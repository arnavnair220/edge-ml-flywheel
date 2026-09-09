"""The labeled set as the files Ultralytics reads.

YOLO wants a directory of images, a parallel directory of one text file per
image, and a small YAML naming the classes. The conversion is three decisions
worth stating, because each one silently changes what the model learns:

**Coordinates are normalized against `NATIVE_IMAGE_SIZE`, not against the file.**
Every box in this project is in the archive's 1280x720 frame -- the manifest, the
oracle's boxes, the eval's ground truth -- and YOLO's format is a fraction of the
image. Reading the size off each JPEG would make the conversion depend on a
decode, and ingest already refuses an image of another size.

**A box outside the class set is dropped, and dropping is counted.** Same rule as
`evaluation.coco.ground_truth`, which is the point: a class the model is not
trained on must not be a class it is scored on. `train` is the category this
mostly means, at 151 boxes archive-wide.

**A box hanging over the edge is clipped, and clipping is counted.** The archive
states corners past the frame on objects entering or leaving it. YOLO refuses a
coordinate outside [0, 1], so the alternatives are to clip or to drop the box
entirely -- and dropping teaches the model that a half-visible car is background.
A box left with no area after clipping is dropped as degenerate, the way ingest
drops one the archive states backwards.

**An image with no boxes is kept, with an empty label file.** That is YOLO's own
spelling of a background frame, and "there is nothing here" is an answer the
model has to learn. It is not a missing label, and the counts below keep the two
distinguishable.
"""

import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import NATIVE_IMAGE_SIZE, ClassSet, ImageId
from edge_ml_flywheel.ingest.labels import Box

log = logging.getLogger(__name__)

IMAGES_DIR: Final = "images"
LABELS_DIR: Final = "labels"
DATA_YAML: Final = "data.yaml"

_IMAGE_SUFFIX: Final = ".jpg"
_LABEL_SUFFIX: Final = ".txt"

# Six decimal places against a 1280 px frame is a sixth of a pixel, which is
# below what the archive's own corners resolve. Fixed rather than repr-shortest
# so two runs over the same labels write byte-identical files.
_PRECISION: Final = 6

_REPORTED: Final = 5


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """What the conversion did, for the job log and for the model manifest.

    Each count is a different question. `images` and `boxes` are the size of the
    training set; `empty_images` says how much of it is background; and the two
    drop counts are the ones worth watching, because both are the conversion
    quietly deciding a box does not exist.
    """

    images: int
    boxes: int
    empty_images: int
    dropped_out_of_class: int
    dropped_degenerate: int
    clipped: int


def _yolo_row(box: Box, category_id: int) -> tuple[str, int]:
    """One box as `<class> <cx> <cy> <w> <h>`, and whether it was clipped.

    Class IDs are zero-based here and one-based in `ClassSet`, which follows COCO.
    The subtraction happens once, in this function, rather than at whichever call
    site remembers.
    """
    width, height = NATIVE_IMAGE_SIZE

    x1 = min(max(box.x1, 0.0), float(width))
    y1 = min(max(box.y1, 0.0), float(height))
    x2 = min(max(box.x2, 0.0), float(width))
    y2 = min(max(box.y2, 0.0), float(height))
    clipped = int((x1, y1, x2, y2) != (box.x1, box.y1, box.x2, box.y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box encloses no area inside the frame: {box}")

    centre_x = (x1 + x2) / 2 / width
    centre_y = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height

    values = " ".join(
        f"{value:.{_PRECISION}f}" for value in (centre_x, centre_y, box_width, box_height)
    )
    return f"{category_id - 1} {values}", clipped


def _link(source: Path, destination: Path) -> None:
    """Symlink the channel's image into the dataset, or copy if we cannot.

    A symlink because the images channel is already on local disk and 8,000
    copies is a gigabyte written for nothing. The fallback exists because a
    developer running the conversion on a machine without symlink permission
    should get a slower test, not a failure.
    """
    try:
        destination.symlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _data_yaml(root: Path, classes: ClassSet) -> str:
    """The dataset descriptor, written by hand rather than through a YAML writer.

    Three keys and a list of nine names, against a dependency the pure layer
    would otherwise carry for this one file. `val` points at the training images
    and is never read: validation is turned off in the job, because the metric
    this project promotes on is the eval plane's over the frozen cohort, and a
    number YOLO prints against the training set is not that metric.
    """
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes.names))
    return f"path: {root}\ntrain: {IMAGES_DIR}\nval: {IMAGES_DIR}\nnames:\n{names}\n"


def write(
    root: Path,
    image_dir: Path,
    labels: Mapping[ImageId, Sequence[Box]],
    classes: ClassSet,
) -> DatasetStats:
    """Lay out one cycle's training set under `root`, and say what it holds.

    The image channel and the labels have to name the same set, and disagreeing
    either way is refused. A labeled image with no file means the manifest was
    written from a different label set than the one that arrived; an image with
    no label means the manifest named something nothing bought. Both are cheap to
    detect here and expensive to notice later, when the symptom is a model
    trained on fewer images than the cycle paid for.
    """
    images = {path.stem: path for path in sorted(image_dir.glob(f"*{_IMAGE_SUFFIX}"))}

    unlabeled = sorted(set(images) - set(labels))
    if unlabeled:
        raise ValueError(
            f"{len(unlabeled):,} image(s) in the channel have no label, so the manifest names "
            f"images this run has not bought: {unlabeled[:_REPORTED]}"
        )
    missing = sorted(set(labels) - set(images))
    if missing:
        raise ValueError(
            f"{len(missing):,} labeled image(s) did not arrive on the images channel, so the "
            f"manifest is not the label set: {missing[:_REPORTED]}"
        )

    image_root = root / IMAGES_DIR
    label_root = root / LABELS_DIR
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    category_ids = classes.category_ids
    boxes_written = 0
    empty_images = 0
    out_of_class = 0
    degenerate = 0
    clipped = 0

    for image_id, source in images.items():
        rows: list[str] = []
        for box in labels[ImageId(image_id)]:
            category_id = category_ids.get(box.category)
            if category_id is None:
                out_of_class += 1
                continue
            try:
                row, was_clipped = _yolo_row(box, category_id)
            except ValueError:
                degenerate += 1
                continue
            rows.append(row)
            clipped += was_clipped

        (label_root / f"{image_id}{_LABEL_SUFFIX}").write_text(
            "".join(f"{row}\n" for row in rows), encoding="utf-8"
        )
        _link(source, image_root / source.name)

        boxes_written += len(rows)
        empty_images += not rows

    (root / DATA_YAML).write_text(_data_yaml(root, classes), encoding="utf-8")

    stats = DatasetStats(
        images=len(images),
        boxes=boxes_written,
        empty_images=empty_images,
        dropped_out_of_class=out_of_class,
        dropped_degenerate=degenerate,
        clipped=clipped,
    )
    log.info(
        "dataset: %d images, %d boxes, %d with no boxes; dropped %d outside the class set and "
        "%d with no area, clipped %d to the frame",
        stats.images,
        stats.boxes,
        stats.empty_images,
        stats.dropped_out_of_class,
        stats.dropped_degenerate,
        stats.clipped,
    )
    return stats

"""BDD100K boxes and a model's detections, in the shapes `pycocotools` reads.

Three conversions:

- **Coordinates.** The archive states corners, `x1 y1 x2 y2`; COCO states a
  corner and an extent, `x y width height`. Both are in `NATIVE_IMAGE_SIZE`
  pixels and nothing here rescales, since the small-object slice is defined at
  native resolution rather than at the model's 416 px (design section 4.4).
- **Class IDs.** `pycocotools` matches on integers, so `car` becomes `1`. The
  mapping is `ClassSet`'s, from the frozen tuple in `conventions` rather than
  from whatever categories appear in the input, which would make the IDs a
  function of the data and relabel cached arrays scored on a different subset.
- **Image IDs.** `pycocotools` wants integers, and a BDD100K image ID is
  `0000f77c-6257be58`. `ImageIndex` holds that mapping.

Ground truth is filtered and detections are not. A ground-truth box outside the
class set is dropped, `train` among them. A detection outside the class set
raises: the archive is data to filter, but a model predicting an untrained class
means the eval ran under a different class set than the one reported.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Self, cast

from pycocotools.coco import COCO

from edge_ml_flywheel.conventions import NATIVE_IMAGE_SIZE, ClassSet, ImageId
from edge_ml_flywheel.ingest.labels import Box

# BDD100K's legacy documents carry no crowd flag, so every annotation is
# `iscrowd=0`. Named because it is a property of the archive rather than a
# default: with no crowd regions, the only ignore flags `COCOeval` raises come
# from the area-range filter.
_ISCROWD: Final = 0


@dataclass(frozen=True, slots=True)
class Detection:
    """One box a model predicted, in native pixels, with its confidence.

    Corners and native pixels, the frame `Box` is in, so one conversion serves
    both. Whatever ran the model owns the rescale from 416 px.

    AP is the area under a precision-recall curve swept by lowering a confidence
    threshold, so `score` is what the metric ranks by; a constant score makes the
    ranking arbitrary.
    """

    category: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True, slots=True)
class ImageIndex:
    """The eval set's image IDs in a fixed order, and their integer COCO IDs.

    An image's integer is its position in the tuple, so the mapping is a function
    of the whole set: index a different set and every ID names a different image.
    Those integers are stored in the cached match arrays and drawn by the
    bootstrap resample, so the index is persisted beside the cache rather than
    re-derived by each reader.

    Sorted rather than in caller order, so two components assembling the same
    eval cohort from different sources agree without coordinating.

    Zero-based, unlike `ClassSet`'s category IDs, because these double as row
    offsets into the cached arrays: a resample draws `rng.integers(0, len(index))`
    and indexes with it directly.
    """

    image_ids: tuple[ImageId, ...]

    @classmethod
    def of(cls, image_ids: Iterable[ImageId]) -> Self:
        """Build the index. Refuses a repeat and refuses an empty set.

        A repeat is not deduplicated. The eval cohort is 5,000 distinct images,
        so a repeated ID means an upstream join produced the wrong set, and both
        available responses -- two integers for one image, or folding them
        together -- misweight the metric without reporting it.
        """
        ordered = sorted(image_ids)
        if not ordered:
            raise ValueError("an eval set of no images has no index")
        duplicates = sorted({image for image in ordered if ordered.count(image) > 1})
        if duplicates:
            raise ValueError(f"image index repeats an image: {duplicates}")
        return cls(image_ids=tuple(ordered))

    def __len__(self) -> int:
        return len(self.image_ids)

    def numeric(self, image_id: ImageId) -> int:
        try:
            return self.image_ids.index(image_id)
        except ValueError:
            raise ValueError(f"{image_id} is not in this eval index") from None

    def image_id(self, numeric: int) -> ImageId:
        """The inverse. A cached array holds integers, and a report needs IDs."""
        if not 0 <= numeric < len(self.image_ids):
            raise ValueError(
                f"image ID {numeric} is outside an index of {len(self.image_ids)} images"
            )
        return self.image_ids[numeric]

    @property
    def numeric_ids(self) -> tuple[int, ...]:
        """Every integer ID, in index order. `COCOeval.params.imgIds`."""
        return tuple(range(len(self.image_ids)))


def _bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Corners to COCO's `[x, y, width, height]`.

    Refuses a box enclosing no area. Ingest drops degenerate ground truth as
    `ParsedLabel.degenerate_boxes`, so one reaching here is either corners the
    archive states backwards or a broken detection post-process. A zero-area box
    has IoU 0 against everything, which would read as a model detecting nothing.
    """
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"box encloses no area: ({x1}, {y1}) to ({x2}, {y2})")
    return [x1, y1, width, height]


def _images(index: ImageIndex) -> list[dict[str, Any]]:
    """One entry per scored image, whether or not it has boxes.

    `file_name` is carried so a document dumped for inspection says which picture
    each integer is; nothing in the scoring path reads it.
    """
    width, height = NATIVE_IMAGE_SIZE
    return [
        {
            "id": index.numeric(image_id),
            "file_name": f"{image_id}.jpg",
            "width": width,
            "height": height,
        }
        for image_id in index.image_ids
    ]


def _categories(classes: ClassSet) -> list[dict[str, Any]]:
    return [{"id": identifier, "name": name} for name, identifier in classes.category_ids.items()]


def ground_truth(
    labels: Mapping[ImageId, Sequence[Box]],
    classes: ClassSet,
    index: ImageIndex,
) -> dict[str, Any]:
    """A COCO ground-truth document over exactly the indexed images.

    `labels` must key every image in `index` and no others, with an image that
    has no boxes present as an empty sequence rather than absent. `pycocotools`
    cannot distinguish the two -- both leave an image with no ground truth -- and
    they mean opposite things: an empty frame, where every detection is correctly
    a false positive, against labels that failed to load, where the false
    positives are the bug. The second lowers the metric for both models equally
    and passes every gate.
    """
    missing = sorted(set(index.image_ids) - set(labels))
    if missing:
        raise ValueError(
            f"{len(missing)} indexed images have no labels entry, and an image with no boxes "
            f"must be present with an empty one: {missing[:5]}"
        )
    extra = sorted(set(labels) - set(index.image_ids))
    if extra:
        raise ValueError(f"{len(extra)} labelled images are not in the eval index: {extra[:5]}")

    category_ids = classes.category_ids
    annotations: list[dict[str, Any]] = []

    for image_id in index.image_ids:
        numeric = index.numeric(image_id)
        for box in labels[image_id]:
            identifier = category_ids.get(box.category)
            if identifier is None:
                continue  # a category outside the class set, `train` among them
            annotations.append(
                {
                    # 1-based and sequential. `pycocotools` requires annotation
                    # IDs to be unique and otherwise never reads them.
                    "id": len(annotations) + 1,
                    "image_id": numeric,
                    "category_id": identifier,
                    "bbox": _bbox(box.x1, box.y1, box.x2, box.y2),
                    # The area the range filter partitions on, in native pixels,
                    # putting the small-object boundary at 32x32 of the real
                    # frame rather than of a 416 px resize.
                    "area": box.area,
                    "iscrowd": _ISCROWD,
                }
            )

    return {
        "images": _images(index),
        "annotations": annotations,
        "categories": _categories(classes),
    }


def detections(
    predictions: Mapping[ImageId, Sequence[Detection]],
    classes: ClassSet,
    index: ImageIndex,
) -> list[dict[str, Any]]:
    """A COCO results list, in the form `COCO.loadRes` accepts.

    An image may be absent, unlike in `ground_truth`: a model may predict nothing
    on a frame, and absence and an empty sequence mean the same thing. A
    prediction for an image outside the index is refused -- it means the model was
    run over something other than the frozen eval cohort, so the number does not
    compare to any other cycle's.

    Not truncated to `maxDets`. That cap belongs to scoring, which applies it
    after sorting by score; applying it here would fix one cap in a cache built
    to answer later questions without re-scoring.
    """
    unknown_images = sorted(set(predictions) - set(index.image_ids))
    if unknown_images:
        raise ValueError(
            f"{len(unknown_images)} predicted images are not in the eval index: "
            f"{unknown_images[:5]}"
        )

    category_ids = classes.category_ids
    records: list[dict[str, Any]] = []

    for image_id in sorted(predictions):
        numeric = index.numeric(image_id)
        for detection in predictions[image_id]:
            identifier = category_ids.get(detection.category)
            if identifier is None:
                raise ValueError(
                    f"{image_id}: detected {detection.category!r}, which is not in the class "
                    f"set the eval is running under: {list(classes.names)}"
                )
            if not 0.0 <= detection.score <= 1.0:
                raise ValueError(f"{image_id}: detection score {detection.score} is outside [0, 1]")
            records.append(
                {
                    "image_id": numeric,
                    "category_id": identifier,
                    "bbox": _bbox(detection.x1, detection.y1, detection.x2, detection.y2),
                    "score": detection.score,
                }
            )

    return records


def _as_dataset(document: Mapping[str, Any]) -> Any:
    """Hand a plain document to `COCO.dataset`, whose stub is narrower than it.

    `types-pycocotools` types `dataset` as a TypedDict requiring `segmentation`
    on every annotation and `supercategory` on every category. The `bbox` path
    reads neither: `COCOeval` at `iouType="bbox"` touches `bbox`, `area` and
    `iscrowd` only. Satisfying the stub would add two permanently empty fields to
    every annotation of a 5,000-image document, so the widening is stated here
    once instead, at both call sites' expense of one indirection.
    """
    return dict(document)


def as_coco(document: Mapping[str, Any]) -> COCO:
    """Wrap a ground-truth document as a `COCO`, without touching a disk.

    `COCO(path)` is the only documented constructor and it reads JSON from a
    file. Assigning `dataset` then calling `createIndex` is the same pair of
    steps it performs after parsing, without the round trip through a temp file.
    """
    coco = COCO()
    coco.dataset = _as_dataset(document)
    coco.createIndex()
    return coco


def as_coco_results(ground_truth_coco: COCO, records: Sequence[Mapping[str, Any]]) -> COCO:
    """Wrap a results list as a `COCO`, empty list included.

    `COCO.loadRes` reads `anns[0]` to decide whether the results are captions,
    boxes or segmentations, so it raises `IndexError` on an empty list. Design
    section 4.2 hard fails a challenger whose classes collapse to zero
    detections, which the gate can only report if scoring does not raise first,
    so the empty case is assembled here in the steps `loadRes` would have taken.
    """
    if records:
        # `loadRes` is typed as taking a file path, as its docstring claims, and
        # has accepted an in-memory list since 2.0. Passing a path would mean
        # writing these records to a temp file to read them straight back.
        return ground_truth_coco.loadRes(cast(Any, [dict(record) for record in records]))

    empty = COCO()
    empty.dataset = _as_dataset(
        {
            "images": list(ground_truth_coco.dataset["images"]),
            "annotations": [],
            "categories": copy.deepcopy(ground_truth_coco.dataset["categories"]),
        }
    )
    empty.createIndex()
    return empty

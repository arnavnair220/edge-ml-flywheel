"""The scoring job itself, as it runs inside the container.

Started by `container/score.py`, a three-line file at the root of the source
archive beside `container/train.py`.

The order of the steps is the order the failures are worth having in. The model
is fetched and loaded before any image is scored, so a missing or unreadable
checkpoint costs seconds rather than a pass over 5,000 frames. Each cohort is
then scored and uploaded before the next begins, so a job that dies partway
leaves the cohorts before the failure in the store.

**No label is read, and there is nowhere to read one from.** A checkpoint and the
images `scoring.inputs` names arrive; ground truth is matched against these
detections in the step after this one, under an identity that is allowed to hold
it. That division is what lets the scoring identity be denied every label prefix
outright rather than trusted to stay away from one.
"""

import argparse
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from ultralytics import YOLO

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    Cohort,
    DetectionRow,
    ImageId,
    Precision,
    Seed,
    detections_key,
    parse_image_id,
    parse_model_version,
    uri,
)
from edge_ml_flywheel.scoring import detections, inputs
from edge_ml_flywheel.storage import ARTIFACTS_BUCKET, DATA_BUCKET, ObjectStore, bucket_name, tree
from edge_ml_flywheel.storage.filesystem import FsspecStore

log = logging.getLogger("edge_ml_flywheel.scoring")

# Where the inputs land and the detections are written before upload. Local
# scratch on the job's own volume; nothing here survives the job except what is
# uploaded.
WORK: Final = Path("/tmp/scoring")

# Images per forward pass arrive as `--batch` rather than being declared here.
# The two passes need different values -- the int8 graph's batch axis is fixed at
# 1 by `export.to_onnx`'s `dynamic=False` and ONNX Runtime refuses anything else
# -- so the number is a function of the precision being run, and `job.Scoring`
# decides it where a test can read the decision.

_IMAGE_SUFFIX: Final = ".jpg"

# How many offending IDs a refusal lists before summarizing, matching the rest of
# the package.
_REPORTED: Final = 5


def _parser() -> argparse.ArgumentParser:
    """The flags `job.arguments` writes, underscored to match `training.entrypoint`."""
    parser = argparse.ArgumentParser(prog="score.py")
    parser.add_argument("--version", required=True, help="The model version being scored.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--precision",
        type=Precision,
        choices=list(Precision),
        default=Precision.FP32,
        help="Which build of the model to score with.",
    )
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--confidence_floor", type=float, required=True)
    parser.add_argument("--max_detections", type=int, required=True)
    # Required rather than defaulted, because the two passes need different
    # values and only one of them is survivable: the int8 graph has a batch axis
    # fixed at 1 and ONNX Runtime refuses anything else. A default here is a
    # container that quietly batches the artifact that ships.
    parser.add_argument("--batch", type=int, required=True)
    return parser


def _chunks(paths: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def _class_names(model: Any) -> dict[int, str]:
    """The model's own class vocabulary, checked against the one it is scored under.

    A fine-tuned checkpoint carries the names the dataset YAML gave it, which
    `training.dataset` writes from `CLASS_SET`. A mismatch means the checkpoint
    is not the model this cycle trained -- the COCO base, most plausibly, whose
    eighty names overlap this project's nine -- and scoring it would produce
    detections in a vocabulary nothing downstream can read. `evaluation.coco`
    raises on the same condition one step later; raising here costs the job
    seconds instead of an hour.
    """
    names = {int(index): str(name) for index, name in dict(model.names).items()}
    unknown = sorted(set(names.values()) - set(CLASS_SET.names))
    if unknown:
        raise SystemExit(
            f"the checkpoint predicts {len(names)} classes, {len(unknown)} of which are outside "
            f"the class set it is scored under: {unknown[:_REPORTED]}. "
            f"Declared: {list(CLASS_SET.names)}"
        )
    return names


def _rows(image_id: ImageId, result: Any, names: dict[int, str]) -> Iterator[DetectionRow]:
    """One image's boxes as rows, dropping any that enclose no area.

    Coordinates arrive in the original frame rather than at `image_size`:
    Ultralytics rescales its predictions back to the source image, which is
    `NATIVE_IMAGE_SIZE` here because ingest refuses an image of another size. So
    these are already in the frame `Box`, `Detection` and the eval's area
    thresholds all use, and nothing in this project rescales twice.

    A degenerate box is dropped and counted rather than raised on, matching
    `training.dataset`: it is a rounding artifact at the frame edge, and a job
    that dies an hour in over one such box has thrown away every other frame.
    """
    boxes = result.boxes
    if boxes is None:
        return

    for corners, confidence, category in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
    ):
        x1, y1, x2, y2 = (float(value) for value in corners)
        if x2 <= x1 or y2 <= y1:
            continue
        yield DetectionRow(
            image_id=image_id,
            category=names[int(category)],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=float(confidence),
        )


def detect(
    model: Any, paths: Sequence[Path], names: dict[int, str], args: argparse.Namespace
) -> Iterator[DetectionRow]:
    """Run the model over one cohort, yielding rows as they come.

    A generator, so `detections.write` flushes a row group at a time and the peak
    memory is a buffer rather than every box in the cohort. That is what makes
    62,000 images at up to `max_detections` each a job that fits.

    `verbose=False` because Ultralytics prints a line per image by default, and
    67,000 of them is a log nobody can read and a CloudWatch bill for the
    privilege.
    """
    for chunk in _chunks(paths, args.batch):
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.image_size,
            conf=args.confidence_floor,
            max_det=args.max_detections,
            verbose=False,
        )
        for path, result in zip(chunk, results, strict=True):
            yield from _rows(parse_image_id(path.stem), result, names)


def score_cohort(
    model: Any,
    cohort: Cohort,
    paths: Sequence[Path],
    names: dict[int, str],
    args: argparse.Namespace,
) -> Path:
    """Score one cohort and return the detections file it wrote."""
    if not paths:
        raise SystemExit(f"the {cohort.value} manifest names no images, so nothing to score")

    log.info("%s: scoring %d images", cohort.value, len(paths))
    local = WORK / "output" / f"{cohort.value}.parquet"
    written = detections.write(detect(model, paths, names, args), local)
    log.info(
        "%s: %d detections over %d images, %.1f per image",
        cohort.value,
        written,
        len(paths),
        written / len(paths),
    )
    return local


def publish(artifacts: ObjectStore, local: Path, cohort: Cohort, args: argparse.Namespace) -> None:
    """Upload one cohort's detections to the key their readers walk.

    `detections_key` for this precision, so the file a reader finds under
    `detections_prefix` is the one this job wrote.
    """
    key = detections_key(parse_model_version(args.version), Seed(args.seed), cohort, args.precision)
    artifacts.upload(local, key)
    log.info("wrote %s", uri(artifacts.bucket, key))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parser().parse_args(argv)

    data = FsspecStore.connect(bucket_name(DATA_BUCKET), pool=tree.WORKERS)
    artifacts = FsspecStore.connect(bucket_name(ARTIFACTS_BUCKET))
    scored = inputs.Pass(parse_model_version(args.version), Seed(args.seed), args.precision)
    staged = inputs.stage(data, artifacts, scored, WORK / "input")
    log.info(
        "scoring %s seed %d at %s from %s", args.version, args.seed, args.precision, staged.model
    )

    model = YOLO(str(staged.model))
    names = _class_names(model)

    # Sorted, so a job that dies partway leaves the cohorts before the failure
    # uploaded in a fixed order. One cohort makes that moot today and the loop
    # stays, because what this walks is whatever cohorts `inputs` staged.
    for cohort in sorted(staged.images):
        written = score_cohort(model, cohort, staged.images[cohort], names, args)
        publish(artifacts, written, cohort, args)


if __name__ == "__main__":
    main()

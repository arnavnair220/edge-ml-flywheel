"""The scoring job itself, as it runs inside the container.

Started by `container/score.py`, which is a three-line file at the root of the
source archive for `container/train.py`'s reason -- though not for its cause. A
training job gets script mode and an entry point SageMaker finds; a Processing
job gets whatever `job.container_entrypoint` names and nothing else, so the file
is there to be named.

The order of the steps is the order the failures are worth having in. The model
is loaded before any channel is walked, so a missing or unreadable checkpoint
costs seconds rather than a pass over 62,000 frames. Each cohort is then scored
and written before the next begins, so a job that dies on the pool still leaves
the eval detections -- which are the expensive half to regenerate, since they are
the ones every later cycle compares against.

**No label is read, and there is nowhere to read one from.** Two image channels
and a checkpoint arrive; ground truth is matched against these detections in the
step after this one, under a role that is allowed to hold it. That division is
what lets the scoring role be denied every label prefix outright rather than
trusted to stay away from one.
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
    SCORED_COHORTS,
    Cohort,
    DetectionRow,
    ImageId,
    ModelArtifact,
    Seed,
    detections_key,
    parse_image_id,
    parse_model_version,
)
from edge_ml_flywheel.scoring import detections
from edge_ml_flywheel.scoring.job import (
    INPUT_ROOT,
    MODEL_CHANNEL,
    OUTPUT_ROOT,
)

log = logging.getLogger("edge_ml_flywheel.scoring")

# Images per forward pass. Chunked here rather than left to the framework's
# default of one, because the whole job is a pass over 67,000 frames and a batch
# of one leaves the GPU idle between them. Not `Recipe.batch`, which is a
# training decision constrained by gradient memory this job never allocates.
BATCH: Final = 32

_IMAGE_SUFFIX: Final = ".jpg"

# How many offending IDs a refusal lists before summarizing, matching the rest of
# the package.
_REPORTED: Final = 5


def _parser() -> argparse.ArgumentParser:
    """The flags `job.arguments` writes, underscored to match `training.entrypoint`."""
    parser = argparse.ArgumentParser(prog="score.py")
    parser.add_argument("--version", required=True, help="The model version being scored.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--confidence_floor", type=float, required=True)
    parser.add_argument("--max_detections", type=int, required=True)
    return parser


def channel(name: str) -> Path:
    return Path(INPUT_ROOT) / name


def checkpoint() -> Path:
    """The model this job scores with.

    Named by `ModelArtifact.TORCH` rather than globbed, because the channel is a
    single-object prefix and a glob that found two files would pick one of them
    silently.
    """
    path = channel(MODEL_CHANNEL) / ModelArtifact.TORCH.value
    if not path.is_file():
        raise SystemExit(f"the {MODEL_CHANNEL} channel carries no {ModelArtifact.TORCH.value}")
    return path


def cohort_images(cohort: Cohort) -> list[Path]:
    """Every image on one cohort's channel, in a fixed order.

    `rglob` rather than a flat glob, and the difference is a failure mode rather
    than a preference: a manifest channel lands its objects flat today, and if
    that ever changed a flat glob would find nothing, write an empty detections
    file and report it as a model that saw nothing -- which is a verdict design
    section 4.2 acts on. Finding the images either way makes the empty file mean
    what it says.
    """
    root = channel(cohort.value)
    found = sorted(path for path in root.rglob(f"*{_IMAGE_SUFFIX}") if path.is_file())
    if not found:
        raise SystemExit(
            f"the {cohort.value} channel carries no images, so there is nothing to score"
        )
    return found


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
    for chunk in _chunks(paths, BATCH):
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
    model: Any, cohort: Cohort, names: dict[int, str], args: argparse.Namespace
) -> None:
    """Score one cohort and write its detections where the output channel expects.

    The file name is derived from `detections_key` rather than spelled, because
    the output channel uploads this directory's contents under
    `detections_prefix` and the two halves of that key have to agree. A name
    invented here would produce a job that succeeds and an object nothing looks
    for.
    """
    version = parse_model_version(args.version)
    paths = cohort_images(cohort)
    log.info("%s: scoring %d images", cohort.value, len(paths))

    name = Path(detections_key(version, Seed(args.seed), cohort)).name
    written = detections.write(
        detect(model, paths, names, args), Path(OUTPUT_ROOT) / cohort.value / name
    )
    log.info(
        "%s: %d detections over %d images, %.1f per image",
        cohort.value,
        written,
        len(paths),
        written / len(paths),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parser().parse_args(argv)

    base = checkpoint()
    log.info("scoring %s seed %d from %s", args.version, args.seed, base)

    model = YOLO(str(base))
    names = _class_names(model)

    # Sorted for `job.inputs`' reason, and it decides something here that it does
    # not there: a job that dies partway leaves the cohorts before the failure
    # written, so the order is the order they are worth having.
    for cohort in sorted(SCORED_COHORTS):
        score_cohort(model, cohort, names, args)


if __name__ == "__main__":
    main()

"""Accumulation, checked against the implementation it replaces.

Correctness is equality with `pycocotools` rather than numbers written down here:
there is no hand-computed mAP over forty images. So the fixture is generated from
a fixed seed, with enough near-misses, false positives and repeated scores that
the precision envelope, the 101-point interpolation and the stable tie-break all
do something.

`COCOeval.accumulate()` is the reference, not `summarize()`, which indexes
`params.maxDets` positionally and assumes the default three-entry list.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval

from edge_ml_flywheel.conventions import CLASS_SET, MAX_DETS, ImageId
from edge_ml_flywheel.evaluation.coco import (
    Detection,
    ImageIndex,
    detections,
    ground_truth,
)
from edge_ml_flywheel.evaluation.match import (
    AREA_BOUNDS,
    IOU_THRESHOLDS,
    AreaRange,
    MatchCache,
    as_coco,
    as_coco_results,
    score,
)
from edge_ml_flywheel.evaluation.metrics import Scope, average_precision, image_rows
from edge_ml_flywheel.ingest.labels import Box

CLASSES = CLASS_SET

IMAGES = 40
# Three images get exactly the detections their boxes deserve, so a resample over
# them has an AP that is exact rather than a number needing a reference. Two carry
# no boxes at all, which makes "no class has ground truth here" reachable.
PERFECT = (3, 11, 26)
EMPTY = (8, 22)


@dataclass(frozen=True, slots=True)
class Fixture:
    labels: dict[ImageId, list[Box]]
    predictions: dict[ImageId, list[Detection]]
    index: ImageIndex
    cache: MatchCache


def a_fixture(seed: int = 20260907) -> Fixture:
    """Forty images of boxes, and a model that is right about most of them.

    Detections are jittered off their boxes by varying amounts so IoU lands
    between thresholds and the ten of them disagree. Scores are rounded to two
    decimals so ties are common, which is what the stable sort is for.
    """
    rng = np.random.default_rng(seed)
    width, height = 1280, 720

    labels: dict[ImageId, list[Box]] = {}
    predictions: dict[ImageId, list[Detection]] = {}

    for position in range(IMAGES):
        image_id = ImageId(f"{position:08x}-{position:08x}")
        boxes: list[Box] = []
        detected: list[Detection] = []

        if position not in EMPTY:
            for _ in range(int(rng.integers(1, 7))):
                # Spanning the 32x32 and 96x96 boundaries, so all four area
                # ranges are populated.
                size = float(rng.choice([18.0, 28.0, 45.0, 80.0, 140.0, 260.0]))
                aspect = float(rng.uniform(0.6, 1.8))
                box_width = min(size * aspect, width - 1.0)
                box_height = min(size / aspect, height - 1.0)
                x1 = float(rng.uniform(0, width - box_width))
                y1 = float(rng.uniform(0, height - box_height))
                boxes.append(
                    Box(
                        category=str(rng.choice(CLASSES.names)),
                        x1=x1,
                        y1=y1,
                        x2=x1 + box_width,
                        y2=y1 + box_height,
                    )
                )

        for box in boxes:
            if position not in PERFECT and rng.random() > 0.75:
                continue  # a miss
            drift = 0.0 if position in PERFECT else float(rng.uniform(0, 0.22))
            shift_x = (box.x2 - box.x1) * drift
            shift_y = (box.y2 - box.y1) * drift
            confidence = 1.0 if position in PERFECT else round(float(rng.uniform(0.1, 1.0)), 2)
            detected.append(
                Detection(
                    category=box.category,
                    x1=box.x1 + shift_x,
                    y1=box.y1 + shift_y,
                    x2=box.x2 + shift_x,
                    y2=box.y2 + shift_y,
                    score=confidence,
                )
            )

        if position not in PERFECT:
            for _ in range(int(rng.integers(0, 4))):  # false positives
                size = float(rng.uniform(20, 200))
                x1 = float(rng.uniform(0, width - size))
                y1 = float(rng.uniform(0, height - size))
                detected.append(
                    Detection(
                        category=str(rng.choice(CLASSES.names)),
                        x1=x1,
                        y1=y1,
                        x2=x1 + size,
                        y2=y1 + size,
                        score=round(float(rng.uniform(0.1, 1.0)), 2),
                    )
                )

        labels[image_id] = boxes
        predictions[image_id] = detected

    index = ImageIndex.of(labels)
    truth = as_coco(ground_truth(labels, CLASSES, index))
    results = as_coco_results(truth, detections(predictions, CLASSES, index))
    return Fixture(
        labels=labels,
        predictions=predictions,
        index=index,
        cache=score(truth, results, index),
    )


FIXTURE = a_fixture()
ALL_ROWS = image_rows(range(IMAGES))


def rows_for(positions: tuple[int, ...]) -> np.ndarray:
    return image_rows(
        FIXTURE.index.numeric(ImageId(f"{position:08x}-{position:08x}")) for position in positions
    )


def coco_reference(image_ids: list[int] | None = None) -> COCOeval:
    """A full `COCOeval` pass over the same data, accumulated."""
    truth = as_coco(ground_truth(FIXTURE.labels, CLASSES, FIXTURE.index))
    results = as_coco_results(truth, detections(FIXTURE.predictions, CLASSES, FIXTURE.index))
    evaluator = COCOeval(truth, results, iouType="bbox")
    evaluator.params.imgIds = list(image_ids or FIXTURE.index.numeric_ids)
    evaluator.params.catIds = [CLASSES.category_id(name) for name in CLASSES.names]
    evaluator.params.iouThrs = np.array(IOU_THRESHOLDS)
    evaluator.params.areaRng = [list(AREA_BOUNDS[area]) for area in AreaRange]
    evaluator.params.areaRngLbl = [area.value for area in AreaRange]
    evaluator.params.maxDets = [MAX_DETS]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def coco_ap(evaluator: COCOeval, area: AreaRange, iou: float | None = None) -> float:
    """`summarize()`'s own arithmetic: mean of the precision entries above -1."""
    entries = evaluator.eval["precision"][:, :, :, list(AreaRange).index(area), 0]
    if iou is not None:
        entries = entries[np.flatnonzero(np.isclose(IOU_THRESHOLDS, iou))]
    return float(entries[entries > -1].mean())


@pytest.mark.parametrize("iou", [None, 0.5])
@pytest.mark.parametrize("area", list(AreaRange))
def test_the_identity_resample_matches_cocoeval(area: AreaRange, iou: float | None):
    reference = coco_reference()

    assert average_precision(FIXTURE.cache, ALL_ROWS, Scope(area=area, iou=iou)) == pytest.approx(
        coco_ap(reference, area, iou)
    )


def test_a_subset_matches_a_pass_over_only_those_images():
    """The claim that makes a slice readable off a whole-cohort cache."""
    chosen = sorted(np.random.default_rng(7).choice(IMAGES, size=15, replace=False).tolist())
    reference = coco_reference(image_ids=chosen)

    assert average_precision(FIXTURE.cache, image_rows(chosen)) == pytest.approx(
        coco_ap(reference, AreaRange.ALL)
    )


def test_a_perfect_subset_scores_one_even_when_images_repeat():
    """What a bootstrap draw does to a slice the model gets entirely right."""
    drawn = np.random.default_rng(11).choice(rows_for(PERFECT), size=12, replace=True)

    assert average_precision(FIXTURE.cache, drawn) == pytest.approx(1.0)


def test_a_slice_with_no_ground_truth_has_no_ap():
    with pytest.raises(ValueError, match="no AP to report"):
        average_precision(FIXTURE.cache, rows_for(EMPTY))


def test_refuses_an_iou_nothing_was_scored_at():
    """At construction, before any accumulation has been paid for."""
    with pytest.raises(ValueError, match="nothing was scored at IoU"):
        Scope(iou=0.62)

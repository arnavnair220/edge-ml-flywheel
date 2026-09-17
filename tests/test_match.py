"""The scoring pass and the layout of what it caches.

Three images, boxes at readable coordinates, and one detection per case worth
naming, so a failure says which part of the layout is wrong rather than that a
number moved.

The central test compares a gather against `COCOeval`'s own `evalImgs`. That is
the claim the module makes: the cache is what `accumulate()` would have consumed,
addressed by block instead of by position. The arithmetic built on top of it is
`test_metrics`' subject, not this file's.
"""

from pathlib import Path

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
    load,
    save,
    score,
)
from edge_ml_flywheel.ingest.labels import Box

IMAGE_A = ImageId("00000000-0000000a")
IMAGE_B = ImageId("00000000-0000000b")
IMAGE_C = ImageId("00000000-0000000c")

CLASSES = CLASS_SET

CAR = CLASSES.category_id("car")
PERSON = CLASSES.category_id("person")

# A large car and a 20x20 one, which is under the 32x32 boundary and so is the
# only box in the fixture that the `SMALL` range keeps and `LARGE` ignores.
BIG_CAR = Box(category="car", x1=100, y1=100, x2=300, y2=250)
SMALL_CAR = Box(category="car", x1=500, y1=400, x2=520, y2=420)
PERSON_BOX = Box(category="person", x1=700, y1=300, x2=740, y2=420)

LABELS = {
    IMAGE_A: [BIG_CAR, SMALL_CAR],
    IMAGE_B: [PERSON_BOX],
    # An empty frame, present with an empty list. Every detection on it is
    # correctly a false positive.
    IMAGE_C: [],
}

INDEX = ImageIndex.of(LABELS)


def a_detection(box: Box, score_value: float) -> Detection:
    """A detection placed exactly on a box, so it matches at every IoU threshold."""
    return Detection(
        category=box.category, x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, score=score_value
    )


# Deliberately not one detection per box. Image A's second car detection is a
# false positive well away from either car, image C's is a detection on an empty
# frame, and nothing detects the small car -- so the fixture exercises a hit, a
# miss, a false positive and an empty block in one pass.
PREDICTIONS = {
    IMAGE_A: [
        a_detection(BIG_CAR, 0.9),
        Detection(category="car", x1=20, y1=20, x2=120, y2=90, score=0.7),
    ],
    IMAGE_B: [a_detection(PERSON_BOX, 0.6)],
    IMAGE_C: [Detection(category="car", x1=10, y1=10, x2=60, y2=60, score=0.5)],
}


def a_cache() -> MatchCache:
    truth = as_coco(ground_truth(LABELS, CLASSES, INDEX))
    results = as_coco_results(truth, detections(PREDICTIONS, CLASSES, INDEX))
    return score(truth, results, INDEX)


def an_evaluator() -> COCOeval:
    """A second pass, parameterized the same way, kept for its `evalImgs`.

    `score` discards the evaluator, so the reference for a layout test has to be
    built alongside rather than read back out of the cache.
    """
    truth = as_coco(ground_truth(LABELS, CLASSES, INDEX))
    results = as_coco_results(truth, detections(PREDICTIONS, CLASSES, INDEX))
    evaluator = COCOeval(truth, results, iouType="bbox")
    evaluator.params.imgIds = list(INDEX.numeric_ids)
    evaluator.params.catIds = [CLASSES.category_id(name) for name in CLASSES.names]
    evaluator.params.iouThrs = np.array(IOU_THRESHOLDS)
    evaluator.params.areaRng = [list(AREA_BOUNDS[area]) for area in AreaRange]
    evaluator.params.areaRngLbl = [area.value for area in AreaRange]
    evaluator.params.maxDets = [MAX_DETS]
    evaluator.evaluate()
    return evaluator


def accumulate_inputs(
    evaluator: COCOeval, category_id: int, area: AreaRange
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """What `COCOeval.accumulate()` builds for one category and area range.

    Its own slicing, transcribed: the flat `evalImgs` list is indexed
    category-major, `None` entries are dropped, and the pieces are concatenated in
    image order.
    """
    images = len(INDEX)
    base = (category_id - 1) * len(AreaRange) * images + list(AreaRange).index(area) * images
    blocks = [evaluator.evalImgs[base + row] for row in range(images)]
    present = [block for block in blocks if block is not None]
    return (
        np.concatenate([np.array(block["dtScores"]) for block in present]),
        np.concatenate([block["dtMatches"] for block in present], axis=1) > 0,
        np.concatenate([block["dtIgnore"] for block in present], axis=1) > 0,
        int(np.count_nonzero(np.concatenate([block["gtIgnore"] for block in present]) == 0)),
    )


@pytest.mark.parametrize("area", list(AreaRange))
@pytest.mark.parametrize("category_id", [CAR, PERSON])
def test_the_identity_resample_reproduces_accumulates_own_arrays(category_id: int, area: AreaRange):
    """The module's whole claim, for every category and area range."""
    cache = a_cache()
    expected_scores, expected_matched, expected_ignored, expected_truth = accumulate_inputs(
        an_evaluator(), category_id, area
    )

    blocks = cache.blocks(category_id, area, np.arange(len(INDEX)))
    rows = cache.detection_rows(blocks)

    assert np.array_equal(cache.scores[rows], expected_scores)
    assert np.array_equal(cache.matched[:, rows], expected_matched)
    assert np.array_equal(cache.ignored[:, rows], expected_ignored)
    assert cache.truth_count(blocks) == expected_truth


def test_a_block_holds_one_image_and_one_category():
    cache = a_cache()
    row_a = np.array([INDEX.numeric(IMAGE_A)])

    cars = cache.detection_rows(cache.blocks(CAR, AreaRange.ALL, row_a))
    people = cache.detection_rows(cache.blocks(PERSON, AreaRange.ALL, row_a))

    # Image A's two car detections, descending by score, and no person.
    assert cache.scores[cars].tolist() == [0.9, 0.7]
    assert people.size == 0


def test_an_image_with_no_boxes_and_no_detections_is_an_empty_block():
    """Image B has a person and no car, so `evaluateImg` returns nothing."""
    cache = a_cache()
    blocks = cache.blocks(CAR, AreaRange.ALL, np.array([INDEX.numeric(IMAGE_B)]))

    assert cache.detection_rows(blocks).size == 0
    assert cache.truth_count(blocks) == 0


def test_the_small_range_keeps_only_the_small_box():
    cache = a_cache()
    rows = np.arange(len(INDEX))

    assert cache.truth_count(cache.blocks(CAR, AreaRange.SMALL, rows)) == 1  # the 20x20 car
    assert cache.truth_count(cache.blocks(CAR, AreaRange.ALL, rows)) == 2  # both cars


def test_a_repeated_image_contributes_twice():
    """What a bootstrap draw with replacement asks of the layout."""
    cache = a_cache()
    row_a = INDEX.numeric(IMAGE_A)

    once = cache.blocks(CAR, AreaRange.ALL, np.array([row_a]))
    twice = cache.blocks(CAR, AreaRange.ALL, np.array([row_a, row_a]))

    assert cache.detection_rows(twice).tolist() == cache.detection_rows(once).tolist() * 2
    assert cache.truth_count(twice) == 2 * cache.truth_count(once)


def test_max_dets_keeps_the_highest_scoring_detections():
    cache = a_cache()
    blocks = cache.blocks(CAR, AreaRange.ALL, np.array([INDEX.numeric(IMAGE_A)]))

    assert cache.scores[cache.detection_rows(blocks, max_dets=1)].tolist() == [0.9]


def test_a_model_that_detected_nothing_scores_rather_than_raising():
    """Design section 4.2 hard fails this, which needs a number to fail on."""
    truth = as_coco(ground_truth(LABELS, CLASSES, INDEX))
    cache = score(truth, as_coco_results(truth, []), INDEX)

    blocks = cache.blocks(CAR, AreaRange.ALL, np.arange(len(INDEX)))
    assert cache.detection_rows(blocks).size == 0
    # The boxes are still there to be missed, so AP is zero and not undefined.
    assert cache.truth_count(blocks) == 2


def test_a_loaded_cache_answers_identically(tmp_path: Path):
    cache = a_cache()
    path = tmp_path / "matches.npz"
    save(cache, path)

    restored = load(path)

    rows = np.arange(len(INDEX))
    original = cache.detection_rows(cache.blocks(CAR, AreaRange.ALL, rows))
    again = restored.detection_rows(restored.blocks(CAR, AreaRange.ALL, rows))
    assert restored.index == cache.index
    assert np.array_equal(restored.scores[again], cache.scores[original])
    assert np.array_equal(restored.matched[:, again], cache.matched[:, original])
    assert np.array_equal(restored.n_truth, cache.n_truth)


def test_refuses_a_file_scored_at_other_iou_thresholds(tmp_path: Path):
    """The failure this catches is silent: the arrays are indexed by position."""
    path = tmp_path / "matches.npz"
    save(a_cache(), path)
    with np.load(path) as archive:
        members = dict(archive)
    members["iou_thresholds"] = np.array([0.5, 0.75])
    np.savez_compressed(path, **members)

    with pytest.raises(ValueError, match="IoU thresholds"):
        load(path)


def test_refuses_arrays_that_disagree_on_the_detection_count():
    cache = a_cache()

    with pytest.raises(ValueError, match="scores is"):
        MatchCache(
            index=cache.index,
            matched=cache.matched,
            ignored=cache.ignored,
            scores=cache.scores[:-1],
            starts=cache.starts,
            n_truth=cache.n_truth,
        )

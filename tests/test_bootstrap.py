"""The paired delta and its band.

A resampling band has no closed form to check against, so what is tested is
structure rather than a number: that the draws are the same every run, that the
delta is the measurement and not the resample mean, that a model against itself
cancels exactly, and that a mismatched comparison is refused.

The fixture is deliberately blunt. Twelve images, a champion that detects the
first car in each and a challenger that detects them all, so the sign of the
delta is known by construction and any band that disagrees with it is a bug.

Images carry one, two or three cars rather than a uniform count. That is
load-bearing: with identical images every resample returns the identical AP, the
band collapses onto the point estimate, and the tests below pass without the
resampling having done anything.
"""

import numpy as np
import pytest

from edge_ml_flywheel.conventions import ClassSetVersion, ImageId, Seed, class_set
from edge_ml_flywheel.evaluation.bootstrap import RESAMPLE_SEED, paired_delta, resamples
from edge_ml_flywheel.evaluation.coco import (
    Detection,
    ImageIndex,
    as_coco,
    as_coco_results,
    detections,
    ground_truth,
)
from edge_ml_flywheel.evaluation.match import MatchCache, score
from edge_ml_flywheel.evaluation.metrics import average_precision, image_rows
from edge_ml_flywheel.ingest.labels import Box

CLASS_SET_VERSION = ClassSetVersion(1)
CLASSES = class_set(CLASS_SET_VERSION)

IMAGES = 12
# Three, not five. The pairing is what is under test and it does not care how many
# seeds there are, so the fixture pays for three scoring passes rather than five
# (design section 4.2 trains five).
SEEDS = (Seed(1), Seed(2), Seed(3))


def an_image(position: int) -> ImageId:
    return ImageId(f"{position:08x}-{position:08x}")


def boxes_for(position: int) -> list[Box]:
    """One, two or three cars, placed apart, at coordinates that vary by image."""
    left = 50.0 + position * 10
    return [
        Box(
            category="car",
            x1=left + slot * 380,
            y1=100 + slot * 90,
            x2=left + slot * 380 + 120,
            y2=200 + slot * 90,
        )
        for slot in range(1 + position % 3)
    ]


LABELS = {an_image(position): boxes_for(position) for position in range(IMAGES)}
INDEX = ImageIndex.of(LABELS)


def a_cache(boxes_detected: int, image_ids: tuple[ImageId, ...] | None = None) -> MatchCache:
    """A model that finds the first `boxes_detected` cars in every image."""
    index = INDEX if image_ids is None else ImageIndex.of(image_ids)
    labels = {image_id: LABELS[image_id] for image_id in index.image_ids}
    predictions = {
        image_id: [
            Detection(
                category=box.category,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                score=0.9,
            )
            for box in labels[image_id][:boxes_detected]
        ]
        for image_id in index.image_ids
    }
    truth = as_coco(ground_truth(labels, CLASSES, index))
    results = as_coco_results(truth, detections(predictions, CLASSES, index))
    return score(truth, results, CLASS_SET_VERSION, index)


# Seed 3 is a slightly better run than the other two, so averaging over seeds is
# an average of different numbers rather than a no-op.
HALF = a_cache(1)
CHAMPION = {Seed(1): HALF, Seed(2): HALF, Seed(3): a_cache(2)}
CHALLENGER = dict.fromkeys(SEEDS, a_cache(2))

ALL_ROWS = image_rows(range(IMAGES))


def test_resamples_are_the_same_draws_every_time():
    """Cycle eight's band must be computed over cycle one's draws."""
    first = resamples(ALL_ROWS, count=5)
    again = resamples(ALL_ROWS, count=5)

    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert all(sample.size == ALL_ROWS.size for sample in first)
    # Drawn with replacement, so a sample is not a permutation.
    assert any(len(set(sample.tolist())) < sample.size for sample in first)


def test_a_different_seed_gives_different_draws():
    assert not np.array_equal(
        resamples(ALL_ROWS, count=1)[0],
        resamples(ALL_ROWS, count=1, seed=RESAMPLE_SEED + 1)[0],
    )


def test_a_real_improvement_clears_the_band():
    delta = paired_delta(CHAMPION, CHALLENGER, count=100)

    assert delta.observed > 0
    assert delta.lower <= delta.observed <= delta.upper
    assert delta.improved
    # The band has width, so the resampling did something. Without this the test
    # passes on a fixture where every draw returns the same number.
    assert delta.upper > delta.lower


def test_the_observed_delta_is_the_measurement_not_a_resample_mean():
    delta = paired_delta(CHAMPION, CHALLENGER, count=100)

    champion = np.mean([average_precision(cache, ALL_ROWS) for cache in CHAMPION.values()])
    challenger = np.mean([average_precision(cache, ALL_ROWS) for cache in CHALLENGER.values()])
    assert delta.observed == pytest.approx(float(challenger - champion))


def test_a_model_against_itself_cancels_on_every_draw():
    """The A/A shape: the pairing is what makes this exactly zero, not merely small."""
    delta = paired_delta(CHAMPION, dict(CHAMPION), count=100)

    assert (delta.observed, delta.lower, delta.upper) == (0.0, 0.0, 0.0)
    assert not delta.improved


def test_refuses_unmatched_seeds():
    with pytest.raises(ValueError, match="seeds do not match"):
        paired_delta(CHAMPION, {Seed(1): CHALLENGER[Seed(1)]}, count=5)


def test_refuses_caches_from_different_eval_cohorts():
    fewer = a_cache(2, image_ids=INDEX.image_ids[:6])

    with pytest.raises(ValueError, match="same eval cohort"):
        paired_delta(CHAMPION, dict.fromkeys(SEEDS, fewer), count=5)

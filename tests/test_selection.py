"""Scoring, ranking and the batch record, all built rather than read.

Selection is pure, so every fixture here is constructed: a detection is six
numbers and a manifest row is nine fields, which means no case needs a model, a
parquet or a credential.

What is checked is the arithmetic of the score, that the two empty-detection
cases land at opposite ends of the ranking rather than together, that the
tie-break makes a batch reproducible across attempts -- the property the oracle's
idempotency key rests on -- and that the mix report keeps zeros distinguishable
from absences.
"""

import json

import pytest

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    ImageId,
    ManifestRow,
    Scene,
    Split,
    TimeOfDay,
    Weather,
)
from edge_ml_flywheel.evaluation.coco import Detection
from edge_ml_flywheel.selection import (
    BAND_HIGH,
    BLIND_SPOT,
    DECISIVE,
    Mix,
    Predictions,
    image_score,
    score_pool,
    select,
    selection_report,
    uncertainty,
)

CLASSES = CLASS_SET


def an_image(position: int) -> ImageId:
    return ImageId(f"{position:08x}-{position:08x}")


def a_detection(score: float, category: str = "car") -> Detection:
    """Coordinates are inert here -- selection reads `category` and `score`."""
    return Detection(category=category, x1=0.0, y1=0.0, x2=10.0, y2=10.0, score=score)


def predicting(of_image: dict[ImageId, list[Detection]] | None = None) -> Predictions:
    return Predictions(classes=CLASSES, of_image=of_image or {})


def a_row(image_id: ImageId, weather: Weather, timeofday: TimeOfDay) -> ManifestRow:
    return ManifestRow(
        image_id=image_id,
        split=Split.TRAIN,
        weather=weather,
        scene=Scene.CITY_STREET,
        timeofday=timeofday,
        n_boxes=0,
        box_areas=(),
        sha256="0" * 64,
        label_source="scalabel",
    )


POOL = tuple(an_image(position) for position in range(10))


# --- The per-detection score --------------------------------------------------


def test_uncertainty_peaks_at_the_decision_boundary() -> None:
    assert uncertainty(0.5) == 1.0
    assert uncertainty(0.0) == 0.0
    assert uncertainty(1.0) == 0.0


def test_a_near_certain_rejection_is_as_informative_as_a_near_certain_hit() -> None:
    """The reason this is distance from 0.5 rather than `1 - confidence`."""
    assert uncertainty(0.02) == pytest.approx(uncertainty(0.98))


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_a_confidence_outside_the_unit_range_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match="not a confidence"):
        uncertainty(confidence)


# --- The per-image score ------------------------------------------------------


def test_an_image_scores_the_mean_of_its_in_band_detections() -> None:
    score = image_score([a_detection(0.5), a_detection(0.25)])
    assert score == pytest.approx((1.0 + 0.5) / 2)


def test_confident_detections_are_left_out_of_the_mean() -> None:
    """A busy frame of easy objects must not outrank a quiet ambiguous one."""
    quiet = image_score([a_detection(0.5)])
    busy = image_score([a_detection(0.5)] + [a_detection(0.99)] * 20)
    assert busy == pytest.approx(quiet)


def test_one_ambiguous_box_does_not_beat_a_uniformly_unsure_frame() -> None:
    """Per-object rather than per-image maximum, which is the design's choice."""
    one_bad = image_score([a_detection(0.5), a_detection(0.2), a_detection(0.2)])
    uniformly_unsure = image_score([a_detection(0.45), a_detection(0.5), a_detection(0.55)])
    assert uniformly_unsure > one_bad


def test_seeing_nothing_and_seeing_everything_clearly_are_opposite_scores() -> None:
    assert image_score([]) == BLIND_SPOT
    assert image_score([a_detection(0.99), a_detection(0.99)]) == DECISIVE
    assert BLIND_SPOT > DECISIVE


def test_a_detection_exactly_on_the_band_edge_counts() -> None:
    assert image_score([a_detection(BAND_HIGH)]) == pytest.approx(uncertainty(BAND_HIGH))


# --- Scoring the pool ---------------------------------------------------------


def test_every_pool_image_is_scored_including_the_ones_with_no_detections() -> None:
    scores = score_pool(POOL, predicting({POOL[0]: [a_detection(0.5)]}))
    assert set(scores) == set(POOL)
    assert scores[POOL[1]] == BLIND_SPOT


def test_a_detection_for_an_image_outside_the_pool_is_refused() -> None:
    """Inference over the wrong image set -- the eval cohort, most consequentially."""
    with pytest.raises(ValueError, match="not in the pool"):
        score_pool(POOL[:2], predicting({an_image(99): [a_detection(0.5)]}))


def test_an_empty_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to rank"):
        score_pool([], predicting())


def test_a_detection_outside_the_class_set_is_refused() -> None:
    """The archive spells three categories differently from `det_20`, silently."""
    with pytest.raises(ValueError, match="outside the class set"):
        predicting({POOL[0]: [a_detection(0.5, "pedestrian")]})


# --- Ranking and taking -------------------------------------------------------


def test_the_top_of_the_ranking_is_what_is_bought() -> None:
    scores = {image_id: index / 10 for index, image_id in enumerate(POOL)}
    batch = select(POOL, scores, 3)
    assert batch == (POOL[9], POOL[8], POOL[7])


def test_ties_break_on_image_id_so_a_retry_proposes_the_same_batch() -> None:
    """The property `conventions.batch_digest` rests on.

    Every blind spot scores identically, so a tie at the batch boundary is
    guaranteed rather than incidental. Without the tie-break the two attempts
    below would differ and the oracle would charge twice.
    """
    scores = dict.fromkeys(POOL, BLIND_SPOT)
    first = select(POOL, scores, 4)
    second = select(reversed(POOL), scores, 4)
    assert first == second == tuple(sorted(POOL)[:4])


def test_the_batch_does_not_depend_on_the_pools_iteration_order() -> None:
    scores = {image_id: index / 10 for index, image_id in enumerate(POOL)}
    assert select(POOL, scores, 4) == select(reversed(POOL), scores, 4)


def test_an_unscored_pool_image_is_refused_rather_than_ranked_last() -> None:
    with pytest.raises(ValueError, match="no score"):
        select(POOL, {POOL[0]: 1.0}, 1)


@pytest.mark.parametrize(
    ("budget", "message"),
    [(0, "buys no batch"), (len(POOL) + 1, "end of the run")],
)
def test_an_impossible_budget_is_refused(budget: int, message: str) -> None:
    scores = dict.fromkeys(POOL, 0.5)
    with pytest.raises(ValueError, match=message):
        select(POOL, scores, budget)


# --- The batch record ---------------------------------------------------------

MANIFEST = {
    image_id: a_row(
        image_id,
        Weather.SNOWY if index < 4 else Weather.CLEAR,
        TimeOfDay.NIGHT if index < 4 else TimeOfDay.DAYTIME,
    )
    for index, image_id in enumerate(POOL)
}


def test_a_mix_reports_a_zero_for_every_value_it_did_not_see() -> None:
    """A missing key and a zero read the same in a chart and mean opposite things."""
    mix = Mix.of([POOL[0]], MANIFEST)
    assert mix.weather["snowy"] == 1
    assert mix.weather["foggy"] == 0
    assert set(mix.weather) == {member.value for member in Weather}
    assert set(mix.timeofday) == {member.value for member in TimeOfDay}


def test_an_image_with_no_manifest_row_is_refused() -> None:
    with pytest.raises(ValueError, match="no manifest row"):
        Mix.of([an_image(99)], MANIFEST)


def test_a_collapsed_batch_is_visible_against_the_remaining_pool() -> None:
    batch = POOL[:4]
    report = selection_report(batch, POOL, MANIFEST, predicting())
    assert report.batch.weather["snowy"] == 4
    assert report.batch.weather["clear"] == 0
    assert report.remaining.weather["snowy"] == 0
    assert report.remaining.weather["clear"] == 6
    assert report.batch.images + report.remaining.images == len(POOL)


def test_predicted_classes_cover_the_class_set_so_a_zero_is_the_warning() -> None:
    predictions = predicting({POOL[0]: [a_detection(0.5, "car"), a_detection(0.9, "car")]})
    report = selection_report(POOL[:2], POOL, MANIFEST, predictions)
    assert report.predicted_classes["car"] == 2
    assert report.predicted_classes["bus"] == 0
    assert set(report.predicted_classes) == set(CLASSES.names)


def test_blind_spots_in_the_batch_are_counted() -> None:
    predictions = predicting({POOL[0]: [a_detection(0.5)]})
    report = selection_report(POOL[:3], POOL, MANIFEST, predictions)
    assert report.blind_spots == 2


def test_a_batch_from_outside_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the pool"):
        selection_report([an_image(99)], POOL, MANIFEST, predicting())


def test_the_report_serializes_to_json() -> None:
    report = selection_report(POOL[:2], POOL, MANIFEST, predicting())
    document = json.loads(report.as_json())
    assert document["batch"]["images"] == 2
    assert document["remaining"]["images"] == 8
    assert document["blind_spots"] == 2

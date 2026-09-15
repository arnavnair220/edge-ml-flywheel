"""The two gates whose inputs exist, and the thresholds they read.

Both gates are pure, so the fixtures are built rather than read: `Cohorts` is a
frozen mapping and `PairedDelta` is five floats, which means every case here is
constructed directly instead of going through a parquet or a scoring pass.

Most cases run against `TIGHT` rather than the design's numbers, so a batch that
should pass is four images instead of two hundred and fifty. The design's values
are asserted once, on their own, and exercised once end to end -- a fixture sized
to the real floor makes every other test slower and none of them clearer.

What is checked is that each condition fails on its own, that a verdict names
every condition that failed rather than the first, and that the two failures
which look alike -- a class the model cannot detect and a class eval does not
contain -- are reported as the different bugs they are.
"""

import pytest

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    Cohort,
    ImageId,
    PartitionVersion,
)
from edge_ml_flywheel.evaluation.bootstrap import CONFIDENCE, RESAMPLES, PairedDelta
from edge_ml_flywheel.gates import DEFAULT, Gate, Thresholds, data_gate, quality_gate
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle.cohorts import Cohorts
from edge_ml_flywheel.oracle.labels import SoldLabel

VERSION = PartitionVersion(0)
CLASSES = CLASS_SET

# Small enough that a passing batch is four images. `min_mean_delta` keeps the
# design's value, because unlike the other two it is not a fixture-size knob --
# the deltas below are written against it.
TIGHT = Thresholds(min_new_images=4, min_instances_per_class=2, min_mean_delta=0.005)

BUDGET = 1_000


def an_image(position: int) -> ImageId:
    return ImageId(f"{position:08x}-{position:08x}")


POOL = tuple(an_image(position) for position in range(300))
IN_EVAL = an_image(9001)
IN_RESERVE = an_image(9002)
UNASSIGNED = an_image(9003)

COHORTS = Cohorts(
    partition_version=VERSION,
    of_image={
        **dict.fromkeys(POOL, Cohort.POOL),
        IN_EVAL: Cohort.EVAL,
        IN_RESERVE: Cohort.RESERVE,
    },
)


def a_box(category: str) -> Box:
    return Box(category=category, x1=10.0, y1=10.0, x2=110.0, y2=110.0)


def labels_for(*images: ImageId, categories: tuple[str, ...] = CLASSES.names) -> list[SoldLabel]:
    """One box of each named category per image, so instance counts are countable.

    A batch of *n* images therefore carries *n* instances of every class, which
    makes the coverage floor a function of the batch size alone.
    """
    return [
        SoldLabel(image_id=image, boxes=tuple(a_box(category) for category in categories))
        for image in images
    ]


A_CLEAN_BATCH = labels_for(*POOL[:8])


def a_delta(observed: float = 0.02, lower: float = 0.01, upper: float = 0.03) -> PairedDelta:
    return PairedDelta(
        observed=observed,
        lower=lower,
        upper=upper,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
    )


DETECTING = dict.fromkeys(CLASSES.names, 0.42)


# --- Thresholds ---------------------------------------------------------------


class TestThresholds:
    def test_the_defaults_are_the_design_values(self) -> None:
        assert DEFAULT.min_new_images == 250
        assert DEFAULT.min_instances_per_class == 10
        assert DEFAULT.min_mean_delta == 0.005

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("min_new_images", 0),
            ("min_instances_per_class", 0),
            ("min_mean_delta", 0.0),
            ("min_mean_delta", -0.01),
        ],
    )
    def test_a_threshold_that_admits_everything_is_refused(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            Thresholds(**{field: value})  # type: ignore[arg-type]

    def test_the_four_gate_names_are_reserved(self) -> None:
        """Edge and canary are unimplemented and still named (design section 4)."""
        assert [gate.value for gate in Gate] == ["data", "quality", "edge", "canary"]


# --- Data gate ----------------------------------------------------------------


class TestDataGate:
    def test_a_clean_batch_passes(self) -> None:
        result = data_gate(COHORTS, A_CLEAN_BATCH, CLASSES, BUDGET, TIGHT)
        assert result.passed
        assert result.gate == Gate.DATA
        assert "8 images" in result.reason

    def test_a_full_size_batch_passes_the_design_thresholds(self) -> None:
        """The one case sized to the real floors, so the defaults are exercised."""
        result = data_gate(COHORTS, labels_for(*POOL[:250]), CLASSES, BUDGET)
        assert result.passed

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (IN_EVAL, "in eval"),
            (IN_RESERVE, "in reserve"),
            (UNASSIGNED, "not assigned"),
        ],
    )
    def test_an_image_outside_the_pool_fails_and_says_which_cohort(
        self, image: ImageId, expected: str
    ) -> None:
        """The reason distinguishes three different bugs in three components."""
        batch = A_CLEAN_BATCH + labels_for(image)
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert not result.passed
        assert expected in result.reason

    def test_one_leaked_image_fails_the_whole_batch(self) -> None:
        batch = A_CLEAN_BATCH + labels_for(IN_EVAL)
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert not result.passed
        assert "1 of 9" in result.reason

    def test_a_batch_under_the_floor_fails(self) -> None:
        result = data_gate(COHORTS, labels_for(*POOL[:3]), CLASSES, BUDGET, TIGHT)
        assert not result.passed
        assert "under the floor of 4" in result.reason

    def test_a_batch_over_the_registered_budget_fails(self) -> None:
        result = data_gate(COHORTS, A_CLEAN_BATCH, CLASSES, 5, TIGHT)
        assert not result.passed
        assert "over the registered budget" in result.reason

    def test_a_class_short_of_instances_fails_and_is_named(self) -> None:
        """Three images give three of every class except the one left out of two."""
        batch = labels_for(*POOL[:2]) + labels_for(
            *POOL[2:5], categories=("car", "person", "truck")
        )
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert result.passed  # bus has 2, which meets TIGHT's floor of 2

        tighter = Thresholds(min_new_images=4, min_instances_per_class=3, min_mean_delta=0.005)
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, tighter)
        assert not result.passed
        assert "bus (2)" in result.reason
        assert "car" not in result.reason

    def test_a_category_outside_the_class_set_is_not_counted(self) -> None:
        """`train` is in the archive and in no class set, so it is not evidence."""
        batch = labels_for(*POOL[:4], categories=(*CLASSES.names, "train"))
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert result.passed

    def test_boxes_are_counted_not_images(self) -> None:
        """One image of ten cars does not cover a class the way ten images do."""
        crowded = labels_for(*POOL[:4], categories=("car",) * 10)
        result = data_gate(COHORTS, crowded, CLASSES, BUDGET, TIGHT)
        assert not result.passed
        assert "bus (0)" in result.reason

    def test_every_failure_is_reported_not_only_the_first(self) -> None:
        """The next attempt costs a training run, so one verdict names them all."""
        batch = labels_for(IN_EVAL, *POOL[:2])
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert not result.passed
        assert "in eval" in result.reason
        assert "under the floor" in result.reason

    def test_leakage_is_reported_first(self) -> None:
        batch = labels_for(IN_EVAL, *POOL[:2])
        result = data_gate(COHORTS, batch, CLASSES, BUDGET, TIGHT)
        assert result.reason.startswith("1 of 3 purchased images are not in pool")


# --- Quality gate -------------------------------------------------------------


class TestQualityGate:
    def test_a_clear_improvement_passes(self) -> None:
        result = quality_gate(a_delta(), DETECTING, CLASSES)
        assert result.passed
        assert result.gate == Gate.QUALITY
        assert "+0.0200" in result.reason

    def test_a_delta_under_the_threshold_fails(self) -> None:
        result = quality_gate(a_delta(observed=0.001, lower=0.0005), DETECTING, CLASSES)
        assert not result.passed
        assert "under the promotion threshold" in result.reason

    def test_a_band_touching_zero_fails(self) -> None:
        """The A/A test's expected verdict: a real-looking delta inside the noise."""
        result = quality_gate(a_delta(observed=0.02, lower=-0.004), DETECTING, CLASSES)
        assert not result.passed
        assert "does not clear zero" in result.reason

    def test_a_band_at_exactly_zero_fails(self) -> None:
        """`improved` is a strict inequality, and a boundary is not an improvement."""
        result = quality_gate(a_delta(lower=0.0), DETECTING, CLASSES)
        assert not result.passed

    def test_a_collapsed_class_fails_whatever_the_delta_says(self) -> None:
        collapsed = {**DETECTING, "bus": 0.0}
        result = quality_gate(a_delta(observed=0.5, lower=0.4), collapsed, CLASSES)
        assert not result.passed
        assert "scored zero AP" in result.reason
        assert "bus" in result.reason

    def test_a_class_absent_from_eval_is_a_different_failure(self) -> None:
        """Not folded into the collapse case: it says the check could not run."""
        unmeasured = {name: 0.42 for name in CLASSES.names if name != "bus"}
        result = quality_gate(a_delta(), unmeasured, CLASSES)
        assert not result.passed
        assert "no ground truth in eval" in result.reason
        assert "scored zero AP" not in result.reason

    def test_every_failure_is_reported_not_only_the_first(self) -> None:
        result = quality_gate(
            a_delta(observed=0.001, lower=-0.002), {**DETECTING, "bus": 0.0}, CLASSES
        )
        assert not result.passed
        assert "under the promotion threshold" in result.reason
        assert "does not clear zero" in result.reason
        assert "scored zero AP" in result.reason

    def test_the_band_parameters_are_recorded_in_the_reason(self) -> None:
        """A band means nothing without the parameters that produced it."""
        result = quality_gate(a_delta(), DETECTING, CLASSES)
        assert str(RESAMPLES) in result.reason
        assert "95%" in result.reason

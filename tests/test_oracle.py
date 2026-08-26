"""Tests for the cohort gate, which is where the eval guarantee lives.

The oracle reads ground truth straight out of `raw/labels/`, so no storage
boundary separates a purchasable `pool` label from the `eval` labels beside it.
What separates them is `oracle.cohorts`, and that makes these tests the
enforcement rather than a description of it. They are written accordingly:

- **Every non-pool cohort gets its own case**, not one "not pool" case. The three
  are refused for three different reasons and only one of them -- `eval` -- is a
  silent, unrecoverable failure if it ever gets through, so it should fail here
  by name rather than as a member of a set someone can later shrink.
- **The path is tested, not only the verdict.** A gate that returns False and a
  gate that prevents a key being built are different guarantees, and the second
  is the one claimed. `test_no_input_produces_a_val_key` is the whole point of
  the module.
- **The fixture holds all four cohorts on disk.** Refusing an eval image because
  its file is absent would prove nothing about the gate.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionVersion,
    Split,
    assignments_key,
    raw_label_key,
)
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle import cohorts as gate
from edge_ml_flywheel.oracle import labels as sold
from edge_ml_flywheel.partition import assign

VERSION = PartitionVersion(0)

# One image per cohort, so every refusal has a case and the pool has a control.
POOL = ImageId("00000001-00000001")
POOL_EMPTY = ImageId("00000002-00000002")
BOOTSTRAP = ImageId("00000003-00000003")
EVAL = ImageId("00000004-00000004")
RESERVE = ImageId("00000005-00000005")
UNASSIGNED = ImageId("0000dead-0000beef")

ASSIGNED = {
    POOL: Cohort.POOL,
    POOL_EMPTY: Cohort.POOL,
    BOOTSTRAP: Cohort.BOOTSTRAP,
    EVAL: Cohort.EVAL,
    RESERVE: Cohort.RESERVE,
}

# A coordinate that is not exactly representable in binary floating point, so a
# round-trip through anything that reformats numbers shows up.
AWKWARD = 380.1


def a_box(category: str = "car", x1: float = 380.4) -> Box:
    return Box(category=category, x1=x1, y1=404.8, x2=402.3, y2=416.7)


def a_label_document(image_id: ImageId, boxes: tuple[Box, ...]) -> dict[str, Any]:
    """The 2018 Scalabel shape, as `ingest.labels` documents it."""
    return {
        "name": f"{image_id}.jpg",
        "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
        "frames": [
            {
                "timestamp": 10000,
                "objects": [
                    *(
                        {
                            "category": box.category,
                            "box2d": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                        }
                        for box in boxes
                    ),
                    # Carries no `box2d`, so it must never be sold as a box.
                    {"category": "area/drivable", "poly2d": [[0, 0], [1, 1]]},
                ],
            }
        ],
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A tree laid out at the S3 keys, holding all four cohorts' labels.

    Every image gets a label file on disk, including `eval`. The gate has to be
    what refuses them -- if the file were simply missing, a removed check would
    still look like a pass here and fail only in production.
    """
    assign.write_parquet(
        [AssignmentRow(image_id=image, cohort=cohort) for image, cohort in ASSIGNED.items()],
        tmp_path / assignments_key(VERSION),
    )

    for image_id, cohort in ASSIGNED.items():
        drawn = (a_box(), a_box("truck", AWKWARD))
        boxes: tuple[Box, ...] = () if image_id == POOL_EMPTY else drawn
        split = Split.TRAIN if cohort in (Cohort.POOL, Cohort.BOOTSTRAP) else Split.VAL
        path = tmp_path / raw_label_key(image_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(a_label_document(image_id, boxes)), encoding="utf-8")

    return tmp_path


@pytest.fixture
def cohorts(root: Path) -> gate.Cohorts:
    return gate.Cohorts.read(root, VERSION)


class TestCohortIndex:
    def test_reads_every_assignment(self, cohorts: gate.Cohorts) -> None:
        assert {image: cohorts.cohort_of(image) for image in ASSIGNED} == ASSIGNED

    def test_an_unassigned_image_is_none_rather_than_a_cohort(self, cohorts: gate.Cohorts) -> None:
        """Distinct from every cohort, because it is a different bug.

        A known ID in the wrong cohort means the selector picked badly. An
        unknown one means it is not working from this partition at all.
        """
        assert cohorts.cohort_of(UNASSIGNED) is None

    def test_pool_is_the_purchasable_set(self, cohorts: gate.Cohorts) -> None:
        assert cohorts.pool == frozenset({POOL, POOL_EMPTY})

    def test_refuses_a_version_with_no_assignments(self, root: Path) -> None:
        with pytest.raises(ValueError, match="no assignments parquet"):
            gate.Cohorts.read(root, PartitionVersion(1))

    def test_pool_draws_from_train(self) -> None:
        """The split the gate derives its key from, pinned.

        If `COHORT_SPLIT` ever put `pool` and `eval` in one split, the key-path
        guarantee below would quietly weaken to a value check.
        """
        assert gate.PURCHASABLE_SPLIT is Split.TRAIN
        assert gate.PURCHASABLE is Cohort.POOL


class TestTheGate:
    def test_a_pool_batch_passes(self, cohorts: gate.Cohorts) -> None:
        gate.check_purchasable(cohorts, [POOL, POOL_EMPTY])

    @pytest.mark.parametrize(
        ("image_id", "reason"),
        [
            (EVAL, "in eval"),
            (BOOTSTRAP, "in bootstrap"),
            (RESERVE, "in reserve"),
            (UNASSIGNED, "not assigned"),
        ],
    )
    def test_everything_outside_the_pool_is_refused(
        self, cohorts: gate.Cohorts, image_id: ImageId, reason: str
    ) -> None:
        """One case per cohort, by name.

        `eval` above all: it is the only one whose failure is silent and
        unrecoverable, since a model trained on the exam scores better on
        exactly the set it is measured against and every gate passes.
        """
        with pytest.raises(gate.NotPurchasableError, match=reason):
            gate.check_purchasable(cohorts, [image_id])

    def test_one_bad_image_refuses_the_whole_batch(self, cohorts: gate.Cohorts) -> None:
        """Not a filter. Selling the rest would charge for a batch nobody asked
        for and turn a selector reaching into `eval` into a quiet short cycle."""
        with pytest.raises(gate.NotPurchasableError, match="1 of 3"):
            gate.check_purchasable(cohorts, [POOL, EVAL, POOL_EMPTY])

    def test_a_duplicate_is_refused(self, cohorts: gate.Cohorts) -> None:
        """One label billed twice, which the ledger cannot see.

        The batch is a set of images to the oracle and a count to the budget,
        and this is the one place those two views meet.
        """
        with pytest.raises(gate.NotPurchasableError, match="more than once"):
            gate.check_purchasable(cohorts, [POOL, POOL])

    def test_an_empty_batch_is_refused(self, cohorts: gate.Cohorts) -> None:
        with pytest.raises(gate.NotPurchasableError, match="no images"):
            gate.check_purchasable(cohorts, [])

    def test_refusals_name_the_cohort(self, cohorts: gate.Cohorts) -> None:
        """A refusal that says only "not purchasable" sends a reader hunting."""
        assert gate.refusals(cohorts, [POOL, EVAL, BOOTSTRAP]) == {
            EVAL: "in eval",
            BOOTSTRAP: "in bootstrap",
        }


class TestTheKeyPath:
    """The claim is that a refused image is unread, not merely unsold."""

    def test_a_pool_image_resolves_to_a_train_key(self, cohorts: gate.Cohorts) -> None:
        assert sold.label_key(cohorts, POOL) == raw_label_key(POOL, Split.TRAIN)

    @pytest.mark.parametrize("image_id", [EVAL, BOOTSTRAP, RESERVE, UNASSIGNED])
    def test_no_key_exists_for_anything_outside_the_pool(
        self, cohorts: gate.Cohorts, image_id: ImageId
    ) -> None:
        with pytest.raises(gate.NotPurchasableError):
            sold.label_key(cohorts, image_id)

    def test_no_input_produces_a_val_key(self, cohorts: gate.Cohorts) -> None:
        """The centrepiece.

        Over every image the partition knows about plus one it does not, the key
        builder either raises or returns a key under `train/`. `eval` and
        `reserve` live under `val/`, so a key there is the shape contamination
        would take, and this asserts the function cannot produce one -- rather
        than that the callers written so far do not ask it to.
        """
        produced: list[str] = []
        for image_id in [*ASSIGNED, UNASSIGNED]:
            try:
                produced.append(sold.label_key(cohorts, image_id))
            except gate.NotPurchasableError:
                continue

        assert produced == [
            raw_label_key(POOL, Split.TRAIN),
            raw_label_key(POOL_EMPTY, Split.TRAIN),
        ]
        assert not any(f"/{Split.VAL.value}/" in key for key in produced)

    def test_reading_an_eval_label_raises_before_the_file_is_opened(
        self, root: Path, cohorts: gate.Cohorts
    ) -> None:
        """The eval label exists on disk, so this can only be the gate."""
        assert (root / raw_label_key(EVAL, Split.VAL)).is_file()
        with pytest.raises(gate.NotPurchasableError, match="in eval"):
            sold.read_label(root, cohorts, EVAL)


class TestReadingALabel:
    def test_boxes_arrive_intact(self, root: Path, cohorts: gate.Cohorts) -> None:
        label = sold.read_label(root, cohorts, POOL)
        assert label.boxes == (a_box(), a_box("truck", AWKWARD))

    def test_a_polygon_is_not_sold_as_a_box(self, root: Path, cohorts: gate.Cohorts) -> None:
        """Every fixture document carries an `area/drivable` poly2d.

        `ingest.labels` picks boxes by structure rather than by a class list, and
        this asserts the oracle inherited that rather than re-deriving it.
        """
        label = sold.read_label(root, cohorts, POOL)
        assert {box.category for box in label.boxes} == {"car", "truck"}

    def test_an_image_with_no_boxes_is_still_a_label(
        self, root: Path, cohorts: gate.Cohorts
    ) -> None:
        """A real label and a real charge: "nothing here" is a thing to learn."""
        assert sold.read_label(root, cohorts, POOL_EMPTY).boxes == ()

    def test_local_fetch_binds_the_gate(self, root: Path, cohorts: gate.Cohorts) -> None:
        """A `Fetch` carries the gate with it, so a purchase cannot bypass it."""
        fetch = sold.local_fetch(root, cohorts)
        assert fetch(POOL).image_id == POOL
        with pytest.raises(gate.NotPurchasableError):
            fetch(EVAL)


class TestBoxEncoding:
    def test_round_trips(self) -> None:
        boxes = (a_box(), a_box("bus", AWKWARD))
        assert sold.decode_boxes(sold.encode_boxes(boxes)) == boxes

    def test_an_awkward_float_survives_exactly(self) -> None:
        """Not `approx`. A coordinate that comes back near the archive's is a
        different label from the one that was bought."""
        decoded = sold.decode_boxes(sold.encode_boxes((a_box("car", AWKWARD),)))
        assert decoded[0].x1 == AWKWARD

    def test_no_boxes_round_trips(self) -> None:
        assert sold.decode_boxes(sold.encode_boxes(())) == ()

    def test_the_encoding_is_compact(self) -> None:
        assert " " not in sold.encode_boxes((a_box(),))

    def test_a_row_of_the_wrong_width_raises(self) -> None:
        """Rather than defaulting a corner, which is a rectangle in the wrong place."""
        with pytest.raises(ValueError, match="not a"):
            sold.decode_boxes(json.dumps([["car", 1.0, 2.0]]))

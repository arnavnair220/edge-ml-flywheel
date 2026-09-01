"""Unit tests for the bootstrap and eval label files, on a staged temp tree.

The fixture stages label documents for the labeled cohorts **and for nothing
else**. That is deliberate and is load-bearing: the partition role can read all
80,000 label documents in the bucket, so the property worth testing is not that
this code refuses a `pool` label when asked, but that it never asks. A read that
reached for one fails here as a missing file rather than passing quietly.

`TestKeys` is where the guarantee is stated directly, and `TestCheck` is the
assertion that runs in the job -- the one that has to hold even if the key list
is wrong.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    LABELED_COHORTS,
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionVersion,
    assignments_key,
    cohort_labels_key,
    raw_label_key,
)
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle.labels import decode_boxes
from edge_ml_flywheel.partition import assign, cohort_labels

V0 = PartitionVersion(0)

# Two of each cohort, which is enough for every property here: the cohorts are
# distinguished by name rather than by size, and a set comparison over four
# images fails the same way it would over 13,000.
COHORTS = (
    Cohort.BOOTSTRAP,
    Cohort.BOOTSTRAP,
    Cohort.POOL,
    Cohort.POOL,
    Cohort.EVAL,
    Cohort.EVAL,
    Cohort.RESERVE,
    Cohort.RESERVE,
)


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


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
                    {
                        "category": box.category,
                        "box2d": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                    }
                    for box in boxes
                ],
            }
        ],
    }


def assignments() -> tuple[AssignmentRow, ...]:
    return tuple(
        AssignmentRow(image_id=an_image_id(index), cohort=cohort)
        for index, cohort in enumerate(COHORTS)
    )


def boxes_for(image_id: ImageId) -> tuple[Box, ...]:
    """A distinct box per image, so a row landing under the wrong ID is visible."""
    return (a_box(x1=float(int(image_id[:8], 16) % 1000)), a_box(category="person"))


def a_staged_partition(root: Path, rows: tuple[AssignmentRow, ...] | None = None) -> Path:
    """Assignments, plus label documents for the labeled cohorts only.

    A `pool` or `reserve` document is never written. Anything that tries to read
    one raises `FileNotFoundError` at the read, which is the loud version of the
    failure this whole module exists to prevent.
    """
    rows = rows if rows is not None else assignments()
    assign.write_parquet(rows, root / assignments_key(V0))

    for row in rows:
        if row.cohort not in LABELED_COHORTS:
            continue
        path = root / raw_label_key(row.image_id, COHORT_SPLIT[row.cohort])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(a_label_document(row.image_id, boxes_for(row.image_id))), encoding="utf-8"
        )
    return root


def labels_of(rows: tuple[AssignmentRow, ...], cohort: Cohort) -> list[cohort_labels.CohortLabel]:
    return [
        cohort_labels.CohortLabel(image_id=image_id, boxes=boxes_for(image_id))
        for image_id in cohort_labels.image_ids(rows, cohort)
    ]


# --- Which labels are addressable at all ---


class TestKeys:
    def test_a_labeled_cohort_resolves_to_its_own_split(self) -> None:
        image_id = an_image_id(0)
        assert cohort_labels.label_key(Cohort.BOOTSTRAP, image_id).endswith(
            f"train/{image_id}.json"
        )
        assert cohort_labels.label_key(Cohort.EVAL, image_id).endswith(f"val/{image_id}.json")

    @pytest.mark.parametrize("cohort", [Cohort.POOL, Cohort.RESERVE])
    def test_an_unlabeled_cohort_has_no_addressable_label(self, cohort: Cohort) -> None:
        # The refusal happens while the caller still holds a cohort, so no path to
        # a withheld label is ever produced for it to then decline to use.
        with pytest.raises(ValueError, match="has no labels of its own"):
            cohort_labels.label_key(cohort, an_image_id(0))

    def test_the_key_list_is_exactly_the_labeled_cohorts(self) -> None:
        rows = assignments()
        listed = cohort_labels.keys(rows)
        expected = {
            raw_label_key(row.image_id, COHORT_SPLIT[row.cohort])
            for row in rows
            if row.cohort in LABELED_COHORTS
        }
        assert set(listed) == expected
        assert len(listed) == len(expected)

    def test_no_pool_label_is_named(self) -> None:
        # The statement the widened `raw/labels/` grant is worth: the job can read
        # all 80,000 documents and this list is what it asks for.
        rows = assignments()
        withheld = {
            an_image_id(index) for index, cohort in enumerate(COHORTS) if cohort is Cohort.POOL
        }
        assert withheld
        assert not any(image_id in key for key in cohort_labels.keys(rows) for image_id in withheld)

    def test_the_key_list_is_sorted_within_a_cohort(self) -> None:
        # One ordering for the key list, the read and the parquet, which is what
        # makes a re-run byte-identical.
        rows = assignments()
        for cohort in LABELED_COHORTS:
            ids = cohort_labels.image_ids(rows, cohort)
            assert list(ids) == sorted(ids)


# --- The assertion that runs in the job ---


class TestCheck:
    def test_what_collect_produced_passes(self, tmp_path: Path) -> None:
        rows = assignments()
        stage = a_staged_partition(tmp_path)
        for cohort in LABELED_COHORTS:
            cohort_labels.check(cohort_labels.collect(stage, rows, cohort), rows, cohort)

    def test_an_image_from_another_cohort_is_refused(self) -> None:
        # The failure the check exists for. A `pool` image in the eval label file
        # is ground truth escaping the wall by the one route IAM cannot see.
        rows = assignments()
        strayed = [
            *labels_of(rows, Cohort.EVAL),
            cohort_labels.CohortLabel(image_id=an_image_id(2), boxes=()),
        ]
        with pytest.raises(ValueError, match="are not in this cohort"):
            cohort_labels.check(strayed, rows, Cohort.EVAL)

    def test_a_missing_image_is_refused(self) -> None:
        rows = assignments()
        short = labels_of(rows, Cohort.EVAL)[:-1]
        with pytest.raises(ValueError, match="have no label"):
            cohort_labels.check(short, rows, Cohort.EVAL)

    def test_a_repeated_image_is_refused(self) -> None:
        rows = assignments()
        doubled = labels_of(rows, Cohort.EVAL)
        with pytest.raises(ValueError, match="appear more than once"):
            cohort_labels.check([*doubled, doubled[0]], rows, Cohort.EVAL)

    def test_a_stray_is_reported_before_a_shortfall(self) -> None:
        # Both are wrong and only one is a leak, so the leak is what the message
        # names.
        rows = assignments()
        both = [
            *labels_of(rows, Cohort.EVAL)[:-1],
            cohort_labels.CohortLabel(image_id=an_image_id(2), boxes=()),
        ]
        with pytest.raises(ValueError, match="are not in this cohort"):
            cohort_labels.check(both, rows, Cohort.EVAL)


# --- The files ---


class TestWrite:
    def test_both_files_land_at_the_conventional_keys(self, tmp_path: Path) -> None:
        stage = a_staged_partition(tmp_path)
        cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))
        for cohort in LABELED_COHORTS:
            assert (stage / cohort_labels_key(V0, cohort)).is_file()

    def test_no_file_is_written_for_a_withheld_cohort(self, tmp_path: Path) -> None:
        stage = a_staged_partition(tmp_path)
        cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))
        written = {
            path.parent.name
            for path in (stage / f"derived/partition_version=v{V0:03d}/labels").rglob("*.parquet")
        }
        assert written == {f"cohort={cohort.value}" for cohort in LABELED_COHORTS}

    def test_the_columns_are_the_schema_of_record(self, tmp_path: Path) -> None:
        stage = a_staged_partition(tmp_path)
        cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))
        table = pq.read_table(stage / cohort_labels_key(V0, Cohort.EVAL))
        assert tuple(table.column_names) == ("image_id", "boxes")
        assert table.schema == cohort_labels.SCHEMA

    def test_the_boxes_round_trip_through_the_purchase_encoding(self, tmp_path: Path) -> None:
        # Decoded with `oracle.labels.decode_boxes`, because training reads a
        # bootstrap label and a bought one out of the same column.
        rows = assignments()
        stage = a_staged_partition(tmp_path)
        cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))

        table = pq.read_table(stage / cohort_labels_key(V0, Cohort.BOOTSTRAP))
        for image_id, encoded in zip(
            table.column("image_id").to_pylist(), table.column("boxes").to_pylist(), strict=True
        ):
            assert decode_boxes(encoded) == boxes_for(ImageId(image_id))
        assert table.column("image_id").to_pylist() == list(
            cohort_labels.image_ids(rows, Cohort.BOOTSTRAP)
        )

    def test_two_runs_write_byte_identical_parquet(self, tmp_path: Path) -> None:
        first = a_staged_partition(tmp_path / "first")
        second = a_staged_partition(tmp_path / "second")
        cohort_labels.write(first, V0, assign.read_assignments(first, V0))
        cohort_labels.write(second, V0, assign.read_assignments(second, V0))
        for cohort in LABELED_COHORTS:
            assert (first / cohort_labels_key(V0, cohort)).read_bytes() == (
                second / cohort_labels_key(V0, cohort)
            ).read_bytes()

    def test_an_image_with_no_boxes_is_a_legitimate_label(self, tmp_path: Path) -> None:
        # "There is nothing here" is an answer the model has to learn, and it is
        # the same judgement `oracle.labels.SoldLabel` documents.
        rows = assignments()
        stage = a_staged_partition(tmp_path)
        empty = cohort_labels.image_ids(rows, Cohort.EVAL)[0]
        path = stage / raw_label_key(empty, COHORT_SPLIT[Cohort.EVAL])
        path.write_text(json.dumps(a_label_document(empty, ())), encoding="utf-8")

        cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))
        table = pq.read_table(stage / cohort_labels_key(V0, Cohort.EVAL))
        assert decode_boxes(table.column("boxes").to_pylist()[0]) == ()

    def test_a_missing_label_document_stops_the_job(self, tmp_path: Path) -> None:
        # The staged copy is 13,000 separate objects, so a partial stage is the
        # realistic failure. It has to be loud: a label file short by one image is
        # a smaller eval nothing announces.
        rows = assignments()
        stage = a_staged_partition(tmp_path)
        (
            stage
            / raw_label_key(
                cohort_labels.image_ids(rows, Cohort.EVAL)[0], COHORT_SPLIT[Cohort.EVAL]
            )
        ).unlink()
        with pytest.raises(FileNotFoundError):
            cohort_labels.write(stage, V0, assign.read_assignments(stage, V0))

    def test_a_missing_assignments_parquet_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no assignments parquet"):
            assign.read_assignments(tmp_path, V0)

    def test_the_assignments_round_trip(self, tmp_path: Path) -> None:
        stage = a_staged_partition(tmp_path)
        assert assign.read_assignments(stage, V0) == assignments()

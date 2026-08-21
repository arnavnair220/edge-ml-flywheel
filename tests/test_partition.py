"""Unit tests for the partitioner, on a synthetic manifest in a temp directory.

The properties that matter are not about scale -- exactly-once, the split
boundary, and the same seed producing the same cohorts on any machine are all
visible at eighty rows. The exceptions are the tests that go through
`partition.write`, which use a manifest sized for partition version 0 so that the
real spec's quotas are what get drawn.

`TestReproducibility` pins the ticket `sha256` produces. A failure there says
partition version 0 has moved, which is the one change this suite exists to make
loud.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionSeed,
    PartitionSpec,
    PartitionVersion,
    Scene,
    Split,
    TimeOfDay,
    Weather,
    assignments_key,
    columns,
    manifest_key,
    partition_manifest_key,
    partition_spec,
)
from edge_ml_flywheel.partition import assign as partition
from edge_ml_flywheel.partition.__main__ import _parser, main

SEED = PartitionSeed(20260819)
V0 = PartitionVersion(0)

# Sixty train and twenty val, and cohort sizes scaled down from version 0 by a
# thousand -- except `reserve`, which absorbs the difference so that `val` is
# covered exactly.
TRAIN = 60
VAL = 20

SPEC = PartitionSpec(
    seed=SEED,
    sizes={
        Cohort.BOOTSTRAP: 8,
        Cohort.POOL: 52,
        Cohort.EVAL: 5,
        Cohort.RESERVE: 15,
    },
)

_DEFAULT_TAGS = (Weather.CLEAR.value, Scene.CITY_STREET.value, TimeOfDay.DAYTIME.value)


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


def a_manifest(
    train: int = TRAIN,
    val: int = VAL,
    tags: dict[int, tuple[str, str, str]] | None = None,
    **overrides: Any,
) -> pa.Table:
    """A manifest carrying only the columns the partitioner reads.

    `tags` replaces the three tag values for individual rows by index, which is
    what the composition test needs; every other row is a clear city daytime
    frame.
    """
    tags = tags or {}
    rows = [tags.get(index, _DEFAULT_TAGS) for index in range(train + val)]

    data: dict[str, list[Any]] = {
        "image_id": [an_image_id(index) for index in range(train + val)],
        "split": [Split.TRAIN.value] * train + [Split.VAL.value] * val,
        "weather": [row[0] for row in rows],
        "scene": [row[1] for row in rows],
        "timeofday": [row[2] for row in rows],
    }
    return pa.table(data | overrides)


def a_staged_manifest(root: Path) -> Path:
    """A staged tree holding a manifest sized for partition version 0."""
    path = root / manifest_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(a_manifest(train=70_000, val=10_000), path)
    return root


def in_cohort(rows: tuple[AssignmentRow, ...], cohort: Cohort) -> set[ImageId]:
    return {row.image_id for row in rows if row.cohort is cohort}


# --- Partition specs ---


class TestPartitionSpec:
    def test_version_zero_is_the_designed_partition(self) -> None:
        # Golden for the reason the layout strings are golden: every cycle's
        # result is conditional on these four numbers and this seed, and none of
        # them can be corrected once a partition has been drawn from them.
        spec = partition_spec(V0)
        assert spec.seed == 20260819
        assert spec.sizes == {
            Cohort.BOOTSTRAP: 8_000,
            Cohort.POOL: 62_000,
            Cohort.EVAL: 5_000,
            Cohort.RESERVE: 5_000,
        }
        assert sum(spec.sizes.values()) == 80_000

    def test_an_undefined_version_is_refused(self) -> None:
        # A version nobody wrote down has no seed, so it has no partition.
        with pytest.raises(ValueError, match="partition version 7 is not defined"):
            partition_spec(PartitionVersion(7))

    def test_a_spec_must_size_every_cohort(self) -> None:
        # A cohort added to the enum without a size lands here rather than in a
        # draw that leaves images unassigned.
        with pytest.raises(ValueError, match="sizes no cohort"):
            PartitionSpec(seed=SEED, sizes={Cohort.BOOTSTRAP: 8, Cohort.POOL: 52})

    def test_a_spec_must_not_size_a_cohort_negatively(self) -> None:
        with pytest.raises(ValueError, match="negatively"):
            PartitionSpec(
                seed=SEED,
                sizes={Cohort.BOOTSTRAP: -1, Cohort.POOL: 61, Cohort.EVAL: 5, Cohort.RESERVE: 15},
            )

    def test_quotas_are_per_split_in_draw_order(self) -> None:
        # Draw order is what a later version inherits, so it is pinned.
        assert SPEC.quotas(Split.TRAIN) == ((Cohort.BOOTSTRAP, 8), (Cohort.POOL, 52))
        assert SPEC.quotas(Split.VAL) == ((Cohort.EVAL, 5), (Cohort.RESERVE, 15))

    def test_every_cohort_draws_from_a_named_split(self) -> None:
        # A cohort added without a split decision lands here.
        assert set(COHORT_SPLIT) == set(Cohort)


# --- Exactly-once ---


class TestExactlyOnce:
    def test_every_image_gets_one_cohort(self) -> None:
        rows = partition.assign(a_manifest(), SPEC)
        assert len(rows) == TRAIN + VAL
        assert len({row.image_id for row in rows}) == TRAIN + VAL

    def test_cohort_counts_are_the_specified_sizes(self) -> None:
        rows = partition.assign(a_manifest(), SPEC)
        assert {cohort: len(in_cohort(rows, cohort)) for cohort in Cohort} == dict(SPEC.sizes)

    def test_the_cohorts_are_disjoint_and_complete(self) -> None:
        rows = partition.assign(a_manifest(), SPEC)
        drawn = [in_cohort(rows, cohort) for cohort in Cohort]
        assert set.union(*drawn) == {an_image_id(index) for index in range(TRAIN + VAL)}
        assert sum(len(ids) for ids in drawn) == len(set.union(*drawn))

    def test_the_check_passes_what_the_draw_produced(self) -> None:
        manifest = a_manifest()
        partition.check(partition.assign(manifest, SPEC), manifest, SPEC)

    def test_quotas_that_do_not_cover_a_split_are_refused(self) -> None:
        # The failure this catches is a spec written against a different pool
        # size: the draw would fill its quotas and leave the tail of the ordering
        # with no cohort at all.
        short = PartitionSpec(
            seed=SEED,
            sizes={Cohort.BOOTSTRAP: 8, Cohort.POOL: 51, Cohort.EVAL: 5, Cohort.RESERVE: 15},
        )
        with pytest.raises(ValueError, match=r"quotas total 59 against 60 images"):
            partition.assign(a_manifest(), short)

    def test_a_duplicate_manifest_image_is_refused(self) -> None:
        ids = [an_image_id(index) for index in range(TRAIN + VAL)]
        ids[1] = ids[0]
        with pytest.raises(ValueError, match="appears twice in the manifest"):
            partition.assign(a_manifest(image_id=ids), SPEC)

    def test_the_check_catches_a_cohort_drawn_to_the_wrong_size(self) -> None:
        # Splits stay valid here -- both cohorts draw from `train` -- so this is
        # the size arithmetic on its own.
        manifest = a_manifest()
        rows = partition.assign(manifest, SPEC)
        moved = tuple(
            AssignmentRow(image_id=row.image_id, cohort=Cohort.BOOTSTRAP)
            if row.cohort is Cohort.POOL and row.image_id == min(in_cohort(rows, Cohort.POOL))
            else row
            for row in rows
        )
        with pytest.raises(ValueError, match="cohort sizes disagree"):
            partition.check(moved, manifest, SPEC)

    def test_the_check_catches_a_missing_image(self) -> None:
        manifest = a_manifest()
        rows = partition.assign(manifest, SPEC)
        with pytest.raises(ValueError, match="assignments over 80 manifest images"):
            partition.check(rows[:-1], manifest, SPEC)


class TestTheSplitBoundary:
    def test_trainable_cohorts_hold_only_train_images(self) -> None:
        rows = partition.assign(a_manifest(), SPEC)
        train = {an_image_id(index) for index in range(TRAIN)}
        assert in_cohort(rows, Cohort.BOOTSTRAP) | in_cohort(rows, Cohort.POOL) == train

    def test_held_out_cohorts_hold_only_val_images(self) -> None:
        rows = partition.assign(a_manifest(), SPEC)
        val = {an_image_id(index) for index in range(TRAIN, TRAIN + VAL)}
        assert in_cohort(rows, Cohort.EVAL) | in_cohort(rows, Cohort.RESERVE) == val

    def test_the_withheld_test_split_is_refused(self) -> None:
        # The third of the three places the exclusion is enforced, after the
        # extraction filter and the verification suite. `Split` has no TEST
        # member, so those images cannot reach a cohort.
        manifest = a_manifest(split=["test"] * (TRAIN + VAL))
        with pytest.raises(ValueError, match="manifest split is 'test'"):
            partition.assign(manifest, SPEC)

    def test_the_check_catches_an_image_on_the_wrong_side_of_the_split(self) -> None:
        # Unreachable through `assign`, which draws per split. The check is for
        # the version of `_draw` that someone edits later.
        manifest = a_manifest(train=1, val=0)
        tampered = (AssignmentRow(image_id=an_image_id(0), cohort=Cohort.EVAL),)
        spec = PartitionSpec(
            seed=SEED,
            sizes={Cohort.BOOTSTRAP: 0, Cohort.POOL: 1, Cohort.EVAL: 0, Cohort.RESERVE: 0},
        )
        with pytest.raises(ValueError, match="is a train image in eval"):
            partition.check(tampered, manifest, spec)


# --- Reproducibility ---


class TestReproducibility:
    def test_the_same_seed_draws_the_same_cohorts(self) -> None:
        assert partition.assign(a_manifest(), SPEC) == partition.assign(a_manifest(), SPEC)

    def test_a_different_seed_draws_different_cohorts(self) -> None:
        other = PartitionSpec(seed=PartitionSeed(SEED + 1), sizes=SPEC.sizes)
        assert partition.assign(a_manifest(), other) != partition.assign(a_manifest(), SPEC)

    def test_manifest_row_order_does_not_change_the_draw(self) -> None:
        # The reason the draw is a ticket per image rather than a shuffle of the
        # split. The manifest is built by walking a directory, so a re-ingest that
        # lists files in another order must not repartition the dataset.
        order = list(reversed(range(TRAIN))) + list(reversed(range(TRAIN, TRAIN + VAL)))
        shuffled = a_manifest(image_id=[an_image_id(index) for index in order])
        assert partition.assign(shuffled, SPEC) == partition.assign(a_manifest(), SPEC)

    def test_a_ticket_depends_on_the_seed_and_the_image_alone(self) -> None:
        # Pinned rather than recomputed, so a change to the ticket rule is a
        # failing test instead of a silently different partition.
        assert partition.ticket(SEED, ImageId("0000f77c-6257be58")) == (
            "bf53defeb82da94c8d6015aab129671423ec59c0f1dd738f3683f605ee05823e"
        )

    def test_growing_eval_at_the_same_seed_keeps_the_earlier_eval(self) -> None:
        # Why `eval` is drawn before `reserve`: a version that spends the reserve
        # is a superset, so every image an earlier eval measured is still
        # measured.
        grown = PartitionSpec(
            seed=SEED,
            sizes={Cohort.BOOTSTRAP: 8, Cohort.POOL: 52, Cohort.EVAL: 12, Cohort.RESERVE: 8},
        )
        small = in_cohort(partition.assign(a_manifest(), SPEC), Cohort.EVAL)
        large = in_cohort(partition.assign(a_manifest(), grown), Cohort.EVAL)
        assert small < large


class TestUniformity:
    def test_a_drawn_cohort_tracks_the_split_it_came_from(self) -> None:
        # Uniform, so `eval` inherits `val`'s tag distribution to within sampling
        # error and no stratification is applied anywhere. A known answer rather
        # than a statistical gamble: the seed fixes the draw.
        snowy = (Weather.SNOWY.value, Scene.HIGHWAY.value, TimeOfDay.NIGHT.value)
        manifest = a_manifest(
            train=600,
            val=400,
            tags=dict.fromkeys(range(600, 800), snowy),
        )
        spec = PartitionSpec(
            seed=SEED,
            sizes={Cohort.BOOTSTRAP: 100, Cohort.POOL: 500, Cohort.EVAL: 200, Cohort.RESERVE: 200},
        )
        counts = partition.composition(manifest, partition.assign(manifest, spec))

        # Half of `val` is snowy, so about half of a 200-image `eval` should be.
        # The standard error on that draw is five images.
        assert 85 <= counts[Cohort.EVAL]["weather"][Weather.SNOWY.value] <= 115


# --- The files ---


class TestWrite:
    def test_both_files_land_at_the_conventional_keys(self, tmp_path: Path) -> None:
        stage = a_staged_manifest(tmp_path)
        partition.write(stage, V0)
        assert (stage / assignments_key(V0)).is_file()
        assert (stage / partition_manifest_key(V0)).is_file()

    def test_the_parquet_columns_are_the_schema_of_record(self, tmp_path: Path) -> None:
        stage = a_staged_manifest(tmp_path)
        partition.write(stage, V0)
        table = pq.read_table(stage / assignments_key(V0))
        assert tuple(table.column_names) == columns(AssignmentRow)
        assert table.schema == partition.SCHEMA

    def test_the_rows_round_trip(self, tmp_path: Path) -> None:
        stage = a_staged_manifest(tmp_path)
        rows = partition.write(stage, V0)
        table = pq.read_table(stage / assignments_key(V0))
        assert table.column("image_id").to_pylist() == [row.image_id for row in rows]
        assert table.column("cohort").to_pylist() == [row.cohort.value for row in rows]

    def test_two_runs_write_byte_identical_parquet(self, tmp_path: Path) -> None:
        # Sorted by image_id on the way out, so "this is the same partition" is a
        # checksum rather than a claim.
        first = a_staged_manifest(tmp_path / "first")
        second = a_staged_manifest(tmp_path / "second")
        partition.write(first, V0)
        partition.write(second, V0)
        assert (first / assignments_key(V0)).read_bytes() == (
            second / assignments_key(V0)
        ).read_bytes()

    def test_the_document_records_the_seed_and_the_sizes(self, tmp_path: Path) -> None:
        stage = a_staged_manifest(tmp_path)
        partition.write(stage, V0)
        document = json.loads((stage / partition_manifest_key(V0)).read_text(encoding="utf-8"))
        assert document["seed"] == 20260819
        assert document["partition_version"] == 0
        assert document["cohorts"]["eval"] == {"images": 5_000, "split": "val"}
        assert "sha256(seed:image_id)" in document["draw"]

    def test_a_missing_manifest_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no manifest parquet"):
            partition.write(tmp_path, V0)

    def test_every_manifest_part_is_read(self, tmp_path: Path) -> None:
        # Ingest writes one part today. Reading the prefix rather than one key is
        # what keeps that a property of ingest instead of an assumption here.
        for part in range(2):
            ids = [an_image_id(part * 40 + index) for index in range(40)]
            path = tmp_path / manifest_key(part)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(a_manifest(train=30, val=10, image_id=ids), path)
        assert partition.read_manifest(tmp_path).num_rows == 80


# --- The command line ---


class TestRecorded:
    """The guard on redrawing a version that is already in the bucket."""

    def test_a_matching_record_agrees(self) -> None:
        spec = partition_spec(V0)
        assert partition.disagreements(partition.document(V0, spec), V0, spec) == ()

    def test_the_written_document_agrees_with_itself(self, tmp_path: Path) -> None:
        # `drawn_at` is in the file and not in the comparison, so a re-run of one
        # version agrees with what it wrote last time.
        spec = partition_spec(V0)
        path = tmp_path / "_partition.json"
        partition.write_document(V0, spec, path)
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert "drawn_at" in recorded
        assert partition.disagreements(recorded, V0, spec) == ()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", 20260820),
            ("draw", "uniform shuffle with random.sample"),
            ("cohorts", {"eval": {"images": 10_000, "split": "val"}}),
        ],
        ids=["an-edited-seed", "an-edited-rule", "an-edited-size"],
    )
    def test_an_edited_version_disagrees(self, field: str, value: object) -> None:
        # Each of the three changes the assignment under a key every existing run
        # is measured against, which is what makes it a new version.
        spec = partition_spec(V0)
        recorded = partition.document(V0, spec) | {field: value}
        assert partition.disagreements(recorded, V0, spec) == (field,)

    def test_a_missing_record_is_the_first_run(self, tmp_path: Path) -> None:
        main(
            [
                "verify-recorded",
                "--recorded",
                str(tmp_path / "absent.json"),
                "--partition-version",
                "0",
            ]
        )

    def test_an_edited_seed_stops_the_build(self, tmp_path: Path) -> None:
        path = tmp_path / "_partition.json"
        path.write_text(
            json.dumps(partition.document(V0, partition_spec(V0)) | {"seed": 1}), encoding="utf-8"
        )
        with pytest.raises(
            SystemExit, match="already recorded in the bucket with a different seed"
        ):
            main(["verify-recorded", "--recorded", str(path), "--partition-version", "0"])


class TestCli:
    def test_the_manifest_prefix_is_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Printed rather than spelled in the buildspec, so the copy and the writer
        # cannot disagree about where the manifest is.
        main(["prefix", "manifest"])
        assert capsys.readouterr().out.strip() == "derived/manifest/"

    def test_the_partition_prefix_is_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["prefix", "partition", "--partition-version", "0"])
        assert capsys.readouterr().out.strip() == "derived/partition_version=v000/"

    def test_the_partition_prefix_needs_a_version(self) -> None:
        with pytest.raises(SystemExit, match="keyed by a version"):
            main(["prefix", "partition"])

    def test_the_version_is_required(self) -> None:
        with pytest.raises(SystemExit):
            _parser().parse_args(["assign", "--stage-dir", "."])

    def test_an_undefined_version_fails_at_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            _parser().parse_args(["assign", "--stage-dir", ".", "--partition-version", "7"])

    def test_there_is_no_seed_flag(self) -> None:
        # The seed is a property of the version, so there is nothing to mistype.
        with pytest.raises(SystemExit):
            _parser().parse_args(
                ["assign", "--stage-dir", ".", "--partition-version", "0", "--seed", "1"]
            )

    def test_assign_writes_the_partition(self, tmp_path: Path) -> None:
        stage = a_staged_manifest(tmp_path)
        main(["assign", "--stage-dir", str(stage), "--partition-version", "0"])
        assert (stage / assignments_key(V0)).is_file()

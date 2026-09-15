"""Tests for `conventions.py`, the module every S3 key and identifier goes through.

Expected keys are written as literal strings and never re-derived from the module's
own constants. An expected value of `f"{MANIFEST_PREFIX}part-{0:05d}.parquet"` asks
the code under test for the answer and then agrees with it.

Golden strings are the point here. These paths freeze the moment real data exists,
so a test that breaks whenever a key changes is exactly the tripwire wanted: it turns
an accidental edit into a loud diff.
"""

import re
from datetime import UTC, date, datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    ATHENA_RESULTS_PREFIX,
    LABELED_COHORTS,
    MANIFEST_PREFIX,
    PURCHASES_PREFIX,
    RAW_PROVENANCE_PREFIX,
    AssignmentRow,
    Buckets,
    Cohort,
    Cycle,
    GateResult,
    ImageId,
    ManifestRow,
    ModelArtifact,
    ModelManifest,
    ModelVersion,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Scene,
    Seed,
    Split,
    Table,
    TimeOfDay,
    Weather,
    _token,
    assignments_key,
    assignments_prefix,
    cohort_labels_key,
    cohort_labels_prefix,
    columns,
    cycle_prefix,
    eval_matches_key,
    eval_metrics_key,
    eval_prefix,
    gate_report_key,
    manifest_key,
    model_artifact_key,
    model_manifest_key,
    model_prefix,
    model_version_cycle,
    model_version_run_id,
    new_model_version,
    new_run_id,
    parse_image_id,
    parse_model_version,
    parse_run_id,
    partition_prefix,
    purchase_labels_key,
    purchase_labels_prefix,
    raw_image_key,
    raw_label_key,
    run_prefix,
    run_slug,
    run_started_at,
    table_name,
    telemetry_prefix,
    training_manifest_key,
    uri,
)

# --- Shared constants ---------------------------------------------------------
#
# Values, not resources, so module-level rather than fixtures.

# new_run_id(datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC), "v0-skeleton")
RUN = RunId("20260812t143355z-v0-skeleton")
# new_model_version(RUN, Cycle(3))
VERSION = ModelVersion("20260812t143355z-v0-skeleton-c003")
IMAGE = ImageId("0000f77c-6257be58")
SHA = "a" * 64
COMMIT = "b" * 40
ACCOUNT = "123456789012"

CREATED = datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC)
PV = PartitionVersion(1)
CYCLE = Cycle(3)


def a_registration(**overrides: Any) -> RunRegistration:
    """A valid registration with named fields replaced.

    Every rejection test then differs from a valid instance by exactly the field
    under test, which is what makes the failure legible.
    """
    values: dict[str, Any] = {
        "run_id": RUN,
        "created_at": CREATED,
        "git_commit": COMMIT,
        "partition_version": PartitionVersion(1),
        "recipe_version": RecipeVersion(1),
        "label_budget_per_cycle": 1000,
        "note": "first skeleton run",
    }
    return RunRegistration(**(values | overrides))


def a_manifest(**overrides: Any) -> ModelManifest:
    """A valid manifest, agreeing with `a_registration()`, with fields replaced."""
    values: dict[str, Any] = {
        "version": VERSION,
        "created_at": CREATED,
        "git_commit": COMMIT,
        "partition_version": PartitionVersion(1),
        "recipe_version": RecipeVersion(1),
        "cohorts_trained_on": frozenset({Cohort.BOOTSTRAP, Cohort.POOL}),
        "labels_spent": 2000,
        "deployed_seed": Seed(1),
        "artifact_sha256": {Seed(1): SHA},
        "gates": (GateResult(gate="data", passed=True, reason="80,000 rows"),),
    }
    return ModelManifest(**(values | overrides))


def a_manifest_row(**overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "image_id": IMAGE,
        "split": Split.TRAIN,
        "weather": Weather.CLEAR,
        "scene": Scene.CITY_STREET,
        "timeofday": TimeOfDay.DAYTIME,
        "n_boxes": 2,
        "box_areas": (120.0, 4800.5),
        "sha256": SHA,
        "label_source": "scalabel",
    }
    return ManifestRow(**(values | overrides))


# --- Part 1: the layout, frozen -----------------------------------------------

LAYOUT_CASES: list[tuple[str, Any, str]] = [
    (
        "raw_image_key",
        lambda: raw_image_key(IMAGE, Split.TRAIN),
        "raw/images/100k/train/0000f77c-6257be58.jpg",
    ),
    (
        "raw_label_key",
        lambda: raw_label_key(IMAGE, Split.VAL),
        "raw/labels/scalabel/val/0000f77c-6257be58.json",
    ),
    (
        "manifest_key",
        manifest_key,
        "derived/manifest/part-00000.parquet",
    ),
    (
        "manifest_key-part-3",
        lambda: manifest_key(3),
        "derived/manifest/part-00003.parquet",
    ),
    (
        "partition_prefix",
        lambda: partition_prefix(PV),
        "derived/partition_version=v001/",
    ),
    (
        "assignments_prefix",
        lambda: assignments_prefix(PV),
        "derived/partition_version=v001/assignments/",
    ),
    (
        "assignments_key",
        lambda: assignments_key(PV),
        "derived/partition_version=v001/assignments/part-00000.parquet",
    ),
    (
        "cohort_labels_prefix",
        lambda: cohort_labels_prefix(PV, Cohort.BOOTSTRAP),
        "derived/partition_version=v001/labels/cohort=bootstrap/",
    ),
    (
        "cohort_labels_key",
        lambda: cohort_labels_key(PV, Cohort.EVAL, 7),
        "derived/partition_version=v001/labels/cohort=eval/part-00007.parquet",
    ),
    (
        "purchase_labels_prefix",
        lambda: purchase_labels_prefix(RUN, CYCLE),
        "derived/purchases/run_id=20260812t143355z-v0-skeleton/cycle=003/",
    ),
    (
        "purchase_labels_key",
        lambda: purchase_labels_key(RUN, CYCLE, 7),
        "derived/purchases/run_id=20260812t143355z-v0-skeleton/cycle=003/part-00007.parquet",
    ),
    (
        "run_prefix",
        lambda: run_prefix(RUN),
        "run_id=20260812t143355z-v0-skeleton/",
    ),
    (
        "cycle_prefix",
        lambda: cycle_prefix(RUN, CYCLE),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/",
    ),
    (
        "model_prefix",
        lambda: model_prefix(VERSION),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "models/version=20260812t143355z-v0-skeleton-c003/",
    ),
    (
        "model_manifest_key",
        lambda: model_manifest_key(VERSION),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "models/version=20260812t143355z-v0-skeleton-c003/manifest.json",
    ),
    (
        "model_artifact_key",
        lambda: model_artifact_key(VERSION, Seed(1), ModelArtifact.ONNX),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "models/version=20260812t143355z-v0-skeleton-c003/seed=1/model.onnx",
    ),
    (
        "eval_prefix",
        lambda: eval_prefix(VERSION),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "eval/version=20260812t143355z-v0-skeleton-c003/",
    ),
    (
        "eval_metrics_key",
        lambda: eval_metrics_key(VERSION),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "eval/version=20260812t143355z-v0-skeleton-c003/metrics.json",
    ),
    (
        "eval_matches_key",
        lambda: eval_matches_key(VERSION, Seed(1)),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "eval/version=20260812t143355z-v0-skeleton-c003/seed=1/matches.npz",
    ),
    (
        "gate_report_key",
        lambda: gate_report_key(RUN, CYCLE),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/gates/report.json",
    ),
    (
        "training_manifest_key",
        lambda: training_manifest_key(RUN, CYCLE),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/training/images.manifest",
    ),
    (
        "telemetry_prefix",
        lambda: telemetry_prefix(RUN, date(2026, 8, 12)),
        "fleet/run_id=20260812t143355z-v0-skeleton/dt=2026-08-12/",
    ),
    (
        "table_name",
        lambda: table_name(Table.RUNS),
        "edge-ml-flywheel-runs",
    ),
    (
        "uri",
        lambda: uri("b", "k"),
        "s3://b/k",
    ),
    (
        "buckets-data",
        lambda: Buckets.for_account(ACCOUNT).data,
        "edge-ml-flywheel-data-123456789012",
    ),
    (
        "buckets-artifacts",
        lambda: Buckets.for_account(ACCOUNT).artifacts,
        "edge-ml-flywheel-artifacts-123456789012",
    ),
    (
        "buckets-telemetry",
        lambda: Buckets.for_account(ACCOUNT).telemetry,
        "edge-ml-flywheel-telemetry-123456789012",
    ),
]


class TestLayoutFreeze:
    """The single most important test in the file, so it goes first."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [(build, expected) for _, build, expected in LAYOUT_CASES],
        ids=[name for name, _, _ in LAYOUT_CASES],
    )
    def test_key_is_frozen(self, build: Any, expected: str) -> None:
        assert build() == expected

    def test_manifest_row_columns(self) -> None:
        assert columns(ManifestRow) == (
            "image_id",
            "split",
            "weather",
            "scene",
            "timeofday",
            "n_boxes",
            "box_areas",
            "sha256",
            "label_source",
        )

    def test_assignment_row_columns(self) -> None:
        assert columns(AssignmentRow) == ("image_id", "cohort")


# --- Part 2: behaviour, in the module's order ---------------------------------

# --- Identifiers ---


class TestImageIds:
    def test_accepts_a_bdd100k_id(self) -> None:
        assert parse_image_id("0000f77c-6257be58") == IMAGE

    @pytest.mark.parametrize(
        "value",
        [
            "0000F77C-6257be58",
            "0000f77c6257be58",
            "0000f77c-6257be5",
            "0000f77c-6257be58-x",
            "zzzzzzzz-6257be58",
            "",
            " 0000f77c-6257be58",
        ],
        ids=[
            "uppercase-hex",
            "no-hyphen",
            "short-group",
            "trailing-junk",
            "non-hex",
            "empty",
            "leading-space",
        ],
    )
    def test_rejects(self, value: str) -> None:
        with pytest.raises(ValueError, match="not a BDD100K image ID"):
            parse_image_id(value)


# --- Runs ---


class TestRunIdMinting:
    def test_mints_from_an_aware_datetime(self) -> None:
        assert new_run_id(CREATED, "v0-skeleton") == "20260812t143355z-v0-skeleton"

    def test_converts_a_non_utc_datetime_rather_than_truncating(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        started = datetime(2026, 8, 12, 20, 3, 55, tzinfo=ist)
        assert new_run_id(started, "v0") == "20260812t143355z-v0"

    def test_drops_microseconds_rather_than_rounding(self) -> None:
        started = datetime(2026, 8, 12, 14, 33, 55, 999999, tzinfo=UTC)
        assert new_run_id(started, "v0") == "20260812t143355z-v0"

    def test_rejects_a_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            new_run_id(datetime(2026, 8, 12, 14, 33, 55), "v0")

    def test_accepts_a_slug_at_the_length_limit(self) -> None:
        slug = "a" * 32
        assert new_run_id(CREATED, slug) == f"20260812t143355z-{slug}"

    def test_rejects_a_slug_over_the_length_limit(self) -> None:
        with pytest.raises(ValueError, match="over 32 characters"):
            new_run_id(CREATED, "a" * 33)

    @pytest.mark.parametrize(
        "slug",
        ["", "Foo", "foo_bar", "-foo", "foo-", "foo--bar", "foo bar", "föo"],
        ids=[
            "empty",
            "uppercase",
            "underscore",
            "leading-hyphen",
            "trailing-hyphen",
            "double-hyphen",
            "space",
            "non-ascii",
        ],
    )
    def test_rejects_a_malformed_slug(self, slug: str) -> None:
        with pytest.raises(ValueError, match="lowercase alphanumeric words"):
            new_run_id(CREATED, slug)

    @pytest.mark.parametrize(
        "slug",
        ["foo-c003", "c003"],
        ids=["hyphenated-cycle-word", "bare-cycle-word"],
    )
    def test_rejects_a_cycle_shaped_slug(self, slug: str) -> None:
        with pytest.raises(ValueError, match="cycle-shaped word"):
            new_run_id(CREATED, slug)

    @pytest.mark.parametrize(
        "slug",
        ["c03", "c0031", "ac003", "batch7", "v0-skeleton"],
        ids=["two-digits", "four-digits", "no-boundary", "word-then-digits", "multi-word"],
    )
    def test_still_accepts_a_slug_that_only_looks_cycle_shaped(self, slug: str) -> None:
        assert new_run_id(CREATED, slug) == f"20260812t143355z-{slug}"


class TestRunIdParsing:
    def test_accepts_a_minted_id(self) -> None:
        assert parse_run_id("20260812t143355z-v0-skeleton") == RUN

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("20260812t143355z", "not a run ID"),
            ("20260812T143355Z-v0", "not a run ID"),
            ("2026081t143355z-v0", "not a run ID"),
            ("20260812t143355z-foo-c003", "cycle-shaped word"),
        ],
        ids=["no-slug", "uppercase", "seven-digit-date", "cycle-shaped-slug"],
    )
    def test_rejects_a_malformed_id(self, value: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "20260230t120000z-v0",
            "20261301t120000z-v0",
            "20260812t250000z-v0",
            "00000000t000000z-v0",
        ],
        ids=["february-30", "month-13", "hour-25", "month-0"],
    )
    def test_rejects_a_well_shaped_but_impossible_timestamp(self, value: str) -> None:
        with pytest.raises(ValueError, match="real UTC timestamp"):
            parse_run_id(value)

    def test_run_started_at_returns_the_minted_second_in_utc(self) -> None:
        started = run_started_at(RUN)
        assert started == CREATED
        assert started.tzinfo is not None
        assert started.utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        ("run_id", "expected"),
        [
            (RunId("20260812t143355z-v0"), "v0"),
            (RUN, "v0-skeleton"),
        ],
        ids=["single-word", "multi-word"],
    )
    def test_run_slug(self, run_id: RunId, expected: str) -> None:
        assert run_slug(run_id) == expected


class TestRunRegistration:
    def test_constructs_from_valid_fields(self) -> None:
        assert a_registration().run_id == RUN

    def test_rejects_a_bad_run_id(self) -> None:
        with pytest.raises(ValueError, match="not a run ID"):
            a_registration(run_id=RunId("nonsense"))

    def test_rejects_a_naive_created_at(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            a_registration(created_at=datetime(2026, 8, 12, 14, 33, 55))

    @pytest.mark.parametrize(
        "commit",
        ["b" * 39, "b" * 41, "B" * 40, "z" * 40],
        ids=["39-chars", "41-chars", "uppercase", "non-hex"],
    )
    def test_rejects_a_bad_git_commit(self, commit: str) -> None:
        with pytest.raises(ValueError, match="40-character git commit"):
            a_registration(git_commit=commit)

    @pytest.mark.parametrize("budget", [0, -1], ids=["zero", "negative"])
    def test_rejects_a_non_positive_label_budget(self, budget: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            a_registration(label_budget_per_cycle=budget)

    def test_the_label_budget_does_not_supersede(self) -> None:
        """A budget change is a new run, but not a re-baseline of the champion.

        The versions fix the data universe and the metric, and neither of those
        moves when a run buys 2,000 labels a cycle instead of 1,000. Adding this
        to `supersedes` would discard the shared baseline that makes two runs
        comparable.
        """
        assert a_registration(label_budget_per_cycle=2000).supersedes(a_registration()) is False

    def test_identical_versions_do_not_supersede(self) -> None:
        assert a_registration().supersedes(a_registration()) is False

    def test_a_registration_does_not_supersede_itself(self) -> None:
        registration = a_registration()
        assert registration.supersedes(registration) is False

    @pytest.mark.parametrize(
        "field",
        ["partition_version", "recipe_version"],
    )
    def test_a_version_differing_alone_supersedes(self, field: str) -> None:
        assert a_registration(**{field: 2}).supersedes(a_registration()) is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("run_id", RunId("20260812t143355z-v1")),
            ("git_commit", "c" * 40),
            ("created_at", datetime(2026, 9, 1, tzinfo=UTC)),
            ("note", "a different reason"),
        ],
        ids=["run_id", "git_commit", "created_at", "note"],
    )
    def test_anything_outside_the_versions_does_not_supersede(self, field: str, value: Any) -> None:
        assert a_registration(**{field: value}).supersedes(a_registration()) is False


# --- Model versions ---


class TestModelVersions:
    @pytest.mark.parametrize("cycle", [0, 3, 999])
    def test_round_trips_run_and_cycle(self, cycle: int) -> None:
        version = new_model_version(RUN, Cycle(cycle))
        assert model_version_run_id(version) == RUN
        assert model_version_cycle(version) == cycle

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("20260812t143355z-v0-skeleton-c03", "not a model version"),
            ("20260812t143355z-v0-skeleton-c0034", "not a model version"),
            ("20260812t143355z-v0-skeleton-cabc", "not a model version"),
            ("20260812t143355z-v0-skeleton", "not a model version"),
            ("20260230t120000z-v0-c003", "real UTC timestamp"),
        ],
        ids=["two-digit-cycle", "four-digit-cycle", "non-numeric", "no-cycle", "impossible-date"],
    )
    def test_rejects(self, value: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_model_version(value)


# --- Image tags ---


class TestImageTags:
    """Golden vocabularies, for the reason the layout strings are golden.

    Every eval slice and the selection condition cap is a predicate over these
    values. A predicate against a tidied-up `dawn-dusk` or a singular
    `gas station` raises nothing and matches nothing: the slice empties, the cap
    never binds, and both read downstream as a clean pass. So the spellings are
    frozen here against the counts they were measured from, and an edit to one
    is a loud diff rather than a silent hole in the gate.
    """

    def test_weather_members(self) -> None:
        assert {w.value for w in Weather} == {
            "clear",
            "overcast",
            "partly cloudy",
            "rainy",
            "snowy",
            "foggy",
            "undefined",
        }

    def test_scene_members(self) -> None:
        assert {s.value for s in Scene} == {
            "city street",
            "highway",
            "residential",
            "parking lot",
            "tunnel",
            "gas stations",
            "undefined",
        }

    def test_timeofday_members(self) -> None:
        assert {t.value for t in TimeOfDay} == {"daytime", "night", "dawn/dusk", "undefined"}

    def test_undefined_is_a_member_of_all_three(self) -> None:
        # The archive writes it explicitly -- 9,291 images for weather alone --
        # so it is an observation the source records, not a null.
        assert (Weather.UNDEFINED, Scene.UNDEFINED, TimeOfDay.UNDEFINED) == (
            "undefined",
            "undefined",
            "undefined",
        )

    @pytest.mark.parametrize(
        "tag",
        [TimeOfDay.DAWN_DUSK, Scene.GAS_STATIONS, Weather.PARTLY_CLOUDY],
        ids=["a-slash", "a-plural", "a-space"],
    )
    def test_a_tag_is_never_an_s3_key_component(self, tag: StrEnum) -> None:
        # Unlike `Split` and `Cohort`, these values hold slashes and spaces: one
        # would silently add a path level, the other needs escaping downstream.
        # So a tag partitions a query and never a prefix, and the key builders
        # refuse it rather than emitting a key that addresses nothing.
        with pytest.raises(ValueError, match="not usable as an S3 key component"):
            _token("tag", tag)


# --- Cohorts and splits ---


class TestCohorts:
    def test_only_the_partition_time_cohorts_are_labeled(self) -> None:
        # A cohort added to the enum without a decision about its labels lands in
        # neither set, and this is where that shows up.
        assert set(Cohort) - LABELED_COHORTS == {Cohort.POOL, Cohort.RESERVE}
        assert len(Cohort) == 4
        assert {Cohort.BOOTSTRAP, Cohort.EVAL} == LABELED_COHORTS

    def test_eval_labels_do_not_share_a_prefix_with_the_training_cohorts(self) -> None:
        # Distinct prefixes are what lets the eval labels be denied by policy.
        assert cohort_labels_prefix(PV, Cohort.EVAL) != cohort_labels_prefix(PV, Cohort.BOOTSTRAP)


class TestSplit:
    def test_members(self) -> None:
        assert {s.value for s in Split} == {"train", "val"}

    def test_there_is_no_test_split(self) -> None:
        # The archive ships withheld ground truth for the 20,000 test images, so
        # re-adding this member should be a conscious act, not an autocomplete.
        assert not hasattr(Split, "TEST")


# --- Buckets and tables ---

_S3_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


class TestBuckets:
    @pytest.mark.parametrize(
        "account_id",
        ["1" * 11, "1" * 13, "12345678901a", "", "123456789012 "],
        ids=["11-digits", "13-digits", "letter", "empty", "trailing-space"],
    )
    def test_rejects_a_bad_account_id(self, account_id: str) -> None:
        with pytest.raises(ValueError, match="12-digit AWS account ID"):
            Buckets.for_account(account_id)

    @pytest.mark.parametrize("purpose", ["data", "artifacts", "telemetry"])
    def test_every_bucket_name_is_a_legal_s3_name(self, purpose: str) -> None:
        name = getattr(Buckets.for_account(ACCOUNT), purpose)
        assert 3 <= len(name) <= 63
        assert _S3_BUCKET_NAME.match(name)
        assert "_" not in name
        assert ".." not in name


class TestTables:
    @pytest.mark.parametrize("table", list(Table), ids=[t.value for t in Table])
    def test_every_table_name_is_legal(self, table: Table) -> None:
        assert re.match(r"^[A-Za-z0-9_.-]{3,255}$", table_name(table))


# --- Data bucket ---


class TestManifestRow:
    def test_a_valid_row_constructs(self) -> None:
        assert a_manifest_row().n_boxes == 2

    def test_zero_boxes_is_valid(self) -> None:
        # BDD100K has images with nothing to detect. Pinned so nobody later converts
        # this into a rejection.
        row = a_manifest_row(n_boxes=0, box_areas=())
        assert row.n_boxes == 0

    def test_rejects_a_box_count_disagreeing_with_the_areas(self) -> None:
        with pytest.raises(ValueError, match="disagrees with"):
            a_manifest_row(n_boxes=3)

    @pytest.mark.parametrize("area", [0.0, -1.0], ids=["zero", "negative"])
    def test_rejects_a_non_positive_area(self, area: float) -> None:
        with pytest.raises(ValueError, match="box area is not positive"):
            a_manifest_row(n_boxes=1, box_areas=(area,))

    @pytest.mark.parametrize(
        "digest",
        ["A" * 64, "a" * 63, "a" * 65, "z" * 64],
        ids=["uppercase", "63-chars", "65-chars", "non-hex"],
    )
    def test_rejects_a_bad_sha256(self, digest: str) -> None:
        with pytest.raises(ValueError, match="lowercase hex sha256"):
            a_manifest_row(sha256=digest)

    def test_rejects_a_malformed_image_id(self) -> None:
        with pytest.raises(ValueError, match="not a BDD100K image ID"):
            a_manifest_row(image_id=ImageId("nope"))


class TestAssignmentRow:
    def test_constructs(self) -> None:
        row = AssignmentRow(image_id=IMAGE, cohort=Cohort.BOOTSTRAP)
        assert row.cohort is Cohort.BOOTSTRAP

    def test_rejects_a_malformed_image_id(self) -> None:
        with pytest.raises(ValueError, match="not a BDD100K image ID"):
            AssignmentRow(image_id=ImageId("nope"), cohort=Cohort.BOOTSTRAP)


class TestDerivedKeys:
    @pytest.mark.parametrize("cohort", [Cohort.POOL, Cohort.RESERVE])
    def test_an_unlabeled_cohort_has_no_label_prefix(self, cohort: Cohort) -> None:
        with pytest.raises(ValueError, match="no labels of its own"):
            cohort_labels_prefix(PV, cohort)

    @pytest.mark.parametrize(
        "cohort",
        sorted(LABELED_COHORTS),
        ids=[c.value for c in sorted(LABELED_COHORTS)],
    )
    def test_every_labeled_cohort_has_a_prefix(self, cohort: Cohort) -> None:
        assert cohort_labels_prefix(PV, cohort).endswith(f"cohort={cohort.value}/")


# --- Artifacts bucket ---


class TestModelManifest:
    def test_constructs(self) -> None:
        assert a_manifest().version == VERSION

    def test_run_and_cycle_come_from_the_version(self) -> None:
        manifest = a_manifest()
        assert manifest.run_id == RUN
        assert manifest.cycle == 3

    def test_rejects_a_naive_created_at(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            a_manifest(created_at=datetime(2026, 8, 12, 14, 33, 55))

    def test_rejects_a_bad_git_commit(self) -> None:
        with pytest.raises(ValueError, match="40-character git commit"):
            a_manifest(git_commit="b" * 39)

    def test_rejects_a_bad_version(self) -> None:
        with pytest.raises(ValueError, match="not a model version"):
            a_manifest(version=ModelVersion("20260812t143355z-v0-skeleton"))

    def test_rejects_a_deployed_seed_with_no_digest(self) -> None:
        with pytest.raises(ValueError, match="no artifact digest"):
            a_manifest(deployed_seed=Seed(4))

    def test_rejects_a_non_sha256_digest(self) -> None:
        with pytest.raises(ValueError, match="lowercase hex sha256"):
            a_manifest(artifact_sha256={Seed(1): SHA, Seed(2): "nope"})

    def test_rejects_an_empty_cohort_set(self) -> None:
        with pytest.raises(ValueError, match="trained on no cohort"):
            a_manifest(cohorts_trained_on=frozenset())

    def test_rejects_training_on_the_eval_cohort(self) -> None:
        with pytest.raises(ValueError, match="trained on the eval cohort"):
            a_manifest(cohorts_trained_on=frozenset({Cohort.BOOTSTRAP, Cohort.EVAL}))

    def test_no_disagreements_with_its_own_run(self) -> None:
        assert a_manifest().disagreements(a_registration()) == ()

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("run_id", RunId("20260812t143355z-v1"), "run_id"),
            ("partition_version", PartitionVersion(2), "partition_version"),
            ("recipe_version", RecipeVersion(2), "recipe_version"),
        ],
        ids=["run_id", "partition_version", "recipe_version"],
    )
    def test_one_field_differing_names_exactly_that_field(
        self, field: str, value: Any, expected: str
    ) -> None:
        assert a_manifest().disagreements(a_registration(**{field: value})) == (expected,)

    def test_all_three_differing_are_reported_in_declared_order(self) -> None:
        run = a_registration(
            run_id=RunId("20260812t143355z-v1"),
            partition_version=PartitionVersion(2),
            recipe_version=RecipeVersion(2),
        )
        assert a_manifest().disagreements(run) == (
            "run_id",
            "partition_version",
            "recipe_version",
        )

    def test_no_gates_is_not_vacuously_passing(self) -> None:
        assert a_manifest(gates=()).gates_passed is False

    def test_all_gates_passing(self) -> None:
        gates = (
            GateResult(gate="data", passed=True, reason="80,000 rows"),
            GateResult(gate="uplift", passed=True, reason="+2.1 mAP"),
        )
        assert a_manifest(gates=gates).gates_passed is True

    def test_one_failure_among_several(self) -> None:
        gates = (
            GateResult(gate="data", passed=True, reason="80,000 rows"),
            GateResult(gate="uplift", passed=False, reason="-0.3 mAP"),
            GateResult(gate="latency", passed=True, reason="18 ms"),
        )
        assert a_manifest(gates=gates).gates_passed is False


class TestGateReport:
    def test_default_suffix(self) -> None:
        assert gate_report_key(RUN, CYCLE).endswith("gates/report.json")

    def test_an_alternative_suffix(self) -> None:
        assert gate_report_key(RUN, CYCLE, "html").endswith("gates/report.html")

    @pytest.mark.parametrize("suffix", ["a/b", "a=b", ""], ids=["slash", "equals", "empty"])
    def test_rejects_an_unsafe_suffix(self, suffix: str) -> None:
        with pytest.raises(ValueError, match="not usable as an S3 key component"):
            gate_report_key(RUN, CYCLE, suffix)


# --- Part 3: cross-cutting invariants -----------------------------------------

OUT_OF_RANGE_CASES: list[tuple[str, Any]] = [
    ("cycle_prefix--1", lambda: cycle_prefix(RUN, Cycle(-1))),
    ("cycle_prefix-1000", lambda: cycle_prefix(RUN, Cycle(1000))),
    ("new_model_version--1", lambda: new_model_version(RUN, Cycle(-1))),
    ("new_model_version-1000", lambda: new_model_version(RUN, Cycle(1000))),
    ("partition_prefix--1", lambda: partition_prefix(PartitionVersion(-1))),
    ("partition_prefix-1000", lambda: partition_prefix(PartitionVersion(1000))),
    ("assignments_prefix--1", lambda: assignments_prefix(PartitionVersion(-1))),
    ("assignments_prefix-1000", lambda: assignments_prefix(PartitionVersion(1000))),
    ("assignments_key-pv--1", lambda: assignments_key(PartitionVersion(-1))),
    ("assignments_key-pv-1000", lambda: assignments_key(PartitionVersion(1000))),
    (
        "cohort_labels_prefix--1",
        lambda: cohort_labels_prefix(PartitionVersion(-1), Cohort.BOOTSTRAP),
    ),
    (
        "cohort_labels_prefix-1000",
        lambda: cohort_labels_prefix(PartitionVersion(1000), Cohort.BOOTSTRAP),
    ),
    (
        "cohort_labels_key-pv--1",
        lambda: cohort_labels_key(PartitionVersion(-1), Cohort.BOOTSTRAP, 0),
    ),
    (
        "cohort_labels_key-pv-1000",
        lambda: cohort_labels_key(PartitionVersion(1000), Cohort.BOOTSTRAP, 0),
    ),
    ("cohort_labels_key-part--1", lambda: cohort_labels_key(PV, Cohort.BOOTSTRAP, -1)),
    (
        "cohort_labels_key-part-100000",
        lambda: cohort_labels_key(PV, Cohort.BOOTSTRAP, 100000),
    ),
    ("purchase_labels_key-part--1", lambda: purchase_labels_key(RUN, CYCLE, -1)),
    ("purchase_labels_key-part-100000", lambda: purchase_labels_key(RUN, CYCLE, 100000)),
    ("purchase_labels_key-cycle--1", lambda: purchase_labels_key(RUN, Cycle(-1), 0)),
    ("purchase_labels_key-cycle-1000", lambda: purchase_labels_key(RUN, Cycle(1000), 0)),
    ("training_manifest_key-cycle--1", lambda: training_manifest_key(RUN, Cycle(-1))),
    ("training_manifest_key-cycle-1000", lambda: training_manifest_key(RUN, Cycle(1000))),
    ("manifest_key--1", lambda: manifest_key(-1)),
    ("manifest_key-100000", lambda: manifest_key(100000)),
    ("assignments_key-part--1", lambda: assignments_key(PV, -1)),
    ("assignments_key-part-100000", lambda: assignments_key(PV, 100000)),
    ("model_artifact_key--1", lambda: model_artifact_key(VERSION, Seed(-1), ModelArtifact.ONNX)),
    ("model_artifact_key-10", lambda: model_artifact_key(VERSION, Seed(10), ModelArtifact.ONNX)),
    ("eval_matches_key--1", lambda: eval_matches_key(VERSION, Seed(-1))),
    ("eval_matches_key-10", lambda: eval_matches_key(VERSION, Seed(10))),
]

BOUNDARY_CASES: list[tuple[str, Any, str]] = [
    (
        "cycle-0",
        lambda: cycle_prefix(RUN, Cycle(0)),
        "run_id=20260812t143355z-v0-skeleton/cycle=000/",
    ),
    (
        "cycle-999",
        lambda: cycle_prefix(RUN, Cycle(999)),
        "run_id=20260812t143355z-v0-skeleton/cycle=999/",
    ),
    (
        "partition-0",
        lambda: partition_prefix(PartitionVersion(0)),
        "derived/partition_version=v000/",
    ),
    (
        "partition-999",
        lambda: partition_prefix(PartitionVersion(999)),
        "derived/partition_version=v999/",
    ),
    (
        "cohort-labels-0",
        lambda: cohort_labels_key(PV, Cohort.BOOTSTRAP, 0),
        "derived/partition_version=v001/labels/cohort=bootstrap/part-00000.parquet",
    ),
    (
        "cohort-labels-99999",
        lambda: cohort_labels_key(PV, Cohort.BOOTSTRAP, 99999),
        "derived/partition_version=v001/labels/cohort=bootstrap/part-99999.parquet",
    ),
    (
        "part-0",
        lambda: manifest_key(0),
        "derived/manifest/part-00000.parquet",
    ),
    (
        "part-99999",
        lambda: manifest_key(99999),
        "derived/manifest/part-99999.parquet",
    ),
    (
        "seed-0",
        lambda: model_artifact_key(VERSION, Seed(0), ModelArtifact.ONNX),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "models/version=20260812t143355z-v0-skeleton-c003/seed=0/model.onnx",
    ),
    (
        "seed-9",
        lambda: model_artifact_key(VERSION, Seed(9), ModelArtifact.ONNX),
        "run_id=20260812t143355z-v0-skeleton/cycle=003/"
        "models/version=20260812t143355z-v0-skeleton-c003/seed=9/model.onnx",
    ),
]


class TestKeyRanges:
    """Every number entering a key goes through one padding helper."""

    @pytest.mark.parametrize(
        "build",
        [build for _, build in OUT_OF_RANGE_CASES],
        ids=[name for name, _ in OUT_OF_RANGE_CASES],
    )
    def test_rejects_out_of_range(self, build: Any) -> None:
        with pytest.raises(ValueError, match="out of range"):
            build()

    @pytest.mark.parametrize(
        ("build", "expected"),
        [(build, expected) for _, build, expected in BOUNDARY_CASES],
        ids=[name for name, _, _ in BOUNDARY_CASES],
    )
    def test_accepts_at_the_boundary(self, build: Any, expected: str) -> None:
        assert build() == expected


class TestNesting:
    def test_cycle_sits_under_run(self) -> None:
        assert cycle_prefix(RUN, CYCLE).startswith(run_prefix(RUN))

    @pytest.mark.parametrize(
        "build",
        [
            lambda: model_prefix(VERSION),
            lambda: eval_prefix(VERSION),
            lambda: gate_report_key(RUN, CYCLE),
            lambda: training_manifest_key(RUN, CYCLE),
        ],
        ids=["model_prefix", "eval_prefix", "gate_report_key", "training_manifest_key"],
    )
    def test_sits_under_the_cycle(self, build: Any) -> None:
        assert build().startswith(cycle_prefix(RUN, CYCLE))

    @pytest.mark.parametrize(
        "build",
        [
            lambda: assignments_prefix(PV),
            lambda: cohort_labels_prefix(PV, Cohort.BOOTSTRAP),
        ],
        ids=["assignments_prefix", "cohort_labels_prefix"],
    )
    def test_sits_under_the_partition(self, build: Any) -> None:
        assert build().startswith(partition_prefix(PV))

    @pytest.mark.parametrize(
        "build",
        [
            lambda: model_manifest_key(VERSION),
            lambda: model_artifact_key(VERSION, Seed(1), ModelArtifact.ONNX),
        ],
        ids=["model_manifest_key", "model_artifact_key"],
    )
    def test_sits_under_the_model(self, build: Any) -> None:
        assert build().startswith(model_prefix(VERSION))

    @pytest.mark.parametrize(
        "build",
        [
            lambda: eval_metrics_key(VERSION),
            lambda: eval_matches_key(VERSION, Seed(1)),
        ],
        ids=["eval_metrics_key", "eval_matches_key"],
    )
    def test_sits_under_the_eval(self, build: Any) -> None:
        assert build().startswith(eval_prefix(VERSION))

    @pytest.mark.parametrize("version", [0, 1, 999])
    def test_the_manifest_is_not_under_any_partition(self, version: int) -> None:
        # Nothing in a manifest row changes when the partition does, so moving it
        # under there would be an easy tidy-up to get wrong.
        assert not MANIFEST_PREFIX.startswith(partition_prefix(PartitionVersion(version)))

    @pytest.mark.parametrize("version", [0, 1, 999])
    def test_purchases_are_not_under_any_partition(self, version: int) -> None:
        # Which images a cycle bought is a fact about the run, not the partition:
        # two runs over one partition buy differently, and the label-efficiency
        # A/B is the case where they must not collide.
        prefix = purchase_labels_prefix(RUN, CYCLE)
        assert not prefix.startswith(partition_prefix(PartitionVersion(version)))

    def test_a_purchase_is_separated_by_run(self) -> None:
        other = RunId("20260812t143355z-v1-other")
        assert purchase_labels_prefix(RUN, CYCLE) != purchase_labels_prefix(other, CYCLE)


ALL_BUILT_KEYS: dict[str, str] = {
    "raw_image_key": raw_image_key(IMAGE, Split.TRAIN),
    "raw_label_key": raw_label_key(IMAGE, Split.VAL),
    "manifest_key": manifest_key(),
    "partition_prefix": partition_prefix(PV),
    "assignments_prefix": assignments_prefix(PV),
    "assignments_key": assignments_key(PV),
    "cohort_labels_prefix": cohort_labels_prefix(PV, Cohort.BOOTSTRAP),
    "cohort_labels_key": cohort_labels_key(PV, Cohort.EVAL, 7),
    "purchase_labels_prefix": purchase_labels_prefix(RUN, CYCLE),
    "purchase_labels_key": purchase_labels_key(RUN, CYCLE, 7),
    "run_prefix": run_prefix(RUN),
    "cycle_prefix": cycle_prefix(RUN, CYCLE),
    "model_prefix": model_prefix(VERSION),
    "model_manifest_key": model_manifest_key(VERSION),
    "model_artifact_key": model_artifact_key(VERSION, Seed(1), ModelArtifact.ONNX),
    "eval_prefix": eval_prefix(VERSION),
    "eval_metrics_key": eval_metrics_key(VERSION),
    "eval_matches_key": eval_matches_key(VERSION, Seed(1)),
    "gate_report_key": gate_report_key(RUN, CYCLE),
    "training_manifest_key": training_manifest_key(RUN, CYCLE),
    "telemetry_prefix": telemetry_prefix(RUN, date(2026, 8, 12)),
    "MANIFEST_PREFIX": MANIFEST_PREFIX,
    "PURCHASES_PREFIX": PURCHASES_PREFIX,
    "RAW_PROVENANCE_PREFIX": RAW_PROVENANCE_PREFIX,
    "ATHENA_RESULTS_PREFIX": ATHENA_RESULTS_PREFIX,
}


class TestKeyShape:
    @pytest.mark.parametrize(
        ("name", "value"),
        list(ALL_BUILT_KEYS.items()),
        ids=list(ALL_BUILT_KEYS),
    )
    def test_prefixes_end_in_a_slash_and_keys_do_not(self, name: str, value: str) -> None:
        if name.lower().endswith("prefix"):
            assert value.endswith("/")
        else:
            assert not value.endswith("/")

    @pytest.mark.parametrize(
        "value",
        list(ALL_BUILT_KEYS.values()),
        ids=list(ALL_BUILT_KEYS),
    )
    def test_no_empty_component_and_no_space(self, value: str) -> None:
        assert "//" not in value
        assert " " not in value


class TestSortOrder:
    """The reason the padding exists, asserted rather than the padding itself."""

    def test_cycles_sort_numerically_as_strings(self) -> None:
        assert cycle_prefix(RUN, Cycle(2)) < cycle_prefix(RUN, Cycle(10))

    def test_purchases_sort_numerically_as_strings(self) -> None:
        # A cycle trains on every purchase up to it, so the cumulative labeled set
        # is only a key range if lexicographic order matches cycle order.
        assert purchase_labels_prefix(RUN, Cycle(2)) < purchase_labels_prefix(RUN, Cycle(10))

    def test_partition_versions_sort_numerically_as_strings(self) -> None:
        assert partition_prefix(PartitionVersion(2)) < partition_prefix(PartitionVersion(10))

    def test_label_parts_sort_numerically_as_strings(self) -> None:
        assert cohort_labels_key(PV, Cohort.BOOTSTRAP, 2) < cohort_labels_key(
            PV, Cohort.BOOTSTRAP, 10
        )

    def test_seeds_sort_numerically_as_strings(self) -> None:
        assert model_artifact_key(VERSION, Seed(2), ModelArtifact.ONNX) < model_artifact_key(
            VERSION, Seed(9), ModelArtifact.ONNX
        )

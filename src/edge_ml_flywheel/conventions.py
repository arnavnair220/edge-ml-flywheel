"""Storage and identifier conventions, fixed before the first table is written.

Every component builds its S3 keys through this module rather than formatting
its own strings. A key spelled in two places drifts in one of them, and the
failure is silent in the worst way: the writer succeeds, the reader finds
nothing, and the cycle reports having done less work than it did.

Three rules the layout follows, recorded here because they decide every
question about where something new belongs.

**Key a thing by exactly what its content is a function of.** Key by less and
two different things collide at one path; key by more and identical bytes are
stored once per surplus dimension. Cohort labels are a function of the raw data
and the partition version, so they carry ``partition_version`` and deliberately
not ``run_id`` -- two runs sharing a partition share them, which is safe
precisely because a partition change already forces a fresh run and a
re-baselined champion (design section 5). Model artifacts are a function of
run, cycle, version and seed, so all four appear. The rule is also what splits
the manifest in two: an image's attributes and checksum are a function of the
raw archive alone, its cohort is a function of the partition, and the two
therefore cannot share a path.

**Hive-style ``key=value/`` only where something prunes on it** -- a query
engine, an IAM policy, or a lifecycle rule. Elsewhere it is noise, and a
partition holding too little data is worse than none.

**Numbers in keys are zero-padded**, because S3 sorts lexicographically and an
unpadded ``cycle=10`` sorts before ``cycle=2``. A width is a promise, so the
range check that keeps it is part of the padding rather than a duty left to
whichever call site remembers.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NewType, Self

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# --- Identifiers -------------------------------------------------------------

ImageId = NewType("ImageId", str)
PartitionVersion = NewType("PartitionVersion", int)
Cycle = NewType("Cycle", int)
Seed = NewType("Seed", int)

# The partitioner's draw seed, kept distinct from `Seed` because the two are
# different numbers with different lifetimes. `Seed` is a training seed, one of
# the seeds a cycle trains, and a path component capped at one digit by
# `SEED_DIGITS`; a draw seed appears in no key, is fixed once per partition
# version, and is six digits wider than that cap allows.
PartitionSeed = NewType("PartitionSeed", int)

RunId = NewType("RunId", str)

# `<run_id>-c<cycle>`. Built out of the run-id machinery, so the format itself is
# defined under "Model versions" below rather than here.
ModelVersion = NewType("ModelVersion", str)

# A version stamp a paired comparison assumes is held constant (design section
# 5), alongside `PartitionVersion` below. An integer rather than a free-form
# string because the only operation ever performed on it is equality against the
# champion's. The class set is not among these: there is one, so it cannot
# differ between two models being compared.
RecipeVersion = NewType("RecipeVersion", int)

# Widths are properties of the key format, not of any one caller, so they live
# here and nowhere else. Nothing restates them.
#
# `CYCLE_DIGITS` is the one that is not free to change, because a model version
# ends in the padded cycle: widening it after the first version is minted leaves
# every existing version unparseable, and those strings are in S3 keys, in
# `fleet_config`, and in whatever a device last reported. Three digits caps a run
# at 1,000 cycles against a design that plans eight, so the headroom is not the
# question -- but change it before Phase 3 mints anything, or not at all.
CYCLE_DIGITS: Final = 3
PARTITION_VERSION_DIGITS: Final = 3
PART_DIGITS: Final = 5

# One digit, which is to say `seed=1` rather than `seed=01`. A cycle trains five
# seeds (design section 4.2), so nine is headroom and a hundred is a longer key
# bought for nothing. As permanent as the widths above -- it is a path component
# from the first artifact written -- so the cap is the deliberate part, not the
# absence of padding.
SEED_DIGITS: Final = 1

# BDD100K image IDs are two 8-character hex groups, as in `0000f77c-6257be58`.
# Applied at the ingest boundary rather than on every key build: a bad ID should
# fail the Phase 1 verification loudly, not surface as a mysterious 404 in week
# five. Key builders enforce only token safety.
BDD100K_IMAGE_ID: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}$")

# Anything outside this set either breaks the key structure ("/"), breaks Hive
# partition parsing ("="), or needs escaping somewhere downstream.
_SAFE_TOKEN: Final = re.compile(r"^[A-Za-z0-9._-]+$")

_ACCOUNT_ID: Final = re.compile(r"^[0-9]{12}$")


def _token(name: str, value: str) -> str:
    """Reject a value that cannot appear as a single S3 key component."""
    if not _SAFE_TOKEN.match(value):
        raise ValueError(f"{name} is not usable as an S3 key component: {value!r}")
    return value


def _padded(name: str, value: int, digits: int) -> str:
    """Zero-pad a number for a key, refusing one that does not fit the width.

    Padding alone would accept cycle 1,000 and emit four digits, which sorts
    before every three-digit cycle and reads back as a different number. A
    negative pads to ``-01``, a component that looks like a real path and
    addresses nothing. Both failures are silent, so the range check lives with
    the padding rather than at whichever call site remembers -- and it lives
    here once rather than once per width.
    """
    if not 0 <= value < 10**digits:
        raise ValueError(f"{name} is out of range for a {digits}-digit key component: {value}")
    return f"{value:0{digits}d}"


def parse_image_id(value: str) -> ImageId:
    """Validate a BDD100K image ID. Use at ingest, not in hot paths."""
    if not BDD100K_IMAGE_ID.match(value):
        raise ValueError(f"not a BDD100K image ID: {value!r}")
    return ImageId(value)


# --- Selection ----------------------------------------------------------------
#
# There is one selection rule: mean per-object uncertainty, in
# `selection.select.by_uncertainty`. It is not a field on a run and not a value
# anyone passes, because there is nothing to choose between -- every run ranks
# the pool the same way, and a run that did not would not be this project.


# --- Runs ---------------------------------------------------------------------
#
# `run_id` is the blast radius boundary for a whole experiment: the partition
# key on every stateful table and the top prefix of artifacts and telemetry, so
# run 2 physically cannot read run 1's ledger, champion or locks (design
# section 7). Everything about the format follows from that.
#
# `<utc timestamp>-<slug>`, as in `20260812t143355z-v0-skeleton`.
#
# **Second-precision UTC, fixed width, leading.** S3 and DynamoDB both sort
# lexicographically, so a fixed-width leading timestamp is the only thing that
# makes a bucket listing or a query range chronological. Second precision plus
# the slug is what keeps ids distinct without a random suffix, since runs are
# started by a human or a cron tick and never in a burst. It is not proof on its
# own, and a collision is silent in the worst way -- the second run adopts the
# first's spent-label ledger -- so the guard that actually holds is the
# registration write: a conditional put on `attribute_not_exists(run_id)`
# against `Table.RUNS`, which turns a collision into a loud failure at run
# start. That is the reason registration is mandatory rather than a nicety.
#
# **A slug, because the timestamp alone is unreadable.** In week five a bucket
# listing of bare timestamps tells you nothing, and this string is what gets
# pasted into a rollback command under pressure.
#
# **Lowercase, enforced.** S3 keys are case-sensitive and DynamoDB keys are
# compared byte for byte, so `Run` and `run` are two different runs everywhere
# except in a human's reading of them.
#
# **Deliberately not encoded: the versions.** A change to `recipe_version` or
# `partition_version` forces a new run (design section 5), but the id only has to
# be *new*, not to describe the change. Both live in the run registration and in
# every model manifest; putting them in the id too would give two sources of
# truth that drift.

RUN_SLUG_MAX_LEN: Final = 32

# Words of lowercase alphanumerics joined by single hyphens. No leading, double
# or trailing hyphen, so the id splits back apart unambiguously.
_RUN_SLUG: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# A slug may not end in a cycle-shaped word, because a model version is
# `<run_id>-c<cycle>`: the run `...z-foo-c003` and the third model of run
# `...z-foo` are then the same string, and `_locate` splits it into a run that
# was never minted. Only the hyphenated case is ambiguous to the parser -- a
# slug of exactly `c003` cannot be read as a version, since that leaves no slug
# -- but it is rejected too, because "a run is never named after a cycle" is a
# rule that survives being remembered at 2am and a carve-out is not. Derived
# from `CYCLE_DIGITS` so the two cannot drift apart.
_RUN_SLUG_CYCLE_WORD: Final = re.compile(rf"(?:^|-)c[0-9]{{{CYCLE_DIGITS}}}$")

# Named groups, and a pattern kept separate from its anchors, because the model
# version below embeds this verbatim. Positional groups would renumber inside the
# larger pattern and leave both accessors reading the wrong half of a match.
_RUN_ID_PATTERN: Final = (
    r"(?P<date>[0-9]{8})t(?P<time>[0-9]{6})z-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)"
)
_RUN_ID: Final = re.compile(f"^{_RUN_ID_PATTERN}$")

_RUN_TIMESTAMP_FORMAT: Final = "%Y%m%dt%H%M%Sz"


def _check_slug(slug: str) -> None:
    """Applied when minting and again when parsing.

    Both, because a rule enforced only at the minter is true of the ids we
    happened to make ourselves and of nothing arriving from outside -- and the
    ambiguity this rejects is one a hand-typed id walks straight into.
    """
    if len(slug) > RUN_SLUG_MAX_LEN:
        raise ValueError(f"run slug is over {RUN_SLUG_MAX_LEN} characters: {slug!r}")
    if not _RUN_SLUG.match(slug):
        raise ValueError(
            f"run slug must be lowercase alphanumeric words joined by single hyphens: {slug!r}"
        )
    if _RUN_SLUG_CYCLE_WORD.search(slug):
        raise ValueError(
            f"run slug must not end in a cycle-shaped word, which a model version reads as its "
            f"own cycle: {slug!r}"
        )


def new_run_id(started_at: datetime, slug: str) -> RunId:
    """Mint a run ID. Called **once**, at the top of a run, never downstream.

    `started_at` is required rather than read from the clock here, and that is
    the whole point of the signature: a Lambda that calls a zero-argument
    minter invents a fresh run on every invocation and quietly detaches itself
    from the run it belongs to. The id is an input that gets passed down, so
    the one place that reads a clock is the one place that starts a run.
    """
    if started_at.tzinfo is None:
        raise ValueError(f"started_at must be timezone-aware: {started_at!r}")
    _check_slug(slug)
    stamp = started_at.astimezone(UTC).strftime(_RUN_TIMESTAMP_FORMAT)
    return RunId(f"{stamp}-{slug}")


def parse_run_id(value: str) -> RunId:
    """Validate a run ID arriving from outside -- an event payload, a CLI flag.

    Worth doing at every boundary rather than only at ingest, unlike an image
    ID: a malformed run ID does not 404, it reads and writes a table partition
    nobody is watching.
    """
    match = _RUN_ID.match(value)
    if not match:
        raise ValueError(f"not a run ID: {value!r}")
    _check_slug(match["slug"])
    try:
        datetime.strptime(f"{match['date']}t{match['time']}z", _RUN_TIMESTAMP_FORMAT)
    except ValueError:
        raise ValueError(f"run ID does not carry a real UTC timestamp: {value!r}") from None
    return RunId(value)


def run_started_at(run_id: RunId) -> datetime:
    """Recover the mint time. Convenience for sorting and display only.

    Authoritative run metadata lives in the registration item, not in the id.
    """
    parsed = parse_run_id(run_id)
    match = _RUN_ID.match(parsed)
    assert match is not None  # parse_run_id would have raised
    stamp = f"{match['date']}t{match['time']}z"
    return datetime.strptime(stamp, _RUN_TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def run_slug(run_id: RunId) -> str:
    match = _RUN_ID.match(parse_run_id(run_id))
    assert match is not None  # parse_run_id would have raised
    return match["slug"]


_GIT_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class RunRegistration:
    """Written once, when a run is minted, and never updated.

    This is what lets `run_id` stay opaque. The id says *which* run and *when*;
    this says what it was configured as and why it exists, so a bucket listing
    six weeks later resolves to something without reverse-engineering it from
    artifacts. The versions are recorded at the run level, not only per
    model, because they are the *precondition* of the run's whole comparison --
    a model manifest disagreeing with its run's registration is a bug worth
    detecting rather than a fact worth storing twice.

    `label_budget_per_cycle` is the *rule*, and `Table.LABEL_BUDGET` holds what
    is left of it. Those are different facts and only one of them is durable: a
    budget item is a counter the oracle decrements, so it says a cycle has 400
    labels left and never that the cycle was allowed 1,000. Summing purchases
    afterwards recovers what was spent, not what was permitted, and the two
    differ in exactly the case worth auditing -- a cycle that could not spend its
    cap. Model improvement per label spent is the headline number, so the
    denominator's rule is recorded where the run's other preconditions are.

    Being on a write-once item makes the budget fixed for the run, which is the
    intended constraint rather than a side effect: a budget raised at cycle four
    makes the per-cycle curve before and after it two different measurements
    plotted on one axis. A different budget is a different run, which is cheap --
    it shares the partition and the frozen eval.

    `note` is free text and the only unstructured field: the reason this run was
    started, which is exactly the thing no schema anticipates and no artifact
    records.
    """

    run_id: RunId
    created_at: datetime
    git_commit: str
    partition_version: PartitionVersion
    recipe_version: RecipeVersion
    label_budget_per_cycle: int
    note: str

    def __post_init__(self) -> None:
        parse_run_id(self.run_id)
        if self.created_at.tzinfo is None:
            raise ValueError(f"created_at must be timezone-aware: {self.created_at!r}")
        if not _GIT_COMMIT.match(self.git_commit):
            raise ValueError(f"not a full 40-character git commit SHA: {self.git_commit!r}")
        # Zero is refused rather than treated as a dry run. A run that can buy
        # nothing trains the same model every cycle, and the loop reports eight
        # clean no-change cycles rather than a configuration error.
        if self.label_budget_per_cycle <= 0:
            raise ValueError(
                f"label budget per cycle must be positive: {self.label_budget_per_cycle}"
            )

    def supersedes(self, other: "RunRegistration") -> bool:
        """True when `other`'s cached comparisons cannot carry into this run.

        The design's re-baseline rule (section 5) stated once, here, rather than
        as an `if` in the promotion path that someone later extends by one field
        and forgets in the other place.

        The class set is not one of these, because there is only one: it cannot
        differ between two runs, so it cannot be what forces a re-baseline.
        """
        return (
            self.partition_version,
            self.recipe_version,
        ) != (
            other.partition_version,
            other.recipe_version,
        )


# --- Model versions -----------------------------------------------------------
#
# `<run_id>-c<cycle>`, as in `20260812t143355z-v0-skeleton-c003`.
#
# **The name is an address; the manifest is the facts.** The versions, the git
# commit, the artifact digests, the gate results -- every one of them is read by
# opening `ModelManifest`, and none of them is needed to find it. So none of them
# appear here. Encoding one would create a second place for the same fact to be
# stated, and this is the copy that cannot be corrected: the version is a path
# component and the artifacts bucket is write-once, so a name that disagrees with
# its manifest disagrees permanently. Same rule that keeps them out of `run_id`,
# applied to the same kind of field.
#
# **What the name must carry is a locator.** A device asks `fleet_config` for
# `desired_version` and gets back one string (design section 6). It does not know
# what run is current or what cycle produced the model, and that string is the
# whole input to finding `run_id=.../cycle=.../models/version=.../seed=1`.
# Encoding run and cycle makes that resolution pure string parsing. The
# alternative is a version-to-path lookup table -- a second store to keep
# consistent, bought for the sake of a shorter string.
#
# **One version per cycle**, which is what makes the cycle enough to identify it.
# A cycle trains one challenger, at one seed by design (design section 4.2), and the
# champion it is compared against is not retrained -- its cached seed runs are
# reused (design section 7), so the cycle produces exactly one new model. Work
# that falls outside that rhythm already has somewhere to live: the random arm of
# the label-efficiency A/B is its own run lineage, and the budget re-calibration
# is its own throwaway cycle (design section 8, section 2). The A/A test's
# optional symmetric variant trains two models, and they go in two consecutive
# cycles -- comparing a model against one from an adjacent cycle is the gate's
# ordinary shape, not a special case.
#
# The run and cycle appear again in the prefix built from them, so the builders
# below take the version alone and recover the rest. Passing all three would be
# three ways to state two facts, and eventually one of them is wrong.

_MODEL_VERSION: Final = re.compile(
    f"^(?P<run_id>{_RUN_ID_PATTERN})-c(?P<cycle>[0-9]{{{CYCLE_DIGITS}}})$"
)


def _locate(version: ModelVersion) -> tuple[RunId, Cycle]:
    """Split a version into the run and cycle its key prefix is built from.

    Shape only, deliberately: this is on the path of every key build, and the
    expensive check -- that the timestamp is a real date -- is a boundary
    concern that `parse_model_version` owns.
    """
    match = _MODEL_VERSION.match(version)
    if not match:
        raise ValueError(f"not a model version: {version!r}")
    return RunId(match["run_id"]), Cycle(int(match["cycle"]))


def new_model_version(run_id: RunId, cycle: Cycle) -> ModelVersion:
    """Name the model a cycle produces. Called once, where the model is trained.

    Both arguments are already in hand wherever this is called -- a training job
    knows its run and its cycle -- so there is nothing to look up and no clock to
    read, unlike `new_run_id`.
    """
    return ModelVersion(f"{parse_run_id(run_id)}-c{_padded('cycle', cycle, CYCLE_DIGITS)}")


def parse_model_version(value: str) -> ModelVersion:
    """Validate a version arriving from outside -- `desired_version`, a telemetry
    heartbeat, a rollback argument.

    Worth doing at every boundary, for `parse_run_id`'s reason: a malformed
    version does not 404 in any useful way, it names a prefix that was never
    written, and the device reports itself as running something that does not
    exist.
    """
    run_id, _ = _locate(ModelVersion(value))
    parse_run_id(run_id)  # rejects a well-shaped but impossible timestamp
    return ModelVersion(value)


def model_version_run_id(version: ModelVersion) -> RunId:
    return _locate(version)[0]


def model_version_cycle(version: ModelVersion) -> Cycle:
    return _locate(version)[1]


# SageMaker's ceiling on the names it calls entities, which a model package group
# is one of. Stated here because the group is named below and the check that it
# fits belongs beside the name rather than in the module that makes the call.
MAX_ENTITY_NAME: Final = 63


def model_package_group(run_id: RunId) -> str:
    """The Model Registry group a run's versions are registered into.

    One group per run, named the run and nothing else. No project prefix, for two
    reasons. The account holds this project alone, so a prefix would namespace
    nothing; and a run ID is already 49 characters at the longest slug
    `RUN_SLUG_MAX_LEN` permits, which leaves 14 for a prefix that would then make
    a legal run unregisterable at its first cycle -- the worst place to discover
    that a name is too long, since the run is minted and its cycle is spent.
    Identifying the group as this project's is what the tags on it are for.

    A group per run rather than per project because the champion a version is
    compared against is a fact about its run (design section 5): a partition or
    recipe change forces a new run and a re-baselined champion, so versions from
    two runs are not two entries on one ladder.
    """
    name = str(parse_run_id(run_id))
    if len(name) > MAX_ENTITY_NAME:
        raise ValueError(
            f"model package group name is {len(name)} characters, over SageMaker's "
            f"{MAX_ENTITY_NAME}: {name}"
        )
    return name


# --- Image tags ---------------------------------------------------------------
#
# The three BDD100K attributes the eval slices and the batch's condition mix
# are written against. Each vocabulary is exhaustive because it was counted over
# all 80,000 images rather than recalled, which is also what makes an unexpected
# value at ingest a statement that the archive changed rather than a gap here.
#
# **Values are the archive's own spellings, carried verbatim.** `dawn/dusk` holds
# a slash, four members hold a space, and `gas stations` is plural. Which is the
# one way these differ from `Split` and `Cohort`: a tag is never an S3 key
# component -- a slash would silently add a path level and a space needs escaping
# somewhere downstream -- so a tag partitions a query and never a prefix.
# `_token` refuses every one of them, so the mistake fails at the key builder
# rather than producing a key that addresses nothing.
#
# **`undefined` is a member of all three.** The archive states it explicitly --
# 9,291 images for `weather` alone -- so it is an observation the source records,
# not a value it withholds. A vocabulary omitting it would reject data that is
# genuinely fine, and a nullable column would put two spellings of the same fact
# in one place.


class Weather(StrEnum):
    """Pool shares run from `clear` at 53% down to `foggy` at 143 images.

    That spread is why the regression gate is written against overall eval and
    not against these values: `foggy` is 143 images in the whole archive and a
    few in any eval drawn from it, so a per-value verdict would be noise wearing
    a threshold. The vocabulary is here because ingest validates against it and
    the selection mix record tallies over it, not because anything
    gates on it.
    """

    CLEAR = "clear"
    OVERCAST = "overcast"
    PARTLY_CLOUDY = "partly cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    FOGGY = "foggy"
    UNDEFINED = "undefined"


class Scene(StrEnum):
    """`GAS_STATIONS` is plural, which is the archive's spelling and not a typo."""

    CITY_STREET = "city street"
    HIGHWAY = "highway"
    RESIDENTIAL = "residential"
    PARKING_LOT = "parking lot"
    TUNNEL = "tunnel"
    GAS_STATIONS = "gas stations"
    UNDEFINED = "undefined"


class TimeOfDay(StrEnum):
    """`DAWN_DUSK` is one value, `dawn/dusk`, not two joined by a separator."""

    DAYTIME = "daytime"
    NIGHT = "night"
    DAWN_DUSK = "dawn/dusk"
    UNDEFINED = "undefined"


# --- Raw image geometry -------------------------------------------------------

# BDD100K's stated resolution, and the frame every box coordinate in this
# project is expressed in. Here rather than in `ingest` because three components
# depend on it: ingest rejects an image of another size, `ManifestRow.box_areas`
# is in these pixels, and eval defines its small-object slice at this resolution
# rather than at the model's 416 px (design section 4.4).
NATIVE_IMAGE_SIZE: Final = (1280, 720)


# --- Detections per image -----------------------------------------------------

# The cap the scoring pass writes at and the match cache is built at, and the
# largest one anything can ask for later. `COCOeval` applies it inside the
# per-image step, after sorting by score, so a block holds at most this many
# detections and a smaller cap is a truncation of each block rather than a
# re-score (see `evaluation.match.detection_rows`).
#
# Here rather than beside the matcher because the two planes either side of it
# must agree: `scoring.job` writes at this cap and `evaluation.match` caches at
# it, and a number the writer and the reader spell separately is the silent
# drift this module exists to prevent. It is also what keeps the control
# function's imports clear of `pycocotools` -- the Lambda is a zip of pure
# Python that builds job requests, and importing the matcher for one integer
# put a C extension in its import graph that its package cannot carry.
MAX_DETS: Final = 100


# --- Object classes -----------------------------------------------------------
#
# Which object categories a model predicts, and so which ones the metric covers.
# One set, fixed for the project: changing it would redefine what the headline
# number means and void every comparison already made, which is a new project
# rather than a new run.
#
# The names are the archive's, measured at ingest and recorded in
# `raw/_provenance/integrity.json`. A class set is checked against them because
# the legacy Scalabel archive spells three categories `person`, `motor` and
# `bike` where the `det_20` release says `pedestrian`, `motorcycle` and
# `bicycle`: a `det_20` spelling matches no box in the archive, so that class
# scores 0.0 AP every cycle and nothing raises.
#
# A category ID is a position in a tuple, so declaration order is permanent --
# the IDs are stored in every cached match array, which holds `2` and not
# `truck`. Order is measured frequency descending, the order the integrity
# report records.

# The ten categories carrying a `box2d` in the legacy archive. Not a class set:
# this is what the archive contains, and a class set is what a model predicts.
# `train` is here and in neither class set (design section 3), at 151 boxes
# archive-wide.
BOX_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "car",
        "traffic sign",
        "traffic light",
        "person",
        "truck",
        "bus",
        "bike",
        "rider",
        "motor",
        "train",
    }
)


@dataclass(frozen=True, slots=True)
class ClassSet:
    """The categories a model predicts. There is one, `CLASS_SET`.

    A type rather than a bare tuple because the ID arithmetic and the archive
    check below are what a class set *is*, and both are easy to get subtly wrong
    at a call site.

    IDs are 1-based, as COCO's own categories are.
    """

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a class set with no classes is not a class set")
        duplicates = sorted({name for name in self.names if self.names.count(name) > 1})
        if duplicates:
            raise ValueError(f"class set repeats a class: {duplicates}")
        unknown = sorted(set(self.names) - BOX_CATEGORIES)
        if unknown:
            listed = ", ".join(sorted(BOX_CATEGORIES))
            raise ValueError(
                f"class set names categories the archive has no boxes for: {unknown}. "
                f"Measured at ingest: {listed}"
            )

    @property
    def category_ids(self) -> Mapping[str, int]:
        """Name to COCO category ID, for a caller converting many boxes.

        `category_id` is the one-off form.
        """
        return {name: index for index, name in enumerate(self.names, start=1)}

    def category_id(self, name: str) -> int:
        try:
            return self.names.index(name) + 1
        except ValueError:
            raise ValueError(f"{name!r} is not in this class set: {list(self.names)}") from None

    def category_name(self, category_id: int) -> str:
        """The inverse. A cached match array holds IDs, and a report needs names."""
        if not 1 <= category_id <= len(self.names):
            raise ValueError(
                f"category ID {category_id} is outside a class set of {len(self.names)}"
            )
        return self.names[category_id - 1]


# The nine categories every model in this project predicts (design section 3).
# The four COCO-native classes alone are a set a COCO-pretrained detector already
# starts strong on, which leaves a cycle's labels little to move; these nine are
# what the loop is measured over.
#
# One set rather than a table of versions, because the project trains one kind of
# model: there is nothing to select between, so there is no version to carry and
# no lookup that can fail. The declaration order is still permanent -- a category
# ID is a position in this tuple and is what every cached match array stores --
# so a class is appended, never inserted or reordered.
CLASS_SET: Final = ClassSet(
    names=(
        "car",
        "traffic sign",
        "traffic light",
        "person",
        "truck",
        "bus",
        "bike",
        "rider",
        "motor",
    )
)


# --- Cohorts and splits ------------------------------------------------------


class Split(StrEnum):
    """BDD100K's own split, which survives into `raw/` paths and the manifest.

    There is deliberately no `TEST` member. The archive ships ground-truth boxes
    for the 20,000 test images the benchmark withholds, and those are dropped at
    ingest. Making the value unrepresentable is the cheap half of the leakage
    guard; the Phase 1 assertion that the pool is exactly 80,000 is the other.
    """

    TRAIN = "train"
    VAL = "val"


class Cohort(StrEnum):
    """The partitioner's exactly-once assignment for each of the 80,000 images.

    Disjoint and complete is exactly the property a partition key needs, so the
    Phase 1 partition assertion and the validity of the `cohort=` prefix are the
    same statement.

    `BOOTSTRAP` and `EVAL` are labeled when the partitioner runs; `POOL` is the
    withheld remainder that the fleet drives through and selection ranks. An
    image leaves the pool by being bought, and which images a cycle bought is a
    fact about a run rather than about the partition, so a purchase is recorded
    under its run -- see `purchase_labels_prefix` -- and never by reassigning a
    cohort. Cohort assignment is immutable for the life of a partition version.

    `RESERVE` is deliberately inert. Growing `EVAL` mid-run moves the ruler every
    earlier cycle was measured against, so the spare `val` images are held out of
    `POOL` where selection would otherwise spend them, and a larger eval becomes
    a new partition rather than a dead end.
    """

    BOOTSTRAP = "bootstrap"
    POOL = "pool"
    EVAL = "eval"
    RESERVE = "reserve"


# The two cohorts the partitioner labels, and so the two with boxes of their own
# under the partition prefix. `pool` labels are withheld and sold a batch at a
# time, so what a cycle buys is filed under its run instead; `reserve` is inert
# and has labels nowhere.
LABELED_COHORTS: Final = frozenset({Cohort.BOOTSTRAP, Cohort.EVAL})

# The two cohorts a cycle puts in front of a model, which is a different question
# from which cohorts have boxes and has a different answer. `eval` is scored
# because it is the ruler, and `pool` because its scores are the ranking the
# budget is spent down. `bootstrap` is absent because a model's confidence over
# its own training set measures nothing anyone acts on, and `reserve` because it
# is inert.
#
# Both are scored in one pass over each, and never again for that model: the
# cohorts are fixed for the cycle, so a second pass would re-derive an answer
# already on disk (design section 7).
SCORED_COHORTS: Final = frozenset({Cohort.EVAL, Cohort.POOL})

# Which split each cohort draws from: everything trainable out of `train`,
# everything held out of `val`. The leakage rule as data rather than as a
# predicate somewhere in the partitioner, so a leakage question is answered by
# naming a cohort's source instead of re-deriving an image-ID set.
#
# Declaration order is draw order within a split, which is what a later partition
# version inherits. `eval` before `reserve` means a version that grows `eval` at
# the same seed holds every earlier `eval` image, and `bootstrap` before `pool`
# the same for a larger bootstrap.
COHORT_SPLIT: Final[Mapping[Cohort, Split]] = {
    Cohort.BOOTSTRAP: Split.TRAIN,
    Cohort.POOL: Split.TRAIN,
    Cohort.EVAL: Split.VAL,
    Cohort.RESERVE: Split.VAL,
}


# --- Partition versions -------------------------------------------------------
#
# A partition version is not a label attached to whatever the partitioner did on
# some afternoon; it is the draw. One seed and four cohort sizes produce one
# assignment of the 80,000 images, so both are recorded here per version and the
# partitioner takes a version and nothing else.
#
# **The seed is the irreversible half.** Every cycle's numbers are conditional on
# which 8,000 images the champion started from, and an unrecorded seed makes that
# set unrecoverable. A `--seed` flag would be a number a hand can mistype into a
# partition that is valid, different, and indistinguishable from the intended
# one; there is no flag, so a version can only be drawn one way.
#
# **Sizes belong beside it.** Growing `eval` moves the ruler every earlier cycle
# was measured against, so it is a new version rather than a re-run of an
# existing one -- which is only expressible if a version fixes its sizes.


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """What one `partition_version` means. Add a version, never edit one.

    Editing an entry re-defines a partition that has already been drawn, and
    every artifact keyed by that version disagrees with it from then on.
    """

    seed: PartitionSeed
    sizes: Mapping[Cohort, int]

    def __post_init__(self) -> None:
        missing = sorted(cohort.value for cohort in set(Cohort) - set(self.sizes))
        if missing:
            raise ValueError(f"partition spec sizes no cohort: {missing}")
        negative = {cohort.value: size for cohort, size in self.sizes.items() if size < 0}
        if negative:
            raise ValueError(f"partition spec sizes a cohort negatively: {negative}")

    def quotas(self, split: Split) -> tuple[tuple[Cohort, int], ...]:
        """The cohorts drawing from one split, in draw order, with their sizes."""
        return tuple(
            (cohort, self.sizes[cohort])
            for cohort, source in COHORT_SPLIT.items()
            if source is split
        )


PARTITIONS: Final[Mapping[PartitionVersion, PartitionSpec]] = {
    # The sizes are the decision, and each one is load-bearing on its own.
    # `eval` at 5,000 is what puts the overall metric's noise band narrow enough
    # for a cycle's gain to clear it; `bootstrap` at 8,000 leaves the model
    # headroom for a 1,000-label cycle to move the metric; `pool` at 62,000 makes
    # one cycle 1.6% of what was scored, which is the selectivity a ranking needs
    # to diverge from a random draw.
    #
    # The seed is the date the cohort sizes were settled, which is a way of
    # saying it means nothing -- recorded so that its arbitrariness is on the
    # record and nobody improves it.
    PartitionVersion(0): PartitionSpec(
        seed=PartitionSeed(20260819),
        sizes={
            Cohort.BOOTSTRAP: 8_000,
            Cohort.POOL: 62_000,
            Cohort.EVAL: 5_000,
            Cohort.RESERVE: 5_000,
        },
    ),
}


def partition_spec(partition_version: PartitionVersion) -> PartitionSpec:
    """The seed and sizes a version is drawn with, or a refusal.

    A version nobody wrote down is a partition nobody can reproduce, so it is a
    partition nobody can write.
    """
    spec = PARTITIONS.get(partition_version)
    if spec is None:
        listed = ", ".join(str(version) for version in sorted(PARTITIONS))
        raise ValueError(f"partition version {partition_version} is not defined. Defined: {listed}")
    return spec


class ModelArtifact(StrEnum):
    """Filenames under a seed's model prefix.

    `TORCH` is the checkpoint the scoring job loads. `ONNX` is the int8 graph the
    fleet runs and the file the manifest's digest names. `SHA256` lists the
    digests of the other two in the format `sha256sums` reads.
    """

    ONNX = "model.onnx"
    TORCH = "model.pt"
    SHA256 = "model.sha256"


# A `sha256sum -c` line is the digest and the filename, and nothing else. A
# third field is a file this package did not write, not a line to read past.
_SHA256SUM_FIELDS: Final = 2


def sha256sums_document(digests: Mapping[ModelArtifact, str]) -> str:
    """The `ModelArtifact.SHA256` file, as the training job writes it.

    `sha256sum -c` format -- `<digest>  <filename>`, two spaces, one line per
    artifact -- so the device verifies its download with the tool it already has
    rather than with a parser this project ships to it.

    Written here rather than in the job so that the writer and `sha256sums`
    below cannot disagree about the format. They are the two halves of one file,
    separated by an S3 object and several hours.
    """
    return "".join(
        f"{digests[artifact]}  {artifact.value}\n" for artifact in sorted(digests, key=str)
    )


def sha256sums(document: str) -> dict[ModelArtifact, str]:
    """The digests a `ModelArtifact.SHA256` file lists, keyed by artifact.

    Strict about both halves of every line, because this is the last place a
    malformed digest can be caught with the bytes still nearby: the next reader
    is a device refusing to load a model, hours later and out of reach. A line
    naming a file the enum does not know is refused rather than skipped -- an
    artifact this package cannot name is one nothing here can deploy.
    """
    digests: dict[ModelArtifact, str] = {}
    for number, line in enumerate(document.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != _SHA256SUM_FIELDS:
            raise ValueError(f"line {number} is not `<digest>  <filename>`: {line!r}")
        digest, filename = fields
        if not _SHA256.match(digest):
            raise ValueError(f"line {number} is not a lowercase hex sha256: {digest!r}")
        try:
            artifact = ModelArtifact(filename)
        except ValueError:
            listed = ", ".join(sorted(str(item) for item in ModelArtifact))
            raise ValueError(
                f"line {number} names {filename!r}, which is not a model artifact. Named: {listed}"
            ) from None
        if artifact in digests:
            raise ValueError(
                f"line {number} repeats {filename}, with no way to tell which line is the file"
            )
        digests[artifact] = digest
    return digests


# --- Buckets -----------------------------------------------------------------

PROJECT: Final = "edge-ml-flywheel"


@dataclass(frozen=True, slots=True)
class Buckets:
    """The three buckets, on the `<project>-<purpose>-<account_id>` pattern the
    state bucket already established.

    Split by immutability rule, writer and lifecycle rather than by tidiness.
    `telemetry` is not versioned -- Firehose writes a large number of objects and
    every retained version is billed forever -- and it is the only bucket with
    expiry rules, which would otherwise have to be scoped carefully around real
    data in the same bucket. Buckets cost nothing; a lifecycle rule that matches
    one prefix too many costs the data.

    Terraform state lives in its own hand-bootstrapped bucket and is not here.
    """

    data: str
    artifacts: str
    telemetry: str

    @classmethod
    def for_account(cls, account_id: str) -> Self:
        if not _ACCOUNT_ID.match(account_id):
            raise ValueError(f"not a 12-digit AWS account ID: {account_id!r}")
        return cls(
            data=f"{PROJECT}-data-{account_id}",
            artifacts=f"{PROJECT}-artifacts-{account_id}",
            telemetry=f"{PROJECT}-telemetry-{account_id}",
        )


def uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


# --- Tables -------------------------------------------------------------------


class Table(StrEnum):
    """Every table carries `run_id` in its partition key (design section 7).

    `RUNS` is the one whose partition key is `run_id` alone -- it is the
    registration itself, one item per run. Key design cannot be changed after a
    table is created, which is what puts these names here rather than in
    whichever module happens to touch a table first.
    """

    RUNS = "runs"
    LABEL_BUDGET = "label_budget"
    FLEET_CONFIG = "fleet_config"
    AUDIT_LOG = "audit_log"
    RUN_LOCKS = "run_locks"


def table_name(table: Table) -> str:
    """No account ID, unlike buckets: table names are per account and region."""
    return f"{PROJECT}-{table.value}"


# --- fleet_config entities ----------------------------------------------------
#
# `Table.FLEET_CONFIG` sorts on `entity`, and one shape of item uses it: the
# run's own control item, at `run`.
#
# **There is no per-device item, and that is the fleet plane's answer rather than
# a gap it left.** The design reserved `device#<n>` to carry a `desired_version`
# a device would read, and asked which of that item or the Greengrass deployment
# would be the record of intent (design section 6). A deployment already is one:
# it names a component version, the service holds it, and a device is told what
# to run rather than polling for it. An item beside it would be a second copy of
# one fact, written by a different call, with nothing to say which is right when
# they disagree -- and the disagreement would be invisible, because the copy the
# device acts on is the one nothing here reads.
#
# So the spelling stays reserved and unused. What a device is running is read
# back from the deployment and from what the device itself reports, both of which
# are produced by the thing that actually decides it.
#
# The run item is where the cycle counter lives, which makes this string
# load-bearing beyond addressing. Advancing the counter is a conditional update
# against exactly this item, and that update is the single-flight lock (design
# section 5): it is the one place two overlapping cycles become representable,
# so it is the one place they can be refused. Two spellings of the entity would
# be two locks, which is no lock.

RUN_ENTITY: Final = "run"


# --- Audit log sort keys ------------------------------------------------------
#
# `Table.AUDIT_LOG` sorts on a composed `event` string rather than a timestamp,
# and the choice is load-bearing rather than stylistic. The oracle's idempotency
# key is `(run_id, cycle, sha256(sorted image ids))`; a conditional put on a sort
# key built from exactly those three fields makes one write do three jobs --
# record the charge, refuse the double charge, and cache what a retry replays. A
# timestamp cannot, because a retry carries a different one and would land beside
# the original as a second charge.
#
# The grammar lives here rather than in the module that writes it, for the reason
# the whole module exists: a key spelled at two call sites drifts at one of them,
# and this one cannot be corrected afterwards -- the item it addresses is the
# evidence that a charge happened.
#
# **The cycle is padded inside the string.** A DynamoDB `N` attribute sorts
# numerically and needs no padding, but this is an `S`, so `cycle=10` would sort
# before `cycle=2` and a query for a run's events in order would be wrong.
#
# **The digest is over the sorted image IDs**, so the same batch proposed in a
# different order is the same purchase. Selection ranks its output, and a retry
# that re-ranks under a tie would otherwise pay twice for one batch.

AUDIT_SEPARATOR: Final = "#"

# Digest input is newline-joined rather than concatenated. Image IDs are fixed
# width today, so the two agree; a separator means they still agree if a future
# `label_source` ever changes that, instead of two different batches hashing
# alike.
_DIGEST_SEPARATOR: Final = "\n"


class AuditEvent(StrEnum):
    """What kind of thing an audit item records.

    Leading component of the sort key, so a run's purchases are one `begins_with`
    query and stay separable from the promotions and rejections that join them
    later. Only the oracle's is defined; the registry adds its own.
    """

    PURCHASE = "purchase"


def batch_digest(image_ids: Iterable[ImageId]) -> str:
    """Identify a batch by its contents, independent of order.

    Refuses a duplicate rather than folding it away. Deduplicating here would
    make two genuinely different requests -- one image asked for once, and the
    same image asked for twice -- hash to one key, and the second is a request
    the budget would charge twice for. The oracle's gate refuses it upstream;
    this refuses to give it a name.
    """
    ordered = sorted(image_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("a batch digest is over distinct image IDs, and this batch repeats one")
    if not ordered:
        raise ValueError("a batch of no images has no digest")
    return hashlib.sha256(_DIGEST_SEPARATOR.join(ordered).encode()).hexdigest()


def purchase_event(cycle: Cycle, digest: str) -> str:
    """The `audit_log` sort key one purchase claims.

    Built from the two halves of the idempotency key that are not already the
    partition key. `run_id` is deliberately absent: it is the partition, and
    repeating it here would put the same fact in the item twice with no way to
    say which is right when they disagree.
    """
    if not _SHA256.match(digest):
        raise ValueError(f"not a lowercase hex sha256 batch digest: {digest!r}")
    padded = _padded("cycle", cycle, CYCLE_DIGITS)
    return AUDIT_SEPARATOR.join((AuditEvent.PURCHASE.value, f"c{padded}", digest))


def parse_purchase_event(event: str) -> tuple[Cycle, str]:
    """Split a purchase sort key back into its cycle and digest.

    The inverse exists so that reading the ledger is not string surgery at the
    call site, and so the format has a test that would fail on a one-sided
    change to either half.
    """
    kind, cycle, digest = event.split(AUDIT_SEPARATOR, maxsplit=2)
    if kind != AuditEvent.PURCHASE.value:
        raise ValueError(f"not a {AuditEvent.PURCHASE.value} event: {event!r}")
    if not cycle.startswith("c") or len(cycle) != CYCLE_DIGITS + 1:
        raise ValueError(f"purchase event has no padded cycle: {event!r}")
    if not _SHA256.match(digest):
        raise ValueError(f"purchase event carries no sha256 digest: {event!r}")
    return Cycle(int(cycle[1:])), digest


# --- Data bucket: raw ---------------------------------------------------------
#
# Immutable after ingest. Every reprocessing reads from it, which is what makes
# the "confidence scores are regenerable offline" claim (design section 8) true.
# Enforced by a bucket policy denying writes to every principal but the ingest
# role, not by everyone remembering.
#
# `raw/labels/` carries the ground truth for all 80,000 images, of every cohort.
# The oracle reads it directly and keeps `eval` out of reach in code, against the
# assignments -- cohort is a column there and not a component of any key, so no
# policy can draw that line. What a policy *can* do is keep everyone else out
# entirely, and that is the division: a training job that could GET these files
# would bypass the oracle, the ledger and the cost-per-label deliverable while
# every gate still passed, so the training role is denied this prefix in its own
# policy and again in the bucket policy. An explicit deny beats any allow,
# including one granted later somewhere else.

RAW_PREFIX: Final = "raw/"
RAW_IMAGES_PREFIX: Final = "raw/images/100k/"

# The label format, spelled once. It is both the prefix `raw/labels/` is keyed
# by and the value every manifest row carries in `label_source`, because those
# are the same fact: switching formats moves the prefix and rewrites the column
# together, which is what makes it a re-ingest rather than a second manifest.
LABEL_SOURCE: Final = "scalabel"
RAW_LABELS_PREFIX: Final = f"raw/labels/{LABEL_SOURCE}/"

# Leads with an underscore on purpose: Hive, Glue and Athena skip paths
# beginning with "_" or ".", so a crawler pointed at `raw/` ignores it. The
# corollary is that queryable data must never live behind one.
RAW_PROVENANCE_PREFIX: Final = "raw/_provenance/"


def raw_image_key(image_id: ImageId, split: Split) -> str:
    return f"{RAW_IMAGES_PREFIX}{split.value}/{_token('image_id', image_id)}.jpg"


def raw_label_key(image_id: ImageId, split: Split) -> str:
    return f"{RAW_LABELS_PREFIX}{split.value}/{_token('image_id', image_id)}.json"


# --- Data bucket: derived -----------------------------------------------------
#
# Two tables, not one. The manifest holds what each image *is* -- attributes,
# box counts, checksum -- and the assignments table holds which cohort the
# partitioner put it in. They are written by different jobs at different times:
# ingest cannot fill in a cohort because the partitioner has not run yet.
#
# Folding cohort into the manifest would mean rewriting all 80,000 rows of image
# facts on every re-partition, once per partition version, with nothing to say
# which copy is authoritative when two disagree. The cost of the split is a join
# on `image_id`, which at 80,000 rows is a hash join over a few megabytes.


def columns(row: "type[DataclassInstance]") -> tuple[str, ...]:
    """Column names in declaration order.

    Writers and readers take their column list from the schema rather than
    restating it, so a renamed field cannot leave one side spelling the old name.
    """
    return tuple(field.name for field in fields(row))


_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One row per image in `derived/manifest/`. Written once, at ingest.

    Every Phase 1 question is a query over this: cohort sizing, eval composition,
    per-slice counts. It has to exist before the cohort labels, which cannot be
    written until the partitioner has assigned cohorts.

    `weather`, `scene` and `timeofday` are enums because their vocabularies were
    measured over all 80,000 images before being written down. The regression
    report's slices and the selection mix record are written against these
    three columns, and a misspelled tag is the one kind of wrong predicate that
    does not fail: it matches nothing, so the slice empties or the mix row reads
    zero, and neither says so -- an empty series charts as a flat line and a zero
    row as a condition the selector simply did not buy. The parquet column stays
    `string` either way -- `StrEnum` serializes to the identical value -- so the
    typing is a parse-boundary guarantee bought without a re-ingest.

    `sha256` is over the image bytes. Design section 4.1's data gate and section
    6's artifact verification both need the raw data content-addressed.

    `label_source` is constant across all 80,000 rows and stored anyway, so a
    single downloaded parquet says which label format produced it. Switching it
    changes what the metric means and forces a fresh champion baseline (design
    section 5), which makes it exactly the fact worth carrying in the file.

    **Every column is a fact the archive states, never a verdict we reached.**
    `box_areas` in native 1280x720 px rather than a small-object count, because a
    count would bake in a threshold and a resolution that appear nowhere in the
    file: change either and every stored count is wrong with nothing to say so.
    It would also be a second definition of "small" competing with the one the
    eval module applies at 416 px, which is the definition the metric actually
    uses. Storing the evidence keeps that threshold in one place -- eval, which
    owns the metric -- and makes any threshold at any resolution a query.

    `n_boxes` is `len(box_areas)` and stored anyway. It is parameter-free, so it
    cannot drift, and it is the column every sanity check and slice count reads;
    a scalar spares those queries from scanning the list column at all.
    """

    image_id: ImageId
    split: Split
    weather: Weather
    scene: Scene
    timeofday: TimeOfDay
    n_boxes: int
    box_areas: tuple[float, ...]
    sha256: str
    label_source: str

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)
        if not _SHA256.match(self.sha256):
            raise ValueError(f"not a lowercase hex sha256 digest: {self.sha256!r}")
        if self.n_boxes != len(self.box_areas):
            raise ValueError(
                f"n_boxes {self.n_boxes} disagrees with {len(self.box_areas)} box areas"
            )
        if any(area <= 0 for area in self.box_areas):
            raise ValueError(f"box area is not positive: {self.box_areas!r}")


@dataclass(frozen=True, slots=True)
class AssignmentRow:
    """One row per image in `derived/partition_version=<v>/assignments/`.

    Written by the partitioner, whose exactly-once guarantee is the statement
    that this table has 80,000 rows with unique `image_id` and no cohort outside
    `Cohort` -- the disjoint-and-complete assertion and the primary leakage
    insurance (design section 5).

    No `partition_version` column: it is in the prefix, so a Hive-partitioned
    table supplies it and storing it again would be the same fact in two places.
    """

    image_id: ImageId
    cohort: Cohort

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)


MANIFEST_PREFIX: Final = "derived/manifest/"


def manifest_key(part: int = 0) -> str:
    """One ~5 MB parquet of 80,000 rows.

    Deliberately not under `partition_prefix`: nothing in a row here changes when
    the partition does. A change of `label_source` does change it, but that is a
    re-ingest -- it moves `RAW_LABELS_PREFIX` too -- not a second manifest.

    The `part` argument exists so a writer that emits multiple files does not
    have to invent a name.
    """
    return f"{MANIFEST_PREFIX}part-{_padded('part', part, PART_DIGITS)}.parquet"


def partition_prefix(partition_version: PartitionVersion) -> str:
    """Keyed by partition version so a re-partition writes alongside, not over."""
    version = _padded("partition_version", partition_version, PARTITION_VERSION_DIGITS)
    return f"derived/partition_version=v{version}/"


def partition_manifest_key(partition_version: PartitionVersion) -> str:
    """The seed and sizes the assignments beside it were drawn with.

    Underscore-prefixed for `RAW_PROVENANCE_PREFIX`'s reason: Glue and Athena
    skip paths beginning with `_`, so a crawler over the partition prefix reads
    `assignments/` and walks past this.

    A copy of the `PARTITIONS` entry, which stays authoritative. Written anyway,
    because the question "which seed produced these 80,000 rows" is asked by
    someone looking at a bucket, and answering it from a source checkout requires
    knowing which commit was current when the partitioner ran.
    """
    return f"{partition_prefix(partition_version)}_partition.json"


def assignments_prefix(partition_version: PartitionVersion) -> str:
    return f"{partition_prefix(partition_version)}assignments/"


def assignments_key(partition_version: PartitionVersion, part: int = 0) -> str:
    """80,000 rows of two columns, well under a megabyte.

    `cohort` is a column here, not a prefix: splitting a file this small four
    ways would make every query slower and buy nothing. The cohort labels under
    this same partition prefix do use `cohort=` as a prefix, because there an IAM
    policy prunes on it.
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{assignments_prefix(partition_version)}part-{name}.parquet"


def cohort_labels_prefix(partition_version: PartitionVersion, cohort: Cohort) -> str:
    """Where the boxes for one labeled cohort live.

    `cohort=` is a prefix rather than a column for one reason, and it is not query
    pruning: `eval`'s boxes are a second copy of ground truth outside the
    `raw/labels/` deny, and an IAM policy can only name a path. Freezing `eval`
    after Phase 1 is the same statement about writes.
    """
    if cohort not in LABELED_COHORTS:
        raise ValueError(f"{cohort.value} has no labels of its own")
    return f"{partition_prefix(partition_version)}labels/cohort={cohort.value}/"


def cohort_labels_key(partition_version: PartitionVersion, cohort: Cohort, part: int = 0) -> str:
    """One cohort's boxes: `image_id` and its encoded boxes, and nothing else.

    Labels only, because the images are already in `raw/images/` and training
    reads them from there. 13,000 images at roughly 18 boxes each is a few
    megabytes, against the ~750 MB that bundling the same images beside their
    boxes would have duplicated.

    One object per cohort rather than a packed set, which `File` input mode
    settles: the channel is copied to local disk once per job, so every epoch
    after it reads disk. Whether object count makes that copy slow enough to
    revisit is measured on the first training jobs rather than guessed (design
    section 11).
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{cohort_labels_prefix(partition_version, cohort)}part-{name}.parquet"


PURCHASES_PREFIX: Final = "derived/purchases/"


def purchases_run_prefix(run_id: RunId) -> str:
    """Everything one run has ever bought, as a single prefix.

    The cumulative labeled set stated as one string, which is what a training
    channel needs: a cycle trains on the bootstrap plus every purchase before it,
    and this prefix holds exactly those because a cycle buys after it trains. A
    channel per cycle would be the same objects named `cycle` times over, and
    would put a cap on how many cycles a run can have.
    """
    return f"{PURCHASES_PREFIX}{run_prefix(run_id)}"


def purchase_labels_prefix(run_id: RunId, cycle: Cycle) -> str:
    """One cycle's bought boxes.

    Keyed by run and cycle rather than by partition version, because which images
    a cycle bought is a fact about that run's selector and budget: two runs over
    one partition buy different images, and the label-efficiency A/B is exactly
    the case where they must not collide. A cycle's training set is therefore the
    bootstrap labels plus every purchase prefix from cycle 1 up to it, which
    states the cumulative labeled set as a key range. That holds for the boxes
    only; the images it names are scattered across one flat prefix and are
    addressed by `training_manifest_key` instead.
    """
    return f"{PURCHASES_PREFIX}{cycle_prefix(run_id, cycle)}"


def purchase_labels_key(run_id: RunId, cycle: Cycle, part: int = 0) -> str:
    """A cycle's purchase in one file, padded to sort with the rest.

    A thousand images of boxes is well under a megabyte, so the part number is
    here to match `cohort_labels_key` rather than because a cycle is expected to
    need a second file.
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{purchase_labels_prefix(run_id, cycle)}part-{name}.parquet"


# --- Artifacts bucket ---------------------------------------------------------
#
# Write-once. Nothing under a cycle prefix is ever rewritten and a re-run is a
# new `run_id`, which is what makes gate reports and model manifests audit
# evidence rather than current state.

# The COCO-pretrained checkpoint every seed of every cycle fine-tunes from, held
# in the bucket rather than downloaded by the training job. Two reasons, and the
# second is the one that matters: a download is an unpinned dependency on a host
# outside this project, and "from the COCO base every time" (design section 3) is
# a claim about the same bytes, which only a stored file can settle. Its digest
# is logged by every job.
#
# Outside the run prefixes because it is a function of the recipe rather than of
# any run, and named by file because the file name *is* the model: a
# `recipe_version` that changes the base changes this string.
BASE_WEIGHTS_PREFIX: Final = "base/"
BASE_WEIGHTS_FILE: Final = "yolo11n.pt"

# Where the object comes from the one time it is absent. `launch.ensure_base`
# fetches it on the first cycle of a fresh account and every later cycle finds it
# already in the bucket, so the claim above holds from the second job onward
# without a setup command anyone has to know to run. Pinned to a release tag
# rather than `latest`, since the tag is the only part of this URL that says
# which checkpoint the recipe means.
BASE_WEIGHTS_URL: Final = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)


def base_weights_key(file: str = BASE_WEIGHTS_FILE) -> str:
    return f"{BASE_WEIGHTS_PREFIX}{_token('base weights', file)}"


def run_prefix(run_id: RunId) -> str:
    return f"run_id={_token('run_id', run_id)}/"


def cycle_prefix(run_id: RunId, cycle: Cycle) -> str:
    return f"{run_prefix(run_id)}cycle={_padded('cycle', cycle, CYCLE_DIGITS)}/"


def model_prefix(version: ModelVersion) -> str:
    """Run and cycle are recovered from the version, never passed beside it."""
    return f"{cycle_prefix(*_locate(version))}models/version={version}/"


def model_manifest_key(version: ModelVersion) -> str:
    """A model with no manifest cannot be promoted (design section 5)."""
    return f"{model_prefix(version)}manifest.json"


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's verdict. Pass or fail is never recorded without its reason.

    The rejection log is a feature (design section 5), and a bare `False` six
    weeks later is a fact nobody can act on.
    """

    gate: str
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """The document at `model_manifest_key`. Written once, never updated.

    A model file on its own is an anonymous blob: nothing in it says what data
    trained it, what code produced it, or what it was measured against. Every
    field here is cheap to record at training time and impossible to reconstruct
    afterwards, which is why "no manifest, no promotion" is a precondition rather
    than paperwork.

    The versions are repeated from the run registration on purpose. They are not
    a second source of truth -- `RunRegistration` remains authoritative -- but a
    manifest that disagrees with its run is a bug the registration step can only
    detect if the manifest states its own view.

    Run and cycle are *not* repeated, and the difference is the point: a version
    is a fact this model claims about itself and can be wrong about, while the
    run and cycle are already inside `version` and could only ever be restated.
    They are properties here rather than fields, so the serialized document has
    one spelling of them and readers still get both typed.

    `cohorts_trained_on` records which cohorts the training set drew from:
    `bootstrap` alone at cycle 0, and `bootstrap` with `pool` once any labels
    have been bought. Exactly *which* pool images is a function of run and cycle,
    both already inside `version`, so this field is not an inventory -- it is the
    leakage statement, and `eval` appearing in it is the failure the field exists
    to make representable and then refuse.

    `artifact_sha256` is the digest of each seed's deployable file, every seed
    the cycle trained and not only the deployed one -- the matched-seed saving in
    design section 7 depends on each one existing and being identifiable at the
    cycle it was trained in. The device agent verifies the digest before loading
    (design section 6), so a truncated download becomes a rejection instead of a
    model that silently returns nonsense.

    The file it names is `ModelArtifact.ONNX`, the int8 graph, because that is
    what the device loads and a digest of anything else verifies nothing it
    does. Every seed exports one, so the field means the same thing for the seed
    that ships and for the seeds retained beside it. The digest is taken where
    the bytes were produced and published beside them as `ModelArtifact.SHA256`,
    which `sha256sums` reads the line out of.

    `registry.manifest` serializes this, and the registration step writes it
    before it reads it back.
    """

    version: ModelVersion
    created_at: datetime
    git_commit: str
    partition_version: PartitionVersion
    recipe_version: RecipeVersion
    cohorts_trained_on: frozenset[Cohort]
    labels_spent: int
    deployed_seed: Seed
    artifact_sha256: Mapping[Seed, str]
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        parse_model_version(self.version)
        if self.created_at.tzinfo is None:
            raise ValueError(f"created_at must be timezone-aware: {self.created_at!r}")
        if not _GIT_COMMIT.match(self.git_commit):
            raise ValueError(f"not a full 40-character git commit SHA: {self.git_commit!r}")
        if self.deployed_seed not in self.artifact_sha256:
            raise ValueError(
                f"deployed seed {self.deployed_seed} has no artifact digest in the manifest"
            )
        for seed, digest in self.artifact_sha256.items():
            if not _SHA256.match(digest):
                raise ValueError(f"seed {seed} digest is not a lowercase hex sha256: {digest!r}")
        if not self.cohorts_trained_on:
            raise ValueError("a model trained on no cohort is not a model")
        if Cohort.EVAL in self.cohorts_trained_on:
            raise ValueError(f"trained on the {Cohort.EVAL.value} cohort")

    @property
    def run_id(self) -> RunId:
        return model_version_run_id(self.version)

    @property
    def cycle(self) -> Cycle:
        return model_version_cycle(self.version)

    def disagreements(self, run: RunRegistration) -> tuple[str, ...]:
        """Field names where this model contradicts the run it claims to be in.

        Empty means the model is admissible for comparison inside `run`. Anything
        else is a hard refusal to promote (design section 5), not a low score: a
        mismatch does not say the model is worse, it says the comparison the
        gates ran was not the comparison they reported. Checked at promotion for
        the same reason "no manifest, no promotion" is -- it is a precondition of
        the decision, and afterwards there is nothing left to check it against.

        Field names rather than a bool, because a rejection is recorded with its
        reason (`GateResult`), and "rejected" without "recipe_version was 3 and
        the run declared 4" is not something anyone can act on six weeks later.

        Two distinct failures fall out together. `run_id` disagreeing means the
        model belongs to another run entirely -- checkable at all only because
        the version encodes its run. A version disagreeing means the training job
        ran a configuration the run never declared, which is exactly what this
        class restates the versions to expose. Copying the registration's values
        in at construction would make a disagreement unrepresentable, and would
        do it by recording the run's intent in place of the job's behaviour --
        silencing the witness rather than believing it.
        """
        return tuple(
            name
            for name, claimed, declared in (
                ("run_id", self.run_id, run.run_id),
                ("partition_version", self.partition_version, run.partition_version),
                ("recipe_version", self.recipe_version, run.recipe_version),
            )
            if claimed != declared
        )

    @property
    def gates_passed(self) -> bool:
        """Every gate reported and every one green.

        An empty `gates` is false rather than vacuously true: a model whose gates
        never ran is the exact case this is here to stop.
        """
        return bool(self.gates) and all(gate.passed for gate in self.gates)


def model_seed_prefix(version: ModelVersion, seed: Seed) -> str:
    """One seed's own prefix, which is what a training job is pointed at.

    Per seed, because every champion artifact is retained. Seed 1 is the one that
    ships, by convention, and the only one a single-seed cycle produces. A cycle
    trained at more keeps the rest as well: they are what the matched-seed cost
    saving in design section 7 depends on, and keeping only seed 1 quietly
    removes it.
    """
    return f"{model_prefix(version)}seed={_padded('seed', seed, SEED_DIGITS)}/"


def model_artifact_key(version: ModelVersion, seed: Seed, artifact: ModelArtifact) -> str:
    return f"{model_seed_prefix(version, seed)}{artifact.value}"


def eval_prefix(version: ModelVersion) -> str:
    """Under the cycle that produced the model, not the cycle being decided.

    A champion is re-compared every cycle without being re-evaluated: the eval
    cohort is frozen, so its scores are a function of the model alone and the
    cached results stay valid where they were first written (design section 7).
    Keying this by the deciding cycle instead would write a fresh copy of an
    unchanged answer every cycle and make the reuse impossible to express.
    """
    return f"{cycle_prefix(*_locate(version))}eval/version={version}/"


def eval_metrics_key(version: ModelVersion) -> str:
    return f"{eval_prefix(version)}metrics.json"


def eval_matches_key(version: ModelVersion, seed: Seed) -> str:
    """The cached per-image match arrays the paired bootstrap resamples.

    One file per model-seed, not per image. The bootstrap reads it a thousand
    times, so it has to arrive in a single GET and live in memory; 16,000
    separate objects would be unusable.
    """
    return f"{eval_prefix(version)}seed={_padded('seed', seed, SEED_DIGITS)}/matches.npz"


# --- Artifacts bucket: detections ---------------------------------------------
#
# What one model saw in the images of one cohort, before anything has been
# compared against ground truth. It is the input to both halves of the cycle that
# follow -- the match arrays the gates read, and the uncertainty ranking the
# budget is spent down -- and it is produced once because scoring 67,000 images
# is the expensive step and every later question is a query over its output
# (design section 4.3).
#
# **Detections are not a metric.** Nothing here has been matched, scored or
# judged; a detections file says what the model emitted and stays true whatever
# is later decided about it. That is what lets one file serve two readers who
# disagree about everything else: evaluation cares about the low-confidence tail
# because AP sweeps it, and selection throws that tail away because a box at
# 0.001 is a decision the model already made.


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """One box a model predicted, as a row of a detections file.

    The fields of `evaluation.coco.Detection` plus the image it was found in, in
    the same frame -- corners in `NATIVE_IMAGE_SIZE` pixels. Deliberately the
    same seven facts in the same units, so reading a detections file back is a
    field-for-field construction rather than a conversion with a rescale hidden
    in it. Exactly one component rescales from the model's 416 px, and it is the
    one holding the model.

    `category` is the archive's name rather than a `ClassSet` position, unlike a
    cached match array, which holds `2` and not `truck`. Those arrays are numpy,
    where an integer is what there is to store; this is parquet, where a repeated
    string is dictionary-encoded to the same integer anyway. What the name buys
    is a file that still means something read without the class set beside it,
    and `selection.score.Predictions` matches on names -- so storing IDs would be
    converting twice to arrive back where the data started.

    No `cohort` column and no `version`: both are in the prefix, for
    `AssignmentRow`'s reason.
    """

    image_id: ImageId
    category: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)
        # The same range `selection.score.uncertainty` refuses, checked where the
        # number enters the project rather than where it is first divided by.
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"a confidence outside [0, 1] is not a confidence: {self.score}")
        # Corners in the order `Box` and `Detection` state them. A reversed pair
        # converts to a negative width, which COCO's area filter reads as a box
        # smaller than every threshold rather than as a malformed one -- so it
        # would drop out of the small-object slice silently instead of raising.
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"detection corners do not enclose a positive area: "
                f"({self.x1}, {self.y1}) to ({self.x2}, {self.y2})"
            )


def _scored(cohort: Cohort) -> Cohort:
    """Refuse a cohort nothing scores, while the caller still holds a cohort.

    `cohort_labels_prefix`'s arrangement: the gate is the thing that produces the
    path rather than a check someone performs before building one, so there is no
    argument to these builders that names a detections file for `bootstrap`.
    """
    if cohort not in SCORED_COHORTS:
        listed = ", ".join(sorted(member.value for member in SCORED_COHORTS))
        raise ValueError(f"{cohort.value} is not a scored cohort ({listed} are)")
    return cohort


def scoring_manifest_key(run_id: RunId, cycle: Cycle, cohort: Cohort) -> str:
    """The image keys one cycle scores for one cohort, in `ManifestFile` form.

    `training_manifest_key`'s argument applied to the other set of images a cycle
    puts in front of a model, and split by cohort for a reason that document does
    not have. A manifest names keys below one shared prefix, `eval` draws from
    `val` and `pool` from `train`, so a single document covering both would have
    to name the prefix above the split -- which is the prefix holding all 80,000
    images, and an input channel pointed there is one that can reach a frame the
    cycle did not mean to score. The split `COHORT_SPLIT` already records is
    therefore the split between two manifests.

    Per cycle rather than per seed, like the training manifest: every seed of a
    cycle scores the same images, and a document per seed would be one list
    written once per seed with nothing to say which was authoritative.

    Under the write-once cycle prefix because it is also the record of what was
    scored. The remaining pool shrinks every cycle, so "which 61,000 images was
    cycle two's ranking over" is not recoverable afterwards from a partition and
    a ledger without replaying every purchase in order.
    """
    return f"{cycle_prefix(run_id, cycle)}scoring/images-{_scored(cohort).value}.manifest"


class Precision(StrEnum):
    """Which build of a model produced a set of detections.

    `FP32` is the checkpoint, which is what the paired comparison, the selector
    and every reported metric are computed from. `INT8` is the quantized ONNX
    graph the fleet runs, scored once per cycle over `eval` alone so the edge
    gate can say what quantization cost.

    A dimension of the detections key rather than a flag on a file, because the
    two are produced by separate jobs and read by separate consumers, and a
    quantized box sitting in the prefix the selector reads would rank the pool by
    the wrong model's uncertainty.
    """

    FP32 = "fp32"
    INT8 = "int8"


def detections_prefix(
    version: ModelVersion,
    seed: Seed,
    cohort: Cohort,
    precision: Precision = Precision.FP32,
) -> str:
    """Where one model-seed's boxes for one cohort land.

    Keyed by version and seed because detections are a function of the model, and
    by cohort because the two are written by separate output channels of one job
    and read by separate consumers -- which is the `key=value/` rule's "something
    prunes on it", here a job's upload and a reader's prefix rather than a query
    engine. By precision for the same reason: the int8 pass is a second job over
    the same cohort, and the two sets of boxes answer different questions.

    `FP32` is the default because it is the pass every cycle runs and the one
    every existing reader wants. The segment is written for both, so neither is
    the unmarked case a reader has to know about.

    Under the cycle that produced the model, not the cycle being decided, for
    `eval_prefix`'s reason: a champion is re-compared every cycle without being
    re-scored, and keying this by the deciding cycle would write a fresh copy of
    an unchanged answer every time.
    """
    padded = _padded("seed", seed, SEED_DIGITS)
    return (
        f"{cycle_prefix(*_locate(version))}detections/version={version}/"
        f"seed={padded}/cohort={_scored(cohort).value}/precision={precision.value}/"
    )


def detections_key(
    version: ModelVersion,
    seed: Seed,
    cohort: Cohort,
    part: int = 0,
    precision: Precision = Precision.FP32,
) -> str:
    """One cohort's detections in one parquet.

    Parquet rather than the `.npz` the match cache uses, because these have two
    readers that want different rows: evaluation takes the whole file and
    selection takes a confidence band out of it, and a columnar file answers the
    second without decompressing the first. The match arrays have exactly one
    reader and are indexed rather than filtered, which is why they stay numpy.
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{detections_prefix(version, seed, cohort, precision)}part-{name}.parquet"


def gate_report_prefix(run_id: RunId, cycle: Cycle) -> str:
    """The directory the report lands in, which is what an output channel names.

    Split out of `gate_report_key` for `detections_prefix`' reason: SageMaker
    uploads a Processing output channel by prefix and the container writes a file
    under it, so the two halves of that key are addressed by two callers. Deriving
    the directory by trimming the key at the call site would be the one spelling
    `conventions` exists to prevent.
    """
    return f"{cycle_prefix(run_id, cycle)}gates/"


def gate_report_key(run_id: RunId, cycle: Cycle, suffix: str = "json") -> str:
    """Per cycle, not per model: the report covers the comparison, not one side.

    Too fat to live in the audit log -- the DynamoDB item cap is 400 KB and it
    would be a blob in a database. The log holds the event and a `detail_uri`
    pointing here.
    """
    return f"{gate_report_prefix(run_id, cycle)}report.{_token('suffix', suffix)}"


@dataclass(frozen=True, slots=True)
class SelectionRow:
    """One pool image's place in a cycle's ranking, and whether it was bought.

    `rank` is the position the ordering put it in, 0 for the most uncertain. It is
    derivable from `score` plus the image ID tie-break, and it is stored anyway
    for the reason `ManifestRow.n_boxes` is: a reader that re-derives it has to
    know the tie-break rule, and a query engine hands back rows in whatever order
    it likes. Storing it makes the file say what the ranking *was* rather than
    what it can be reconstructed as.

    `selected` is the batch. The top `budget` rows carry it, so the batch is a
    predicate over this file rather than a second document that can disagree with
    it -- which is what lets the purchase read its image IDs out of the evidence
    that they were chosen.

    No `version` or `seed` column, and the absence is worth stating: a cycle ranks
    once, on one model, so they would be constant down all 62,000 rows. Which
    model did the scoring is in `detections_prefix`, and which cycle this is is in
    the key.
    """

    image_id: ImageId
    score: float
    rank: int
    selected: bool

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)
        # The range `selection.score` produces, checked where the number is
        # written rather than where it is next compared. A score outside it is a
        # scorer that changed under a file already in the bucket.
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"an uncertainty score outside [0, 1]: {self.score}")
        if self.rank < 0:
            raise ValueError(f"a rank is a position and cannot be negative: {self.rank}")


def selection_prefix(run_id: RunId, cycle: Cycle) -> str:
    """Where one cycle's selection files land.

    Per cycle rather than per model, like the gate report: selection describes the
    purchase, and a cycle makes exactly one.
    """
    return f"{cycle_prefix(run_id, cycle)}selection/"


def selection_ranking_key(run_id: RunId, cycle: Cycle, part: int = 0) -> str:
    """The whole ranked pool, and which of it this cycle bought.

    One file rather than a ranking and a batch beside it. The batch is the top of
    the ranking by definition, so two documents would be one fact with a way to
    disagree -- and the purchase reads its image IDs out of the same rows the
    ranking is evidence of, rather than out of a second document that says it
    agrees.

    Under the write-once cycle prefix because it is the evidence the selector
    works, and it is not recoverable afterwards. The ledger records which images
    were bought; only this says what they were chosen *over* -- the 61,000 that
    scored lower, which is the comparison the whole ranking rests on.

    A parquet rather than JSON: it is one row per pool image, which is 62,000 of
    them at cycle one, and the chart that reads it wants two columns of the set
    rather than the document.
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{selection_prefix(run_id, cycle)}ranking-part-{name}.parquet"


def selection_report_key(run_id: RunId, cycle: Cycle) -> str:
    """What this cycle's batch was made of, beside what it left in the pool.

    Per cycle rather than per model, like the gate report: it describes the
    purchase, and a cycle makes exactly one. Under the write-once cycle prefix
    because it is evidence about a decision already taken -- a batch's condition
    mix is not recoverable later from the ledger, which records which images were
    bought and nothing about the pool they were drawn out of.

    Not written yet. The mix is a join of `selection_ranking_key` onto the image
    manifest, and the manifest is zstd -- which the control plane's pyarrow cannot
    open (see `partition.assign._COMPRESSION`). So the condition mix is a query
    over two files in the bucket rather than a third file, and it lands with the
    composition chart that is its only reader.
    """
    return f"{selection_prefix(run_id, cycle)}report.json"


# The source archive's file name, named because two jobs address it and only one
# of them has a framework that unpacks it. See `training_code_key`.
TRAINING_CODE_FILE: Final = "sourcedir.tar.gz"


def training_manifest_key(run_id: RunId, cycle: Cycle) -> str:
    """The image keys one cycle trains on, in SageMaker `ManifestFile` form.

    The labels are a key range -- bootstrap plus every purchase prefix up to this
    cycle -- but the images are not. They sit flat under `raw_image_key`'s train
    prefix among all 70,000, and the cumulative labeled set is a scattered subset
    of those. An `S3Prefix` channel would take the whole prefix, so the subset has
    to be named object by object, which is what `ManifestFile` is for.

    The document is a JSON array whose first element is `{"prefix": <s3 uri>}`
    and whose rest are keys relative to it. Every listed object therefore shares
    one prefix, which holds here because `raw_image_key` varies only by split and
    a training set never crosses one.

    Written at prepare time and read by every seed of the cycle, so it doubles as the
    record of what the challenger trained on: one list under a write-once cycle
    prefix, rather than a set reconstructed later from a ledger and a partition.
    """
    return f"{cycle_prefix(run_id, cycle)}training/images.manifest"


def training_code_key(run_id: RunId, cycle: Cycle) -> str:
    """The package as the training container receives it.

    SageMaker's script mode takes the code as one archive in S3 and unpacks it
    beside the entry point, so the archive exists whatever else is true. Filing
    it under the write-once cycle prefix rather than in a scratch location is
    what makes it evidence: the manifest records a `git_commit`, and this is the
    tree that commit produced, uploaded before the job that read it started.

    Beside `training_manifest_key` because they are the two objects one cycle
    hands its seeds, and they are read by the same role under the same grant.

    Read by the scoring job as well, which is why the file name is a constant: a
    Processing job has no script mode to unpack an archive for it, so the
    container command names this file directly and a second spelling would be a
    command that unpacks nothing. Scoring reusing the training archive is also
    the property worth having -- the code that scored a model is the same tree
    that trained it, and the manifest's `git_commit` covers both.
    """
    return f"{cycle_prefix(run_id, cycle)}training/{TRAINING_CODE_FILE}"


# --- Artifacts bucket: the fleet's own objects --------------------------------

# The file name `replay_manifest_key` ends in, named because two readers address
# it and only one of them builds the key. `TRAINING_CODE_FILE`'s situation
# exactly: Greengrass names a downloaded artifact after the last component of its
# key, so the component recipe spells the file directly, and a second spelling
# would be a component that starts and finds no frames.
REPLAY_MANIFEST_FILE: Final = "replay.json"


def replay_manifest_key(run_id: RunId, cycle: Cycle) -> str:
    """The pool frames one cycle put in front of the fleet.

    `scoring_manifest_key`'s argument for the third set of images a cycle shows a
    model, and the one design section 7.2 requires by name: the sample is drawn
    per cycle because the pool shrinks as the run proceeds, and it is the small
    per-cycle list of image IDs that later ties a telemetry record back to a
    scoring decision. Without it a confidence reported from a device is a number
    about a frame nobody can identify.

    Drawn out of `selection_ranking_key`'s rows rather than recomputed from the
    partition and the ledger, so the frames the fleet replays are by construction
    frames the cycle also ranked -- which is what the realism check compares and
    what makes the telemetry regenerable offline from retained images.

    A JSON array of image IDs rather than a manifest of S3 keys: it is read by
    the device, which resolves each ID through `raw_image_key` under its own
    grant, and by a later join against the ranking. Neither wants SageMaker's
    `ManifestFile` envelope.
    """
    return f"{cycle_prefix(run_id, cycle)}fleet/{REPLAY_MANIFEST_FILE}"


# Where the device-side code is held, outside every run prefix because it is a
# function of the commit and of nothing else -- `BASE_WEIGHTS_PREFIX`'s placement
# and its reason. Two cycles built from one tree ship one object, and a component
# version built twice from the same commit names the identical bytes.
REPLAY_CODE_PREFIX: Final = "fleet/code/"

# The archive's file name, and the commit is the *directory* above it rather than
# the name itself. That is not tidiness: Greengrass unpacks an archive into a
# directory named after the file, so a key ending in `<sha>.zip` would put the
# package under a path that changes with every commit -- and the recipe sets
# `PYTHONPATH` to that path, so it would be a component that starts and cannot
# import itself on the first cycle after any change. Holding the name fixed and
# moving the commit above it makes the archive content-addressed and the unpacked
# path constant.
REPLAY_CODE_FILE: Final = "replay.zip"


def replay_code_key(git_commit: str) -> str:
    """The package as the replay component receives it, addressed by commit.

    `training_code_key`'s object for the device rather than for a job, and keyed
    differently for a reason that document does not have. A training archive is
    evidence about one cycle and lives under that cycle's write-once prefix; this
    one is an input to any number of them, and naming it by the tree that
    produced it is what lets a redeploy of the same commit be a no-op instead of
    an overwrite of an object a live component is still pointed at.

    The commit is checked rather than trusted, because the name is the whole of
    the addressing: a truncated SHA would quietly claim a prefix a full one never
    writes to, and the component would ship a tree nobody can identify.
    """
    if not _GIT_COMMIT.match(git_commit):
        raise ValueError(f"not a full 40-character git commit SHA: {git_commit!r}")
    return f"{REPLAY_CODE_PREFIX}{git_commit}/{REPLAY_CODE_FILE}"


# --- Fleet: Greengrass components ---------------------------------------------
#
# A promoted model reaches a device as a Greengrass component (design section 6),
# and a component is addressed by a name and a version. A model version cannot be
# either of them as it stands: Greengrass requires the version to be semantic --
# `<major>.<minor>.<patch>` -- and a model version is a timestamp, a slug and a
# cycle.
#
# So the two halves of a model version are split across the two halves of a
# component's address. **The name carries the run and the version carries the
# cycle**, which is `model_package_group`'s arrangement applied to the other
# registry a model is held in: one component per run, one version per cycle,
# because the champion a version is compared against is a fact about its run
# (design section 5). Two runs therefore cannot collide on a component version,
# which they would if the version were the cycle alone -- and a Greengrass
# component version is immutable once created, so that collision is one nothing
# can clear up afterwards.
#
# **The cycle is not padded, and this is the one place in this module where
# padding would be wrong.** Semver forbids a leading zero in a numeric
# identifier, so `0.003.0` is a string no Greengrass API accepts. Nor is padding
# needed: semver compares those identifiers numerically, so `0.10.0` already
# sorts above `0.9.0` without the help that `CYCLE_DIGITS` exists to give every
# key that sorts lexicographically.

# The major and patch are fixed at zero and only the minor moves. A major of 0
# says what is true -- nothing about this component's interface is promised
# between cycles -- and spending the other two identifiers on anything would mean
# inventing a second number the cycle does not supply.
_COMPONENT_MAJOR: Final = 0
_COMPONENT_PATCH: Final = 0

_COMPONENT_VERSION: Final = re.compile(
    rf"^{_COMPONENT_MAJOR}\.(?P<cycle>0|[1-9][0-9]*)\.{_COMPONENT_PATCH}$"
)

# What appears as the publisher on every component version. A required field with
# no addressing role, so it is the project and nothing more.
COMPONENT_PUBLISHER: Final = PROJECT

# Greengrass's ceiling on a component name, and the reason the name is the run
# rather than the run plus a description of it: `PROJECT` and the longest run ID
# `RUN_SLUG_MAX_LEN` permits come to 66 characters, which leaves room and is
# checked anyway for `model_package_group`'s reason -- the worst place to find a
# name too long is the first deployment of a run already several cycles in.
COMPONENT_NAME_MAX: Final = 128


def model_component(run_id: RunId) -> str:
    """The Greengrass component one run's models are deployed as.

    Dot-separated rather than hyphenated, which is the reverse-DNS shape every
    AWS-published component uses and which reads as one name with two parts
    rather than as a longer run ID.
    """
    name = f"{PROJECT}.{parse_run_id(run_id)}"
    if len(name) > COMPONENT_NAME_MAX:
        raise ValueError(
            f"component name is {len(name)} characters, over Greengrass's "
            f"{COMPONENT_NAME_MAX}: {name}"
        )
    return name


def component_version(cycle: Cycle) -> str:
    """One cycle's component version. Unpadded, for the reason stated above."""
    if cycle < 0:
        raise ValueError(f"a cycle is not negative and semver has no sign for it: {cycle}")
    return f"{_COMPONENT_MAJOR}.{cycle}.{_COMPONENT_PATCH}"


def parse_component_version(value: str) -> Cycle:
    """The cycle a component version names.

    The inverse exists so that reading a deployment back is not string surgery at
    the call site, and so a one-sided change to either half fails a test --
    `parse_purchase_event`'s arrangement. It is also what a rollback reads: the
    revision names a version, and which cycle that was is the fact the audit
    record wants.
    """
    match = _COMPONENT_VERSION.match(value)
    if not match:
        raise ValueError(
            f"not a component version this project mints: {value!r}. Expected "
            f"{_COMPONENT_MAJOR}.<cycle>.{_COMPONENT_PATCH} with no leading zero on the cycle"
        )
    return Cycle(int(match["cycle"]))


def component_address(version: ModelVersion) -> tuple[str, str]:
    """The component name and version one model is deployed as.

    Both halves from the one string, never passed beside it -- `model_prefix`'s
    rule, and the reason this exists rather than two calls at every site: a
    caller holding a model version has no business splitting it itself.
    """
    run_id, cycle = _locate(version)
    return model_component(run_id), component_version(cycle)


# --- Telemetry bucket ---------------------------------------------------------

ATHENA_RESULTS_PREFIX: Final = "athena-results/"


def telemetry_run_prefix(run_id: RunId) -> str:
    """Everything one run's fleet has ever said, as a single prefix.

    `purchases_run_prefix`'s shape and its reason. A replay is read back by
    listing rather than by date -- the reader knows which version it is asking
    about and not which afternoon the device ran -- so the day partition below is
    for a query engine and this is for the listing that feeds the canary.
    """
    return f"fleet/{run_prefix(run_id)}"


def telemetry_prefix(run_id: RunId, day: date) -> str:
    """Partitioned by run then day, both filtered on constantly.

    Day granularity, not hour: at this volume hourly partitions produce small
    files and Athena gets slower, not faster.
    """
    return f"{telemetry_run_prefix(run_id)}dt={day.isoformat()}/"


# --- Telemetry: what a device says --------------------------------------------
#
# A device publishes to `telemetry_topic` and an IoT rule lands the message in
# the telemetry bucket under `telemetry_prefix`. Nothing on the device writes a
# file (design section 7), so these shapes are the whole of what a replay leaves
# behind.
#
# **The topic's segments are load-bearing.** The rule's S3 key template addresses
# the run and the thing by position -- `${topic(3)}` and `${topic(4)}` -- which
# makes the layout below a contract with a Terraform file rather than a string
# this module is free to rearrange. It is stated here for the reason the whole
# module exists, and the cost of getting it wrong is the usual one: the publish
# succeeds and the object lands under a prefix nothing reads.
#
# **Frames are batched and the summary is not.** A replay is several hundred
# frames of a few hundred bytes each, and one object per frame would be one S3
# GET per frame for a reader that wants all of them. One object per hundred is
# the same bytes in a fiftieth of the requests, well inside IoT Core's 128 KB
# message ceiling. The summary is a single record because there is one.

TELEMETRY_ROOT: Final = f"{PROJECT}/fleet"

# What the IoT rule subscribes to: every run, every thing. The two wildcards are
# the two segments below.
TELEMETRY_TOPIC_FILTER: Final = f"{TELEMETRY_ROOT}/+/+"

# Frames per published message. Sized so a full message stays well under IoT
# Core's ceiling at the detection counts BDD100K produces, and so a 500-frame
# replay is five objects rather than five hundred.
TELEMETRY_BATCH: Final = 100


def telemetry_topic_prefix(run_id: RunId) -> str:
    """Everything in a device's topic except which device it is.

    Split out because the last segment is not always a name this module can
    check. A Greengrass recipe names the publishing device as `{iot:thingName}`,
    which the nucleus substitutes on the device -- so the recipe needs this
    prefix and supplies its own final segment, while `telemetry_topic` below is
    what every reader and every test uses. Building the topic in the recipe
    instead would be a second spelling of the layout the IoT rule depends on.
    """
    return f"{TELEMETRY_ROOT}/{parse_run_id(run_id)}/"


def telemetry_topic(run_id: RunId, thing: str) -> str:
    """Where one device publishes one run's replay.

    The thing name is the last segment so that the rule can name it in the object
    it writes, which is what keeps two devices' records apart in a prefix neither
    of them partitions. Checked as a key component for exactly that reason: a
    slash in it would move the object a level down and change which segment the
    rule reads the run out of.
    """
    return f"{telemetry_topic_prefix(run_id)}{_token('thing', thing)}"


class TelemetryKind(StrEnum):
    """Which of the two records a telemetry object holds.

    A discriminator rather than two topics, because two topics would be two rules
    and two prefixes for records a reader always wants together: a batch of
    frames means nothing without the summary saying how many there should have
    been.
    """

    FRAMES = "frames"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One replayed frame, as an entry inside a `TelemetryKind.FRAMES` record.

    Three facts and deliberately not the boxes. `inference_ms` is the measurement
    design section 4.3 asks for and the only one that must come off the device --
    p95 is computed from these, and mean hides the stalls that matter. `scores`
    is the confidence of each detection, which is what a later distribution
    comparison reads; the coordinates are omitted because nothing downstream of
    the fleet matches a device's boxes against ground truth, and the offline pass
    already wrote every box this model draws.

    `image_id` is what ties the record to the ranking that chose the frame, which
    is `replay_manifest_key`'s whole purpose.

    A frame the model found nothing in carries an empty `scores` and is not an
    absent row. It is a real observation -- the champion's blind spots are what
    `selection.score` ranks highest -- and dropping it would make a replay's
    frame count disagree with its summary for a reason nothing recorded.
    """

    image_id: ImageId
    inference_ms: float
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)
        if self.inference_ms <= 0.0:
            raise ValueError(
                f"a frame that took {self.inference_ms} ms is not a frame that was inferred"
            )
        outside = [score for score in self.scores if not 0.0 <= score <= 1.0]
        if outside:
            raise ValueError(f"a confidence outside [0, 1] is not a confidence: {outside[:5]}")


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """One device's whole replay of one model version, frames and summary joined.

    What the canary gate reads. It is a reduction of the objects under
    `telemetry_prefix` rather than a document anything writes, which is why the
    two counts are separate fields: `replayed` is what the device said it put
    through the model and `latencies_ms` is what actually arrived. A replay whose
    summary claims 500 frames and whose batches carry 300 is a replay that lost
    messages, and a report that stored one number could not say so.

    `artifact_sha256` is the digest the device computed over the file it loaded,
    not the one Greengrass verified on download. The service checks its own copy
    against its own recipe, which is a closed loop; this is the independent half,
    and comparing it against `ModelManifest.artifact_sha256` is what says the
    bytes that ran are the bytes the gates were reported over.

    `starts` is how many times the component has started for this version, read
    off a counter in its work directory. One is a clean install. More is
    Greengrass restarting something that exited, which is the failure design
    section 4.5 calls "hot-swapping did not crash the agent" and the reason the
    check is worth making at all.
    """

    version: ModelVersion
    thing: str
    artifact_sha256: str
    cold_start_ms: float
    starts: int
    replayed: int
    latencies_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        parse_model_version(self.version)
        if not _SHA256.match(self.artifact_sha256):
            raise ValueError(f"not a lowercase hex sha256 digest: {self.artifact_sha256!r}")
        if self.cold_start_ms <= 0.0:
            raise ValueError(f"a cold start of {self.cold_start_ms} ms was never measured")
        if self.starts < 1:
            raise ValueError(
                f"a component that started {self.starts} times published nothing, so this report "
                f"could not exist"
            )
        if self.replayed < 0:
            raise ValueError(f"a replay cannot have put {self.replayed} frames through a model")

    @property
    def reported(self) -> int:
        """Frames that actually arrived, against `replayed` frames claimed."""
        return len(self.latencies_ms)

    @property
    def complete(self) -> bool:
        """Every frame the summary claimed is a frame a batch carried.

        Equality rather than a floor, in both directions. Fewer is lost messages;
        more is two replays of one version landing in one prefix, which makes
        every number below an average over two runs of the component.
        """
        return self.reported == self.replayed

    @property
    def p95_ms(self) -> float:
        """Nearest-rank p95, which is a latency this device actually recorded.

        Not an interpolation between two of them. At a few hundred frames the two
        differ by less than the measurement's own spread, and a reported figure
        that appears in no sample is one nobody can go back and find.
        """
        if not self.latencies_ms:
            raise ValueError(f"{self.thing} reported no frame, so it has no p95")
        ordered = sorted(self.latencies_ms)
        rank = -(-len(ordered) * 95 // 100)  # ceil, without importing math for it
        return ordered[rank - 1]

    @property
    def throughput_fps(self) -> float:
        """Frames a second of inference, which is what the canary compares.

        Off the mean rather than the p95: throughput is how much work the device
        got through, so the slow frames should count exactly as much as they
        cost. p95 is the separate question of how bad the worst of them was.
        """
        if not self.latencies_ms:
            raise ValueError(f"{self.thing} reported no frame, so it has no throughput")
        return 1000.0 * len(self.latencies_ms) / sum(self.latencies_ms)

"""Storage and identifier conventions, fixed before the first table is written.

Every component builds its S3 keys through this module rather than formatting
its own strings. A key spelled in two places drifts in one of them, and the
failure is silent in the worst way: the writer succeeds, the reader finds
nothing, and the cycle reports having done less work than it did.

Three rules the layout follows, recorded here because they decide every
question about where something new belongs.

**Key a thing by exactly what its content is a function of.** Key by less and
two different things collide at one path; key by more and identical bytes are
stored once per surplus dimension. Shards are a function of the raw data and
the partition version, so they carry ``partition_version`` and deliberately not
``run_id`` -- two runs sharing a partition share the shards, which is safe
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
# the five a cycle trains, and a path component capped at one digit by
# `SEED_DIGITS`; a draw seed appears in no key, is fixed once per partition
# version, and is six digits wider than that cap allows.
PartitionSeed = NewType("PartitionSeed", int)

RunId = NewType("RunId", str)

# `<run_id>-c<cycle>`. Built out of the run-id machinery, so the format itself is
# defined under "Model versions" below rather than here.
ModelVersion = NewType("ModelVersion", str)

# The three version stamps a paired comparison assumes are held constant
# (design section 5). They are integers rather than free-form strings because
# the only operation ever performed on them is equality against the champion's.
ClassSetVersion = NewType("ClassSetVersion", int)
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
SHARD_DIGITS: Final = 5
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
# Defined ahead of runs because a run carries one: the rule it buys labels by is
# part of what the run *is*, not a per-cycle choice.


class Selector(StrEnum):
    """Which rule a run ranks the pool by, recorded on the run registration.

    All three members exist before any of them is needed. "Uncertain frames
    teach the most" is the hypothesis the project sets out to test rather than
    a premise it may assume, and what tests it is a run buying by a different
    rule over the same bootstrap and the same eval. Naming the alternatives
    here keeps selection a swappable function rather than one rule with a
    second added alongside it later, which is the arrangement under which two
    runs come to differ in more than the selector.

    `UNCERTAINTY` is the rule the loop runs by, and the only one a cycle uses
    by default.

    `RANDOM` needs the remaining pool and a seed and no inference at all, which
    makes it both the smoke test for the ranking-to-purchase path before a
    champion exists to score with, and the control arm of the deferred
    label-efficiency comparison (design section 8).

    `CERTAINTY` inverts the ranking, buying what the champion is most sure of.
    Those frames carry the least new information, so a cycle run this way
    should gain close to nothing; one that gains as much as a real cycle says
    the ranking is not what is doing the work.
    """

    UNCERTAINTY = "uncertainty"
    RANDOM = "random"
    CERTAINTY = "certainty"


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
# **Deliberately not encoded: the version trio.** A change to `class_set_version`,
# `recipe_version` or `partition_version` forces a new run (design section 5),
# but the id only has to be *new*, not to describe the change. The trio lives in
# the run registration and in every model manifest; putting it in the id too
# would give two sources of truth that drift.

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
    artifacts. The version trio is recorded at the run level, not only per
    model, because it is the *precondition* of the run's whole comparison --
    a model manifest disagreeing with its run's registration is a bug worth
    detecting rather than a fact worth storing twice.

    `selector` is here for the trio's reason and not with the trio: it is a
    precondition of what the run's numbers mean, so a bucket listing has to be
    able to say which rule bought the labels -- but it is deliberately outside
    `supersedes`, for the reason that method gives.

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
    class_set_version: ClassSetVersion
    recipe_version: RecipeVersion
    selector: Selector
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
        and forgets in the other two places.

        `selector` is not one of these fields, and the omission is load-bearing.
        The trio fixes the data universe and the metric; the selector changes
        only which images inside that universe get bought. A label-efficiency
        comparison is *paired* -- both arms start from the same bootstrap
        champion and score against the same frozen eval -- so a selector change
        forcing a re-baseline would discard the shared baseline that makes the
        arms comparable at all, leaving them different in two respects instead
        of one. Adding it here resembles tightening the rule and instead voids
        the comparison.
        """
        return (
            self.partition_version,
            self.class_set_version,
            self.recipe_version,
        ) != (
            other.partition_version,
            other.class_set_version,
            other.recipe_version,
        )


# --- Model versions -----------------------------------------------------------
#
# `<run_id>-c<cycle>`, as in `20260812t143355z-v0-skeleton-c003`.
#
# **The name is an address; the manifest is the facts.** The version trio, the
# git commit, the artifact digests, the gate results -- every one of them is read
# by opening `ModelManifest`, and none of them is needed to find it. So none of
# them appear here. Encoding one would create a second place for the same fact to
# be stated, and this is the copy that cannot be corrected: the version is a path
# component and the artifacts bucket is write-once, so a name that disagrees with
# its manifest disagrees permanently. Same rule that keeps the trio out of
# `run_id`, applied to the same kind of field.
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
# A cycle trains one challenger over five seeds (design section 4.2), and the
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


# --- Image tags ---------------------------------------------------------------
#
# The three BDD100K attributes the eval slices and the selection condition cap
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

    That spread is why the regression gate carries a slice floor rather than
    gating every member it can name: `foggy` is not measurable at any eval size
    drawn from this archive, so a slice too small to discriminate is reported and
    never vetoes (design section 4.4).
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
    under its run -- see `purchase_shards_prefix` -- and never by reassigning a
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


# `pool` is read one object at a time -- by the fleet, and by the selection pass
# scoring it -- and `reserve` is not read at all, so neither earns a WebDataset
# shard. What a cycle buys out of the pool is sharded under its run instead.
SHARDED_COHORTS: Final = frozenset({Cohort.BOOTSTRAP, Cohort.EVAL})

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
    # `eval` at 5,000 is what puts twelve slices over the 300-image gating floor;
    # `bootstrap` at 8,000 leaves the model headroom for a 1,000-label cycle to
    # move the metric; `pool` at 62,000 makes one cycle 1.6% of what was scored,
    # which is the selectivity a ranking needs to diverge from a random draw.
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
    """Filenames under a seed's model prefix."""

    ONNX = "model.onnx"
    TORCH = "model.pt"
    SHA256 = "model.sha256"


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
    ORACLE_LABELS = "oracle_labels"
    LABEL_BUDGET = "label_budget"
    FLEET_CONFIG = "fleet_config"
    AUDIT_LOG = "audit_log"
    RUN_LOCKS = "run_locks"


def table_name(table: Table) -> str:
    """No account ID, unlike buckets: table names are per account and region."""
    return f"{PROJECT}-{table.value}"


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
# `raw/labels/` carries the ground truth for all 80,000 images, where
# `oracle_labels` is loaded with `pool` alone. A training job that can GET these
# files bypasses the oracle entirely and reaches every cohort, so the training
# role is denied this prefix in its own policy and again in the bucket policy. An
# explicit deny beats any allow, including one granted later somewhere else.

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

    Every Phase 1 question is a query over this: cohort sizing, eval
    stratification, per-slice counts. It has to exist before the shards, which
    cannot be written until the partitioner has assigned cohorts.

    `weather`, `scene` and `timeofday` are enums because their vocabularies were
    measured over all 80,000 images before being written down. Eval
    stratification and the selection condition cap are predicates over these
    three columns, and a misspelled tag is the one kind of wrong predicate that
    does not fail: it matches nothing, so the slice empties or the cap never
    binds, and both read downstream as a clean pass. The parquet column stays
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
    ways would make every query slower and buy nothing. The shards under this
    same partition prefix do use `cohort=` as a prefix, because there the pruning
    is the point.
    """
    name = _padded("part", part, PART_DIGITS)
    return f"{assignments_prefix(partition_version)}part-{name}.parquet"


def shards_prefix(partition_version: PartitionVersion, cohort: Cohort) -> str:
    """Sharded by cohort, so a bulk read is a prefix selection rather than a
    filter over everything.

    Only the two cohorts labeled at partition time are sharded. `eval` is frozen
    after Phase 1 by denying writes under its prefix -- which is only expressible
    because cohort is a prefix.
    """
    if cohort not in SHARDED_COHORTS:
        raise ValueError(f"{cohort.value} is never sharded")
    return f"{partition_prefix(partition_version)}shards/cohort={cohort.value}/"


def shard_key(partition_version: PartitionVersion, cohort: Cohort, index: int) -> str:
    """One ~200 MB WebDataset shard.

    Two representations of the same images, because there are two access
    patterns: random single-object GET for the fleet and for scoring the pool,
    bulk sequential reads for five seeds of training every cycle. Only the
    labeled images are duplicated -- 13,000 at partition time, growing by what
    each cycle buys -- so the second copy is about 1.5 GB, a few cents a month.
    """
    name = _padded("index", index, SHARD_DIGITS)
    return f"{shards_prefix(partition_version, cohort)}shard-{name}.tar"


PURCHASES_PREFIX: Final = "derived/purchases/"


def purchase_shards_prefix(run_id: RunId, cycle: Cycle) -> str:
    """One cycle's bought labels, sharded for the same bulk read the cohort
    shards serve.

    Keyed by run and cycle rather than by partition version, because which images
    a cycle bought is a fact about that run's selector and budget: two runs over
    one partition buy different images, and the label-efficiency A/B is exactly
    the case where they must not collide. A cycle's training set is therefore the
    bootstrap shards plus every purchase prefix from cycle 1 up to it, which
    states the cumulative labeled set as a key range.
    """
    return f"{PURCHASES_PREFIX}{cycle_prefix(run_id, cycle)}"


def purchase_shard_key(run_id: RunId, cycle: Cycle, index: int) -> str:
    """One shard of a single cycle's purchase, padded to sort with the rest."""
    name = _padded("index", index, SHARD_DIGITS)
    return f"{purchase_shards_prefix(run_id, cycle)}shard-{name}.tar"


# --- Artifacts bucket ---------------------------------------------------------
#
# Write-once. Nothing under a cycle prefix is ever rewritten and a re-run is a
# new `run_id`, which is what makes gate reports and model manifests audit
# evidence rather than current state.


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

    The version trio is repeated from the run registration on purpose. It is not
    a second source of truth -- `RunRegistration` remains authoritative -- but a
    manifest that disagrees with its run is a bug the registration step can only
    detect if the manifest states its own view.

    Run and cycle are *not* repeated, and the difference is the point: the trio
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

    `artifact_sha256` is the digest of each seed's `model.onnx`, all five, not
    only the deployed one -- the matched-seed saving in design section 7 depends
    on the other four existing and being identifiable. The device agent verifies
    the digest before loading (design section 6), so a truncated download becomes
    a rejection instead of a model that silently returns nonsense.

    Serialization is deliberately absent. It lands with Phase 3's manifest
    emission, next to the registration step that reads it back.
    """

    version: ModelVersion
    created_at: datetime
    git_commit: str
    partition_version: PartitionVersion
    class_set_version: ClassSetVersion
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
        the version encodes its run. A trio field disagreeing means the training
        job ran a configuration the run never declared, which is exactly what
        this class restates the trio to expose. Copying the registration's values
        in at construction would make a disagreement unrepresentable, and would
        do it by recording the run's intent in place of the job's behaviour --
        silencing the witness rather than believing it.
        """
        return tuple(
            name
            for name, claimed, declared in (
                ("run_id", self.run_id, run.run_id),
                ("partition_version", self.partition_version, run.partition_version),
                ("class_set_version", self.class_set_version, run.class_set_version),
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


def model_artifact_key(version: ModelVersion, seed: Seed, artifact: ModelArtifact) -> str:
    """Per seed, because all five champion artifacts are retained.

    Seed 1 is the one that ships, by convention. The other four are what the
    matched-seed cost saving in design section 7 depends on -- keeping only seed
    1 quietly removes it.
    """
    return f"{model_prefix(version)}seed={_padded('seed', seed, SEED_DIGITS)}/{artifact.value}"


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
    separate objects would be unusable. Same access-pattern reasoning that
    produces shards.
    """
    return f"{eval_prefix(version)}seed={_padded('seed', seed, SEED_DIGITS)}/matches.npz"


def gate_report_key(run_id: RunId, cycle: Cycle, suffix: str = "json") -> str:
    """Per cycle, not per model: the report covers the comparison, not one side.

    Too fat to live in the audit log -- the DynamoDB item cap is 400 KB and it
    would be a blob in a database. The log holds the event and a `detail_uri`
    pointing here.
    """
    return f"{cycle_prefix(run_id, cycle)}gates/report.{_token('suffix', suffix)}"


# --- Telemetry bucket ---------------------------------------------------------

ATHENA_RESULTS_PREFIX: Final = "athena-results/"


def telemetry_prefix(run_id: RunId, day: date) -> str:
    """Partitioned by run then day, both filtered on constantly.

    Day granularity, not hour: at this volume hourly partitions produce small
    files and Athena gets slower, not faster.
    """
    return f"fleet/{run_prefix(run_id)}dt={day.isoformat()}/"

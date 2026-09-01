"""The seeded draw, and the two files it writes.

Each of the 80,000 images gets exactly one cohort. The draw is uniform within a
split and carries no predicate over `weather`, `scene` or `timeofday`: the
partition decides who is trainable, not what the model sees when, so there is no
carve-out to keep disjoint and the assertion is a row count plus a uniqueness
check.

**A ticket per image, not a shuffle of the split.** `sha256(seed:image_id)`
orders each split, and the first `n` tickets are the cohort. A shuffle is a
function of the order its input arrived in, and this input is a directory walk:
a re-ingest that lists files differently would produce a different partition
under the same seed with nothing to announce it. A ticket depends on the seed and
the image ID alone. It is also independent of `random`, whose sampling algorithm
CPython does not promise across versions, against a partition that has to stay
re-derivable for the life of the project.

Both files land in the staged tree at the keys `conventions` builds --
`assignments/part-00000.parquet` and the `_partition.json` recording the seed.
"""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    MANIFEST_PREFIX,
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionSeed,
    PartitionSpec,
    PartitionVersion,
    Split,
    assignments_key,
    columns,
    parse_image_id,
    partition_manifest_key,
    partition_spec,
)

log = logging.getLogger(__name__)

SCHEMA: Final = pa.schema(
    [
        ("image_id", pa.string()),
        ("cohort", pa.string()),
    ]
)

# zstd for `manifest.SCHEMA`'s reason: written once, read whole.
_COMPRESSION: Final = "zstd"

# The three tag columns, read for the composition log and for nothing else. No
# partition decision is a predicate over them.
_TAG_COLUMNS: Final = ("weather", "scene", "timeofday")

_MANIFEST_COLUMNS: Final = ("image_id", "split", *_TAG_COLUMNS)

# The two cohorts whose composition is worth a log line. `eval` because
# proportional sampling is the claim overall mAP rests on, and `pool` because its
# tag shares are the reference line every cycle's purchase mix is read against.
_REPORTED_COHORTS: Final = (Cohort.EVAL, Cohort.POOL)

# Recorded in `_partition.json`, and compared when a partition already in the
# bucket is checked against this code: a seed reproduces a draw only for someone
# who knows what was done with it, so a changed rule under an unchanged version
# is the same failure as a changed seed.
DRAW_RULE: Final = "per split, ascending sha256(seed:image_id), quotas in cohort order"


def read_manifest(stage_dir: Path) -> pa.Table:
    """Every manifest part, identity and tags only.

    `box_areas` is most of the file's bytes and nothing here is a predicate over
    it, so the column stays on disk.
    """
    parts = sorted((stage_dir / MANIFEST_PREFIX).glob("part-*.parquet"))
    if not parts:
        raise ValueError(f"no manifest parquet under {stage_dir / MANIFEST_PREFIX}")
    return pa.concat_tables(
        [pq.read_table(part, columns=list(_MANIFEST_COLUMNS)) for part in parts]
    )


def read_assignments(
    stage_dir: Path, partition_version: PartitionVersion
) -> tuple[AssignmentRow, ...]:
    """The draw read back out of the staged tree.

    The label copy runs as its own step, after the shell has staged the documents
    the draw says to fetch, so it reads the assignments this module wrote rather
    than being handed them. Sorted by `image_id` for `assign`'s reason, which
    makes the round trip through parquet an identity on ordering as well as on
    content.
    """
    path = stage_dir / assignments_key(partition_version)
    if not path.is_file():
        raise ValueError(f"no assignments parquet at {path}")

    table = pq.read_table(path, columns=list(columns(AssignmentRow)))
    rows = [
        AssignmentRow(image_id=parse_image_id(value), cohort=Cohort(cohort))
        for value, cohort in zip(
            table.column("image_id").to_pylist(),
            table.column("cohort").to_pylist(),
            strict=True,
        )
    ]
    return tuple(sorted(rows, key=lambda row: row.image_id))


def ticket(seed: PartitionSeed, image_id: ImageId) -> str:
    """One image's place in the draw, as lowercase hex.

    Hex rather than an int because ordering is the only operation performed on it
    and fixed-width hex sorts identically.
    """
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def _split_of(manifest: pa.Table) -> dict[ImageId, Split]:
    """The manifest as image ID to split, validating both on the way through.

    `Split` has no `TEST` member, so a manifest carrying the withheld split fails
    here rather than assigning 20,000 images to a cohort. That is the third of the
    three places the exclusion is enforced, after the extraction filter and the
    verification suite.
    """
    found: dict[ImageId, Split] = {}
    for value, split in zip(
        manifest.column("image_id").to_pylist(),
        manifest.column("split").to_pylist(),
        strict=True,
    ):
        image_id = parse_image_id(value)
        if image_id in found:
            raise ValueError(f"{image_id} appears twice in the manifest")
        try:
            found[image_id] = Split(split)
        except ValueError:
            listed = ", ".join(member.value for member in Split)
            raise ValueError(
                f"{image_id}: manifest split is {split!r}, which is not one of: {listed}"
            ) from None
    return found


def _draw(
    image_ids: Sequence[ImageId],
    seed: PartitionSeed,
    quotas: Sequence[tuple[Cohort, int]],
) -> list[AssignmentRow]:
    """Hand one split's images to its cohorts, in quota order.

    Exactly-once is a property of walking one ordering once rather than a rule
    applied afterwards: an image has one ticket, the ordering has one position for
    it, and the quotas consume that ordering without overlap or remainder. The
    quota total is checked against the split first, since a shortfall would
    otherwise leave the tail of the ordering silently unassigned.
    """
    total = sum(quota for _, quota in quotas)
    if total != len(image_ids):
        named = ", ".join(f"{cohort.value} {quota:,}" for cohort, quota in quotas)
        raise ValueError(
            f"quotas total {total:,} against {len(image_ids):,} images in the split ({named})"
        )

    ordered = sorted(image_ids, key=lambda image_id: (ticket(seed, image_id), image_id))

    rows: list[AssignmentRow] = []
    start = 0
    for cohort, quota in quotas:
        rows.extend(
            AssignmentRow(image_id=image_id, cohort=cohort)
            for image_id in ordered[start : start + quota]
        )
        start += quota
    return rows


def assign(manifest: pa.Table, spec: PartitionSpec) -> tuple[AssignmentRow, ...]:
    """One cohort per manifest row, drawn per split.

    Per split rather than over the 80,000 at once, because `COHORT_SPLIT` decides
    where each cohort draws from and a draw confined to one split cannot take an
    image from the wrong side of the rule.

    Sorted by `image_id` on the way out, so two runs of one version produce
    byte-identical parquet and "this is the same partition" is checkable rather
    than asserted.
    """
    split_of = _split_of(manifest)

    rows: list[AssignmentRow] = []
    for split in Split:
        in_split = [image_id for image_id, source in split_of.items() if source is split]
        rows.extend(_draw(in_split, spec.seed, spec.quotas(split)))

    return tuple(sorted(rows, key=lambda row: row.image_id))


def check(rows: Sequence[AssignmentRow], manifest: pa.Table, spec: PartitionSpec) -> None:
    """Disjoint and complete, asserted before anything is written.

    `_draw` produces this by construction, so nothing here should ever fire. It
    runs anyway: `cohort=` is valid as a storage prefix only while the property
    holds, and by the time an assignments parquet is in the bucket every later
    artifact is keyed against it (design section 5). This is the primary leakage
    insurance, and it costs a second over 80,000 rows.
    """
    split_of = _split_of(manifest)

    if len(rows) != len(split_of):
        raise ValueError(f"{len(rows):,} assignments over {len(split_of):,} manifest images")

    assigned = {row.image_id for row in rows}
    if len(assigned) != len(rows):
        raise ValueError(f"{len(rows) - len(assigned):,} images are assigned more than once")

    unassigned = sorted(set(split_of) - assigned)
    if unassigned:
        raise ValueError(
            f"{len(unassigned):,} manifest images have no cohort, first few: {unassigned[:5]}"
        )

    # Before the sizes, because this is the leakage statement and a partition that
    # fails it should be reported as failing it rather than as an arithmetic
    # disagreement that happens to accompany it.
    for row in rows:
        source = COHORT_SPLIT[row.cohort]
        if split_of[row.image_id] is not source:
            raise ValueError(
                f"{row.image_id} is a {split_of[row.image_id].value} image in "
                f"{row.cohort.value}, which draws from {source.value}"
            )

    counts = Counter(row.cohort for row in rows)
    wrong = {
        cohort.value: f"{counts[cohort]:,} drawn against {size:,} specified"
        for cohort, size in spec.sizes.items()
        if counts[cohort] != size
    }
    if wrong:
        raise ValueError(f"cohort sizes disagree with the partition spec: {wrong}")


def write_parquet(rows: Sequence[AssignmentRow], path: Path) -> None:
    """Columns from the schema of record, for `manifest.write_parquet`'s reason."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: [getattr(row, name) for row in rows] for name in columns(AssignmentRow)}
    pq.write_table(pa.table(data, schema=SCHEMA), path, compression=_COMPRESSION)


def document(partition_version: PartitionVersion, spec: PartitionSpec) -> dict[str, Any]:
    """A partition's description of itself, everything but when it was drawn.

    Split out from the writer so that the comparison below reads the fields the
    writer writes. Two spellings of this shape would let a recorded partition and
    a spec agree on the fields someone remembered to compare.
    """
    return {
        "partition_version": int(partition_version),
        "seed": int(spec.seed),
        "draw": DRAW_RULE,
        "cohorts": {
            cohort.value: {"images": spec.sizes[cohort], "split": COHORT_SPLIT[cohort].value}
            for cohort in COHORT_SPLIT
        },
    }


def write_document(partition_version: PartitionVersion, spec: PartitionSpec, path: Path) -> None:
    """Record the seed beside the assignments it produced.

    `draw` states the rule in the file rather than only in this module, because
    the seed alone reproduces the partition only for someone who knows what was
    done with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document(partition_version, spec) | {"drawn_at": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def disagreements(
    recorded: Mapping[str, Any], partition_version: PartitionVersion, spec: PartitionSpec
) -> tuple[str, ...]:
    """Fields where a partition already in the bucket contradicts this code.

    Empty means a re-run would reproduce what is there, which is the only case in
    which overwriting it is safe. `PartitionSpec` says to add a version rather
    than edit one; this is what makes that enforceable instead of remembered,
    because an edited seed produces a partition that is valid, differently drawn,
    and keyed identically to the one every existing run was measured against.

    Field names rather than a bool, for `ModelManifest.disagreements`' reason: a
    refusal is only actionable with the field that caused it.
    """
    expected = document(partition_version, spec)
    return tuple(name for name, value in expected.items() if recorded.get(name) != value)


def composition(
    manifest: pa.Table, rows: Sequence[AssignmentRow]
) -> dict[Cohort, dict[str, Counter[str]]]:
    """Tag counts per reported cohort, per axis.

    Returned and logged rather than written. It is a join of the manifest onto the
    assignments, both of which are in the bucket, so a third file would hold the
    same fact a query answers.
    """
    cohort_of = {row.image_id: row.cohort for row in rows}
    image_ids = manifest.column("image_id").to_pylist()

    counted: dict[Cohort, dict[str, Counter[str]]] = {
        cohort: {column: Counter[str]() for column in _TAG_COLUMNS} for cohort in _REPORTED_COHORTS
    }
    for column in _TAG_COLUMNS:
        for image_id, value in zip(image_ids, manifest.column(column).to_pylist(), strict=True):
            cohort = cohort_of[ImageId(image_id)]
            if cohort in counted:
                counted[cohort][column][value] += 1
    return counted


def _log_composition(counted: Mapping[Cohort, Mapping[str, Counter[str]]]) -> None:
    for cohort, axes in counted.items():
        for column, counts in axes.items():
            listed = ", ".join(f"{value} {count:,}" for value, count in counts.most_common())
            log.info("%s %s: %s", cohort.value, column, listed)


def write(stage_dir: Path, partition_version: PartitionVersion) -> tuple[AssignmentRow, ...]:
    """Draw one version's partition into the staged tree."""
    spec = partition_spec(partition_version)
    manifest = read_manifest(stage_dir)
    log.info(
        "partition version %d over %d manifest rows, seed %d",
        partition_version,
        manifest.num_rows,
        spec.seed,
    )

    rows = assign(manifest, spec)
    check(rows, manifest, spec)

    write_parquet(rows, stage_dir / assignments_key(partition_version))
    write_document(partition_version, spec, stage_dir / partition_manifest_key(partition_version))

    for cohort, size in spec.sizes.items():
        log.info("%s: %d images from %s", cohort.value, size, COHORT_SPLIT[cohort].value)

    _log_composition(composition(manifest, rows))
    log.info("wrote %s", assignments_key(partition_version))
    return rows

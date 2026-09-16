"""The boxes for the two cohorts the partition labels.

`bootstrap` and `eval` are the cohorts that own labels at partition time, so
their boxes are copied out of `raw/labels/` and filed under the partition prefix
the draw already writes. `pool` labels stay withheld and reach a run only through
the oracle; `reserve` has none anywhere.

**The copy is not a convenience.** `bootstrap` and `pool` are both drawn from
`train`, so their label documents are siblings in one directory and cohort is a
column rather than a path component. There is no prefix that means "the bootstrap
labels", which makes `raw/labels/scalabel/train/` ungrantable to training: the
same read that gives it the 8,000 it owns would give it the 62,000 it is supposed
to buy. Writing the 8,000 to their own prefix is what makes the grant
expressible.

`eval` does not share that problem -- `val` holds only `eval` and `reserve`, and
neither is trainable -- and is copied anyway. Both cohorts are then one mechanism
with one set of guarantees, against a second path that would exist to save a file.

**No key this module builds addresses a `pool` label.** `label_key` takes a
cohort, refuses anything outside `LABELED_COHORTS`, and derives the split from
`COHORT_SPLIT`, so the refusal happens while the caller still holds a cohort and
before a path exists. That is `oracle.labels`' arrangement for the same reason:
the gate is not a check performed before the work, it is the thing that produces
the path the work is done on.

**Boxes are encoded exactly as a purchase encodes them.** `oracle.labels`
serializes the archive's own corners, and training reads a bootstrap label and a
bought one out of the same column with the same decoder. A second encoding here
would be a second way to be a hair off the geometry BDD published.
"""

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    LABELED_COHORTS,
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionVersion,
    cohort_labels_key,
    raw_label_key,
)
from edge_ml_flywheel.ingest.labels import Box, parse_label
from edge_ml_flywheel.oracle import labels as sold
from edge_ml_flywheel.oracle.labels import encode_boxes

log = logging.getLogger(__name__)

# `boxes` is the compact JSON `oracle.labels.encode_boxes` produces, so a reader
# decodes a bootstrap label and a bought one with one function. A list column of
# structs would be the better parquet and the worse arrangement: it would make
# the two label sources different shapes, and the whole point of a cumulative
# labeled set is that they are not.
#
# Taken from `oracle.labels` rather than restated, because that is the same
# statement about the column names: two identical schemas are two places for one
# of them to be renamed. This module already takes the encoder from there, so the
# format arrives from one place entire.
SCHEMA: Final = sold.SCHEMA

# snappy for `assign.write_parquet`'s reason, and this is the file that found it:
# `prepare` reads the bootstrap labels in a Lambda, where pyarrow comes from the
# managed AWSSDKPandas layer with zstd trimmed out of the build.
_COMPRESSION: Final = sold.COMPRESSION

# How many offending IDs a refusal lists before summarizing, for
# `oracle.cohorts._REPORTED`'s reason.
_REPORTED: Final = 5


@dataclass(frozen=True, slots=True)
class CohortLabel:
    """One image and all of its boxes, as a row of a cohort's label file.

    The same pair `oracle.labels.SoldLabel` carries, without the charge. Not that
    type reused, because a sold label is the record of a transaction and these
    were never bought: sharing the name would make a free label look like an
    unbilled one in every log line that prints it.
    """

    image_id: ImageId
    boxes: tuple[Box, ...]


def label_key(cohort: Cohort, image_id: ImageId) -> str:
    """The S3 key of one labeled cohort image's raw label document.

    Total on what it returns and partial on what it accepts, for
    `oracle.labels.label_key`'s reason: there is no cohort argument for which
    this produces a key to a label the partition withheld.
    """
    if cohort not in LABELED_COHORTS:
        listed = ", ".join(sorted(member.value for member in LABELED_COHORTS))
        raise ValueError(
            f"{cohort.value} has no labels of its own, so no key addresses them ({listed} do)"
        )
    return raw_label_key(image_id, COHORT_SPLIT[cohort])


def image_ids(rows: Iterable[AssignmentRow], cohort: Cohort) -> tuple[ImageId, ...]:
    """One cohort's images, sorted.

    Sorted here rather than at the writer so that the key list, the read order
    and the parquet all follow one ordering, which is what makes a re-run
    byte-identical.
    """
    return tuple(sorted(row.image_id for row in rows if row.cohort is cohort))


def keys(rows: Sequence[AssignmentRow]) -> tuple[str, ...]:
    """Every raw label key this step reads, for the shell to copy.

    The buildspec stages exactly this list rather than the label tree, which is
    the difference between a job that has 13,000 label documents on local disk
    and one that has 80,000. It is also the audit artifact: the copy fetched what
    the package named, and the package cannot name a `pool` label.
    """
    return tuple(
        label_key(cohort, image_id)
        for cohort in sorted(LABELED_COHORTS)
        for image_id in image_ids(rows, cohort)
    )


def read(stage_dir: Path, cohort: Cohort, image_id: ImageId) -> CohortLabel:
    """One label out of a tree laid out at its S3 keys.

    `parse_label` does the archive-shaped validation, so a document that is not
    what this code was written against fails here rather than becoming a label
    with no boxes -- which would read downstream as an image containing nothing.
    """
    path = stage_dir / label_key(cohort, image_id)
    parsed = parse_label(json.loads(path.read_text(encoding="utf-8")), image_id)
    return CohortLabel(image_id=image_id, boxes=parsed.boxes)


def collect(stage_dir: Path, rows: Sequence[AssignmentRow], cohort: Cohort) -> list[CohortLabel]:
    """Every label for one cohort, in image ID order."""
    return [read(stage_dir, cohort, image_id) for image_id in image_ids(rows, cohort)]


def check(
    labels: Sequence[CohortLabel],
    rows: Sequence[AssignmentRow],
    cohort: Cohort,
) -> None:
    """Exactly this cohort's images, asserted before the file is written.

    `collect` produces this by construction, so nothing here should ever fire. It
    runs anyway, and it is the check the widened label grant is worth: this role
    can now read all 80,000 label documents, and this is the statement that only
    two cohorts' worth reached an output file.

    Set equality rather than a count. A count would pass a file holding 8,000
    rows of which one was a `pool` image and one a `bootstrap` image missing --
    which is precisely the shape a bug in the key list would produce.
    """
    expected = set(image_ids(rows, cohort))
    written = {label.image_id for label in labels}

    duplicated = sorted(
        image_id
        for image_id, times in Counter(label.image_id for label in labels).items()
        if times > 1
    )
    if duplicated:
        raise ValueError(
            f"{cohort.value}: {len(duplicated):,} image(s) appear more than once: "
            f"{duplicated[:_REPORTED]}"
        )

    # Reported before the shortfall, because a stray image is a leak and a missing
    # one is an incomplete copy, and a failure that is both should be named as the
    # first of the two.
    stray = sorted(written - expected)
    if stray:
        raise ValueError(
            f"{cohort.value}: {len(stray):,} image(s) are not in this cohort and must not be "
            f"written to its labels: {stray[:_REPORTED]}"
        )

    missing = sorted(expected - written)
    if missing:
        raise ValueError(
            f"{cohort.value}: {len(missing):,} of {len(expected):,} images have no label, "
            f"first few: {missing[:_REPORTED]}"
        )


def write_parquet(labels: Sequence[CohortLabel], path: Path) -> None:
    """Columns from `SCHEMA`, for `assign.write_parquet`'s reason."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_id": [label.image_id for label in labels],
        "boxes": [encode_boxes(label.boxes) for label in labels],
    }
    pq.write_table(pa.table(data, schema=SCHEMA), path, compression=_COMPRESSION)


def write(
    stage_dir: Path, partition_version: PartitionVersion, rows: Sequence[AssignmentRow]
) -> Mapping[Cohort, int]:
    """Write both labeled cohorts' boxes into the staged tree.

    Each cohort is checked before its own file is written, so a failure on the
    second leaves the first staged and neither uploaded -- the buildspec's
    `post_build` guard is what makes that a job that wrote nothing.
    """
    written: dict[Cohort, int] = {}
    for cohort in sorted(LABELED_COHORTS):
        labels = collect(stage_dir, rows, cohort)
        check(labels, rows, cohort)

        key = cohort_labels_key(partition_version, cohort)
        write_parquet(labels, stage_dir / key)

        boxes = sum(len(label.boxes) for label in labels)
        log.info(
            "%s: %d images and %d boxes from %s, wrote %s",
            cohort.value,
            len(labels),
            boxes,
            COHORT_SPLIT[cohort].value,
            key,
        )
        written[cohort] = len(labels)
    return written

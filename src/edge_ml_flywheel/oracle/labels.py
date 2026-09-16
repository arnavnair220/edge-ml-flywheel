"""What one sold label is, and how the oracle gets hold of it.

A label reaches the oracle only through `fetch`, and `fetch` takes a `Cohorts`
rather than a split. That is the whole design of this module: the key is built
from a cohort lookup, so there is no code path here that can address
`raw/labels/scalabel/val/` -- where `eval` lives -- however the caller is wrong.
The gate is not a check this module performs before doing its work; it is the
thing that produces the path the work is done on.

Boxes are serialized to compact JSON on the way into a purchase's parquet. The
encoding is here rather than in the label writer because it is the same
serialization the archive uses, and a box that reads back a hair off the corners
BDD published is a different label from the one that was bought.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import ImageId, parse_image_id, raw_label_key
from edge_ml_flywheel.ingest.labels import Box, parse_label
from edge_ml_flywheel.oracle.cohorts import Cohorts, purchasable_split

log = logging.getLogger(__name__)

# What a file of labels looks like, wherever it came from. `partition.cohort_labels`
# writes the two cohorts the draw labels and this writes what a cycle bought, and
# they are one schema rather than two identical ones: `training.labels` reads both
# with one function, and that only stays true if there is one place for the column
# names to change.
SCHEMA: Final = pa.schema(
    [
        ("image_id", pa.string()),
        ("boxes", pa.string()),
    ]
)

# snappy for `partition.assign._COMPRESSION`'s reason: this file is written by the
# oracle Lambda, and a Lambda's pyarrow is the managed layer, built without zstd.
COMPRESSION: Final = "snappy"

# The four corners, in the order they are written and read. A positional row
# rather than an object per box: `[category, x1, y1, x2, y2]` costs four numbers
# where a mapping costs four numbers and four keys, across roughly 18 boxes an
# image and 8,000 images a run.
_CORNERS: Final = ("x1", "y1", "x2", "y2")

# No spaces. `json.dumps` puts ", " and ": " between every element by default,
# which is readability nothing reads.
_COMPACT: Final = (",", ":")

# How a purchase fetches one label. A callable rather than a client, so the
# purchase path is testable against a local tree and runs in a Lambda against S3
# without either knowing about the other.
type Fetch = Callable[[ImageId], "SoldLabel"]


@dataclass(frozen=True, slots=True)
class SoldLabel:
    """One image and all of its boxes, which is what a cycle pays for.

    The unit is the image rather than the box because annotation is priced per
    image, so this is also the unit the budget counts. An image with no boxes is
    a legitimate label and a legitimate charge -- "there is nothing here" is an
    answer the model has to learn -- so an empty `boxes` is not refused.
    """

    image_id: ImageId
    boxes: tuple[Box, ...]

    def __post_init__(self) -> None:
        parse_image_id(self.image_id)


def encode_boxes(boxes: Sequence[Box]) -> str:
    """Boxes as the compact JSON a purchase files beside the image ID."""
    return json.dumps(
        [[box.category, *(getattr(box, corner) for corner in _CORNERS)] for box in boxes],
        separators=_COMPACT,
    )


def decode_boxes(encoded: str) -> tuple[Box, ...]:
    """The inverse, strict about shape.

    A row of the wrong width means the encoding changed under labels already
    written, which is unrecoverable rather than degraded: the boxes are the
    label. So it raises rather than yielding a box with a corner defaulted to
    zero, which would be a plausible rectangle in the wrong place.
    """
    rows = json.loads(encoded)
    if not isinstance(rows, list):
        raise ValueError(f"encoded boxes are not a list: {encoded[:80]!r}")

    boxes: list[Box] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(_CORNERS) + 1:
            raise ValueError(f"not a [category, x1, y1, x2, y2] row: {row!r}")
        category, *corners = row
        boxes.append(Box(category, *(float(corner) for corner in corners)))
    return tuple(boxes)


def label_key(cohorts: Cohorts, image_id: ImageId) -> str:
    """The S3 key of one purchasable image's label.

    Routed through `purchasable_split`, which re-runs the gate and raises for
    anything outside the pool. So this function is total on what it returns and
    partial on what it accepts: there is no argument for which it produces a key
    under the split `eval` is drawn from.
    """
    return raw_label_key(image_id, purchasable_split(cohorts, image_id))


def read_label(root: Path, cohorts: Cohorts, image_id: ImageId) -> SoldLabel:
    """One label out of a tree laid out at its S3 keys.

    The local half of `Fetch`, used by tests and by any job that has staged the
    labels. `parse_label` does the archive-shaped validation, so a document that
    is not what this code was written against fails here rather than becoming a
    label with no boxes.
    """
    path = root / label_key(cohorts, image_id)
    parsed = parse_label(json.loads(path.read_text(encoding="utf-8")), image_id)
    return SoldLabel(image_id=image_id, boxes=parsed.boxes)


def local_fetch(root: Path, cohorts: Cohorts) -> Fetch:
    """Bind a staged tree and a cohort index into a `Fetch`."""

    def fetch(image_id: ImageId) -> SoldLabel:
        return read_label(root, cohorts, image_id)

    return fetch


def s3_fetch(client: Any, bucket: str, cohorts: Cohorts) -> Fetch:
    """Bind a bucket and a cohort index into a `Fetch`.

    The other half of `local_fetch`, and deliberately the same three lines in a
    different order: the key comes from `label_key`, which routes through the
    gate, so this reaches S3 only for an image the gate has already admitted. A
    refused image produces no `GetObject` at all -- which is what makes "unread
    rather than merely unsold" a statement about the network and not only about
    the code.

    One object per image rather than a prefix download. A batch is a thousand
    scattered keys out of 80,000, so there is no prefix that names them; and the
    labels this role may read are exactly the ones it can build a key for.
    """

    def fetch(image_id: ImageId) -> SoldLabel:
        key = label_key(cohorts, image_id)
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        parsed = parse_label(json.loads(body), image_id)
        return SoldLabel(image_id=image_id, boxes=parsed.boxes)

    return fetch


def write_parquet(labels: Sequence[SoldLabel], path: Path) -> int:
    """Write a purchase and return how many labels landed.

    Two columns and no cohort, run or cycle: all three are in the key, for
    `AssignmentRow`'s reason. The boxes go through `encode_boxes` rather than a
    list column of structs, which is the worse parquet and the better arrangement
    -- a bought label and a bootstrap one are then the same shape, which is what
    makes the cumulative labeled set one thing rather than two.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_id": [label.image_id for label in labels],
        "boxes": [encode_boxes(label.boxes) for label in labels],
    }
    pq.write_table(pa.table(data, schema=SCHEMA), path, compression=COMPRESSION)

    log.info("wrote %d labels to %s", len(labels), path)
    return len(labels)

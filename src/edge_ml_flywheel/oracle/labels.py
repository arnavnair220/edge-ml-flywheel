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
from typing import Final

from edge_ml_flywheel.conventions import ImageId, parse_image_id, raw_label_key
from edge_ml_flywheel.ingest.labels import Box, parse_label
from edge_ml_flywheel.oracle.cohorts import Cohorts, purchasable_split

log = logging.getLogger(__name__)

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

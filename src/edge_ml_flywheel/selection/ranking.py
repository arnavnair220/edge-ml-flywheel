"""The ranked pool as a file: what scored what, and which of it was bought.

`conventions.SelectionRow` is the schema of record and this is the codec. Pure
and local-path only, like `scoring.detections`: the launcher downloads and
uploads, and nothing here knows about S3.

**One file carries the ranking and the batch.** The batch is the top of the
ranking, so a second document naming its image IDs would be the same fact with a
way to disagree -- and it is the fact a purchase is charged against. The oracle
reads the selected rows out of the evidence that they were selected.

**Written in rank order, and carrying the rank.** The order is how a person reads
it; the column is how a query engine does. Neither is redundant with the other
once the file has been through something that sorts.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import ImageId, SelectionRow, columns, parse_image_id

log = logging.getLogger(__name__)

SCHEMA: Final = pa.schema(
    [
        ("image_id", pa.string()),
        ("score", pa.float64()),
        ("rank", pa.int32()),
        ("selected", pa.bool_()),
    ]
)

# float64 for `scoring.detections.SCHEMA`'s reason one step later: the score is
# what the ordering is built from, and a rounded tie reorders two images and moves
# the batch boundary.

# snappy for `partition.assign._COMPRESSION`'s reason. This file is written by the
# control Lambda and read by the oracle Lambda, and a Lambda's pyarrow is the
# managed layer, which is built without zstd.
_COMPRESSION: Final = "snappy"

_COLUMNS: Final = columns(SelectionRow)


def rows(
    ranked: Sequence[ImageId], scores: Mapping[ImageId, float], batch: Iterable[ImageId]
) -> tuple[SelectionRow, ...]:
    """The ranking as rows, in rank order.

    `ranked` is the whole pool already ordered, and `batch` is the prefix of it
    that was bought. Taking both rather than deriving the batch from a budget,
    because `selection.select` is what decides the batch and re-applying its rule
    here would be a second implementation of the tie-break.
    """
    bought = set(batch)
    outside = sorted(bought - set(ranked))
    if outside:
        raise ValueError(
            f"{len(outside)} image(s) in the batch are not in the ranking it came from: "
            f"{outside[:5]}"
        )

    return tuple(
        SelectionRow(
            image_id=image_id,
            score=scores[image_id],
            rank=position,
            selected=image_id in bought,
        )
        for position, image_id in enumerate(ranked)
    )


def write(ranked: Sequence[SelectionRow], path: Path) -> int:
    """Write the ranking and return how many rows landed.

    Columns from the schema of record, for `partition.assign.write_parquet`'s
    reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: [getattr(row, name) for row in ranked] for name in _COLUMNS}
    pq.write_table(pa.table(data, schema=SCHEMA), path, compression=_COMPRESSION)

    log.info("wrote %d ranked images to %s", len(ranked), path)
    return len(ranked)


def read(path: Path) -> tuple[SelectionRow, ...]:
    """The ranking read back, in the order the file holds it.

    Every row goes through `SelectionRow`, so a file written under a different
    schema fails here rather than becoming a batch of plausible image IDs with the
    wrong flag on them -- which the oracle would charge for.
    """
    table = pq.read_table(path, columns=list(_COLUMNS))
    return tuple(
        SelectionRow(
            image_id=parse_image_id(str(image_id)),
            score=float(score),
            rank=int(rank),
            selected=bool(selected),
        )
        for image_id, score, rank, selected in zip(
            *(table.column(name).to_pylist() for name in _COLUMNS), strict=True
        )
    )


def selected(ranked: Iterable[SelectionRow]) -> tuple[ImageId, ...]:
    """The batch, in rank order.

    A refusal when nothing is flagged, rather than an empty batch: the oracle
    refuses a purchase of no images anyway, and the message it would give names
    the batch rather than the file that failed to describe one.
    """
    batch = tuple(row.image_id for row in ranked if row.selected)
    if not batch:
        raise ValueError(
            "the ranking marks no image as selected, so there is no batch to buy. Selection "
            "writes the top of the ranking flagged; a file with none is one nothing selected."
        )
    return batch

"""The detections file: what one model saw, written once and read by two planes.

`conventions.DetectionRow` is the schema of record and this is the codec. Pure
and local-path only, like `training.labels`: the container writes to a Processing
output directory that SageMaker uploads, and a reader opens objects something
else has already downloaded, so nothing here knows about S3.

**Written as a stream, not as a list.** The remaining pool is 62,000 images at up
to `MAX_DETECTIONS` boxes each, and a list of that many dataclasses is most of a
gigabyte of Python objects held while the model is still on the GPU. `write`
takes an iterable and flushes a row group at a time, so the container's peak
memory is one row group rather than one cohort.

**An empty file is written rather than skipped.** A model that detects nothing is
a real and reportable outcome -- design section 4.2 hard fails a challenger whose
classes collapse to zero detections -- and the gate can only report it if the
absence arrives as a file with no rows. A missing object is indistinguishable
from a job that died before it wrote anything, which is the one case that must
not be read as a verdict.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import DetectionRow, ImageId, columns, parse_image_id
from edge_ml_flywheel.evaluation.coco import Detection

log = logging.getLogger(__name__)

SCHEMA: Final = pa.schema(
    [
        ("image_id", pa.string()),
        ("category", pa.string()),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("x2", pa.float64()),
        ("y2", pa.float64()),
        ("score", pa.float64()),
    ]
)

# float64 rather than float32 throughout, for `match.MatchCache`'s reason applied
# one step earlier: AP sorts every detection in the pass by score, and a rounded
# tie reorders two of them and moves the number. Coordinates follow the scores
# rather than being decided separately, so that a row read back is the row that
# was written and not a nearby one.

# snappy for `partition.cohort_labels`' reason: pyarrow in a Lambda comes from the
# managed AWSSDKPandas layer, which is built without zstd. Selection reads these
# files and the control plane is where it will run.
_COMPRESSION: Final = "snappy"

# Rows per row group, and so the write buffer's depth. A few hundred images'
# worth: large enough that the parquet is not a pile of tiny groups a reader has
# to seek through, small enough that the buffer is megabytes while a model is
# holding a GPU.
ROW_GROUP: Final = 50_000

_COLUMNS: Final = columns(DetectionRow)


def _table(rows: Sequence[DetectionRow]) -> pa.Table:
    """Columns from the schema of record, for `assign.write_parquet`'s reason."""
    data = {name: [getattr(row, name) for row in rows] for name in _COLUMNS}
    return pa.table(data, schema=SCHEMA)


def _chunks(rows: Iterable[DetectionRow], size: int) -> Iterator[list[DetectionRow]]:
    iterator = iter(rows)
    while chunk := list(islice(iterator, size)):
        yield chunk


def write(rows: Iterable[DetectionRow], path: Path) -> int:
    """Write one cohort's detections and return how many rows landed.

    The writer is opened whatever the iterable holds, including nothing, so the
    empty case is a file with a schema and no rows rather than an absent object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with pq.ParquetWriter(path, SCHEMA, compression=_COMPRESSION) as writer:
        for chunk in _chunks(rows, ROW_GROUP):
            writer.write_table(_table(chunk))
            written += len(chunk)

    log.info("wrote %d detections to %s", written, path)
    return written


def parquet_files(root: Path) -> list[Path]:
    """Every parquet under a directory, in a fixed order.

    Recursive and sorted for `training.labels.parquet_files`' reason: a prefix
    downloaded to disk reproduces the key structure below it, and a re-run should
    read the parts in one order.
    """
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def read(path: Path) -> Iterator[DetectionRow]:
    """One detections file, row by row.

    Every row is reconstructed through `DetectionRow`, so a file written under a
    different schema fails here rather than becoming boxes with plausible numbers
    in the wrong fields -- which would score as a model that is simply bad.
    """
    table = pq.read_table(path, columns=list(_COLUMNS))
    for values in zip(*(table.column(name).to_pylist() for name in _COLUMNS), strict=True):
        record = dict(zip(_COLUMNS, values, strict=True))
        yield DetectionRow(
            image_id=parse_image_id(str(record["image_id"])),
            category=str(record["category"]),
            x1=float(record["x1"]),
            y1=float(record["y1"]),
            x2=float(record["x2"]),
            y2=float(record["y2"]),
            score=float(record["score"]),
        )


def collect(root: Path) -> list[DetectionRow]:
    """Every detection under a directory, from however many parts it is in."""
    rows = [row for path in parquet_files(root) for row in read(path)]
    log.info("read %d detections from %s", len(rows), root)
    return rows


def group(rows: Iterable[DetectionRow]) -> dict[ImageId, tuple[Detection, ...]]:
    """Rows to the per-image mapping both consumers take.

    `evaluation.coco.detections` and `selection.score.Predictions` want the same
    shape -- image ID to the boxes found in it -- so the conversion happens once,
    here, rather than once per plane with two chances to drop the score field or
    transpose a corner.

    An image with no detections is simply absent, which is what both consumers
    already expect: `coco.detections` documents that an image may be missing, and
    `score.score_pool` is driven by the pool rather than by this mapping so that
    a blind spot is scored rather than dropped.
    """
    found: defaultdict[ImageId, list[Detection]] = defaultdict(list)
    for row in rows:
        found[row.image_id].append(
            Detection(
                category=row.category,
                x1=row.x1,
                y1=row.y1,
                x2=row.x2,
                y2=row.y2,
                score=row.score,
            )
        )
    return {image_id: tuple(boxes) for image_id, boxes in found.items()}

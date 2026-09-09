"""The cumulative labeled set, read back out of the files it is spread across.

A cycle trains on the bootstrap cohort plus every batch its run has bought, and
those are the same two columns in the same shape: `image_id` and the compact JSON
`oracle.labels.encode_boxes` produces. So one reader serves both, and a bootstrap
label and a bought one are the same thing by the time anything trains on them --
which is the property `partition.cohort_labels` gave up a nicer parquet schema
for.

Pure and local-path only. The container reads a channel that SageMaker has
already copied to disk, and `prepare` reads objects it has already downloaded, so
nothing here knows about S3.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import ImageId, parse_image_id
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle.labels import decode_boxes

log = logging.getLogger(__name__)

# The two columns every label file carries, whichever prefix it came from.
_IMAGE_ID: Final = "image_id"
_BOXES: Final = "boxes"

# How many offending IDs a refusal lists before summarizing, matching
# `partition.cohort_labels`.
_REPORTED: Final = 5


def parquet_files(root: Path) -> list[Path]:
    """Every parquet under a channel directory, in a fixed order.

    Recursive, because an `S3Prefix` channel reproduces the key structure below
    its prefix: the purchases channel arrives as one directory per cycle, and a
    flat listing would find nothing. Sorted so a re-run reads them in one order.
    """
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def read(path: Path) -> Iterator[tuple[ImageId, tuple[Box, ...]]]:
    """One label file, row by row.

    `decode_boxes` is strict about the row shape, so a file written under a
    different encoding fails here rather than becoming an image with no boxes --
    which trains as a frame that genuinely contains nothing.
    """
    table = pq.read_table(path, columns=[_IMAGE_ID, _BOXES])
    for image_id, encoded in zip(
        table.column(_IMAGE_ID).to_pylist(),
        table.column(_BOXES).to_pylist(),
        strict=True,
    ):
        yield parse_image_id(image_id), decode_boxes(encoded)


def collect(roots: Sequence[Path]) -> dict[ImageId, tuple[Box, ...]]:
    """The labeled set, from every channel that carries part of it.

    A repeated image is refused rather than merged or overwritten. The cohorts
    are disjoint by construction and the oracle refuses to sell an image twice,
    so a duplicate here is one of those two invariants having failed upstream --
    and the two available alternatives are worse than stopping: overwriting hides
    it, and merging invents a label that has boxes from two sources.
    """
    labels: dict[ImageId, tuple[Box, ...]] = {}
    duplicated: list[ImageId] = []

    for root in roots:
        if not root.is_dir():
            continue
        for path in parquet_files(root):
            rows = 0
            for image_id, boxes in read(path):
                if image_id in labels:
                    duplicated.append(image_id)
                labels[image_id] = boxes
                rows += 1
            log.info("read %d labels from %s", rows, path)

    if duplicated:
        raise ValueError(
            f"{len(duplicated):,} image(s) are labeled twice across the training channels, which "
            f"means a cohort or a purchase is not disjoint: {sorted(duplicated)[:_REPORTED]}"
        )
    return labels


def box_count(labels: Mapping[ImageId, Sequence[Box]]) -> int:
    return sum(len(boxes) for boxes in labels.values())

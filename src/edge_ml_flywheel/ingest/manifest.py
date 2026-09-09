"""The 80,000-row manifest parquet, and the integrity report beside it.

Every Phase 1 question is a query against this file -- cohort sizing, eval
stratification, per-slice counts, the minimum-slice thresholds -- which is why
it is built before the cohort labels rather than alongside them. Those cannot be
written until the partitioner has assigned cohorts, and the partitioner cannot
run until these counts exist.

`ManifestRow` in `conventions` is the schema of record. The parquet schema below
restates the types, because parquet needs them and a dataclass field does not
carry a column type, but the *names* are asserted equal to `columns(ManifestRow)`
in the unit tests rather than copied and hoped over.

The integrity findings do not live in the manifest. A row is what the archive
says about an image; whether that image is usable is a verdict, and verdicts go
in `_provenance/integrity.json` where the verification suite reads them and
refuses the upload.
"""

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import (
    LABEL_SOURCE,
    NATIVE_IMAGE_SIZE,
    RAW_IMAGES_PREFIX,
    RAW_LABELS_PREFIX,
    RAW_PROVENANCE_PREFIX,
    ImageId,
    ManifestRow,
    Split,
    columns,
    manifest_key,
    parse_image_id,
    raw_image_key,
)
from edge_ml_flywheel.ingest.images import ImageFacts, inspect_image
from edge_ml_flywheel.ingest.labels import parse_label

log = logging.getLogger(__name__)

SCHEMA: Final = pa.schema(
    [
        ("image_id", pa.string()),
        ("split", pa.string()),
        ("weather", pa.string()),
        ("scene", pa.string()),
        ("timeofday", pa.string()),
        ("n_boxes", pa.int32()),
        # The evidence rather than a count of small objects, so any threshold at
        # any resolution stays a query and lives only in the eval module that
        # owns the metric.
        ("box_areas", pa.list_(pa.float64())),
        ("sha256", pa.string()),
        ("label_source", pa.string()),
    ]
)

# zstd over parquet's snappy default. The file is read whole by every query and
# written once, so decompression speed is not the constraint and the smaller
# object is a smaller download every time a cohort or slice is sized.
_COMPRESSION: Final = "zstd"

INTEGRITY_KEY: Final = f"{RAW_PROVENANCE_PREFIX}integrity.json"

_PROGRESS_EVERY: Final = 10_000


def image_dir(stage_dir: Path, split: Split) -> Path:
    return stage_dir / RAW_IMAGES_PREFIX / split.value


def label_dir(stage_dir: Path, split: Split) -> Path:
    return stage_dir / RAW_LABELS_PREFIX / split.value


@dataclass(frozen=True, slots=True)
class IntegrityFinding:
    """One image that design section 4.1's data gate would refuse."""

    image_id: ImageId
    split: str
    problem: str
    detail: str


@dataclass(frozen=True, slots=True)
class ManifestBuild:
    rows: tuple[ManifestRow, ...]
    findings: tuple[IntegrityFinding, ...]
    degenerate_boxes: int

    # The boxed-category vocabulary, measured. `labels` picks boxes by structure
    # rather than by a hardcoded class list, so this is the record of what that
    # structure actually contained -- and the input to `class_set_version` when
    # Phase 2 has to name the classes.
    box_categories: dict[str, int]


def _findings(image_id: ImageId, split: Split, facts: ImageFacts) -> list[IntegrityFinding]:
    if facts.decode_error is not None:
        return [
            IntegrityFinding(
                image_id=image_id,
                split=split.value,
                problem="undecodable",
                detail=facts.decode_error,
            )
        ]

    found: list[IntegrityFinding] = []
    if facts.size != NATIVE_IMAGE_SIZE:
        found.append(
            IntegrityFinding(
                image_id=image_id,
                split=split.value,
                problem="resolution",
                detail=f"{facts.size} is not {NATIVE_IMAGE_SIZE}",
            )
        )
    if facts.is_blank:
        found.append(
            IntegrityFinding(
                image_id=image_id,
                split=split.value,
                problem="blank",
                detail=f"single luminance value {facts.extrema}",
            )
        )
    return found


def build(stage_dir: Path) -> ManifestBuild:
    """Walk the staged tree and produce every manifest row.

    Ordered by split then image ID so that two ingests of the same archive
    produce the same file. Nothing downstream depends on the order, but a
    manifest that is byte-comparable between runs is what makes "this is the
    same data" checkable rather than asserted.
    """
    rows: list[ManifestRow] = []
    findings: list[IntegrityFinding] = []
    categories: Counter[str] = Counter()
    degenerate = 0

    for split in Split:
        for label_path in sorted(label_dir(stage_dir, split).glob("*.json")):
            image_id = parse_image_id(label_path.stem)
            parsed = parse_label(json.loads(label_path.read_text(encoding="utf-8")), image_id)

            facts = inspect_image(stage_dir / raw_image_key(image_id, split))
            findings.extend(_findings(image_id, split, facts))

            categories.update(parsed.box_categories)
            degenerate += parsed.degenerate_boxes

            rows.append(
                ManifestRow(
                    image_id=image_id,
                    split=split,
                    weather=parsed.weather,
                    scene=parsed.scene,
                    timeofday=parsed.timeofday,
                    n_boxes=len(parsed.box_areas),
                    box_areas=parsed.box_areas,
                    sha256=facts.sha256,
                    label_source=LABEL_SOURCE,
                )
            )

            if len(rows) % _PROGRESS_EVERY == 0:
                log.info("%d rows", len(rows))

    return ManifestBuild(
        rows=tuple(rows),
        findings=tuple(findings),
        degenerate_boxes=degenerate,
        box_categories=dict(categories.most_common()),
    )


def write_parquet(rows: Sequence[ManifestRow], path: Path) -> None:
    """Columns come from the schema of record, never from a list restated here.

    `getattr` over `columns(ManifestRow)` rather than nine named comprehensions:
    a renamed field then fails at the writer instead of producing a parquet with
    a column nobody reads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: [getattr(row, name) for row in rows] for name in columns(ManifestRow)}
    pq.write_table(pa.table(data, schema=SCHEMA), path, compression=_COMPRESSION)


def write(stage_dir: Path) -> ManifestBuild:
    """Build the manifest and the integrity report into the staged tree."""
    built = build(stage_dir)

    parquet_path = stage_dir / manifest_key()
    write_parquet(built.rows, parquet_path)

    integrity_path = stage_dir / INTEGRITY_KEY
    integrity_path.parent.mkdir(parents=True, exist_ok=True)
    integrity_path.write_text(
        json.dumps(
            {
                "rows": len(built.rows),
                "findings": [asdict(finding) for finding in built.findings],
                "degenerate_boxes": built.degenerate_boxes,
                "box_categories": built.box_categories,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    log.info(
        "%d rows, %d integrity findings, %d degenerate boxes, %d boxed categories",
        len(built.rows),
        len(built.findings),
        built.degenerate_boxes,
        len(built.box_categories),
    )
    return built


def read_parquet(path: Path) -> pa.Table:
    return pq.read_table(path)


def staged_ids(stage_dir: Path) -> dict[str, dict[str, set[ImageId]]]:
    """Both modalities' staged IDs per split, read off the tree.

    Read from the directories rather than from the manifest on purpose: this is
    what the manifest gets checked *against*. The equality of the two sets is
    what validates the images archive, which has no published digest, and it is
    the check that catches a truncated download.
    """

    def ids(directory: Path, pattern: str) -> set[ImageId]:
        return {parse_image_id(path.stem) for path in directory.glob(pattern)}

    return {
        "images": {split.value: ids(image_dir(stage_dir, split), "*.jpg") for split in Split},
        "labels": {split.value: ids(label_dir(stage_dir, split), "*.json") for split in Split},
    }

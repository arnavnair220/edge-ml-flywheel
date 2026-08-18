"""Where the bytes came from, recorded next to the bytes.

`raw/_provenance/` answers the question a bucket listing cannot: which host
served these archives, on what day, verified against what, under which license,
from which commit of which build. None of it is reconstructable afterwards and
all of it is a line of JSON at ingest time, which is the same trade `ModelManifest`
makes for a model.

The prefix leads with an underscore deliberately -- Hive, Glue and Athena skip
paths beginning with `_` or `.`, so a crawler pointed at `raw/` walks past this.

Verification runs in two places and writes here once. `verify_archives` checks
what a download produced, before an extract has cost anything, and leaves its
result in the work directory; `write` folds that into the record. Splitting them
is what lets the expensive steps fail fast.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import LABEL_SOURCE, RAW_PROVENANCE_PREFIX
from edge_ml_flywheel.ingest.source import ARCHIVES, LICENSE, Archive

log = logging.getLogger(__name__)

SOURCE_KEY: Final = f"{RAW_PROVENANCE_PREFIX}source.json"

# What `verify_archives` leaves in the work directory for `write` to fold in.
# Not under the staged tree: it describes files that are deleted before the
# upload, and it is an intermediate rather than something `raw/` should carry.
ARCHIVES_REPORT: Final = "archives.json"

# Hashing 5.7 GB a megabyte at a time rather than reading it into a 3 GB
# container.
_CHUNK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class ArchiveCheck:
    name: str
    size_bytes: int
    sha256: str
    verified_sha256: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive: Archive, path: Path) -> ArchiveCheck:
    """Check one downloaded archive against what the source claims it is.

    The size check is not ceremony for the images archive -- it is the only
    check it has, because no digest is published for it. It catches the
    truncation a resumed download can leave behind. What it cannot catch is a
    host that has changed what it serves, and that is what the image-ID set
    check in the verification suite is for.
    """
    size = path.stat().st_size
    if size != archive.size_bytes:
        raise ValueError(
            f"{archive.name}: expected {archive.size_bytes:,} bytes, got {size:,}. "
            f"A resumed download that cannot resume looks exactly like this."
        )

    digest = sha256_file(path)
    if archive.sha256 is not None and digest != archive.sha256:
        raise ValueError(
            f"{archive.name}: sha256 is {digest}, expected {archive.sha256}. "
            f"The host is not serving the archive this code was written against."
        )

    return ArchiveCheck(
        name=archive.name,
        size_bytes=size,
        sha256=digest,
        verified_sha256=archive.sha256 is not None,
    )


def verify_archives(work_dir: Path, host: str) -> list[ArchiveCheck]:
    """Check every downloaded archive and record the result in `work_dir`."""
    checks = [verify_archive(archive, work_dir / archive.name) for archive in ARCHIVES.values()]

    (work_dir / ARCHIVES_REPORT).write_text(
        json.dumps(
            {
                "host": host,
                "fetched_at": datetime.now(UTC).isoformat(),
                "archives": [
                    {
                        "name": check.name,
                        "url": ARCHIVES[key].url(host),
                        "size_bytes": check.size_bytes,
                        "sha256": check.sha256,
                        "verified_sha256": check.verified_sha256,
                    }
                    for key, check in zip(ARCHIVES, checks, strict=True)
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for check in checks:
        log.info(
            "%s: %d bytes, sha256 %s (%s)",
            check.name,
            check.size_bytes,
            check.sha256,
            "verified against a published digest" if check.verified_sha256 else "recorded only",
        )
    return checks


def _build_identity() -> dict[str, str | None]:
    """Which build wrote this, from the variables CodeBuild sets itself.

    `None` outside CodeBuild, which is the honest answer when someone has run a
    step by hand -- and is itself the thing a reader six weeks later wants to
    know about a `raw/` prefix.
    """
    return {
        "build_id": os.environ.get("CODEBUILD_BUILD_ID"),
        "build_number": os.environ.get("CODEBUILD_BUILD_NUMBER"),
        "git_commit": os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION"),
        "source_version": os.environ.get("CODEBUILD_SOURCE_VERSION"),
    }


def write(stage_dir: Path, work_dir: Path) -> None:
    """Write `raw/_provenance/source.json` into the staged tree."""
    archives = json.loads((work_dir / ARCHIVES_REPORT).read_text(encoding="utf-8"))

    document = {
        "dataset": "BDD100K",
        "label_source": LABEL_SOURCE,
        "label_format": "legacy 2018 Scalabel, per-image JSON",
        # Recorded because it is a decision, not an accident. The archive ships
        # ground truth for the 20,000 test images the benchmark withholds, and
        # they are excluded at extraction.
        "splits_ingested": ["train", "val"],
        "test_split": "withheld by the benchmark, excluded at extraction, never written to disk",
        "license": LICENSE,
        "build": _build_identity(),
        **archives,
    }

    path = stage_dir / SOURCE_KEY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote %s", SOURCE_KEY)

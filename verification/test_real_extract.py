"""Ingest verification, run against a real extract before anything is uploaded.

Not under `tests/`, and not collected by `pytest` with no arguments, because
these need 5.8 GB of BDD100K on local disk. `buildspecs/ingest.yml` runs them
explicitly with `INGEST_STAGE_DIR` and `INGEST_WORK_DIR` set, between building
the manifest and copying anything to S3. `raw/` is write-once and versioned, so
bad bytes are far cheaper to refuse than to reverse.

Two archives on this host turned out not to match their filenames, so none of
this is ceremony.

Expected numbers are written as literals and never re-derived from the code
under test. A check that asks the manifest how many rows it has and then agrees
with it is not a check.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from edge_ml_flywheel.conventions import (
    RAW_PROVENANCE_PREFIX,
    ImageId,
    ManifestRow,
    columns,
    manifest_key,
)
from edge_ml_flywheel.ingest import manifest as manifest_module
from edge_ml_flywheel.ingest.source import LABELS
from edge_ml_flywheel.ingest.stage import WITHHELD_SPLIT

# BDD100K ships 100,000 labelled images: 70,000 train, 10,000 val and 20,000
# test. The test entries are not stubs -- 367,728 boxes, and an attribute
# distribution indistinguishable from train -- but the benchmark's maintainers
# state they will not publish test labels, so this archive leaks ground truth
# the benchmark deliberately withholds. Dropping it is the leakage guard, and
# these three numbers are the cheapest possible regression test against a future
# re-ingest quietly pulling it back in.
TRAIN_IMAGES = 70_000
VAL_IMAGES = 10_000
POOL_IMAGES = 80_000

SPLITS = ("train", "val")


def _required_dir(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.fail(
            f"{variable} is not set. This suite runs against a real extract; "
            f"see buildspecs/ingest.yml."
        )
    return Path(value)


@pytest.fixture(scope="session")
def stage_dir() -> Path:
    return _required_dir("INGEST_STAGE_DIR")


@pytest.fixture(scope="session")
def work_dir() -> Path:
    return _required_dir("INGEST_WORK_DIR")


@pytest.fixture(scope="session")
def table(stage_dir: Path) -> pa.Table:
    return manifest_module.read_parquet(stage_dir / manifest_key())


@pytest.fixture(scope="session")
def staged(stage_dir: Path) -> dict[str, dict[str, set[ImageId]]]:
    return manifest_module.staged_ids(stage_dir)


@pytest.fixture(scope="session")
def integrity(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / manifest_module.INTEGRITY_KEY
    return dict(json.loads(path.read_text(encoding="utf-8")))


# --- Check 1: the one archive with a published digest -------------------------


class TestLabelsArchive:
    def test_sha256_matches_the_published_digest(self, work_dir: Path) -> None:
        """Re-hashed here rather than trusted from the pre_build check.

        The buildspec verifies this before extracting, so that a host serving
        something else fails in under a minute. That is a fail-fast duplicate of
        this, on purpose: this is the copy that runs immediately before the
        upload, with everything else that has to be true also being checked.
        """
        digest = hashlib.sha256((work_dir / LABELS.name).read_bytes()).hexdigest()
        assert digest == LABELS.sha256


# --- Check 2: the manifest is the shape the pool is ---------------------------


class TestManifestShape:
    def test_columns_are_the_schema_of_record(self, table: pa.Table) -> None:
        assert tuple(table.column_names) == columns(ManifestRow)

    def test_row_count(self, table: pa.Table) -> None:
        assert table.num_rows == POOL_IMAGES

    def test_image_ids_are_unique(self, table: pa.Table) -> None:
        ids = table.column("image_id").to_pylist()
        assert len(set(ids)) == POOL_IMAGES

    def test_label_source_is_constant(self, table: pa.Table) -> None:
        assert set(table.column("label_source").to_pylist()) == {"scalabel"}

    def test_box_counts_agree_with_the_areas(self, table: pa.Table) -> None:
        counts = table.column("n_boxes").to_pylist()
        areas = table.column("box_areas").to_pylist()
        assert all(count == len(area) for count, area in zip(counts, areas, strict=True))

    def test_every_box_area_is_positive(self, table: pa.Table) -> None:
        assert all(area > 0 for row in table.column("box_areas").to_pylist() for area in row)

    def test_every_digest_is_a_lowercase_hex_sha256(self, table: pa.Table) -> None:
        digests = table.column("sha256").to_pylist()
        assert all(len(digest) == 64 for digest in digests)
        assert all(digest == digest.lower() for digest in digests)


# --- Check 3: the withheld test split is not here -----------------------------


class TestWithheldSplit:
    def test_only_train_and_val(self, table: pa.Table) -> None:
        assert set(table.column("split").to_pylist()) == set(SPLITS)

    @pytest.mark.parametrize(
        ("split", "expected"),
        [("train", TRAIN_IMAGES), ("val", VAL_IMAGES)],
    )
    def test_split_counts(self, table: pa.Table, split: str, expected: int) -> None:
        """The real leakage check.

        `split in {train, val}` alone would pass on a manifest where 20,000 test
        images had been relabelled; the counts are what pin the pool to the two
        splits the benchmark publishes labels for.
        """
        splits = table.column("split").to_pylist()
        assert splits.count(split) == expected

    def test_no_test_directory_survived_the_extract(self, stage_dir: Path) -> None:
        offenders = [path for path in stage_dir.rglob("*") if WITHHELD_SPLIT in path.parts]
        assert offenders == []


# --- Check 4: the archive with no digest --------------------------------------


class TestImageAndLabelSets:
    """The check that matters most.

    No sha256 is published for the 5.7 GB images archive, so nothing else
    establishes that the right images arrived. Image IDs equalling label IDs as
    an exact set validates it, catches a truncated download, and is what would
    carry the whole provenance weight if the Hugging Face mirror were ever used
    instead.
    """

    @pytest.mark.parametrize("split", SPLITS)
    def test_image_ids_equal_label_ids(
        self, staged: dict[str, dict[str, set[ImageId]]], split: str
    ) -> None:
        assert staged["images"][split] == staged["labels"][split]

    @pytest.mark.parametrize("split", SPLITS)
    @pytest.mark.parametrize("modality", ["images", "labels"])
    def test_staged_counts(
        self, staged: dict[str, dict[str, set[ImageId]]], modality: str, split: str
    ) -> None:
        expected = TRAIN_IMAGES if split == "train" else VAL_IMAGES
        assert len(staged[modality][split]) == expected

    def test_the_manifest_covers_exactly_the_staged_pool(
        self, table: pa.Table, staged: dict[str, dict[str, set[ImageId]]]
    ) -> None:
        pool = set().union(*staged["labels"].values())
        assert set(table.column("image_id").to_pylist()) == pool


# --- Design section 4.1's data gate, run at ingest ----------------------------


class TestIntegrity:
    def test_no_findings(self, integrity: dict[str, Any]) -> None:
        """Undecodable files, wrong resolution, blank frames.

        Reported with the offending image IDs rather than as a bare count,
        because the decision this failure forces -- re-download, or accept and
        record -- is not one a number can inform.
        """
        assert integrity["findings"] == []

    def test_the_report_agrees_with_the_manifest(
        self, integrity: dict[str, Any], table: pa.Table
    ) -> None:
        assert integrity["rows"] == table.num_rows

    def test_the_boxed_category_vocabulary_was_recorded(self, integrity: dict[str, Any]) -> None:
        """Not asserted equal to a list of ten names, deliberately.

        `labels` picks boxes by structure rather than by category, so this is
        the measurement that says what the vocabulary is. Asserting a
        remembered class list here would fail on a naming difference between the
        legacy and det_20 releases while proving nothing about the data.
        """
        assert integrity["box_categories"]


# --- Provenance ---------------------------------------------------------------


class TestProvenance:
    def test_source_record_exists(self, stage_dir: Path) -> None:
        assert (stage_dir / f"{RAW_PROVENANCE_PREFIX}source.json").is_file()

    def test_provenance_is_hidden_from_crawlers(self) -> None:
        """Glue, Hive and Athena skip a path component starting with `_`.

        Pinned because the corollary is load-bearing in the other direction: no
        queryable data may ever live behind one.
        """
        assert RAW_PROVENANCE_PREFIX.split("/")[1].startswith("_")

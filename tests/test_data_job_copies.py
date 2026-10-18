"""Tests for the copies the ingest and partition builds make through the store.

The buildspecs used to make these copies with the AWS CLI, which reaches AWS and
nothing else. They are now subcommands, and these tests drive each one against
an in-process bucket, for `test_label_staging`'s reason: the property is where
the bytes land, and a mock that returned whatever the test needed could not
show that.

**Ingest** uploads the staged tree at its relative paths and then checks every
staged key arrived. A key the build did not stage is not a failure: a later
step, or a re-ingest, leaves keys this build never wrote.

**Partition** stages the manifest and the version's record, where an absent
record is the first run, and uploads the partition prefix and nothing else. The
staged tree also holds the manifest and raw label documents, which the role may
not write back.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from edge_ml_flywheel import storage
from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    Cohort,
    ImageId,
    PartitionVersion,
    Split,
    assignments_key,
    manifest_key,
    partition_manifest_key,
    partition_prefix,
    partition_spec,
    raw_image_key,
    raw_label_key,
)
from edge_ml_flywheel.ingest.__main__ import main as ingest
from edge_ml_flywheel.partition import assign
from edge_ml_flywheel.partition.__main__ import main as partition
from edge_ml_flywheel.storage import StorageConfig
from edge_ml_flywheel.storage.filesystem import FsspecStore

BUCKET = "edge-ml-flywheel-data-test"

V0 = PartitionVersion(0)

IMAGE = ImageId("0000f77c-6257be58")
OTHER = ImageId("00054602-3bf57337")


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch, memory_storage: StorageConfig) -> FsspecStore:
    """An empty data bucket in the in-memory store, named by configuration."""
    monkeypatch.setenv(storage.DATA_BUCKET, BUCKET)
    return FsspecStore.connect(BUCKET, memory_storage)


def bucket_keys(data: FsspecStore) -> set[str]:
    return set(data.keys(""))


def stage_file(root: Path, key: str, data: bytes | None = None) -> bytes:
    """A file at `root / key`, holding bytes distinct to its key."""
    contents = data if data is not None else key.encode("utf-8")
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return contents


# --- ingest -------------------------------------------------------------------


def an_ingest_stage(root: Path) -> dict[str, bytes]:
    """An image, a label and a manifest, under both prefixes ingest writes."""
    return {
        key: stage_file(root, key)
        for key in (
            raw_image_key(IMAGE, Split.TRAIN),
            raw_label_key(IMAGE, Split.TRAIN),
            manifest_key(),
        )
    }


class TestIngestUpload:
    def test_every_staged_file_lands_at_its_relative_path(
        self, data: FsspecStore, tmp_path: Path
    ) -> None:
        staged = an_ingest_stage(tmp_path)

        ingest(["upload", "--stage-dir", str(tmp_path)])

        assert bucket_keys(data) == set(staged)
        for key, contents in staged.items():
            assert data.get(key) == contents


class TestVerifyUpload:
    def test_a_complete_upload_passes(self, data: FsspecStore, tmp_path: Path) -> None:
        an_ingest_stage(tmp_path)
        ingest(["upload", "--stage-dir", str(tmp_path)])

        ingest(["verify-upload", "--stage-dir", str(tmp_path)])

    def test_a_missing_key_is_reported(self, data: FsspecStore, tmp_path: Path) -> None:
        staged = an_ingest_stage(tmp_path)
        for key in staged:
            if key != raw_label_key(IMAGE, Split.TRAIN):
                data.put(key, staged[key])

        with pytest.raises(SystemExit, match=r"1 of 3 staged keys are not in the bucket") as raised:
            ingest(["verify-upload", "--stage-dir", str(tmp_path)])

        assert raw_label_key(IMAGE, Split.TRAIN) in str(raised.value)

    def test_keys_this_build_did_not_stage_are_allowed(
        self, data: FsspecStore, tmp_path: Path
    ) -> None:
        """A later step or a re-ingest leaves keys here that this build never wrote."""
        an_ingest_stage(tmp_path)
        ingest(["upload", "--stage-dir", str(tmp_path)])
        data.put(raw_image_key(OTHER, Split.VAL), b"")
        data.put(f"{partition_prefix(V0)}_partition.json", b"{}")

        ingest(["verify-upload", "--stage-dir", str(tmp_path)])


# --- partition ----------------------------------------------------------------


def a_record(**overrides: Any) -> bytes:
    """Version 0's `_partition.json`, as this code would write it."""
    document = assign.document(V0, partition_spec(V0)) | overrides
    return json.dumps(document).encode("utf-8")


def fetch_inputs(stage_dir: Path, recorded: Path) -> None:
    partition(
        [
            "fetch-inputs",
            "--stage-dir",
            str(stage_dir),
            "--recorded",
            str(recorded),
            "--partition-version",
            "0",
        ]
    )


class TestFetchInputs:
    def test_a_first_run_stages_the_manifest_and_no_record(
        self, data: FsspecStore, tmp_path: Path
    ) -> None:
        data.put(manifest_key(), b"manifest")
        stage_dir, recorded = tmp_path / "stage", tmp_path / "recorded.json"

        fetch_inputs(stage_dir, recorded)

        assert (stage_dir / manifest_key()).read_bytes() == b"manifest"
        assert not recorded.exists()

    def test_a_re_run_stages_the_record_outside_the_tree(
        self, data: FsspecStore, tmp_path: Path
    ) -> None:
        data.put(manifest_key(), b"manifest")
        data.put(partition_manifest_key(V0), a_record())
        stage_dir, recorded = tmp_path / "stage", tmp_path / "recorded.json"

        fetch_inputs(stage_dir, recorded)

        assert recorded.read_bytes() == a_record()
        assert not (stage_dir / partition_manifest_key(V0)).exists()

    def test_a_stale_local_record_is_not_checked_in_place_of_the_bucket(
        self, data: FsspecStore, tmp_path: Path
    ) -> None:
        data.put(manifest_key(), b"manifest")
        recorded = tmp_path / "recorded.json"
        recorded.write_bytes(a_record(seed=1))

        fetch_inputs(tmp_path / "stage", recorded)

        assert not recorded.exists()

    def test_an_absent_manifest_stops_the_build(self, data: FsspecStore, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="no manifest"):
            fetch_inputs(tmp_path / "stage", tmp_path / "recorded.json")


def a_partition_stage(root: Path) -> set[str]:
    """A staged tree as the build leaves it, and the keys under the partition prefix.

    The manifest and a raw label document sit beside the partition prefix, as
    they do after `fetch-inputs` and `stage-labels`.
    """
    stage_file(root, manifest_key())
    stage_file(root, raw_label_key(IMAGE, COHORT_SPLIT[Cohort.BOOTSTRAP]))

    written = {
        assignments_key(V0),
        f"{partition_prefix(V0)}labels/cohort=bootstrap/part-00000.parquet",
    }
    for key in written:
        stage_file(root, key)
    stage_file(root, partition_manifest_key(V0), a_record())
    return written | {partition_manifest_key(V0)}


def upload(stage_dir: Path) -> None:
    partition(["upload", "--stage-dir", str(stage_dir), "--partition-version", "0"])


class TestPartitionUpload:
    def test_only_the_partition_prefix_is_written(self, data: FsspecStore, tmp_path: Path) -> None:
        written = a_partition_stage(tmp_path)

        upload(tmp_path)

        assert bucket_keys(data) == written
        assert all(key.startswith(partition_prefix(V0)) for key in bucket_keys(data))

    def test_the_record_that_landed_is_checked(self, data: FsspecStore, tmp_path: Path) -> None:
        a_partition_stage(tmp_path)
        stage_file(tmp_path, partition_manifest_key(V0), a_record(seed=1))

        with pytest.raises(SystemExit, match="different seed"):
            upload(tmp_path)

    def test_an_unstaged_partition_is_refused(self, data: FsspecStore, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nothing to upload"):
            upload(tmp_path)

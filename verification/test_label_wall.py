"""The label walls, checked against the deployed data bucket.

Three rules keep the project's numbers honest, and each fails with no symptom
if it stops holding:

- Only the oracle reads `raw/labels/`. Otherwise a training job reads the
  labels directly and the cost-per-label ledger measures nothing.
- Only the evaluation job reads `cohort=eval/`. Otherwise anyone can read the
  answer key to every reported metric.
- Nobody but the partitioner writes the labeled cohorts. Otherwise cycle eight
  is not measured against the same labels as cycle one.

Not under `tests/`: this needs real credentials and a deployed bucket, so
`pytest` with no arguments must not collect it. Run it through the object store,
configured like any job:

    STORAGE_DATA_BUCKET=<bucket> uv run pytest verification/test_label_wall.py

`STORAGE_PROTOCOL` and `STORAGE_ENDPOINT_URL` point it at MinIO or another
provider. A refusal is asserted as `PermissionError`, which `s3fs` raises for a
denied S3 or MinIO request.

**The identity under test is the one running the suite.** Nothing here assumes
another identity, so each run checks what one identity can and cannot do.

**A negative test needs a positive control.** A refusal is also what a wrong key
or a wrong bucket produces. So each refused key is first shown to exist by
listing, and an image is read before a label is refused.

**Writes are probed at a key nothing claims.** A cohort's boxes are `part-000`,
so a write probe aims at `part-999`. If a freeze ever fails, the damage is a
stray object beside the labels, not a frozen file overwritten.
"""

import pytest

from edge_ml_flywheel.conventions import (
    PARTITIONS,
    RAW_IMAGES_PREFIX,
    Cohort,
    PartitionVersion,
    Split,
    cohort_labels_key,
    cohort_labels_prefix,
    parse_image_id,
    raw_image_key,
    raw_label_key,
)
from edge_ml_flywheel.storage import DATA_BUCKET, ObjectStore, bucket_name
from edge_ml_flywheel.storage.filesystem import FsspecStore

# `val`, not `train`. Both sit behind the same rule, and a stray success on the
# split no cohort trains on is not a read of anything a model could learn from.
PROBE_SPLIT = Split.VAL

# The part number the write probes aim at. Nothing writes it.
UNCLAIMED_PART = 999


@pytest.fixture(scope="session")
def store() -> ObjectStore:
    return FsspecStore.connect(bucket_name(DATA_BUCKET))


@pytest.fixture(scope="session")
def partition_version() -> PartitionVersion:
    """The newest version defined, which is the one in the bucket."""
    return max(PARTITIONS)


@pytest.fixture(scope="session")
def image_id(store: ObjectStore) -> str:
    """One real image ID, taken from the bucket rather than written down here."""
    prefix = f"{RAW_IMAGES_PREFIX}{PROBE_SPLIT.value}/"
    key = next(store.keys(prefix), None)
    if key is None:
        pytest.fail(f"no images under {prefix} in {store.bucket}")
    name = key.rsplit("/", 1)[-1]
    return str(parse_image_id(name.removesuffix(".jpg")))


class TestThePositiveControl:
    """What makes a refusal below mean something."""

    def test_an_image_reads(self, store: ObjectStore, image_id: str) -> None:
        """Credentials work, the bucket is real, and the walls are narrow."""
        key = raw_image_key(parse_image_id(image_id), PROBE_SPLIT)
        assert len(store.get(key)) > 0

    def test_the_refused_label_exists(self, store: ObjectStore, image_id: str) -> None:
        key = raw_label_key(parse_image_id(image_id), PROBE_SPLIT)
        assert store.exists(key)

    def test_the_refused_eval_labels_exist(
        self, store: ObjectStore, partition_version: PartitionVersion
    ) -> None:
        prefix = cohort_labels_prefix(partition_version, Cohort.EVAL)
        assert next(store.keys(prefix), None) is not None


class TestTheLabelWall:
    """Only the oracle reads `raw/labels/`."""

    def test_a_withheld_label_cannot_be_read(self, store: ObjectStore, image_id: str) -> None:
        key = raw_label_key(parse_image_id(image_id), PROBE_SPLIT)
        with pytest.raises(PermissionError):
            store.get(key)

    def test_the_wall_covers_train(self, store: ObjectStore) -> None:
        """`train` holds the labels the oracle sells, so it gets its own check.

        The key is constructed rather than listed. A refused caller is refused
        before the store checks the key exists, so `NoSuchKeyError` here means
        the wall stopped applying to `train`.
        """
        key = raw_label_key(parse_image_id("00000000-00000000"), Split.TRAIN)
        with pytest.raises(PermissionError):
            store.get(key)


class TestTheEvalWall:
    """Only the evaluation job reads `cohort=eval/`."""

    def test_an_eval_label_cannot_be_read(
        self, store: ObjectStore, partition_version: PartitionVersion
    ) -> None:
        key = cohort_labels_key(partition_version, Cohort.EVAL)
        with pytest.raises(PermissionError):
            store.get(key)


class TestTheFreeze:
    """Nobody but the partitioner writes the labeled cohorts.

    Both cohorts are checked, because it would be easy to narrow the freeze to
    `eval` while believing it intact. Only writes are probed: a delete aimed at
    a prefix no rerun can reproduce is not worth the check.
    """

    def test_the_eval_labels_cannot_be_overwritten(
        self, store: ObjectStore, partition_version: PartitionVersion
    ) -> None:
        key = cohort_labels_key(partition_version, Cohort.EVAL, part=UNCLAIMED_PART)
        with pytest.raises(PermissionError):
            store.put(key, b"")

    def test_the_bootstrap_labels_cannot_be_overwritten(
        self, store: ObjectStore, partition_version: PartitionVersion
    ) -> None:
        key = cohort_labels_key(partition_version, Cohort.BOOTSTRAP, part=UNCLAIMED_PART)
        with pytest.raises(PermissionError):
            store.put(key, b"")

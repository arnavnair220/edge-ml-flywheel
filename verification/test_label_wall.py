"""The label walls, checked against the deployed bucket policy.

Every cost-per-label number this project reports rests on one statement:
`WithheldLabelsAreOracleOnly` in `infra/storage.tf`, which denies `s3:GetObject`
on `raw/labels/*` to every principal outside `label_reader_arns`. If that deny
stops holding, a training job reads 80,000 label files directly, the ledger
measures nothing, and every gate still passes -- a failure with no symptom. So it
is checked rather than assumed.

Two further statements carry that same shape now that a partition is drawn, and
are checked here for the same reason. `EvalLabelsAreScoringOnly` denies reads of
`labels/cohort=eval/`, the second copy of the ground truth that lives outside the
prefix above; what a read there costs is the eval rather than the budget.
`LabelsAreFrozenExceptThePartitioner` denies writes to both labeled cohorts, and
is what makes cycle eight's number comparable to cycle one's rather than merely
intended to be. Neither failure has a symptom either.

Not under `tests/`, for `test_real_extract.py`'s reason: this needs real
credentials and a deployed stack, so `pytest` with no arguments must not collect
it. Run it as whoever administers the account:

    DATA_BUCKET=edge-ml-flywheel-data-<account> uv run pytest verification/test_label_wall.py

**The identity under test is the one running the suite**, and no role is created
for the purpose. An operator with full administrative rights is not on the
allowlist either, so the ordinary developer session is already the case the wall
has to refuse -- and it is the case that matters most, because it is the one
standing exemption anybody would be tempted to grant.

**A negative test needs a positive control.** `AccessDenied` on its own is also
what a typo in the key, an expired session and a wrong bucket name produce. So
the suite reads an image before it reads a label, and a `bootstrap` label before
an `eval` one. The read that succeeds is what makes the read that fails evidence
rather than coincidence.

**Writes are probed at a key nothing claims.** A frozen prefix is checked by
attempting to write into it, and the whole point of the test is that the attempt
should fail -- so it names a part number no writer emits. If the freeze ever
stops holding, the damage is a stray object beside the labels rather than a
frozen file overwritten by its own test.

**What is deliberately not checked here.** That an allowlist still *admits* its
members: the oracle loader exercises `label_reader_arns` as a real job whose
whole function is reading those files, `eval_label_reader_arns` is exercised by
the evaluation job -- whose whole function is matching against those boxes, so a
cycle that produces a gate report is that allowlist working -- and the
partitioner's write exemption is exercised by the partitioner. Nor that the
training and scoring roles are refused, which needs a suite that can assume them.
Creating a stand-in for any of them would mean a standing role that can read or
rewrite ground truth and be assumed from anywhere in the account, which is the
exemption `storage.tf` refuses on purpose.
"""

import os
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError

from edge_ml_flywheel.conventions import (
    PARTITIONS,
    RAW_IMAGES_PREFIX,
    RAW_LABELS_PREFIX,
    Cohort,
    PartitionVersion,
    Split,
    cohort_labels_key,
    cohort_labels_prefix,
    parse_image_id,
    raw_image_key,
    raw_label_key,
)

# What the bucket policy returns to a principal it refuses. S3 reports a denial
# and a missing object identically to a caller with no `ListBucket`, which is why
# the suite establishes the object exists before asserting on the code.
ACCESS_DENIED = "AccessDenied"

# `val`, not `train`. Both sit behind the same deny, and reading the split the
# trainable cohorts do not come from means a stray success here is not a read of
# something a model could have been trained on.
PROBE_SPLIT = Split.VAL

# The part number the write probes aim at. A cohort's boxes are one file,
# `part-000`, so nothing writes this one and a failure of the freeze leaves an
# object beside the labels rather than on top of one.
UNCLAIMED_PART = 999


def _required(variable: str) -> str:
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is not set. This suite runs against the deployed bucket.")
    return value


def _error_code(raised: ClientError) -> str:
    return str(raised.response["Error"]["Code"])


@pytest.fixture(scope="session")
def bucket() -> str:
    return _required("DATA_BUCKET")


@pytest.fixture(scope="session")
def s3() -> Any:
    """S3 as whoever is running the suite. Normally an account administrator."""
    return boto3.client("s3")


@pytest.fixture(scope="session")
def partition_version() -> PartitionVersion:
    """The newest version anyone has defined, which is the one in the bucket.

    Read from `PARTITIONS` rather than from an environment variable: a version
    absent from that registry is one nobody could have written, and the deny is
    wildcarded over `partition_version=` anyway, so the check does not depend on
    which version this happens to be.
    """
    return max(PARTITIONS)


@pytest.fixture(scope="session")
def image_id(s3: Any, bucket: str) -> str:
    """One real image ID, taken from the bucket rather than written down here.

    A hardcoded ID that a re-ingest had moved would fail as NoSuchKey, which is
    exactly the false negative the positive control exists to rule out.
    """
    listing = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{RAW_IMAGES_PREFIX}{PROBE_SPLIT.value}/",
        MaxKeys=1,
    )
    contents = listing.get("Contents")
    if not contents:
        pytest.fail(f"no images under {RAW_IMAGES_PREFIX}{PROBE_SPLIT.value}/ in {bucket}")

    name = contents[0]["Key"].rsplit("/", 1)[-1]
    return str(parse_image_id(name.removesuffix(".jpg")))


class TestThePositiveControl:
    """The wall is a refusal, and these are what make a refusal mean something."""

    def test_an_image_reads(self, s3: Any, bucket: str, image_id: str) -> None:
        """Credentials work, the bucket is real, and the deny is narrow.

        Without this, every assertion below is satisfied by having no access at
        all -- or by no network.
        """
        key = raw_image_key(parse_image_id(image_id), PROBE_SPLIT)
        assert s3.get_object(Bucket=bucket, Key=key)["ContentLength"] > 0

    def test_the_label_that_is_denied_below_actually_exists(
        self, s3: Any, bucket: str, image_id: str
    ) -> None:
        """So the denial is a denial and not a 404.

        `HeadObject` is denied along with `GetObject`, so existence is
        established by listing -- which the wall deliberately does not cover.
        """
        key = raw_label_key(parse_image_id(image_id), PROBE_SPLIT)
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        assert [entry["Key"] for entry in listing.get("Contents", [])] == [key]

    def test_a_bootstrap_label_reads(
        self, s3: Any, bucket: str, partition_version: PartitionVersion
    ) -> None:
        """The control for the eval deny, and the asymmetry it turns on.

        These two files sit side by side under one partition prefix and are
        treated as opposites on purpose: `bootstrap` is training's input and is
        reachable by whatever `derived/` grant a role carries, while `eval` is
        denied to everyone. Reading this one is what makes the refusal next door
        a statement about `cohort=eval/` rather than about `derived/`.
        """
        key = cohort_labels_key(partition_version, Cohort.BOOTSTRAP)
        assert s3.get_object(Bucket=bucket, Key=key)["ContentLength"] > 0

    def test_the_eval_labels_that_are_denied_below_actually_exist(
        self, s3: Any, bucket: str, partition_version: PartitionVersion
    ) -> None:
        """So the denial is a denial and not a 404, as above.

        Listed rather than headed for the same reason: `HeadObject` is denied
        alongside `GetObject`, and the deny covers objects rather than the
        listing.
        """
        prefix = cohort_labels_prefix(partition_version, Cohort.EVAL)
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        assert listing["KeyCount"] > 0


class TestTheWall:
    def test_a_withheld_label_cannot_be_read(self, s3: Any, bucket: str, image_id: str) -> None:
        """The statement the budget rests on, executed rather than believed."""
        key = raw_label_key(parse_image_id(image_id), PROBE_SPLIT)
        with pytest.raises(ClientError) as raised:
            s3.get_object(Bucket=bucket, Key=key)
        assert _error_code(raised.value) == ACCESS_DENIED

    def test_there_is_no_administrative_exemption(
        self, s3: Any, bucket: str, image_id: str
    ) -> None:
        """Stated separately from the test above, because it is a second claim.

        That one says the wall refuses *someone*. This one says who: the identity
        running the suite holds whatever rights its operator holds, and is
        refused anyway. `storage.tf` scopes the deny to object actions, so this
        principal can still replace the bucket policy -- the route back to a
        label is a reviewable Terraform diff, never a session that happened to be
        privileged enough.
        """
        key = raw_label_key(parse_image_id(image_id), PROBE_SPLIT)
        with pytest.raises(ClientError) as raised:
            s3.get_object(Bucket=bucket, Key=key)
        assert _error_code(raised.value) == ACCESS_DENIED

    def test_the_deny_covers_train_and_not_only_val(self, s3: Any, bucket: str) -> None:
        """`train` holds the 62,000 the oracle sells, so it gets its own check.

        The key is constructed rather than listed, and whether the object exists
        is beside the point: a caller with no read on the prefix is refused
        before S3 decides. NoSuchKey here would mean the deny stopped applying to
        the split that matters most.
        """
        key = raw_label_key(parse_image_id("00000000-00000000"), Split.TRAIN)
        with pytest.raises(ClientError) as raised:
            s3.get_object(Bucket=bucket, Key=key)
        assert _error_code(raised.value) == ACCESS_DENIED


class TestTheEvalWall:
    """`EvalLabelsAreScoringOnly`: the copy of ground truth the scorer reads."""

    def test_an_eval_label_cannot_be_read(
        self, s3: Any, bucket: str, partition_version: PartitionVersion
    ) -> None:
        """Nobody outside an allowlist of exactly one.

        `eval_label_reader_arns` holds the evaluation role and nothing else, so
        this is now the same shape of test as the label wall above: the identity
        running the suite is an administrator, is not that role, and is refused.
        An operator who can read these boxes can read the answer key to every
        number the project reports, which is why the exemption is one job rather
        than one person.
        """
        key = cohort_labels_key(partition_version, Cohort.EVAL)
        with pytest.raises(ClientError) as raised:
            s3.get_object(Bucket=bucket, Key=key)
        assert _error_code(raised.value) == ACCESS_DENIED


class TestTheFreeze:
    """`LabelsAreFrozenExceptThePartitioner`: one statement over both cohorts.

    So both are checked. The two prefixes differ in every other way -- one is
    training's input and one is denied to everybody -- and it would be easy to
    narrow this deny to `cohort=eval/` while believing the freeze intact.

    `PutObject` alone, though the statement also names the delete actions. An
    overwrite is what a job does by accident; a delete is not, and probing it
    would mean pointing a destructive call at a prefix whose contents no rerun
    can reproduce.
    """

    def test_the_eval_labels_cannot_be_overwritten(
        self, s3: Any, bucket: str, partition_version: PartitionVersion
    ) -> None:
        key = cohort_labels_key(partition_version, Cohort.EVAL, part=UNCLAIMED_PART)
        with pytest.raises(ClientError) as raised:
            s3.put_object(Bucket=bucket, Key=key, Body=b"")
        assert _error_code(raised.value) == ACCESS_DENIED

    def test_the_bootstrap_labels_cannot_be_overwritten(
        self, s3: Any, bucket: str, partition_version: PartitionVersion
    ) -> None:
        """Readable by training and still frozen, which is the whole claim.

        The cohort a job may read is the one it might be tempted to rewrite, and
        a bootstrap set that changed between cycles would make the label
        efficiency curve a comparison of two different experiments.
        """
        key = cohort_labels_key(partition_version, Cohort.BOOTSTRAP, part=UNCLAIMED_PART)
        with pytest.raises(ClientError) as raised:
            s3.put_object(Bucket=bucket, Key=key, Body=b"")
        assert _error_code(raised.value) == ACCESS_DENIED


class TestTheWallsScope:
    """What the wall does not cover, pinned so that a change to it is deliberate."""

    def test_listing_the_label_prefix_is_permitted(self, s3: Any, bucket: str) -> None:
        """A label key carries an image ID, and image IDs are in the manifest.

        Denying the listing would withhold a fact every role already reads while
        making the real deny harder to reason about. Asserted because the
        reasoning holds only while a label key carries nothing else -- a future
        `label_source` that encoded more into the key would need this revisited.
        """
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=RAW_LABELS_PREFIX, MaxKeys=3)
        assert listing["KeyCount"] > 0

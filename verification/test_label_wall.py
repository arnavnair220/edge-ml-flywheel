"""The label wall, checked against the deployed bucket policy.

Every cost-per-label number this project reports rests on one statement:
`WithheldLabelsAreOracleOnly` in `infra/storage.tf`, which denies `s3:GetObject`
on `raw/labels/*` to every principal outside `label_reader_arns`. If that deny
stops holding, a training job reads 80,000 label files directly, the ledger
measures nothing, and every gate still passes -- a failure with no symptom. So it
is checked rather than assumed.

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
the suite reads an image before it reads a label. The image succeeding is what
makes the label failing evidence rather than coincidence.

**What is deliberately not checked here.** That the allowlist still *admits* its
members, and that Phase 3's training role is refused. The first is exercised by
the oracle loader -- a real job, on the list, whose whole function is reading
these files -- and the second joins this suite when that role exists. Creating a
stand-in for either would mean a standing role that can read labels and be
assumed from anywhere in the account, which is the exemption `storage.tf`
refuses on purpose.
"""

import os
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError

from edge_ml_flywheel.conventions import (
    RAW_IMAGES_PREFIX,
    RAW_LABELS_PREFIX,
    Split,
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

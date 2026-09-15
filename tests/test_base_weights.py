"""Tests for staging the COCO base, which `prepare` does so nobody has to.

The behaviour worth pinning is the branch, not the download: the base is fetched
on the first cycle of a fresh account and every later cycle finds the object and
makes no request at all. A second fetch would be a second set of bytes claiming
to be the checkpoint forty jobs started from, so "already there" has to mean
"leave it alone" rather than "overwrite it with the same thing".

`urlopen` is monkeypatched rather than allowed out to GitHub. A test that
reaches the network tests GitHub's availability, and the one that matters here
asserts the opposite -- that nothing leaves the account on the ordinary path.
"""

import urllib.request
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import boto3
import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import base_weights_key
from edge_ml_flywheel.training import launch

REGION = "us-east-1"

# What moto's STS hands back, which is what `Buckets.for_account` names the
# bucket from.
ACCOUNT = "123456789012"

WEIGHTS = b"not a checkpoint, but the bytes that land"


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """An empty artifacts bucket, under fake credentials.

    The credentials are fake for `test_control`'s reason: a test whose mock
    failed to engage would otherwise reach a real account, and this one writes.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        session = boto3.Session(region_name=REGION)
        session.client("s3").create_bucket(Bucket=f"edge-ml-flywheel-artifacts-{ACCOUNT}")
        yield session


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every URL opened, and serves `WEIGHTS` for each."""
    urls: list[str] = []

    def fake(url: str, *args: Any, **kwargs: Any) -> BytesIO:
        urls.append(url)
        return BytesIO(WEIGHTS)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return urls


def stored(aws: boto3.Session) -> bytes:
    body = aws.client("s3").get_object(
        Bucket=launch.buckets(aws).artifacts, Key=base_weights_key()
    )["Body"]
    return bytes(body.read())


class TestEnsureBase:
    def test_an_empty_bucket_is_staged_from_the_release(
        self, artifacts: boto3.Session, fetched: list[str]
    ) -> None:
        key = launch.ensure_base(artifacts)

        assert key == base_weights_key()
        assert stored(artifacts) == WEIGHTS
        assert len(fetched) == 1

    def test_a_staged_base_is_not_fetched_again(
        self, artifacts: boto3.Session, fetched: list[str]
    ) -> None:
        launch.ensure_base(artifacts)
        launch.ensure_base(artifacts)

        assert fetched == fetched[:1]

    def test_a_staged_base_is_never_overwritten(
        self, artifacts: boto3.Session, fetched: list[str]
    ) -> None:
        """The bytes forty jobs started from survive a later cycle's check.

        Distinct from the call count above: a fetch that was skipped and a fetch
        whose result was discarded look the same from the caller and differ
        entirely in the bucket.
        """
        artifacts.client("s3").put_object(
            Bucket=launch.buckets(artifacts).artifacts,
            Key=base_weights_key(),
            Body=b"the checkpoint this project actually trained from",
        )

        launch.ensure_base(artifacts)

        assert stored(artifacts) == b"the checkpoint this project actually trained from"
        assert fetched == []

"""The storage contract, run against every S3-compatible backend this project uses.

One suite, parametrized over backends, because the claim `storage` makes is that
the same code works against all of them. A test written per backend would prove
three implementations agree with three sets of expectations.

- **moto** runs on every invocation. It is in-process and needs nothing.
- **MinIO** runs in a container started once for the session, and is skipped
  where no Docker daemon answers. It is the backend the local cluster uses, and
  a second, independent implementation of the API.
- **AWS S3** runs only when `STORAGE_TEST_AWS_BUCKET` names a bucket, with
  credentials and region from the caller's environment. Each test writes under
  its own prefix and removes what it wrote.

Every test addresses keys through `Backend.key`, which puts them under a prefix
unique to the test, so the backends that share one bucket for the session never
see another test's objects.
"""

import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import boto3
import docker
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto import mock_aws
from testcontainers.community.minio import MinioContainer

from edge_ml_flywheel.storage import (
    ARTIFACTS_BUCKET,
    DATA_BUCKET,
    ENDPOINT_URL,
    REGION,
    TELEMETRY_BUCKET,
    NoSuchKeyError,
    StorageConfig,
    bucket_name,
    buckets,
)
from edge_ml_flywheel.storage.s3 import S3Store, client

MOTO_REGION: Final = "us-east-1"

# Chainguard's build of MinIO from upstream source. MinIO no longer publishes
# images: `minio/minio` is gone from Docker Hub and quay.io refuses anonymous
# pulls. Chainguard's free tier tags only `latest`, so the image is pinned by
# digest, and a MinIO release cannot change what this suite tests against
# between two runs of the same commit.
MINIO_IMAGE: Final = (
    "cgr.dev/chainguard/minio"
    "@sha256:709099319add48dfb44aa596a7070d3ddd8dd04d7ae881765719aec29df9e510"
)

# The bucket the MinIO container and moto are given. AWS uses the caller's.
LOCAL_BUCKET: Final = "storage-contract"

# The opt-in for AWS S3. Unset means the backend is skipped, so a plain
# `pytest` never writes to a real account.
AWS_BUCKET_VARIABLE: Final = "STORAGE_TEST_AWS_BUCKET"

# One more than a single `ListObjectsV2` page, which every backend caps at 1,000.
PAGED: Final = 1_001


@dataclass(frozen=True, slots=True)
class Backend:
    """A store under test, and the raw client that set it up and cleans it up."""

    name: str
    store: S3Store
    admin: Any
    prefix: str

    def key(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"


def _docker_answers() -> bool:
    try:
        docker.from_env().ping()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def minio() -> Iterator[tuple[StorageConfig, str, str]]:
    """One MinIO server for the session: its config and its two credentials."""
    if not _docker_answers():
        pytest.skip("no Docker daemon, so there is no MinIO to test against")

    with MinioContainer(MINIO_IMAGE) as container:
        endpoint = (
            f"http://{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(container.port)}"
        )
        config = StorageConfig(endpoint_url=endpoint, region=MOTO_REGION)
        admin = _minio_client(config, container.access_key, container.secret_key)
        admin.create_bucket(Bucket=LOCAL_BUCKET)
        yield config, container.access_key, container.secret_key


def _minio_client(config: StorageConfig, access_key: str, secret_key: str) -> Any:
    return boto3.session.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key
    ).client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name=config.region,
        config=Config(s3={"addressing_style": "path"}),
    )


def _fake_credentials(monkeypatch: pytest.MonkeyPatch, key: str, secret: str) -> None:
    """Credentials in the environment, which is where the SDK's chain looks first.

    Set for moto as well as MinIO, for `test_base_weights`' reason: a test whose
    mock failed to engage would otherwise reach a real account.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(params=["moto", "minio", "aws"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Backend]:
    prefix = f"contract/{uuid.uuid4().hex}/"

    if request.param == "moto":
        _fake_credentials(monkeypatch, "testing", "testing")
        with mock_aws():
            config = StorageConfig(region=MOTO_REGION)
            admin = client(config)
            admin.create_bucket(Bucket=LOCAL_BUCKET)
            yield Backend("moto", S3Store.connect(LOCAL_BUCKET, config), admin, prefix)
        return

    if request.param == "minio":
        config, access_key, secret_key = request.getfixturevalue("minio")
        _fake_credentials(monkeypatch, access_key, secret_key)
        admin = _minio_client(config, access_key, secret_key)
        yield Backend("minio", S3Store.connect(LOCAL_BUCKET, config), admin, prefix)
        _remove(admin, LOCAL_BUCKET, prefix)
        return

    bucket = os.environ.get(AWS_BUCKET_VARIABLE, "")
    if not bucket:
        pytest.skip(f"{AWS_BUCKET_VARIABLE} is not set, so AWS S3 is not tested")
    config = StorageConfig.from_env()
    admin = client(config)
    yield Backend("aws", S3Store.connect(bucket, config), admin, prefix)
    _remove(admin, bucket, prefix)


def _remove(admin: Any, bucket: str, prefix: str) -> None:
    pages = admin.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        found = [{"Key": entry["Key"]} for entry in page.get("Contents", ())]
        if found:
            admin.delete_objects(Bucket=bucket, Delete={"Objects": found, "Quiet": True})


class TestGetAndPut:
    def test_what_is_put_is_what_is_got(self, backend: Backend) -> None:
        backend.store.put(backend.key("a.json"), b'{"a":1}')

        assert backend.store.get(backend.key("a.json")) == b'{"a":1}'

    def test_a_put_replaces_what_was_there(self, backend: Backend) -> None:
        backend.store.put(backend.key("a.json"), b"first")
        backend.store.put(backend.key("a.json"), b"second")

        assert backend.store.get(backend.key("a.json")) == b"second"

    def test_an_empty_object_is_an_object(self, backend: Backend) -> None:
        backend.store.put(backend.key("empty"), b"")

        assert backend.store.get(backend.key("empty")) == b""
        assert backend.store.exists(backend.key("empty"))

    def test_an_absent_key_raises_missing(self, backend: Backend) -> None:
        with pytest.raises(NoSuchKeyError):
            backend.store.get(backend.key("absent.json"))

    def test_the_store_names_its_bucket(self, backend: Backend) -> None:
        assert backend.store.bucket


class TestFiles:
    def test_an_upload_reads_back_byte_for_byte(self, backend: Backend, tmp_path: Path) -> None:
        local = tmp_path / "model.onnx"
        local.write_bytes(bytes(range(256)) * 64)

        backend.store.upload(local, backend.key("model.onnx"))

        assert backend.store.get(backend.key("model.onnx")) == local.read_bytes()

    def test_a_download_creates_the_directories_it_lands_in(
        self, backend: Backend, tmp_path: Path
    ) -> None:
        backend.store.put(backend.key("x/y/z.json"), b"nested")
        destination = tmp_path / "a" / "b" / "c" / "z.json"

        backend.store.download(backend.key("x/y/z.json"), destination)

        assert destination.read_bytes() == b"nested"

    def test_downloading_an_absent_key_raises_missing(
        self, backend: Backend, tmp_path: Path
    ) -> None:
        with pytest.raises(NoSuchKeyError):
            backend.store.download(backend.key("absent.pt"), tmp_path / "absent.pt")


class TestKeys:
    def test_only_keys_under_the_prefix_are_listed(self, backend: Backend) -> None:
        for suffix in ("in/a", "in/b", "out/c"):
            backend.store.put(backend.key(suffix), b"")

        assert sorted(backend.store.keys(backend.key("in/"))) == [
            backend.key("in/a"),
            backend.key("in/b"),
        ]

    def test_an_empty_prefix_lists_nothing(self, backend: Backend) -> None:
        assert list(backend.store.keys(backend.key("nothing/"))) == []

    def test_the_listing_continues_past_one_page(self, backend: Backend) -> None:
        """A listing that stopped at the first page would drop images silently."""
        wanted = [backend.key(f"many/{index:05d}") for index in range(PAGED)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda key: backend.store.put(key, b""), wanted))

        assert sorted(backend.store.keys(backend.key("many/"))) == wanted


class TestExists:
    def test_a_written_key_exists(self, backend: Backend) -> None:
        backend.store.put(backend.key("a/b"), b"")

        assert backend.store.exists(backend.key("a/b"))

    def test_an_absent_key_does_not(self, backend: Backend) -> None:
        assert not backend.store.exists(backend.key("a/b"))

    def test_a_key_that_only_prefixes_another_does_not(self, backend: Backend) -> None:
        """`a/b` is not there because `a/b.json` is."""
        backend.store.put(backend.key("a/b.json"), b"")

        assert not backend.store.exists(backend.key("a/b"))

    def test_a_directory_like_prefix_is_not_a_key(self, backend: Backend) -> None:
        backend.store.put(backend.key("a/b/c"), b"")

        assert not backend.store.exists(backend.key("a/b"))


class TestDenied:
    """A refused read stays an error. `NoSuchKeyError` would turn a permission fault into
    a first run, and the caller would proceed as though nothing had been written.
    """

    def test_a_denied_get_is_not_missing(self) -> None:
        s3 = boto3.session.Session(
            aws_access_key_id="testing", aws_secret_access_key="testing"
        ).client("s3", region_name=MOTO_REGION)
        with Stubber(s3) as stub:
            stub.add_client_error(
                "get_object", service_error_code="AccessDenied", http_status_code=403
            )
            with pytest.raises(ClientError) as raised:
                S3Store("bucket", s3).get("labels/cohort=eval/x.json")

        assert not isinstance(raised.value, NoSuchKeyError)

    def test_wrong_credentials_are_refused_rather_than_missing(
        self,
        minio: tuple[StorageConfig, str, str],
        tmp_path: Path,
    ) -> None:
        """Against a real server, for both read paths: a GET and a download's HEAD."""
        config, access_key, secret_key = minio
        _minio_client(config, access_key, secret_key).put_object(
            Bucket=LOCAL_BUCKET, Key="denied/x", Body=b"x"
        )
        stranger = S3Store(LOCAL_BUCKET, _minio_client(config, "stranger", "not-the-secret"))

        with pytest.raises(ClientError) as got:
            stranger.get("denied/x")
        with pytest.raises(ClientError) as downloaded:
            stranger.download("denied/x", tmp_path / "x")

        assert not isinstance(got.value, NoSuchKeyError)
        assert not isinstance(downloaded.value, NoSuchKeyError)


class TestConfig:
    def test_unset_means_the_sdk_decides(self) -> None:
        assert StorageConfig.from_env({}) == StorageConfig(endpoint_url=None, region=None)

    def test_an_empty_value_is_unset(self) -> None:
        """A manifest that templates an absent value writes an empty string."""
        assert StorageConfig.from_env({ENDPOINT_URL: "", REGION: ""}) == StorageConfig()

    def test_set_values_are_read(self) -> None:
        environ = {ENDPOINT_URL: "http://minio:9000", REGION: "us-east-1"}

        assert StorageConfig.from_env(environ) == StorageConfig(
            endpoint_url="http://minio:9000", region="us-east-1"
        )

    def test_an_endpoint_switches_to_path_style(self) -> None:
        configured = client(StorageConfig(endpoint_url="http://minio:9000", region="us-east-1"))

        assert configured.meta.config.s3["addressing_style"] == "path"
        assert configured.meta.endpoint_url == "http://minio:9000"

    def test_no_endpoint_leaves_aws_addressing_alone(self) -> None:
        configured = client(StorageConfig(region="us-east-1"))

        assert (configured.meta.config.s3 or {}).get("addressing_style") != "path"


class TestBucketNames:
    def test_a_configured_name_is_read(self) -> None:
        assert bucket_name(DATA_BUCKET, {DATA_BUCKET: "data"}) == "data"

    def test_an_unset_name_stops_the_job(self) -> None:
        with pytest.raises(SystemExit, match=DATA_BUCKET):
            bucket_name(DATA_BUCKET, {})

    def test_an_empty_name_stops_the_job(self) -> None:
        with pytest.raises(SystemExit, match=DATA_BUCKET):
            bucket_name(DATA_BUCKET, {DATA_BUCKET: ""})

    def test_all_three_are_read(self) -> None:
        environ = {DATA_BUCKET: "d", ARTIFACTS_BUCKET: "a", TELEMETRY_BUCKET: "t"}

        named = buckets(environ)

        assert (named.data, named.artifacts, named.telemetry) == ("d", "a", "t")

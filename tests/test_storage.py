"""The storage contract, run against every backend this project uses.

One suite, parametrized over backends, because the claim `storage` makes is that
the same code works against all of them. A test written per backend would prove
four implementations agree with four sets of expectations.

- **memory** runs on every invocation. It is fsspec's in-process filesystem,
  and the one the job tests run against.
- **moto** runs on every invocation, as a threaded server on loopback. It is a
  model of the S3 API, reached through `s3fs` exactly as AWS is.
- **MinIO** runs in a container started once for the session, and is skipped
  where no Docker daemon answers. It is the backend the local cluster uses, and
  a second, independent implementation of the S3 API.
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
from fsspec.implementations.memory import MemoryFileSystem
from testcontainers.community.minio import MinioContainer

from edge_ml_flywheel.storage import (
    ARTIFACTS_BUCKET,
    DATA_BUCKET,
    ENDPOINT_URL,
    PROTOCOL,
    REGION,
    TELEMETRY_BUCKET,
    NoSuchKeyError,
    StorageConfig,
    bucket_name,
    buckets,
    tree,
)
from edge_ml_flywheel.storage.filesystem import FsspecStore, filesystem

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

# The bucket the local backends are given. AWS uses the caller's.
LOCAL_BUCKET: Final = "storage-contract"

# The opt-in for AWS S3. Unset means the backend is skipped, so a plain
# `pytest` never writes to a real account.
AWS_BUCKET_VARIABLE: Final = "STORAGE_TEST_AWS_BUCKET"

# One more than a single `ListObjectsV2` page, which every S3 backend caps at 1,000.
PAGED: Final = 1_001


@dataclass(frozen=True, slots=True)
class Backend:
    """A store under test, and the prefix this test's keys live under."""

    name: str
    store: FsspecStore
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
        admin = _admin(config, container.access_key, container.secret_key)
        admin.create_bucket(Bucket=LOCAL_BUCKET)
        yield config, container.access_key, container.secret_key


def _admin(config: StorageConfig, access_key: str, secret_key: str) -> Any:
    """A plain boto3 client, which sets a backend up and cleans it up.

    Outside the store on purpose: a fixture that wrote and removed objects
    through the code under test would pass when that code was wrong both ways.
    """
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
    endpoint failed to apply would otherwise reach a real account.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(params=["memory", "moto", "minio", "aws"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Backend]:
    prefix = f"contract/{uuid.uuid4().hex}/"

    if request.param == "memory":
        config = request.getfixturevalue("memory_storage")
        yield Backend("memory", FsspecStore.connect(LOCAL_BUCKET, config), prefix)
        return

    if request.param == "moto":
        _fake_credentials(monkeypatch, "testing", "testing")
        config = StorageConfig(
            endpoint_url=request.getfixturevalue("moto_server"), region=MOTO_REGION
        )
        admin = _admin(config, "testing", "testing")
        admin.create_bucket(Bucket=LOCAL_BUCKET)
        yield Backend("moto", FsspecStore.connect(LOCAL_BUCKET, config), prefix)
        _remove(admin, LOCAL_BUCKET, prefix)
        return

    if request.param == "minio":
        config, access_key, secret_key = request.getfixturevalue("minio")
        _fake_credentials(monkeypatch, access_key, secret_key)
        admin = _admin(config, access_key, secret_key)
        yield Backend("minio", FsspecStore.connect(LOCAL_BUCKET, config), prefix)
        _remove(admin, LOCAL_BUCKET, prefix)
        return

    bucket = os.environ.get(AWS_BUCKET_VARIABLE, "")
    if not bucket:
        pytest.skip(f"{AWS_BUCKET_VARIABLE} is not set, so AWS S3 is not tested")
    config = StorageConfig.from_env()
    yield Backend("aws", FsspecStore.connect(bucket, config), prefix)
    _remove(boto3.session.Session().client("s3", region_name=config.region), bucket, prefix)


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

    def test_a_prefix_need_not_end_at_a_directory(self, backend: Backend) -> None:
        """A key prefix, not a path: `labels/cohort=` names keys in several directories."""
        for suffix in ("cohort=a/x", "cohort=b/y", "other/z"):
            backend.store.put(backend.key(suffix), b"")

        assert sorted(backend.store.keys(backend.key("cohort="))) == [
            backend.key("cohort=a/x"),
            backend.key("cohort=b/y"),
        ]

    def test_an_empty_prefix_lists_nothing(self, backend: Backend) -> None:
        assert list(backend.store.keys(backend.key("nothing/"))) == []

    def test_a_key_written_after_a_listing_is_listed(self, backend: Backend) -> None:
        """No listing is cached, so a second listing sees what the first could not."""
        backend.store.put(backend.key("late/a"), b"")
        list(backend.store.keys(backend.key("late/")))

        backend.store.put(backend.key("late/b"), b"")

        assert sorted(backend.store.keys(backend.key("late/"))) == [
            backend.key("late/a"),
            backend.key("late/b"),
        ]

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

    def test_a_key_written_after_an_absent_answer_exists(self, backend: Backend) -> None:
        """A first run asks, finds nothing, writes, and asks again."""
        assert not backend.store.exists(backend.key("record.json"))

        backend.store.put(backend.key("record.json"), b"{}")

        assert backend.store.exists(backend.key("record.json"))


class _Refusing(MemoryFileSystem):  # type: ignore[misc]
    """A filesystem that refuses every read, the way a denied identity is refused."""

    def cat_file(self, path: str, *args: Any, **kwargs: Any) -> bytes:
        raise PermissionError(f"refused {path}")


class TestDenied:
    """A refused read stays an error. `NoSuchKeyError` would turn a permission fault into
    a first run, and the caller would proceed as though nothing had been written.
    """

    def test_a_denied_get_is_not_missing(self) -> None:
        store = FsspecStore("bucket", _Refusing(skip_instance_cache=True))

        with pytest.raises(PermissionError) as raised:
            store.get("labels/cohort=eval/x.json")

        assert not isinstance(raised.value, NoSuchKeyError)

    def test_wrong_credentials_are_refused_rather_than_missing(
        self,
        minio: tuple[StorageConfig, str, str],
        tmp_path: Path,
    ) -> None:
        """Against a real server, for both read paths: a GET and a download."""
        config, access_key, secret_key = minio
        _admin(config, access_key, secret_key).put_object(
            Bucket=LOCAL_BUCKET, Key="denied/x", Body=b"x"
        )
        stranger = FsspecStore(
            LOCAL_BUCKET, filesystem(config, key="stranger", secret="not-the-secret")
        )

        with pytest.raises(PermissionError) as got:
            stranger.get("denied/x")
        with pytest.raises(PermissionError) as downloaded:
            stranger.download("denied/x", tmp_path / "x")

        assert not isinstance(got.value, NoSuchKeyError)
        assert not isinstance(downloaded.value, NoSuchKeyError)


class _RefusesOnePut:
    """A store that refuses writes to one key and passes everything else through."""

    def __init__(self, store: FsspecStore, refused: str) -> None:
        self._store = store
        self._refused = refused

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def put(self, key: str, body: bytes) -> None:
        if key == self._refused:
            raise PermissionError(f"refused {key}")
        self._store.put(key, body)


class TestTreeUpload:
    def test_every_file_lands_at_the_prefix_plus_its_relative_path(
        self, backend: Backend, tmp_path: Path
    ) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "top.json").write_bytes(b"top")
        (tmp_path / "a" / "b" / "deep.json").write_bytes(b"deep")

        landed = tree.upload(backend.store, tmp_path, backend.prefix)

        assert landed == 2
        assert sorted(backend.store.keys(backend.prefix)) == [
            backend.key("a/b/deep.json"),
            backend.key("top.json"),
        ]
        assert backend.store.get(backend.key("a/b/deep.json")) == b"deep"

    def test_every_file_is_attempted_before_a_failure_is_raised(
        self, backend: Backend, tmp_path: Path
    ) -> None:
        """The count says whether one object failed or the whole prefix is denied."""
        for name in ("a", "b", "c"):
            (tmp_path / name).write_bytes(name.encode())
        store = _RefusesOnePut(backend.store, backend.key("b"))

        with pytest.raises(RuntimeError, match=r"1 of 3 objects could not be uploaded"):
            tree.upload(store, tmp_path, backend.prefix)

        assert sorted(backend.store.keys(backend.prefix)) == [backend.key("a"), backend.key("c")]

    def test_a_missing_directory_is_refused(self, backend: Backend, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tree.upload(backend.store, tmp_path / "absent", backend.prefix)


class TestConfig:
    def test_unset_means_s3_and_the_sdk_decides(self) -> None:
        assert StorageConfig.from_env({}) == StorageConfig(
            protocol="s3", endpoint_url=None, region=None
        )

    def test_an_empty_value_is_unset(self) -> None:
        """A manifest that templates an absent value writes an empty string."""
        assert StorageConfig.from_env({PROTOCOL: "", ENDPOINT_URL: "", REGION: ""}) == (
            StorageConfig()
        )

    def test_set_values_are_read(self) -> None:
        environ = {PROTOCOL: "s3", ENDPOINT_URL: "http://minio:9000", REGION: "us-east-1"}

        assert StorageConfig.from_env(environ) == StorageConfig(
            protocol="s3", endpoint_url="http://minio:9000", region="us-east-1"
        )

    def test_the_protocol_selects_the_filesystem(self) -> None:
        assert isinstance(filesystem(StorageConfig(protocol="memory")), MemoryFileSystem)

    def test_an_endpoint_switches_to_path_style(self) -> None:
        configured = filesystem(StorageConfig(endpoint_url="http://minio:9000", region="us-east-1"))

        assert configured.config_kwargs["s3"]["addressing_style"] == "path"
        assert configured.endpoint_url == "http://minio:9000"

    def test_no_endpoint_leaves_aws_addressing_alone(self) -> None:
        configured = filesystem(StorageConfig(region="us-east-1"))

        assert "s3" not in configured.config_kwargs


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

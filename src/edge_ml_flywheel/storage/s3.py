"""`ObjectStore` over the S3 API, which is the only implementation.

One implementation because one API reaches every provider this project runs on:
AWS S3 natively, MinIO by design, and GCS through its XML API's S3
interoperability. What differs between them is configuration, and
`StorageConfig` is where that difference lives.

Two settings change when an endpoint is configured, and both are provider
behaviours rather than preferences:

- **Path-style addressing.** The SDK otherwise puts the bucket in the hostname,
  which a MinIO service on a cluster address does not answer.
- **Checksums only when required.** The SDK attaches CRC checksums to every
  upload by default, and GCS interoperability refuses headers it does not
  implement. AWS S3 keeps the default.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Self

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from edge_ml_flywheel.storage import NoSuchKeyError, StorageConfig

# Retries inside the SDK rather than a loop at each call site. Many small reads
# against one prefix is exactly the shape a store answers with a throttle, and a
# job that fails on one after thousands of good reads has thrown away the work
# for a condition the SDK handles.
_ATTEMPTS: Final = 5

# The SDK's own connection-pool default. A caller that shares one store across a
# thread pool passes its thread count instead, so the threads do not queue behind
# connections.
POOL: Final = 10

# The error codes that mean "no such key". `NoSuchKey` comes from a GET; a HEAD
# has no body to carry a code, so the transfer manager's reports the status.
_ABSENT: Final = frozenset({"NoSuchKey", "404"})


def client(config: StorageConfig, pool: int = POOL) -> Any:
    """An S3 client for `config`, safe to share across threads once built."""
    options: dict[str, Any] = {
        "max_pool_connections": pool,
        "retries": {"max_attempts": _ATTEMPTS, "mode": "standard"},
    }
    if config.endpoint_url is not None:
        options["s3"] = {"addressing_style": "path"}
        options["request_checksum_calculation"] = "when_required"
        options["response_checksum_validation"] = "when_required"

    return boto3.session.Session().client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name=config.region,
        config=Config(**options),
    )


class S3Store:
    """One bucket behind an S3 client."""

    def __init__(self, bucket: str, s3: Any) -> None:
        self._bucket = bucket
        self._s3 = s3

    @classmethod
    def connect(cls, bucket: str, config: StorageConfig | None = None, pool: int = POOL) -> Self:
        """A store for `bucket`, configured from the environment unless told."""
        return cls(bucket, client(config if config is not None else StorageConfig.from_env(), pool))

    @property
    def bucket(self) -> str:
        return self._bucket

    def get(self, key: str) -> bytes:
        try:
            body: bytes = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as error:
            if _absent(error):
                raise NoSuchKeyError(f"s3://{self._bucket}/{key}") from error
            raise
        return body

    def put(self, key: str, body: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body)

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self._bucket, key, str(path))
        except ClientError as error:
            if _absent(error):
                raise NoSuchKeyError(f"s3://{self._bucket}/{key}") from error
            raise

    def upload(self, path: Path, key: str) -> None:
        self._s3.upload_file(str(path), self._bucket, key)

    def keys(self, prefix: str) -> Iterator[str]:
        pages = self._s3.get_paginator("list_objects_v2").paginate(
            Bucket=self._bucket, Prefix=prefix
        )
        for page in pages:
            for entry in page.get("Contents", ()):
                yield str(entry["Key"])

    def exists(self, key: str) -> bool:
        """A listing rather than a HEAD, and an exact match on what it returns.

        A HEAD on an absent key answers 403 rather than 404 to an identity that
        may read objects but not list them, so the answer would depend on the
        grant rather than on the bucket. The listing asks the question the
        callers mean, and the exact match keeps `a/b` from being reported
        present because `a/b.json` is.
        """
        response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=key, MaxKeys=1)
        return any(entry["Key"] == key for entry in response.get("Contents", ()))


def _absent(error: ClientError) -> bool:
    """Whether the SDK's error means the key is not there.

    A denied read is not absent, and stays a `ClientError`. Reporting it as
    `NoSuchKeyError` would turn a permission fault into a first run.
    """
    return error.response.get("Error", {}).get("Code") in _ABSENT

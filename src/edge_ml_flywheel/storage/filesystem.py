"""`ObjectStore` over an fsspec filesystem, which is the only implementation.

One implementation because fsspec already is the abstraction over providers:
`s3fs` reaches AWS S3 and MinIO, `gcsfs` reaches GCS through its own API, and
`memory` is a whole object store in the process for tests. What differs between
them is the protocol and its settings, and `StorageConfig` is where that
difference lives.

Two settings are fixed for every filesystem this module builds:

- **No listings cache.** `s3fs` caches directory listings by default, so a key
  written after the first listing would read as absent to `exists` and `keys`
  for the life of the process. Every listing here goes to the store.
- **Path-style addressing when an endpoint is configured.** The SDK otherwise
  puts the bucket in the hostname, which a MinIO service on a cluster address
  does not answer.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, Self

import fsspec
from fsspec import AbstractFileSystem

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


def filesystem(config: StorageConfig, pool: int = POOL, **options: Any) -> AbstractFileSystem:
    """A filesystem for `config`, safe to share across threads once built.

    `options` go to the filesystem's constructor as they are, for the S3 caller
    that must name its credentials -- a profile, a key pair -- rather than take
    the SDK's chain.
    """
    if config.protocol != "s3":
        return fsspec.filesystem(config.protocol, **options)

    settings: dict[str, Any] = {
        "max_pool_connections": pool,
        "retries": {"max_attempts": _ATTEMPTS, "mode": "standard"},
    }
    if config.endpoint_url is not None:
        settings["s3"] = {"addressing_style": "path"}

    return fsspec.filesystem(
        "s3",
        endpoint_url=config.endpoint_url,
        client_kwargs={"region_name": config.region},
        config_kwargs=settings,
        use_listings_cache=False,
        skip_instance_cache=True,
        **options,
    )


class FsspecStore:
    """One bucket on an fsspec filesystem."""

    def __init__(self, bucket: str, fs: AbstractFileSystem) -> None:
        self._bucket = bucket
        self._fs = fs

    @classmethod
    def connect(cls, bucket: str, config: StorageConfig | None = None, pool: int = POOL) -> Self:
        """A store for `bucket`, configured from the environment unless told."""
        configured = config if config is not None else StorageConfig.from_env()
        return cls(bucket, filesystem(configured, pool))

    @classmethod
    def from_session(cls, aws: Any, bucket: str) -> Self:
        """A store for `bucket` under a boto3 session's credentials and region.

        For the old stack's callers, which hold a session rather than storage
        configuration: an operator's profile, or a Lambda's role. The session's
        credentials are passed as they are now, because `s3fs` builds its own
        client and cannot take one from boto3. The endpoint still comes from the
        environment, which is unset in AWS and names a local server in tests.
        """
        credentials = aws.get_credentials().get_frozen_credentials()
        fs = filesystem(
            replace(StorageConfig.from_env(), protocol="s3", region=aws.region_name),
            key=credentials.access_key,
            secret=credentials.secret_key,
            token=credentials.token,
        )
        return cls(bucket, fs)

    @property
    def bucket(self) -> str:
        return self._bucket

    def get(self, key: str) -> bytes:
        try:
            body: bytes = self._fs.cat_file(self._path(key))
        except FileNotFoundError as error:
            raise NoSuchKeyError(self._url(key)) from error
        return body

    def put(self, key: str, body: bytes) -> None:
        self._fs.pipe_file(self._path(key), body)

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fs.get_file(self._path(key), str(path))
        except FileNotFoundError as error:
            raise NoSuchKeyError(self._url(key)) from error

    def upload(self, path: Path, key: str) -> None:
        self._fs.put_file(str(path), self._path(key))

    def keys(self, prefix: str) -> Iterator[str]:
        """One flat listing from the bucket's root, filtered by `prefix`.

        `prefix` is passed to the listing rather than folded into the path,
        because it is a key prefix and not a directory: `s3fs` and `gcsfs` send
        it as the request's own prefix, and a filesystem that ignores it lists
        more than it needs to, which the filter below then drops. A path ending
        in a directory would also make `s3fs` probe that path with a HEAD when
        the listing is empty, and a HEAD answers an identity that may not list
        with 403.
        """
        root = f"{self._bucket}/"
        for name in self._fs.find(self._bucket, prefix=prefix):
            key = str(name).lstrip("/").removeprefix(root)
            if key.startswith(prefix):
                yield key

    def exists(self, key: str) -> bool:
        """A listing rather than a HEAD, and an exact match on what it returns.

        A HEAD on an absent key answers 403 rather than 404 to an identity that
        may read objects but not list them, so the answer would depend on the
        grant rather than on the bucket. The listing asks the question the
        callers mean, and the exact match keeps `a/b` from being reported
        present because `a/b.json` is.
        """
        return any(found == key for found in self.keys(key))

    def _path(self, key: str) -> str:
        return f"{self._bucket}/{key}"

    def _url(self, key: str) -> str:
        protocol = self._fs.protocol
        scheme = protocol if isinstance(protocol, str) else protocol[0]
        return f"{scheme}://{self._bucket}/{key}"

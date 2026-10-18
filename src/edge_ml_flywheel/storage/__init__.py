"""The object store, as the operations the jobs perform on it.

Six operations and nothing else, each one a call some job already makes: read a
small object, write one, move a large file in either direction, list a prefix,
and ask whether one key is there. A capability nothing calls is surface a second
implementation would have to match for no reader.

**Where the store is comes from configuration, never from the provider.** The
endpoint, the region and the three bucket names are environment variables, so
the same code reaches AWS S3, MinIO and GCS through its S3-compatible API by a
change of settings rather than of code. Credentials are not among them: the
SDK's default chain resolves them, which is environment keys locally and the
cluster's workload identity in a cloud.

**The store holds no access rule of its own.** What a job may read is decided by
the identity its credentials belong to and, for labels, by the key builders in
`oracle.labels` and `partition.cohort_labels`, which cannot produce an `eval`
key. A store that filtered keys would be a second copy of that rule.
"""

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from edge_ml_flywheel.conventions import Buckets

# The variables, named once. `docs/storage.md` lists them for an operator; this
# is the list the code reads.
ENDPOINT_URL = "STORAGE_ENDPOINT_URL"
REGION = "STORAGE_REGION"
DATA_BUCKET = "STORAGE_DATA_BUCKET"
ARTIFACTS_BUCKET = "STORAGE_ARTIFACTS_BUCKET"
TELEMETRY_BUCKET = "STORAGE_TELEMETRY_BUCKET"


class NoSuchKeyError(LookupError):
    """The key is not in the bucket.

    One exception for every implementation, so a caller that treats an absent
    object as an answer -- a first run, a cycle not yet prepared -- does not
    have to know which SDK's error code means "not found".
    """


class ObjectStore(Protocol):
    """One bucket, and the six things a job does to it."""

    @property
    def bucket(self) -> str: ...

    def get(self, key: str) -> bytes:
        """The whole object, in memory. Raises `NoSuchKeyError` if it is not there."""
        ...

    def put(self, key: str, body: bytes) -> None: ...

    def download(self, key: str, path: Path) -> None:
        """The object to a local file, creating its directory. Raises `NoSuchKeyError`."""
        ...

    def upload(self, path: Path, key: str) -> None: ...

    def keys(self, prefix: str) -> Iterator[str]:
        """Every key under `prefix`, across as many listing pages as it takes."""
        ...

    def exists(self, key: str) -> bool:
        """Whether this exact key is there -- not merely something it prefixes."""
        ...


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Where the S3-compatible API is.

    `endpoint_url` is `None` for AWS S3, which the SDK resolves from the region.
    Any other provider is an explicit URL: MinIO's service address locally, and
    `https://storage.googleapis.com` for GCS. `region` is `None` to leave it to
    the SDK's own chain, which is what a pod on EKS or a Greengrass device
    already sets.
    """

    endpoint_url: str | None = None
    region: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        return cls(
            endpoint_url=environ.get(ENDPOINT_URL) or None,
            region=environ.get(REGION) or None,
        )


def bucket_name(variable: str, environ: Mapping[str, str] = os.environ) -> str:
    """One bucket's name, as configured.

    Read rather than derived. A name built from an AWS account ID is a name only
    AWS can produce, and on MinIO or GCS there is no account to build it from;
    the infrastructure that created the buckets knows their names and hands them
    to every job that needs one.
    """
    name = environ.get(variable, "")
    if not name:
        raise SystemExit(f"{variable} is not set. Bucket names come from configuration.")
    return name


def buckets(environ: Mapping[str, str] = os.environ) -> Buckets:
    """All three bucket names, for a caller that addresses more than one.

    A job that touches one bucket asks for that one through `bucket_name`, so
    it does not need configuration for buckets it never opens.
    """
    return Buckets(
        data=bucket_name(DATA_BUCKET, environ),
        artifacts=bucket_name(ARTIFACTS_BUCKET, environ),
        telemetry=bucket_name(TELEMETRY_BUCKET, environ),
    )

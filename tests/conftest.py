"""Fixtures shared across test modules."""

import logging
from collections.abc import Iterator

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from moto.server import ThreadedMotoServer

from edge_ml_flywheel import storage
from edge_ml_flywheel.storage import StorageConfig

# The configuration every in-memory store is built from, in a test and in the
# job under test alike.
MEMORY = StorageConfig(protocol="memory")


@pytest.fixture(scope="session")
def moto_server() -> Iterator[str]:
    """moto as an HTTP server on loopback, for the session, and its URL.

    `s3fs` sends requests through aiohttp, which `mock_aws` does not intercept,
    so a store reaches moto here instead. The server runs in this process and
    shares `mock_aws`'s state, so a bucket a test creates through a mocked
    boto3 client is the bucket a store on this URL sees.

    Loopback rather than moto's default `0.0.0.0`, which is a listening address
    and not one a client on Windows can connect to.
    """
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def memory_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageConfig]:
    """An empty in-memory object store, and the configuration that reaches it.

    The environment selects the `memory` protocol too, so a job entrypoint that
    connects from configuration reaches the same objects the test wrote. fsspec
    keeps one memory store per process, so it is emptied before and after each
    test rather than relied on to start clean.
    """
    monkeypatch.setenv(storage.PROTOCOL, MEMORY.protocol)
    monkeypatch.delenv(storage.ENDPOINT_URL, raising=False)
    _empty()
    yield MEMORY
    _empty()


def _empty() -> None:
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs.clear()
    MemoryFileSystem.pseudo_dirs.append("")

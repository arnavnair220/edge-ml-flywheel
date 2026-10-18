"""Copying many objects between the store and local disk, below one prefix.

An object lands at `root / <key with prefix removed>`, and a file goes up to
`prefix + <path relative to root>`. With no prefix that is `ingest.stage`'s rule
-- the tree is the bucket -- which the ingest and partition jobs use. The other
jobs pass the prefix of each input and a short directory of their own for it,
because a full key is long enough that `root / key` passes the 260-character
path limit Windows still applies by default.

Many small requests from one process are request latency, which is what a thread
pool is for -- and the reason this is not a loop of `download` or `upload`
calls, whose transfer manager starts its own threads to multipart something that
fits in one request.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Final

from edge_ml_flywheel.storage import ObjectStore

log = logging.getLogger(__name__)

# Threads, and the size a caller should give the store's connection pool. A pool
# narrower than the thread count serialises the threads behind connections and
# logs a warning per wait.
WORKERS: Final = 32

# How often the progress line is printed. Silence for two minutes in a job log is
# indistinguishable from a hang.
_EVERY: Final = 1_000

# Failures named in the error. The count is the number that matters; the first
# few keys are what says whether it was one bad object or the whole prefix.
_REPORTED: Final = 5


def fetch(
    store: ObjectStore,
    keys: Sequence[str],
    root: Path,
    prefix: str = "",
    what: str = "objects",
) -> int:
    """Copy every key below `prefix` into `root`, and return how many landed.

    Every key is attempted before anything is raised. A partial copy fails the
    job either way, and the count of what failed is the difference between "one
    object is missing" and "this identity cannot read the prefix at all" --
    which are the two things that go wrong here and want different fixes.
    """
    outside = [key for key in keys if not key.startswith(prefix)]
    if outside:
        raise ValueError(f"{len(outside)} key(s) are not under {prefix!r}: {outside[:_REPORTED]}")

    targets = {key: root / key.removeprefix(prefix) for key in keys}
    directories = {path.parent for path in targets.values()}
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    log.info("staging %d %s into %d directories", len(keys), what, len(directories))

    landed = _attempt_all(
        {key: partial(_get, store, key, path) for key, path in targets.items()}, "staged"
    )
    if landed.failures:
        raise RuntimeError(
            f"{len(landed.failures)} of {len(keys)} {what} could not be staged, so anything "
            f"read from this tree would be short: {landed.named()}"
        )

    log.info("staged %d %s", landed.count, what)
    return landed.count


def fetch_prefix(store: ObjectStore, prefix: str, root: Path, what: str = "objects") -> int:
    """Copy everything under `prefix` into `root`, and return how many landed.

    Zero is a count rather than an error. Whether an empty prefix is a fault is
    the caller's question: a run's purchases are empty at cycle 0, and its eval
    labels never are.
    """
    return fetch(store, sorted(store.keys(prefix)), root, prefix, what)


def upload(store: ObjectStore, root: Path, prefix: str = "", what: str = "objects") -> int:
    """Copy every file under `root` to `prefix + <relative path>`, and return how many landed.

    `fetch` in the other direction, with its rule: every file is attempted
    before anything is raised, and the count says whether one object failed or
    the identity cannot write the prefix at all. Every key starts with `prefix`,
    so nothing outside it is written whatever else sits beside `root`.

    A missing `root` is an error rather than zero, because the callers upload a
    tree an earlier step was meant to have written.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"nothing to upload: {root} is not a directory")

    files = {
        prefix + path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    log.info("uploading %d %s to %s/%s", len(files), what, store.bucket, prefix)

    landed = _attempt_all(
        {key: partial(_put, store, key, path) for key, path in files.items()}, "uploaded"
    )
    if landed.failures:
        raise RuntimeError(
            f"{len(landed.failures)} of {len(files)} {what} could not be uploaded, so the "
            f"bucket holds a partial copy: {landed.named()}"
        )

    log.info("uploaded %d %s", landed.count, what)
    return landed.count


class _Attempted:
    """What a pool of copies came to: how many landed, and what failed."""

    def __init__(self) -> None:
        self.count = 0
        self.failures: list[tuple[str, Exception]] = []

    def named(self) -> str:
        return ", ".join(f"{key} ({error})" for key, error in self.failures[:_REPORTED])


def _attempt_all(copies: Mapping[str, Callable[[], None]], verb: str) -> _Attempted:
    """Run every copy on the pool, logging progress, and collect failures by key."""
    attempted = _Attempted()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(copy): key for key, copy in copies.items()}
        for done in as_completed(futures):
            try:
                done.result()
            except Exception as error:
                attempted.failures.append((futures[done], error))
                continue

            attempted.count += 1
            if attempted.count % _EVERY == 0:
                log.info("%s %d of %d", verb, attempted.count, len(copies))

    return attempted


def _get(store: ObjectStore, key: str, path: Path) -> None:
    path.write_bytes(store.get(key))


def _put(store: ObjectStore, key: str, path: Path) -> None:
    store.put(key, path.read_bytes())

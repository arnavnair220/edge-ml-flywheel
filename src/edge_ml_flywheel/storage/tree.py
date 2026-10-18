"""Copying many objects onto local disk, below one prefix.

An object lands at `root / <key with prefix removed>`. With no prefix that is
`ingest.stage`'s rule -- the tree is the bucket -- which the partition job uses.
The jobs pass the prefix of each input and a short directory of their own for
it, because a full key is long enough that `root / key` passes the 260-character
path limit Windows still applies by default.

Many small reads from one process are request latency, which is what a thread
pool is for -- and the reason this is not a loop of `download` calls, whose
transfer manager starts its own threads to multipart something that arrives in
one response.
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    failures: list[tuple[str, Exception]] = []
    landed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_get, store, key, path): key for key, path in targets.items()}
        for done in as_completed(futures):
            try:
                done.result()
            except Exception as error:
                failures.append((futures[done], error))
                continue

            landed += 1
            if landed % _EVERY == 0:
                log.info("staged %d of %d", landed, len(keys))

    if failures:
        named = ", ".join(f"{key} ({error})" for key, error in failures[:_REPORTED])
        raise RuntimeError(
            f"{len(failures)} of {len(keys)} {what} could not be staged, so anything read from "
            f"this tree would be short: {named}"
        )

    log.info("staged %d %s", landed, what)
    return landed


def fetch_prefix(store: ObjectStore, prefix: str, root: Path, what: str = "objects") -> int:
    """Copy everything under `prefix` into `root`, and return how many landed.

    Zero is a count rather than an error. Whether an empty prefix is a fault is
    the caller's question: a run's purchases are empty at cycle 0, and its eval
    labels never are.
    """
    return fetch(store, sorted(store.keys(prefix)), root, prefix, what)


def _get(store: ObjectStore, key: str, path: Path) -> None:
    path.write_bytes(store.get(key))

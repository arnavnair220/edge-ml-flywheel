"""Copying the label documents a partition names onto local disk.

The one place in the data jobs where the package makes the copy rather than
printing keys for the shell to copy, and it is here for a measured reason: the
buildspec's `xargs -P 32 -I {} aws s3 cp` ran one `aws s3 cp` per document, and
13,000 of them timed the build out at thirty minutes without finishing. Each of
those is a Python interpreter and a botocore import before a request leaves, so
on the two vCPUs a small CodeBuild container has, the work is process startup and
the parallelism buys nothing. The same 13,000 GETs from one process are request
latency, which is what a thread pool is for.

The division the buildspec keeps everywhere else survives the move: the keys are
still named by `cohort_labels.keys`, and this module fetches a list it is handed
rather than a prefix it walks. `stage-labels` derives that list itself instead of
reading one from stdin, so there is no argument anyone can pass that makes this
fetch a `pool` label -- the refusal stays where `cohort_labels.label_key` puts
it, before a path exists.

**The tree is the bucket, laid out locally**, which is `ingest.stage`'s rule and
what lets `cohort_labels.read` find a document at `stage_dir / <key>` with no
second path convention to keep in step.
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Final

import boto3
from botocore.config import Config

from edge_ml_flywheel.conventions import Buckets

log = logging.getLogger(__name__)

# Threads, and the client's connection pool with them. A pool narrower than the
# thread count serialises the threads behind connections and logs a warning per
# wait, which is the failure this constant exists to keep impossible.
#
# 32 because that is what the buildspec asked `xargs` for, and the bottleneck it
# hit was never this number.
WORKERS: Final = 32

# Retries inside botocore rather than a loop here. These are 13,000 small GETs
# against one prefix, which is exactly the shape S3 answers with a 503 Slow Down,
# and a build that fails on one throttle after staging 12,000 documents has
# thrown away four minutes for a condition the SDK handles.
_ATTEMPTS: Final = 5

# How often the progress line is printed. Silence for two minutes in a build log
# is indistinguishable from the hang this module was written to fix.
_EVERY: Final = 1_000

# Failures named in the error. The count is the number that matters; the first
# few keys are what says whether it was one bad document or the whole prefix.
_REPORTED: Final = 5


def client(aws: boto3.Session, workers: int = WORKERS) -> Any:
    """An S3 client sized for the pool that will share it.

    One client across every thread rather than one per thread: botocore clients
    are safe to call concurrently once built, and building 32 of them pays the
    import and credential resolution this module exists to pay once.
    """
    return aws.client(
        "s3",
        config=Config(
            max_pool_connections=workers,
            retries={"max_attempts": _ATTEMPTS, "mode": "standard"},
        ),
    )


def data_bucket(aws: boto3.Session) -> str:
    """The data bucket, composed from the caller's account.

    The same arrangement `run.control.cycle_machine_arn` uses, and for its
    reason: a `--bucket` flag would be a second answer to a question
    `conventions.Buckets` already answers, and the wrong answer is a job that
    reads a bucket nobody else in the project writes.
    """
    account = str(aws.client("sts").get_caller_identity()["Account"])
    return Buckets.for_account(account).data


def stage(s3: Any, bucket: str, keys: Sequence[str], stage_dir: Path) -> int:
    """Copy every key to `stage_dir / key`, and return how many landed.

    Every key is attempted before anything is raised. A partial stage is a failed
    build either way, and the count of what failed is the difference between "one
    document is missing from the archive" and "this role cannot read the label
    prefix at all" -- which are the two things that go wrong here and want
    different fixes.
    """
    directories = {(stage_dir / key).parent for key in keys}
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    log.info("staging %d label documents into %d directories", len(keys), len(directories))

    failures: list[tuple[str, Exception]] = []
    landed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_get, s3, bucket, key, stage_dir): key for key in keys}
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
            f"{len(failures)} of {len(keys)} label documents could not be staged, so the cohort "
            f"files written from this tree would be short: {named}"
        )

    log.info("staged %d label documents", landed)
    return landed


def _get(s3: Any, bucket: str, key: str, stage_dir: Path) -> None:
    """One document, whole, in memory.

    `get_object` rather than `download_file`: these are a few kilobytes each, and
    the transfer manager behind `download_file` spins up its own threads to
    multipart something that arrives in one response.
    """
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    (stage_dir / key).write_bytes(body)

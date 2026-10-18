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

from collections.abc import Sequence
from pathlib import Path

from edge_ml_flywheel.storage import DATA_BUCKET, ObjectStore, bucket_name, tree
from edge_ml_flywheel.storage.s3 import S3Store

# 32 because that is what the buildspec asked `xargs` for, and the bottleneck it
# hit was never this number.
WORKERS = tree.WORKERS


def data_store(workers: int = WORKERS) -> ObjectStore:
    """The data bucket, behind a store sized for the pool that will share it.

    One store across every thread rather than one per thread: the client behind
    it is safe to call concurrently once built, and building 32 of them pays the
    import and credential resolution this module exists to pay once.

    The bucket is named by configuration rather than by a `--bucket` flag, for
    the reason a flag would be wrong: it would be a second answer to a question
    the deployment already answers, and the wrong answer is a job that reads a
    bucket nobody else in the project writes.
    """
    return S3Store.connect(bucket_name(DATA_BUCKET), pool=workers)


def stage(store: ObjectStore, keys: Sequence[str], stage_dir: Path) -> int:
    """Copy every key to `stage_dir / key`, and return how many landed.

    `storage.tree.fetch`, which attempts every key before raising: one document
    missing from the archive and a role that cannot read the label prefix at all
    are the two things that go wrong here, and the failure count tells them
    apart.
    """
    return tree.fetch(store, keys, stage_dir, what="label documents")

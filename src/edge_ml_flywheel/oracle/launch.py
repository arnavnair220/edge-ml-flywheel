"""The caller's side of a purchase: assemble the oracle, buy the batch, file it.

`selection.launch`'s shape for the step after it, and it reuses
`training.launch`'s account helpers rather than restating them -- the session,
the bucket names, the run registration and the S3 primitives are the same facts
about the same account.

**This is the only module in the project that runs the oracle against S3.**
`purchase` and `cohorts` were written pure and local-path, which is what let the
charge be tested against a real transaction engine without a bucket; this is
where those two meet a live account, and it is deliberately the only place. The
gate, the ledger and the label read all happen inside `Oracle.purchase` exactly
as they do in the tests.

**The batch is read out of the ranking, not passed in.** A purchase names a
thousand image IDs and every one of them is chargeable, so the list has to come
from the record of how they were chosen rather than from an execution input a
retry could carry differently. That is also what makes the idempotency key
stable: the same cycle re-run reads the same file and produces the same digest.

**The labels are written after the charge and never before.** `Oracle.purchase`
returns the receipt beside them and this files them, which is the split that
keeps "charge, then serve" literal -- a batch nobody can pay for reaches this
function as an exception rather than as a file.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import boto3

from edge_ml_flywheel.conventions import (
    Cycle,
    ImageId,
    RunId,
    RunRegistration,
    assignments_prefix,
    purchase_labels_key,
    selection_ranking_key,
    uri,
)
from edge_ml_flywheel.oracle import labels as sold
from edge_ml_flywheel.oracle import purchase as buying
from edge_ml_flywheel.oracle.cohorts import Cohorts
from edge_ml_flywheel.selection import ranking
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Purchased:
    """What one cycle bought, for the state machine to act on.

    `pool_remaining` is what the loop branches on, and it is the ranked pool minus
    this batch rather than a fresh count over the partition. The ranking names
    exactly the images this cycle had left to buy -- the partition's pool minus
    everything bought before it -- so subtracting the batch is the number by
    definition, and re-deriving it from the assignments would be the same answer
    reached by a route that can disagree.

    `replayed` distinguishes a fresh charge from a retry of one that already
    happened. The labels are the same either way, which is the point of a replay,
    so nothing downstream branches on it -- but a cost report counting distinct
    purchases reads it, and so does anyone asking why a cycle shows two attempts.
    """

    run_id: RunId
    cycle: Cycle
    images: int
    pool_remaining: int
    replayed: bool


def purchase(aws: boto3.Session, run_id: RunId, cycle: Cycle) -> Purchased:
    """Buy this cycle's batch and write its boxes under the run.

    The order is the order the failures are worth having in. The ranking is read
    first, because without it there is no batch and nothing else matters; the
    cohort index second, because it is what the gate is; and the charge last, so
    that everything which could refuse for a reason other than money has already
    run.

    Nothing here checks whether the run has already bought these images. The
    scoring manifest the ranking was drawn from is the pool minus every previous
    purchase, so a repeat cannot be proposed; and `training.labels.collect`
    refuses a duplicate across purchase files at the next cycle's prepare. A third
    check would be a third place to keep one rule.
    """
    run = base.registration(aws, run_id)
    buckets = base.buckets(aws)

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        batch = _batch(aws, buckets.artifacts, run_id, cycle, work)
        cohorts = _cohorts(aws, buckets.data, run, work)

        oracle = buying.Oracle(
            resource=aws.resource("dynamodb"),
            run=run,
            cohorts=cohorts,
            fetch=sold.s3_fetch(aws.client("s3"), buckets.data, cohorts),
        )
        receipt, labels = oracle.purchase(cycle, batch.images)

        key = purchase_labels_key(run_id, cycle)
        local = work / Path(key).name
        sold.write_parquet(labels, local)
        aws.client("s3").upload_file(str(local), buckets.data, key)

    log.info(
        "%s cycle %d: %d labels%s, filed at %s",
        run_id,
        cycle,
        receipt.labels_spent,
        " (replayed)" if receipt.replayed else "",
        uri(buckets.data, key),
    )
    return Purchased(
        run_id=run_id,
        cycle=cycle,
        images=receipt.images,
        pool_remaining=batch.ranked - receipt.images,
        replayed=receipt.replayed,
    )


@dataclass(frozen=True, slots=True)
class _Batch:
    """The selected images, and how many were ranked to choose them from."""

    images: tuple[ImageId, ...]
    ranked: int


def _batch(aws: boto3.Session, artifacts: str, run_id: RunId, cycle: Cycle, work: Path) -> _Batch:
    """This cycle's batch, read out of the ranking that chose it."""
    key = selection_ranking_key(run_id, cycle)
    if not base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} does not exist, so nothing says what this cycle chose to buy. "
            f"Run selection for this cycle."
        )

    local = work / Path(key).name
    aws.client("s3").download_file(artifacts, key, str(local))

    rows = ranking.read(local)
    batch = ranking.selected(rows)
    log.info("the ranking offers %d of %d images for purchase", len(batch), len(rows))
    return _Batch(images=batch, ranked=len(rows))


def _cohorts(aws: boto3.Session, data: str, run: RunRegistration, work: Path) -> Cohorts:
    """The partition's assignments, downloaded and indexed.

    The same index `scoring.launch.partition_cohorts` reads, from the same files,
    for `oracle.cohorts.Cohorts`' reason: a cohort is a fact about a partition, so
    two runs over one version are answered identically and there is nothing
    per-run to keep in step.
    """
    prefix = assignments_prefix(run.partition_version)
    found = base.download_prefix(aws, data, prefix, work / prefix)
    if not found:
        raise SystemExit(
            f"partition v{run.partition_version} has no assignments in {data}, so there is no "
            f"cohort index and nothing that can say an image is purchasable."
        )
    return Cohorts.read(work, run.partition_version)

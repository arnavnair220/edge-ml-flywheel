"""The caller's side of an evaluation job: resolve the request, or start it.

`scoring.launch`'s shape for the step after it, and it reuses `training.launch`'s
account helpers rather than restating them -- the session, the bucket names, the
run registration and the S3 primitives are the same facts about the same account.

**There is no preparation step here, and the absence is the point.** The other
two jobs are handed documents a cycle had to write first; this one is handed what
they produced. The eval manifest names what was scored, the detections are the
scoring job's output, and the eval boxes were frozen at partition time -- so
there is nothing for a prepare to compute, and a step that wrote something would
be writing a fourth copy of a set three existing objects already fix.

**Nothing here reads a label.** The eval boxes are a channel on the job, read
inside the container by the role that runs it. This module resolves names and
checks that objects exist, which is a listing; the control plane it runs in is
denied that prefix outright.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    PROJECT,
    Cohort,
    ModelVersion,
    Seed,
    detections_prefix,
    eval_matches_key,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
    training_code_key,
    uri,
)
from edge_ml_flywheel.evaluation import job
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)

# The role an evaluation job runs as, and deliberately not the scoring one. This
# is the single ARN on `eval_label_reader_arns` -- see `infra/evaluation.tf` for
# what the separation buys.
ROLE: Final = "evaluation"


def request(
    aws: boto3.Session,
    version: ModelVersion,
    seeds: tuple[Seed, ...],
    compute: job.Compute,
    champion: ModelVersion | None = None,
) -> dict[str, Any]:
    """One cycle's `CreateProcessingJob` request, resolved against the account.

    Separate from `start` for `scoring.launch.request`'s reason: the state
    machine wants the request handed back so `createProcessingJob.sync` makes the
    call and owns the wait.

    Every input is checked here rather than left to SageMaker. A missing
    detections prefix fails the job minutes in as a download error naming a key,
    which is the least diagnosable place to find out that a seed never scored --
    and for the champion's cache specifically it is the difference between "the
    champion was never evaluated" and "the comparison is broken".

    The eval boxes are the one channel not checked, because this caller cannot
    read them: the bucket policy admits one principal to that prefix and it is
    the job, not the control plane. Their absence is a partition that was never
    drawn, which fails every earlier step of the cycle first.
    """
    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)
    run = base.registration(aws, run_id)

    artifacts = base.buckets(aws).artifacts
    required = [
        training_code_key(run_id, cycle),
        scoring_manifest_key(run_id, cycle, Cohort.EVAL),
    ]
    for key in required:
        if not base.exists(aws, artifacts, key):
            raise SystemExit(
                f"{uri(artifacts, key)} does not exist, so there is nothing to evaluate. Run "
                f"prepare and score for this cycle."
            )

    # A prefix rather than an object, so the check is a listing of what the
    # scoring job uploaded rather than a guess at its part numbering.
    for seed in sorted(seeds):
        prefix = detections_prefix(version, seed, Cohort.EVAL)
        if not _any_object(aws, artifacts, prefix):
            raise SystemExit(
                f"{uri(artifacts, prefix)} holds no detections, so seed {seed} was never scored "
                f"over {Cohort.EVAL.value}. Run the scoring job for this seed."
            )

    if champion is not None:
        for seed in sorted(seeds):
            key = eval_matches_key(champion, seed)
            if not base.exists(aws, artifacts, key):
                raise SystemExit(
                    f"{uri(artifacts, key)} does not exist, so champion {champion} has no cached "
                    f"matches for seed {seed} and there is nothing to pair this cycle against."
                )

    target = job.Target(
        buckets=base.buckets(aws),
        region=str(aws.region_name),
        role_arn=base.role_arn(aws, ROLE),
        version=version,
        seeds=tuple(sorted(seeds)),
        partition_version=run.partition_version,
        champion=champion,
        tags={"project": PROJECT, "recipe_version": str(run.recipe_version)},
    )

    return job.processing_job(target=target, compute=compute, attempt=datetime.now(UTC))


def _any_object(aws: boto3.Session, bucket: str, prefix: str) -> bool:
    """Whether a prefix holds anything at all.

    `training.launch.exists` answers the question for one key and this one for a
    directory, which is what a prefix channel is pointed at.
    """
    response = aws.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def start(
    aws: boto3.Session,
    version: ModelVersion,
    seeds: tuple[Seed, ...],
    compute: job.Compute,
    champion: ModelVersion | None = None,
) -> str:
    """Create the cycle's evaluation job and return its name."""
    created = request(aws, version, seeds, compute, champion)

    aws.client("sagemaker").create_processing_job(**created)
    log.info("started %s on %s", created["ProcessingJobName"], compute.instance_type)
    return str(created["ProcessingJobName"])

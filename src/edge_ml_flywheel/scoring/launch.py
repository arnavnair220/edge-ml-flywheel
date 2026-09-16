"""The caller's side of a scoring job: prepare the manifests, resolve the request.

`training.launch`'s two halves for the other job a cycle runs, and it reuses that
module's account helpers rather than restating them -- the session, the bucket
names, the run registration and the S3 primitives are the same facts about the
same account, and a second copy is a second thing to keep true.

**Preparation is once per cycle; the request is once per seed.** Every seed of a
cycle scores the same images, so the two manifests are written before the Map
rather than inside it -- the same split `training.launch.prepare` makes, and for
the same reason: seeds of one challenger must differ in the seed and in nothing
else.

**Nothing here reads a label.** The purchase files are opened, and only to
recover which image IDs the run has already bought so they can be subtracted from
the pool. The boxes in them are decoded on the way past and thrown away, which is
`training.launch.labeled_set`'s arrangement and is why the control plane's grant
stops at the labels a run already owns.
"""

import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    PROJECT,
    SCORED_COHORTS,
    Cohort,
    Cycle,
    ImageId,
    ModelArtifact,
    ModelVersion,
    RunId,
    RunRegistration,
    Seed,
    assignments_prefix,
    model_artifact_key,
    model_version_cycle,
    model_version_run_id,
    purchases_run_prefix,
    scoring_manifest_key,
    training_code_key,
    uri,
)
from edge_ml_flywheel.oracle.cohorts import Cohorts
from edge_ml_flywheel.scoring import cohorts as sets
from edge_ml_flywheel.scoring import job
from edge_ml_flywheel.training import images, labels
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)

# The role a scoring job runs as, and deliberately not the training one. The
# suffix is here rather than in `base.role_arn` because this is the module that
# means it -- see `infra/scoring.tf` for what the difference buys.
ROLE: Final = "scoring"


def purchased(aws: boto3.Session, run: RunRegistration, work: Path) -> frozenset[ImageId]:
    """Every image this run has already bought, by ID.

    Read through `training.labels`, which is the reader of record for a purchase
    file, rather than by pulling one column out of the parquet here. It costs a
    box decode per image that is thrown away immediately, and it buys the
    duplicate check that reader performs -- a repeated ID across two cycles means
    the oracle sold one image twice, which is a ledger fault this is the first
    thing downstream to notice.

    Empty at cycle 0, and the emptiness is ordinary: nothing has been bought yet,
    so the pool to score is the whole pool.
    """
    root = work / "purchases"
    found = base.download_prefix(
        aws, base.buckets(aws).data, purchases_run_prefix(run.run_id), root
    )
    if not found:
        log.info("%s has bought nothing yet, so the pool is whole", run.run_id)
        return frozenset()
    return frozenset(labels.collect([root]))


def partition_cohorts(aws: boto3.Session, run: RunRegistration, work: Path) -> Cohorts:
    """The partition's assignments, downloaded and indexed.

    The same index the purchase gate reads, from the same files, for
    `oracle.cohorts.Cohorts`' reason: a cohort is a fact about a partition, so
    two runs over one version are answered identically and there is nothing
    per-run to keep in step.
    """
    prefix = assignments_prefix(run.partition_version)
    found = base.download_prefix(aws, base.buckets(aws).data, prefix, work / prefix)
    if not found:
        raise SystemExit(
            f"partition v{run.partition_version} has no assignments in "
            f"{base.buckets(aws).data}. The partitioner writes them; run it before scoring."
        )
    return Cohorts.read(work, run.partition_version)


@dataclass(frozen=True, slots=True)
class Preparation:
    """What a cycle's scoring is prepared from, as against which cycle.

    `training.launch.Preparation` without `code`: the source archive is staged
    once per cycle by the training preparation and both jobs unpack that same
    object, so there is nothing for this step to package.

    `max_images` caps each cohort independently and is 0 for all of them. It is
    the lever that makes a skeleton run cost cents, and it is a parameter of the
    cycle rather than of the job for `training.launch.Preparation`'s reason: a
    short run is a short manifest, and the manifest is the record of what was
    scored.

    `replace` permits rather than supplies, also for that class's reason. A
    manifest rewritten after a seed has scored against it leaves the record
    describing a set nothing was ranked over.
    """

    max_images: int = 0
    replace: bool = False


def prepare(
    aws: boto3.Session,
    run_id: RunId,
    cycle: Cycle,
    preparation: Preparation,
) -> Mapping[Cohort, int]:
    """Write this cycle's scoring manifests and return how many images each names.

    Both cohorts at once, and once per cycle for `training.launch.prepare`'s
    reason. The refusal to overwrite comes first, before anything is downloaded,
    so a cycle that has already been prepared costs a listing rather than a copy
    of the assignments.
    """
    run = base.registration(aws, run_id)
    artifacts = base.buckets(aws).artifacts

    if not preparation.replace:
        for cohort in sorted(SCORED_COHORTS):
            key = scoring_manifest_key(run_id, cycle, cohort)
            if base.exists(aws, artifacts, key):
                raise SystemExit(
                    f"{uri(artifacts, key)} already exists, and it is the record of what this "
                    f"cycle scored. Pass --replace only if no seed has scored against it."
                )

    # The scratch directory belongs to this function rather than to its caller,
    # for `training.launch.prepare`'s reason: the purchase files are downloaded
    # to read IDs out of and are wanted nowhere else, so nothing outside this
    # holds a path to a copy of them after it returns.
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        index = partition_cohorts(aws, run, work)
        spent = purchased(aws, run, work)
        sets.check_purchases(index, spent)

        named: dict[Cohort, int] = {}
        for cohort in sorted(SCORED_COHORTS):
            image_ids = images.capped(sets.to_score(index, cohort, spent), preparation.max_images)
            local = work / f"{cohort.value}.manifest"
            named[cohort] = images.write(
                local, base.buckets(aws).data, image_ids, COHORT_SPLIT[cohort]
            )

            key = scoring_manifest_key(run_id, cycle, cohort)
            aws.client("s3").upload_file(str(local), artifacts, key)
            log.info("wrote %s", uri(artifacts, key))

    return named


def request(
    aws: boto3.Session,
    version: ModelVersion,
    seed: Seed,
    scoring: job.Scoring,
    compute: job.Compute,
) -> dict[str, Any]:
    """One seed's `CreateProcessingJob` request, resolved against the account.

    Separate from `start` for `training.launch.request`'s reason: the state
    machine wants the request handed back so `createProcessingJob.sync` makes the
    call and owns the wait.

    Every input the job reads is checked here rather than left to SageMaker. A
    missing manifest, archive or checkpoint fails the job minutes in as a
    download error naming a key, which is the least diagnosable place to find out
    that a step upstream did not run -- and for the checkpoint specifically it is
    the difference between "training did not write its model" and "scoring is
    broken".
    """
    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)
    run = base.registration(aws, run_id)

    artifacts = base.buckets(aws).artifacts
    required = [
        training_code_key(run_id, cycle),
        model_artifact_key(version, seed, ModelArtifact.TORCH),
        *(scoring_manifest_key(run_id, cycle, cohort) for cohort in sorted(SCORED_COHORTS)),
    ]
    for key in required:
        if not base.exists(aws, artifacts, key):
            raise SystemExit(
                f"{uri(artifacts, key)} does not exist, so there is nothing to score. Run prepare "
                f"and train for this cycle."
            )

    target = job.Target(
        buckets=base.buckets(aws),
        region=str(aws.region_name),
        role_arn=base.role_arn(aws, ROLE),
        version=version,
        seed=seed,
        tags={"project": PROJECT, "recipe_version": str(run.recipe_version)},
    )

    return job.processing_job(
        target=target,
        scoring=scoring,
        compute=compute,
        attempt=datetime.now(UTC),
    )


def start(
    aws: boto3.Session,
    version: ModelVersion,
    seed: Seed,
    scoring: job.Scoring,
    compute: job.Compute,
) -> str:
    """Create one seed's scoring job and return its name."""
    created = request(aws, version, seed, scoring, compute)

    aws.client("sagemaker").create_processing_job(**created)
    log.info("started %s on %s", created["ProcessingJobName"], compute.instance_type)
    return str(created["ProcessingJobName"])

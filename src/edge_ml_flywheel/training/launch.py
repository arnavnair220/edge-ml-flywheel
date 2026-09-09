"""The operator's side of a training job: prepare, package, start, report.

Four steps that Step Functions will own one at a time, written here first
because the walking skeleton has to run before the state machine that sequences
it exists. What survives that transition is `job.training_job`, which is the
definition; this module is the caller, and the ASL becomes a second caller of the
same function rather than a second definition.

**Nothing here decides anything about a run.** The partition and class set come
from the run registration, not from flags: they are preconditions of the whole
comparison (design section 5), and a flag is how a job comes to train under a
class set its run never declared. The bucket names come from the account ID, the
way `conventions.Buckets` builds them, so there is no configuration to keep in
step with Terraform.
"""

import gzip
import io
import logging
import tarfile
import tempfile
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    PROJECT,
    Buckets,
    Cohort,
    Cycle,
    ImageId,
    ModelVersion,
    RunId,
    RunRegistration,
    Seed,
    base_weights_key,
    cohort_labels_prefix,
    model_version_cycle,
    model_version_run_id,
    purchases_run_prefix,
    training_code_key,
    training_manifest_key,
    uri,
)
from edge_ml_flywheel.run import registration as reg
from edge_ml_flywheel.training import images, job, labels

log = logging.getLogger(__name__)

# What goes into the source archive. The package itself, plus the two files at
# its root that SageMaker's script mode requires there.
_CODE_ROOT: Final = "edge_ml_flywheel"
_CONTAINER_DIR: Final = "container"
_REQUIREMENTS: Final = "requirements.txt"

# Excluded from the archive: compiled bytecode is a function of an interpreter
# that is not the container's, and shipping it invites a stale `.pyc` shadowing
# a module that changed.
_EXCLUDED: Final = ("__pycache__", ".pyc")

# A fixed timestamp in every tar header, so the archive is a function of the tree
# and not of when it was built. Two builds of one commit then have one digest,
# which is what makes the archive under the cycle prefix evidence of what ran.
_EPOCH: Final = 0

_POLL_SECONDS: Final = 30

# Where a training job stops. `Stopped` is in here because a managed spot job
# that ran out of `MaxWaitTimeInSeconds` waiting for capacity ends this way, and
# it is a result rather than a hang.
_TERMINAL: Final = frozenset({"Completed", "Failed", "Stopped"})


def session(profile: str | None = None) -> boto3.Session:
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


def account_id(aws: boto3.Session) -> str:
    return str(aws.client("sts").get_caller_identity()["Account"])


def buckets(aws: boto3.Session) -> Buckets:
    return Buckets.for_account(account_id(aws))


def role_arn(aws: boto3.Session) -> str:
    """The training role, composed from the account and the name Terraform gives
    it. A `terraform output` would be the same string behind a second tool."""
    return f"arn:aws:iam::{account_id(aws)}:role/{PROJECT}-training"


def registration(aws: boto3.Session, run_id: RunId) -> RunRegistration:
    entry = reg.read(reg.runs_table(aws.resource("dynamodb")), run_id)
    if entry is None:
        raise SystemExit(
            f"{run_id} was never registered, so there is no partition or class set to train "
            f"under. Register the run first."
        )
    return entry


def repo_root() -> Path:
    """The checkout this module was imported from.

    `src/edge_ml_flywheel/training/launch.py`, so three levels up is `src/` and
    four is the repository. The launcher is an operator tool run from a clone;
    from an installed wheel there is no `container/` directory and this fails
    with the message below rather than building an archive with no entry point.
    """
    root = Path(__file__).resolve().parents[3]
    if not (root / _CONTAINER_DIR / job.ENTRY_POINT).is_file():
        raise SystemExit(
            f"cannot find {_CONTAINER_DIR}/{job.ENTRY_POINT} above {__file__}. Run this from a "
            f"checkout of the repository."
        )
    return root


def _members(root: Path) -> Iterator[tuple[Path, str]]:
    """Every file in the archive, as (source, name inside the archive), sorted."""
    package = root / "src" / _CODE_ROOT
    for path in sorted(package.rglob("*")):
        if path.is_file() and not any(marker in str(path) for marker in _EXCLUDED):
            yield path, f"{_CODE_ROOT}/{path.relative_to(package).as_posix()}"

    container = root / _CONTAINER_DIR
    for name in (job.ENTRY_POINT, _REQUIREMENTS):
        yield container / name, name


def archive(root: Path) -> bytes:
    """The source archive, byte-identical for a given tree.

    The gzip layer is opened explicitly rather than through `w:gz`, because a
    gzip header carries its own timestamp: left to the default, two archives of
    one commit differ in eight bytes and the digest stops meaning anything.
    """
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=_EPOCH) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for source, name in _members(root):
            info = tar.gettarinfo(str(source), arcname=name)
            info.mtime = _EPOCH
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                tar.addfile(info, handle)
    return buffer.getvalue()


def package(aws: boto3.Session, run_id: RunId, cycle: Cycle) -> str:
    """Upload the tree this cycle's five seeds run, and return its key."""
    key = training_code_key(run_id, cycle)
    artifacts = buckets(aws).artifacts
    payload = archive(repo_root())

    aws.client("s3").put_object(Bucket=artifacts, Key=key, Body=payload)
    log.info("packaged %d KB of source to %s", len(payload) // 1024, uri(artifacts, key))
    return key


def _download_prefix(aws: boto3.Session, bucket: str, prefix: str, root: Path) -> int:
    """Copy a prefix to local disk, keeping the key structure below it."""
    client = aws.client("s3")
    copied = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", ()):
            key = entry["Key"]
            if key.endswith("/"):
                continue
            destination = root / key.removeprefix(prefix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(destination))
            copied += 1
    return copied


def labeled_set(aws: boto3.Session, run: RunRegistration, work: Path) -> Sequence[ImageId]:
    """Every image this run has labels for: the bootstrap cohort and its purchases.

    The same two sources the training channels carry, read here through the same
    function the container uses. That is deliberate -- the manifest has to name
    exactly the set the job will find labels for, and the cheapest way to
    guarantee it is for both sides to compute it the same way.
    """
    data = buckets(aws).data
    bootstrap = work / Cohort.BOOTSTRAP.value
    purchases = work / "purchases"

    found = _download_prefix(
        aws, data, cohort_labels_prefix(run.partition_version, Cohort.BOOTSTRAP), bootstrap
    )
    if not found:
        raise SystemExit(
            f"partition v{run.partition_version} has no bootstrap labels in {data}. The "
            f"partitioner writes them; run it before training."
        )
    _download_prefix(aws, data, purchases_run_prefix(run.run_id), purchases)

    return sorted(labels.collect([bootstrap, purchases]))


def prepare(
    aws: boto3.Session,
    run_id: RunId,
    cycle: Cycle,
    max_images: int = 0,
    replace: bool = False,
) -> int:
    """Write this cycle's two inputs -- the image manifest and the source archive
    -- and return how many images the manifest names.

    Both at once, and once per cycle rather than once per seed, because the five
    seeds of a challenger have to differ in the seed and in nothing else. A
    package step on each launch would let seed 4 run a tree seed 1 never saw,
    which is not a comparison between five seeds of one model.

    Refuses to overwrite what is already there, because the manifest is the
    record of what the challenger trained on: replacing it after a seed has run
    leaves the record describing a set no model was trained on. `replace` is for
    iterating on the skeleton, where the cycle is being re-run on purpose.
    """
    run = registration(aws, run_id)
    artifacts = buckets(aws).artifacts
    key = training_manifest_key(run_id, cycle)

    if not replace and _exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} already exists, and it is the record of what this cycle "
            f"trained on. Pass --replace only if no seed has run against it."
        )

    # The labels are downloaded to read the image IDs out of them and are wanted
    # nowhere else, so the scratch directory belongs to this function rather than
    # to its caller: nothing outside it can hold a path to a copy of the labeled
    # set after this returns.
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        image_ids = images.capped(labeled_set(aws, run, work), max_images)
        local = work / "images.manifest"
        named = images.write(local, buckets(aws).data, image_ids)

        aws.client("s3").upload_file(str(local), artifacts, key)
        log.info("wrote %s", uri(artifacts, key))

    package(aws, run_id, cycle)
    return named


def stage_base(aws: boto3.Session, weights: Path) -> str:
    """Put the COCO base in the bucket, once, for every job to start from."""
    key = base_weights_key(weights.name)
    artifacts = buckets(aws).artifacts
    aws.client("s3").upload_file(str(weights), artifacts, key)
    log.info("staged %s", uri(artifacts, key))
    return key


def _exists(aws: boto3.Session, bucket: str, key: str) -> bool:
    client = aws.client("s3")
    response = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(entry["Key"] == key for entry in response.get("Contents", ()))


def start(
    aws: boto3.Session,
    version: ModelVersion,
    seed: Seed,
    recipe: job.Recipe,
    compute: job.Compute,
) -> str:
    """Create one seed's training job and return its name.

    Takes the model version rather than a run and a cycle, for the reason every
    key builder in `conventions` does: they are inside it, and passing all three
    is how a job comes to write its artifacts under a cycle it did not train.

    Both of the cycle's inputs are checked here rather than left to SageMaker. A
    missing manifest or archive fails the job several minutes in, as a download
    error naming a key, which is the least diagnosable place to find out that
    `prepare` was never run for this cycle.
    """
    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)
    run = registration(aws, run_id)

    artifacts = buckets(aws).artifacts
    for key in (training_manifest_key(run_id, cycle), training_code_key(run_id, cycle)):
        if not _exists(aws, artifacts, key):
            raise SystemExit(f"{uri(artifacts, key)} does not exist. Run prepare for this cycle.")

    target = job.Target(
        buckets=buckets(aws),
        region=str(aws.region_name),
        role_arn=role_arn(aws),
        version=version,
        seed=seed,
        partition_version=run.partition_version,
        class_set_version=run.class_set_version,
        tags={"project": PROJECT, "recipe_version": str(run.recipe_version)},
    )

    request = job.training_job(
        target=target,
        recipe=recipe,
        compute=compute,
        attempt=datetime.now(UTC),
    )

    aws.client("sagemaker").create_training_job(**request)
    log.info("started %s on %s", request["TrainingJobName"], compute.instance_type)
    return str(request["TrainingJobName"])


def wait(aws: boto3.Session, name: str) -> str:
    """Block until the job stops, then report what each phase cost.

    The `Downloading` transition is the number design section 11 asks for: `File`
    mode copies the channel before the first step, the cost is driven by object
    count rather than bytes, and nothing packs the images into shards until this
    reads back in minutes. SageMaker records it and it cannot be recovered from
    the log afterwards, so it is printed here on every job.
    """
    client = aws.client("sagemaker")
    while True:
        described = client.describe_training_job(TrainingJobName=name)
        status = str(described["TrainingJobStatus"])
        if status in _TERMINAL:
            break
        log.info("%s: %s", status, described.get("SecondaryStatus", ""))
        time.sleep(_POLL_SECONDS)

    for transition in described.get("SecondaryStatusTransitions", ()):
        started = transition["StartTime"]
        ended = transition.get("EndTime", started)
        log.info(
            "%s: %.0fs -- %s",
            transition["Status"],
            (ended - started).total_seconds(),
            transition.get("StatusMessage", ""),
        )

    billed = described.get("BillableTimeInSeconds")
    if billed is not None:
        log.info("billed %ds on %s", billed, described["ResourceConfig"]["InstanceType"])
    if status != "Completed":
        raise SystemExit(f"{name} {status.lower()}: {described.get('FailureReason', '')}")

    log.info("%s completed", name)
    return status


def describe(aws: boto3.Session, name: str) -> dict[str, Any]:
    return dict(aws.client("sagemaker").describe_training_job(TrainingJobName=name))

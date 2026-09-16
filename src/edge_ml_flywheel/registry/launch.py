"""The caller's side of a registration: assemble the manifest, or refuse.

`evaluation.launch`'s shape for the step after it, and it reuses
`training.launch`'s account helpers rather than restating them -- the session,
the bucket names, the run registration and the S3 primitives are the same facts
about the same account.

**This step reaches a verdict about nothing.** The gates ran in the evaluation
job and their result is at `conventions.gate_report_key`; what happens here is
that the report is read, a manifest is built from what the cycle actually
produced, and the two are checked against the run. The decision was made
upstream, in a job that could see the labels this one cannot.

**There are two kinds of "no" here and they are not the same.** A gate that
failed is a verdict: the manifest records it, the version is registered as
`Rejected`, and the cycle carries on to selection with its labels bought. A
manifest that disagrees with its run -- a partition or recipe the run never
declared -- is not a verdict at all, and it raises: it means the comparison the
gates ran is not the comparison they reported, so the right outcome is a failed
execution rather than a rejection that implies the model was measured and found
wanting.

**Nothing here reads a label box.** The purchase files are opened for the image
IDs they name, the way `training.launch.labeled_set` opens them, and the count is
what the manifest records. The withheld labels are behind a prefix this role is
denied outright.
"""

import json
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    Cohort,
    ModelArtifact,
    ModelManifest,
    ModelVersion,
    RunId,
    RunRegistration,
    Seed,
    gate_report_key,
    model_artifact_key,
    model_manifest_key,
    model_package_group,
    model_seed_prefix,
    model_version_cycle,
    model_version_run_id,
    purchases_run_prefix,
    sha256sums,
    uri,
)
from edge_ml_flywheel.registry import manifest as document
from edge_ml_flywheel.registry import package
from edge_ml_flywheel.training import job
from edge_ml_flywheel.training import labels as label_files
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)

# Where SageMaker's own copy of the model lands, under the seed prefix. The
# training job is handed this prefix as its `OutputDataConfig` and the service
# appends `<job name>/output/model.tar.gz` under it -- so the tarball's full key
# is a function of a job name this step does not know, and finding it is a
# listing rather than a build. See `training.job.training_job`.
SAGEMAKER_PREFIX: Final = "_sagemaker/"
SAGEMAKER_MODEL: Final = "output/model.tar.gz"


@dataclass(frozen=True, slots=True)
class Registered:
    """What the registration step produced, for the state machine to act on.

    `passed` is the branch, and it is the manifest's own `gates_passed` rather
    than the report's `passed` field. Both say the same thing today; taking it
    off the manifest means the version that gets registered and the branch the
    cycle takes are one value, so a manifest that lost a gate on the way into the
    document cannot approve a model the state machine then treats as rejected.

    `request` is built and not sent, for the reason `training.launch.request`
    gives: the call is the state machine's, in the open.
    """

    version: ModelVersion
    passed: bool
    group: str
    request: dict[str, Any]


def _read_json(aws: boto3.Session, bucket: str, key: str, missing: str) -> dict[str, Any]:
    """One JSON object out of S3, or a refusal naming what should have written it."""
    if not base.exists(aws, bucket, key):
        raise SystemExit(f"{uri(bucket, key)} does not exist. {missing}")
    body = aws.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return dict(json.loads(body))


def artifact_digests(
    aws: boto3.Session, bucket: str, version: ModelVersion, seeds: Sequence[Seed]
) -> dict[Seed, str]:
    """Each seed's digest, read from the file the training job published beside
    the model.

    Read back rather than recomputed, and that is the point of the field: the
    digest was taken where the bytes were produced, inside the job, before the
    upload it then verified. Hashing the object here would produce a digest of
    whatever is in the bucket now, which is the copy rather than the artifact --
    and it is precisely a disagreement between those two that the device's check
    exists to catch.

    `sha256sum -c` format, so the file is one `<digest>  <filename>` line per
    artifact the job published. The line this reads is `ModelArtifact.ONNX`: the
    manifest's digest is the one the device checks before loading, and what the
    device loads is the int8 graph. Selected by name rather than by position,
    because a file whose first line is the answer is a file that silently
    changes answer when a third artifact sorts above it.
    """
    digests: dict[Seed, str] = {}
    for seed in sorted(seeds):
        key = model_artifact_key(version, seed, ModelArtifact.SHA256)
        if not base.exists(aws, bucket, key):
            raise SystemExit(
                f"{uri(bucket, key)} does not exist, so seed {seed} of {version} published no "
                f"digest. A model whose artifact cannot be identified cannot be registered."
            )
        body = aws.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        published = sha256sums(body.decode())
        if ModelArtifact.ONNX not in published:
            raise SystemExit(
                f"{uri(bucket, key)} lists no digest for {ModelArtifact.ONNX.value}, so seed "
                f"{seed} of {version} published no deployable artifact. The manifest's digest is "
                f"what the device verifies, and it cannot name a file that was never exported."
            )
        digests[seed] = published[ModelArtifact.ONNX]

    return digests


def model_data_url(aws: boto3.Session, bucket: str, version: ModelVersion, seed: Seed) -> str:
    """The deployed seed's `model.tar.gz`, found by listing.

    SageMaker writes it under a directory named for the training job, and a cycle
    that retried a failed job has two of those -- so the key is not derivable and
    a second attempt makes the listing ambiguous. The most recent object wins,
    which is the attempt that produced the model this cycle is registering: an
    earlier one failed, and a job that failed uploaded no model.
    """
    prefix = f"{model_seed_prefix(version, seed)}{SAGEMAKER_PREFIX}"
    found = aws.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix)
    tarballs = [
        item for item in found.get("Contents", ()) if str(item["Key"]).endswith(SAGEMAKER_MODEL)
    ]
    if not tarballs:
        raise SystemExit(
            f"{uri(bucket, prefix)} holds no {SAGEMAKER_MODEL}, so the training job for seed "
            f"{seed} of {version} never wrote one. There is nothing to register."
        )

    latest = max(tarballs, key=lambda item: item["LastModified"])
    return uri(bucket, str(latest["Key"]))


def purchased(aws: boto3.Session, run: RunRegistration) -> tuple[int, frozenset[Cohort]]:
    """How many labels this run has bought, and which cohorts trained the model.

    Counted off the purchase files rather than off the ledger, because the files
    are what the training job was handed: the ledger records what was charged for
    and this records what the model was taught from, and the manifest is a
    statement about the model.

    `bootstrap` is always in the set -- a cycle trains on the cumulative labeled
    set and that always includes it -- and `pool` joins it once anything has been
    bought. `eval` cannot appear: `ModelManifest` refuses a manifest naming it,
    and nothing under this prefix is drawn from it.

    Zero and `{bootstrap}` is the honest answer for every cycle until the
    purchase step lands, rather than a placeholder: no run has bought a label, so
    no model has trained on one.
    """
    data = base.buckets(aws).data
    with tempfile.TemporaryDirectory() as scratch:
        purchases = Path(scratch)
        found = base.download_prefix(aws, data, purchases_run_prefix(run.run_id), purchases)
        if not found:
            return 0, frozenset({Cohort.BOOTSTRAP})

        bought = label_files.collect([purchases])

    return len(bought), frozenset({Cohort.BOOTSTRAP, Cohort.POOL})


def register(
    aws: boto3.Session,
    run_id: RunId,
    version: ModelVersion,
    seeds: Sequence[Seed],
) -> Registered:
    """Write this cycle's manifest and build the request that records its verdict.

    The order is the order the failures are worth having in. Everything the
    manifest is made of is read first, so a cycle missing an input fails before
    anything is written; the manifest is validated against the run before it is
    written, because a manifest that disagrees with its run is not a document
    worth filing; and the request is built last, from the manifest that landed
    rather than from the one that was assembled.

    `deployed_seed` is the lowest seed the cycle trained, which is seed 1 by
    convention (design section 4.2) and never the best-scoring one -- picking on
    the eval set biases the number the gate reported. `min` rather than a literal,
    so a run that trained some other set still names one seed.

    `run_id` is an argument as well as being inside `version`, and it is checked
    against it. The execution carries the run it claimed a cycle from and the
    version arrives from the training step several states later, so the two are
    separate values by the time they meet here -- which is the only reason
    checking them is a check rather than a restatement.

    **`ModelManifest.disagreements` cannot fire today, and that is worth saying
    plainly.** The versions it compares are read off the run registration a few
    lines above, so both sides of the comparison are one item and the answer is
    always empty. It is called anyway, at the point the design puts it: the check
    becomes live the moment a version reaches this step from the job that ran
    under it rather than from the item the job was configured from, and a
    precondition added later is a precondition added somewhere else.
    """
    if not seeds:
        raise SystemExit(f"{version} was trained at no seed, so there is no artifact to register")
    if model_version_run_id(version) != run_id:
        raise SystemExit(
            f"{version} belongs to run {model_version_run_id(version)}, and this execution is "
            f"running {run_id}. Nothing is registered under a run that did not train it."
        )

    cycle = model_version_cycle(version)
    run = base.registration(aws, run_id)
    artifacts = base.buckets(aws).artifacts

    report = _read_json(
        aws,
        artifacts,
        gate_report_key(run_id, cycle),
        "The evaluation job writes it, and a model is not registered without a verdict.",
    )
    gates = document.gates(report, version)

    digests = artifact_digests(aws, artifacts, version, seeds)
    spent, cohorts = purchased(aws, run)
    deployed = min(seeds)

    built = ModelManifest(
        version=version,
        created_at=datetime.now(UTC),
        # The commit the run was registered at, which is the tree this cycle's
        # jobs ran unless the control plane was redeployed mid-run. It is the one
        # commit available to a function with no checkout; recording the deployed
        # tree's own SHA is a Lambda environment variable set at deploy time, and
        # it lands when there is a second commit in a run to tell apart.
        git_commit=run.git_commit,
        partition_version=run.partition_version,
        recipe_version=run.recipe_version,
        cohorts_trained_on=cohorts,
        labels_spent=spent,
        deployed_seed=deployed,
        artifact_sha256=digests,
        gates=gates,
    )

    disagreements = built.disagreements(run)
    if disagreements:
        raise SystemExit(
            f"{version} contradicts run {run_id} on {list(disagreements)}, so the comparison the "
            f"gates ran is not the comparison they reported. Nothing is registered."
        )

    _write_manifest(aws, artifacts, built)

    request = package.create_model_package(
        manifest=built,
        buckets=base.buckets(aws),
        image=job.image_uri(str(aws.region_name)),
        model_data_url=model_data_url(aws, artifacts, version, deployed),
        tags={"recipe_version": str(run.recipe_version)},
    )

    log.info(
        "%s: %s, %d gate(s), seed %d at %s",
        version,
        package.approval_status(built),
        len(gates),
        deployed,
        digests[deployed],
    )
    return Registered(
        version=version,
        passed=built.gates_passed,
        group=model_package_group(run_id),
        request=request,
    )


def _write_manifest(aws: boto3.Session, bucket: str, built: ModelManifest) -> None:
    """Write the manifest and confirm what landed.

    The read-back is the partitioner's and the training job's arrangement: the
    step does not report a registered model on the strength of a `put_object`
    that returned. It is a stronger check here than a digest comparison would be,
    because what the next cycle needs is not the same bytes but the same
    *manifest* -- so the document is parsed back through `from_document` and
    re-encoded, and a field that does not survive the round trip fails at the
    cycle that wrote it rather than at the cycle that tried to promote it.
    """
    key = model_manifest_key(built.version)
    body = document.to_document(built)

    client = aws.client("s3")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, sort_keys=True).encode(),
        ContentType="application/json",
    )

    landed = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    if document.to_document(document.from_document(landed)) != body:
        raise SystemExit(
            f"the manifest read back from {uri(bucket, key)} is not the one written. The model is "
            f"not registered, because a manifest is the precondition of promoting it."
        )

    log.info("wrote %s", uri(bucket, key))

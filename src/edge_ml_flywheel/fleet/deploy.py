"""The operator's side of a deployment: sample, package, publish, deploy, read.

`training.launch`'s shape for the fleet, and it shares that module's session and
bucket helpers rather than restating them -- the arrangement every other
`launch` in this package uses.

**Nothing here decides whether a rollout continues.** This stages what a device
needs, asks Greengrass to put it there, and reads back what the device said.
The verdict is `gates.canary`, over a `ReplayReport` that `fleet.telemetry`
reduces, and the division is the one design section 7.1 asks for: the judgement
is a pure function, and this is the part that cannot be tested without AWS.

**A rollback is a deployment, so there is no rollback function.** `redeploy`
takes a version and is called with the previous one, which means the path a
rollback takes is the path every cycle has already exercised. A dedicated
rollback would be code first run on the day it is needed.
"""

import io
import json
import logging
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    PROJECT,
    Buckets,
    Cycle,
    ImageId,
    ModelVersion,
    RunId,
    component_address,
    model_component,
    model_version_cycle,
    model_version_run_id,
    parse_image_id,
    parse_model_version,
    replay_code_key,
    replay_manifest_key,
    telemetry_run_prefix,
    uri,
)
from edge_ml_flywheel.fleet import component, telemetry
from edge_ml_flywheel.gates.canary import canary_gate, detections_stand
from edge_ml_flywheel.registry import launch as registry
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)

# The instance Terraform tags as the fleet's one device, spelled here for
# `__main__.THING_GROUP`'s reason: the name is a fact about the account that a
# second tool would otherwise have to be run to discover.
DEVICE_NAME: Final = f"{PROJECT}-device-1"

# Fixed timestamp in every zip entry, for `training.launch.archive`'s reason: a
# ZIP entry carries its own modification time, so two archives of one tree would
# otherwise differ in bytes and Greengrass would hash them differently. DOS time
# has no year before 1980, which is why this is not the epoch the tar uses.
_ZIP_EPOCH: Final = (1980, 1, 1, 0, 0, 0)

# Excluded from the archive, matching `training.launch`: compiled bytecode is a
# function of an interpreter that is not the device's, and a stale `.pyc` would
# shadow a module that changed.
_EXCLUDED: Final = ("__pycache__", ".pyc")

_CODE_ROOT: Final = "edge_ml_flywheel"


# ---------------------------------------------------------------------------
# Staging what the device is given
# ---------------------------------------------------------------------------


def code_archive(package: Path) -> bytes:
    """The package as a ZIP, byte-identical for a given tree.

    ZIP rather than the gzipped tar the jobs get, because `Unarchive` is the only
    unpacking Greengrass does and `"ZIP"` is the only value it takes. The
    determinism is worth the same care it gets there and for a sharper reason:
    this archive is addressed by commit, so two builds of one tree must be one
    object or the second silently replaces a file a live component is pointed at.

    Only the package. There is no entry point beside it -- the recipe runs
    `-m edge_ml_flywheel.fleet.replay`, so the module system finds it -- and the
    three container root files a job needs are files a device never runs.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in _members(package):
            info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Fixed rather than inherited, for the reason the timestamp is: the
            # tree arrives from a git checkout whose modes depend on the platform
            # that cloned it, and a mode that differs is a digest that differs
            # for a tree that does not.
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def _members(package: Path) -> Iterator[tuple[Path, str]]:
    """Every file in the archive, as (source, name inside it), sorted.

    Sorted because a zip's entry order is part of its bytes, and a directory walk
    is ordered by whatever the filesystem says.
    """
    if not package.is_dir():
        raise SystemExit(f"{package} is not the package directory. Run this from a checkout.")
    for source in sorted(package.rglob("*.py")):
        if any(part in str(source) for part in _EXCLUDED):
            continue
        yield source, str(Path(_CODE_ROOT) / source.relative_to(package)).replace("\\", "/")


def checkout_archive() -> bytes:
    """The archive as built from the repository this module was imported from.

    `training.launch.checkout_archive`'s counterpart, and it reuses that module's
    `repo_root` so the checkout layout is stated once for both.
    """
    return code_archive(base.repo_root() / "src" / _CODE_ROOT)


def stage_code(aws: boto3.Session, git_commit: str, code: bytes) -> str:
    """Upload the tree the component runs, and return its key.

    Unconditional. The key is the commit, so re-uploading writes identical bytes
    to the same place -- and a conditional put would be a check bought to avoid
    a few hundred kilobytes.
    """
    key = replay_code_key(git_commit)
    artifacts = base.buckets(aws).artifacts
    aws.client("s3").put_object(Bucket=artifacts, Key=key, Body=code)
    log.info("packaged %d KB of source to %s", len(code) // 1024, uri(artifacts, key))
    return key


def sampled_frames(aws: boto3.Session, run_id: RunId, cycle: Cycle) -> tuple[ImageId, ...]:
    """The pool frames this cycle's device scores, read off what the cycle drew.

    **The draw is not made here.** `scoring.launch.prepare` makes it, early in
    the cycle and before any model exists, because the sample is what the device
    is shipped and because a draw made at deployment time could not be recorded
    under the write-once cycle prefix the rest of the cycle was prepared under.
    What this does is read it back and refuse a deployment whose sample is
    missing.

    The refusal is a real ordering rather than a defensive check. A deployment
    made before `ScorePrepare` has run would otherwise fail on the device, an
    hour later, as a component that started and found no frames.
    """
    key = replay_manifest_key(run_id, cycle)
    artifacts = base.buckets(aws).artifacts
    if not base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} does not exist, so cycle {cycle} has not drawn the frames its "
            f"device would score. Score prepare writes it; run it before deploying."
        )

    document = json.loads(aws.client("s3").get_object(Bucket=artifacts, Key=key)["Body"].read())
    if not isinstance(document, list) or not document:
        raise SystemExit(f"{uri(artifacts, key)} is not a non-empty list of image IDs")

    drawn = tuple(parse_image_id(str(value)) for value in document)
    log.info("cycle %d put %d pool frames in front of the fleet", cycle, len(drawn))
    return drawn


# ---------------------------------------------------------------------------
# Greengrass
# ---------------------------------------------------------------------------


def iot_endpoint(aws: boto3.Session) -> str:
    """The account's ATS data endpoint, which is where a device publishes.

    Resolved here and passed into the recipe rather than discovered on the
    device, for `component.recipe`'s reason. ATS specifically: the legacy
    endpoint is served under a certificate chain Amazon no longer issues.
    """
    return str(aws.client("iot").describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"])


def publish_component(aws: boto3.Session, recipe: dict[str, Any]) -> str:
    """Create the component version and return its ARN.

    Greengrass hashes every artifact the recipe names as part of this call, so a
    key that does not exist fails here -- before a deployment, while the cycle is
    still the thing in front of you -- rather than on a device an hour later.

    A component version is immutable, so publishing one that exists is an error
    rather than an overwrite. That is the property worth having and the reason
    `component_version` carries the run in the *name*: without it, the second run
    to reach cycle 3 would collide with the first, and the collision would be
    permanent.
    """
    response = aws.client("greengrassv2").create_component_version(
        inlineRecipe=json.dumps(recipe).encode()
    )
    arn = str(response["arn"])
    log.info("published %s %s", recipe["ComponentName"], recipe["ComponentVersion"])
    return arn


def redeploy(
    aws: boto3.Session, version: ModelVersion, target_arn: str, task_token: str = ""
) -> str:
    """Deploy one component version to the fleet, and return the deployment ID.

    The whole of both directions. Called with this cycle's version it is a
    rollout; called with the previous one it is the rollback, and the two are the
    same call because they are the same act -- a revision naming a version
    (design section 6).

    `task_token` is what makes a rollout also the start of the cycle's pool pass:
    the device reads it out of its configuration and resumes the waiting
    execution when its detections are durable. A rollback passes none, because
    nothing is waiting on a rollback -- the cycle that was waiting is the one
    that failed.
    """
    request = component.deployment(version, target_arn, task_token=task_token)
    response = aws.client("greengrassv2").create_deployment(**request)
    deployment_id = str(response["deploymentId"])
    log.info("deployment %s puts %s on %s", deployment_id, version, target_arn)
    return deployment_id


def deployed(aws: boto3.Session, run_id: RunId, target_arn: str) -> ModelVersion | None:
    """The model version the fleet's latest deployment names, or `None`.

    Read off the deployment rather than out of a table, which is what makes the
    deployment the record of intent rather than one of two records that can
    disagree. `None` means this run has deployed nothing yet, which is a run's
    first cycle and nothing else.

    **Out of the configuration, not out of the component version.** A component
    version is numbered by the cycle that deployed it, and a cycle that rejected
    its challenger deploys the standing champion -- so `0.5.0` can perfectly well
    carry a model trained in cycle three, and deriving the model from the number
    would report a version that was never on the device. `component.deployment`
    merges the model version into every deployment for exactly this read.

    Only the effective deployment is asked for. A target's history holds every
    revision, and the one before the current is not the one to roll back *to* --
    a rollback is named by the caller, which holds the champion pointer.
    """
    client = aws.client("greengrassv2")
    latest = client.list_deployments(targetArn=target_arn, historyFilter="LATEST_ONLY")
    found = latest.get("deployments", ())
    if not found:
        return None

    detail = client.get_deployment(deploymentId=found[0]["deploymentId"])
    name = model_component(run_id)
    entry = detail.get("components", {}).get(name)
    if entry is None:
        log.info("the effective deployment carries no %s", name)
        return None

    merged = entry.get("configurationUpdate", {}).get("merge")
    if not merged:
        raise SystemExit(
            f"the effective deployment of {name} names no model version. Every deployment this "
            f"project makes merges one; this one was made by something else."
        )
    return parse_model_version(str(json.loads(merged)["version"]))


# ---------------------------------------------------------------------------
# Reading back what the device said
# ---------------------------------------------------------------------------


def records(aws: boto3.Session, run_id: RunId) -> list[dict[str, Any]]:
    """Every telemetry object one run's fleet has produced, parsed.

    The whole run's prefix rather than one version's, because that is what
    `telemetry.report` is documented to take: narrowing belongs in the reducer,
    where a champion's report and a challenger's are built by one rule. At a few
    objects per replay and a handful of cycles this is a listing of tens, so the
    generality costs nothing worth optimizing.

    A malformed object is refused rather than skipped. These are written by a
    rule this project configured from messages this project's own code composed,
    so a document that will not parse means one of those two changed -- and
    dropping it would report a short replay instead.
    """
    client = aws.client("s3")
    telemetry_bucket = base.buckets(aws).telemetry
    prefix = telemetry_run_prefix(run_id)

    found: list[dict[str, Any]] = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=telemetry_bucket, Prefix=prefix
    ):
        for entry in page.get("Contents", ()):
            body = client.get_object(Bucket=telemetry_bucket, Key=entry["Key"])["Body"].read()
            try:
                found.append(json.loads(body))
            except json.JSONDecodeError as broken:
                raise SystemExit(
                    f"{uri(telemetry_bucket, str(entry['Key']))} is not the JSON this project "
                    f"publishes: {broken}"
                ) from None

    log.info("%d telemetry records under %s", len(found), uri(telemetry_bucket, prefix))
    return found


@dataclass(frozen=True, slots=True)
class Verdict:
    """What one device pass decided, for the two decisions that read it.

    `passed` is the canary gate: whether the rollout stands. `ranks` is whether
    the detections may be bought from, which fails only when the digest or the
    completion check did. A slow model wrote a perfectly good ranking, so the two
    fields disagree exactly in that case -- see `gates.canary.detections_stand`.

    `reason` is the gate's, unedited, because a verdict without its reason is
    what this project exists not to record.
    """

    version: ModelVersion
    passed: bool
    ranks: bool
    reason: str


def judge(
    aws: boto3.Session,
    run_id: RunId,
    version: ModelVersion,
    champion: ModelVersion | None = None,
) -> Verdict:
    """Reduce the telemetry, apply the canary gate, and say what may be bought.

    The expected digest comes out of the model manifest rather than from a
    caller, which is what makes the check mean anything: a digest passed in is a
    digest that matches whatever the caller read it from, and the manifest is the
    copy written where the bytes were produced.

    The champion's report is read from the telemetry of the cycle it was deployed
    in rather than measured again, for `gates.canary`'s reason. A champion that
    is also this version -- a cycle that rejected its challenger and redeployed
    the incumbent -- compares against itself, so it is dropped here rather than
    producing a throughput check whose answer is always zero drift.
    """
    manifest = registry.read_manifest(aws, base.buckets(aws).artifacts, version)
    documents = records(aws, run_id)

    try:
        report = telemetry.report(documents, version)
    except telemetry.NoReplayError as missing:
        raise SystemExit(str(missing)) from None

    against = None
    if champion is not None and champion != version:
        against = telemetry.report(documents, champion)

    expected = manifest.artifact_sha256[manifest.deployed_seed]
    verdict = canary_gate(report=report, expected_sha256=expected, champion=against)
    return Verdict(
        version=version,
        passed=verdict.passed,
        ranks=detections_stand(report, expected),
        reason=verdict.reason,
    )


def require_device(aws: boto3.Session) -> str:
    """Refuse a deployment the device cannot act on, and name it if it can.

    The instance is stopped between runs to stay under the budget alarm, and a
    cycle now blocks on it: a deployment to a stopped device is a Greengrass
    deployment that sits `IN_PROGRESS` until the cycle's two-hour wait expires.
    Two hours later the operator learns something an API call answers now.

    Found by tag rather than by an ID in configuration, for
    `component.thing_group_arn`'s reason -- a `terraform output` would be the same
    string behind a second tool that has to be run in the right directory.
    """
    instances = aws.client("ec2").describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [DEVICE_NAME]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    running = [
        instance
        for reservation in instances.get("Reservations", ())
        for instance in reservation.get("Instances", ())
    ]
    if not running:
        raise SystemExit(
            f"no running instance tagged {DEVICE_NAME}. The cycle's pool pass happens on the "
            f"device, so a stopped device is a cycle that cannot proceed. Start it and run again."
        )

    found = str(running[0]["InstanceId"])
    log.info("%s is running as %s", DEVICE_NAME, found)
    return found


def recipe_for(
    aws: boto3.Session, release: component.Release, replay: component.Replay
) -> dict[str, Any]:
    """`component.recipe` with the two account-shaped values filled in.

    A thin seam, and it exists so that the pure builder never takes a session.
    The buckets and the endpoint are the only things in a recipe a checkout
    cannot know, which is what keeps the recipe itself testable.
    """
    return component.recipe(
        release=release,
        buckets=base.buckets(aws),
        iot_endpoint=iot_endpoint(aws),
        replay=replay,
    )


def describe(version: ModelVersion, buckets: Buckets) -> str:
    """One line naming what a deployment of this version consists of.

    For the CLI's log rather than for a record. Everything in it is derivable,
    which is the point: an operator about to roll something out reads back what
    the strings they passed resolved to.
    """
    name, semver = component_address(version)
    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)
    return (
        f"{name} {semver} -- cycle {cycle} of {run_id}, frames from "
        f"{uri(buckets.artifacts, replay_manifest_key(run_id, cycle))}"
    )

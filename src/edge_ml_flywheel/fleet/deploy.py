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

import hashlib
import io
import json
import logging
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from random import Random
from typing import Any, Final

import boto3
import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import (
    CYCLE_DIGITS,
    Buckets,
    Cycle,
    ImageId,
    ModelVersion,
    RunId,
    component_address,
    model_component,
    model_version_cycle,
    model_version_run_id,
    new_model_version,
    parse_component_version,
    parse_image_id,
    replay_code_key,
    replay_manifest_key,
    selection_ranking_key,
    telemetry_run_prefix,
    uri,
)
from edge_ml_flywheel.fleet import component
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)

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


def sample_frames(
    aws: boto3.Session, run_id: RunId, cycle: Cycle, frames: int
) -> tuple[ImageId, ...]:
    """The pool frames this cycle's device replays, drawn out of its own ranking.

    **Random, not the top of the ranking.** The fleet is the footage a device
    drives through, and the selector is what acts on it (design section 7.2).
    Replaying the most uncertain frames would make every confidence the device
    reports come from the low-confidence tail, so the distribution a later check
    compares would be a property of the draw rather than of the model.

    **Out of the rows this cycle did not buy.** The ranking file marks its own
    batch, and those images have labels by the time the model is deployed -- so
    they are not the unlabeled pool any more, and including them would put frames
    in the device's stream that the next cycle's selector can no longer sell.

    **Seeded by the run and the cycle**, so a redeploy of one cycle replays the
    identical frames. Anything else would make two attempts at one deployment
    incomparable, and the second attempt is usually the one made after a
    rollback.
    """
    ranked = _unbought(aws, run_id, cycle)
    if len(ranked) < frames:
        raise SystemExit(
            f"cycle {cycle} left {len(ranked)} unbought pool images, fewer than the {frames} "
            f"frames asked for. The pool is spent, which is the end of the run rather than a "
            f"deployment"
        )

    drawn = tuple(sorted(Random(_draw_seed(run_id, cycle)).sample(ranked, frames)))
    log.info("sampled %d of %d unbought pool frames for cycle %d", frames, len(ranked), cycle)
    return drawn


def _draw_seed(run_id: RunId, cycle: Cycle) -> int:
    """A draw seed that is a function of the run and the cycle and nothing else.

    A digest rather than `hash()`, which is salted per process: the "same frames
    every time" property would otherwise hold within one invocation and nowhere
    else, which is exactly the case a redeploy after a rollback is not.

    The cycle is padded into the string for `purchase_event`'s reason -- it is
    text here, so cycle 10 and cycle 1 followed by a zero must not be one seed.
    """
    stamp = f"{run_id}-c{cycle:0{CYCLE_DIGITS}d}"
    return int(hashlib.sha256(stamp.encode()).hexdigest()[:16], 16)


def _unbought(aws: boto3.Session, run_id: RunId, cycle: Cycle) -> list[ImageId]:
    """Image IDs from the cycle's ranking that its purchase left behind.

    The refusal below is a real ordering, not a defensive check. `Select` runs
    after `Promote` in the cycle state machine, so there is a window -- seconds
    wide, and wide enough to hit by hand -- where a version is promoted and its
    ranking has not been written. Deploying inside it would otherwise fail as a
    missing S3 key, which reads as a broken pipeline rather than as a deployment
    made slightly too early.
    """
    key = selection_ranking_key(run_id, cycle)
    artifacts = base.buckets(aws).artifacts
    if not base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} does not exist, so cycle {cycle} has not yet ranked its pool. "
            f"Selection runs after promotion, so a model can be promoted a moment before the "
            f"frames its device would replay have been chosen. Wait for the cycle to reach "
            f"Purchase and deploy again."
        )

    body = aws.client("s3").get_object(Bucket=artifacts, Key=key)["Body"].read()
    table = pq.read_table(io.BytesIO(body), columns=["image_id", "selected"])
    return [
        parse_image_id(image_id)
        for image_id, selected in zip(
            table.column("image_id").to_pylist(),
            table.column("selected").to_pylist(),
            strict=True,
        )
        if not selected
    ]


def stage_frames(
    aws: boto3.Session, run_id: RunId, cycle: Cycle, image_ids: Sequence[ImageId]
) -> str:
    """Write the sampled frame list where the recipe names it as an artifact.

    Under the cycle's write-once prefix because it is the record design section
    7.2 asks for: the small per-cycle list that ties a telemetry record back to a
    scoring decision, and the thing that makes the device's confidences
    regenerable offline from retained images.
    """
    key = replay_manifest_key(run_id, cycle)
    artifacts = base.buckets(aws).artifacts
    aws.client("s3").put_object(
        Bucket=artifacts,
        Key=key,
        Body=json.dumps(sorted(str(image_id) for image_id in image_ids)).encode(),
        ContentType="application/json",
    )
    log.info("staged %d frames to %s", len(image_ids), uri(artifacts, key))
    return key


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


def redeploy(aws: boto3.Session, version: ModelVersion, target_arn: str) -> str:
    """Deploy one component version to the fleet, and return the deployment ID.

    The whole of both directions. Called with this cycle's version it is a
    rollout; called with the previous one it is the rollback, and the two are the
    same call because they are the same act -- a revision naming a version
    (design section 6).
    """
    request = component.deployment(version, target_arn)
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

    return new_model_version(run_id, parse_component_version(entry["componentVersion"]))


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

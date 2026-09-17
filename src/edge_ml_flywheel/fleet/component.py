"""What one promoted model is, as the Greengrass recipe that deploys it.

`registry.package`'s shape for the step after promotion: pure functions returning
the documents, with no client and no call, so the CLI and a test read one
definition rather than two.

**One component, not two.** The design describes a model component and a replay
component beside it (design section 6, design section 7.2), and at one device
that separation costs more than it buys. Two components mean the replay has to
discover which model version is deployed, which is cross-component configuration
or a shared directory -- a second mechanism to keep correct so that either can be
redeployed without the other, which at one device and one model per cycle never
happens. One component carries both, and a deployment then brings the model and
the code that exercises it or neither. What it appears to cost is re-uploading
unchanged code every cycle, except that `replay_code_key` addresses that archive
by commit, so two cycles built from one tree name one object.

**The sampled frames are an artifact, not a runtime fetch.** Greengrass hashes
every artifact when a component version is created and verifies it on download,
so shipping the frame list that way makes it immutable for the life of the
version and spares the device a bucket grant it would otherwise need. The images
themselves are fetched at runtime -- fifty megabytes is not a component artifact
-- which is the one grant the device holds over the data bucket, and it is over
pixels: the label wall applies to a device exactly as it applies to a training
job.

**The recipe is built here rather than in Terraform.** A component version is a
function of the model being deployed, so it is minted once per cycle by the step
that deploys it. Terraform owns what exists for the life of the project -- the
device, its identity, the rule that lands telemetry -- and owns no component at
all.
"""

import json
from dataclasses import dataclass
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    COMPONENT_PUBLISHER,
    POOL_SAMPLE,
    REPLAY_CODE_FILE,
    REPLAY_MANIFEST_FILE,
    Buckets,
    Cohort,
    Cycle,
    ModelArtifact,
    ModelVersion,
    Precision,
    Seed,
    component_address,
    detections_key,
    model_artifact_key,
    model_version_cycle,
    model_version_run_id,
    parse_model_version,
    replay_code_key,
    replay_manifest_key,
    telemetry_topic_prefix,
    uri,
)
from edge_ml_flywheel.selection.score import BAND_LOW

# The only recipe format Greengrass v2 publishes, and a required field.
RECIPE_FORMAT: Final = "2020-01-25"

# Graviton. Stated rather than left open, because a manifest with no platform
# matches every device: an x86 one would install an int8 graph exported for ARM
# and fail at its first inference rather than at the deployment.
PLATFORM: Final = {"os": "linux", "architecture": "aarch64"}

# Where the instance's user data puts the interpreter this component runs under.
# A virtualenv built once at boot rather than an install step in this recipe:
# onnxruntime and its wheels are tens of megabytes and do not change between
# cycles, so installing them per deployment would make every rollout as slow as
# the slowest pip resolve and would put a network failure in the path of a
# rollback.
INTERPRETER: Final = "/opt/edge-ml-flywheel/venv/bin/python"

# The directory `Unarchive` puts the code in, which Greengrass names after the
# archive file without its suffix. Derived from the file name rather than spelled
# for exactly that reason: the two are one fact, and the failure when they
# disagree is a component that starts and cannot import itself.
CODE_DIRECTORY: Final = REPLAY_CODE_FILE.removesuffix(".zip")

# How the nucleus spells a device's own name inside a recipe. The device names
# itself rather than being told what it is called, so a second device needs no
# edit here and cannot be configured into publishing under the first one's name.
THING_NAME: Final = "{iot:thingName}"


@dataclass(frozen=True, slots=True)
class Release:
    """What one deployment consists of: a model, the seed of it that ships, and
    the tree the device runs it under.

    Three facts that travel together everywhere in this package, and that are
    wrong together when any one of them is wrong: the digest the device reports
    is the seed's, the frames it replays are the version's cycle's, and the code
    that hashes and reports both is the commit's.

    `seed` is a field rather than a constant for `model_artifact_key`'s reason.
    Seed 1 ships by convention, and a convention spelled at the call site is one
    a caller can be wrong about loudly -- the manifest records which seed it was,
    and a release naming a different one produces a digest the canary rejects.

    `cycle` is the cycle doing the deploying, and it is the model's own on every
    cycle that promoted. It differs on one that did not: the champion stays, the
    frames do not, so the release is last cycle's model with this cycle's sample
    and a component version of its own. Absent, the two are one number.
    """

    version: ModelVersion
    seed: Seed
    git_commit: str
    cycle: Cycle | None = None

    def __post_init__(self) -> None:
        parse_model_version(self.version)
        # `replay_code_key` refuses a short SHA, and it refuses it where the key
        # is built -- which is after a component version has been named. Checked
        # here as well so a release that cannot be staged is refused before
        # anything has been uploaded under it.
        replay_code_key(self.git_commit)
        if self.cycle is not None and self.cycle < model_version_cycle(self.version):
            raise ValueError(
                f"cycle {self.cycle} cannot deploy {self.version}, which was trained in cycle "
                f"{model_version_cycle(self.version)}. A deploying cycle is at or after the one "
                f"that produced the model"
            )

    @property
    def deploying(self) -> Cycle:
        """The cycle whose component version and sample this release carries."""
        return self.cycle if self.cycle is not None else model_version_cycle(self.version)


@dataclass(frozen=True, slots=True)
class Replay:
    """How a device is asked to score, and how much of the pool it is given.

    One object rather than four arguments, because they are one decision. A pass
    over `POOL_SAMPLE` frames warming on 50 of them is a different measurement
    from a pass over 60 warming on 50, and settings that travelled separately
    would let a caller assemble the second while believing it had asked for the
    first.

    `frames` is the cycle's whole ranking universe and not a measurement sample.
    Design section 4.3 wanted 500 frames for a p95; this pass is also what
    selection ranks, so the number is `POOL_SAMPLE` and the latency percentile
    comes out of a far larger set than it was specified over.

    `image_size` has to match what the model was exported against. The floor is
    `BAND_LOW` rather than the cloud pass's 0.001: this file is ranked, never
    integrated, and `selection.score` discards everything below the band before
    it scores anything -- so the tail costs a tenfold larger file on a device
    with two cores and changes no ranking. Both are defaults here for
    `gates.thresholds`' reason: a run that changed them would be measuring
    something else.
    """

    frames: int = POOL_SAMPLE
    warmup: int = 50
    image_size: int = 416
    confidence_floor: float = BAND_LOW

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError(f"a warmup of {self.warmup} frames is not a warmup")
        if self.frames <= self.warmup:
            raise ValueError(
                f"{self.frames} frames with {self.warmup} of them warmup leaves nothing measured, "
                f"and a replay that records no frame reports as one that lost its telemetry"
            )
        if self.image_size < 1:
            raise ValueError(f"an input of {self.image_size} px is not an image")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError(
                f"a confidence floor outside [0, 1] admits nothing or everything: "
                f"{self.confidence_floor}"
            )

    @property
    def measured(self) -> int:
        """Frames that reach the telemetry, which is what the summary claims."""
        return self.frames - self.warmup


# Design section 4.3's replay, and what every deployment gets unless a flag says
# otherwise. A shared instance because `Replay` is frozen, so one object serves
# every signature rather than each constructing its own -- `thresholds.DEFAULT`'s
# arrangement.
STANDARD: Final = Replay()


def run_script() -> str:
    """The command the nucleus runs, as one line.

    Every path is a recipe variable rather than a literal. Greengrass decides
    where an artifact lands and where a component may write, and a path guessed
    here would be one that works until the nucleus changes its layout.
    """
    return " ".join(
        (
            INTERPRETER,
            "-m edge_ml_flywheel.fleet.replay",
            f"--model {{artifacts:path}}/{ModelArtifact.ONNX.value}",
            f"--frames {{artifacts:path}}/{REPLAY_MANIFEST_FILE}",
            "--work {work:path}",
            f"--thing {THING_NAME}",
            "--run_id {configuration:/runId}",
            "--version {configuration:/version}",
            "--topic {configuration:/topic}",
            "--endpoint {configuration:/iotEndpoint}",
            "--bucket {configuration:/dataBucket}",
            "--detections_bucket {configuration:/detectionsBucket}",
            "--detections_key {configuration:/detectionsKey}",
            "--task_token {configuration:/taskToken}",
            "--cycle {configuration:/cycle}",
            "--warmup {configuration:/warmup}",
            "--image_size {configuration:/imageSize}",
            "--confidence_floor {configuration:/confidenceFloor}",
        )
    )


def recipe(
    release: Release,
    buckets: Buckets,
    iot_endpoint: str,
    replay: Replay = STANDARD,
) -> dict[str, Any]:
    """The whole component version for one promoted model.

    `iot_endpoint` is resolved by the caller and baked into the configuration
    rather than discovered on the device. It is account-specific, so a device
    that looked it up would need `iot:DescribeEndpoint` and a call before its
    first publish -- a grant and a failure mode bought to avoid passing a string
    the deploying side already holds.

    No `ComponentDependencies`. The nucleus runs this and is not a dependency a
    component declares; everything else it needs is in the interpreter the
    instance built at boot.
    """
    cycle = release.deploying
    name, semver = component_address(release.version, cycle)
    run_id = model_version_run_id(release.version)
    model = model_artifact_key(release.version, release.seed, ModelArtifact.ONNX)

    return {
        "RecipeFormatVersion": RECIPE_FORMAT,
        "ComponentName": name,
        "ComponentVersion": semver,
        "ComponentDescription": (
            f"Scores cycle {cycle}'s pool sample with {release.version} and reports what it saw."
        ),
        "ComponentPublisher": COMPONENT_PUBLISHER,
        "ComponentConfiguration": {
            "DefaultConfiguration": {
                "runId": str(run_id),
                "version": str(release.version),
                # The deploying cycle, which the device keys its start counter
                # by. Not derivable on the device from the model version: a
                # rejected cycle redeploys the champion, so one version can be
                # three cycles' deployments.
                "cycle": cycle,
                "topic": f"{telemetry_topic_prefix(run_id)}{THING_NAME}",
                "iotEndpoint": iot_endpoint,
                "dataBucket": buckets.data,
                "detectionsBucket": buckets.artifacts,
                "detectionsKey": detections_key(
                    release.version,
                    release.seed,
                    Cohort.POOL,
                    precision=Precision.INT8,
                    cycle=cycle,
                ),
                # Empty by default and filled by the deployment, because a task
                # token does not exist when a component version is published and
                # a component version is immutable once it does. A pass with no
                # token still writes its file; what it does not do is resume an
                # execution.
                "taskToken": "",
                "warmup": replay.warmup,
                "imageSize": replay.image_size,
                "confidenceFloor": replay.confidence_floor,
            }
        },
        "Manifests": [
            {
                "Platform": PLATFORM,
                "Lifecycle": {
                    "Run": {
                        # The package is put on the path rather than installed
                        # into the interpreter, so the venv stays a function of
                        # the instance and the code stays a function of the
                        # deployment. A pip install here would make a rollback
                        # depend on an uninstall.
                        "Setenv": {"PYTHONPATH": _code_path()},
                        "Script": run_script(),
                    }
                },
                "Artifacts": [
                    {"URI": uri(buckets.artifacts, model)},
                    {"URI": uri(buckets.artifacts, replay_manifest_key(run_id, cycle))},
                    {
                        "URI": uri(buckets.artifacts, replay_code_key(release.git_commit)),
                        "Unarchive": "ZIP",
                    },
                ],
            }
        ],
    }


def _code_path() -> str:
    """Where `Unarchive` leaves the package, as the recipe variable for it."""
    return f"{{artifacts:decompressedPath}}/{CODE_DIRECTORY}"


def deployment(
    version: ModelVersion,
    target_arn: str,
    name: str | None = None,
    task_token: str = "",
    cycle: Cycle | None = None,
) -> dict[str, Any]:
    """The `CreateDeployment` request that puts one component version on the fleet.

    **The task token rides here and not in the recipe.** A component version is
    published before the cycle reaches the state that issues a token, and it is
    immutable once published, so the token arrives as a configuration update on
    the deployment -- which is created after the token exists and is the only
    part of this pair that can carry a value that late. An empty token deploys
    the component with the default, which is a pass nobody is waiting on.

    **The model version rides here too, always.** A component version names the
    cycle that deployed it rather than the cycle that trained the model, so the
    number in `0.<cycle>.0` no longer answers "what is this device running". The
    merge makes the deployment say so itself, which is what keeps it the single
    record of intent rather than a pointer something else has to interpret.

    **A deployment is the record of intent, and there is no second copy.** The
    design left open whether `fleet_config` or the Greengrass deployment holds
    what a device should be running (design section 6); this is the answer, and
    `conventions`' fleet_config section records it. The service already stores
    this document, revises it, and reports what each device made of it, so an
    item beside it would be one fact written twice by two calls, with nothing to
    say which is right when they disagree.

    **Rollback is this same request naming the previous version**, which is why
    there is no separate builder for one. A rollback down its own code path is a
    path first exercised on the day it is needed.

    `failureHandlingPolicy` is `ROLLBACK`, which is the health check design
    section 6 describes: a device that cannot install or start the new component
    returns to the one it was running, with nothing here being asked. That is the
    automatic half. The canary gate is the half that judges a deployment which
    installed perfectly well and is slower, or loaded the wrong file.
    """
    component, semver = component_address(version, cycle)
    merge: dict[str, str] = {"version": str(version)}
    if task_token:
        merge["taskToken"] = task_token

    update: dict[str, dict[str, Any]] = {
        component: {
            "componentVersion": semver,
            "configurationUpdate": {"merge": json.dumps(merge)},
        }
    }

    return {
        "targetArn": target_arn,
        "deploymentName": name or f"{component}-{semver}",
        "components": update,
        "deploymentPolicies": {
            "failureHandlingPolicy": "ROLLBACK",
            # The component is a batch job rather than a service with clients, so
            # there is nothing on the device to notify and nothing to wait for. A
            # notification policy would ask a component that does not subscribe
            # for permission to replace it.
            "componentUpdatePolicy": {"action": "SKIP_NOTIFY_COMPONENTS"},
        },
    }


def thing_group_arn(region: str, account_id: str, group: str) -> str:
    """The deployment target, composed rather than looked up.

    `run.control.cycle_machine_arn`'s arrangement and its reason: a
    `terraform output` would be the same string behind a second tool that has to
    be run in the right directory.
    """
    return f"arn:aws:iot:{region}:{account_id}:thinggroup/{group}"

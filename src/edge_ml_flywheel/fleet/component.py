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

from dataclasses import dataclass
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    COMPONENT_PUBLISHER,
    REPLAY_CODE_FILE,
    REPLAY_MANIFEST_FILE,
    Buckets,
    ModelArtifact,
    ModelVersion,
    Seed,
    component_address,
    model_artifact_key,
    model_version_cycle,
    model_version_run_id,
    parse_model_version,
    replay_code_key,
    replay_manifest_key,
    telemetry_topic_prefix,
    uri,
)

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
    """

    version: ModelVersion
    seed: Seed
    git_commit: str

    def __post_init__(self) -> None:
        parse_model_version(self.version)
        # `replay_code_key` refuses a short SHA, and it refuses it where the key
        # is built -- which is after a component version has been named. Checked
        # here as well so a release that cannot be staged is refused before
        # anything has been uploaded under it.
        replay_code_key(self.git_commit)


@dataclass(frozen=True, slots=True)
class Replay:
    """How a device is asked to replay, and how much of the pool it is given.

    Design section 4.3's measurement, as an object rather than four arguments:
    500 frames after 50 warmup at batch size 1, which is what the p95 and the
    cold start in the writeup are measured over.

    `frames` is read by the sampler and the other three by the device, and they
    are one object anyway, because they are one decision. A replay of 500 frames
    warming on 50 of them is a different measurement from a replay of 60 warming
    on 50, and settings that travelled separately would let a caller assemble the
    second while believing it had asked for the first.

    `image_size` and `confidence_floor` have to match the cloud pass or the
    device's confidences are not comparable to it, which is the comparison the
    realism check rests on. They are defaults here for `gates.thresholds`'
    reason: a run that changed them would be measuring something else.
    """

    frames: int = 500
    warmup: int = 50
    image_size: int = 416
    confidence_floor: float = 0.001

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
    name, semver = component_address(release.version)
    run_id = model_version_run_id(release.version)
    cycle = model_version_cycle(release.version)
    model = model_artifact_key(release.version, release.seed, ModelArtifact.ONNX)

    return {
        "RecipeFormatVersion": RECIPE_FORMAT,
        "ComponentName": name,
        "ComponentVersion": semver,
        "ComponentDescription": (
            f"Replays cycle {cycle}'s pool sample through {release.version} and publishes "
            f"what it saw."
        ),
        "ComponentPublisher": COMPONENT_PUBLISHER,
        "ComponentConfiguration": {
            "DefaultConfiguration": {
                "runId": str(run_id),
                "version": str(release.version),
                "topic": f"{telemetry_topic_prefix(run_id)}{THING_NAME}",
                "iotEndpoint": iot_endpoint,
                "dataBucket": buckets.data,
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
) -> dict[str, Any]:
    """The `CreateDeployment` request that puts one component version on the fleet.

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
    component, semver = component_address(version)
    return {
        "targetArn": target_arn,
        "deploymentName": name or f"{component}-{semver}",
        "components": {component: {"componentVersion": semver}},
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

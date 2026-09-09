"""What one training job is, as the request that creates it.

A pure function returning the `CreateTrainingJob` argument dictionary, with no
client and no call. That is the point: the walking skeleton starts a job from a
CLI and every later cycle starts five from a Step Functions `Map`, and both read
the definition out of here. An ASL block spelling out channels and
hyperparameters would be a second definition of the same job, in a language with
no tests, that nothing checks against this one.

**Three things this request deliberately does not carry.**

`CheckpointConfig` is absent, and its absence is the interrupt policy (design
section 3). A managed spot job with nowhere to checkpoint has nothing to resume
from, so an interrupted run restarts from the beginning rather than resuming
half-trained state -- which is what keeps "seed *k* fixes the run" true. The
rule is a property of the job definition and not of an interrupt handler
somebody maintains, so it is enforced by a test over this function rather than
described in a comment somewhere.

`VpcConfig` is absent, so the job runs on SageMaker's own network and reaches S3
without a NAT gateway.

`max_images` is absent, and it is a parameter of the *cycle* rather than of the
job: the images a job trains on are exactly the ones named in
`training_manifest_key`, and that document doubles as the record of what the
challenger was trained on. Shrinking the training set inside the container would
make the record and the run disagree, so a short skeleton run is a short
manifest -- see `training.images`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    Buckets,
    ClassSetVersion,
    Cohort,
    Cycle,
    ModelVersion,
    PartitionVersion,
    RunId,
    Seed,
    base_weights_key,
    cohort_labels_prefix,
    model_seed_prefix,
    model_version_cycle,
    model_version_run_id,
    purchases_run_prefix,
    training_code_key,
    training_manifest_key,
    uri,
)

# The AWS-owned account that publishes the Deep Learning Containers in every
# region. The same number everywhere except the opt-in regions, which this
# project does not run in.
DLC_ACCOUNT: Final = "763104351884"

# PyTorch 2.9 on Python 3.12, which is the reason this tag rather than the newer
# one beside it: `requires-python` in `pyproject.toml` pins the interpreter the
# lockfile resolves against, and 3.12 was provisional until a base image was
# chosen. It is chosen here, and the two now agree. CUDA 13 still builds for
# Turing, so the `ml.g4dn.xlarge` T4 in the cost model is unaffected.
#
# A tag and not `latest`. The container is half of what `recipe_version` means.
DLC_REPOSITORY: Final = "pytorch-training"
DLC_TAG: Final = "2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

# Where SageMaker puts a channel and where it collects the model, both fixed by
# the service. Spelled here because `entrypoint` reads them and this module
# writes the request that fills them.
CHANNEL_ROOT: Final = "/opt/ml/input/data"
MODEL_DIR: Final = "/opt/ml/model"

# The four channels, named once. `entrypoint` joins these onto `CHANNEL_ROOT`,
# so a rename is one edit rather than a job that starts and then cannot find its
# own data.
IMAGES_CHANNEL: Final = "images"
BOOTSTRAP_CHANNEL: Final = "bootstrap"
PURCHASES_CHANNEL: Final = "purchases"
BASE_CHANNEL: Final = "base"

# Script mode's contract: the archive to unpack and the file inside it to run.
# `train.py` sits at the root of the archive because SageMaker requires the entry
# point there, and it does nothing but call into this package.
ENTRY_POINT: Final = "train.py"

# SageMaker caps a training job name at 63 characters, and the name is built
# from parts this project already fixed the width of: a 16-character timestamp,
# the run slug, `-c` and three digits, `-s` and one, and a seven-character
# attempt suffix. Everything but the slug is 33, so a slug of 31 characters fits
# exactly and `RUN_SLUG_MAX_LEN`'s 32 is one too many at this one boundary.
#
# Not worth widening the cap for, and deliberately not enforced back in
# `conventions`: 32 characters of lowercase hyphenated words is a slug nobody
# writes -- the real ones are `v1-uncertainty` -- and a run ID has readers that
# have nothing to do with SageMaker. So the refusal lives here, where it can name
# the fix, rather than as a rule about run IDs in general.
MAX_JOB_NAME: Final = 63
_ATTEMPT_FORMAT: Final = "%H%M%S"


def image_uri(region: str) -> str:
    """The prebuilt container this job runs in.

    No ECR image of our own, so there is nothing to build, scan or keep patched:
    the container plus a `requirements.txt` is the whole environment. SageMaker
    pulls a first-party image with the service's own credentials, which is why
    the training role holds no ECR grant.
    """
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{DLC_REPOSITORY}:{DLC_TAG}"


def job_name(version: ModelVersion, seed: Seed, attempt: datetime) -> str:
    """`<model version>-s<seed>-<hhmmss>`.

    The version and seed are what a human reads off a console listing, so they
    lead. The attempt stamp is there because a job name is unique per account
    forever: a retry after a spot interrupt, or a second skeleton run of the same
    cycle, would otherwise be refused as a name collision rather than run. It
    identifies an attempt and nothing else -- what the artifacts are keyed by is
    the version and the seed, which two attempts share.
    """
    name = f"{version}-s{seed}-{attempt.strftime(_ATTEMPT_FORMAT)}"
    if len(name) > MAX_JOB_NAME:
        raise ValueError(
            f"training job name is {len(name)} characters, over SageMaker's {MAX_JOB_NAME}: "
            f"{name}. The run slug is what to shorten."
        )
    return name


@dataclass(frozen=True, slots=True)
class Recipe:
    """The training knobs, all of them, in one object.

    Together these are what `recipe_version` names. They are a dataclass rather
    than arguments spread across a function signature so that the run
    registration's claim to have trained under recipe *n* has something concrete
    to be a claim about, and so that a change to one of them is visible as a
    change to this class.

    `image_size` is 416 and is a handicap as much as an edge constraint (design
    section 3): a COCO-pretrained detector at full resolution starts strong
    enough that a cycle's 1,000 labels cannot move the metric.

    `freeze` holds the first ten modules, which is YOLO11n's backbone. Fine-tune
    the head, never train from scratch: it costs 10-100x more and lands in the
    same place. Unfreezing is a `recipe_version` change and a new run.
    """

    epochs: int
    image_size: int = 416
    batch: int = 16
    freeze: int = 10

    # Pinned rather than left to the framework's CPU-count default, because the
    # matched-seed comparison shares an augmentation order between champion and
    # challenger and that order is a function of how many workers drew it.
    workers: int = 4

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive: {self.epochs}")
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError(f"image size must be a positive multiple of 32: {self.image_size}")


@dataclass(frozen=True, slots=True)
class Target:
    """Which model this job produces, and the account it produces it in.

    The three objects a request is built from split by the question they answer:
    `Target` is what and where, `Recipe` is how, `Compute` is on what. That is
    also how they change -- a recipe change is a `recipe_version`, a compute
    change is a cost decision, and a target changes every single job.

    The run and the cycle are not fields. They are inside `version`, and every
    key this module builds recovers them from it, for the reason `conventions`
    keeps the version trio out of a model version in the first place.

    `partition_version` and `class_set_version` are here rather than defaulted,
    because both are preconditions of the comparison the run is making (design
    section 5): they come off the run registration, and a default would be a job
    training under a configuration its run never declared.
    """

    buckets: Buckets
    region: str
    role_arn: str
    version: ModelVersion
    seed: Seed
    partition_version: PartitionVersion
    class_set_version: ClassSetVersion

    # Cost attribution, and the only field here that decides nothing. Terraform's
    # `default_tags` covers what Terraform creates; a training job is created by
    # an API call and carries what it is given.
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def run_id(self) -> RunId:
        return model_version_run_id(self.version)

    @property
    def cycle(self) -> Cycle:
        return model_version_cycle(self.version)


@dataclass(frozen=True, slots=True)
class Compute:
    """What the job runs on and how long it is allowed to take.

    Spot by default, with `max_wait_seconds` covering the wait for capacity on
    top of the run itself. The runtime ceiling is a bound on a hang rather than
    an estimate, and it is deliberately close to design section 3's ~90-minute
    figure for a seed run: discard-and-restart is cheap while a run is short and
    ruinous when it is hours, so a job that overruns is a recipe that has drifted
    off the constraint rather than a job to wait out.
    """

    instance_type: str = "ml.g4dn.xlarge"
    volume_size_gb: int = 30
    max_runtime_seconds: int = 2 * 60 * 60
    max_wait_seconds: int = 3 * 60 * 60
    use_spot: bool = True

    def __post_init__(self) -> None:
        if self.use_spot and self.max_wait_seconds < self.max_runtime_seconds:
            raise ValueError(
                f"a spot job's max wait ({self.max_wait_seconds}s) must cover its max runtime "
                f"({self.max_runtime_seconds}s), since the wait includes the run"
            )


def _channel(name: str, s3_uri: str, data_type: str) -> dict[str, Any]:
    """One input channel, in `File` mode.

    `File` and not `FastFile` or `Pipe`: the channel is copied to local disk once
    and every epoch after that reads disk. What that copy costs is driven by
    object count rather than bytes, which is the open question design section 11
    leaves to measurement -- so every job logs the download time and nothing is
    packed into shards until a number says it should be.
    """
    return {
        "ChannelName": name,
        "InputMode": "File",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": data_type,
                "S3Uri": s3_uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def input_channels(target: Target) -> list[dict[str, Any]]:
    """The four channels a cycle's training set arrives on.

    `images` is a `ManifestFile` and the other three are prefixes, and the
    difference is not a preference. The labeled set is a scattered subset of the
    70,000 `train` images sitting flat under one prefix, so an `S3Prefix` channel
    would take all of them; the manifest names the subset object by object.
    Labels are the opposite -- the bootstrap file and each cycle's purchase are
    whole prefixes, appended to and never rewritten -- so a prefix channel is
    exactly the cumulative set with nothing to enumerate.

    `purchases` is absent at cycle 0, and the absence is required rather than
    tidy: nothing has been bought yet, and SageMaker fails a job whose channel
    prefix matches no object.
    """
    buckets = target.buckets
    channels = [
        _channel(
            IMAGES_CHANNEL,
            uri(buckets.artifacts, training_manifest_key(target.run_id, target.cycle)),
            "ManifestFile",
        ),
        _channel(
            BOOTSTRAP_CHANNEL,
            uri(buckets.data, cohort_labels_prefix(target.partition_version, Cohort.BOOTSTRAP)),
            "S3Prefix",
        ),
        _channel(
            BASE_CHANNEL,
            uri(buckets.artifacts, base_weights_key()),
            "S3Prefix",
        ),
    ]

    if target.cycle > 0:
        channels.append(
            _channel(
                PURCHASES_CHANNEL,
                uri(buckets.data, purchases_run_prefix(target.run_id)),
                "S3Prefix",
            )
        )
    return channels


def hyperparameters(target: Target, recipe: Recipe) -> dict[str, str]:
    """What the container is told, all of it strings, as SageMaker requires.

    The bucket is passed rather than discovered because the job writes its own
    artifacts: SageMaker's `model.tar.gz` is what the transform job and the model
    registry consume, and the loose `model.pt` beside its digest is what the
    manifest and the device verify. Two forms with two readers, from one set of
    bytes hashed where they were produced.

    `class_set_version` decides which of an image's boxes are trained on at all,
    so it is an input to the job and not a label attached afterwards. The job
    fails on a version `conventions.CLASS_SETS` does not define.

    The run and the cycle are not passed. They are inside `version`, and the
    container recovers them the way every key builder does -- three spellings of
    two facts is how a job comes to write its artifacts under a cycle it did not
    train.
    """
    return {
        "version": target.version,
        "seed": str(target.seed),
        "class_set_version": str(target.class_set_version),
        "epochs": str(recipe.epochs),
        "image_size": str(recipe.image_size),
        "batch": str(recipe.batch),
        "freeze": str(recipe.freeze),
        "workers": str(recipe.workers),
        "artifacts_bucket": target.buckets.artifacts,
    }


def environment(region: str) -> dict[str, str]:
    """Container environment, all four entries about Ultralytics behaving.

    `YOLO_AUTOINSTALL` off is the load-bearing one: Ultralytics will pip-install
    a missing dependency mid-run by default, which would put a package nobody
    pinned into a job whose whole claim is that it is reproducible from a
    lockfile and a requirements file.

    `YOLO_CONFIG_DIR` is redirected because the default is under `$HOME`, which a
    training container does not guarantee is writable.
    """
    return {
        "AWS_DEFAULT_REGION": region,
        "YOLO_AUTOINSTALL": "false",
        "YOLO_CONFIG_DIR": "/tmp/ultralytics",
        "MPLBACKEND": "Agg",
    }


def training_job(
    target: Target,
    recipe: Recipe,
    compute: Compute,
    attempt: datetime,
) -> dict[str, Any]:
    """The whole `CreateTrainingJob` request for one seed of one cycle.

    Four arguments, and each is one of the questions the request answers: which
    model, trained how, on what, started when. `attempt` is separate from
    `Target` because it is the one input that differs between two runs of an
    otherwise identical job -- a retry after a spot interrupt is the same target
    at a new attempt.
    """
    buckets = target.buckets
    version = target.version
    seed = target.seed
    output_prefix = f"{model_seed_prefix(version, seed)}_sagemaker/"

    request: dict[str, Any] = {
        "TrainingJobName": job_name(version, seed, attempt),
        "RoleArn": target.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri(target.region),
            "TrainingInputMode": "File",
        },
        "InputDataConfig": input_channels(target),
        # Underscore-prefixed for `RAW_PROVENANCE_PREFIX`'s reason: SageMaker
        # writes `<job name>/output/model.tar.gz` under whatever path it is
        # given, and a job-named directory beside the artifacts `conventions`
        # addresses would make the seed prefix two layouts at once. The tarball is
        # what a transform job and the registry read; `entrypoint` writes the
        # loose files that everything else does.
        "OutputDataConfig": {"S3OutputPath": uri(buckets.artifacts, output_prefix)},
        "ResourceConfig": {
            "InstanceType": compute.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": compute.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": compute.max_runtime_seconds},
        "HyperParameters": {
            **hyperparameters(target, recipe),
            "sagemaker_program": ENTRY_POINT,
            "sagemaker_submit_directory": uri(
                buckets.artifacts, training_code_key(target.run_id, target.cycle)
            ),
        },
        "Environment": environment(target.region),
        "EnableManagedSpotTraining": compute.use_spot,
        "Tags": [
            {"Key": key, "Value": value}
            for key, value in {
                "run_id": target.run_id,
                "cycle": str(target.cycle),
                "seed": str(seed),
                **target.tags,
            }.items()
        ],
    }

    if compute.use_spot:
        request["StoppingCondition"]["MaxWaitTimeInSeconds"] = compute.max_wait_seconds

    return request

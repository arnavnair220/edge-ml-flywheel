"""What one scoring job is, as the request that creates it.

`training.job`'s shape for the other job a cycle runs: a pure function returning
the `CreateProcessingJob` argument dictionary, with no client and no call, so the
state machine and a CLI read one definition rather than two.

**A Processing job rather than a Batch Transform.** Both would produce the same
detections, and the difference is what each costs to own. A transform needs a
SageMaker `Model` resource, an inference handler answering a request-response
contract, and it writes one output object per input object -- 5,000 objects and a
compaction step to turn them into the one parquet each consumer reads. A
Processing job loads the model once, walks the channel, and writes the file. The
service is not what makes this the evaluation plane; the detections are, and both
services emit the same ones.

**One job per seed, one cohort inside it.** This job covers `eval`: the 5,000
frozen images every reported metric is computed over. The pool is scored on the
device by the model deployed to it, which is what makes the selector's
uncertainty the fleet's uncertainty rather than a cloud checkpoint's. See
`fleet.replay` and `COHORTS` below.

**Three things this request deliberately does not carry**, all for
`training.job`'s reasons. No `VpcConfig`, so the job reaches S3 over SageMaker's
own network without a NAT gateway. No label channel of any kind: the detections
are matched against ground truth in the step after this one, by a role that is
allowed to read it. And no `max_images`, which is a parameter of the cycle rather
than of the job -- the images scored are exactly those named in
`scoring_manifest_key`, and that document is the record of what was ranked.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    MAX_DETS,
    TRAINING_CODE_FILE,
    Buckets,
    Cohort,
    Cycle,
    ModelArtifact,
    ModelVersion,
    Precision,
    RunId,
    Seed,
    detections_prefix,
    model_artifact_key,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
    training_code_key,
    uri,
)
from edge_ml_flywheel.training import job as training

# Where Processing puts an input and looks for an output, both fixed by the
# service's convention rather than by the API. Spelled here because `entrypoint`
# reads these paths and this module writes the request that fills them.
#
# The three absolute roots are built from one prefix and their own directory
# names, because `container_entrypoint` needs both halves: the absolute form is
# what the request and the container agree on, and the relative form is what
# keeps that command inside the 256 characters SageMaker allows one member of
# `ContainerEntrypoint`. Deriving both from one pair is what stops a `cd` in the
# command from drifting away from the paths it makes relative.
PROCESSING_ROOT: Final = "/opt/ml/processing"

_INPUT_DIR: Final = "input"
_OUTPUT_DIR: Final = "output"
_CODE_DIR: Final = "code"

INPUT_ROOT: Final = f"{PROCESSING_ROOT}/{_INPUT_DIR}"
OUTPUT_ROOT: Final = f"{PROCESSING_ROOT}/{_OUTPUT_DIR}"

# The two channels that are not a cohort. A cohort's channel is named by its own
# value, so `eval` arrives at `<INPUT_ROOT>/eval` and needs no constant.
CODE_CHANNEL: Final = "code"
MODEL_CHANNEL: Final = "model"

# The entry point at the root of the source archive, beside `train.py`. It exists
# for the same reason that one does -- a file the container command can name --
# though for the opposite cause: script mode *requires* `train.py` there, while
# this is there because a Processing job has no script mode at all.
ENTRY_POINT: Final = "score.py"

# What the archive's requirements file is called. `launch._REQUIREMENTS` packs it
# and the container command below installs it, which is the one place those two
# facts meet.
REQUIREMENTS: Final = "requirements.txt"

# Where the archive is unpacked. Not into the channel directory it arrives in:
# an input channel is the service's to populate, and unpacking beside the tarball
# would put a directory tree inside something a reader expects to hold one file.
UNPACKED: Final = f"{PROCESSING_ROOT}/{_CODE_DIR}"

# Images per forward pass for the fp32 pass. Here rather than in the container
# for `MAX_DETS`' reason one field over: the number belongs to the request the
# job is built from, and `entrypoint` cannot be imported without a GPU stack, so
# a constant declared there is a constant no test on a laptop ever reads.
_FP32_BATCH: Final = 32

# What one member of `ContainerEntrypoint` may be. The API rejects the request
# rather than the job failing, so the cost of exceeding it is a validation error
# minutes into a cycle that has already trained a model.
MAX_ENTRYPOINT_MEMBER: Final = 256


def container_entrypoint(entry_point: str = ENTRY_POINT) -> list[str]:
    """The command the container runs, unpacking the cycle's own source archive.

    A Processing job has no framework toolkit: SageMaker starts the image with
    whatever this says and nothing else happens on its behalf. So the three steps
    script mode performs invisibly for the training job are written out here --
    unpack, install, run -- and they are written as one `bash -c` because
    `ContainerEntrypoint` is a command rather than a shell.

    `set -euo pipefail` is the load-bearing part. Without `-e` a failed `tar` or
    a failed `pip install` is a job that goes on to run `score.py` against a
    missing package and reports the import error as the fault, several minutes
    after the real one.

    The trailing `entry_point` is `$0`, which is what makes `"$@"` expand to the
    `ContainerArguments` SageMaker appends rather than swallowing the first one.

    The argument is what makes this one definition rather than two. A cycle runs
    two Processing jobs -- scoring and evaluation -- from one archive under one
    `git_commit`, and they differ in the file at the root that is run and in
    nothing else about how the container starts. A second copy of these four
    lines would be a second place for the unpack path or the `-e` to go missing.

    **It opens with a `cd` because the command has a length limit.** One member
    of `ContainerEntrypoint` may be 256 characters and the absolute form of this
    was 265, which the API refuses -- after a cycle has trained a model, since
    the scoring request is built from what training produced. `/opt/ml/processing`
    appeared four times and is stated once instead, which is the whole saving.
    Nothing downstream is relative: the paths the request and the container agree
    on stay absolute, and `PROCESSING_ROOT` is what both forms are built from.
    """
    script = "; ".join(
        (
            "set -euo pipefail",
            f"cd {PROCESSING_ROOT}",
            f"mkdir -p {_CODE_DIR}",
            f"tar xzf {_INPUT_DIR}/{CODE_CHANNEL}/{TRAINING_CODE_FILE} -C {_CODE_DIR}",
            f"pip install --no-cache-dir --quiet -r {_CODE_DIR}/{REQUIREMENTS}",
            f'exec python {_CODE_DIR}/{entry_point} "$@"',
        )
    )
    if len(script) > MAX_ENTRYPOINT_MEMBER:
        raise ValueError(
            f"the container command is {len(script)} characters and SageMaker allows "
            f"{MAX_ENTRYPOINT_MEMBER}, so this request would be refused: {script}"
        )
    return ["bash", "-c", script, entry_point]


def artifact_for(precision: Precision) -> ModelArtifact:
    """The file one pass scores with.

    The fp32 pass loads the checkpoint; the int8 pass loads the quantized graph
    that ships, so what the edge gate measures is the artifact the device runs
    rather than a second conversion of it.
    """
    return ModelArtifact.ONNX if precision is Precision.INT8 else ModelArtifact.TORCH


# Which cohorts this job covers, at either precision: `eval`, and nothing else.
#
# **The pool is not scored in the cloud at all.** It is scored on the device, by
# the int8 artifact actually deployed, as the cycle's fleet round trip -- so the
# ranking is the uncertainty of the model driving rather than of a checkpoint
# that never left the account. This job produces the ruler; the fleet produces
# the catalogue. See `fleet.replay`.
#
# A constant rather than the function of `Precision` this was, because the answer
# stopped depending on the argument. `SCORED_COHORTS` still holds both, since a
# pool detections key is still a key something writes -- just not this job.
COHORTS: Final = frozenset({Cohort.EVAL})


@dataclass(frozen=True, slots=True)
class Target:
    """Which model is being scored, and the account it is scored in.

    `training.job.Target` without `partition_version`, and the absence is the
    point rather than an omission: a training job derives its channels from the
    partition, while a scoring job is handed two manifests that already name
    every image. There is nothing here for a partition to decide, so carrying one
    would be a field that could disagree with the documents.
    """

    buckets: Buckets
    region: str
    role_arn: str
    version: ModelVersion
    seed: Seed

    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def run_id(self) -> RunId:
        return model_version_run_id(self.version)

    @property
    def cycle(self) -> Cycle:
        return model_version_cycle(self.version)


@dataclass(frozen=True, slots=True)
class Scoring:
    """How the model is run over the images. The scoring half of the recipe.

    `image_size` is `training.job.IMAGE_SIZE` and not a number of its own,
    because a detector run at a resolution it was not trained at is a worse
    detector for a reason that appears nowhere in the report.

    `confidence_floor` is 0.001, which is COCO's convention and is far below
    anything a person would call a detection. AP is the area under a curve swept
    by lowering a threshold, so the low-confidence tail is most of what the metric
    integrates: raising this floor to something that looks sensible would truncate
    the curve and quietly lower every number the project reports. Selection wants
    the opposite and gets it by filtering -- `selection.score.BAND_LOW` keeps the
    band it ranks on at 0.05 -- which is the arrangement that lets one file serve
    both readers.

    `max_detections` is `conventions.MAX_DETS` rather than a number beside it.
    The match cache is built at that cap and cannot be asked for more later, so a
    job emitting more rows would write boxes no metric can ever read, and a job
    emitting fewer would silently cap the cache below its own limit. It comes
    from `conventions` rather than from the matcher that reads it because this
    module is imported by the control function, whose deployment package cannot
    carry `pycocotools`.

    `precision` is which build of the model the pass runs. `FP32` is the
    checkpoint and the pass every cycle makes; `INT8` is the quantized graph that
    ships, run over `eval` alone so the edge gate can say what quantization cost.
    It belongs here rather than on `Target` because it is a property of the run
    and not of the model -- one trained model has both builds -- and it is what
    decides the file loaded, the cohorts covered and the prefix written.
    """

    image_size: int = training.IMAGE_SIZE
    confidence_floor: float = 0.001
    max_detections: int = MAX_DETS
    precision: Precision = Precision.FP32

    @property
    def batch(self) -> int:
        """Images per forward pass, which is a function of the build being run.

        The int8 graph takes one frame and only one: `export.to_onnx` passes
        `dynamic=False` so the quantizer can fold shapes into constants, which
        fixes the batch axis at 1, and ONNX Runtime rejects anything else
        outright. Batching it would also measure something the fleet never does
        -- the device replays one frame at a time -- and this pass exists to say
        what quantization cost the artifact that ships.

        The fp32 pass batches, because it is a GPU over thousands of frames and
        one image per forward leaves the card idle between them. That is a
        throughput decision and nothing the metric can see: detection is per
        image, so how many share a forward pass changes only the wall clock.
        """
        return 1 if self.precision is Precision.INT8 else _FP32_BATCH

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_floor < 1.0:
            raise ValueError(
                f"the confidence floor must be inside (0, 1), and 0 is not a floor: "
                f"{self.confidence_floor}"
            )
        if not 1 <= self.max_detections <= MAX_DETS:
            raise ValueError(
                f"max detections must be between 1 and the {MAX_DETS} the match cache is built "
                f"at: {self.max_detections}"
            )


@dataclass(frozen=True, slots=True)
class Compute:
    """What the job runs on and how long it is allowed to take.

    The same T4 the training job uses, on demand for its reason. Inference over
    67,000 frames at 416 px is minutes of GPU; what the wall clock is actually
    spent on is the channel download, which is 67,000 objects copied to local
    disk before the first one is read. That is the same open question design
    section 11 leaves for training's channels, at eight times the object count,
    so the ceiling below is a bound on a hang rather than an estimate and the
    first job is what measures the rest.

    `volume_size_gb` has to hold every image the job scores. BDD100K frames run
    around 100 KB, so 67,000 of them is about 7 GB and 30 GB is the same headroom
    training runs with.
    """

    instance_type: str = "ml.g4dn.xlarge"
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3 * 60 * 60

    @classmethod
    def for_precision(cls, precision: Precision) -> "Compute":
        """What each pass runs on.

        The int8 pass is CPU, and not as a saving: int8 is a CPU format. ONNX
        Runtime's CUDA provider has no kernel for most of what `quantize_static`
        emits and falls back to float, which would measure the accuracy of a
        model the device will never run. `ml.m5.xlarge` is the instance the
        evaluation job already uses.

        It also covers 5,000 images rather than 67,000, so the volume and the
        ceiling come down with it.
        """
        if precision is not Precision.INT8:
            return cls()
        return cls(instance_type="ml.m5.xlarge", volume_size_gb=10, max_runtime_seconds=60 * 60)


def _manifest_input(target: Target, cohort: Cohort) -> dict[str, Any]:
    """One cohort's images, named object by object.

    `ManifestFile` for `training.job.input_channels`' reason and one more: the
    images to score are a subset of a split's prefix, and an `S3Prefix` channel
    pointed at `val/` would hand the job the `reserve` cohort as well -- images
    no cycle has any business scoring, and 5,000 of them.
    """
    return {
        "InputName": cohort.value,
        "S3Input": {
            "S3Uri": uri(
                target.buckets.artifacts,
                scoring_manifest_key(target.run_id, target.cycle, cohort),
            ),
            "LocalPath": f"{INPUT_ROOT}/{cohort.value}",
            "S3DataType": "ManifestFile",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
        },
    }


def _object_input(name: str, s3_uri: str) -> dict[str, Any]:
    """One named object, as a prefix channel that matches exactly it.

    `S3Prefix` against a full key rather than the directory holding it, so the
    model channel takes `model.pt` and not the `_sagemaker/` tarball beside it,
    and the code channel takes the archive and not the image manifest.
    """
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": f"{INPUT_ROOT}/{name}",
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
        },
    }


def inputs(target: Target, scoring: Scoring) -> list[dict[str, Any]]:
    """The channels a scoring job reads: its code, its model, and two cohorts.

    Cohorts in a fixed order, from `SCORED_COHORTS` sorted, so two calls produce
    the same request and a diff of two executions is about what changed.
    """
    buckets = target.buckets
    channels = [
        _object_input(
            CODE_CHANNEL,
            uri(buckets.artifacts, training_code_key(target.run_id, target.cycle)),
        ),
        _object_input(
            MODEL_CHANNEL,
            uri(
                buckets.artifacts,
                model_artifact_key(target.version, target.seed, artifact_for(scoring.precision)),
            ),
        ),
    ]
    channels.extend(_manifest_input(target, cohort) for cohort in sorted(COHORTS))
    return channels


def check_channel_names(
    processing_inputs: Sequence[Mapping[str, Any]],
    processing_outputs: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse a request whose channel names SageMaker will reject.

    Input and output names share one namespace, and the service enforces it at
    `CreateProcessingJob` -- so a collision is a `ValidationException` minutes
    into a cycle, against a request built from a model that has already been
    trained. Both Processing jobs are checked by the same function because the
    constraint belongs to the API rather than to either of them, and evaluation
    grows a channel per seed.

    `MAX_ENTRYPOINT_MEMBER` is the same bargain one field over: what the API
    will refuse is worth refusing here, where it costs a test rather than a GPU
    hour.
    """
    names = [str(channel["InputName"]) for channel in processing_inputs]
    names.extend(str(channel["OutputName"]) for channel in processing_outputs)

    collisions = sorted({name for name in names if names.count(name) > 1})
    if collisions:
        raise ValueError(
            f"input and output channel names must be unique across both lists, and "
            f"SageMaker rejects the request rather than the job: {collisions}"
        )


def output_name(cohort: Cohort) -> str:
    """What the request calls one cohort's output channel.

    Not the cohort's own value, which is what its *input* channel is called:
    SageMaker requires input and output names to be unique across both lists,
    and a job scoring `eval` named both of its `eval`. That is a rejected
    request rather than a failed job, and it arrives in a cycle that has already
    trained a model.

    The suffix says what the channel carries rather than disambiguating for its
    own sake, so the name is still readable in a console listing beside the
    input it is paired with.

    The container is unaffected: it writes to `OUTPUT_ROOT/<cohort>` from the
    cohort, and `LocalPath` below still says exactly that. This name is the
    API's label for the upload and nothing reads it back.
    """
    return f"{cohort.value}-detections"


def outputs(target: Target, scoring: Scoring) -> list[dict[str, Any]]:
    """One output channel per cohort, uploaded when the job finishes.

    `EndOfJob` rather than `Continuous`: a partially uploaded detections file is
    a file a later step would read as a complete answer, and there is nothing to
    watch in progress -- the consumer is a step that has not started yet.
    """
    return [
        {
            "OutputName": output_name(cohort),
            "S3Output": {
                "S3Uri": uri(
                    target.buckets.artifacts,
                    detections_prefix(target.version, target.seed, cohort, scoring.precision),
                ),
                "LocalPath": f"{OUTPUT_ROOT}/{cohort.value}",
                "S3UploadMode": "EndOfJob",
            },
        }
        for cohort in sorted(COHORTS)
    ]


def arguments(target: Target, scoring: Scoring) -> list[str]:
    """What the container is told, as `--flag value` pairs.

    Underscored flags, matching `training.entrypoint`'s. SageMaker forces the
    style there -- a hyperparameter key arrives verbatim as a flag -- and nothing
    forces it here, so this is one package with one flag style rather than two
    entry points that spell their arguments differently for a reason a reader has
    to know the service to see.

    The class set is not passed, and neither are the run and the cycle, for
    `training.job.hyperparameters`' reasons: there is one class set, and the
    other two are inside `version`.
    """
    return [
        "--version",
        target.version,
        "--seed",
        str(target.seed),
        "--precision",
        scoring.precision.value,
        "--image_size",
        str(scoring.image_size),
        "--confidence_floor",
        str(scoring.confidence_floor),
        "--max_detections",
        str(scoring.max_detections),
        "--batch",
        str(scoring.batch),
    ]


def processing_job(
    target: Target,
    scoring: Scoring,
    compute: Compute,
    attempt: datetime,
) -> dict[str, Any]:
    """The whole `CreateProcessingJob` request for one seed of one cycle.

    Four arguments for `training.job.training_job`'s four questions: which model,
    scored how, on what, started when.

    The job name comes from `training.job.job_name`, which is the same format
    with the same cap. Two resource types do not share a name space in SageMaker,
    so a scoring job and the training job that produced its model may carry one
    string -- and having one format means one length check and one place the run
    slug is reported as the thing to shorten.
    """
    processing_inputs = inputs(target, scoring)
    processing_outputs = outputs(target, scoring)
    check_channel_names(processing_inputs, processing_outputs)

    return {
        "ProcessingJobName": training.job_name(target.version, target.seed, attempt),
        "RoleArn": target.role_arn,
        "AppSpecification": {
            # The training container, deliberately. Processing overrides the
            # image's entry point, so the toolkit it ships is simply unused --
            # and reusing the tag means one pinned container in the project
            # instead of a second one that can drift a torch version away from
            # the weights it is loading.
            "ImageUri": training.image_uri(target.region),
            "ContainerEntrypoint": container_entrypoint(),
            "ContainerArguments": arguments(target, scoring),
        },
        "ProcessingInputs": processing_inputs,
        "ProcessingOutputConfig": {"Outputs": processing_outputs},
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": compute.instance_type,
                "VolumeSizeInGB": compute.volume_size_gb,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": compute.max_runtime_seconds},
        "Environment": training.environment(target.region),
        "Tags": [
            {"Key": key, "Value": value}
            for key, value in {
                "run_id": target.run_id,
                "cycle": str(target.cycle),
                "seed": str(target.seed),
                **target.tags,
            }.items()
        ],
    }

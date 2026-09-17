"""What one scoring job is, as the request that creates it.

`training.job`'s shape for the other job a cycle runs: a pure function returning
the `CreateProcessingJob` argument dictionary, with no client and no call, so the
state machine and a CLI read one definition rather than two.

**A Processing job rather than a Batch Transform.** Both would produce the same
detections, and the difference is what each costs to own. A transform needs a
SageMaker `Model` resource, an inference handler answering a request-response
contract, and it writes one output object per input object -- which at 67,000
images is 67,000 objects and a compaction step to turn them into the one parquet
each consumer reads. A Processing job loads the model once, walks the channel,
and writes the file. The service is not what makes this the evaluation plane; the
detections are, and both services emit the same ones.

**One job per seed, both cohorts inside it.** `eval` and the remaining pool are
two input channels and two output channels of a single job, because the expensive
parts -- acquiring a GPU, loading the checkpoint -- are paid per job and not per
cohort. Two jobs would pay them twice to keep two lists apart that the container
keeps apart anyway.

**Three things this request deliberately does not carry**, all for
`training.job`'s reasons. No `VpcConfig`, so the job reaches S3 over SageMaker's
own network without a NAT gateway. No label channel of any kind: the detections
are matched against ground truth in the step after this one, by a role that is
allowed to read it. And no `max_images`, which is a parameter of the cycle rather
than of the job -- the images scored are exactly those named in
`scoring_manifest_key`, and that document is the record of what was ranked.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    MAX_DETS,
    SCORED_COHORTS,
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
INPUT_ROOT: Final = "/opt/ml/processing/input"
OUTPUT_ROOT: Final = "/opt/ml/processing/output"

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
UNPACKED: Final = "/opt/ml/processing/code"


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
    """
    script = "; ".join(
        (
            "set -euo pipefail",
            f"mkdir -p {UNPACKED}",
            f"tar xzf {INPUT_ROOT}/{CODE_CHANNEL}/{TRAINING_CODE_FILE} -C {UNPACKED}",
            f"pip install --no-cache-dir --quiet -r {UNPACKED}/{REQUIREMENTS}",
            f'exec python {UNPACKED}/{entry_point} "$@"',
        )
    )
    return ["bash", "-c", script, entry_point]


def artifact_for(precision: Precision) -> ModelArtifact:
    """The file one pass scores with.

    The fp32 pass loads the checkpoint; the int8 pass loads the quantized graph
    that ships, so what the edge gate measures is the artifact the device runs
    rather than a second conversion of it.
    """
    return ModelArtifact.ONNX if precision is Precision.INT8 else ModelArtifact.TORCH


def cohorts_for(precision: Precision) -> frozenset[Cohort]:
    """Which cohorts one pass covers.

    The int8 pass takes `eval` alone. The pool is scored to be ranked, and the
    ranking is the selector's -- built from the model the cycle trained, not from
    a quantized copy of it. Scoring 62,000 frames a second time would spend an
    hour producing boxes nothing reads.

    A function rather than only a property on `Target`, because the container
    reads it too: a job that scored a cohort it was given no output channel for
    would do the work and then drop it.
    """
    return frozenset({Cohort.EVAL}) if precision is Precision.INT8 else SCORED_COHORTS


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
    channels.extend(
        _manifest_input(target, cohort) for cohort in sorted(cohorts_for(scoring.precision))
    )
    return channels


def outputs(target: Target, scoring: Scoring) -> list[dict[str, Any]]:
    """One output channel per cohort, uploaded when the job finishes.

    `EndOfJob` rather than `Continuous`: a partially uploaded detections file is
    a file a later step would read as a complete answer, and there is nothing to
    watch in progress -- the consumer is a step that has not started yet.
    """
    return [
        {
            "OutputName": cohort.value,
            "S3Output": {
                "S3Uri": uri(
                    target.buckets.artifacts,
                    detections_prefix(target.version, target.seed, cohort, scoring.precision),
                ),
                "LocalPath": f"{OUTPUT_ROOT}/{cohort.value}",
                "S3UploadMode": "EndOfJob",
            },
        }
        for cohort in sorted(cohorts_for(scoring.precision))
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
        "ProcessingInputs": inputs(target, scoring),
        "ProcessingOutputConfig": {"Outputs": outputs(target, scoring)},
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

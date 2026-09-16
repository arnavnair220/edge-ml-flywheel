"""What one evaluation job is, as the request that creates it.

`scoring.job`'s shape for the step after it: a pure function returning the
`CreateProcessingJob` argument dictionary, with no client and no call, so the
state machine and a CLI read one definition rather than two.

**This is the one job in the account that reads a ground-truth box for the eval
cohort.** Scoring deliberately holds no label grant -- it loads a checkpoint,
decodes JPEGs and writes down what came back -- so the matching has to happen
somewhere, and putting it in a second job with a second role is what lets the
first one be denied every label prefix outright rather than trusted to leave them
alone. The role this request is handed is the single ARN on
`eval_label_reader_arns` in `infra/storage.tf`.

**One job per cycle, not one per seed.** The paired delta is a mean over
same-seed differences, so every seed has to be in one process for the comparison
to happen at all. That is the opposite split from scoring, where a seed is an
independent pass over 67,000 images, and it follows from the same rule: a job is
the unit of work that cannot be divided without redoing it.

**A CPU instance, and no GPU image.** Nothing here loads a model. The expensive
part is a thousand resamples of `metrics.average_precision` over the cached match
arrays, which is numpy, so the GPU build of the container would be gigabytes of
CUDA pulled for drivers nothing opens.

**Three things this request deliberately does not carry.** No `VpcConfig`, for
`training.job`'s reason. No pool detections: the uncertainty ranking is
selection's input and is read by the step that spends the budget, and handing
them to the gate would put 62,000 images of boxes in a job that scores 5,000. And
no instance type from the execution input -- the one an execution may set is the
GPU type the training and scoring jobs share, and passing it here would put this
job on a GPU to run numpy.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    Buckets,
    Cohort,
    Cycle,
    ModelVersion,
    PartitionVersion,
    Precision,
    RunId,
    Seed,
    cohort_labels_prefix,
    detections_prefix,
    eval_matches_key,
    eval_metrics_key,
    eval_prefix,
    gate_report_key,
    gate_report_prefix,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
    training_code_key,
    uri,
)
from edge_ml_flywheel.scoring.job import (
    CODE_CHANNEL,
    INPUT_ROOT,
    OUTPUT_ROOT,
    container_entrypoint,
)
from edge_ml_flywheel.training import job as training

# The entry point at the root of the source archive, beside `train.py` and
# `score.py`. There for `scoring.job.ENTRY_POINT`'s reason -- a Processing job
# has no script mode, so the container command names a file directly.
ENTRY_POINT: Final = "evaluate.py"

# The channels this job reads, named once. `entrypoint` joins these onto
# `INPUT_ROOT`, so a rename is one edit rather than a job that starts and then
# cannot find what it was given.
MANIFEST_CHANNEL: Final = "manifest"
LABELS_CHANNEL: Final = "labels"
CHAMPION_CHANNEL: Final = "champion"

# One detections channel per seed, since each seed's boxes are a separate prefix.
# A prefix rather than a suffix so a listing of the input directory groups them.
DETECTIONS_CHANNEL: Final = "detections-seed"

# The deployed seed's int8 boxes over `eval`, written by the quantized scoring
# pass. One channel and not one per seed: exactly one artifact ships, so exactly
# one is quantized and measured.
INT8_CHANNEL: Final = "detections-int8"

# The two output channels: the cached arrays and the metrics beside them, and the
# verdict. Two rather than one because they are keyed differently -- the metrics
# are a function of the model and the report is a function of the cycle -- and an
# output channel uploads one directory to one prefix.
EVAL_OUTPUT: Final = "eval"
GATES_OUTPUT: Final = "gates"

# SageMaker caps a Processing job at ten input channels. Three are fixed, a
# fourth appears when there is a champion, and the rest are one per seed -- so
# six seeds is the ceiling and a run asking for more is refused here rather than
# by an API error naming a limit without naming the seed list. A cycle trains
# one (design section 4.2), so this is headroom and not a constraint anyone
# meets.
MAX_INPUTS: Final = 10


def detections_channel(seed: Seed) -> str:
    return f"{DETECTIONS_CHANNEL}-{seed}"


@dataclass(frozen=True, slots=True)
class Target:
    """Which comparison is being run, and the account it is run in.

    `seeds` rather than one seed, for the reason in the module docstring: the
    delta is a mean over same-seed differences, so the job takes every seed the
    cycle trained.

    `champion` is the version this challenger is measured against, and `None` is
    a run's first cycle rather than a missing argument. There is no champion
    before one has been promoted, so the first model is the baseline and the
    quality gate says so -- see `gates.quality`.

    `partition_version` is here where `scoring.job.Target` has no use for one:
    the eval boxes live under the partition prefix, so this is the one job whose
    channels are a function of the draw rather than only of the run.
    """

    buckets: Buckets
    region: str
    role_arn: str
    version: ModelVersion
    seeds: tuple[Seed, ...]
    partition_version: PartitionVersion
    champion: ModelVersion | None = None

    # The size of the int8 artifact the deployed seed exported, in bytes, read
    # off the object by the caller. `None` skips the edge gate entirely.
    #
    # Passed in rather than measured here because this job holds no model grant
    # at all -- `infra/evaluation.tf` keeps `models/*` off every statement, which
    # is what lets the account answer "what could have contaminated the eval" by
    # naming two identities. A file size is not a reason to widen that, and the
    # caller building this request already reads the bucket.
    artifact_bytes: int | None = None

    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def deployed_seed(self) -> Seed:
        """The seed whose artifact ships, and so the one the edge gate is about.

        `min` rather than a literal 1, matching `entrypoint.deployed_seed`.
        """
        return min(self.seeds)

    @property
    def gates_edge(self) -> bool:
        """Whether this job has an int8 pass to judge."""
        return self.artifact_bytes is not None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("an evaluation over no seed has nothing to compare")
        if self.artifact_bytes is not None and self.artifact_bytes < 1:
            raise ValueError(
                f"an int8 artifact of {self.artifact_bytes} bytes is not an artifact. Pass None "
                f"to run the cycle without an edge verdict rather than an empty file's size."
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"a seed appears twice in {list(self.seeds)}")
        if self.champion == self.version:
            raise ValueError(
                f"{self.version} cannot be compared against itself. A cycle's challenger and its "
                f"champion are two models, and a paired delta between one model and itself is zero "
                f"by construction."
            )

        channels = (
            len(_FIXED_CHANNELS)
            + len(self.seeds)
            + (1 if self.champion else 0)
            + (1 if self.gates_edge else 0)
        )
        if channels > MAX_INPUTS:
            raise ValueError(
                f"{len(self.seeds)} seeds need {channels} input channels, over the {MAX_INPUTS} a "
                f"Processing job accepts. Score fewer seeds, or give the detections one manifest "
                f"channel instead of one prefix channel each."
            )

    @property
    def run_id(self) -> RunId:
        return model_version_run_id(self.version)

    @property
    def cycle(self) -> Cycle:
        return model_version_cycle(self.version)


# Named for the count alone, which is what `Target.__post_init__` needs. The
# channels themselves are built in `inputs` below, where their URIs are.
_FIXED_CHANNELS: Final = (CODE_CHANNEL, MANIFEST_CHANNEL, LABELS_CHANNEL)


@dataclass(frozen=True, slots=True)
class Compute:
    """What the job runs on and how long it is allowed to take.

    `ml.m5.xlarge` and not the cycle's GPU type. Building the match cache is
    `pycocotools`' greedy assignment over 5,000 images, and the bootstrap is a
    thousand numpy gathers over what that produced -- minutes of one core, with
    the arrays for two models and a seed each held in memory at a few hundred
    megabytes.

    The runtime ceiling is a bound on a hang rather than an estimate, at roughly
    ten times the design's few-minutes figure for the resampling (design section
    4.2). A job that overruns it has found an eval cohort far larger than the
    5,000 this was sized for, which is a partition change and not a job to wait
    out.

    `volume_size_gb` holds the detections, the eval labels and the champion's
    cached arrays -- tens of megabytes between them. 30 is the same headroom the
    other two jobs run with, kept identical so there is one number to reason
    about rather than three.
    """

    instance_type: str = "ml.m5.xlarge"
    volume_size_gb: int = 30
    max_runtime_seconds: int = 60 * 60


def _prefix_input(name: str, s3_uri: str) -> dict[str, Any]:
    """One channel, as the prefix SageMaker copies before the container starts.

    Every channel here is an `S3Prefix`, unlike scoring's image channels: what
    this job reads are whole small prefixes -- one manifest document, one label
    file, one detections file per seed -- rather than a scattered subset of a
    directory holding 70,000 objects.
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


def inputs(target: Target) -> list[dict[str, Any]]:
    """Every channel this job reads: its code, what was scored, the boxes, the
    detections, and the champion it is compared against.

    The manifest is read for the images it names rather than for the images
    themselves, which is why it is a prefix channel over one JSON document and
    not the `ManifestFile` the scoring job was handed. It is the statement of
    what was put in front of the model: a frame the challenger missed entirely
    contributes no detection row, so a cache built from the detections alone
    would silently drop every miss and score the model on its hits.

    Seeds in ascending order, so two calls produce the same request and a diff of
    two executions is about what changed.
    """
    buckets = target.buckets
    channels = [
        _prefix_input(
            CODE_CHANNEL,
            uri(buckets.artifacts, training_code_key(target.run_id, target.cycle)),
        ),
        _prefix_input(
            MANIFEST_CHANNEL,
            uri(
                buckets.artifacts,
                scoring_manifest_key(target.run_id, target.cycle, Cohort.EVAL),
            ),
        ),
        _prefix_input(
            LABELS_CHANNEL,
            uri(buckets.data, cohort_labels_prefix(target.partition_version, Cohort.EVAL)),
        ),
    ]

    channels.extend(
        _prefix_input(
            detections_channel(seed),
            uri(buckets.artifacts, detections_prefix(target.version, seed, Cohort.EVAL)),
        )
        for seed in sorted(target.seeds)
    )

    # The deployed seed's int8 boxes, when an artifact was exported to judge.
    # Same cohort and same seed as one of the channels above, at the other
    # precision -- which is the distinction `Precision` exists to keep in the key.
    if target.gates_edge:
        channels.append(
            _prefix_input(
                INT8_CHANNEL,
                uri(
                    buckets.artifacts,
                    detections_prefix(
                        target.version, target.deployed_seed, Cohort.EVAL, Precision.INT8
                    ),
                ),
            )
        )

    # The champion's whole eval prefix, every seed's cache in one channel. Under
    # the cycle that produced the champion rather than this one, which is
    # `eval_prefix`'s reason for being keyed that way: the eval cohort is frozen,
    # so the champion's arrays stay valid where they were first written and are
    # read rather than recomputed every cycle (design section 7).
    if target.champion is not None:
        channels.append(
            _prefix_input(CHAMPION_CHANNEL, uri(buckets.artifacts, eval_prefix(target.champion)))
        )

    return channels


def outputs(target: Target) -> list[dict[str, Any]]:
    """Where the cached arrays, the metrics and the verdict land.

    `EndOfJob` for `scoring.job.outputs`' reason: a partially uploaded report is
    a document a later step reads as a complete verdict, and there is nothing to
    watch in progress.
    """
    return [
        {
            "OutputName": EVAL_OUTPUT,
            "S3Output": {
                "S3Uri": uri(target.buckets.artifacts, eval_prefix(target.version)),
                "LocalPath": f"{OUTPUT_ROOT}/{EVAL_OUTPUT}",
                "S3UploadMode": "EndOfJob",
            },
        },
        {
            "OutputName": GATES_OUTPUT,
            "S3Output": {
                "S3Uri": uri(
                    target.buckets.artifacts,
                    gate_report_prefix(target.run_id, target.cycle),
                ),
                "LocalPath": f"{OUTPUT_ROOT}/{GATES_OUTPUT}",
                "S3UploadMode": "EndOfJob",
            },
        },
    ]


def output_names(version: ModelVersion, seeds: Sequence[Seed]) -> dict[str, str]:
    """What each produced file is called, relative to its output channel.

    Derived from the key builders rather than spelled, for
    `scoring.entrypoint.score_cohort`'s reason: the output channel uploads a
    directory's contents under a prefix, and the two halves of that key have to
    agree. A name invented in the container would be a job that succeeds and an
    object nothing looks for.

    Returned as a mapping the container reads rather than as constants, because
    `eval_matches_key` puts the seed in a path component and only the caller
    knows the seeds.
    """
    root = eval_prefix(version)
    names = {"metrics": eval_metrics_key(version).removeprefix(root)}
    for seed in seeds:
        names[f"matches-{seed}"] = eval_matches_key(version, seed).removeprefix(root)

    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)
    names["report"] = gate_report_key(run_id, cycle).removeprefix(gate_report_prefix(run_id, cycle))
    return names


def arguments(target: Target) -> list[str]:
    """What the container is told, as `--flag value` pairs.

    Underscored flags, matching the other two entry points. The class set, the
    run and the cycle are not passed for `scoring.job.arguments`' reasons: there
    is one class set, and the other two are inside `version`.

    The thresholds are not passed either, and that is the stronger omission.
    They are the pre-declared promotion rule (`gates.thresholds`), so a job
    argument for one would be a way to gate two cycles of a run differently --
    and the gate report records the numbers it applied, which is only evidence if
    they could not have come from the request.
    """
    flags = [
        "--version",
        target.version,
        "--seeds",
        *(str(seed) for seed in sorted(target.seeds)),
    ]
    if target.champion is not None:
        flags.extend(("--champion", target.champion))
    if target.artifact_bytes is not None:
        flags.extend(("--artifact_bytes", str(target.artifact_bytes)))
    return flags


def processing_job(
    target: Target,
    compute: Compute,
    attempt: datetime,
) -> dict[str, Any]:
    """The whole `CreateProcessingJob` request for one cycle's evaluation.

    Three arguments where the other two jobs take four: which comparison, on
    what, started when. There is no recipe here -- this job runs no model, so
    there is no resolution or confidence floor to fix, and the numbers it applies
    are the gates' own.

    The job name is `training.job.job_name` at the cycle's first seed. Two
    resource types do not share a name space in SageMaker, so this may carry the
    same string as the training and scoring jobs that produced its inputs, and
    one format means one length check.
    """
    return {
        "ProcessingJobName": training.job_name(target.version, min(target.seeds), attempt),
        "RoleArn": target.role_arn,
        "AppSpecification": {
            # The CPU build of the cycle's pinned container. Processing overrides
            # the entry point, so the toolkit it ships is unused, and reusing the
            # framework version means the interpreter and the pyarrow that read a
            # detections file here are the ones that wrote it.
            "ImageUri": training.image_uri(target.region, training.DLC_CPU_TAG),
            "ContainerEntrypoint": container_entrypoint(ENTRY_POINT),
            "ContainerArguments": arguments(target),
        },
        "ProcessingInputs": inputs(target),
        "ProcessingOutputConfig": {"Outputs": outputs(target)},
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
                **target.tags,
            }.items()
        ],
    }

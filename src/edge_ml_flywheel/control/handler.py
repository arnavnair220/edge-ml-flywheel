"""One Lambda, one entry point, dispatching on a step name.

Five steps today, because five of a cycle's states need Python that already
exists: `prepare` writes the cycle's image manifest and source archive,
`train_request` builds one seed's `CreateTrainingJob` request, `score_prepare`
writes the two manifests naming what this cycle scores, `score_request` builds
one seed's `CreateProcessingJob` request, and `evaluate_request` builds the one
that matches those detections against ground truth. The state machine hands the
three request steps to `sagemaker:createTrainingJob.sync` and
`sagemaker:createProcessingJob.sync`.

**The `*_request` steps build and do not call**, for the reason given below, and
the two `*_prepare` steps run once per cycle rather than once per seed. Both are
the same split: what a cycle hands its seeds is written before the Map, and what
one seed does with it is resolved inside. `evaluate_request` sits on the cycle
side of that line even though it follows a Map -- the paired delta is a mean over
same-seed differences, so every seed belongs to one job.

**One function rather than one Lambda per step.** The steps share their whole
context -- a session, the bucket names, the run registration -- and none of them
is hot, large, or differently privileged. Five functions would be five
deployment packages, five log groups and five roles for the sake of a dispatch
that is a dictionary lookup. The step name is in the ASL, so a typo is a failed
execution naming the step it could not find rather than a call that silently
does nothing.

**The training request is built here and created there.** A Lambda that started
the job would have to either wait ninety minutes for it or hand the polling back
to the state machine anyway, and `.sync` already owns waiting, retrying and
stopping the job if the execution is aborted. So this returns the request and
makes no SageMaker call at all -- which is also why the Lambda's role holds no
SageMaker grant and cannot pass the training role.

**Where the source tree comes from.** `prepare` uploads the archive the cycle's
seeds run, and it is built from what Terraform deployed rather than from a git
checkout a Lambda does not have: the package is unpacked at `LAMBDA_TASK_ROOT`,
and `container/train.py` with `requirements.txt` beside it arrive in a layer at
`/opt`, which is the one place a Lambda can be handed files that live outside
`src/`. Both are fixed at deploy time, so the archive is a function of the
deployed function rather than of when it ran, and `launch.archive` flattens the
tar headers so it is byte-identical every invocation.
"""

import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import (
    Cycle,
    Seed,
    new_model_version,
    parse_model_version,
    parse_run_id,
)
from edge_ml_flywheel.evaluation import job as evaluation_job
from edge_ml_flywheel.evaluation import launch as evaluation_launch
from edge_ml_flywheel.scoring import job as scoring_job
from edge_ml_flywheel.scoring import launch as scoring_launch
from edge_ml_flywheel.training import job, launch

log = logging.getLogger()
log.setLevel(logging.INFO)

# Where the deployment package is unpacked, and where a layer's contents land.
# Both are the Lambda runtime's, not ours. `LAMBDA_TASK_ROOT` is read from the
# environment rather than hardcoded so the module imports and tests outside a
# Lambda, where the two paths are a fixture.
TASK_ROOT: Final = Path(os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
LAYER_ROOT: Final = Path("/opt")

# The directory the package occupies inside the deployment zip. Terraform points
# `archive_file` at `src/`, so the zip root holds `edge_ml_flywheel/` and this is
# the same name `launch._CODE_ROOT` writes into the tar.
PACKAGE_DIR: Final = "edge_ml_flywheel"

STEP: Final = "step"


class ControlError(Exception):
    """A step refused the work it was asked to do.

    Its own type so that a `Retry` in the ASL can distinguish it from a throttle
    or a timeout: this is the shape of failure that will fail again on the next
    attempt -- an unregistered run, a manifest that already exists, a cycle
    nobody prepared -- and retrying it wastes the execution's time before it
    reports the same thing.

    It also exists to convert `SystemExit`. The functions this calls were
    written for a CLI and raise `SystemExit` with a message meant for an
    operator's terminal; a `BaseException` escaping a Lambda handler kills the
    process and reports itself as a runtime crash, which loses that message.
    """


def _session() -> boto3.Session:
    """A session per invocation.

    Not cached at module scope: a Lambda container is reused across invocations
    and a session holds credentials that expire, so the saving is milliseconds
    against a failure mode that appears only on a long-lived container.
    """
    return boto3.Session()


def deployed_archive() -> bytes:
    """The source archive, built from what was deployed.

    Two directories because a Lambda has two: the package as `archive_file`
    zipped it, and the layer carrying the two files SageMaker's script mode
    requires at the root of the tar. See the module docstring for why they
    arrive separately.
    """
    return launch.archive(TASK_ROOT / PACKAGE_DIR, LAYER_ROOT)


def prepare(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Write the cycle's image manifest and source archive.

    `max_images` and `replace` are read with defaults because both are skeleton
    knobs the state machine may simply not pass: 0 is the whole labeled set, and
    a cycle that has already been prepared is an error rather than something to
    quietly overwrite.
    """
    run_id = parse_run_id(str(event["run_id"]))
    cycle = Cycle(int(event["cycle"]))

    images = launch.prepare(
        aws,
        run_id,
        cycle,
        launch.Preparation(
            code=deployed_archive(),
            max_images=int(event.get("max_images", 0)),
            replace=bool(event.get("replace", False)),
        ),
    )
    return {"run_id": run_id, "cycle": cycle, "images": images}


def train_request(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Build one seed's `CreateTrainingJob` request and return it unstarted.

    `epochs` is required and has no default here, unlike `instance_type`. It is
    the recipe, and a control plane that quietly trains one epoch because nobody
    passed a number produces a model whose `recipe_version` is a claim about a
    recipe it did not run. The compute fields default to `job.Compute`, which is
    where the purchasing mode and the runtime ceiling are decided.
    """
    version = new_model_version(parse_run_id(str(event["run_id"])), Cycle(int(event["cycle"])))
    seed = Seed(int(event["seed"]))

    recipe = job.Recipe(epochs=int(event["epochs"]))
    compute = job.Compute(
        instance_type=str(event.get("instance_type", job.Compute().instance_type)),
    )

    built = launch.request(aws, version, seed, recipe, compute)
    log.info("built the request for %s seed %d", version, seed)
    return {"version": version, "seed": seed, "request": built}


def score_prepare(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Write the two manifests naming what this cycle scores.

    `max_images` and `replace` are read with defaults for `prepare`'s reason, and
    they are the same two knobs meaning the same two things one step later: 0 is
    every image in the cohort, and a cycle already prepared for scoring is an
    error rather than something to overwrite.

    The counts come back per cohort so the execution history records what each
    manifest named. A pool that has stopped shrinking, or an eval that is not
    5,000, is visible in the state output rather than only in a log group.
    """
    run_id = parse_run_id(str(event["run_id"]))
    cycle = Cycle(int(event["cycle"]))

    named = scoring_launch.prepare(
        aws,
        run_id,
        cycle,
        scoring_launch.Preparation(
            max_images=int(event.get("max_images", 0)),
            replace=bool(event.get("replace", False)),
        ),
    )
    return {
        "run_id": run_id,
        "cycle": cycle,
        "images": {cohort.value: count for cohort, count in named.items()},
    }


def score_request(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Build one seed's `CreateProcessingJob` request and return it unstarted.

    Takes the version rather than a run and a cycle, unlike the two `prepare`
    steps: by this point in a cycle the model exists and is what is being scored,
    and the run and cycle are inside its version. Passing all three is how a job
    comes to score a model under a cycle it was not trained in.

    Every field of `Scoring` is a default here. They are the scoring half of the
    recipe -- resolution, confidence floor, detection cap -- and each is pinned to
    a value another component depends on, so an execution input for any of them
    would be a way to score two cycles of one run differently.
    """
    version = parse_model_version(str(event["version"]))
    seed = Seed(int(event["seed"]))

    compute = scoring_job.Compute(
        instance_type=str(event.get("instance_type", scoring_job.Compute().instance_type)),
    )

    built = scoring_launch.request(aws, version, seed, scoring_job.Scoring(), compute)
    log.info("built the scoring request for %s seed %d", version, seed)
    return {"version": version, "seed": seed, "request": built}


def evaluate_request(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Build the cycle's `CreateProcessingJob` request and return it unstarted.

    Takes every seed at once, unlike `score_request`. A paired delta is a mean
    over same-seed differences, so the comparison cannot be divided across jobs
    and the seed list is the argument rather than the Map item.

    `champion` is optional and absent today. Nothing records a champion yet --
    `Promote` is still a stub -- so every cycle evaluates as its run's baseline,
    and the quality gate says so rather than comparing against nothing. The
    argument exists here because the comparison path is built and tested; what is
    missing is the step that would name a model to compare against.

    No `instance_type`, unlike the other two request steps. The one an execution
    may set is the GPU type training and scoring share, and this job is numpy
    over cached arrays -- so `job.Compute`'s CPU default is not a default a caller
    may override into something that costs ten times as much to run the same
    addition.
    """
    version = parse_model_version(str(event["version"]))
    seeds = tuple(Seed(int(seed)) for seed in event["seeds"])

    champion = event.get("champion")
    built = evaluation_launch.request(
        aws,
        version,
        seeds,
        evaluation_job.Compute(),
        parse_model_version(str(champion)) if champion else None,
    )
    log.info("built the evaluation request for %s at seeds %s", version, list(seeds))
    return {"version": version, "seeds": list(seeds), "request": built}


# The dispatch table, and the whole of this module's control flow. The keys are
# the strings the ASL passes, so they are part of the interface between the two
# files and are spelled once each.
STEPS: Final[Mapping[str, Callable[[boto3.Session, Mapping[str, Any]], dict[str, Any]]]] = {
    "prepare": prepare,
    "train_request": train_request,
    "score_prepare": score_prepare,
    "score_request": score_request,
    "evaluate_request": evaluate_request,
}


def handler(event: Mapping[str, Any], context: object = None) -> dict[str, Any]:
    """The Lambda entry point.

    `context` is accepted and unused, as the runtime requires it positionally.
    """
    name = str(event.get(STEP, ""))
    step = STEPS.get(name)
    if step is None:
        raise ControlError(
            f"no control step named {name!r}. The state machine passes one of: "
            f"{', '.join(sorted(STEPS))}"
        )

    log.info("step %s", name)
    try:
        return step(_session(), event)
    except SystemExit as refusal:
        raise ControlError(str(refusal)) from None

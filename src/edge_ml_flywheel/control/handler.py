"""One Lambda, one entry point, dispatching on a step name.

Ten steps, because ten of a cycle's states need Python that already exists:
`prepare` writes the cycle's image manifest and source archive, `train_request`
builds one seed's `CreateTrainingJob` request, `score_prepare` writes the eval
manifest and draws the sample the fleet will score, `score_request` builds one
seed's `CreateProcessingJob` request, `evaluate_request` builds the one that
matches those detections against ground truth, `register` writes the model
manifest and builds the `CreateModelPackage` that records the verdict, `deploy`
publishes the cycle's component version, `fleet_score` deploys it with the
cycle's task token and leaves the task open, `canary` judges what came back, and
`select` ranks it and writes what the cycle chose to buy. The state machine hands
the three request steps to `sagemaker:createTrainingJob.sync` and
`sagemaker:createProcessingJob.sync`, and the registration to the `aws-sdk`
integration for `sagemaker:createModelPackage`.

**Three steps act on hardware, and they are the cycle's round trip to the
fleet.** `deploy` and `fleet_score` put a model and a frame list on a device;
`canary` reads back what it did with them. They are here rather than in a fleet
Lambda of their own because a second function would be a second package, log
group and role for three calls that share this one's session and bucket names --
`select`, which reads what the device wrote, is already here.

**The purchase is deliberately not here.** It is the eighth step of a cycle and
it runs in its own function under its own role, because this one is denied
`raw/labels/` outright and the oracle is the single principal that must read
exactly those files. See `oracle.handler`.

**`register` and `select` are the two steps that write something read back
later.** The other five produce inputs to a job that is about to run. `register`
writes the manifest, which is the precondition of ever promoting the model
(design section 5); `select` writes the ranking, which is what the purchase is
charged against and the only record of what a batch was chosen over.

**The `*_request` steps build and do not call**, for the reason given below, and
the two `*_prepare` steps run once per cycle rather than once per seed. Both are
the same split: what a cycle hands its seeds is written before the Map, and what
one seed does with it is resolved inside. `evaluate_request` sits on the cycle
side of that line even though it follows a Map -- the paired delta is a mean over
same-seed differences, so every seed belongs to one job.

**One function rather than one Lambda per step.** The steps share their whole
context -- a session, the bucket names, the run registration -- and none of them
is hot, large, or differently privileged. Ten functions would be ten deployment
packages, ten log groups and ten roles for the sake of a dispatch that is a
dictionary lookup. The step name is in the ASL, so a typo is a failed execution
naming the step it could not find rather than a call that silently does nothing.

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
    PROJECT,
    Cycle,
    Precision,
    Seed,
    new_model_version,
    parse_model_version,
    parse_run_id,
)
from edge_ml_flywheel.evaluation import job as evaluation_job
from edge_ml_flywheel.evaluation import launch as evaluation_launch
from edge_ml_flywheel.fleet import component
from edge_ml_flywheel.fleet import deploy as fleet_deploy
from edge_ml_flywheel.registry import launch as registry_launch
from edge_ml_flywheel.scoring import job as scoring_job
from edge_ml_flywheel.scoring import launch as scoring_launch
from edge_ml_flywheel.selection import launch as selection_launch
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

# The thing group a deployment targets, spelled as `fleet.__main__` spells it.
# One group holding one device, so a second device joins the fleet without a
# deployment step changing.
THING_GROUP: Final = f"{PROJECT}-devices"


def _target(aws: boto3.Session) -> str:
    """The deployment target, composed rather than looked up."""
    return component.thing_group_arn(str(aws.region_name), launch.account_id(aws), THING_GROUP)


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

    `max_images` is read with a default for `prepare`'s reason, and it must be
    the same number `prepare` was given: that step caps the image manifest and
    this one caps the labels the container collects, and the two caps are one
    decision resolved twice. A cap reaching one and not the other is the pair
    `dataset.write` refuses.
    """
    version = new_model_version(parse_run_id(str(event["run_id"])), Cycle(int(event["cycle"])))
    seed = Seed(int(event["seed"]))

    recipe = job.Recipe(
        epochs=int(event["epochs"]),
        max_images=int(event.get("max_images", 0)),
    )
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

    Every field of `Scoring` but one is a default here. They are the scoring half
    of the recipe -- resolution, confidence floor, detection cap -- and each is
    pinned to a value another component depends on, so an execution input for any
    of them would be a way to score two cycles of one run differently.

    `precision` is the exception, and it is not a recipe value: it names which
    build of the model this pass runs, and a cycle makes two passes. It comes
    from the state machine rather than the execution input for that reason -- the
    two calls are two states, not a setting someone chooses per run.

    The int8 pass takes its compute from `Compute.for_precision` and ignores the
    execution's `instance_type`, which is the GPU type training and scoring
    share. int8 is a CPU format, so honouring that input would put the quantized
    graph on hardware whose runtime falls back to float and measure a model the
    device will never run.
    """
    version = parse_model_version(str(event["version"]))
    seed = Seed(int(event["seed"]))
    precision = Precision(str(event.get("precision", Precision.FP32.value)))

    compute = (
        scoring_job.Compute.for_precision(precision)
        if precision is Precision.INT8
        else scoring_job.Compute(
            instance_type=str(event.get("instance_type", scoring_job.Compute().instance_type)),
        )
    )

    built = scoring_launch.request(
        aws, version, seed, scoring_job.Scoring(precision=precision), compute
    )
    log.info("built the %s scoring request for %s seed %d", precision.value, version, seed)
    return {"version": version, "seed": seed, "request": built}


def evaluate_request(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Build the cycle's `CreateProcessingJob` request and return it unstarted.

    Takes every seed at once, unlike `score_request`. A paired delta is a mean
    over same-seed differences, so the comparison cannot be divided across jobs
    and the seed list is the argument rather than the Map item.

    `champion` is optional, and absent exactly once per run. It arrives as the
    pointer `Promote` wrote on the run's control item at the end of the previous
    cycle, handed back by the claim that opened this one -- so a run's first
    cycle passes nothing and is evaluated as its own baseline, and every cycle
    after it is a paired comparison. Read with `.get` and treated as absent when
    null for that reason: the state machine passes the key either way, and a
    `None` reaching `parse_model_version` would be a first cycle that fails
    instead of a first cycle that has no champion.

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


def _deployable(event: Mapping[str, Any], cycle: Cycle) -> str:
    """The version to put on the device, refusing the one case where there is none.

    A run's first cycle that fails to promote has no champion to fall back on and
    nothing deployed to score its sample with. The gates make that nearly
    impossible -- with no champion the quality gate is the collapse check alone --
    but "nearly" is not a thing to leave to a `None` reaching a parser several
    frames down, so it is refused here with the reason.

    There is deliberately no cloud fallback. Scoring the pool in the account
    would produce a ranking from a model the fleet never ran, which is the exact
    thing this arrangement exists to stop producing.
    """
    version = event.get("version")
    if not version:
        raise ControlError(
            f"cycle {cycle} has no model to deploy: its challenger did not pass the gates and the "
            f"run has no champion yet. A run's first cycle must promote, because the pool is "
            f"scored on the device and nothing is deployed until one does."
        )
    return str(version)


def deploy(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Publish this cycle's component version, over whichever model is champion.

    Runs after `Promote`, so `version` is what the fleet should now be running:
    the challenger on a cycle that promoted, the standing champion on one that
    did not. Either way the component version is numbered by *this* cycle,
    because the recipe names this cycle's sample and a component version is
    immutable once published.

    Publishing is separate from deploying because only the second can carry a
    task token. This step makes the thing that will be deployed and checks the
    device can run it; `fleet_score` is where the deployment is created and the
    cycle begins waiting.
    """
    run_id = parse_run_id(str(event["run_id"]))
    cycle = Cycle(int(event["cycle"]))
    seed = Seed(int(event["seed"]))
    version = parse_model_version(_deployable(event, cycle))
    run = launch.registration(aws, run_id)

    fleet_deploy.require_device(aws)
    fleet_deploy.sampled_frames(aws, run_id, cycle)
    fleet_deploy.stage_code(aws, run.git_commit, fleet_deploy.code_archive(TASK_ROOT / PACKAGE_DIR))

    release = component.Release(version=version, seed=seed, git_commit=run.git_commit, cycle=cycle)
    recipe = fleet_deploy.recipe_for(aws, release, component.STANDARD)
    fleet_deploy.publish_component(aws, recipe)

    log.info("published %s for cycle %d over %s", recipe["ComponentVersion"], cycle, version)
    return {
        "run_id": run_id,
        "cycle": cycle,
        "version": version,
        "seed": seed,
        "component_version": recipe["ComponentVersion"],
    }


def fleet_score(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Deploy the cycle's component with its task token, and leave the task open.

    **This returns while the work is still running, which is the point.** The
    token reaches the device as a configuration value of the deployment, the
    device scores the sample and calls `SendTaskSuccess` when its detections are
    durable, and the state machine waits in between. A Lambda that waited for the
    pass itself would time out at fifteen minutes against an hour of ARM
    inference.

    The token is read from the event because `.waitForTaskToken` puts it there.
    Its absence is a state machine wired without the integration pattern, which
    is worth refusing loudly: the deployment would go out, the device would score
    the sample, and nothing would ever resume the cycle.
    """
    run_id = parse_run_id(str(event["run_id"]))
    cycle = Cycle(int(event["cycle"]))
    version = parse_model_version(_deployable(event, cycle))

    token = str(event.get("task_token", ""))
    if not token:
        raise ControlError(
            "fleet_score was invoked without a task token, so nothing could resume this cycle "
            "once the device finished. The state resource must end in .waitForTaskToken."
        )

    deployment = fleet_deploy.redeploy(aws, version, _target(aws), task_token=token)
    log.info("deployment %s hands cycle %d to the fleet", deployment, cycle)
    return {"run_id": run_id, "cycle": cycle, "version": version, "deployment": deployment}


def canary(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Judge the pass the device just finished, and roll back a failed rollout.

    Two decisions out of one verdict, and they are not the same decision.
    `passed` says whether the rollout stands. `ranks` says whether the cycle may
    buy from what the device wrote -- false only when the digest or the frame
    count failed, because those are statements about the detections rather than
    about the speed they were produced at. See `gates.canary`.

    The rollback is performed here rather than left to a later state, so that a
    device is never left running an artifact this step has already judged. There
    is nothing to roll back to on a run's first cycle, and nothing that needs it:
    a first cycle with no champion has no previous version.
    """
    run_id = parse_run_id(str(event["run_id"]))
    version = parse_model_version(str(event["version"]))
    champion = event.get("champion")
    previous = parse_model_version(str(champion)) if champion else None

    verdict = fleet_deploy.judge(aws, run_id, version, previous)
    if not verdict.passed and previous is not None and previous != version:
        fleet_deploy.redeploy(aws, previous, _target(aws))
        log.info("rolled %s back to %s", version, previous)

    log.info("canary %s: %s", "passed" if verdict.passed else "failed", verdict.reason)
    return {
        "version": version,
        "passed": verdict.passed,
        "ranks": verdict.ranks,
        "reason": verdict.reason,
    }


def select(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Rank what the fleet reported and write the file the purchase is charged
    against.

    Takes the version and one seed, unlike `evaluate_request` and `register`. A
    cycle ranks once, on one model: the uncertainty score is a statement about
    what a detector found in a frame, so averaging it over seeds would rank the
    pool by a model nothing in the fleet is or will be.

    The seed is the lowest the cycle scored, which is the one `register` calls
    deployed -- so the model that ranks the pool is the model that ships, and a
    run training more seeds does not buy its labels by a checkpoint it discards.

    `cycle` is passed beside the version because the two can name different
    cycles: a cycle that rejected its challenger ranks with the champion, whose
    version was minted earlier. The ranking belongs to the cycle doing the
    buying.

    `replace` is read with a default for `prepare`'s reason, and it is the more
    consequential of the two uses: this file is what the oracle charges against.
    """
    version = parse_model_version(str(event["version"]))
    seed = Seed(int(event["seed"]))
    cycle = Cycle(int(event["cycle"]))

    ranked = selection_launch.rank(aws, version, seed, cycle, bool(event.get("replace", False)))
    log.info("%s ranked %d images and chose %d", version, ranked.pool, ranked.batch)
    return {
        "version": ranked.version,
        "cycle": ranked.cycle,
        "pool": ranked.pool,
        "batch": ranked.batch,
        "blind_spots": ranked.blind_spots,
    }


def register(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Write the cycle's model manifest and build the request that records its
    verdict.

    Takes the version and every seed, as `evaluate_request` does and for the same
    reason: the manifest records a digest per seed, so the seed list is an
    argument rather than a Map item. The run is passed beside the version even
    though the version contains one, because the execution claimed its cycle
    under that run and the version was built several states later -- so the two
    are separate values here and checking them against each other is a check.

    `passed` is the value the cycle branches on, and this is the first step that
    returns a real one -- the gates ran in the evaluation job and this reads
    their report. Returned beside the request rather than derived from it later,
    because the branch and the approval status the registry records have to be
    one decision.

    The group is returned separately from the request it also appears in. The
    state machine opens it before registering into it, and a state that had to
    reach inside a request to find the name of the thing it creates would be a
    second spelling of it.
    """
    run_id = parse_run_id(str(event["run_id"]))
    version = parse_model_version(str(event["version"]))
    seeds = tuple(Seed(int(seed)) for seed in event["seeds"])

    registered = registry_launch.register(aws, run_id, version, seeds)
    log.info("%s registered as %s", version, "approved" if registered.passed else "rejected")
    return {
        "version": registered.version,
        "passed": registered.passed,
        "group": registered.group,
        "request": registered.request,
    }


# The dispatch table, and the whole of this module's control flow. The keys are
# the strings the ASL passes, so they are part of the interface between the two
# files and are spelled once each.
STEPS: Final[Mapping[str, Callable[[boto3.Session, Mapping[str, Any]], dict[str, Any]]]] = {
    "prepare": prepare,
    "train_request": train_request,
    "score_prepare": score_prepare,
    "score_request": score_request,
    "evaluate_request": evaluate_request,
    "register": register,
    "deploy": deploy,
    "fleet_score": fleet_score,
    "canary": canary,
    "select": select,
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

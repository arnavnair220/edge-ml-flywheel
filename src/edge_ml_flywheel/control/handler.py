"""One Lambda, one entry point, dispatching on a step name.

Two steps today, because two of a cycle's states need Python that already
exists: `prepare` writes the cycle's image manifest and source archive, and
`train_request` builds one seed's `CreateTrainingJob` request for the state
machine to hand to `sagemaker:createTrainingJob.sync`.

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

from edge_ml_flywheel.conventions import Cycle, Seed, new_model_version, parse_run_id
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

    `epochs` is required and has no default here, unlike the compute fields. It
    is the recipe, and a control plane that quietly trains one epoch because
    nobody passed a number produces a model whose `recipe_version` is a claim
    about a recipe it did not run. The compute fields default to `job.Compute`,
    which is where the spot policy and the runtime ceiling are decided.
    """
    version = new_model_version(parse_run_id(str(event["run_id"])), Cycle(int(event["cycle"])))
    seed = Seed(int(event["seed"]))

    recipe = job.Recipe(epochs=int(event["epochs"]))
    compute = job.Compute(
        instance_type=str(event.get("instance_type", job.Compute().instance_type)),
        use_spot=bool(event.get("use_spot", job.Compute().use_spot)),
    )

    built = launch.request(aws, version, seed, recipe, compute)
    log.info("built the request for %s seed %d", version, seed)
    return {"version": version, "seed": seed, "request": built}


# The dispatch table, and the whole of this module's control flow. The keys are
# the strings the ASL passes, so they are part of the interface between the two
# files and are spelled once each.
STEPS: Final[Mapping[str, Callable[[boto3.Session, Mapping[str, Any]], dict[str, Any]]]] = {
    "prepare": prepare,
    "train_request": train_request,
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

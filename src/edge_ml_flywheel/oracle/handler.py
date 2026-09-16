"""The oracle's Lambda entry point. One step, and its own function.

`control.handler`'s shape with one step in it, and the separation is the whole
point rather than a packaging detail. The control function is denied
`raw/labels/` outright -- orchestrating a cycle is not a reason to be able to
read a withheld label -- and the oracle is the one principal in the account that
must read exactly those files. Two identities cannot live in one Lambda, so they
do not live in one Lambda.

Both functions are built from the same deployment package, so this is a second
handler over one tree rather than a second copy of the code. What differs is the
role, which is the thing that had to differ.

**The request is not built here and handed back.** The three job steps in
`control.handler` return a request for the state machine to execute, because the
call they would make is one their role deliberately cannot. This step is the
opposite case: the whole action is a DynamoDB transaction and an S3 read that
only this role can perform, so it happens here and the state machine gets the
result.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

import boto3

from edge_ml_flywheel.conventions import Cycle, parse_run_id
from edge_ml_flywheel.oracle import launch

log = logging.getLogger()
log.setLevel(logging.INFO)

STEP: Final = "step"


class OracleError(Exception):
    """A purchase refused the work it was asked to do.

    Its own type for `control.handler.ControlError`'s two reasons. A `Retry` in
    the ASL can tell it from a throttle -- an unregistered run, a cycle nobody
    selected, a batch reaching into `eval` -- and every one of those is refused
    identically on a second attempt.

    It also converts `SystemExit`, which the functions underneath raise because
    they were written for an operator's terminal. A `BaseException` escaping a
    Lambda handler kills the process and reports a runtime crash, which loses the
    message.

    `OverBudgetError` deliberately does not become one of these. It is the one
    refusal here that is not a bug: a cycle can legitimately run out, and it
    should arrive at the execution history under its own name.
    """


def _session() -> boto3.Session:
    """A session per invocation, for `control.handler._session`'s reason: a Lambda
    container outlives its credentials."""
    return boto3.Session()


def purchase(aws: boto3.Session, event: Mapping[str, Any]) -> dict[str, Any]:
    """Buy the batch this cycle's selection chose.

    Takes the run and the cycle and nothing else. The batch is read out of the
    ranking rather than passed in, so there is no argument to this step that can
    name a different set of images from the one the record says was chosen --
    which is what keeps a retry a replay rather than a second purchase.
    """
    run_id = parse_run_id(str(event["run_id"]))
    cycle = Cycle(int(event["cycle"]))

    bought = launch.purchase(aws, run_id, cycle)
    log.info(
        "%s cycle %d bought %d labels, %d left in the pool",
        run_id,
        cycle,
        bought.images,
        bought.pool_remaining,
    )
    return {
        "run_id": run_id,
        "cycle": cycle,
        "images": bought.images,
        "pool_remaining": bought.pool_remaining,
        "replayed": bought.replayed,
    }


# One step today, and a dispatch table anyway. The shape is `control.handler`'s so
# that the two functions are read the same way, and the ASL passes a step name to
# each -- so a typo is a failed execution naming the step it could not find.
STEPS: Final[Mapping[str, Callable[[boto3.Session, Mapping[str, Any]], dict[str, Any]]]] = {
    "purchase": purchase,
}


def handler(event: Mapping[str, Any], context: object = None) -> dict[str, Any]:
    """The Lambda entry point.

    `context` is accepted and unused, as the runtime requires it positionally.
    """
    name = str(event.get(STEP, ""))
    step = STEPS.get(name)
    if step is None:
        raise OracleError(
            f"no oracle step named {name!r}. The state machine passes one of: "
            f"{', '.join(sorted(STEPS))}"
        )

    log.info("step %s", name)
    try:
        return step(_session(), event)
    except SystemExit as refusal:
        raise OracleError(str(refusal)) from None

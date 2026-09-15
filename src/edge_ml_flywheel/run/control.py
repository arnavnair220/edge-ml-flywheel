"""The run's control item, and the counter a cycle is claimed from.

Split from `registration` the way the two items are split in storage. A
registration is what a run *is* and is written once; this is what a run is
*doing* and changes on every cycle. Putting a counter on a write-once item means
giving up either the write-once rule or the counter, so they are two items in
two tables.

**`next_cycle` is claimed, never read-then-written.** The state machine's first
act is a conditional `UpdateItem` that adds one to this field and returns the
item as it was, so the number it gets back is the cycle it now owns and no other
execution can own. A read followed by a write is the same code with a race in
it, and the race is exactly the case worth refusing -- a double-fired cron tick
opening a second cycle against the same budget. That conditional write is the
single-flight lock, which is why the counter lives here rather than in a field
the orchestrator carries between executions.

`cycle_cap` is on this item and not on the registration for one reason: a
condition expression can only name attributes of the item it writes. A cap
stored anywhere else would be a cap some second call has to fetch and pass in,
which is a cap two executions can disagree about.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from edge_ml_flywheel.conventions import (
    PROJECT,
    RUN_ENTITY,
    Cycle,
    RunId,
    Table,
    columns,
    parse_run_id,
    table_name,
)
from edge_ml_flywheel.run.registration import RunAlreadyRegisteredError

log = logging.getLogger(__name__)

# The error DynamoDB returns when a `ConditionExpression` is not satisfied.
# Restated rather than imported from `registration`, which keeps its own copy
# private: it is botocore's constant, not that module's.
_CONDITION_FAILED = "ConditionalCheckFailedException"

# A cap of zero is a run that can never claim a cycle, and one that would have to
# be noticed by wondering why nothing ever happened. Refused where it is stated.
MIN_CYCLE_CAP = 1


@dataclass(frozen=True, slots=True)
class RunControl:
    """The `fleet_config` item a run is started with.

    Frozen like every other row in this project, and for the same reason it is
    frozen despite naming a counter: the *item* advances, this object does not.
    A cycle is claimed by a conditional update against DynamoDB, so a mutable
    `next_cycle` here would be a second, local, unsynchronized copy of the only
    number that must have exactly one.

    `entity` is not a field. It is the sort key and it is a constant for this
    shape of item, so a field would be a value that can be set to something the
    item is then unaddressable at.

    `next_cycle` is the cycle the *next* claim gets, not the one running. It is 0
    at registration, so the first execution claims cycle 0 and leaves 1 behind
    it.
    """

    run_id: RunId
    next_cycle: Cycle
    cycle_cap: int

    def __post_init__(self) -> None:
        parse_run_id(self.run_id)
        if self.next_cycle < 0:
            raise ValueError(f"next cycle cannot be negative: {self.next_cycle}")
        if self.cycle_cap < MIN_CYCLE_CAP:
            raise ValueError(
                f"cycle cap must be at least {MIN_CYCLE_CAP}: {self.cycle_cap}. A run that can "
                f"claim no cycle trains nothing and reports it as a clean finish."
            )


def to_item(control: RunControl) -> dict[str, Any]:
    """A control item as DynamoDB stores it.

    Both counters are `N` rather than the padded strings a cycle becomes inside
    an audit sort key, and the difference is what compares them. A `N` compares
    numerically, which is what `next_cycle < cycle_cap` in the claim's condition
    expression needs; the same values as strings would compare lexicographically,
    which is right up to cycle 9 and wrong at 10.

    The field set is checked against the schema of record on the way out, for
    `registration.to_item`'s reason. `entity` is expected to be *extra* here --
    it is the sort key, and the check is for fields that lost their encoding.
    """
    item: dict[str, Any] = {
        "run_id": str(control.run_id),
        "entity": RUN_ENTITY,
        "next_cycle": int(control.next_cycle),
        "cycle_cap": int(control.cycle_cap),
    }

    expected = set(columns(RunControl))
    missing = sorted(expected - set(item))
    if missing:
        raise ValueError(f"control fields with no encoding here: {missing}")

    return item


def from_item(item: dict[str, Any]) -> RunControl:
    """The inverse, with `RunControl.__post_init__` as the validator.

    The `int` conversions are the one thing this has to get right, for
    `registration.from_item`'s reason: the resource API returns every `N` as a
    `Decimal`, and a `Decimal` cycle compares unequal to the same cycle as an
    `int` everywhere that reads it back.
    """
    return RunControl(
        run_id=parse_run_id(item["run_id"]),
        next_cycle=Cycle(int(item["next_cycle"])),
        cycle_cap=int(item["cycle_cap"]),
    )


def fleet_config_table(dynamodb: Any = None) -> Any:
    """The `fleet_config` table, built on demand for `runs_table`'s reason: a
    resource created at import time fixes the region and the credentials before
    any caller has had a chance to choose them."""
    resource = dynamodb if dynamodb is not None else boto3.resource("dynamodb")
    return resource.Table(table_name(Table.FLEET_CONFIG))


def register(table: Any, control: RunControl) -> None:
    """Open the run's cycle counter, or refuse.

    Conditional for the registration's reason and then some. This item *is* the
    cycle counter, so a second write does not duplicate a record -- it resets a
    live run to cycle 0 and points the next execution at prefixes an earlier one
    has already written under, which the write-once artifacts bucket will then
    refuse halfway through a cycle rather than at its start.

    The condition names the partition key, which is how DynamoDB spells "no item
    at this key" for a composite-key table.
    """
    try:
        table.put_item(
            Item=to_item(control),
            ConditionExpression="attribute_not_exists(run_id)",
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != _CONDITION_FAILED:
            raise
        raise RunAlreadyRegisteredError(
            f"{control.run_id} already has a control item, so it is a run that has already been "
            f"started. Writing this one would reset its cycle counter to 0 and send the next "
            f"execution over prefixes an earlier one has written."
        ) from None

    log.info("cycle counter open at %d, capped at %d", control.next_cycle, control.cycle_cap)


def cycle_machine_arn(aws: boto3.Session) -> str:
    """The cycle state machine, composed from the account and the name Terraform
    gives it -- the same arrangement `training.launch.role_arn` uses, and for the
    same reason: a `terraform output` would be the same string behind a second
    tool that has to be run in the right directory."""
    account = aws.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:states:{aws.region_name}:{account}:stateMachine:{PROJECT}-cycle"


def start(
    aws: boto3.Session,
    run_id: RunId,
    epochs: int,
    seeds: Sequence[int],
    max_images: int = 0,
) -> str:
    """Start the run and return its execution ARN.

    One execution is the whole run, not one cycle. The machine loops from
    `MoreCycles` back to `ClaimCycle` and leaves through `Done` when the cap or
    the pool is spent, so there is nothing to tick and nothing to call again --
    which is also why a schedule firing once per cycle was never built.

    The execution takes the run ID as its name. Executions are unique by name, so
    a second start against a run already going is refused by Step Functions
    rather than by a check here that has to remember to run -- the same argument
    `register` makes for the conditional write above.

    `max_images` is omitted from the payload when it is 0 rather than passed as a
    zero, because the state machine's default *is* the whole labeled set and a
    cap nobody chose should not appear in the record of what an execution was
    asked for.
    """
    payload: dict[str, Any] = {"run_id": str(run_id), "epochs": epochs, "seeds": list(seeds)}
    if max_images:
        payload["max_images"] = max_images

    response = aws.client("stepfunctions").start_execution(
        stateMachineArn=cycle_machine_arn(aws),
        name=str(run_id),
        input=json.dumps(payload),
    )

    log.info("started %s over seeds %s at %d epochs", run_id, list(seeds), epochs)
    return str(response["executionArn"])


def read(table: Any, run_id: RunId) -> RunControl | None:
    """A run's control item, or `None` if its counter was never opened.

    `None` rather than a raise for `registration.read`'s reason, and
    `ConsistentRead` for the same one: the only caller that reads immediately
    after a write is the registration confirming its own.
    """
    response = table.get_item(
        Key={"run_id": str(parse_run_id(run_id)), "entity": RUN_ENTITY},
        ConsistentRead=True,
    )
    item = response.get("Item")
    return from_item(item) if item else None

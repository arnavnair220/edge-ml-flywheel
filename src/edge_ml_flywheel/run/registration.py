"""The registration item, and the write that claims a run's name.

Two halves, deliberately separable. `to_item` and `from_item` are pure and hold
the entire storage schema of a run; `register` and `read` are four lines of
boto3 around them. So the encoding -- the part that is written once and can
never be corrected -- is exercisable without credentials, and the AWS surface is
small enough to read in one screen.

**The conditional put is the whole point of this module.** `run_id` is
second-precision UTC plus a slug (`conventions`), which makes a collision
unlikely and not impossible, and a collision is silent in the worst way: the
second run adopts the first's spent-label ledger, its champion and its locks,
because those are addressed by `run_id` and nothing else. `attribute_not_exists`
turns that into a refusal at run start, before anything downstream has written a
byte. It is also why registration is mandatory rather than a nicety -- skipping
it does not skip the check, it removes it.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

from edge_ml_flywheel.conventions import (
    ClassSetVersion,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Selector,
    Table,
    columns,
    parse_run_id,
    table_name,
)

log = logging.getLogger(__name__)

# The error DynamoDB returns when a `ConditionExpression` is not satisfied. A
# string rather than a typed exception because botocore raises one `ClientError`
# for every API failure and distinguishes them by this code alone.
_CONDITION_FAILED = "ConditionalCheckFailedException"


class RunAlreadyRegisteredError(Exception):
    """A run by this ID is already in the table.

    Its own type rather than a re-raised `ClientError`, because the two callers
    want different things from it: a human starting a run needs to know the id
    collided, and a retry of the same registration wants to distinguish "already
    done" from a permissions or network failure that should be retried.
    """


def to_item(registration: RunRegistration) -> dict[str, Any]:
    """A registration as a DynamoDB item.

    Every field is encoded explicitly rather than by walking the dataclass,
    because three of them need a decision -- a `datetime` has no DynamoDB type,
    an enum has to lose its class, and a `NewType` int has to survive as a
    number. What a generic walk would have bought is drift protection, so that
    is bought directly instead: the field set is checked against the schema of
    record on the way out. Adding a field to `RunRegistration` and forgetting it
    here fails at the write rather than producing an item that is missing it
    permanently -- there is no second write to fix.
    """
    item: dict[str, Any] = {
        "run_id": str(registration.run_id),
        # Normalized to UTC so the stored strings sort chronologically, which is
        # the only ordering anyone reads this column for.
        "created_at": registration.created_at.astimezone(UTC).isoformat(),
        "git_commit": registration.git_commit,
        "partition_version": int(registration.partition_version),
        "class_set_version": int(registration.class_set_version),
        "recipe_version": int(registration.recipe_version),
        "selector": registration.selector.value,
        "label_budget_per_cycle": registration.label_budget_per_cycle,
        "note": registration.note,
    }

    expected = set(columns(RunRegistration))
    missing = sorted(expected - set(item))
    if missing:
        raise ValueError(f"registration fields with no encoding here: {missing}")

    return item


def from_item(item: dict[str, Any]) -> RunRegistration:
    """The inverse, with `RunRegistration.__post_init__` as the validator.

    Nothing here re-checks a value that the dataclass already refuses. The one
    thing this does have to get right is the numbers: the DynamoDB resource API
    returns every `N` attribute as a `Decimal`, so an unconverted
    `partition_version` compares unequal to the same version as an `int` and a
    `ModelManifest.disagreements` check would report a mismatch that is only a
    difference of type.
    """
    return RunRegistration(
        run_id=parse_run_id(item["run_id"]),
        created_at=datetime.fromisoformat(item["created_at"]),
        git_commit=item["git_commit"],
        partition_version=PartitionVersion(int(item["partition_version"])),
        class_set_version=ClassSetVersion(int(item["class_set_version"])),
        recipe_version=RecipeVersion(int(item["recipe_version"])),
        selector=Selector(item["selector"]),
        label_budget_per_cycle=int(item["label_budget_per_cycle"]),
        note=item["note"],
    )


def runs_table(dynamodb: Any = None) -> Any:
    """The `runs` table, named out of `conventions` rather than spelled here.

    Built on demand and passed in, never held as a module global: a resource
    created at import time fixes the region and the credentials before any
    caller has had a chance to choose them, and is the usual reason a test
    reaches the real account.
    """
    resource = dynamodb if dynamodb is not None else boto3.resource("dynamodb")
    return resource.Table(table_name(Table.RUNS))


def register(table: Any, registration: RunRegistration) -> None:
    """Claim the run's name, or refuse.

    The write and the uniqueness check are one operation on purpose. Reading
    first and writing if absent is the same code with a race in it, and the race
    window is exactly the case this guards -- two runs started in the same second
    by a human and a cron tick.
    """
    try:
        table.put_item(
            Item=to_item(registration),
            ConditionExpression="attribute_not_exists(run_id)",
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != _CONDITION_FAILED:
            raise
        raise RunAlreadyRegisteredError(
            f"{registration.run_id} is already registered. The id is a UTC second plus a slug, "
            f"so this is either a re-run of a completed registration or two runs started in the "
            f"same second -- either way, writing under it would adopt that run's ledger. Use a "
            f"different slug."
        ) from None

    log.info("registered %s", registration.run_id)


def read(table: Any, run_id: RunId) -> RunRegistration | None:
    """A run's registration, or `None` if it was never minted.

    `None` rather than a raise, because the two callers disagree about whether
    absence is an error: a promotion checking a model against its run has been
    handed a broken reference, and a human listing what exists has not.

    `ConsistentRead`, because the only caller that reads immediately after a
    write is the registration step confirming its own write, and an eventually
    consistent read there reports a run that does not exist yet.
    """
    response = table.get_item(Key={"run_id": str(parse_run_id(run_id))}, ConsistentRead=True)
    item = response.get("Item")
    return from_item(item) if item else None

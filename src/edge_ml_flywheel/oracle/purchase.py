"""Charging for a batch of labels, then serving it.

Three things have to be true of a purchase, and only one of them is about money:

1. **A retry must not charge twice.** Retries and redeliveries are normal in a
   step-function-driven loop, and a double debit has no undo -- it overstates the
   cost of every cycle after it, in the project's headline number.
2. **A cycle must not spend past its cap**, and must not be able to go negative
   to find that out.
3. **The charge and the ledger entry must not be able to disagree**, in either
   direction. A debit with no audit item is spend nobody can account for; an
   audit item with no debit is a purchase the budget never saw.

The third is what decides the implementation. The idempotency key and the ledger
live in two different tables, so guarding them with two conditional writes leaves
a window in which one has happened and the other has not -- and a crash in that
window is unrecoverable, because the retry cannot tell which half it is resuming.
`TransactWriteItems` closes it: one atomic write, one condition on each item,
both applied or neither.

    audit_log     put   `attribute_not_exists(event)`   -> refuses the second charge
    label_budget  update `remaining >= :n`              -> refuses the overspend

That single write is also what makes the ledger item's lifecycle simple enough to
state in a sentence. The item is created by the same expression that first debits
it, seeded from the registered budget with `if_not_exists`, so its only writer is
the step that spends against it and its value never rises.

**Serving happens after the transaction, never before or beside it.** A label is
read only once the charge has committed, which is what "charge, then serve"
means literally rather than as a description. A batch that fails the gate, fails
the budget or is a replay of an earlier purchase causes no label file to be
opened.

**A replay is a return value, not an error.** Retrying a purchase that already
happened is the normal case this is built for, so it returns the original
receipt marked `replayed` and re-serves the same labels. The caller re-writes the
same shards over the same keys, which is why the shard write is idempotent.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import boto3
from botocore.exceptions import ClientError

from edge_ml_flywheel.conventions import (
    Cycle,
    ImageId,
    RunId,
    RunRegistration,
    Table,
    batch_digest,
    purchase_event,
    purchase_shards_prefix,
    table_name,
)
from edge_ml_flywheel.oracle.cohorts import Cohorts, check_purchasable
from edge_ml_flywheel.oracle.labels import Fetch, SoldLabel

log = logging.getLogger(__name__)

# What DynamoDB reports when a transaction is refused. Strings rather than typed
# exceptions because botocore raises one `ClientError` for every API failure and
# distinguishes them by these codes alone.
#
# The two spellings are not a typo. A standalone conditional write raises
# `ConditionalCheckFailedException`; inside a transaction the same refusal
# appears in `CancellationReasons` as `ConditionalCheckFailed`, without the
# suffix. Matching on the wrong one turns every refusal here into an unhandled
# error -- a double charge would surface as a crash rather than a replay.
_TRANSACTION_CANCELLED: Final = "TransactionCanceledException"
_CANCELLED_BY_CONDITION: Final = "ConditionalCheckFailed"

# Position of each item in the transaction, which is how a cancellation is read
# back: `CancellationReasons` is a list parallel to `TransactItems`, so the index
# is what says *which* condition refused. Named rather than written as 0 and 1 at
# the call site, because reversing the two silently swaps "already purchased" for
# "over budget" and both are plausible answers.
_AUDIT_ITEM: Final = 0
_BUDGET_ITEM: Final = 1


class OverBudgetError(Exception):
    """The cycle does not have enough left to cover this batch.

    Its own type because it is the one failure here that is not a bug. A cycle
    can legitimately run out, and the caller's response is to buy less or to stop
    -- not to retry, which is what an undifferentiated `ClientError` would invite.
    """


@dataclass(frozen=True, slots=True)
class Receipt:
    """What one purchase charged, and where its labels were filed.

    Returned for a fresh purchase and for a replay alike; `replayed` is the only
    field that distinguishes them. Callers that need to know -- a cost report
    counting distinct purchases, a log line -- read it, and callers that just
    want the labels do not have to care.
    """

    run_id: RunId
    cycle: Cycle
    event: str
    digest: str
    images: int
    shard_prefix: str
    replayed: bool

    @property
    def labels_spent(self) -> int:
        """One label per image: annotation is priced per image, so the two are
        the same number by definition rather than by coincidence."""
        return self.images


def dynamodb(resource: Any = None) -> Any:
    """The resource, built on demand and passed in.

    For `run.registration.runs_table`'s reason: one created at import time fixes
    the region and credentials before a caller has had a chance to choose them.
    """
    return resource if resource is not None else boto3.resource("dynamodb")


def _audit_item(
    run_id: RunId, cycle: Cycle, digest: str, images: int, at: datetime
) -> dict[str, Any]:
    """The evidence that a charge happened.

    Deliberately holds a count and a digest rather than the image IDs. A
    thousand IDs is well inside the 400 KB item limit and still the wrong place
    for them: the shards under `shard_prefix` are the record of *which* images
    were bought, and the digest is what lets a claimed batch be checked against
    this item without storing it twice. Same split as the gate report, which
    lives in S3 with a pointer here.
    """
    return {
        "run_id": str(run_id),
        "event": purchase_event(cycle, digest),
        "cycle": cycle,
        "digest": digest,
        "images": images,
        "shard_prefix": purchase_shards_prefix(run_id, cycle),
        # Normalized to UTC so the column sorts chronologically, which is the
        # only reason anyone reads it -- the sort key already orders by cycle.
        "created_at": at.astimezone(UTC).isoformat(),
    }


def _cancellation_codes(error: ClientError) -> list[str]:
    """Which item of the transaction refused, by position.

    `CancellationReasons` is parallel to the items submitted, with `"None"` where
    an item was fine. It is absent on failures that are not cancellations, which
    is why the caller checks the error code first.
    """
    reasons = error.response.get("CancellationReasons") or []
    return [str(reason.get("Code")) for reason in reasons]


def charge(
    resource: Any,
    run: RunRegistration,
    cycle: Cycle,
    image_ids: Sequence[ImageId],
) -> Receipt:
    """Claim the idempotency key and debit the ledger, atomically.

    Returns a receipt whether the charge is new or a replay. Raises
    `OverBudgetError` when the cycle cannot cover the batch, and lets anything
    else -- a throttle, a missing table, a denied call -- propagate, because a
    retry is the right answer to those and is the wrong answer to the two
    conditions here.
    """
    digest = batch_digest(image_ids)
    event = purchase_event(cycle, digest)
    images = len(image_ids)

    # Checked before the call rather than in the condition. The create branch
    # seeds `remaining` with the registered budget and subtracts in the same
    # expression, so a batch larger than the whole cycle's cap would write a
    # negative on the very first purchase -- the one case the `remaining >= :n`
    # condition cannot catch, because there is no `remaining` yet.
    if images > run.label_budget_per_cycle:
        raise OverBudgetError(
            f"{images} labels is more than cycle {cycle}'s entire budget of "
            f"{run.label_budget_per_cycle}"
        )

    client = resource.meta.client
    try:
        client.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": table_name(Table.AUDIT_LOG),
                        "Item": _audit_item(run.run_id, cycle, digest, images, datetime.now(UTC)),
                        "ConditionExpression": "attribute_not_exists(#event)",
                        "ExpressionAttributeNames": {"#event": "event"},
                    }
                },
                {
                    "Update": {
                        "TableName": table_name(Table.LABEL_BUDGET),
                        "Key": {"run_id": str(run.run_id), "cycle": cycle},
                        # Creation and the first debit in one expression, so the
                        # item's only writer is the step that spends against it
                        # and its value never rises.
                        "UpdateExpression": (
                            "SET remaining = if_not_exists(remaining, :budget) - :n"
                        ),
                        "ConditionExpression": (
                            "attribute_not_exists(remaining) OR remaining >= :n"
                        ),
                        "ExpressionAttributeValues": {
                            ":budget": run.label_budget_per_cycle,
                            ":n": images,
                        },
                    }
                },
            ]
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != _TRANSACTION_CANCELLED:
            raise

        codes = _cancellation_codes(error)

        # The audit condition is checked first, and the order matters. When both
        # refuse, this is a retry of a purchase that already succeeded: the
        # budget was debited by the original, so reporting "over budget" would
        # send the caller to buy less when the right answer is that it is done.
        if codes[_AUDIT_ITEM : _AUDIT_ITEM + 1] == [_CANCELLED_BY_CONDITION]:
            log.info("%s cycle %d: %s is a replay", run.run_id, cycle, digest[:12])
            return Receipt(
                run_id=run.run_id,
                cycle=cycle,
                event=event,
                digest=digest,
                images=images,
                shard_prefix=purchase_shards_prefix(run.run_id, cycle),
                replayed=True,
            )

        if codes[_BUDGET_ITEM : _BUDGET_ITEM + 1] == [_CANCELLED_BY_CONDITION]:
            raise OverBudgetError(
                f"cycle {cycle} of {run.run_id} has less than {images} labels left of its "
                f"{run.label_budget_per_cycle}. Nothing was charged and no label was read."
            ) from None

        raise

    log.info("%s cycle %d: charged %d labels", run.run_id, cycle, images)
    return Receipt(
        run_id=run.run_id,
        cycle=cycle,
        event=event,
        digest=digest,
        images=images,
        shard_prefix=purchase_shards_prefix(run.run_id, cycle),
        replayed=False,
    )


def remaining(resource: Any, run_id: RunId, cycle: Cycle) -> int | None:
    """What is left of one cycle's cap, or `None` before its first purchase.

    `None` rather than the registered budget, because "nothing has been spent"
    and "the cap is 1,000" are different facts and this table only knows the
    first. The cap is on the registration.
    """
    table = resource.Table(table_name(Table.LABEL_BUDGET))
    item = table.get_item(Key={"run_id": str(run_id), "cycle": cycle}, ConsistentRead=True).get(
        "Item"
    )
    return int(item["remaining"]) if item else None


@dataclass(frozen=True, slots=True)
class Oracle:
    """The shop, open for one run.

    Everything a purchase needs that does not vary per batch, bound once: the
    run whose ledger is charged, the partition whose cohorts gate the request,
    and how a label is fetched. A cycle then asks for a batch and nothing else.

    Bound rather than passed because three of the four are the same for a run's
    whole life, and a per-call `cohorts` is a per-call opportunity to hand the
    gate a different partition from the one the run was registered against.
    """

    resource: Any
    run: RunRegistration
    cohorts: Cohorts
    fetch: Fetch

    def __post_init__(self) -> None:
        # The gate is only the eval guarantee while it is gating the partition
        # this run's bootstrap and eval were drawn from. A mismatch here would
        # refuse and admit images by the wrong draw entirely.
        if self.cohorts.partition_version != self.run.partition_version:
            raise ValueError(
                f"{self.run.run_id} is registered against partition "
                f"v{self.run.partition_version} and the cohort index is "
                f"v{self.cohorts.partition_version}"
            )

    def charge(self, cycle: Cycle, image_ids: Sequence[ImageId]) -> Receipt:
        return charge(self.resource, self.run, cycle, image_ids)

    def remaining(self, cycle: Cycle) -> int | None:
        return remaining(self.resource, self.run.run_id, cycle)

    def purchase(
        self, cycle: Cycle, image_ids: Sequence[ImageId]
    ) -> tuple[Receipt, tuple[SoldLabel, ...]]:
        """Gate, charge, then serve. In that order, which is the whole guarantee.

        The gate runs first and on image IDs alone, so a batch reaching into
        `eval` is refused before a key exists. The charge runs second, so a batch
        nobody can pay for reads nothing. Only then is a label fetched.

        Returns the receipt beside the labels rather than writing shards here:
        what a purchase *is* is the charge, and where its labels are filed is the
        shard writer's business.
        """
        check_purchasable(self.cohorts, image_ids)
        receipt = self.charge(cycle, image_ids)
        return receipt, tuple(self.fetch(image_id) for image_id in image_ids)

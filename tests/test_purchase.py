"""Tests for the charge, against a real transaction engine.

The three properties under test -- no double charge, no overspend, no way for the
ledger and the audit trail to disagree -- are all properties of condition
expressions evaluated atomically. A fake that raised on the second call would
prove only that the fake was written to raise, so these run against `moto`, which
implements `TransactWriteItems` and its cancellation reasons rather than
simulating them. Every idempotency test is a genuine second write against tables
that already hold the first.

Two things are asserted after every refusal: that nothing was charged, and that
nothing was read. The second is what makes "charge, then serve" a guarantee
rather than an ordering that happens to hold in the code as written, so the
fetch used here counts its calls.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    AssignmentRow,
    ClassSetVersion,
    Cohort,
    Cycle,
    ImageId,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Selector,
    Split,
    Table,
    assignments_key,
    batch_digest,
    parse_purchase_event,
    purchase_event,
    purchase_labels_prefix,
    raw_label_key,
    table_name,
)
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle import cohorts as gate
from edge_ml_flywheel.oracle import labels as sold
from edge_ml_flywheel.oracle import purchase as buying
from edge_ml_flywheel.partition import assign

RUN = RunId("20260812t143355z-v0-skeleton")
CREATED = datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC)
COMMIT = "d" * 40
REGION = "us-east-1"
VERSION = PartitionVersion(0)
CYCLE = Cycle(1)

# A small budget, so the overspend cases are two purchases rather than two
# thousand. The arithmetic is the same at either size.
BUDGET = 5

POOL = tuple(ImageId(f"0000000{n}-0000000{n}") for n in range(1, 7))
EVAL = ImageId("000000ee-000000ee")
BOOTSTRAP = ImageId("000000bb-000000bb")

ASSIGNED: dict[ImageId, Cohort] = {
    **dict.fromkeys(POOL, Cohort.POOL),
    EVAL: Cohort.EVAL,
    BOOTSTRAP: Cohort.BOOTSTRAP,
}


def a_registration(**overrides: Any) -> RunRegistration:
    values: dict[str, Any] = {
        "run_id": RUN,
        "created_at": CREATED,
        "git_commit": COMMIT,
        "partition_version": VERSION,
        "class_set_version": ClassSetVersion(1),
        "recipe_version": RecipeVersion(1),
        "selector": Selector.UNCERTAINTY,
        "label_budget_per_cycle": BUDGET,
        "note": "purchase tests",
    }
    return RunRegistration(**(values | overrides))


class CountingFetch:
    """A `Fetch` that records what it was asked for.

    The refusal tests assert on `asked` rather than only on the exception,
    because "the purchase failed" and "no label was read" are different claims
    and only the second is the one the wall rests on.
    """

    def __init__(self, root: Path, cohorts: gate.Cohorts) -> None:
        self._fetch = sold.local_fetch(root, cohorts)
        self.asked: list[ImageId] = []

    def __call__(self, image_id: ImageId) -> sold.SoldLabel:
        self.asked.append(image_id)
        return self._fetch(image_id)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    assign.write_parquet(
        [AssignmentRow(image_id=image, cohort=cohort) for image, cohort in ASSIGNED.items()],
        tmp_path / assignments_key(VERSION),
    )
    for image_id, cohort in ASSIGNED.items():
        split = Split.VAL if cohort is Cohort.EVAL else Split.TRAIN
        path = tmp_path / raw_label_key(image_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": f"{image_id}.jpg",
                    "attributes": {
                        "weather": "clear",
                        "scene": "highway",
                        "timeofday": "daytime",
                    },
                    "frames": [
                        {
                            "timestamp": 10000,
                            "objects": [
                                {
                                    "category": "car",
                                    "box2d": {
                                        "x1": 380.4,
                                        "y1": 404.8,
                                        "x2": 402.3,
                                        "y2": 416.7,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def cohorts(root: Path) -> gate.Cohorts:
    return gate.Cohorts.read(root, VERSION)


@pytest.fixture
def resource(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """`audit_log` and `label_budget` with the key schemas `infra/tables.tf` declares.

    Fake credentials for `test_run.py`'s reason: without them a mock that failed
    to engage would reach a real account -- here, one holding a ledger with no
    undo on a charge.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        dynamodb.create_table(
            TableName=table_name(Table.AUDIT_LOG),
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "event", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "event", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName=table_name(Table.LABEL_BUDGET),
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "cycle", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "cycle", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield dynamodb


@pytest.fixture
def fetch(root: Path, cohorts: gate.Cohorts) -> CountingFetch:
    return CountingFetch(root, cohorts)


@pytest.fixture
def oracle(resource: Any, cohorts: gate.Cohorts, fetch: CountingFetch) -> buying.Oracle:
    return buying.Oracle(resource=resource, run=a_registration(), cohorts=cohorts, fetch=fetch)


def audit_items(resource: Any) -> list[dict[str, Any]]:
    return list(resource.Table(table_name(Table.AUDIT_LOG)).scan()["Items"])


class TestTheIdempotencyKey:
    def test_the_same_batch_in_a_different_order_is_one_purchase(self) -> None:
        """Selection ranks its output, and a retry can re-rank under a tie.

        If order changed the key, that retry would pay a second time for the
        same thousand images.
        """
        assert batch_digest(POOL[:3]) == batch_digest(tuple(reversed(POOL[:3])))

    def test_a_different_batch_is_a_different_purchase(self) -> None:
        assert batch_digest(POOL[:3]) != batch_digest(POOL[:4])

    def test_the_event_carries_the_cycle_and_the_digest(self) -> None:
        digest = batch_digest(POOL[:3])
        assert parse_purchase_event(purchase_event(CYCLE, digest)) == (CYCLE, digest)

    def test_the_cycle_is_padded_so_the_sort_key_orders(self) -> None:
        """The sort key is a string, so `cycle=10` would otherwise precede 2."""
        digest = batch_digest(POOL[:3])
        assert purchase_event(Cycle(2), digest) < purchase_event(Cycle(10), digest)

    def test_a_batch_that_repeats_an_image_has_no_digest(self) -> None:
        """Deduplicating would give two different requests one name, and the
        second is one the budget would charge twice for."""
        with pytest.raises(ValueError, match="repeats"):
            batch_digest([POOL[0], POOL[0]])


class TestChargingOnce:
    def test_a_purchase_debits_the_cycle(self, oracle: buying.Oracle) -> None:
        receipt = oracle.charge(CYCLE, POOL[:2])
        assert receipt.replayed is False
        assert receipt.labels_spent == 2
        assert oracle.remaining(CYCLE) == BUDGET - 2

    def test_the_ledger_item_is_created_by_the_first_debit(self, oracle: buying.Oracle) -> None:
        """No seeding step, so the item's only writer is the one that spends."""
        assert oracle.remaining(CYCLE) is None
        oracle.charge(CYCLE, POOL[:2])
        assert oracle.remaining(CYCLE) == BUDGET - 2

    def test_a_second_distinct_batch_debits_again(self, oracle: buying.Oracle) -> None:
        oracle.charge(CYCLE, POOL[:2])
        oracle.charge(CYCLE, POOL[2:4])
        assert oracle.remaining(CYCLE) == BUDGET - 4

    def test_the_audit_item_records_the_charge(self, oracle: buying.Oracle, resource: Any) -> None:
        receipt = oracle.charge(CYCLE, POOL[:2])
        (item,) = audit_items(resource)
        assert item["event"] == receipt.event
        assert int(item["images"]) == 2
        assert item["labels_prefix"] == purchase_labels_prefix(RUN, CYCLE)


class TestRefusingTheSecondCharge:
    def test_replaying_a_purchase_does_not_debit_again(self, oracle: buying.Oracle) -> None:
        """The property with no undo. A retried purchase that double debits
        overstates the cost of every cycle after it."""
        first = oracle.charge(CYCLE, POOL[:2])
        second = oracle.charge(CYCLE, POOL[:2])

        assert first.replayed is False
        assert second.replayed is True
        assert second.event == first.event
        assert oracle.remaining(CYCLE) == BUDGET - 2

    def test_a_replay_leaves_one_audit_item(self, oracle: buying.Oracle, resource: Any) -> None:
        oracle.charge(CYCLE, POOL[:2])
        oracle.charge(CYCLE, POOL[:2])
        assert len(audit_items(resource)) == 1

    def test_a_reordered_replay_is_still_a_replay(self, oracle: buying.Oracle) -> None:
        oracle.charge(CYCLE, POOL[:3])
        assert oracle.charge(CYCLE, tuple(reversed(POOL[:3]))).replayed is True
        assert oracle.remaining(CYCLE) == BUDGET - 3

    def test_the_same_batch_in_another_cycle_is_a_new_purchase(self, oracle: buying.Oracle) -> None:
        """The key is `(run, cycle, digest)`, and a later cycle legitimately
        re-buys nothing -- but the budget is per cycle, so it must charge."""
        oracle.charge(CYCLE, POOL[:2])
        later = oracle.charge(Cycle(2), POOL[:2])

        assert later.replayed is False
        assert oracle.remaining(CYCLE) == BUDGET - 2
        assert oracle.remaining(Cycle(2)) == BUDGET - 2


class TestTheBudget:
    def test_a_batch_over_what_is_left_is_refused(self, oracle: buying.Oracle) -> None:
        oracle.charge(CYCLE, POOL[:4])
        with pytest.raises(buying.OverBudgetError, match="less than"):
            oracle.charge(CYCLE, POOL[4:6])

    def test_a_refused_batch_debits_nothing(self, oracle: buying.Oracle) -> None:
        """A refusal is not a partial charge."""
        oracle.charge(CYCLE, POOL[:4])
        with pytest.raises(buying.OverBudgetError):
            oracle.charge(CYCLE, POOL[4:6])
        assert oracle.remaining(CYCLE) == BUDGET - 4

    def test_a_refused_batch_writes_no_audit_item(
        self, oracle: buying.Oracle, resource: Any
    ) -> None:
        """The atomic half that a two-write implementation would get wrong: an
        audit item with no debit is a purchase the budget never saw."""
        oracle.charge(CYCLE, POOL[:4])
        with pytest.raises(buying.OverBudgetError):
            oracle.charge(CYCLE, POOL[4:6])
        assert len(audit_items(resource)) == 1

    def test_a_batch_over_the_whole_cap_is_refused_before_the_call(
        self, oracle: buying.Oracle, resource: Any
    ) -> None:
        """The one case `remaining >= :n` cannot catch, because on the first
        purchase there is no `remaining` yet -- the create branch would write a
        negative."""
        with pytest.raises(buying.OverBudgetError, match="entire budget"):
            oracle.charge(CYCLE, POOL[:6])
        assert oracle.remaining(CYCLE) is None
        assert audit_items(resource) == []

    def test_spending_the_cap_exactly_is_allowed(self, oracle: buying.Oracle) -> None:
        """`>=`, not `>`. A cycle may spend its last label."""
        oracle.charge(CYCLE, POOL[:BUDGET])
        assert oracle.remaining(CYCLE) == 0

    def test_a_replay_of_the_batch_that_emptied_the_budget_still_replays(
        self, oracle: buying.Oracle
    ) -> None:
        """Both conditions refuse here, and the order they are read in decides
        the answer. Reporting "over budget" would send the caller to buy less
        when the truth is that this purchase already happened.
        """
        first = oracle.charge(CYCLE, POOL[:BUDGET])
        second = oracle.charge(CYCLE, POOL[:BUDGET])
        assert second.replayed is True
        assert second.event == first.event
        assert oracle.remaining(CYCLE) == 0


class TestChargeThenServe:
    def test_a_purchase_returns_the_labels(self, oracle: buying.Oracle) -> None:
        receipt, labels = oracle.purchase(CYCLE, POOL[:2])
        assert [label.image_id for label in labels] == list(POOL[:2])
        assert labels[0].boxes == (Box("car", 380.4, 404.8, 402.3, 416.7),)
        assert receipt.labels_spent == 2

    def test_a_batch_the_gate_refuses_reads_nothing_and_charges_nothing(
        self, oracle: buying.Oracle, fetch: CountingFetch, resource: Any
    ) -> None:
        """The gate runs before the charge, so an eval image costs nothing and,
        more importantly, is never opened."""
        with pytest.raises(gate.NotPurchasableError, match="in eval"):
            oracle.purchase(CYCLE, [POOL[0], EVAL])

        assert fetch.asked == []
        assert audit_items(resource) == []
        assert oracle.remaining(CYCLE) is None

    def test_a_batch_over_budget_reads_nothing(
        self, oracle: buying.Oracle, fetch: CountingFetch
    ) -> None:
        """Charge, then serve -- so a batch nobody can pay for opens no file."""
        oracle.purchase(CYCLE, POOL[:4])
        fetch.asked.clear()

        with pytest.raises(buying.OverBudgetError):
            oracle.purchase(CYCLE, POOL[4:6])
        assert fetch.asked == []

    def test_a_bootstrap_image_is_refused_by_the_gate(
        self, oracle: buying.Oracle, fetch: CountingFetch
    ) -> None:
        """Already owned. Paying again is wasted budget rather than leakage, and
        it is still refused before anything is read."""
        with pytest.raises(gate.NotPurchasableError, match="in bootstrap"):
            oracle.purchase(CYCLE, [BOOTSTRAP])
        assert fetch.asked == []

    def test_a_replay_serves_the_same_labels_again(self, oracle: buying.Oracle) -> None:
        """Which is what lets the caller rewrite the same boxes over the same
        keys after a crash between the charge and the label write."""
        _, first = oracle.purchase(CYCLE, POOL[:2])
        receipt, second = oracle.purchase(CYCLE, POOL[:2])

        assert receipt.replayed is True
        assert first == second


class TestTheOracleIsBoundToItsRun:
    def test_a_cohort_index_from_another_partition_is_refused(
        self, resource: Any, cohorts: gate.Cohorts, fetch: CountingFetch
    ) -> None:
        """The gate is the eval guarantee only while it gates the partition the
        run's own bootstrap and eval were drawn from."""
        elsewhere = a_registration(partition_version=PartitionVersion(1))
        with pytest.raises(ValueError, match="registered against partition"):
            buying.Oracle(resource=resource, run=elsewhere, cohorts=cohorts, fetch=fetch)

    def test_two_runs_debit_separate_ledgers(
        self, resource: Any, cohorts: gate.Cohorts, fetch: CountingFetch
    ) -> None:
        """`run_id` is the partition key, so a second run cannot see the first's
        spend -- the blast-radius claim, at the ledger."""
        other = RunId("20260812t143355z-v0-control")
        first = buying.Oracle(resource, a_registration(), cohorts, fetch)
        second = buying.Oracle(resource, a_registration(run_id=other), cohorts, fetch)

        first.charge(CYCLE, POOL[:2])
        second.charge(CYCLE, POOL[:2])

        assert first.remaining(CYCLE) == BUDGET - 2
        assert second.remaining(CYCLE) == BUDGET - 2

    def test_the_same_batch_under_two_runs_is_two_purchases(
        self, resource: Any, cohorts: gate.Cohorts, fetch: CountingFetch
    ) -> None:
        other = RunId("20260812t143355z-v0-control")
        first = buying.Oracle(resource, a_registration(), cohorts, fetch)
        second = buying.Oracle(resource, a_registration(run_id=other), cohorts, fetch)

        assert first.charge(CYCLE, POOL[:2]).replayed is False
        assert second.charge(CYCLE, POOL[:2]).replayed is False
        assert len(audit_items(resource)) == 2

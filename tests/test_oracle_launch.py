"""The purchase against an account: the first time the oracle touches S3.

`oracle.purchase` and `oracle.cohorts` were written pure and local-path, and
their tests cover the charge and the gate against a real transaction engine. What
is this module's own is everything those two were deliberately kept away from:
that the batch comes out of the ranking rather than off an argument, that the
label read goes through the gate on its way to a key, and that the boxes land in
the format the next cycle's training channel reads.

Two things are asserted after every refusal, matching `test_purchase`: that
nothing was charged, and that nothing was written. A purchase that fails and
leaves a file is a cycle whose labels exist without a ledger entry.
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
    Buckets,
    Cohort,
    Cycle,
    ImageId,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Split,
    Table,
    assignments_key,
    new_model_version,
    purchase_labels_key,
    raw_label_key,
    selection_ranking_key,
    table_name,
)
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.oracle import cohorts as gate
from edge_ml_flywheel.oracle import labels as sold
from edge_ml_flywheel.oracle import launch as buying
from edge_ml_flywheel.oracle import purchase as charging
from edge_ml_flywheel.partition import assign
from edge_ml_flywheel.run import registration as reg
from edge_ml_flywheel.selection import ranking
from edge_ml_flywheel.training import labels as training_labels

RUN = RunId("20260812t143355z-v1-uncertainty")
CYCLE = Cycle(1)
VERSION = new_model_version(RUN, CYCLE)
REGION = "us-east-1"
ACCOUNT = "123456789012"
BUCKETS = Buckets.for_account(ACCOUNT)
PARTITION = PartitionVersion(0)

BUDGET = 3

POOL = tuple(ImageId(f"0000000{n}-0000000{n}") for n in range(1, 6))
EVAL = ImageId("000000ee-000000ee")
BOOTSTRAP = ImageId("000000bb-000000bb")

ASSIGNED: dict[ImageId, Cohort] = {
    **dict.fromkeys(POOL, Cohort.POOL),
    EVAL: Cohort.EVAL,
    BOOTSTRAP: Cohort.BOOTSTRAP,
}

BOX = Box("car", 380.4, 404.8, 402.3, 416.7)


def a_label_document(image_id: ImageId) -> dict[str, Any]:
    return {
        "name": f"{image_id}.jpg",
        "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
        "frames": [
            {
                "timestamp": 10000,
                "objects": [
                    {
                        "category": BOX.category,
                        "box2d": {"x1": BOX.x1, "y1": BOX.y1, "x2": BOX.x2, "y2": BOX.y2},
                    }
                ],
            }
        ],
    }


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[boto3.Session]:
    """An account with both buckets, all three tables, the partition and the labels.

    Every cohort gets a label object in the bucket, `eval` included. The gate has
    to be what refuses it -- a missing file would make a removed check still look
    like a pass here and fail only in production.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        session = boto3.Session(region_name=REGION)
        client = session.client("s3")
        for bucket in (BUCKETS.data, BUCKETS.artifacts):
            client.create_bucket(Bucket=bucket)

        _tables(session)
        _register(session)

        local = tmp_path / "assignments.parquet"
        assign.write_parquet(
            [AssignmentRow(image_id=image, cohort=cohort) for image, cohort in ASSIGNED.items()],
            local,
        )
        client.upload_file(str(local), BUCKETS.data, assignments_key(PARTITION))

        for image_id, cohort in ASSIGNED.items():
            split = Split.VAL if cohort is Cohort.EVAL else Split.TRAIN
            client.put_object(
                Bucket=BUCKETS.data,
                Key=raw_label_key(image_id, split),
                Body=json.dumps(a_label_document(image_id)).encode(),
            )

        yield session


def _tables(aws: boto3.Session) -> None:
    resource = aws.resource("dynamodb")
    resource.create_table(
        TableName=table_name(Table.RUNS),
        KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    resource.create_table(
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
    resource.create_table(
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


def _register(aws: boto3.Session, **overrides: Any) -> None:
    values: dict[str, Any] = {
        "run_id": RUN,
        "created_at": datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC),
        "git_commit": "d" * 40,
        "partition_version": PARTITION,
        "recipe_version": RecipeVersion(1),
        "label_budget_per_cycle": BUDGET,
        "note": "purchase tests",
    }
    table = reg.runs_table(aws.resource("dynamodb"))
    table.put_item(Item=reg.to_item(RunRegistration(**(values | overrides))))


def _rank(aws: boto3.Session, tmp_path: Path, batch: tuple[ImageId, ...], pool: Any = POOL) -> None:
    """Write a ranking naming `batch`, the way `selection.launch` would."""
    local = tmp_path / "ranking.parquet"
    ranking.write(ranking.rows(pool, dict.fromkeys(pool, 0.5), batch), local)
    aws.client("s3").upload_file(str(local), BUCKETS.artifacts, selection_ranking_key(RUN, CYCLE))


def _purchased(aws: boto3.Session, tmp_path: Path) -> dict[ImageId, tuple[Box, ...]]:
    """The purchase file, read back through the reader training uses.

    Into a directory of its own, because `training.labels.collect` walks a
    channel rather than opening a file -- and `tmp_path` already holds the
    assignments and the ranking, which are parquets of other schemas.
    """
    root = tmp_path / "read-back"
    root.mkdir(exist_ok=True)
    aws.client("s3").download_file(
        BUCKETS.data, purchase_labels_key(RUN, CYCLE), str(root / "part-00000.parquet")
    )
    return training_labels.collect([root])


def _audit(aws: boto3.Session) -> list[dict[str, Any]]:
    return list(aws.resource("dynamodb").Table(table_name(Table.AUDIT_LOG)).scan()["Items"])


def _written(aws: boto3.Session) -> bool:
    found = aws.client("s3").list_objects_v2(
        Bucket=BUCKETS.data, Prefix=purchase_labels_key(RUN, CYCLE)
    )
    return bool(found.get("Contents"))


class TestBuyingTheBatch:
    def test_it_charges_for_what_the_ranking_selected(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        _rank(aws, tmp_path, POOL[:BUDGET])

        bought = buying.purchase(aws, RUN, CYCLE)

        assert bought.images == BUDGET
        assert bought.replayed is False
        assert len(_audit(aws)) == 1

    def test_the_boxes_land_where_training_reads_them(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """Read back through `training.labels`, which is the reader of record: a
        bought label and a bootstrap one are the same two columns decoded by one
        function, and that is what makes the cumulative labeled set one thing."""
        _rank(aws, tmp_path, POOL[:BUDGET])
        buying.purchase(aws, RUN, CYCLE)

        labels = _purchased(aws, tmp_path)

        assert set(labels) == set(POOL[:BUDGET])
        assert labels[POOL[0]] == (BOX,)

    def test_what_is_left_of_the_pool_is_what_the_loop_branches_on(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The ranking names exactly the images this cycle had left to buy, so
        subtracting the batch is the number by definition."""
        _rank(aws, tmp_path, POOL[:BUDGET])

        assert buying.purchase(aws, RUN, CYCLE).pool_remaining == len(POOL) - BUDGET

    def test_the_ledger_is_debited(self, aws: boto3.Session, tmp_path: Path) -> None:
        _rank(aws, tmp_path, POOL[:BUDGET])
        buying.purchase(aws, RUN, CYCLE)

        assert charging.remaining(aws.resource("dynamodb"), RUN, CYCLE) == 0


class TestRetryingIt:
    def test_a_second_purchase_of_one_cycle_is_a_replay(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The property with no undo. The batch comes out of the ranking rather
        than off an argument, so a retry names the same images, hashes to the same
        digest, and replays instead of buying a second batch."""
        _rank(aws, tmp_path, POOL[:BUDGET])

        first = buying.purchase(aws, RUN, CYCLE)
        second = buying.purchase(aws, RUN, CYCLE)

        assert first.replayed is False
        assert second.replayed is True
        assert len(_audit(aws)) == 1

    def test_a_replay_does_not_debit_again(self, aws: boto3.Session, tmp_path: Path) -> None:
        _rank(aws, tmp_path, POOL[:BUDGET])
        buying.purchase(aws, RUN, CYCLE)
        buying.purchase(aws, RUN, CYCLE)

        assert charging.remaining(aws.resource("dynamodb"), RUN, CYCLE) == 0

    def test_a_replay_rewrites_the_same_labels(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Which is what makes the label write safe to repeat after a crash
        between the charge and the file."""
        _rank(aws, tmp_path, POOL[:BUDGET])
        buying.purchase(aws, RUN, CYCLE)
        first = _purchased(aws, tmp_path)

        buying.purchase(aws, RUN, CYCLE)
        assert _purchased(aws, tmp_path) == first


class TestRefusals:
    def test_a_cycle_that_never_selected_is_refused(self, aws: boto3.Session) -> None:
        with pytest.raises(SystemExit, match="what this cycle chose to buy"):
            buying.purchase(aws, RUN, CYCLE)

        assert _audit(aws) == []
        assert not _written(aws)

    def test_an_eval_image_in_the_batch_refuses_the_whole_purchase(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The centrepiece. The eval label is in the bucket and this role could
        read it, so the refusal can only be the gate -- and it happens while the
        batch is still a list of image IDs, before a key exists.
        """
        _rank(aws, tmp_path, (POOL[0], EVAL), pool=(*POOL, EVAL))

        with pytest.raises(gate.NotPurchasableError, match="in eval"):
            buying.purchase(aws, RUN, CYCLE)

        assert _audit(aws) == []
        assert not _written(aws)

    def test_a_bootstrap_image_is_refused_too(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Already owned. Paying again is wasted budget rather than leakage, and
        it is still refused before anything is read."""
        _rank(aws, tmp_path, (BOOTSTRAP,), pool=(*POOL, BOOTSTRAP))

        with pytest.raises(gate.NotPurchasableError, match="in bootstrap"):
            buying.purchase(aws, RUN, CYCLE)

        assert not _written(aws)

    def test_a_batch_over_the_budget_is_refused(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Under its own name rather than as a generic failure: a cycle can
        legitimately run out, and the caller's answer is to buy less or stop."""
        _rank(aws, tmp_path, POOL[: BUDGET + 1])

        with pytest.raises(charging.OverBudgetError):
            buying.purchase(aws, RUN, CYCLE)

        assert _audit(aws) == []
        assert not _written(aws)

    def test_a_ranking_selecting_nothing_is_refused(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        _rank(aws, tmp_path, ())

        with pytest.raises(ValueError, match="no image as selected"):
            buying.purchase(aws, RUN, CYCLE)

    def test_a_run_registered_against_another_partition_is_refused(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The gate is the eval guarantee only while it gates the partition the
        run's own bootstrap and eval were drawn from."""
        _register(aws, partition_version=PartitionVersion(1))
        _rank(aws, tmp_path, POOL[:BUDGET])

        with pytest.raises(SystemExit, match="no assignments"):
            buying.purchase(aws, RUN, CYCLE)


class TestTheLabelRead:
    def test_no_key_this_module_builds_addresses_a_val_label(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The S3 fetch is the local one with a client in it, so the guarantee is
        the same one: the key comes from `label_key`, which routes through the
        gate, and there is no image for which it returns a `val` path.
        """
        cohorts = gate.Cohorts(partition_version=PARTITION, of_image=ASSIGNED)
        produced = []
        for image_id in ASSIGNED:
            try:
                produced.append(sold.label_key(cohorts, image_id))
            except gate.NotPurchasableError:
                continue

        assert produced == [sold.label_key(cohorts, image) for image in POOL]
        assert not any(f"/{Split.VAL.value}/" in key for key in produced)

    def test_the_fetch_reads_the_object_the_gate_admitted(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        cohorts = gate.Cohorts(partition_version=PARTITION, of_image=ASSIGNED)
        fetch = sold.s3_fetch(aws.client("s3"), BUCKETS.data, cohorts)

        assert fetch(POOL[0]).boxes == (BOX,)

    def test_the_fetch_refuses_an_eval_image_without_a_request(self, aws: boto3.Session) -> None:
        """The eval object exists in the bucket, so this can only be the gate."""
        cohorts = gate.Cohorts(partition_version=PARTITION, of_image=ASSIGNED)
        fetch = sold.s3_fetch(aws.client("s3"), BUCKETS.data, cohorts)

        with pytest.raises(gate.NotPurchasableError, match="in eval"):
            fetch(EVAL)

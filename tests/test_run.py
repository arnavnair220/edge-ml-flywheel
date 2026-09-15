"""Tests for the registration item and the write that claims a run.

Two things are under test and they need different machinery. The encoding is
pure, so it is tested by round-tripping a dataclass through a dict. The
conditional put is not testable that way at all -- a fake that raises on the
second call proves only that the fake was written to raise -- so these run
against `moto`, which implements condition expressions rather than simulating
them, and the collision test is a genuine second write against a table that
already holds the item.

`Decimal` appears deliberately in one test. It is what a real DynamoDB read
returns for an `N` attribute, and an unconverted one compares unequal to the int
it encodes, so the round-trip has to survive it rather than only surviving the
dict `to_item` produced.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    ClassSetVersion,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Selector,
    Table,
    columns,
    new_run_id,
    parse_run_id,
    table_name,
)
from edge_ml_flywheel.run import __main__ as cli
from edge_ml_flywheel.run import control as ctl
from edge_ml_flywheel.run import registration as reg

RUN = RunId("20260812t143355z-v0-skeleton")
OTHER = RunId("20260812t143355z-v0-control")
CREATED = datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC)
COMMIT = "b" * 40
REGION = "us-east-1"


class _FrozenClock:
    """A clock stuck on one second.

    `run_id` is minted from the current second plus a slug, so two runs collide
    only when they are started in the same second. Reproducing that needs the
    clock held still -- there is no other way for a test to get the CLI to mint
    the same id twice.
    """

    @staticmethod
    def now(tz: Any = None) -> datetime:
        return CREATED


def a_registration(**overrides: Any) -> RunRegistration:
    values: dict[str, Any] = {
        "run_id": RUN,
        "created_at": CREATED,
        "git_commit": COMMIT,
        "partition_version": PartitionVersion(0),
        "class_set_version": ClassSetVersion(1),
        "recipe_version": RecipeVersion(1),
        "selector": Selector.UNCERTAINTY,
        "label_budget_per_cycle": 1000,
        "note": "first skeleton run",
    }
    return RunRegistration(**(values | overrides))


@pytest.fixture
def tables(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Any]]:
    """The `runs` and `fleet_config` tables, with the key schemas
    `infra/tables.tf` declares.

    Both, because `register` writes to both: the registration claims the name
    and the control item opens the cycle counter, and a fixture with only the
    first would make the CLI tests fail on a missing table rather than on
    whatever they are about.

    Credentials are set to obvious fakes rather than left to the environment.
    Without them boto3 falls back to whatever the developer's machine has
    configured, and a test whose mock failed to engage would reach a real
    account -- against tables whose whole purpose is refusing to be written
    twice.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        resource = boto3.resource("dynamodb", region_name=REGION)
        resource.create_table(
            TableName=table_name(Table.RUNS),
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        resource.create_table(
            TableName=table_name(Table.FLEET_CONFIG),
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "entity", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "entity", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield reg.runs_table(resource), ctl.fleet_config_table(resource)


@pytest.fixture
def table(tables: tuple[Any, Any]) -> Any:
    """The `runs` table alone, for the tests that are only about registration."""
    return tables[0]


@pytest.fixture
def fleet(tables: tuple[Any, Any]) -> Any:
    """The `fleet_config` table alone, for the tests about the cycle counter."""
    return tables[1]


class TestItemEncoding:
    def test_round_trips(self) -> None:
        entry = a_registration()
        assert reg.from_item(reg.to_item(entry)) == entry

    def test_covers_every_field(self) -> None:
        """The item is write-once, so a field missing from it is missing forever.

        Asserted against the dataclass rather than a literal list, which is the
        one place a comparison to the schema of record is worth more than a
        golden value: the failure this catches is a field added to
        `RunRegistration` and not to `to_item`, and a literal list here would
        have to be updated by the same person who forgot.
        """
        assert set(reg.to_item(a_registration())) == set(columns(RunRegistration))

    def test_survives_the_decimals_dynamodb_returns(self) -> None:
        item = reg.to_item(a_registration()) | {
            "partition_version": Decimal(0),
            "class_set_version": Decimal(1),
            "recipe_version": Decimal(1),
            "label_budget_per_cycle": Decimal(1000),
        }
        assert reg.from_item(item) == a_registration()

    def test_normalizes_created_at_to_utc(self) -> None:
        """The column is read for chronological order, which offsets break.

        `2026-08-12T09:33:55-05:00` sorts before `2026-08-12T08:00:00+00:00` as a
        string and after it in time.
        """
        elsewhere = CREATED.astimezone(timezone(timedelta(hours=-5)))
        item = reg.to_item(a_registration(created_at=elsewhere))
        assert item["created_at"] == "2026-08-12T14:33:55+00:00"

    def test_the_item_is_json_serializable(self) -> None:
        """`show` prints it, and a stray enum or datetime would fail there."""
        assert json.loads(json.dumps(reg.to_item(a_registration())))["selector"] == "uncertainty"

    def test_stores_the_selector_as_its_value(self) -> None:
        item = reg.to_item(a_registration(selector=Selector.CERTAINTY))
        assert item["selector"] == "certainty"
        assert type(item["selector"]) is str


class TestRegister:
    def test_writes_a_registration_that_reads_back(self, table: Any) -> None:
        entry = a_registration()
        reg.register(table, entry)
        assert reg.read(table, RUN) == entry

    def test_refuses_a_second_write_under_the_same_id(self, table: Any) -> None:
        """The collision guard, against a real condition expression.

        This is what makes registration mandatory rather than a nicety: a second
        run adopting the first's `run_id` would read its ledger, its champion and
        its locks, because all three are addressed by `run_id` alone.
        """
        reg.register(table, a_registration())
        with pytest.raises(reg.RunAlreadyRegisteredError, match=RUN):
            reg.register(table, a_registration(note="a second run in the same second"))

    def test_the_refused_write_leaves_the_first_registration_intact(self, table: Any) -> None:
        """A refusal is not a partial write. The first run's config survives it."""
        reg.register(table, a_registration())
        with pytest.raises(reg.RunAlreadyRegisteredError):
            reg.register(table, a_registration(selector=Selector.RANDOM, note="clobber"))

        stored = reg.read(table, RUN)
        assert stored is not None
        assert stored.selector is Selector.UNCERTAINTY
        assert stored.note == "first skeleton run"

    def test_two_runs_in_the_same_second_differ_by_slug(self, table: Any) -> None:
        """The ordinary resolution of a collision, and proof it is available.

        Second precision means the timestamp half can repeat; the slug is what
        keeps two runs started at once distinct without a random suffix.
        """
        started = CREATED
        first = new_run_id(started, "v0-skeleton")
        second = new_run_id(started, "v0-control")

        reg.register(table, a_registration(run_id=first))
        reg.register(table, a_registration(run_id=second, selector=Selector.RANDOM))

        stored = reg.read(table, second)
        assert stored is not None
        assert stored.selector is Selector.RANDOM

    def test_a_client_error_that_is_not_a_collision_is_not_swallowed(self, table: Any) -> None:
        """Only the condition failure becomes `RunAlreadyRegisteredError`.

        A denied `PutItem` and a colliding one are both `ClientError`, and
        reporting an IAM problem as "this run already exists" sends whoever is
        starting a run to change their slug over and over.
        """
        table.meta.client.delete_table(TableName=table_name(Table.RUNS))

        with pytest.raises(ClientError) as raised:
            reg.register(table, a_registration())
        assert raised.value.response["Error"]["Code"] == "ResourceNotFoundException"


class TestRead:
    def test_returns_none_for_a_run_never_minted(self, table: Any) -> None:
        assert reg.read(table, OTHER) is None

    def test_rejects_a_malformed_run_id_rather_than_querying(self, table: Any) -> None:
        """A bad run ID does not 404 -- it names a partition nobody is watching.

        So it is refused at the boundary, which is `parse_run_id`'s reason for
        being applied at every entry point rather than only at ingest.
        """
        with pytest.raises(ValueError, match="not a run ID"):
            reg.read(table, RunId("nonsense"))


REGISTER_ARGS = [
    "register",
    "--slug",
    "v1-uncertainty",
    "--selector",
    "uncertainty",
    "--partition-version",
    "0",
    "--class-set-version",
    "1",
    "--recipe-version",
    "1",
    "--label-budget",
    "1000",
    "--cycle-cap",
    "8",
    "--note",
    "first real loop",
]


class TestCli:
    """The buildspec's view: argv in, a run ID on stdout, everything else stderr.

    `table` is requested even where it is unused, because it is what puts the
    mock and the fake credentials in place. Without it these would reach a real
    account.
    """

    def test_register_prints_the_run_id_and_nothing_else_on_stdout(
        self, table: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The buildspec captures stdout into `RUN_ID`, so a stray line breaks it."""
        monkeypatch.setenv("CODEBUILD_RESOLVED_SOURCE_VERSION", COMMIT)
        cli.main(REGISTER_ARGS)

        run_id = capsys.readouterr().out.strip()
        stored = reg.read(table, parse_run_id(run_id))
        assert stored is not None
        assert stored.git_commit == COMMIT
        assert stored.label_budget_per_cycle == 1000
        assert stored.selector is Selector.UNCERTAINTY

    def test_register_refuses_without_a_git_commit(
        self, table: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rather than shelling out to `git rev-parse` in a tree that may be dirty."""
        monkeypatch.delenv("CODEBUILD_RESOLVED_SOURCE_VERSION", raising=False)
        with pytest.raises(SystemExit, match="no git commit"):
            cli.main(REGISTER_ARGS)

    def test_register_turns_a_collision_into_a_non_zero_exit(
        self, table: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The buildspec's guard: a collided run must not reach `post_build`.

        Both calls are given the same second, which is what a double-fired start
        looks like -- the id is minted from the clock, so only a frozen clock
        reproduces it here.
        """
        monkeypatch.setenv("CODEBUILD_RESOLVED_SOURCE_VERSION", COMMIT)
        monkeypatch.setattr(cli, "datetime", _FrozenClock)

        cli.main(REGISTER_ARGS)
        capsys.readouterr()

        with pytest.raises(SystemExit, match="already registered"):
            cli.main(REGISTER_ARGS)

    def test_show_prints_the_registration(
        self, table: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("CODEBUILD_RESOLVED_SOURCE_VERSION", COMMIT)
        cli.main(REGISTER_ARGS)
        run_id = capsys.readouterr().out.strip()

        cli.main(["show", "--run-id", run_id])
        shown = json.loads(capsys.readouterr().out)
        assert shown["registration"]["run_id"] == run_id
        assert shown["registration"]["note"] == "first real loop"
        assert shown["control"] == {
            "run_id": run_id,
            "entity": "run",
            "next_cycle": 0,
            "cycle_cap": 8,
        }

    def test_show_refuses_a_run_that_was_never_minted(self, table: Any) -> None:
        with pytest.raises(SystemExit, match="never registered"):
            cli.main(["show", "--run-id", OTHER])

    def test_rejects_a_selector_outside_the_three(self, table: Any) -> None:
        """Argparse `choices` off the enum, so a typo fails before the clock is read."""
        args = [*REGISTER_ARGS]
        args[args.index("--selector") + 1] = "confidence"
        with pytest.raises(SystemExit):
            cli.main(args)

    def test_rejects_a_non_positive_budget_before_writing(
        self, table: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`RunRegistration.__post_init__` runs before the put, not after it."""
        monkeypatch.setenv("CODEBUILD_RESOLVED_SOURCE_VERSION", COMMIT)
        args = [*REGISTER_ARGS]
        args[args.index("--label-budget") + 1] = "0"

        with pytest.raises(ValueError, match="must be positive"):
            cli.main(args)
        assert table.scan()["Count"] == 0

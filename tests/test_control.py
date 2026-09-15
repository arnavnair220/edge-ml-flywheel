"""Tests for the control plane: the cycle counter, the Lambda, and the ASL.

Three things with three different kinds of evidence.

The counter's encoding is pure and round-trips through a dict, like the
registration's. Its conditional put is not testable that way -- a fake that
raises on the second call proves only that the fake was written to raise -- so it
runs against `moto`, and the second write is a genuine second write.

The handler is tested at its dispatch and its adapter behaviour rather than at
the work it delegates: `launch.prepare` and `launch.request` have their own
callers and their own account. What is this module's own is which step name maps
to which function, and that a `SystemExit` raised by CLI-shaped code underneath
becomes an exception a Lambda can report rather than a process that exits.

The ASL is checked as a graph. It is JSON with no type checker and no linter over
it, and the failure it is exposed to is a `Next` naming a state that does not
exist -- which AWS rejects at deploy time and which a test can reject in a
second. The stub inventory is asserted too, so that a `Pass` state quietly
becoming permanent is a test that has to be edited rather than a thing nobody
notices.
"""

import json
import tarfile
from collections.abc import Iterator
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from edge_ml_flywheel.control import handler as ctrl
from edge_ml_flywheel.conventions import Cycle, RunId, Table, table_name
from edge_ml_flywheel.run import control as ctl
from edge_ml_flywheel.training import launch

RUN = RunId("20260812t143355z-v0-skeleton")
OTHER = RunId("20260812t143355z-v0-control")
REGION = "us-east-1"

# The definition the Terraform reads. A relative path from this file rather than
# a fixture, because a test that cannot find it should fail as a missing file and
# not as an empty parse.
ASL = Path(__file__).resolve().parents[1] / "infra" / "cycle.asl.json"

# The six steps the diagram draws as stubs. Named here so that implementing one
# and leaving it a `Pass` is a failing test.
STUBS = frozenset({"Score", "Evaluate", "Register", "Promote", "Select", "Purchase"})


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A `fleet_config` table with the composite key `infra/tables.tf` declares.

    Fake credentials for `test_run`'s reason: without them a test whose mock
    failed to engage reaches a real account, against the item that decides which
    cycle a run is on.
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
        yield ctl.fleet_config_table(resource)


class TestControlItem:
    def test_round_trips(self) -> None:
        control = ctl.RunControl(run_id=RUN, next_cycle=Cycle(3), cycle_cap=8)
        assert ctl.from_item(ctl.to_item(control)) == control

    def test_carries_the_entity_sort_key(self) -> None:
        """The item is unaddressable without it, and it is not a field."""
        item = ctl.to_item(ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8))
        assert item["entity"] == "run"

    def test_survives_the_decimals_a_real_read_returns(self) -> None:
        """DynamoDB returns every `N` as a `Decimal`, and a `Decimal` cycle
        compares unequal to the same cycle as an `int` everywhere downstream."""
        item = ctl.to_item(ctl.RunControl(run_id=RUN, next_cycle=Cycle(2), cycle_cap=8))
        read_back = ctl.from_item({**item, "next_cycle": Decimal(2), "cycle_cap": Decimal(8)})

        assert read_back.next_cycle == 2
        assert isinstance(read_back.next_cycle, int)
        assert not isinstance(read_back.next_cycle, Decimal)

    def test_refuses_a_cap_of_zero(self) -> None:
        """A run that can claim no cycle trains nothing and reports it as a
        clean finish, which is the failure worth refusing where it is stated."""
        with pytest.raises(ValueError, match="cycle cap must be at least"):
            ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=0)

    def test_refuses_a_negative_cycle(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            ctl.RunControl(run_id=RUN, next_cycle=Cycle(-1), cycle_cap=8)


class TestOpeningTheCounter:
    def test_writes_and_reads_back(self, fleet: Any) -> None:
        control = ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8)
        ctl.register(fleet, control)
        assert ctl.read(fleet, RUN) == control

    def test_a_run_with_no_counter_reads_as_none(self, fleet: Any) -> None:
        """`None` rather than a raise: the state machine's refusal to start is a
        better place to find out than a read that cannot say which run it was."""
        assert ctl.read(fleet, OTHER) is None

    def test_refuses_a_second_write(self, fleet: Any) -> None:
        """Not a duplicate record -- a reset. A second write puts a live run back
        at cycle 0 and sends the next execution over prefixes an earlier one has
        already written under.
        """
        ctl.register(fleet, ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8))

        with pytest.raises(ctl.RunAlreadyRegisteredError, match="already has a control item"):
            ctl.register(fleet, ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8))

    def test_two_runs_do_not_collide(self, fleet: Any) -> None:
        """`run_id` is the partition key, so a second run's counter is a second
        item rather than a refusal."""
        ctl.register(fleet, ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8))
        ctl.register(fleet, ctl.RunControl(run_id=OTHER, next_cycle=Cycle(0), cycle_cap=1))

        held = ctl.read(fleet, OTHER)
        assert held is not None
        assert held.cycle_cap == 1


class TestTheClaim:
    """The conditional update the state machine performs, run here against the
    same engine.

    **The expressions are read out of the ASL rather than restated.** A copy here
    would be a second spelling of the one write that stops two cycles overlapping,
    and it would go on passing after the definition changed under it -- which is
    the only way this test could be worse than nothing. What it exercises is that
    DynamoDB accepts a condition comparing two attributes of the item, and that
    the cap therefore stops the counter with nothing else having to look.

    The values are passed at the resource API's level and the ASL passes them at
    the wire level (`{"N": "1"}`), because those are two different clients of one
    expression. The expression is the part that can be wrong.
    """

    @pytest.fixture
    def claim(self) -> dict[str, Any]:
        definition = json.loads(ASL.read_text(encoding="utf-8"))
        return dict(definition["States"]["ClaimCycle"]["Arguments"])

    def _claim(self, fleet: Any, arguments: dict[str, Any], run_id: RunId) -> dict[str, Any]:
        return dict(
            fleet.update_item(
                Key={"run_id": str(run_id), "entity": "run"},
                UpdateExpression=arguments["UpdateExpression"],
                ConditionExpression=arguments["ConditionExpression"],
                ExpressionAttributeValues={":one": 1},
                ReturnValues=arguments["ReturnValues"],
            )
        )

    def test_hands_back_the_cycle_it_claimed_and_the_cap(
        self, fleet: Any, claim: dict[str, Any]
    ) -> None:
        """ALL_OLD rather than UPDATED_OLD, because the cap comes off the same
        call and a second read would be a second thing to be wrong."""
        ctl.register(fleet, ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=2))

        first = self._claim(fleet, claim, RUN)["Attributes"]
        assert int(first["next_cycle"]) == 0
        assert int(first["cycle_cap"]) == 2

        second = self._claim(fleet, claim, RUN)["Attributes"]
        assert int(second["next_cycle"]) == 1

    def test_the_cap_refuses_the_claim_past_it(self, fleet: Any, claim: dict[str, Any]) -> None:
        """The cap is enforced by the write that claims, not by a check upstream
        of it -- which is what makes a double-fired tick lose rather than open a
        second cycle against a spent budget."""
        ctl.register(fleet, ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=1))
        self._claim(fleet, claim, RUN)

        with pytest.raises(ClientError, match="ConditionalCheckFailed"):
            self._claim(fleet, claim, RUN)

    def test_a_run_with_no_counter_cannot_be_claimed(
        self, fleet: Any, claim: dict[str, Any]
    ) -> None:
        """`attribute_exists(next_cycle)` is why an unregistered run fails at the
        first state rather than several minutes into a training job."""
        with pytest.raises(ClientError, match="ConditionalCheckFailed"):
            self._claim(fleet, claim, OTHER)


class TestTheHandler:
    def test_refuses_a_step_it_does_not_have(self) -> None:
        with pytest.raises(ctrl.ControlError, match="no control step"):
            ctrl.handler({"step": "promote"})

    def test_refuses_an_event_with_no_step_at_all(self) -> None:
        with pytest.raises(ctrl.ControlError, match="no control step"):
            ctrl.handler({"run_id": RUN})

    def test_names_the_steps_it_does_have(self) -> None:
        """The message is what an operator reads off a failed execution."""
        with pytest.raises(ctrl.ControlError, match="prepare, train_request"):
            ctrl.handler({"step": "nope"})

    def test_the_step_names_are_the_ones_the_asl_passes(self) -> None:
        """The two files are an interface, and this is the half that can drift
        without either one failing to parse."""
        definition = json.loads(ASL.read_text(encoding="utf-8"))
        passed = {
            payload["step"]
            for payload in _payloads(definition["States"])
            if isinstance(payload.get("step"), str)
        }
        assert passed == set(ctrl.STEPS)

    def test_a_refusal_underneath_becomes_a_reportable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The functions underneath were written for a CLI and raise `SystemExit`.
        Left alone that is a `BaseException`, which kills the Lambda process and
        reports a runtime crash instead of the message an operator needs.
        """

        def refuse(*_: Any, **__: Any) -> dict[str, Any]:
            raise SystemExit("was never registered, so there is no partition")

        monkeypatch.setattr(ctrl, "STEPS", {**ctrl.STEPS, "prepare": refuse})

        with pytest.raises(ctrl.ControlError, match="was never registered"):
            ctrl.handler({"step": "prepare", "run_id": RUN, "cycle": 0})


class TestTheDeployedArchive:
    """The wrinkle the Lambda exists around: it has no git checkout, so the tree
    it packs is what Terraform deployed -- the package under `LAMBDA_TASK_ROOT`
    and the two script-mode root files in a layer at `/opt`."""

    def _deployment(self, root: Path) -> tuple[Path, Path]:
        package = root / "task" / "edge_ml_flywheel"
        (package / "training").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "training" / "job.py").write_text("# job", encoding="utf-8")

        layer = root / "opt"
        layer.mkdir()
        (layer / "train.py").write_text("# entry point", encoding="utf-8")
        (layer / "requirements.txt").write_text("ultralytics==8.4.146\n", encoding="utf-8")

        return package, layer

    def test_puts_the_entry_point_at_the_root_beside_the_package(self, tmp_path: Path) -> None:
        """Script mode requires `train.py` at the root of the archive, and the
        package below it -- which is the whole reason the two directories are
        separate arguments."""
        package, layer = self._deployment(tmp_path)
        names = _tar_names(launch.archive(package, layer))

        assert "train.py" in names
        assert "requirements.txt" in names
        assert "edge_ml_flywheel/training/job.py" in names

    def test_is_byte_identical_across_builds(self, tmp_path: Path) -> None:
        """The archive is filed under the write-once cycle prefix as evidence of
        what ran, so it has to be a function of the tree and not of when or where
        it was built. Timestamps, ownership and mode are all flattened."""
        package, layer = self._deployment(tmp_path)
        assert launch.archive(package, layer) == launch.archive(package, layer)


class TestTheDefinition:
    """The ASL as a graph. Nothing here runs it -- what it catches is the class of
    mistake JSON with no linter is exposed to."""

    @pytest.fixture
    def definition(self) -> dict[str, Any]:
        return dict(json.loads(ASL.read_text(encoding="utf-8")))

    def test_every_transition_names_a_state_that_exists(self, definition: dict[str, Any]) -> None:
        for states in _scopes(definition["States"]):
            for name, state in states.items():
                for target in _targets(state):
                    assert target in states, f"{name} goes to {target}, which is not a state"

    def test_every_state_either_transitions_or_ends(self, definition: dict[str, Any]) -> None:
        """A state with no `Next`, no `End` and no `Default` is a dangling branch
        that fails at deploy time."""
        terminal = {"Succeed", "Fail"}
        for states in _scopes(definition["States"]):
            for name, state in states.items():
                if state["Type"] in terminal:
                    continue
                assert state.get("End") or _targets(state), f"{name} goes nowhere"

    def test_the_stubs_are_the_six_the_diagram_draws(self, definition: dict[str, Any]) -> None:
        """A `Pass` becoming a `Task` is a step landing, and it should be a test
        that has to be edited rather than a change nobody notices."""
        passes = {name for name, state in definition["States"].items() if state["Type"] == "Pass"}
        assert passes == STUBS

    def test_every_stub_says_what_replaces_it(self, definition: dict[str, Any]) -> None:
        for name in STUBS:
            comment = definition["States"][name].get("Comment", "")
            assert comment.startswith("STUB."), f"{name} does not say what replaces it"

    def test_the_loop_closes(self, definition: dict[str, Any]) -> None:
        """The only edge that goes backwards, and the reason this is a flywheel
        rather than a pipeline."""
        choice = definition["States"]["MoreCycles"]
        assert choice["Choices"][0]["Next"] == definition["StartAt"]

    def test_a_rejected_cycle_still_reaches_the_purchase(self, definition: dict[str, Any]) -> None:
        """A failed gate leaves the champion in place and the labels stay bought,
        so there is no path on which a rejection costs the run its purchase."""
        assert definition["States"]["Passed"]["Default"] == "Select"
        assert definition["States"]["Promote"]["Next"] == "Select"

    def test_the_training_job_is_started_by_the_state_machine(
        self, definition: dict[str, Any]
    ) -> None:
        """`.sync`, so the wait, the retry and the stop-on-abort are the ASL's and
        nothing in this project polls a training job."""
        seed = definition["States"]["Train"]["ItemProcessor"]["States"]["TrainSeed"]
        assert seed["Resource"].endswith("sagemaker:createTrainingJob.sync")

    def test_the_claim_is_conditional_on_the_cap(self, definition: dict[str, Any]) -> None:
        """The single-flight lock. Without the condition this is a counter two
        executions can read the same value from."""
        claim = definition["States"]["ClaimCycle"]
        assert "next_cycle < cycle_cap" in claim["Arguments"]["ConditionExpression"]
        assert claim["Arguments"]["ReturnValues"] == "ALL_OLD"


def _scopes(states: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every set of states that share a namespace: the top level, and one per
    `Map` or `Parallel` branch. A `Next` may only name a state in its own scope,
    which is exactly what makes them separate scopes."""
    yield states
    for state in states.values():
        processor = state.get("ItemProcessor")
        if processor:
            yield from _scopes(processor["States"])
        for branch in state.get("Branches", ()):
            yield from _scopes(branch["States"])


def _targets(state: dict[str, Any]) -> list[str]:
    """Every state name this one can transition to."""
    targets = []
    if "Next" in state:
        targets.append(state["Next"])
    if "Default" in state:
        targets.append(state["Default"])
    for choice in state.get("Choices", ()):
        targets.append(choice["Next"])
    for catcher in state.get("Catch", ()):
        targets.append(catcher["Next"])
    return targets


def _payloads(states: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every Lambda payload in the definition, at any depth."""
    for scope in _scopes(states):
        for state in scope.values():
            arguments = state.get("Arguments")
            if isinstance(arguments, dict) and isinstance(arguments.get("Payload"), dict):
                yield arguments["Payload"]


def _tar_names(payload: bytes) -> set[str]:
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as tar:
        return set(tar.getnames())

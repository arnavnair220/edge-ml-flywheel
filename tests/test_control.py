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

The other failure is the one deploy time does *not* catch, because the definition
is valid and the expression is well formed: a field reading an execution input
the execution left out. That is a failed run rather than a failed apply, and it
is what the payload shape and the test over it are for.
"""

import json
import re
import subprocess
import sys
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
from edge_ml_flywheel.conventions import (
    PROJECT,
    Cycle,
    RunId,
    Table,
    new_model_version,
    table_name,
)
from edge_ml_flywheel.oracle import handler as oracle
from edge_ml_flywheel.run import control as ctl
from edge_ml_flywheel.training import launch

RUN = RunId("20260812t143355z-v0-skeleton")
OTHER = RunId("20260812t143355z-v0-control")
REGION = "us-east-1"

# What moto's STS hands back, which is what `ctl.cycle_machine_arn` composes the
# machine's ARN from.
ACCOUNT = "123456789012"

# The definition the Terraform reads. A relative path from this file rather than
# a fixture, because a test that cannot find it should fail as a missing file and
# not as an empty parse.
ASL = Path(__file__).resolve().parents[1] / "infra" / "cycle.asl.json"

# The steps the diagram draws as stubs. Empty, and kept rather than deleted: the
# assertion is now that every state of a cycle does something, which is the
# property that made the set worth writing down in the first place. A `Pass`
# appearing here again would be a step going back to being a placeholder.
STUBS: frozenset[str] = frozenset()

# The two Lambdas a cycle calls, as the ASL names them before Terraform fills
# them in. Two rather than one because the purchase reads `raw/labels/` and the
# control function is denied it, so the label wall is drawn between two functions.
CONTROL_FUNCTION = "${control_function_arn}"
ORACLE_FUNCTION = "${oracle_function_arn}"

# The execution inputs an execution may leave out, and the handler defaults. They
# are listed here rather than derived because the point of the test below is that
# leaving one out is allowed, and a list derived from the ASL would agree with the
# ASL by construction.
OPTIONAL_INPUTS = frozenset({"max_images", "replace", "instance_type"})

# The step name inside a payload expression, which is where it now lives.
STEP_IN_PAYLOAD = re.compile(r"'step':\s*'(\w+)'")


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

    def test_a_run_with_no_champion_writes_no_pointer(self) -> None:
        """An absent attribute is how the claim's `ALL_OLD` says "no champion
        yet". A `None` written as a string would be a first challenger compared
        against a model called "None"."""
        item = ctl.to_item(ctl.RunControl(run_id=RUN, next_cycle=Cycle(0), cycle_cap=8))
        assert "champion_version" not in item

    def test_the_champion_round_trips(self) -> None:
        control = ctl.RunControl(
            run_id=RUN,
            next_cycle=Cycle(4),
            cycle_cap=8,
            champion_version=new_model_version(RUN, Cycle(3)),
        )
        assert ctl.from_item(ctl.to_item(control)) == control

    def test_refuses_a_champion_from_another_run(self) -> None:
        """The one pointer error that cannot be seen by looking at it: the string
        is well formed and the model exists, and the comparison it produces is
        against a model trained under a partition this run never declared."""
        with pytest.raises(ValueError, match="cannot be"):
            ctl.RunControl(
                run_id=RUN,
                next_cycle=Cycle(4),
                cycle_cap=8,
                champion_version=new_model_version(OTHER, Cycle(3)),
            )

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


class TestStartingTheRun:
    """One execution is the whole run, so this is the only call that starts one.

    Run against `moto` rather than a fake client for `TestTheClaim`'s reason: the
    refusal of a second start is Step Functions' own uniqueness rule on execution
    names, and a fake written to raise proves only that it was written to raise.
    """

    @pytest.fixture
    def machine(self, fleet: Any) -> Iterator[Any]:
        """The cycle machine at the ARN `ctl.cycle_machine_arn` composes.

        Built from the real definition, so a state the ASL renamed cannot leave
        this test starting a machine that no longer matches it. `fleet` is taken
        for its credentials and its region, not its table.
        """
        aws = boto3.Session(region_name=REGION)
        aws.client("stepfunctions").create_state_machine(
            name=f"{PROJECT}-cycle",
            definition=ASL.read_text(encoding="utf-8"),
            roleArn=f"arn:aws:iam::{ACCOUNT}:role/{PROJECT}-cycle",
        )
        yield aws

    def started(self, aws: Any, arn: str) -> dict[str, Any]:
        described = aws.client("stepfunctions").describe_execution(executionArn=arn)
        return dict(json.loads(described["input"]))

    def test_the_payload_names_the_run_the_epochs_and_the_seeds(self, machine: Any) -> None:
        arn = ctl.start(machine, RUN, epochs=1, seeds=[1, 2, 3])

        assert self.started(machine, arn) == {
            "run_id": str(RUN),
            "epochs": 1,
            "seeds": [1, 2, 3],
        }

    def test_an_uncapped_run_carries_no_max_images(self, machine: Any) -> None:
        """0 is the whole labeled set, which is the machine's own default. A cap
        nobody chose has no business in the record of what was asked for."""
        arn = ctl.start(machine, RUN, epochs=1, seeds=[1])

        assert "max_images" not in self.started(machine, arn)

    def test_a_skeleton_cap_is_carried(self, machine: Any) -> None:
        arn = ctl.start(machine, RUN, epochs=1, seeds=[1], max_images=300)

        assert self.started(machine, arn)["max_images"] == 300

    def test_the_execution_is_named_for_the_run(self, machine: Any) -> None:
        """The name is the whole mechanism refusing a second start of a run
        already going, so it is asserted rather than the refusal itself: `moto`
        does not enforce Step Functions' uniqueness rule on execution names, and
        a test of a rule the engine under it does not have would pass by
        agreeing with nothing.
        """
        arn = ctl.start(machine, RUN, epochs=1, seeds=[1])

        assert arn.endswith(f":{RUN}")

    def test_a_second_run_starts_alongside_the_first(self, machine: Any) -> None:
        ctl.start(machine, RUN, epochs=1, seeds=[1])

        assert ctl.start(machine, OTHER, epochs=1, seeds=[1]).endswith(f":{OTHER}")


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
        """The message is what an operator reads off a failed execution, so it
        names every step rather than a sample -- and asserting on all of them
        means a step added without being offered here is a failing test."""
        with pytest.raises(ctrl.ControlError) as refusal:
            ctrl.handler({"step": "nope"})

        assert all(step in str(refusal.value) for step in ctrl.STEPS)

    def test_a_first_cycle_with_nothing_to_deploy_is_refused_by_name(self) -> None:
        """The pool is scored on the device, so a run whose first cycle failed
        its gates has no model anywhere to rank with. There is no cloud fallback
        on purpose -- one would produce a ranking from a model the fleet never
        ran -- so this is a refusal with the reason rather than a stack trace."""
        with pytest.raises(ctrl.ControlError, match="must promote"):
            ctrl.handler({"step": "deploy", "run_id": RUN, "cycle": 1, "seed": 1, "version": None})

    def test_the_pool_pass_without_a_token_is_refused(self) -> None:
        """A state machine wired without `.waitForTaskToken` would deploy, the
        device would score the sample, and nothing would ever resume the cycle.
        Better to refuse before the deployment goes out."""
        event = {
            "step": "fleet_score",
            "run_id": RUN,
            "cycle": 1,
            "version": new_model_version(RUN, Cycle(1)),
        }

        with pytest.raises(ctrl.ControlError, match="waitForTaskToken"):
            ctrl.handler(event)

    def test_the_step_names_are_the_ones_the_asl_passes(self) -> None:
        """The two files are an interface, and this is the half that can drift
        without either one failing to parse.

        Read out of the payload expression rather than off a key, because the
        payload is one JSONata object and not a field per key -- which is what
        `TestTheDefinition` checks and why.
        """
        definition = json.loads(ASL.read_text(encoding="utf-8"))
        assert _steps_of(definition["States"], CONTROL_FUNCTION) == set(ctrl.STEPS)

    def test_the_purchase_is_asked_of_the_other_function(self) -> None:
        """The label wall as a fact about the definition.

        The oracle reads `raw/labels/` and this function is denied it, so the two
        are two Lambdas -- and a purchase step appearing in `ctrl.STEPS` would
        mean the read had been moved inside the identity that must not have it.
        """
        definition = json.loads(ASL.read_text(encoding="utf-8"))

        assert _steps_of(definition["States"], ORACLE_FUNCTION) == set(oracle.STEPS)
        assert "purchase" not in ctrl.STEPS

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
    and the root files in a layer at `/opt`."""

    def _deployment(self, root: Path) -> tuple[Path, Path]:
        package = root / "task" / "edge_ml_flywheel"
        (package / "training").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "training" / "job.py").write_text("# job", encoding="utf-8")

        layer = root / "opt"
        layer.mkdir()
        (layer / "train.py").write_text("# entry point", encoding="utf-8")
        (layer / "score.py").write_text("# entry point", encoding="utf-8")
        (layer / "evaluate.py").write_text("# entry point", encoding="utf-8")
        (layer / "requirements.txt").write_text("ultralytics==8.4.146\n", encoding="utf-8")

        return package, layer

    def test_puts_every_entry_point_at_the_root_beside_the_package(self, tmp_path: Path) -> None:
        """Script mode requires `train.py` at the root of the archive and
        `scoring.job.container_entrypoint` names the other two there, which is
        the whole reason the two directories are separate arguments.

        One archive serving all three jobs is also the provenance claim: the code
        that gated a model is the tree that trained and scored it, under one
        `git_commit`.
        """
        package, layer = self._deployment(tmp_path)
        names = _tar_names(launch.archive(package, layer))

        assert "train.py" in names
        assert "score.py" in names
        assert "evaluate.py" in names
        assert "requirements.txt" in names
        assert "edge_ml_flywheel/training/job.py" in names

    def test_is_byte_identical_across_builds(self, tmp_path: Path) -> None:
        """The archive is filed under the write-once cycle prefix as evidence of
        what ran, so it has to be a function of the tree and not of when or where
        it was built. Timestamps, ownership and mode are all flattened."""
        package, layer = self._deployment(tmp_path)
        assert launch.archive(package, layer) == launch.archive(package, layer)


class TestTheImportGraph:
    """What the deployment package can carry, asserted at the import that would
    otherwise discover it in a live execution.

    The Lambda is `archive_file` over `src/`: pure Python, plus `boto3` from the
    runtime and `numpy` and `pyarrow` from the managed layer. Everything in
    `container/requirements.txt` is a compiled wheel that cannot be zipped into
    it, and every one of them is reachable from this handler through a module it
    legitimately imports -- `scoring.job` and `evaluation.job` build the two
    processing requests, and the entrypoints beside them do the work those jobs
    describe.

    So the failure has a shape: a module-level import taken for one name -- a
    constant, a dataclass, a type -- pulls a C extension into a function that
    only ever builds JSON. It costs nothing at test time, nothing at apply time,
    and fails every step of every cycle at `Runtime.ImportModuleError`.

    A subprocess rather than a `meta_path` blocker here, because the suite has
    already imported all of these by the time this runs and `sys.modules` would
    hand them straight back.
    """

    # `container/requirements.txt` and what `ultralytics` brings with it. Named
    # rather than derived from that file: the claim is about these packages being
    # absent from a zip, and a requirements file the container gains a pure
    # Python line in should not quietly widen what this refuses.
    CONTAINER_ONLY = (
        "pycocotools",
        "torch",
        "torchvision",
        "ultralytics",
        "cv2",
        "PIL",
        "onnx",
        "onnxruntime",
    )

    def test_reaches_no_module_the_lambda_cannot_carry(self) -> None:
        script = (
            "import sys\n"
            f"blocked = {self.CONTAINER_ONLY!r}\n"
            "class Refuse:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in blocked:\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Refuse())\n"
            "import edge_ml_flywheel.control.handler\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr


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

    def test_the_stubs_are_the_ones_still_to_land(self, definition: dict[str, Any]) -> None:
        """A `Pass` becoming a `Task` is a step landing, and it should be a test
        that has to be edited rather than a change nobody notices.

        `STUBS` is empty now, so this asserts every state of a cycle does
        something. The last two to land were `Select` and `Purchase`, which is
        what closed the loop: until they did, a run ranked nothing and bought
        nothing and stopped after one cycle.
        """
        passes = {name for name, state in definition["States"].items() if state["Type"] == "Pass"}
        assert passes == STUBS

    def test_the_cycle_ranks_the_pool_and_buys_the_top_of_it(
        self, definition: dict[str, Any]
    ) -> None:
        """The two steps that make this a flywheel rather than a training
        pipeline that happens to run eight times."""
        select = definition["States"]["Select"]
        assert select["Type"] == "Task"
        assert "'step': 'select'" in select["Arguments"]["Payload"]
        assert select["Next"] == "Purchase"

        purchase = definition["States"]["Purchase"]
        assert purchase["Type"] == "Task"
        assert "'step': 'purchase'" in purchase["Arguments"]["Payload"]
        assert purchase["Next"] == "MoreCycles"

    def test_the_purchase_runs_as_the_oracle_and_nothing_else_does(
        self, definition: dict[str, Any]
    ) -> None:
        """The label wall, as which function each step is sent to.

        Every other Lambda step goes to the control function, which is denied
        `raw/labels/` outright. This one goes to the identity that may read a
        withheld label, and it is the only one that does.
        """
        sent = {
            name: state["Arguments"]["FunctionName"]
            for name, state in definition["States"].items()
            if state.get("Resource", "").endswith("lambda:invoke")
        }

        assert sent["Purchase"] == ORACLE_FUNCTION
        assert {name for name, fn in sent.items() if fn == ORACLE_FUNCTION} == {"Purchase"}

    def test_selection_ranks_on_one_seed_and_it_is_the_deployed_one(
        self, definition: dict[str, Any]
    ) -> None:
        """Not every seed, unlike `EvaluateRequest`. An uncertainty score is what
        one detector found in one frame, so a mean over seeds would rank the pool
        by a model that does not exist -- and the seed it takes is the lowest,
        which is the one `Register` calls deployed.
        """
        payload = definition["States"]["Select"]["Arguments"]["Payload"]

        assert "'seed': $seeds_scored[0].seed" in payload
        assert "'seeds'" not in payload

    def test_the_batch_is_not_an_argument_to_the_purchase(self, definition: dict[str, Any]) -> None:
        """The oracle reads its image IDs out of the ranking `Select` wrote.

        A batch passed through the state machine is a batch a retry can carry
        differently, which would hash to a second digest and charge a second time
        for the same thousand images. There is no undo on that.
        """
        payload = definition["States"]["Purchase"]["Arguments"]["Payload"]

        assert "'run_id'" in payload
        assert "'cycle': $cycle" in payload
        assert "image" not in payload

    def test_the_loop_branches_on_what_the_oracle_left(self, definition: dict[str, Any]) -> None:
        """`pool_remaining` was a literal 0 while `Purchase` was a stub, which
        stopped every run after one cycle. It is now assigned from what was
        actually bought, which is what lets the machine go round."""
        purchase = definition["States"]["Purchase"]
        assert purchase["Assign"]["pool_remaining"] == "{% $states.result.Payload.pool_remaining %}"

        condition = definition["States"]["MoreCycles"]["Choices"][0]["Condition"]
        assert "$pool_remaining > 0" in condition

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
        so there is no path on which a gate verdict costs the run its purchase.
        Both branches meet again at the fleet round trip, which is what produces
        the ranking either of them buys from."""
        states = definition["States"]

        assert states["Passed"]["Default"] == "Deploy"
        assert states["Promote"]["Next"] == "Deploy"
        assert states["Deploy"]["Next"] == "FleetScore"
        assert states["FleetScore"]["Next"] == "Canary"
        assert states["Canary"]["Next"] == "Ranked"
        assert states["Ranked"]["Choices"][0]["Next"] == "Select"

    def test_the_pool_pass_waits_for_the_device_and_is_not_retried(
        self, definition: dict[str, Any]
    ) -> None:
        """The token is the completion signal: a device that cannot report is
        caught by the timeout, and a retry would re-deploy onto the same hardware
        that just failed to answer."""
        state = definition["States"]["FleetScore"]

        assert state["Resource"].endswith("lambda:invoke.waitForTaskToken")
        assert state["TimeoutSeconds"] == 7200
        assert "Retry" not in state

    def test_untrustworthy_detections_end_the_run_rather_than_buying(
        self, definition: dict[str, Any]
    ) -> None:
        """A ranking computed from bytes that do not hash to what the gates were
        reported over is not a ranking. This is the one path on which a cycle
        buys nothing, and it is a device fault rather than a gate verdict."""
        states = definition["States"]

        assert states["Ranked"]["Default"] == "PassNotTrusted"
        assert states["PassNotTrusted"]["Type"] == "Fail"

    def test_no_job_is_retried_after_it_has_been_created(self, definition: dict[str, Any]) -> None:
        """A SageMaker job name is minted once, with the request, by the step
        before the one that submits it. So a retry of the submitting state sends
        the same name, and the service answers `ResourceInUse` -- which is not a
        second attempt at anything. It replaces the real failure with a name
        clash, and that is how an int8 scoring crash came back from a cycle
        reported as `SageMaker.ResourceInUseException`.

        What is left is the failures that happen before a job exists, where the
        name is still free and a second attempt is a real one.
        """
        states = definition["States"]
        submitting = {
            "TrainSeed": states["Train"]["ItemProcessor"]["States"]["TrainSeed"],
            "ScoreSeed": states["Score"]["ItemProcessor"]["States"]["ScoreSeed"],
            "ScoreQuantized": states["ScoreQuantized"],
            "Evaluate": states["Evaluate"],
        }

        for name, state in submitting.items():
            assert state["Resource"].endswith(".sync"), name
            retried = {error for retrier in state["Retry"] for error in retrier["ErrorEquals"]}

            assert "States.TaskFailed" not in retried, (
                f"{name} retries a job that was created and failed, which can only collide "
                f"with its own name and hide what actually went wrong"
            )
            assert retried == {
                "SageMaker.ResourceLimitExceeded",
                "SageMaker.ThrottlingException",
            }, name

    def test_the_training_job_is_started_by_the_state_machine(
        self, definition: dict[str, Any]
    ) -> None:
        """`.sync`, so the wait, the retry and the stop-on-abort are the ASL's and
        nothing in this project polls a training job."""
        seed = definition["States"]["Train"]["ItemProcessor"]["States"]["TrainSeed"]
        assert seed["Resource"].endswith("sagemaker:createTrainingJob.sync")

    def test_the_evaluation_runs_once_over_every_seed_that_scored(
        self, definition: dict[str, Any]
    ) -> None:
        """Not a `Map`, unlike the two steps before it. A paired delta is a mean
        over same-seed differences, so a comparison split across jobs is not a
        comparison -- and the seed list comes from what `Score` assigned, so a
        seed that failed to score is one nothing tries to evaluate."""
        request = definition["States"]["EvaluateRequest"]
        assert request["Type"] == "Task"

        payload = request["Arguments"]["Payload"]
        assert "'step': 'evaluate_request'" in payload
        assert "$seeds_scored" in payload

        assert definition["States"]["Evaluate"]["Resource"].endswith(
            "sagemaker:createProcessingJob.sync"
        )

    def test_the_evaluation_job_is_not_given_the_cycles_instance_type(
        self, definition: dict[str, Any]
    ) -> None:
        """The one an execution may set is the GPU type training and scoring
        share, and this job is numpy over cached arrays. Passing it here would be
        a way to run the addition on a GPU."""
        payload = definition["States"]["EvaluateRequest"]["Arguments"]["Payload"]
        assert "instance_type" not in payload

    def test_the_cap_reaches_the_training_job_and_not_only_the_manifest(
        self, definition: dict[str, Any]
    ) -> None:
        """The bug that stopped every skeleton run there has ever been.

        `Prepare` caps the image manifest; the labels arrive on whole prefixes
        that no cap can be expressed on, so the container has to apply the same
        number itself. Until it was passed here, a capped cycle delivered a few
        hundred images beside every label the run had bought, and the training
        job refused the pair -- correctly, and after paying for the instance.
        """
        prepare = definition["States"]["Prepare"]["Arguments"]["Payload"]
        training = definition["States"]["Train"]["ItemProcessor"]["States"]["TrainingRequest"][
            "Arguments"
        ]["Payload"]

        assert "Execution.Input.max_images" in prepare
        assert "Execution.Input.max_images" in training

    def test_an_optional_input_is_only_read_where_absence_is_allowed(
        self, definition: dict[str, Any]
    ) -> None:
        """The bug that stopped the first execution ever started, at `Prepare`.

        A field whose JSONata resolves to nothing is `States.QueryEvaluationError`
        and the end of the run -- Step Functions has no setting that makes it an
        omission. Inside an object constructor the same value simply drops its
        key, so the handler's `.get(name, default)` is what decides, and the
        defaults stay in the one place the CLI reads them from too.
        """
        for expression in _expressions(definition):
            for field in OPTIONAL_INPUTS:
                if f"Execution.Input.{field}" not in expression:
                    continue
                assert expression.startswith("{% {"), (
                    f"{field} is read outside an object constructor, so an execution that "
                    f"leaves it out fails here instead of taking the handler's default: "
                    f"{expression}"
                )

    def test_the_verdict_the_cycle_branches_on_comes_from_the_gates(
        self, definition: dict[str, Any]
    ) -> None:
        """`passed` was a literal `true` while `Register` was a stub, which made
        the branch below decorative. It is now assigned from what the step
        returned, and the step reads the gate report."""
        register = definition["States"]["Register"]
        assert register["Type"] == "Task"
        assert register["Assign"]["passed"] == "{% $states.result.Payload.passed %}"

    def test_the_version_is_registered_whatever_the_verdict(
        self, definition: dict[str, Any]
    ) -> None:
        """The rejection log is a feature (design section 5). Registration sits
        before the branch, so there is no path on which a refused challenger goes
        unrecorded."""
        assert definition["States"]["Register"]["Next"] == "OpenTheRegistryGroup"
        assert definition["States"]["OpenTheRegistryGroup"]["Next"] == "RegisterTheVersion"
        assert definition["States"]["RegisterTheVersion"]["Next"] == "Passed"

    def test_the_registry_calls_are_the_state_machines(self, definition: dict[str, Any]) -> None:
        """The control function builds the request and holds no SageMaker grant,
        so the call happens here, where the execution history records it."""
        for name, action in (
            ("OpenTheRegistryGroup", "sagemaker:createModelPackageGroup"),
            ("RegisterTheVersion", "sagemaker:createModelPackage"),
        ):
            assert definition["States"][name]["Resource"] == f"arn:aws:states:::aws-sdk:{action}"

    def test_a_group_that_already_exists_is_not_a_failed_cycle(
        self, definition: dict[str, Any]
    ) -> None:
        """One group per run, so every cycle after the first finds it there."""
        catch = definition["States"]["OpenTheRegistryGroup"]["Catch"]
        assert catch[0]["ErrorEquals"] == ["States.ALL"]
        assert catch[0]["Next"] == "RegisterTheVersion"

    def test_the_champion_arrives_with_the_cycle_it_was_claimed_for(
        self, definition: dict[str, Any]
    ) -> None:
        """`ALL_OLD` already returns the whole item, so the pointer `Promote`
        wrote at the end of the previous cycle costs no second read. The ternary
        is what makes a run's first cycle a baseline rather than a failed
        expression: a JSONata path resolving to nothing fails the state."""
        champion = definition["States"]["ClaimCycle"]["Assign"]["champion"]
        assert "$exists(" in champion
        assert champion.endswith(": null %}")

    def test_the_evaluation_is_handed_the_champion_to_pair_against(
        self, definition: dict[str, Any]
    ) -> None:
        """Without this the gate has nothing to compare against and every cycle
        of a run is its own baseline."""
        payload = definition["States"]["EvaluateRequest"]["Arguments"]["Payload"]
        assert "'champion': $champion" in payload

    def test_promotion_advances_the_pointer_the_claim_reads(
        self, definition: dict[str, Any]
    ) -> None:
        """One item, written here and read at `ClaimCycle`."""
        promote = definition["States"]["Promote"]
        assert promote["Type"] == "Task"
        assert promote["Resource"].endswith("dynamodb:updateItem")
        assert promote["Arguments"]["UpdateExpression"] == "SET champion_version = :version"
        assert promote["Arguments"]["Key"]["entity"]["S"] == "run"

    def test_promotion_only_ever_moves_forward(self, definition: dict[str, Any]) -> None:
        """A version sorts lexicographically by its padded cycle within one run,
        so a cycle re-run against an older model is refused rather than silently
        regressing the champion every later comparison is paired against."""
        condition = definition["States"]["Promote"]["Arguments"]["ConditionExpression"]
        assert "attribute_not_exists(champion_version)" in condition
        assert "champion_version < :version" in condition

    def test_only_a_passing_challenger_is_promoted(self, definition: dict[str, Any]) -> None:
        """The registry recorded the verdict either way; promotion is what a
        rejected challenger is denied."""
        assert definition["States"]["Passed"]["Choices"][0]["Next"] == "Promote"
        assert definition["States"]["Passed"]["Default"] == "Deploy"

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


def _payloads(states: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Every Lambda invocation in the definition, as (function, payload).

    The payload is a string rather than a dict, because each one is a single
    JSONata object constructor -- which is what makes a key optional; see
    `_expressions`.

    The function name comes with it because a cycle calls two of them, and which
    one a step goes to is the label wall: the control function does every step
    that does not read a withheld label and the oracle does the one that does.
    """
    for scope in _scopes(states):
        for state in scope.values():
            arguments = state.get("Arguments")
            if isinstance(arguments, dict) and isinstance(arguments.get("Payload"), str):
                yield str(arguments.get("FunctionName", "")), arguments["Payload"]


def _steps_of(states: dict[str, Any], function: str) -> set[str]:
    """The step names one function is asked for."""
    return {
        match.group(1)
        for name, payload in _payloads(states)
        if name == function and (match := STEP_IN_PAYLOAD.search(payload))
    }


def _expressions(node: Any) -> Iterator[str]:
    """Every JSONata expression in the definition, at any depth.

    A walk of the parsed JSON rather than a scan of the text, so what is yielded
    is one field's expression and the test below can ask where it sits.
    """
    if isinstance(node, str):
        if node.startswith("{%"):
            yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _expressions(value)
    elif isinstance(node, list):
        for value in node:
            yield from _expressions(value)


def _tar_names(payload: bytes) -> set[str]:
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as tar:
        return set(tar.getnames())

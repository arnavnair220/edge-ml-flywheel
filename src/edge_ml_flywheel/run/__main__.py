"""The run steps, one subcommand each.

**`start` is the whole of starting a run.** It registers the run and starts one
execution, and that execution is the run rather than one cycle: the state machine
loops from `MoreCycles` back to `ClaimCycle` and leaves through `Done` when the
cycle cap or the pool is spent. There is nothing to tick and nothing to call
again.

`register` survives beside it for the case that ordering has to be broken -- a
run registered before the machine it will be started on exists, which is what
`buildspecs/register.yml` does. `show` is how anyone asks what a `run_id` in a
bucket listing was configured as, and it is also the post-build check that the
registration landed.

**`register` writes two items, not one.** The registration in `runs` is what the
run is; the control item in `fleet_config` is the cycle counter the state
machine claims from, and a run with no counter is a run nothing can start. They
are separate for the reason `run.control` gives -- one is write-once and the
other advances every cycle -- and they are written together here because a run
that has one and not the other is not a run anyone asked for.

Logging goes to stderr so that `register` can put the bare run ID on stdout for
the shell to capture, the way `partition prefix` does with a prefix. That string
is the input to every later step.

**Nothing here reads a clock except `main`, once.** `new_run_id` takes the
timestamp as an argument for exactly this reason, and the same value becomes
`created_at`, so the id and the item cannot disagree about when the run started.

**Nothing here invents a git commit either.** `--git-commit` defaults to
CodeBuild's resolved source version and fails if it is absent rather than
shelling out to `git rev-parse`: a SHA read from a working tree with uncommitted
changes names a commit that did not produce this run, and the field is written
once.
"""

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime

import boto3

from edge_ml_flywheel.conventions import (
    PARTITIONS,
    Cycle,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    new_run_id,
    parse_run_id,
)
from edge_ml_flywheel.run import control as ctl
from edge_ml_flywheel.run import registration as reg

log = logging.getLogger("edge_ml_flywheel.run")


def _definition() -> argparse.ArgumentParser:
    """What a run *is*, shared by `register` and `start`.

    A parent parser rather than two copies, because these nine fields are the
    definition of a run and `start` is `register` plus an execution. Two lists
    would be two places a new field has to be added, and the one that gets
    forgotten is the one that silently defaults.
    """
    minted = argparse.ArgumentParser(add_help=False)
    minted.add_argument(
        "--slug",
        required=True,
        help="Readable half of the run ID. Lowercase words joined by hyphens.",
    )
    minted.add_argument(
        "--partition-version",
        type=int,
        required=True,
        choices=sorted(PARTITIONS),
        help="A version defined in conventions.PARTITIONS, which fixes its seed and sizes.",
    )
    # No registry to validate this against yet -- unlike the partition, it is not
    # a draw this repo can reproduce, so it is an integer the caller states and
    # the model manifest is later checked against.
    minted.add_argument("--recipe-version", type=int, required=True)
    minted.add_argument(
        "--label-budget",
        type=int,
        required=True,
        help="New labels purchasable per cycle. Fixed for the run; changing it is a new run.",
    )
    # The ceiling the cycle counter is opened against. Required rather than
    # defaulted for `--selector`'s reason: how many cycles a run gets is part of
    # what the run is, and a default is how a run comes to stop somewhere nobody
    # chose. It is on the control item rather than the registration because the
    # conditional update that claims a cycle can only name attributes of the item
    # it writes -- see `run.control`.
    minted.add_argument(
        "--cycle-cap",
        type=int,
        required=True,
        help="How many cycles this run may claim. The state machine stops at it.",
    )

    # Required, because a note nobody is made to write is a note nobody writes,
    # and this is the only field that says why the run exists.
    minted.add_argument("--note", required=True, help="Why this run was started.")
    minted.add_argument(
        "--git-commit",
        default=os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", ""),
        help="40-character SHA of the commit being run. Defaults to CodeBuild's.",
    )

    return minted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.run")
    sub = parser.add_subparsers(dest="command", required=True)
    definition = _definition()

    # `start` first, because it is the one anyone runs. `register` stays its own
    # command for the case where a run has to exist before it is started.
    started = sub.add_parser(
        "start",
        parents=[definition],
        help="register a run and start it. One execution is the whole run.",
    )
    started.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Epochs per training job. Required for `control.handler`'s reason: it is the recipe.",
    )
    # A list rather than a count, so a run that wants the seed spread the design
    # no longer trains asks for it without editing the state machine, and so the
    # seeds a run trained are recorded in its execution input. One seed is the
    # design (design section 4.2): the cycle waits on one GPU job either way, and
    # every extra seed is another on-demand hour against a quota that has to
    # cover it.
    started.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="The seeds each cycle trains, one job per seed. Seed 1 is what ships.",
    )
    started.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap each cycle's training set, for a short skeleton run. 0 is the whole set.",
    )

    sub.add_parser(
        "register", parents=[definition], help="mint a run ID and claim it, without starting it"
    )

    shown = sub.add_parser("show", help="print a run's registration")
    shown.add_argument("--run-id", required=True)

    return parser


def _require_commit(git_commit: str) -> str:
    if not git_commit:
        raise SystemExit(
            "no git commit: pass --git-commit, or run where "
            "CODEBUILD_RESOLVED_SOURCE_VERSION is set"
        )
    return git_commit


def _register(args: argparse.Namespace, started_at: datetime) -> RunId:
    """Mint, claim, open the cycle counter, and read both back.

    The read-back is not ceremony. `register` is the one write in the project
    with no possible correction -- the item is write-once and everything else is
    keyed by what it names -- so the run does not proceed on the strength of a
    call that returned without raising.

    Two writes and not one, in this order, because they fail differently. The
    registration is the claim on the name and the thing every later key is built
    from; the control item is a counter that is only meaningful once that name is
    claimed. A run left with a registration and no counter is a run the state
    machine refuses to start, which is a loud and fixable state -- the reverse
    would be a counter belonging to a run that does not exist.
    """
    run_id = new_run_id(started_at, args.slug)
    entry = RunRegistration(
        run_id=run_id,
        created_at=started_at,
        git_commit=_require_commit(args.git_commit),
        partition_version=PartitionVersion(args.partition_version),
        recipe_version=RecipeVersion(args.recipe_version),
        label_budget_per_cycle=args.label_budget,
        note=args.note,
    )

    table = reg.runs_table()
    try:
        reg.register(table, entry)
    except reg.RunAlreadyRegisteredError as collision:
        raise SystemExit(str(collision)) from None

    stored = reg.read(table, run_id)
    if stored != entry:
        raise SystemExit(
            f"{run_id} read back as something other than what was written, which means the "
            f"encoding is lossy: {stored!r}"
        )

    counter = ctl.RunControl(run_id=run_id, next_cycle=Cycle(0), cycle_cap=args.cycle_cap)
    fleet = ctl.fleet_config_table()
    try:
        ctl.register(fleet, counter)
    except reg.RunAlreadyRegisteredError as collision:
        raise SystemExit(str(collision)) from None

    opened = ctl.read(fleet, run_id)
    if opened != counter:
        raise SystemExit(
            f"{run_id} claimed its name but its cycle counter read back as {opened!r}. The run "
            f"cannot be started until that item is what it should be."
        )

    log.info("budget %d labels a cycle", args.label_budget)
    log.info(
        "partition v%d, recipe v%d",
        entry.partition_version,
        entry.recipe_version,
    )
    log.info("cycles 0 to %d", counter.cycle_cap - 1)
    return run_id


def _show(run_id: str) -> None:
    """Both items, because a run is both of them.

    The registration says what the run was configured as and the control item
    says where it has got to, and the question anyone actually asks -- what is
    this `run_id` in a bucket listing -- is answered by the pair. A missing
    control item is printed as `null` rather than raised: it means the counter
    was never opened, which is worth seeing rather than being told the run does
    not exist.
    """
    entry = reg.read(reg.runs_table(), parse_run_id(run_id))
    if entry is None:
        raise SystemExit(f"{run_id} was never registered, so nothing was ever written under it")

    counter = ctl.read(ctl.fleet_config_table(), parse_run_id(run_id))
    document = {
        "registration": reg.to_item(entry),
        "control": ctl.to_item(counter) if counter else None,
    }
    print(json.dumps(document, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "start":
        run_id = _register(args, datetime.now(UTC))
        # The run ID first and on its own line, so that a failure to start still
        # leaves the operator holding the name of the run that now exists. It is
        # registered either way, and a run nobody can name is a run nobody can
        # start on a second attempt.
        print(run_id)
        print(
            ctl.start(
                boto3.Session(),
                run_id,
                epochs=args.epochs,
                seeds=args.seeds,
                max_images=args.max_images,
            )
        )

    elif args.command == "register":
        print(_register(args, datetime.now(UTC)))

    elif args.command == "show":
        _show(args.run_id)


if __name__ == "__main__":
    main()

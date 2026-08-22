"""The run steps, one subcommand each.

`buildspecs/register.yml` calls these. Two subcommands rather than one because
the read-back is worth being able to do on its own: `show` is how anyone asks
what a `run_id` in a bucket listing was configured as, and it is also the
post-build check that the registration landed.

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

from edge_ml_flywheel.conventions import (
    PARTITIONS,
    ClassSetVersion,
    PartitionVersion,
    RecipeVersion,
    RunRegistration,
    Selector,
    new_run_id,
    parse_run_id,
)
from edge_ml_flywheel.run import registration as reg

log = logging.getLogger("edge_ml_flywheel.run")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.run")
    sub = parser.add_subparsers(dest="command", required=True)

    minted = sub.add_parser("register", help="mint a run ID and claim it in the runs table")
    minted.add_argument(
        "--slug",
        required=True,
        help="Readable half of the run ID. Lowercase words joined by hyphens.",
    )
    # Required rather than defaulted to `uncertainty`. A control arm differs from
    # the real loop in this one field, and a default is how an arm gets run as
    # the thing it was meant to be the control for.
    minted.add_argument(
        "--selector",
        required=True,
        choices=[member.value for member in Selector],
        help="The rule this run ranks the pool by, fixed for its whole length.",
    )
    minted.add_argument(
        "--partition-version",
        type=int,
        required=True,
        choices=sorted(PARTITIONS),
        help="A version defined in conventions.PARTITIONS, which fixes its seed and sizes.",
    )
    # No registry to validate these against yet -- unlike the partition, neither
    # is a draw this repo can reproduce, so both are integers the caller states
    # and the model manifest is later checked against.
    minted.add_argument("--class-set-version", type=int, required=True)
    minted.add_argument("--recipe-version", type=int, required=True)
    minted.add_argument(
        "--label-budget",
        type=int,
        required=True,
        help="New labels purchasable per cycle. Fixed for the run; changing it is a new run.",
    )
    # Required, because a note nobody is made to write is a note nobody writes,
    # and this is the only field that says why the run exists.
    minted.add_argument("--note", required=True, help="Why this run was started.")
    minted.add_argument(
        "--git-commit",
        default=os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", ""),
        help="40-character SHA of the commit being run. Defaults to CodeBuild's.",
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


def _register(args: argparse.Namespace, started_at: datetime) -> None:
    """Mint, claim, and read back.

    The read-back is not ceremony. `register` is the one write in the project
    with no possible correction -- the item is write-once and everything else is
    keyed by what it names -- so the run does not proceed on the strength of a
    call that returned without raising.
    """
    run_id = new_run_id(started_at, args.slug)
    entry = RunRegistration(
        run_id=run_id,
        created_at=started_at,
        git_commit=_require_commit(args.git_commit),
        partition_version=PartitionVersion(args.partition_version),
        class_set_version=ClassSetVersion(args.class_set_version),
        recipe_version=RecipeVersion(args.recipe_version),
        selector=Selector(args.selector),
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

    log.info("selector %s, budget %d labels a cycle", entry.selector.value, args.label_budget)
    log.info(
        "partition v%d, class set v%d, recipe v%d",
        entry.partition_version,
        entry.class_set_version,
        entry.recipe_version,
    )
    print(run_id)


def _show(run_id: str) -> None:
    entry = reg.read(reg.runs_table(), parse_run_id(run_id))
    if entry is None:
        raise SystemExit(f"{run_id} was never registered, so nothing was ever written under it")
    print(json.dumps(reg.to_item(entry), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "register":
        _register(args, datetime.now(UTC))

    elif args.command == "show":
        _show(args.run_id)


if __name__ == "__main__":
    main()

"""The partition step as a subcommand.

One subcommand today, and a subparser anyway, for `ingest.__main__`'s reason: the
shard writer and the eval freeze are steps of the same job, and a job whose only
entry point is bare `python -m` has nowhere to put the second one.

`--partition-version` is required rather than defaulted. The versions are listed
as choices, so an undefined one fails at the parser, and there is deliberately no
`--seed`: the seed is a property of the version (`conventions.PARTITIONS`).

Logging goes to stderr, matching ingest.
"""

import argparse
import logging
import sys
from pathlib import Path

from edge_ml_flywheel.conventions import PARTITIONS, PartitionVersion
from edge_ml_flywheel.partition import assign

log = logging.getLogger("edge_ml_flywheel.partition")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.partition")
    sub = parser.add_subparsers(dest="command", required=True)

    drawn = sub.add_parser("assign", help="draw cohorts and write the assignments parquet")
    drawn.add_argument("--stage-dir", type=Path, required=True)
    drawn.add_argument(
        "--partition-version",
        type=int,
        choices=sorted(PARTITIONS),
        required=True,
        help="A version defined in conventions.PARTITIONS, which fixes its seed and sizes.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "assign":
        rows = assign.write(args.stage_dir, PartitionVersion(args.partition_version))
        log.info("assigned %d images to a cohort", len(rows))


if __name__ == "__main__":
    main()

"""The partition steps, one subcommand each.

`buildspecs/partition.yml` calls these in order, and the split between them is
`ingest.__main__`'s: the shell fetches and copies, and every decision about what
the data is goes through the package. No S3 key is spelled in the buildspec --
`prefix` prints the ones the copies need, the way `ingest url` prints a URL.

`--partition-version` is required rather than defaulted, and the defined versions
are the parser's choices, so an undefined one fails before anything is read.
There is deliberately no `--seed`: the seed is a property of the version
(`conventions.PARTITIONS`).

Logging goes to stderr so that `prefix` can put a bare prefix on stdout for the
shell to capture.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from edge_ml_flywheel.conventions import (
    MANIFEST_PREFIX,
    PARTITIONS,
    PartitionVersion,
    partition_prefix,
    partition_spec,
)
from edge_ml_flywheel.partition import assign, cohort_labels

log = logging.getLogger("edge_ml_flywheel.partition")

# What `prefix` can be asked for: the input to read and the output to write.
PREFIXES = ("manifest", "partition")


def _add_version(parser: argparse.ArgumentParser) -> None:
    """On each subparser that reads it, for `ingest._add_host`'s reason.

    Declared on the top-level parser it would only be accepted before the
    subcommand name, which is not where anyone types it.
    """
    parser.add_argument(
        "--partition-version",
        type=int,
        choices=sorted(PARTITIONS),
        required=True,
        help="A version defined in conventions.PARTITIONS, which fixes its seed and sizes.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.partition")
    sub = parser.add_subparsers(dest="command", required=True)

    printed = sub.add_parser("prefix", help="print one S3 prefix for the shell to copy")
    printed.add_argument("which", choices=PREFIXES)
    printed.add_argument(
        "--partition-version",
        type=int,
        choices=sorted(PARTITIONS),
        help="Required for the partition prefix, which is keyed by it.",
    )

    recorded = sub.add_parser(
        "verify-recorded", help="check a partition already in the bucket against this code"
    )
    recorded.add_argument("--recorded", type=Path, required=True)
    _add_version(recorded)

    drawn = sub.add_parser("assign", help="draw cohorts and write the assignments parquet")
    drawn.add_argument("--stage-dir", type=Path, required=True)
    _add_version(drawn)

    listed = sub.add_parser(
        "label-keys", help="print the raw label keys the labeled cohorts need, one per line"
    )
    listed.add_argument("--stage-dir", type=Path, required=True)
    _add_version(listed)

    labelled = sub.add_parser(
        "labels", help="write the bootstrap and eval boxes from staged label documents"
    )
    labelled.add_argument("--stage-dir", type=Path, required=True)
    _add_version(labelled)

    return parser


def _prefix(which: str, partition_version: int | None) -> str:
    if which == "manifest":
        return MANIFEST_PREFIX
    if partition_version is None:
        raise SystemExit("the partition prefix is keyed by a version: pass --partition-version")
    return partition_prefix(PartitionVersion(partition_version))


def _verify_recorded(recorded: Path, partition_version: PartitionVersion) -> None:
    """Refuse to redraw a version whose recorded draw is not the one in this code.

    Absent is fine and is the first run of a version. Present and disagreeing is
    the case this exists for: the assignments in the bucket are what every run
    keyed to this version was measured against, and a re-run under an edited seed
    would replace them with a partition that is equally valid and differently
    drawn.
    """
    if not recorded.is_file():
        log.info("no partition recorded for version %d yet", partition_version)
        return

    document = json.loads(recorded.read_text(encoding="utf-8"))
    wrong = assign.disagreements(document, partition_version, partition_spec(partition_version))
    if wrong:
        raise SystemExit(
            f"partition version {partition_version} is already recorded in the bucket with a "
            f"different {', '.join(wrong)}. Its assignments are what existing runs were measured "
            f"against, so this is a new version rather than a re-run of this one."
        )

    log.info(
        "version %d matches what is already recorded; a re-run reproduces it", partition_version
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "prefix":
        print(_prefix(args.which, args.partition_version))

    elif args.command == "verify-recorded":
        _verify_recorded(args.recorded, PartitionVersion(args.partition_version))

    elif args.command == "assign":
        rows = assign.write(args.stage_dir, PartitionVersion(args.partition_version))
        log.info("assigned %d images to a cohort", len(rows))

    elif args.command == "label-keys":
        # To stdout, one per line, for the shell to copy -- the same division as
        # `prefix`. The list is derived from the assignments rather than from the
        # sizes, so it names the images this version actually drew.
        rows = assign.read_assignments(args.stage_dir, PartitionVersion(args.partition_version))
        needed = cohort_labels.keys(rows)
        log.info("%d label documents to stage", len(needed))
        print("\n".join(needed))

    elif args.command == "labels":
        rows = assign.read_assignments(args.stage_dir, PartitionVersion(args.partition_version))
        written = cohort_labels.write(
            args.stage_dir, PartitionVersion(args.partition_version), rows
        )
        log.info(
            "labeled %d images across %s",
            sum(written.values()),
            ", ".join(cohort.value for cohort in written),
        )


if __name__ == "__main__":
    main()

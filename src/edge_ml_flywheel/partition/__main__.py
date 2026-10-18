"""The partition steps, one subcommand each.

`buildspecs/partition.yml` calls these in order. Every copy between the bucket
and local disk is one of these commands, through `storage.ObjectStore`, so the
job runs against any store fsspec reaches and no key is spelled in the buildspec:

- `fetch-inputs` stages the manifest and the version's recorded draw, if any.
- `stage-labels` stages the label documents the draw names, for the reason
  `partition.fetch` gives.
- `upload` writes the partition prefix back and checks what landed.

`label-keys` still prints the label list, and is what the build log records as
the audit of which 13,000 of the 80,000 labels the job read.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from edge_ml_flywheel.conventions import (
    MANIFEST_PREFIX,
    PARTITIONS,
    PartitionVersion,
    partition_manifest_key,
    partition_prefix,
    partition_spec,
)
from edge_ml_flywheel.partition import assign, cohort_labels, fetch
from edge_ml_flywheel.storage import NoSuchKeyError, ObjectStore, tree

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

    inputs = sub.add_parser(
        "fetch-inputs", help="stage the manifest, and the recorded draw if there is one"
    )
    inputs.add_argument("--stage-dir", type=Path, required=True)
    inputs.add_argument("--recorded", type=Path, required=True)
    _add_version(inputs)

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

    fetched = sub.add_parser(
        "stage-labels", help="copy those label documents out of the bucket onto local disk"
    )
    fetched.add_argument("--stage-dir", type=Path, required=True)
    _add_version(fetched)

    labelled = sub.add_parser(
        "labels", help="write the bootstrap and eval boxes from staged label documents"
    )
    labelled.add_argument("--stage-dir", type=Path, required=True)
    _add_version(labelled)

    sent = sub.add_parser(
        "upload", help="write the staged partition prefix to the bucket and check it landed"
    )
    sent.add_argument("--stage-dir", type=Path, required=True)
    _add_version(sent)

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

    _check_recorded(json.loads(recorded.read_text(encoding="utf-8")), partition_version)


def _check_recorded(document: Mapping[str, Any], partition_version: PartitionVersion) -> None:
    """The comparison itself, for a document read from a file or from the bucket."""
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


def _fetch_inputs(
    store: ObjectStore, stage_dir: Path, recorded: Path, partition_version: PartitionVersion
) -> None:
    """The manifest into the staged tree, and the recorded draw beside it.

    The manifest is staged at the key it has in the bucket, so the partitioner
    finds it exactly where `conventions` says it is. An empty manifest prefix is
    an error: there is nothing to draw from.

    The record goes to `recorded`, outside the staged tree, so it is never part
    of what `upload` writes. Absent is the ordinary first run of a version, and
    leaves no file for `verify-recorded` to find -- including one left by an
    earlier call, which would otherwise be checked in place of the bucket's.
    """
    landed = tree.fetch_prefix(
        store, MANIFEST_PREFIX, stage_dir / MANIFEST_PREFIX, what="manifest files"
    )
    if not landed:
        raise SystemExit(f"no manifest under {store.bucket}/{MANIFEST_PREFIX}; run ingest first")

    try:
        store.download(partition_manifest_key(partition_version), recorded)
    except NoSuchKeyError:
        recorded.unlink(missing_ok=True)
        log.info(
            "no partition recorded for version %d yet; this is its first run", partition_version
        )
        return

    log.info("staged the recorded partition for version %d", partition_version)


def _upload(store: ObjectStore, stage_dir: Path, partition_version: PartitionVersion) -> None:
    """The partition prefix, and nothing else, then a read-back of its record.

    Only the partition prefix: the manifest and the label documents were staged
    to be read, and writing either back is a permission this role deliberately
    does not hold. `tree.upload` builds every key under the prefix it is given.

    The read-back makes `verify-recorded`'s comparison against what landed, so
    the document in the bucket is confirmed to describe the partition this
    commit draws.
    """
    prefix = partition_prefix(partition_version)
    tree.upload(store, stage_dir / prefix, prefix, what="partition files")
    _check_recorded(
        json.loads(store.get(partition_manifest_key(partition_version))), partition_version
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "prefix":
        print(_prefix(args.which, args.partition_version))

    elif args.command == "fetch-inputs":
        _fetch_inputs(
            fetch.data_store(),
            args.stage_dir,
            args.recorded,
            PartitionVersion(args.partition_version),
        )

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

    elif args.command == "stage-labels":
        # The same derivation `label-keys` prints, rather than the list it
        # printed: a downloader that takes keys from its caller is a downloader
        # someone can hand a `pool` label, and the refusal in
        # `cohort_labels.label_key` only helps while the keys come from there.
        rows = assign.read_assignments(args.stage_dir, PartitionVersion(args.partition_version))
        needed = cohort_labels.keys(rows)
        fetch.stage(fetch.data_store(), needed, args.stage_dir)

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

    elif args.command == "upload":
        _upload(fetch.data_store(), args.stage_dir, PartitionVersion(args.partition_version))


if __name__ == "__main__":
    main()

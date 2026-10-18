"""The ingest steps, one subcommand each.

`buildspecs/ingest.yml` calls these in order. Separate subcommands rather than a
single `run` because each one is a checkpoint in a build measured in tens of
minutes: the phase boundaries in the CodeBuild log are where you look first when
something failed, and a step that has to be re-run by hand is a step you can
re-run by hand.

Logging goes to stderr so that `url` can put a bare URL on stdout for the shell
to capture.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from edge_ml_flywheel.ingest import manifest, provenance
from edge_ml_flywheel.ingest.source import ARCHIVES
from edge_ml_flywheel.ingest.stage import stage
from edge_ml_flywheel.storage import DATA_BUCKET, ObjectStore, bucket_name, tree
from edge_ml_flywheel.storage.filesystem import FsspecStore

log = logging.getLogger("edge_ml_flywheel.ingest")

# The prefixes ingest writes, listed separately: the ingest role's ListBucket is
# conditioned on an `s3:prefix` of one of these, so a root listing is denied by
# design -- it cannot enumerate the artifacts of runs it has nothing to do with.
_WRITTEN_PREFIXES = ("raw/", "derived/")


def _add_host(parser: argparse.ArgumentParser) -> None:
    """`--host` belongs to the subcommands that read it, not to the top level.

    On the top-level parser it would only be accepted *before* the subcommand
    name -- `ingest --host X url images` -- and the natural spelling the
    buildspec used, `ingest url images --host X`, fails as an unrecognized
    argument. Declared here it is accepted where anyone would type it.

    Never on both parsers at once. Argparse writes the subparser's value into
    the same namespace field last, so a top-level `--host` would be silently
    overwritten by this one's empty default -- the same failure, but quiet
    rather than loud.
    """
    parser.add_argument(
        "--host",
        default=os.environ.get("BDD100K_HOST", ""),
        help="Host serving the archives. Defaults to $BDD100K_HOST.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    url = sub.add_parser("url", help="print one archive's download URL")
    url.add_argument("archive", choices=sorted(ARCHIVES))
    _add_host(url)

    verify = sub.add_parser("verify-archives", help="check the downloads against the source")
    verify.add_argument("--work-dir", type=Path, required=True)
    _add_host(verify)

    staging = sub.add_parser("stage", help="move the extract to its S3 keys")
    staging.add_argument("--extract-dir", type=Path, required=True)
    staging.add_argument("--stage-dir", type=Path, required=True)

    built = sub.add_parser("manifest", help="write the manifest parquet and integrity report")
    built.add_argument("--stage-dir", type=Path, required=True)

    prov = sub.add_parser("provenance", help="write raw/_provenance/")
    prov.add_argument("--stage-dir", type=Path, required=True)
    prov.add_argument("--work-dir", type=Path, required=True)

    sent = sub.add_parser("upload", help="copy the staged tree to the data bucket")
    sent.add_argument("--stage-dir", type=Path, required=True)

    uploaded = sub.add_parser("verify-upload", help="check every staged key arrived in the bucket")
    uploaded.add_argument("--stage-dir", type=Path, required=True)

    return parser


def _require_host(host: str) -> str:
    if not host:
        raise SystemExit("no host: pass --host or set BDD100K_HOST")
    return host


def _staged_keys(stage_dir: Path) -> set[str]:
    """Every staged file as the S3 key it will have.

    The staged tree *is* the key set -- that is what staging is for -- so this
    is a walk rather than a second construction of the same paths.
    """
    return {
        path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*") if path.is_file()
    }


def _data_store() -> ObjectStore:
    """The data bucket, behind a store sized for the upload's thread pool."""
    return FsspecStore.connect(bucket_name(DATA_BUCKET), pool=tree.WORKERS)


def _upload(store: ObjectStore, stage_dir: Path) -> None:
    """The staged tree, every file at the key it was staged at.

    160,000 small objects, so the request rate is the whole cost of this step,
    and `tree.upload` spends it on a thread pool.
    """
    tree.upload(store, stage_dir, what="staged files")


def _listed_keys(store: ObjectStore) -> set[str]:
    """Every key under the prefixes ingest writes, one listing per prefix."""
    return {key for prefix in _WRITTEN_PREFIXES for key in store.keys(prefix)}


def _verify_upload(store: ObjectStore, stage_dir: Path) -> None:
    """The check the pre-upload suite structurally cannot make: that every staged key arrived.

    Catches a partial upload, where the earlier ID-set check catches a partial
    download.
    """
    staged = _staged_keys(stage_dir)
    listed = _listed_keys(store)
    missing = staged - listed

    if missing:
        sample = sorted(missing)[:10]
        raise SystemExit(
            f"{len(missing):,} of {len(staged):,} staged keys are not in the bucket. "
            f"First few: {sample}"
        )

    # Not a failure. A re-ingest under a new label_source, or a partition
    # written by a later step, both leave keys here that this build never
    # staged.
    extra = len(listed - staged)
    log.info("all %d staged keys are present", len(staged))
    log.info("%d other keys under raw/ and derived/ this build did not write", extra)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)

    if args.command == "url":
        print(ARCHIVES[args.archive].url(_require_host(args.host)))

    elif args.command == "verify-archives":
        provenance.verify_archives(args.work_dir, _require_host(args.host))

    elif args.command == "stage":
        pool = stage(args.extract_dir, args.stage_dir)
        log.info("staged %d images and %d labels", pool.image_count, pool.label_count)
        for split, ids in sorted(pool.images.items()):
            log.info("  %s: %d images, %d labels", split.value, len(ids), len(pool.labels[split]))
        if pool.ignored:
            log.info("ignored %d files outside the pool layout", len(pool.ignored))
            log.info("  first few: %s", pool.ignored[:5])

    elif args.command == "manifest":
        manifest.write(args.stage_dir)

    elif args.command == "provenance":
        provenance.write(args.stage_dir, args.work_dir)

    elif args.command == "upload":
        _upload(_data_store(), args.stage_dir)

    elif args.command == "verify-upload":
        _verify_upload(_data_store(), args.stage_dir)


if __name__ == "__main__":
    main()

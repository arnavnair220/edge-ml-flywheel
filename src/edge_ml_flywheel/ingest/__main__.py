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

log = logging.getLogger("edge_ml_flywheel.ingest")

# `aws s3 ls --recursive` prints date, time, size and key.
_LISTING_FIELDS = 4


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

    uploaded = sub.add_parser("verify-upload", help="check every staged key arrived in S3")
    uploaded.add_argument("--stage-dir", type=Path, required=True)
    uploaded.add_argument("--listing", type=Path, required=True)

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


def _listed_keys(listing: Path) -> set[str]:
    """Keys out of `aws s3 ls --recursive` output.

    Lines are `<date> <time> <size> <key>`. Zero-byte entries whose key ends in
    a slash are the console's directory placeholders and are not objects anyone
    staged.
    """
    keys: set[str] = set()
    for line in listing.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=_LISTING_FIELDS - 1)
        if len(parts) != _LISTING_FIELDS:
            continue
        key = parts[-1]
        if not key.endswith("/"):
            keys.add(key)
    return keys


def _verify_upload(stage_dir: Path, listing: Path) -> None:
    staged = _staged_keys(stage_dir)
    listed = _listed_keys(listing)
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

    elif args.command == "verify-upload":
        _verify_upload(args.stage_dir, args.listing)


if __name__ == "__main__":
    main()

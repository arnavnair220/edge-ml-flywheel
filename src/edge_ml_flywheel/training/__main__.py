"""The training steps, one subcommand each.

Three commands in the order a cycle uses them:

    python -m edge_ml_flywheel.training stage-base --weights yolo11n.pt
    python -m edge_ml_flywheel.training prepare --run-id <id> --cycle 0 --max-images 300
    python -m edge_ml_flywheel.training launch --run-id <id> --cycle 0 --seed 1 --epochs 1 --wait

`stage-base` is not a step anyone has to run: `prepare` stages the base itself
when the bucket has none, so a fresh account bootstraps on its first cycle. The
command survives for the case that file has to come from somewhere other than
its release URL, which is what `--weights` supplies. `prepare` is run once per
cycle, because the manifest it writes is what all five seeds train on and is the
record of what the challenger was trained on. `launch` is run once per seed.

That split is why the seed count is not a flag here. Five seeds are five jobs
over one prepared cycle, run in parallel from a Step Functions `Map` once the
state machine exists (design section 4.2); by hand, they are five invocations of
`launch`. A `--seeds 5` flag would be a loop in a CLI competing with the `Map`
state that is meant to own it.

Logging goes to stderr and the job name to stdout, the way `run register` puts
the run ID there: the name is what `wait` and every console lookup take.
"""

import argparse
import logging
import sys
from pathlib import Path

from edge_ml_flywheel.conventions import Cycle, Seed, new_model_version, parse_run_id
from edge_ml_flywheel.training import job, launch

log = logging.getLogger("edge_ml_flywheel.training")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.training")
    parser.add_argument("--profile", help="AWS profile. Defaults to the environment's.")
    sub = parser.add_subparsers(dest="command", required=True)

    staged = sub.add_parser("stage-base", help="upload the COCO base weights, once")
    # Optional, because `prepare` stages the base itself on the first cycle of a
    # fresh account and this command is then only needed to supply the file from
    # somewhere other than its release URL.
    staged.add_argument(
        "--weights",
        type=Path,
        help="Local yolo11n.pt. Omitted, the release copy is fetched if the bucket has none.",
    )

    prepared = sub.add_parser(
        "prepare", help="write the image manifest and source archive for one cycle"
    )
    prepared.add_argument("--run-id", required=True)
    prepared.add_argument("--cycle", type=int, required=True)
    # 0 rather than a number, because "all of them" is the real default and any
    # positive default would be a cap nobody chose silently shrinking a real run.
    prepared.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap the training set, for a short skeleton run. 0 trains on the whole labeled set.",
    )
    prepared.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite an existing manifest. Only safe before any seed has trained on it.",
    )

    started = sub.add_parser("launch", help="start one seed's training job")
    started.add_argument("--run-id", required=True)
    started.add_argument("--cycle", type=int, required=True)
    started.add_argument("--seed", type=int, required=True)
    started.add_argument("--epochs", type=int, required=True)
    started.add_argument("--batch", type=int, default=job.Recipe(epochs=1).batch)
    started.add_argument("--instance-type", default=job.Compute().instance_type)
    started.add_argument(
        "--on-demand",
        action="store_true",
        help="Turn managed spot off. Spot is the default and an interrupt restarts the job.",
    )
    started.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job stops, then report the channel download time.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)
    aws = launch.session(args.profile)

    if args.command == "stage-base":
        print(launch.stage_base(aws, args.weights) if args.weights else launch.ensure_base(aws))

    elif args.command == "prepare":
        print(
            launch.prepare(
                aws,
                parse_run_id(args.run_id),
                Cycle(args.cycle),
                launch.Preparation(
                    # The checkout this command was run from. The control Lambda
                    # passes its deployed tree instead -- see `launch.archive`.
                    code=launch.checkout_archive(),
                    max_images=args.max_images,
                    replace=args.replace,
                ),
            )
        )

    elif args.command == "launch":
        version = new_model_version(parse_run_id(args.run_id), Cycle(args.cycle))
        log.info("model version %s", version)

        name = launch.start(
            aws,
            version,
            Seed(args.seed),
            job.Recipe(epochs=args.epochs, batch=args.batch),
            job.Compute(instance_type=args.instance_type, use_spot=not args.on_demand),
        )
        print(name)

        if args.wait:
            launch.wait(aws, name)


if __name__ == "__main__":
    main()

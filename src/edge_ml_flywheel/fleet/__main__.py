"""The fleet steps, one subcommand each.

`run.__main__`'s shape for the plane after promotion. **The cycle performs all of
this itself** -- `Deploy`, `FleetScore` and `Canary` are states, and the pool pass
they drive is what selection ranks -- so these commands are the hand-operated
copy of the same three calls, for the cases a state machine is the wrong tool
for: re-running a pass after a device failure, restoring a version once the
execution that deployed it has ended, and looking at what the fleet is doing.

Each one goes through the same functions the control steps do, so an operator and
the state machine cannot form two different opinions about one rollout.

Four subcommands, in the order a cycle uses them:

    deploy    stage the code, publish the component, roll it out
    canary    read what the device said and run the gate over it
    rollback  deploy the previous cycle's version again
    status    what the fleet is running, and what it last reported

Logging goes to stderr so `deploy` can put the deployment ID on stdout for a
shell to capture, the way `run register` does with a run ID.
"""

import argparse
import json
import logging
import sys
from typing import Any

from edge_ml_flywheel.conventions import (
    PROJECT,
    Cycle,
    ModelVersion,
    RunId,
    Seed,
    model_version_cycle,
    parse_model_version,
    parse_run_id,
)
from edge_ml_flywheel.fleet import component, deploy, telemetry
from edge_ml_flywheel.training import launch as base

log = logging.getLogger("edge_ml_flywheel.fleet")

# The thing group Terraform creates, and the only deployment target. One group
# rather than a device ARN even at one device, because a second device joins a
# group without any of this changing -- targeting the thing directly would make
# growing the fleet a change to every command here.
THING_GROUP = "devices"

# Design section 4.3's replay, from the one place it is stated. The flags below
# offer it back rather than restating the numbers: a first deployment onto a
# fresh instance is worth being able to run short, and every deployment after it
# should be the standard measurement without anyone typing four values.
STANDARD = component.STANDARD


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edge_ml_flywheel.fleet")
    sub = parser.add_subparsers(dest="command", required=True)

    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--run-id", required=True)

    deployed = argparse.ArgumentParser(add_help=False, parents=[run])
    deployed.add_argument(
        "--version",
        required=True,
        help="The model version to act on. A cycle's challenger, or the champion being restored.",
    )

    rolled = sub.add_parser(
        "deploy", parents=[deployed], help="stage this version's frames and code and roll it out"
    )
    rolled.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed whose artifact ships. Seed 1 by convention; the manifest records which.",
    )
    rolled.add_argument(
        "--git-commit",
        required=True,
        help="40-character SHA of the tree the device runs. The code archive is keyed by it.",
    )
    rolled.add_argument(
        "--cycle",
        type=int,
        default=None,
        help=(
            "The cycle doing the deploying, when it is not the model's own. A cycle that "
            "rejected its challenger redeploys the champion with its own sample, and needs a "
            "component version of its own to do it. Defaults to the version's cycle."
        ),
    )
    rolled.add_argument("--frames", type=int, default=STANDARD.frames)
    rolled.add_argument("--warmup", type=int, default=STANDARD.warmup)
    rolled.add_argument("--image-size", type=int, default=STANDARD.image_size)
    rolled.add_argument("--confidence-floor", type=float, default=STANDARD.confidence_floor)

    checked = sub.add_parser(
        "canary", parents=[deployed], help="run the canary gate over what the device reported"
    )
    checked.add_argument(
        "--champion",
        help="The version this one replaces, whose replay supplies the throughput floor. "
        "Omit on a run's first deployment, which is its own baseline.",
    )

    back = sub.add_parser("rollback", parents=[run], help="deploy a previous cycle's version again")
    back.add_argument(
        "--to",
        required=True,
        help="The model version to restore. Usually the champion the failed cycle challenged.",
    )

    sub.add_parser("status", parents=[run], help="what the fleet is running and last reported")
    return parser


def _target(aws: Any) -> str:
    return component.thing_group_arn(
        str(aws.region_name), base.account_id(aws), f"{PROJECT}-{THING_GROUP}"
    )


def _deploy(aws: Any, args: argparse.Namespace) -> str:
    """Stage, publish, roll out. In that order, and the order is the point.

    Everything a device downloads exists in S3 before a component version names
    it, and the component version exists before a deployment names it. Each step
    fails against the previous one's output, so the failure an operator sees is
    the first thing that was wrong rather than a device timing out on an artifact
    nobody uploaded.
    """
    run_id: RunId = parse_run_id(args.run_id)
    version: ModelVersion = parse_model_version(args.version)
    cycle = Cycle(args.cycle) if args.cycle is not None else model_version_cycle(version)

    release = component.Release(
        version=version, seed=Seed(args.seed), git_commit=args.git_commit, cycle=cycle
    )
    replay = component.Replay(
        frames=args.frames,
        warmup=args.warmup,
        image_size=args.image_size,
        confidence_floor=args.confidence_floor,
    )

    deploy.sampled_frames(aws, run_id, cycle)
    deploy.stage_code(aws, release.git_commit, deploy.checkout_archive())
    deploy.publish_component(aws, deploy.recipe_for(aws, release, replay))

    log.info("%s", deploy.describe(version, base.buckets(aws)))
    return deploy.redeploy(aws, version, _target(aws), cycle=cycle)


def _canary(aws: Any, args: argparse.Namespace) -> bool:
    """Read the pass, run the gate, print the verdict. Return whether it passed.

    `deploy.judge` rather than the gate directly, because the cycle's `Canary`
    step calls the same function: a verdict an operator reads by hand and a
    verdict the state machine acts on must be one judgement over one reduction,
    not two that can disagree about a rollout.

    `ranks` is printed beside `passed` because they can differ, and the
    difference is the whole point of the pair -- a slow model is rolled back and
    its detections are still bought from.
    """
    run_id: RunId = parse_run_id(args.run_id)
    version: ModelVersion = parse_model_version(args.version)
    champion = parse_model_version(args.champion) if args.champion else None

    verdict = deploy.judge(aws, run_id, version, champion)
    print(
        json.dumps(
            {"passed": verdict.passed, "ranks": verdict.ranks, "reason": verdict.reason}, indent=2
        )
    )
    return verdict.passed


def _status(aws: Any, run_id: RunId) -> None:
    """What the fleet was told to run, beside what it says it ran.

    Both, for `run.__main__._show`'s reason: the question anyone actually asks is
    whether the device is on the version it was sent, and that is answered by the
    pair. They come from different places on purpose -- one from the deployment,
    one from the device -- so a disagreement between them is visible rather than
    reconciled by whichever is read.
    """
    intended = deploy.deployed(aws, run_id, _target(aws))
    documents = deploy.records(aws, run_id)

    reported: dict[str, Any] | None = None
    if intended is not None:
        try:
            report = telemetry.report(documents, intended)
            reported = {
                "thing": report.thing,
                "frames": report.reported,
                "of": report.replayed,
                "starts": report.starts,
                "p95_ms": round(report.p95_ms, 1),
                "cold_start_ms": round(report.cold_start_ms, 1),
                "throughput_fps": round(report.throughput_fps, 2),
            }
        except telemetry.NoReplayError:
            log.info("%s is deployed but has reported no completed replay", intended)

    print(json.dumps({"deployed": intended, "reported": reported}, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parser().parse_args(argv)
    aws = base.session()

    if args.command == "deploy":
        print(_deploy(aws, args))

    elif args.command == "canary":
        # A failed canary exits non-zero, so a shell that chains a rollback
        # behind it does the right thing without reading the JSON. The verdict
        # and its reason are on stdout either way -- a rejection is a deliverable
        # (design section 5), not an error to be swallowed by the exit code.
        if not _canary(aws, args):
            raise SystemExit(1)

    elif args.command == "rollback":
        run_id = parse_run_id(args.run_id)
        target = parse_model_version(args.to)
        log.info("rolling %s back to cycle %d", run_id, model_version_cycle(target))
        print(deploy.redeploy(aws, target, _target(aws)))

    elif args.command == "status":
        _status(aws, parse_run_id(args.run_id))


if __name__ == "__main__":
    main()

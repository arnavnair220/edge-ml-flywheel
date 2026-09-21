"""Gathering a finished run's artifacts, and writing the one document they reduce to.

The impure half, split from `reporting.summary` the way `registry.launch` is
split from `registry.manifest`: the reads and the write are here, the schema and
the fold are there, so the encoding is testable without a bucket.

**Every read is of something already written.** Nothing here runs a job, asks a
device anything, or re-measures a number -- a summary is assembled out of
manifests, gate reports and telemetry that the cycles left behind, and a run
whose artifacts are intact can be summarized again next year to the same bytes.

**The canary verdict is `fleet.deploy.judge`'s, not this module's.** It could be
recomputed here from the same telemetry in six lines, and that is exactly what it
must not be: the cycle's `Canary` state and an operator's `fleet canary` already
go through one judgement so they cannot disagree about a rollout, and a summary
forming a third opinion would be the one place in the project where the record
and the decision could differ.
"""

import json
import logging
from typing import Any

import boto3

from edge_ml_flywheel.conventions import (
    Cycle,
    GateResult,
    ModelVersion,
    RunId,
    gate_report_key,
    new_model_version,
    run_summary_key,
    uri,
)
from edge_ml_flywheel.fleet import deploy, telemetry
from edge_ml_flywheel.gates.thresholds import Gate
from edge_ml_flywheel.registry import launch as registry
from edge_ml_flywheel.reporting import summary
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)


def _gate_report(aws: boto3.Session, bucket: str, run_id: RunId, cycle: Cycle) -> dict[str, Any]:
    """One cycle's gate report, or a refusal naming what should have written it.

    Refused rather than skipped. A cycle with no report is a cycle whose
    evaluation job did not finish, and a summary that quietly omitted its delta
    would describe a run that went better than it did.
    """
    key = gate_report_key(run_id, cycle)
    if not base.exists(aws, bucket, key):
        raise SystemExit(
            f"{uri(bucket, key)} does not exist, so cycle {cycle} of {run_id} reached no verdict. "
            f"A run cannot be summarized while one of its cycles has no gate report."
        )
    body = aws.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return dict(json.loads(body))


def _measured(
    aws: boto3.Session,
    run_id: RunId,
    version: ModelVersion,
    champion: ModelVersion | None,
    documents: list[dict[str, Any]],
) -> tuple[GateResult | None, float | None]:
    """The canary verdict and the p95 for a version that reached a device.

    Both `None` together for a version no device finished replaying. At the end
    of a run that is a challenger which never shipped, not a measurement still
    outstanding -- which is the whole reason the summary is written here rather
    than maintained as the run goes.

    `judge` is given the champion the version was rolled out *over*, so the
    throughput comparison is against what the device was running before. A first
    promotion has no predecessor and passes `None`, which `judge` reads as no
    drift check rather than as a champion of zero throughput.
    """
    try:
        report = telemetry.report(documents, version)
    except telemetry.NoReplayError:
        log.info("%s was never replayed on a device, so it has no latency", version)
        return None, None

    verdict = deploy.judge(aws, run_id, version, champion)
    return (
        GateResult(gate=str(Gate.CANARY), passed=verdict.passed, reason=verdict.reason),
        report.p95_ms,
    )


def facts(aws: boto3.Session, run_id: RunId, cycles: int) -> list[summary.CycleFacts]:
    """What each of a run's cycles recorded, in cycle order.

    `cycles` is how many the run claimed, so the range is what the counter
    reached rather than what a listing of the bucket happens to show. A cycle
    that claimed its number and failed before registration has no manifest and
    stops the summary here, which is correct: that run did not finish, and the
    document is written once at the end of one that did.

    The champion is carried forward as the cloud gates decide it, because that is
    what `deploy` was told to roll out. Whether the device then kept it is the
    canary's answer, asked below, and it cannot be known before the rollout it
    judges.
    """
    artifacts = base.buckets(aws).artifacts
    documents = deploy.records(aws, run_id)

    gathered: list[summary.CycleFacts] = []
    champion: ModelVersion | None = None
    for index in range(cycles):
        cycle = Cycle(index)
        version = new_model_version(run_id, cycle)
        manifest = registry.read_manifest(aws, artifacts, version)
        report = _gate_report(aws, artifacts, run_id, cycle)

        rolled_out = bool(manifest.gates) and all(gate.passed for gate in manifest.gates)

        canary: GateResult | None = None
        p95: float | None = None
        if rolled_out:
            canary, p95 = _measured(aws, run_id, version, champion, documents)

        gathered.append(
            summary.CycleFacts(
                manifest=manifest,
                delta=summary.delta_of(report),
                canary=canary,
                p95_ms=p95,
            )
        )

        if rolled_out:
            champion = version

    return gathered


def write(aws: boto3.Session, run_id: RunId, cycles: int) -> str:
    """Assemble the run's summary and put it where a reader can fetch it.

    Read back after the write, like the model manifest and the partition: the
    document is the deliverable, and a deliverable is not reported as written on
    the strength of a `put_object` that returned. The round trip through
    `from_document` is what proves a field survived, which matters more here than
    elsewhere because this is the one document meant to be opened away from the
    code that wrote it.

    Returns the URI, so the state machine's output and an operator's terminal
    both name the object rather than describing it.
    """
    document = summary.to_document(run_id, summary.rows(facts(aws, run_id, cycles)))

    artifacts = base.buckets(aws).artifacts
    key = run_summary_key(run_id)
    aws.client("s3").put_object(
        Bucket=artifacts,
        Key=key,
        Body=json.dumps(document, indent=2, sort_keys=True).encode(),
        ContentType="application/json",
    )

    body = aws.client("s3").get_object(Bucket=artifacts, Key=key)["Body"].read()
    restored = summary.from_document(json.loads(body))
    if len(restored) != len(document["cycles"]):
        raise SystemExit(
            f"{uri(artifacts, key)} read back with {len(restored)} of "
            f"{len(document['cycles'])} cycles, so the write was lossy."
        )

    log.info("summarized %d cycles of %s", len(restored), run_id)
    return uri(artifacts, key)

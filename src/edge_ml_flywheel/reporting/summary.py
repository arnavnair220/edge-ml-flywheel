"""The run summary's rows, and the document they are stored as.

Pure, for `registry.manifest`'s reason: this holds the entire storage schema of
the deliverable's one document, so the encoding round-trips in a test with no
credentials, no bucket and no run.

**Nothing here decides anything.** Every field arrives measured -- the manifest
counted the labels, the evaluation job bootstrapped the delta, the device timed
the frames -- and the only computation in the module is `deployed`, which is a
fold over verdicts that were already recorded. So summarizing one run twice
cannot reach two answers unless one of the artifacts changed underneath.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from edge_ml_flywheel.conventions import (
    Cycle,
    CycleSummary,
    GateResult,
    ModelManifest,
    ModelVersion,
    RunId,
    model_version_cycle,
    parse_model_version,
    parse_run_id,
)

# The schema version of the document itself, not of anything it describes. It is
# here because this is the one document meant to be downloaded and read away from
# the code that wrote it, so the reader that opens a run summarized months ago is
# not necessarily the reader this field was written by.
SCHEMA: int = 1


@dataclass(frozen=True, slots=True)
class CycleFacts:
    """What one cycle recorded, gathered from the three places it recorded it.

    The manifest carries the version, the cumulative labels and the three gates
    the evaluation job applied. `delta` comes from the gate report, which is the
    only artifact that holds the band. `canary` and `p95_ms` come from the
    telemetry, and both are absent for a cycle no device finished replaying --
    which by the time a run is summarized means a cycle whose challenger never
    reached a device, rather than one whose device has not reported yet.

    The canary arrives separately from the other three gates rather than being
    read off the manifest with them, because the manifest is written at
    registration and the canary is not asked until the artifact has reached a
    device. A manifest that carried it would have to be rewritten, and it is one
    of the write-once documents this one is a reduction of.
    """

    manifest: ModelManifest
    delta: tuple[float, float, float] | None
    canary: GateResult | None
    p95_ms: float | None


def _gates(facts: CycleFacts) -> tuple[GateResult, ...]:
    """The cycle's verdicts, the canary last.

    Last because that is the order they were reached in, and the order is the
    one thing a flat list of verdicts can say about when each was asked.
    """
    if facts.canary is None:
        return facts.manifest.gates
    return (*facts.manifest.gates, facts.canary)


def rows(facts: Sequence[CycleFacts]) -> tuple[CycleSummary, ...]:
    """One row per cycle, in cycle order, with the fleet's version folded through.

    `deployed` is the only field not read straight off an artifact, and it is a
    carry rather than a computation: a cycle whose gates all passed put its own
    model on the device, and a cycle that failed any of them left whatever was
    already there. That one rule covers both ways a cycle can fail -- a
    challenger rejected before it shipped and a rollout undone by the canary --
    because in both the device ends the cycle on the previous version.

    Sorted here rather than trusted from the caller. The fold is order-dependent,
    so a caller that listed cycle 3 before cycle 2 would carry the wrong version
    forward, and a reader of the stored document would have no way to tell.
    """
    ordered = sorted(facts, key=lambda entry: model_version_cycle(entry.manifest.version))

    built: list[CycleSummary] = []
    deployed: ModelVersion | None = None
    for entry in ordered:
        verdicts = _gates(entry)
        if verdicts and all(verdict.passed for verdict in verdicts):
            deployed = entry.manifest.version

        built.append(
            CycleSummary(
                cycle=model_version_cycle(entry.manifest.version),
                version=entry.manifest.version,
                labels_spent=entry.manifest.labels_spent,
                gates=verdicts,
                delta=entry.delta,
                deployed=deployed,
                p95_ms=entry.p95_ms,
            )
        )

    return tuple(built)


def delta_of(report: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """The paired delta out of a gate report, as the triple a row stores.

    `None` for the baseline cycle, which had no champion to be compared against
    and whose report records that as a null rather than as a zero delta. Those
    are different facts: one is a comparison that could not be made, the other a
    comparison that came out even.
    """
    measured = report.get("delta")
    if measured is None:
        return None
    return (
        float(measured["observed"]),
        float(measured["lower"]),
        float(measured["upper"]),
    )


def to_document(run_id: RunId, summaries: Sequence[CycleSummary]) -> dict[str, Any]:
    """The `run_summary_key` document: the run, and a row per cycle.

    `run_id` is written even though it is in the key, unlike the manifest, which
    deliberately omits what its version already names. The difference is that
    this document is the one meant to be downloaded and read on its own -- a
    file on a desk with no prefix above it has to say which run it describes.

    `gates` keeps the spelling `report_document` and `manifest.to_document` use,
    so a verdict is one shape across every document in the project that carries
    one.
    """
    return {
        "schema": SCHEMA,
        "run_id": parse_run_id(run_id),
        "cycles": [
            {
                "cycle": int(row.cycle),
                "version": str(row.version),
                "labels_spent": row.labels_spent,
                "gates": [
                    {"gate": str(gate.gate), "passed": gate.passed, "reason": gate.reason}
                    for gate in row.gates
                ],
                "delta": (
                    None
                    if row.delta is None
                    else {
                        "observed": row.delta[0],
                        "lower": row.delta[1],
                        "upper": row.delta[2],
                    }
                ),
                "deployed": None if row.deployed is None else str(row.deployed),
                "p95_ms": row.p95_ms,
            }
            for row in summaries
        ],
    }


def from_document(stored: Mapping[str, Any]) -> tuple[CycleSummary, ...]:
    """The inverse, with `CycleSummary.__post_init__` as the validator.

    Used by the round-trip test and by anything that later reads a finished run.
    A schema it does not know is refused rather than parsed on the assumption
    that the fields it recognizes still mean what they did: this document
    outlives the code that wrote it by design, being the one a reader downloads.
    """
    version = int(stored.get("schema", 0))
    if version != SCHEMA:
        raise ValueError(
            f"a run summary at schema {version}, and this reader knows schema {SCHEMA}. "
            f"Regenerate it rather than reading it as though the fields still line up."
        )

    return tuple(
        CycleSummary(
            cycle=Cycle(int(row["cycle"])),
            version=parse_model_version(str(row["version"])),
            labels_spent=int(row["labels_spent"]),
            gates=tuple(
                GateResult(
                    gate=str(entry["gate"]),
                    passed=bool(entry["passed"]),
                    reason=str(entry["reason"]),
                )
                for entry in row["gates"]
            ),
            delta=delta_of(row),
            deployed=(
                None if row["deployed"] is None else parse_model_version(str(row["deployed"]))
            ),
            p95_ms=None if row["p95_ms"] is None else float(row["p95_ms"]),
        )
        for row in stored["cycles"]
    )

"""What a device publishes, and how a replay is read back out of it.

Both halves in one module for `conventions.sha256sums_document`'s reason: the
writer runs on a device and the reader runs days later in a notebook, separated
by an IoT rule, an S3 object and a Firehose this project does not have. They are
the two ends of one format, and a format spelled at two ends drifts at one of
them.

**Nothing here calls AWS.** The device's publish and the reader's listing are
`replay` and `deploy`; this is the documents and the reduction, so the shape a
device emits is testable without a device and the gate's input is testable
without a bucket.

**A record says what was observed, never what it meant.** No record carries a
pass, a threshold or a comparison -- `gates.canary` owns those and is given a
`ReplayReport`. The division is `DetectionRow`'s: one file serves the gate and
the charts precisely because it took no view.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from edge_ml_flywheel.conventions import (
    FrameRow,
    ImageId,
    ModelVersion,
    ReplayReport,
    RunId,
    TelemetryKind,
    parse_image_id,
    parse_model_version,
    parse_run_id,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Source:
    """Who is publishing, and about what. Every record carries these three.

    A type rather than three arguments repeated on both builders, and it earns
    that by being the thing a reader filters on: a telemetry prefix holds every
    cycle's records and every device's, so which run, which version and which
    device is the question asked of every object before anything else is.

    The run and the version are in the record as well as in the topic and the
    landing prefix, which is the one deliberate repetition in this format. An
    object read on its own -- downloaded, pasted into an issue, opened six weeks
    later -- has to say which model produced it, and a prefix is not carried by
    the bytes.
    """

    run_id: RunId
    version: ModelVersion
    thing: str

    def __post_init__(self) -> None:
        parse_run_id(self.run_id)
        parse_model_version(self.version)
        if not self.thing:
            raise ValueError("a record from an unnamed device cannot be told from another's")


class NoReplayError(Exception):
    """No summary record for the version asked about.

    Distinct from an incomplete replay, which is a verdict `gates.canary`
    reaches, and it has to be: a replay that lost half its frames still produces
    a report and a recorded rejection, while a replay nothing was heard from
    produces neither and is a different question -- the component never started,
    or the rule is not landing objects, or the deployment has not reached the
    device yet. Raising keeps those from being reported as a failed canary, which
    would send the next attempt looking at the model.
    """


def _envelope(kind: TelemetryKind, source: Source) -> dict[str, Any]:
    """The four fields every record carries, whatever kind it is."""
    return {
        "kind": kind.value,
        "run_id": str(source.run_id),
        "version": str(source.version),
        "thing": source.thing,
    }


def frames_document(source: Source, seq: int, frames: Sequence[FrameRow]) -> dict[str, Any]:
    """One published batch of replayed frames.

    `seq` counts batches from zero within one start of the component, and it is
    what makes a redelivered message recognizable as one. Without it two copies
    of a batch are two batches, and every count below is wrong by however many
    times the network retried.
    """
    if seq < 0:
        raise ValueError(f"a batch sequence is a position and cannot be negative: {seq}")
    if not frames:
        raise ValueError("a batch of no frames is not a batch")
    return {
        **_envelope(TelemetryKind.FRAMES, source),
        "seq": seq,
        "frames": [
            {
                "image_id": str(frame.image_id),
                "inference_ms": frame.inference_ms,
                "scores": list(frame.scores),
            }
            for frame in frames
        ],
    }


def replay_document(
    source: Source,
    artifact_sha256: str,
    cold_start_ms: float,
    starts: int,
    replayed: int,
) -> dict[str, Any]:
    """The one summary record a completed replay publishes.

    Published last, after the final batch, so that its presence means the run
    finished rather than that it began. A summary written first would turn every
    crash halfway through into a replay that looks complete and is short of
    frames for a reason nothing recorded.
    """
    return {
        **_envelope(TelemetryKind.REPLAY, source),
        "artifact_sha256": artifact_sha256,
        "cold_start_ms": cold_start_ms,
        "starts": starts,
        "replayed": replayed,
    }


def _frames(document: Mapping[str, Any]) -> list[FrameRow]:
    return [
        FrameRow(
            image_id=parse_image_id(str(frame["image_id"])),
            inference_ms=float(frame["inference_ms"]),
            scores=tuple(float(score) for score in frame["scores"]),
        )
        for frame in document["frames"]
    ]


def frames_of(documents: Iterable[Mapping[str, Any]], version: ModelVersion) -> list[FrameRow]:
    """Every distinct frame one version's replay reported, in arrival order.

    Deduplicated by batch sequence rather than by image: a repeated `seq` is one
    message delivered twice, which is ordinary at least-once behaviour and must
    not count as work the device did. A component that restarted republishes from
    `seq` 0, so this folds a second pass onto the first -- deliberately, because
    a restart is `ReplayReport.starts`' question and a check answering it twice
    by two routes is one that can disagree with itself.
    """
    wanted = parse_model_version(version)
    seen: dict[int, list[FrameRow]] = {}
    for document in documents:
        if document.get("kind") != TelemetryKind.FRAMES.value:
            continue
        if str(document.get("version")) != str(wanted):
            continue
        seq = int(document["seq"])
        if seq in seen:
            log.info("batch %d of %s arrived more than once", seq, wanted)
            continue
        seen[seq] = _frames(document)

    return [frame for seq in sorted(seen) for frame in seen[seq]]


def _summary(documents: Iterable[Mapping[str, Any]], version: ModelVersion) -> Mapping[str, Any]:
    """The summary record, or the highest-`starts` one if a restart left two.

    Two summaries mean the component ran to completion twice, and the later run
    is the state the device is actually in -- so it is the one reported, and the
    `starts` it carries is what makes `gates.canary` reject the deployment. The
    alternative, refusing to build a report at all, would turn a finding the gate
    is built to make into an error the operator has to interpret.
    """
    wanted = parse_model_version(version)
    found = [
        document
        for document in documents
        if document.get("kind") == TelemetryKind.REPLAY.value
        and str(document.get("version")) == str(wanted)
    ]
    if not found:
        raise NoReplayError(
            f"no replay summary for {wanted}. Either the component has not finished, it never "
            f"started, or its telemetry is not landing -- none of which is a failed canary"
        )
    if len(found) > 1:
        log.warning("%d replay summaries for %s, reporting the latest start", len(found), wanted)
    return max(found, key=lambda document: int(document["starts"]))


def report(documents: Iterable[Mapping[str, Any]], version: ModelVersion) -> ReplayReport:
    """One version's whole replay, as the canary gate reads it.

    The documents are every object under the run's telemetry prefix, unfiltered:
    a prefix holds every cycle's records and every device's, and narrowing it is
    this function's job rather than the caller's. Passing a pre-filtered set
    would put the filter at each call site, which is where a champion's report
    and a challenger's would come to be built by two slightly different rules and
    compared as though they were not.
    """
    documents = list(documents)
    summary = _summary(documents, version)
    frames = frames_of(documents, version)

    return ReplayReport(
        version=parse_model_version(version),
        thing=str(summary["thing"]),
        artifact_sha256=str(summary["artifact_sha256"]),
        cold_start_ms=float(summary["cold_start_ms"]),
        starts=int(summary["starts"]),
        replayed=int(summary["replayed"]),
        latencies_ms=tuple(frame.inference_ms for frame in frames),
    )


def image_ids(documents: Iterable[Mapping[str, Any]], version: ModelVersion) -> tuple[ImageId, ...]:
    """The frames a replay actually put through the model, in order.

    Separate from `report` because the gate does not want them and the realism
    check does: joining these against `selection_ranking_key` is what compares
    the fleet's view of a frame with the offline pass's, and design section 7.2
    keeps that a report rather than a selector. Kept here so the join reads the
    same records the gate did.
    """
    return tuple(frame.image_id for frame in frames_of(documents, version))

"""Tests for the telemetry format and the reduction the canary gate reads.

The writer and the reader are one module, so these mostly run one against the
other: a document built by `frames_document` is fed to `report`, which is the
round trip the format's two ends are separated by in real life -- a device, an
IoT rule, an S3 object and several hours.

What is asserted beyond the round trip is the behaviour under things that
actually happen in a fleet: a message delivered twice, a batch that never
arrived, a component that restarted and replayed again, and a prefix holding more
than one cycle's records.
"""

from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    FrameRow,
    ImageId,
    ModelVersion,
    RunId,
    TelemetryKind,
)
from edge_ml_flywheel.fleet import telemetry

RUN = RunId("20260812t143355z-v0-skeleton")
VERSION = ModelVersion("20260812t143355z-v0-skeleton-c003")
EARLIER = ModelVersion("20260812t143355z-v0-skeleton-c002")
SHA = "a" * 64

SOURCE = telemetry.Source(run_id=RUN, version=VERSION, thing="device-1")

# Two real BDD100K IDs, since `FrameRow` validates the shape.
IMAGES = (ImageId("0000f77c-6257be58"), ImageId("0000f77c-62c2a288"))


def a_frame(image_id: ImageId, ms: float) -> FrameRow:
    return FrameRow(image_id=image_id, inference_ms=ms)


def a_summary(source: telemetry.Source = SOURCE, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "source": source,
        "artifact_sha256": SHA,
        "cold_start_ms": 800.0,
        "starts": 1,
        "replayed": 2,
    }
    return telemetry.replay_document(**(fields | overrides))


def a_replay(source: telemetry.Source = SOURCE, **overrides: Any) -> list[dict[str, Any]]:
    """A whole replay: one batch of two frames, and the summary that closes it."""
    frames = [a_frame(IMAGES[0], 20.0), a_frame(IMAGES[1], 30.0)]
    return [
        telemetry.frames_document(source, 0, frames),
        a_summary(source, **overrides),
    ]


class TestTheDocuments:
    def test_every_record_says_which_model_produced_it(self) -> None:
        """A prefix is not carried by the bytes, so an object downloaded on its
        own has to say what it is about."""
        for document in a_replay():
            assert document["run_id"] == RUN
            assert document["version"] == VERSION
            assert document["thing"] == "device-1"

    def test_the_two_kinds_are_told_apart_by_one_field(self) -> None:
        batch, summary = a_replay()

        assert batch["kind"] == TelemetryKind.FRAMES
        assert summary["kind"] == TelemetryKind.REPLAY

    def test_a_batch_of_no_frames_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a batch"):
            telemetry.frames_document(SOURCE, 0, [])

    def test_a_negative_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            telemetry.frames_document(SOURCE, -1, [a_frame(IMAGES[0], 20.0)])

    def test_a_source_with_no_device_name_is_refused(self) -> None:
        """Two devices' records share a prefix neither of them partitions, so an
        unnamed one cannot be told from another's."""
        with pytest.raises(ValueError, match="unnamed device"):
            telemetry.Source(run_id=RUN, version=VERSION, thing="")


class TestTheReduction:
    def test_a_whole_replay_round_trips(self) -> None:
        report = telemetry.report(a_replay(), VERSION)

        assert report.thing == "device-1"
        assert report.artifact_sha256 == SHA
        assert report.starts == 1
        assert report.complete
        assert report.latencies_ms == (20.0, 30.0)

    def test_frames_come_back_in_batch_order(self) -> None:
        """Objects arrive from a listing in key order, which is not send order.
        The sequence is what restores it."""
        later = telemetry.frames_document(SOURCE, 1, [a_frame(IMAGES[0], 40.0)])
        first = telemetry.frames_document(SOURCE, 0, [a_frame(IMAGES[1], 10.0)])

        report = telemetry.report([later, first, a_summary(replayed=2)], VERSION)

        assert report.latencies_ms == (10.0, 40.0)

    def test_a_redelivered_batch_is_counted_once(self) -> None:
        """At-least-once delivery is ordinary, and a second copy of a batch is
        not work the device did."""
        batch, summary = a_replay()

        assert telemetry.report([batch, batch, summary], VERSION).reported == 2

    def test_a_lost_batch_shows_as_an_incomplete_replay(self) -> None:
        """Which is the finding, not an error: the summary says how many frames
        there should have been, and the batches say how many arrived."""
        report = telemetry.report([a_summary(replayed=450)], VERSION)

        assert report.reported == 0
        assert not report.complete

    def test_records_of_another_version_are_ignored(self) -> None:
        """One prefix holds every cycle's records, and narrowing is the reducer's
        job rather than the caller's -- so a champion's report and a
        challenger's are built by one rule."""
        other = telemetry.Source(run_id=RUN, version=EARLIER, thing="device-1")
        report = telemetry.report([*a_replay(), *a_replay(other)], VERSION)

        assert report.version == VERSION
        assert report.reported == 2

    def test_both_versions_are_readable_from_one_listing(self) -> None:
        """Which is what lets the canary compare a challenger against a champion
        without re-running the champion's replay."""
        other = telemetry.Source(run_id=RUN, version=EARLIER, thing="device-1")
        documents = [*a_replay(), *a_replay(other)]

        assert telemetry.report(documents, VERSION).version == VERSION
        assert telemetry.report(documents, EARLIER).version == EARLIER

    def test_a_restart_that_replayed_again_reports_the_later_start(self) -> None:
        """Refusing to build a report would turn a finding the gate is built to
        make into an error an operator has to interpret."""
        report = telemetry.report([*a_replay(), *a_replay(starts=2)], VERSION)

        assert report.starts == 2

    def test_no_summary_is_not_a_failed_canary(self) -> None:
        """The component has not finished, never started, or its telemetry is not
        landing. None of those is a verdict about the model."""
        batch = telemetry.frames_document(SOURCE, 0, [a_frame(IMAGES[0], 20.0)])

        with pytest.raises(telemetry.NoReplayError, match="no replay summary"):
            telemetry.report([batch], VERSION)

    def test_an_empty_prefix_is_not_a_failed_canary_either(self) -> None:
        with pytest.raises(telemetry.NoReplayError):
            telemetry.report([], VERSION)


class TestTheFramesThemselves:
    def test_the_replayed_image_ids_come_back_in_order(self) -> None:
        """What the realism check joins against the cycle's ranking. Kept beside
        the gate's reduction so the join reads the same records the gate did."""
        assert telemetry.image_ids(a_replay(), VERSION) == IMAGES

    def test_a_frame_the_model_found_nothing_in_survives_the_round_trip(self) -> None:
        """Dropping it would make a replay's frame count disagree with its
        summary for a reason nothing recorded."""
        report = telemetry.report(a_replay(), VERSION)

        assert report.reported == 2
        assert report.complete

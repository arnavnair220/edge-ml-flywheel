"""Tests for the canary gate, which judges a model that has already shipped.

Three checks over one device's replay, all operational. A detector is
deterministic, so there is no run-to-run spread on a device and nothing
statistical to test (design section 4.5); what these assert is that the gate
catches a wrong artifact, a component that did not run cleanly, and a model that
is slower on ARM than the one it replaces.

`ReplayReport` is constructed directly here rather than reduced from documents.
That reduction is `fleet.telemetry`'s and is tested there -- this suite is about
the verdict, and building the input by hand is what lets a test drive one field
to a boundary without composing a batch of telemetry to carry it.
"""

from typing import Any

import pytest

from edge_ml_flywheel.conventions import ModelVersion, ReplayReport
from edge_ml_flywheel.gates import Gate, Thresholds, canary_gate

VERSION = ModelVersion("20260812t143355z-v0-skeleton-c003")
CHAMPION_VERSION = ModelVersion("20260812t143355z-v0-skeleton-c002")

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def a_report(**overrides: Any) -> ReplayReport:
    """Four frames at 25 ms each, which is 40 frames a second."""
    fields: dict[str, Any] = {
        "version": VERSION,
        "thing": "device-1",
        "artifact_sha256": DIGEST,
        "cold_start_ms": 800.0,
        "starts": 1,
        "replayed": 4,
        "latencies_ms": (25.0, 25.0, 25.0, 25.0),
    }
    return ReplayReport(**(fields | overrides))


def a_champion(fps: float) -> ReplayReport:
    """A champion replay at a chosen frame rate, for the one comparison there is."""
    return a_report(version=CHAMPION_VERSION, latencies_ms=(1000.0 / fps,) * 4)


class TestPassing:
    def test_a_clean_replay_of_the_right_artifact_passes(self) -> None:
        verdict = canary_gate(a_report(), DIGEST, a_champion(40.0))

        assert verdict.passed
        assert verdict.gate == Gate.CANARY

    def test_a_first_deployment_passes_with_no_champion(self) -> None:
        """A run's first promotion has nothing to be slower than, which is the
        situation the quality gate meets at cycle 0 and answers the same way."""
        verdict = canary_gate(a_report(), DIGEST, champion=None)

        assert verdict.passed
        assert "own baseline" in verdict.reason

    def test_a_faster_challenger_is_not_a_failure(self) -> None:
        """Quantizing to a cheaper graph is a result rather than a regression, so
        only a drop is checked."""
        assert canary_gate(a_report(), DIGEST, a_champion(20.0)).passed

    def test_a_drop_exactly_at_the_allowance_passes(self) -> None:
        """A 10% allowance permits a 10% drop. 40 frames a second against a
        champion's 44.4 is the boundary the threshold admits."""
        verdict = canary_gate(a_report(), DIGEST, a_champion(40.0 / 0.9))

        assert verdict.passed

    def test_the_reason_carries_the_numbers_the_writeup_wants(self) -> None:
        """p95 and cold start are measured and gated nowhere (design section
        4.3), so this verdict is the only place a cycle records what the model
        did on real ARM silicon."""
        verdict = canary_gate(a_report(cold_start_ms=812.0), DIGEST, a_champion(40.0))

        assert "p95 25 ms" in verdict.reason
        assert "cold start 812 ms" in verdict.reason
        assert "40.0 frames/s" in verdict.reason


class TestFailing:
    def test_a_different_artifact_fails(self) -> None:
        """Not a corrupt download -- Greengrass refuses those -- but a recipe
        built over the wrong object, which is a rollout of a model no gate saw.
        """
        verdict = canary_gate(a_report(), OTHER_DIGEST, a_champion(40.0))

        assert not verdict.passed
        assert "manifest for" in verdict.reason

    def test_a_restart_fails(self) -> None:
        verdict = canary_gate(a_report(starts=2), DIGEST, a_champion(40.0))

        assert not verdict.passed
        assert "started 2 times" in verdict.reason

    def test_a_short_replay_fails(self) -> None:
        verdict = canary_gate(a_report(replayed=10), DIGEST, a_champion(40.0))

        assert not verdict.passed
        assert "4 of 10 replayed frames arrived" in verdict.reason

    def test_losing_more_throughput_than_the_allowance_fails(self) -> None:
        verdict = canary_gate(a_report(), DIGEST, a_champion(80.0))

        assert not verdict.passed
        assert "50.0% drop" in verdict.reason

    def test_every_failure_is_reported_together(self) -> None:
        """The verdict is the record of what the rollout found, and one that
        stopped at the first failure would describe less than what happened."""
        verdict = canary_gate(a_report(starts=3, replayed=9), OTHER_DIGEST, a_champion(80.0))

        assert not verdict.passed
        assert "manifest for" in verdict.reason
        assert "started 3 times" in verdict.reason
        assert "replayed frames arrived" in verdict.reason
        assert "drop over the" in verdict.reason


class TestWhatItReadsFrom:
    def test_no_check_needs_a_second_device(self) -> None:
        """Design section 4.5 lists five conditions and this implements three.
        Memory flatness over two replay hours and the distance between two
        confidence distributions both need a fleet this one does not have, and
        they are absent rather than approximated."""
        verdict = canary_gate(a_report(), DIGEST, a_champion(40.0))

        assert verdict.passed
        assert "memory" not in verdict.reason.lower()

    def test_the_champion_is_read_rather_than_re_measured(self) -> None:
        """A replay is a function of a model and a device, so the champion's is
        the one from the cycle it was deployed in. What the gate takes is that
        report, which is why it is an argument rather than something fetched."""
        champion = a_champion(40.0)

        assert canary_gate(a_report(), DIGEST, champion).passed
        assert champion.version == CHAMPION_VERSION


class TestTheThreshold:
    def test_a_tighter_allowance_rejects_a_smaller_drop(self) -> None:
        """`Thresholds` is constructed directly where a test drives a boundary,
        which is what the frozen dataclass is for."""
        thresholds = Thresholds(max_throughput_drop=0.01)
        verdict = canary_gate(a_report(), DIGEST, a_champion(42.0), thresholds)

        assert not verdict.passed

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_an_allowance_outside_zero_to_one_is_refused(self, value: float) -> None:
        """Zero refuses every deployment and one admits a device that has
        stopped, so neither is a threshold."""
        with pytest.raises(ValueError, match="not a fraction"):
            Thresholds(max_throughput_drop=value)

    def test_the_default_is_the_design_s_ten_percent(self) -> None:
        assert Thresholds().max_throughput_drop == 0.10

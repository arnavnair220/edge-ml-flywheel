"""Tests for the edge gate, which judges the artifact that actually ships.

Two thresholds, both properties of a file and a number: small enough to deliver,
and enough of the fp32 model left to be worth delivering. Speed is reported off
the fleet rather than gated (design section 4.3), so there is no third condition
here waiting on silicon.

`Thresholds` is constructed directly where a test drives a boundary, which is
what the frozen dataclass is for -- the defaults are the promotion rule and the
suite tightens a copy rather than the rule.
"""

import pytest

from edge_ml_flywheel.gates import Gate, Thresholds, edge_gate

# A passing artifact: comfortably inside the size ceiling, having lost little.
SMALL = 4 * 1024 * 1024
FP32 = 0.400


class TestPassing:
    def test_a_small_artifact_that_kept_its_accuracy_passes(self) -> None:
        verdict = edge_gate(fp32_map=FP32, int8_map=0.395, artifact_bytes=SMALL)

        assert verdict.passed
        assert verdict.gate == Gate.EDGE

    def test_the_reason_carries_both_numbers_and_what_was_retained(self) -> None:
        """A verdict is evidence or it is a bare boolean six weeks later."""
        verdict = edge_gate(fp32_map=0.400, int8_map=0.380, artifact_bytes=SMALL)

        assert "0.3800" in verdict.reason
        assert "0.4000" in verdict.reason
        assert "95.0% retained" in verdict.reason

    def test_a_loss_exactly_at_the_allowance_passes(self) -> None:
        """The threshold admits its own boundary, so a 5% allowance permits a 5%
        loss rather than rejecting it by a rounding error."""
        thresholds = Thresholds(max_quantization_loss=0.05)
        verdict = edge_gate(0.400, 0.380, SMALL, thresholds)

        assert verdict.passed


class TestFailing:
    def test_losing_more_than_the_allowance_fails(self) -> None:
        verdict = edge_gate(fp32_map=0.400, int8_map=0.300, artifact_bytes=SMALL)

        assert not verdict.passed
        assert "25.0% of fp32 mAP" in verdict.reason

    def test_an_artifact_over_the_ceiling_fails(self) -> None:
        """Which catches a cycle that wrote the fp32 graph under the int8
        filename, rather than one that is a little large."""
        verdict = edge_gate(FP32, 0.395, artifact_bytes=30 * 1024 * 1024)

        assert not verdict.passed
        assert "MB ceiling" in verdict.reason

    def test_both_failures_are_reported_together(self) -> None:
        """The next attempt costs a training run, so a verdict says everything
        that is wrong with this one."""
        verdict = edge_gate(0.400, 0.100, artifact_bytes=30 * 1024 * 1024)

        assert not verdict.passed
        assert "MB ceiling" in verdict.reason
        assert "of fp32 mAP" in verdict.reason

    def test_an_fp32_score_of_zero_is_named_as_the_quality_gate_s_finding(self) -> None:
        """There is no relative loss against nothing, and reporting quantization
        for a model that detected nothing sends the next cycle to the wrong
        place."""
        verdict = edge_gate(fp32_map=0.0, int8_map=0.0, artifact_bytes=SMALL)

        assert not verdict.passed
        assert "quality gate's finding" in verdict.reason

    def test_an_int8_model_that_improved_is_not_a_failure(self) -> None:
        """Quantization occasionally scores a hair above fp32 on a fixed eval
        set. That is noise, not a regression, and a gate reading it as a negative
        loss must not reject it."""
        assert edge_gate(0.400, 0.405, SMALL).passed


class TestWhatItReadsFrom:
    @pytest.mark.parametrize(
        ("fp32", "int8", "size"),
        [(FP32, 0.395, SMALL), (FP32, 0.100, SMALL), (FP32, 0.395, 30 * 1024 * 1024)],
        ids=["passing", "failing on accuracy", "failing on size"],
    )
    def test_no_verdict_needs_a_device(self, fp32: float, int8: float, size: int) -> None:
        """Pass or fail, the inputs are two floats and an integer. Speed is
        reported off the fleet rather than gated, so nothing here blocks on
        silicon and the gate is complete before one exists."""
        verdict = edge_gate(fp32, int8, size)

        assert verdict.gate == Gate.EDGE
        assert verdict.reason


class TestTheThresholds:
    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_an_allowance_outside_zero_to_one_is_refused(self, value: float) -> None:
        """Zero refuses every export and one admits a model that detects
        nothing, so neither is a threshold."""
        with pytest.raises(ValueError, match="not a fraction"):
            Thresholds(max_quantization_loss=value)

    def test_a_ceiling_of_zero_bytes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ships nothing"):
            Thresholds(max_artifact_bytes=0)

    def test_the_default_allowance_is_looser_than_the_design_s_two_percent(self) -> None:
        """Deliberately, and recorded here so a later tightening is a decision
        rather than a drift back. See `Thresholds`: a broken export is a 30% loss
        and a working one is a few percent, so the gate is set to catch the
        first."""
        assert Thresholds().max_quantization_loss > 0.02

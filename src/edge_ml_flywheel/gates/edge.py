"""The edge gate: will the artifact that ships still do the job?

Two thresholds, both properties of a file and a number: the int8 artifact is
small enough to deliver, and it kept enough of the fp32 model to be worth
delivering. Neither needs a device, so the gate is complete here and nothing
about it waits on silicon.

**Speed is reported, not gated** (design section 4.3). p95 latency and cold start
are read off the fleet's own telemetry once there is a fleet, and they belong to
the charts beside every other thing the devices say. Gating on them would mean a
promotion decision that cannot be made until a model has already been deployed,
which is the wrong way round.

**Measuring accuracy in the cloud is the whole point of doing it here.**
Post-training quantization of a small detector is where accuracy goes missing,
and finding that on a device means finding it after a rollout. Found here, the
fix is a training-side change to how the model is quantized, made while the cycle
that produced it is still the cycle in front of you.

**The accuracy check is relative, not absolute.** A challenger that improved and
a challenger that regressed both quantize about as well, so the question is what
the conversion cost *this* model rather than where it landed.
`max_quantization_loss` is the fraction of its own fp32 score an artifact may
lose.
"""

from math import isclose

from edge_ml_flywheel.conventions import GateResult
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds

_MEGABYTE = 1024 * 1024


def _size(artifact_bytes: int, thresholds: Thresholds) -> str | None:
    """The artifact fits the device's disk and its download.

    Not a threshold the export is near, which is what it is for: a quantized
    YOLO11n is a few megabytes against a 25 MB ceiling, so this fires when a
    cycle shipped the fp32 graph under the int8 filename, not when an artifact
    is slightly large.
    """
    if artifact_bytes <= thresholds.max_artifact_bytes:
        return None
    return (
        f"the int8 artifact is {artifact_bytes / _MEGABYTE:.1f} MB, over the "
        f"{thresholds.max_artifact_bytes / _MEGABYTE:.0f} MB ceiling"
    )


def _accuracy(fp32: float, int8: float, thresholds: Thresholds) -> str | None:
    """Quantization did not cost more of the model than it is allowed to.

    An fp32 score of zero is its own failure and belongs to the quality gate, not
    here: there is no relative loss to compute against nothing, and reporting a
    quantization failure for a model that never detected anything would send the
    next cycle looking in the wrong place.
    """
    if fp32 <= 0.0:
        return (
            f"fp32 mAP is {fp32:.4f}, so there is no baseline to measure quantization against. "
            f"A model that scores nothing is the quality gate's finding, not this one's"
        )
    loss = (fp32 - int8) / fp32
    # `isclose` alongside the comparison, because the boundary is a declared
    # threshold and binary floating point does not land on it: a model that lost
    # exactly 5% of 0.4 computes as 0.05000000000000002, and a gate that rejected
    # it would be rejecting an artifact for the representation of a number rather
    # than for the number.
    if loss <= thresholds.max_quantization_loss or isclose(
        loss, thresholds.max_quantization_loss, rel_tol=1e-9
    ):
        return None
    return (
        f"quantization cost {loss:.1%} of fp32 mAP ({fp32:.4f} to {int8:.4f}), over the "
        f"{thresholds.max_quantization_loss:.0%} allowance"
    )


def edge_gate(
    fp32_map: float,
    int8_map: float,
    artifact_bytes: int,
    thresholds: Thresholds = DEFAULT,
) -> GateResult:
    """One verdict over both thresholds.

    Both scores are the deployed seed's mAP over the whole `eval` cohort, from
    the same ground truth and the same matching -- the comparison is between two
    builds of one model, so anything that differed between the two passes would
    be measured as quantization loss.

    Every check runs even after one has failed, for `data_gate`'s reason: the
    next attempt costs a training run, so a verdict should say everything that is
    wrong with this one.
    """
    failures = [
        failure
        for failure in (
            _size(artifact_bytes, thresholds),
            _accuracy(fp32_map, int8_map, thresholds),
        )
        if failure is not None
    ]

    if failures:
        return GateResult(gate=Gate.EDGE, passed=False, reason="; ".join(failures))

    retained = int8_map / fp32_map if fp32_map > 0 else 0.0
    return GateResult(
        gate=Gate.EDGE,
        passed=True,
        reason=(
            f"int8 artifact {artifact_bytes / _MEGABYTE:.1f} MB, mAP {int8_map:.4f} against fp32 "
            f"{fp32_map:.4f} ({retained:.1%} retained)"
        ),
    )

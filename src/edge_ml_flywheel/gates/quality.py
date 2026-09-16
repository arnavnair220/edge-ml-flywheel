"""The quality gate: is the challenger actually better?

Three checks, over numbers `evaluation` has already computed. This module does no
scoring and no resampling -- it reads a `PairedDelta` and a per-class AP mapping
and applies the promotion rule to them. The split is what makes the rule
testable: a gate that also scored would need a model to test, and this needs two
floats.

**Both statistical conditions are required, not either.** The mean paired delta
must clear +0.005 *and* the resampling band must clear zero (design section 4.2).
The design predicts the two are close to redundant and says to confirm it against
real numbers, which is only possible if both are evaluated and both are reported.
That arithmetic was worked out over a seed spread, and a cycle now trains one
seed: the band resamples images and not seeds, so the floor carries more of the
decision than the prediction assumed. Which is a reason to check it, not to drop
either condition.

**The band is the condition that makes this more than a demo.** Most pipelines
promote on a raw metric bump sitting inside the noise, and most that do draw
error bars measure only eval-set noise -- typically the smaller of the two
sources. Requiring the interval to clear zero means sometimes correctly refusing
to promote, and a chart of honest rejections is the evidence the project is for.

**Slices cast no vote.** Per-slice scores are computed every cycle and charted,
never gated (design section 4.4). Acquisition is condition-blind, so a cycle
makes no per-condition bet for a per-condition gate to settle. The one per-class
condition here is not a slice test: it asks whether the model produces output at
all, which is a different question from whether a slice moved.
"""

from collections.abc import Mapping
from typing import Final

from edge_ml_flywheel.conventions import ClassSet, GateResult
from edge_ml_flywheel.evaluation.bootstrap import PairedDelta
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds

_REPORTED: Final = 5


def _uplift(delta: PairedDelta, thresholds: Thresholds) -> str | None:
    """The challenger is better by enough to be worth shipping."""
    if delta.observed >= thresholds.min_mean_delta:
        return None
    return (
        f"mean paired delta {delta.observed:+.4f} is under the promotion threshold of "
        f"{thresholds.min_mean_delta:+.4f}"
    )


def _band(delta: PairedDelta) -> str | None:
    """The improvement is bigger than resampling the eval set could explain.

    Reads `lower` alone, which makes the effective test one-sided at 97.5% --
    conservative in the direction a promotion decision should be conservative in.
    """
    if delta.improved:
        return None
    return (
        f"the {delta.confidence:.0%} band over {delta.resamples} resamples runs from "
        f"{delta.lower:+.4f} to {delta.upper:+.4f} and does not clear zero, so the delta is "
        f"inside eval-set noise"
    )


def _collapse(per_class: Mapping[str, float], classes: ClassSet) -> str | None:
    """No class scored zero, whatever the overall delta says.

    A hard failure, because it asserts that the model produces output at all
    rather than asking whether a slice moved more than its noise -- which is why
    it gates here instead of joining the regression report.

    A class missing from the mapping is also a failure, and a distinct one.
    `per_class_average_precision` omits a class with no ground truth over the
    images scored, so absence means the frozen 5,000-image eval cohort contains
    no box of that class: the check could not run, and a check that could not run
    is not a check that passed. Folding the two together would let a class
    disappear from the metric entirely and read as healthy.
    """
    collapsed = sorted(name for name in classes.names if per_class.get(name, 0.0) <= 0.0)
    unmeasured = sorted(name for name in classes.names if name not in per_class)
    if not collapsed and not unmeasured:
        return None

    reasons = []
    # Reported separately even though `unmeasured` is a subset of `collapsed` by
    # the `.get` default above: "the model detects no cars" and "eval contains no
    # cars" are two different bugs in two different components.
    detected = [name for name in collapsed if name not in unmeasured]
    if detected:
        reasons.append(f"{len(detected)} class(es) scored zero AP: {detected[:_REPORTED]}")
    if unmeasured:
        reasons.append(
            f"{len(unmeasured)} class(es) have no ground truth in eval, so the collapse check "
            f"could not run on them: {unmeasured[:_REPORTED]}"
        )
    return "; ".join(reasons)


def quality_gate(
    delta: PairedDelta,
    per_class: Mapping[str, float],
    classes: ClassSet,
    thresholds: Thresholds = DEFAULT,
) -> GateResult:
    """One verdict, naming every condition the challenger failed.

    `per_class` is the *challenger's* per-class AP over the whole eval cohort --
    the collapse check is about the model being promoted, not about the delta.

    Every check runs even after one has failed, for `data_gate`'s reason: the
    next attempt costs a training run, so a verdict should say everything that is
    wrong with this one.
    """
    failures = [
        failure
        for failure in (
            _uplift(delta, thresholds),
            _band(delta),
            _collapse(per_class, classes),
        )
        if failure is not None
    ]

    if failures:
        return GateResult(gate=Gate.QUALITY, passed=False, reason="; ".join(failures))

    return GateResult(
        gate=Gate.QUALITY,
        passed=True,
        reason=(
            f"mean paired delta {delta.observed:+.4f}, {delta.confidence:.0%} band "
            f"[{delta.lower:+.4f}, {delta.upper:+.4f}] over {delta.resamples} resamples, "
            f"all {len(classes.names)} classes detecting"
        ),
    )

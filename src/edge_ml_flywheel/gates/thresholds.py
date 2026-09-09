"""What the gates are called, and the numbers they apply.

A threshold written into a predicate body is a promotion rule that exists only as
code. The project's claim is that models are gated against *fixed, pre-declared*
thresholds (design section 4), and that is only checkable if the numbers are one
object -- readable before a cycle runs, and recordable in the gate report beside
the verdict they produced.

**Defaults, not a config file.** These are design parameters with one setting for
the life of the project; a run that changes them is measuring something else, and
that is a different run, which `RunRegistration` already expresses. So a frozen
dataclass carrying the design's values, which a test can tighten to drive a
predicate over its boundary and nothing else ever constructs. A loader, a YAML
file and a per-run override would be three moving parts serving a substitution
nobody makes.

**`CONFIDENCE` is deliberately not here.** The level at which a delta counts as
real is `bootstrap`'s constant rather than an argument, for the reason that
module gives: a caller free to lower it is a caller who can shop for a level that
promotes. Putting it in a dataclass a caller constructs hands back that freedom.

**The label budget is not here either.** It is per run and already on the run
registration, which is the write-once item that makes it fixed for the run.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Gate(StrEnum):
    """The four checks a cycle runs, in order.

    All four named though only two are implemented, for the reason `Selector`
    names its three: a name reserved in advance is one the report format and the
    state machine can be written against before the thing behind it exists.

    `EDGE` and `CANARY` have no predicate here, and the omission is deliberate
    rather than pending. Their inputs are p95 latency on an ARM64 task and two
    replay hours of device telemetry, and nothing in the project produces either
    yet. A predicate written now would be a pure function over a measurement
    shape invented to suit it, tested against that same invention, and rewritten
    when the device agent reports what it can actually measure. They land with
    their producers: `CANARY` in phase 6, beside shadow mode and the staged
    rollout, and `EDGE` in phase 7, where the int8 ONNX export first exists.
    """

    DATA = "data"
    QUALITY = "quality"
    EDGE = "edge"
    CANARY = "canary"


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Every number a gate compares against. Design section 4.

    `min_new_images` is a floor on the batch, not on the budget. A cycle that
    bought 40 labels has not failed to be affordable -- it has failed to buy
    enough for the comparison downstream to mean anything.

    `min_instances_per_class` is counted over the boxes bought *this cycle*, so
    it asks whether the batch teaches every class something. Coverage of the
    cumulative training set is not in question: the bootstrap is 8,000 random
    images.

    `min_mean_delta` is the absolute mAP improvement a challenger must show
    before its confidence band is consulted. The two conditions are close to
    redundant at five seeds -- design section 4.2 works out that the band alone
    implies a delta near +0.005 -- and both are kept because that redundancy is
    something the design says to confirm against real numbers rather than assume.
    """

    min_new_images: int = 250
    min_instances_per_class: int = 10
    min_mean_delta: float = 0.005

    def __post_init__(self) -> None:
        # Each of these admits everything rather than being obviously broken: a
        # floor of zero images passes a cycle that bought nothing, and a delta
        # floor of zero promotes on noise the band happens not to have caught.
        if self.min_new_images < 1:
            raise ValueError(f"a batch floor of {self.min_new_images} images is not a floor")
        if self.min_instances_per_class < 1:
            raise ValueError(
                f"a class-instance floor of {self.min_instances_per_class} is not a floor"
            )
        if self.min_mean_delta <= 0.0:
            raise ValueError(
                f"a promotion threshold of {self.min_mean_delta} promotes a challenger no better "
                f"than the champion"
            )


# The design's values, and what every predicate defaults to. A shared instance
# because `Thresholds` is frozen, so one object serves every signature rather
# than each constructing its own -- the arrangement `metrics.OVERALL` uses.
DEFAULT: Final = Thresholds()

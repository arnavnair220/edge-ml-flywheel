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

    `CANARY` is the one whose input comes from outside the cloud, and it is the
    reason the four are not run together: the first three are computed from a
    cycle's own artifacts, and this one cannot be asked until the artifact has
    been deployed and a device has replayed with it. So it is read after
    promotion rather than before, and what it decides is whether the rollout
    continues or rolls back -- not whether the model is registered.

    Its predicate landed with its producer, which is what the reservation was
    for: a shape invented before the devices reported anything would have been
    tested against the invention. What they report is `ReplayReport`, and three
    of design section 4.5's conditions are checked against it. The rest --
    memory flatness over two replay hours, and the confidence-distribution
    distance -- wait on a second device and a longer run, which is the fleet
    growing rather than the gate changing.
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
    before its confidence band is consulted. Design section 4.2 works the two
    conditions out as close to redundant -- the band alone implies a delta near
    +0.005 -- over a seed spread a cycle no longer trains, so at one seed this
    floor is the condition doing more of the work. It is left at the design's
    number rather than re-derived from a guess: the design says to confirm it
    against real numbers, and there are none yet.

    `max_quantization_loss` is the edge gate's, and it is deliberately looser
    than design section 4.3's 2%. That number was written before anything had
    been quantized. What the project needs from this gate is that a broken export
    cannot ship, and a broken export is not a 3% model -- it is a 30% one, or a
    graph that detects nothing. A threshold tight enough to reject a slightly
    lossy but working artifact would stop the loop over a number the fleet would
    never notice, and what quantization actually cost is in the verdict's reason
    either way. Tighten it once several cycles have said what the real spread is.

    `max_artifact_bytes` is design section 4.3's, unchanged. It is not a number
    the export is near -- a quantized YOLO11n is a few megabytes against a 25 MB
    ceiling -- which is the point: it catches an export that silently wrote the
    fp32 graph, not one that is a little large.

    `max_throughput_drop` is the canary's, and it is design section 4.5's 10%
    unchanged. It is the one number that gate compares rather than checks: the
    other two conditions are a digest matching and a component running, which are
    facts about whether the deployment worked at all. This one asks whether the
    model that arrived is as fast on the device as the one it replaces, and a
    challenger that lost a tenth of the champion's frame rate has changed
    something about the graph that the cloud passes never saw.
    """

    min_new_images: int = 250
    min_instances_per_class: int = 10
    min_mean_delta: float = 0.005
    max_quantization_loss: float = 0.05
    max_artifact_bytes: int = 25 * 1024 * 1024
    max_throughput_drop: float = 0.10

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
        if not 0.0 < self.max_quantization_loss < 1.0:
            raise ValueError(
                f"a quantization allowance of {self.max_quantization_loss} is not a fraction "
                f"between refusing every export and admitting one that detects nothing"
            )
        if self.max_artifact_bytes < 1:
            raise ValueError(
                f"an artifact ceiling of {self.max_artifact_bytes} bytes ships nothing"
            )
        if self.min_mean_delta <= 0.0:
            raise ValueError(
                f"a promotion threshold of {self.min_mean_delta} promotes a challenger no better "
                f"than the champion"
            )
        if not 0.0 < self.max_throughput_drop < 1.0:
            raise ValueError(
                f"a throughput allowance of {self.max_throughput_drop} is not a fraction between "
                f"refusing every deployment and admitting one that has stopped"
            )


# The design's values, and what every predicate defaults to. A shared instance
# because `Thresholds` is frozen, so one object serves every signature rather
# than each constructing its own -- the arrangement `metrics.OVERALL` uses.
DEFAULT: Final = Thresholds()

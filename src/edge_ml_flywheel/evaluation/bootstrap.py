"""The paired delta between two models, and the band around it.

The question the quality gate asks is not "is the challenger's mAP higher" but
"is it higher by more than resampling the eval set could explain". This answers
it by resampling the 5,000 eval images with replacement and reading the delta off
each draw.

Three properties make the answer meaningful, and all three are structural rather
than conventions a caller is trusted to follow:

**One resample, both models.** Every draw scores champion and challenger on the
same images, and the statistic is the difference. Bootstrapping each model
independently and subtracting the intervals would measure two things that vary
together as though they varied separately, which widens the band by most of the
eval-set noise the pairing exists to cancel (design section 4.2).

**Matched seeds.** A cycle trains five, so a delta is the mean over five
same-seed differences rather than a difference of the best or the first. Seeds
present on one side and not the other are refused: dropping them silently would
compare a five-seed mean against a three-seed one.

**The same draws every cycle.** `RESAMPLE_SEED` is fixed and the eval cohort is
frozen, so cycle eight's band is computed over the same 1,000 image lists as
cycle one's. A per-run seed would make two cycles differ by their draws as well
as by their models.

The cost is real and is the reason `match` exists. A thousand draws over ten
caches is ten thousand accumulations, a few minutes; the same comparison by
re-scoring would be ten thousand full inference passes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from edge_ml_flywheel.conventions import Seed
from edge_ml_flywheel.evaluation.match import MatchCache
from edge_ml_flywheel.evaluation.metrics import OVERALL, Scope, average_precision, image_rows

# A thousand draws puts the percentile endpoints at the 25th and 976th ordered
# delta, so each is an average of neighbours rather than the single most extreme
# value. More draws narrow the Monte Carlo error on the endpoint, not the band
# itself, and the band is already wider than that error at this count.
RESAMPLES: Final = 1000

# Fixed for the life of the project, and arbitrary -- the date the eval cohort
# sizes were settled, recorded the way `PartitionSpec.seed` is so that its
# arbitrariness is on the record and nobody improves it. It is a constant rather
# than an argument for that entry's other reason: a seed a hand can type is a
# seed a hand can mistype into a run that is valid, different, and
# indistinguishable from the intended one.
RESAMPLE_SEED: Final = 20260819

# Two-sided, so `lower` is the 2.5th percentile. The gate reads `lower` alone,
# which makes its effective test one-sided at 97.5% -- conservative in the
# direction a promotion decision should be conservative in.
CONFIDENCE: Final = 0.95


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """A challenger-minus-champion difference in AP, with its resampling band.

    `observed` is the difference on the real eval set, not the mean of the
    resampled differences. Those two differ by the bootstrap's bias, and the
    reported number should be the measurement rather than an estimate of it.

    Every field is recorded in the gate report, `resamples` and `confidence`
    included: a band means nothing without the parameters that produced it, and
    they are the two things a later reader cannot recover from the number.
    """

    observed: float
    lower: float
    upper: float
    resamples: int
    confidence: float

    @property
    def improved(self) -> bool:
        """The whole band is above zero.

        The gate's own threshold is a design parameter and lives with the gate;
        this is the weaker question of whether the delta clears resampling noise
        at all, which is what the A/A test expects a healthy gate to answer no to.
        """
        return self.lower > 0.0


def resamples(
    rows: NDArray[np.int64], count: int = RESAMPLES, seed: int = RESAMPLE_SEED
) -> list[NDArray[np.int64]]:
    """`count` draws of `len(rows)` rows, with replacement, from a fixed seed.

    Materialized as a list rather than yielded, because every model and every
    seed must see the same draws in the same order and a generator can only be
    walked once. At 5,000 images and 1,000 draws this is 40 MB, which is the
    price of the pairing.
    """
    if count < 1:
        raise ValueError(f"a bootstrap of {count} resamples is not a bootstrap")
    if rows.size == 0:
        raise ValueError("there is nothing to resample")
    rng = np.random.default_rng(seed)
    return [rng.choice(rows, size=rows.size, replace=True) for _ in range(count)]


def _mean_over_seeds(
    caches: Mapping[Seed, MatchCache], rows: NDArray[np.int64], scope: Scope
) -> float:
    return float(np.mean([average_precision(cache, rows, scope) for cache in caches.values()]))


def paired_delta(
    champion: Mapping[Seed, MatchCache],
    challenger: Mapping[Seed, MatchCache],
    scope: Scope = OVERALL,
    *,
    rows: NDArray[np.int64] | None = None,
    count: int = RESAMPLES,
) -> PairedDelta:
    """Challenger minus champion, resampled.

    `rows` defaults to the whole eval cohort, which is what the quality gate
    reads. Passing a slice's rows bootstraps that slice instead -- available, and
    not what the gate uses: a slice is a fraction of 5,000 images and its band is
    correspondingly wide (design section 4.4 reports per-slice scores rather than
    gating on them).

    `CONFIDENCE` is deliberately not an argument. The level at which a delta
    counts as real is part of the promotion rule, and a caller free to lower it is
    a caller who can shop for a level that promotes.

    The percentile method, which is the plain reading of the resampled
    distribution: the interval is the middle `CONFIDENCE` of the deltas. It does
    not correct for skew the way BCa would, and that is a deliberate stopping
    point -- the correction matters for small samples, and a 5,000-image paired
    delta is not one.
    """
    _check_comparable(champion, challenger)

    index = next(iter(champion.values())).index
    if rows is None:
        rows = image_rows(range(len(index)))

    def delta(drawn: NDArray[np.int64]) -> float:
        return _mean_over_seeds(challenger, drawn, scope) - _mean_over_seeds(champion, drawn, scope)

    observed = delta(rows)
    # Drawn outside the `try` below, so that a refusal about `count` or about
    # `rows` is reported as itself rather than rewritten as a statement about the
    # slice's ground truth.
    samples = resamples(rows, count)

    drawn: list[float] = []
    for sample in samples:
        try:
            drawn.append(delta(sample))
        except ValueError as absent:
            # `average_precision` refuses a slice with no ground truth. Over the
            # whole cohort that cannot happen; over a narrow slice a draw can miss
            # every box of every class, which says the slice is too small to
            # bootstrap rather than that this one draw was unlucky.
            raise ValueError(
                f"a resample of these {rows.size} images has no ground truth to score, so "
                f"this slice is too small to put a band on: {absent}"
            ) from absent

    tail = (1.0 - CONFIDENCE) / 2.0
    lower, upper = np.percentile(drawn, [100.0 * tail, 100.0 * (1.0 - tail)])
    return PairedDelta(
        observed=observed,
        lower=float(lower),
        upper=float(upper),
        resamples=count,
        confidence=CONFIDENCE,
    )


def _check_comparable(
    champion: Mapping[Seed, MatchCache], challenger: Mapping[Seed, MatchCache]
) -> None:
    """Refuse a comparison whose two sides were not measured the same way.

    Every one of these is a mismatch that produces a number rather than an error:
    a delta against a different eval cohort, under a different class set, or over
    a different set of seeds is arithmetic that succeeds and answers a question
    nobody asked.
    """
    if not champion or not challenger:
        raise ValueError("a paired delta needs a cache on both sides")
    if set(champion) != set(challenger):
        raise ValueError(
            f"seeds do not match: champion has {sorted(champion)} and challenger has "
            f"{sorted(challenger)}"
        )

    caches = list(champion.values()) + list(challenger.values())
    first = caches[0]
    if any(cache.index != first.index for cache in caches):
        raise ValueError("the caches were not scored against the same eval cohort")
    if any(cache.class_set_version != first.class_set_version for cache in caches):
        raise ValueError("the caches were not scored under the same class set")

"""Which images a cycle scores, per cohort, and which of the pool the fleet sees.

Two sets, and only one of them moves. `eval` is the frozen 5,000 and is the same
list in cycle eight as in cycle one, which is the property the whole comparison
rests on. The pool is the 62,000 minus everything the run has bought so far, so
it shrinks by a budget every cycle -- and the difference is why the manifests are
written under the write-once cycle prefix rather than derived on demand: "which
61,000 images was cycle two's ranking over" is not recoverable afterwards without
replaying every purchase in order.

**The pool is sampled, not scored whole.** `to_score` gives what is left of it
and `to_sample` draws `POOL_SAMPLE` frames out of that, because the pass over
them happens on one ARM device rather than on a GPU. The sample is the cycle's
whole ranking universe: an image outside the draw is not ranked low, it simply
waits for a later cycle's draw.

**Pure, and the assignments come from the partition.** `oracle.cohorts.Cohorts`
is the same index the purchase gate reads, loaded from `assignments/` rather than
copied per run, because a cohort is a fact about a partition and two runs over
one `partition_version` are answered identically. Nothing here opens a socket;
`launch` does the downloading.

**Scoring the pool is not buying it.** These are image IDs going to an inference
job, so no label is read and no budget is touched. The wall that matters here is
the other one: `eval` images reach a model that must never have trained on them,
which is a statement about what trained, not about what is scored -- and it is
`ModelManifest.cohorts_trained_on` that refuses it.
"""

import hashlib
import logging
from collections.abc import Iterable, Sequence
from random import Random
from typing import Final

from edge_ml_flywheel.conventions import (
    CYCLE_DIGITS,
    SCORED_COHORTS,
    Cohort,
    Cycle,
    ImageId,
    RunId,
)
from edge_ml_flywheel.oracle.cohorts import Cohorts

log = logging.getLogger(__name__)

# How many offending IDs a refusal lists before summarizing, matching
# `oracle.cohorts._REPORTED`.
_REPORTED: Final = 5


def to_score(cohorts: Cohorts, cohort: Cohort, purchased: Iterable[ImageId]) -> tuple[ImageId, ...]:
    """One cohort's images for this cycle, sorted.

    `purchased` subtracts only from the pool, and passing it for `eval` is not an
    error to guard against -- it is arithmetic that cannot do anything, since the
    oracle refuses to sell an eval image and the two sets are disjoint by the
    partition. The subtraction is written once rather than behind a branch per
    cohort for exactly that reason.

    Sorted on the way out so the manifest, the detections and a re-run all follow
    one ordering.
    """
    if cohort not in SCORED_COHORTS:
        listed = ", ".join(sorted(member.value for member in SCORED_COHORTS))
        raise ValueError(f"{cohort.value} is not a scored cohort ({listed} are)")

    drawn = cohorts.in_cohort(cohort)
    if not drawn:
        raise ValueError(
            f"partition v{cohorts.partition_version} assigned no image to {cohort.value}"
        )

    spent = frozenset(purchased)
    remaining = tuple(sorted(drawn - spent))
    if not remaining:
        raise ValueError(
            f"every one of the {len(drawn):,} images in {cohort.value} has been bought, so there "
            f"is nothing left to score"
        )

    log.info(
        "%s: scoring %d of %d images, %d already bought",
        cohort.value,
        len(remaining),
        len(drawn),
        len(drawn) - len(remaining),
    )
    return remaining


def to_sample(
    remaining: Sequence[ImageId], run_id: RunId, cycle: Cycle, frames: int
) -> tuple[ImageId, ...]:
    """The pool frames this cycle puts in front of the fleet.

    **Random, not the top of anything.** There is no ranking to take the top of:
    this draw is what produces one, because the device's pass over it is the only
    pass anything makes over the pool. A biased draw would be a selector whose
    input was chosen by a different rule than the one it applies.

    **Out of what the run has not bought**, which is what `remaining` already is.
    A bought image has labels and is in the training set, so scoring it would
    rank a frame no cycle can sell.

    **Seeded by the run and the cycle**, so a redeploy of one cycle scores the
    identical frames. Anything else would make two attempts at one deployment
    incomparable, and the second attempt is usually the one made after a
    rollback.

    Sorted on the way out, like `to_score`, so the manifest, the detections and a
    re-run all follow one ordering.
    """
    if frames < 1:
        raise ValueError(f"a sample of {frames} frames is not a sample")
    if len(remaining) < frames:
        raise ValueError(
            f"cycle {cycle} has {len(remaining):,} unbought pool images, fewer than the "
            f"{frames:,} the fleet is asked to score. The pool is spent, which ends the run "
            f"rather than the cycle"
        )

    drawn = tuple(sorted(Random(_draw_seed(run_id, cycle)).sample(list(remaining), frames)))
    log.info("sampled %d of %d unbought pool images for cycle %d", frames, len(remaining), cycle)
    return drawn


def _draw_seed(run_id: RunId, cycle: Cycle) -> int:
    """A draw seed that is a function of the run and the cycle and nothing else.

    A digest rather than `hash()`, which is salted per process: the "same frames
    every time" property would otherwise hold within one invocation and nowhere
    else, which is exactly the case a redeploy after a rollback is not.

    The cycle is padded into the string for `purchase_event`'s reason -- it is
    text here, so cycle 10 and cycle 1 followed by a zero must not be one seed.
    """
    stamp = f"{run_id}-c{cycle:0{CYCLE_DIGITS}d}"
    return int(hashlib.sha256(stamp.encode()).hexdigest()[:16], 16)


def check_purchases(cohorts: Cohorts, purchased: Iterable[ImageId]) -> None:
    """Refuse a ledger naming an image this partition never put in the pool.

    Nothing downstream would notice on its own. A bought image outside the pool
    subtracts from nothing, so the pool to score comes out the right size and the
    cycle proceeds -- while the run's training set holds a frame the partition
    says is `eval`. That is the leakage the gates exist to catch at the far end,
    and catching it here costs a set difference over 62,000 strings.

    Separate from `to_score` because it is a statement about the run rather than
    about one cohort, and running it once per cycle rather than once per cohort
    is what keeps it from reporting the same fault twice.
    """
    wrong = {
        image_id: cohorts.cohort_of(image_id)
        for image_id in purchased
        if cohorts.cohort_of(image_id) is not Cohort.POOL
    }
    if not wrong:
        return

    named = ", ".join(
        f"{image_id} ({cohort.value if cohort is not None else 'unassigned'})"
        for image_id, cohort in sorted(wrong.items())[:_REPORTED]
    )
    raise ValueError(
        f"{len(wrong)} image(s) this run has bought are not in the pool of partition "
        f"v{cohorts.partition_version}: {named}"
    )

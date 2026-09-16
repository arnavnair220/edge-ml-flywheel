"""Which images a cycle scores, per cohort.

Two sets, and only one of them moves. `eval` is the frozen 5,000 and is the same
list in cycle eight as in cycle one, which is the property the whole comparison
rests on. The pool is the 62,000 minus everything the run has bought so far, so
it shrinks by a budget every cycle -- and the difference is why the manifests are
written under the write-once cycle prefix rather than derived on demand: "which
61,000 images was cycle two's ranking over" is not recoverable afterwards without
replaying every purchase in order.

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

import logging
from collections.abc import Iterable
from typing import Final

from edge_ml_flywheel.conventions import SCORED_COHORTS, Cohort, ImageId
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

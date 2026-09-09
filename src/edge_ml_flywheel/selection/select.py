"""Turning a scored pool into the batch a cycle buys.

The three rules `Selector` names, behind one signature, so the deferred
label-efficiency arm (design section 8) changes a field on a run registration
rather than adding a second code path. All three take the same arguments and two
of them ignore one: `random` never looks at a score, and the two ranked rules
never look at the seed. That is the cost of one interface, and it is cheaper than
three call sites that each know which rule they are talking to.

**The tie-break is not cosmetic.** Every image the champion detected nothing in
scores exactly `BLIND_SPOT`, so ties are guaranteed rather than incidental, and
they land at the batch boundary as soon as there are more blind spots than budget.
A sort on score alone leaves the order among them up to the input's iteration
order, which means two attempts at the same cycle propose different sets. The
oracle identifies a purchase by a digest over the sorted image IDs
(`conventions.batch_digest`), so a different set is a different purchase: the
retry does not replay, it buys a second batch and debits the ledger twice. Adding
the image ID as the second sort key costs nothing and closes that.

**Nothing is filtered.** The ranking is bought as it stands, redundancy and blur
included, because every available correction is a threshold no cycle has yet
reported a value for (design section 2). What guards against it is the record
`selection.mix` writes, not a rule applied here.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Final

import numpy as np

from edge_ml_flywheel.conventions import ImageId, Selector

# One rule: the pool to draw from, what each image scored, how many to take, and
# a seed. Returns the batch in ranked order -- the oracle sorts for its digest, so
# what survives here is a record of the order the rule proposed.
type Rule = Callable[[Iterable[ImageId], Mapping[ImageId, float], int, int], tuple[ImageId, ...]]


def _ranked(
    pool: Iterable[ImageId],
    scores: Mapping[ImageId, float],
    budget: int,
    descending: bool,
) -> tuple[ImageId, ...]:
    """Both ranked rules, which differ only in direction.

    One implementation rather than two, so `certainty` cannot drift into being a
    different rule than the one `uncertainty` inverts -- the whole value of the
    control is that it is the same machinery pointed the other way.

    The image ID always sorts ascending, in both directions. It is a tie-break
    and not a second ranking, so flipping it with the score would make the two
    rules disagree about which of two equally-scored images to prefer, for no
    reason anybody could state.

    The completeness check on `scores` lives here rather than in `select`,
    because it is a precondition of ranking and not of selecting: `random` draws
    a valid batch from an empty mapping, and that is the property that makes it
    the smoke test for this path before a champion exists.
    """
    ordered = sorted(pool)
    unscored = [image_id for image_id in ordered if image_id not in scores]
    if unscored:
        raise ValueError(
            f"{len(unscored)} of {len(ordered)} pool images have no score, and an unscored image "
            f"drops out of the ranking rather than ranking last: {unscored[:5]}"
        )

    sign = -1.0 if descending else 1.0
    ordered.sort(key=lambda image_id: (sign * scores[image_id], image_id))
    return tuple(ordered[:budget])


def by_uncertainty(
    pool: Iterable[ImageId], scores: Mapping[ImageId, float], budget: int, seed: int
) -> tuple[ImageId, ...]:
    """The rule the loop runs by. `seed` is unused and part of the signature."""
    del seed
    return _ranked(pool, scores, budget, descending=True)


def by_certainty(
    pool: Iterable[ImageId], scores: Mapping[ImageId, float], budget: int, seed: int
) -> tuple[ImageId, ...]:
    """The control that buys what the champion is most sure of.

    Those frames carry the least new information, so a cycle run this way should
    gain close to nothing. One that gains as much as a real cycle says the
    ranking is not what is doing the work (design section 9).
    """
    del seed
    return _ranked(pool, scores, budget, descending=False)


def at_random(
    pool: Iterable[ImageId], scores: Mapping[ImageId, float], budget: int, seed: int
) -> tuple[ImageId, ...]:
    """The control arm, and the smoke test for this path before a champion exists.

    `scores` is unused, which is the property that makes it the smoke test: the
    ranking-to-purchase path can be exercised end to end with no inference at all.

    The pool is sorted before it is permuted. A seed only reproduces a draw if
    what it draws from is in a fixed order, and a set of image IDs arriving from
    a parquet read is not.
    """
    del scores
    ordered = sorted(pool)
    drawn = np.random.default_rng(seed).permutation(len(ordered))[:budget]
    return tuple(ordered[index] for index in drawn)


RULES: Final[Mapping[Selector, Rule]] = {
    Selector.UNCERTAINTY: by_uncertainty,
    Selector.CERTAINTY: by_certainty,
    Selector.RANDOM: at_random,
}


def select(
    selector: Selector,
    pool: Iterable[ImageId],
    scores: Mapping[ImageId, float],
    budget: int,
    seed: int,
) -> tuple[ImageId, ...]:
    """The batch this cycle proposes to buy, in the order the rule ranked it.

    The checks here are the ones true of every rule -- a budget has to buy
    something, and it cannot buy more than is left. Each describes a caller that
    has already gone wrong upstream and would otherwise produce a batch that
    looks ordinary. What `scores` must contain is a rule's own precondition and
    is checked by the rules that read it.
    """
    if budget < 1:
        raise ValueError(f"a budget of {budget} labels buys no batch")

    remaining = sorted(pool)
    if not remaining:
        raise ValueError("an empty pool has nothing to select from")
    if budget > len(remaining):
        raise ValueError(
            f"a budget of {budget} is more than the {len(remaining)} images left in the pool, so "
            f"this is not a selection -- it is the end of the run"
        )

    return RULES[selector](remaining, scores, budget, seed)

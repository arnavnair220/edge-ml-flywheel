"""Turning a scored pool into the batch a cycle buys.

One rule: rank the remaining pool by mean per-object uncertainty and take the
top of it. There is no selector to pass and no dispatch to route through,
because every run ranks the same way -- a run that bought by a different rule
would be a different project, not a different configuration of this one.

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

from collections.abc import Iterable, Mapping

from edge_ml_flywheel.conventions import ImageId


def by_uncertainty(
    pool: Iterable[ImageId], scores: Mapping[ImageId, float], budget: int
) -> tuple[ImageId, ...]:
    """The ranking itself: most uncertain first, image ID breaking ties.

    The image ID sorts ascending while the score sorts descending. It is a
    tie-break and not a second ranking, which is what keeps two attempts at one
    cycle proposing the same set.

    The completeness check on `scores` is a precondition of ranking: an image
    with no score drops out of the ordering silently rather than ranking last,
    so a partial scoring pass would buy a batch that looks ordinary and is drawn
    from a fraction of the pool.
    """
    ordered = sorted(pool)
    unscored = [image_id for image_id in ordered if image_id not in scores]
    if unscored:
        raise ValueError(
            f"{len(unscored)} of {len(ordered)} pool images have no score, and an unscored image "
            f"drops out of the ranking rather than ranking last: {unscored[:5]}"
        )

    ordered.sort(key=lambda image_id: (-scores[image_id], image_id))
    return tuple(ordered[:budget])


def select(
    pool: Iterable[ImageId],
    scores: Mapping[ImageId, float],
    budget: int,
) -> tuple[ImageId, ...]:
    """The batch this cycle proposes to buy, in ranked order.

    The checks here are about the call rather than the ranking -- a budget has
    to buy something, and it cannot buy more than is left. Each describes a
    caller that has already gone wrong upstream and would otherwise produce a
    batch that looks ordinary. What `scores` must contain is the ranking's own
    precondition and is checked there.
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

    return by_uncertainty(remaining, scores, budget)

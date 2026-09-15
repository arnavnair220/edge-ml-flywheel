"""Which images a cycle spends its budget on, and the record of what it bought.

Three steps, all pure: score the pool from the champion's detections, rank it and
take the top of the ranking, and write down what the batch was made of. The
output is a tuple of image IDs, which is exactly what `oracle` takes -- so
nothing in this package touches a label, a ledger or a model.

**Detections arrive, they are not produced here.** Running the champion over
62,000 images is the control plane's step; what this package consumes is the
boxes that came back. The split is what makes the ranking testable without a
trained detector, and it is also the reason selection sits inside the label wall
by construction rather than by policy: a score is a function of predictions
alone, so there is no path from here to ground truth to close.

**The ranking is bought unfiltered.** Both known failure modes of raw top-N --
a redundant top of the list, and high-uncertainty frames that are simply
unlabelable -- are left in place, because correcting either means picking a
threshold no cycle has reported a value for (design section 2). `mix` is what
turns the first occurrence into a number instead of a guess.

**One rule.** The pool is ranked by mean per-object uncertainty and the top of
that ranking is bought. There is nothing to select between and no rule to pass,
so `select` takes a pool, its scores and a budget and nothing else.
"""

from edge_ml_flywheel.selection.mix import Mix, SelectionReport, selection_report
from edge_ml_flywheel.selection.score import (
    BAND_HIGH,
    BAND_LOW,
    BLIND_SPOT,
    DECISIVE,
    Predictions,
    image_score,
    score_pool,
    uncertainty,
)
from edge_ml_flywheel.selection.select import by_uncertainty, select

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BLIND_SPOT",
    "DECISIVE",
    "Mix",
    "Predictions",
    "SelectionReport",
    "by_uncertainty",
    "image_score",
    "score_pool",
    "select",
    "selection_report",
    "uncertainty",
]

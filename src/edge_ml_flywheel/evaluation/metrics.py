"""AP over an arbitrary list of images, read off a `MatchCache`.

`COCOeval.accumulate()` reimplemented, with one difference that is the reason it
is reimplemented: it folds every image it was given, and this takes an image
list. That turns three questions into one call with a different argument -- the
overall metric is every row, a slice is the rows whose images carry a tag, and a
bootstrap resample is a draw with replacement (design section 4.2).

The arithmetic is `pycocotools`' exactly, down to details that look incidental
and are not:

- The sort is `argsort(-scores, kind="mergesort")`. Stable, so ties break by the
  order blocks were gathered in, which `MatchCache` fixes as image-row order.
- Precision is swept into a right-to-left running maximum before interpolation,
  which is the standard envelope.
- Precision is sampled at 101 recall thresholds, and a threshold above the
  highest recall achieved contributes a zero rather than being skipped.
- A category with no ground truth in the slice is dropped from the mean, not
  scored as zero. Averaging in a zero would report a model as worse on a slice
  that never contained the class.

**COCO's `-1` for an absent category is deliberately not reproduced at the top
level.** Inside, an absent category is excluded exactly as `accumulate()`
excludes it. But a slice where *no* category has any ground truth has no AP, and
returning `-1.0` for it puts a number into a gate comparison that reads as a
catastrophic regression rather than as a missing measurement. That case raises.

The `metrics.json` document at `conventions.eval_metrics_key` is not built here.
Naming the slices means joining image IDs against the manifest's tags, which is
the eval job's work; this module knows about images by row number and nothing
else.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from edge_ml_flywheel.conventions import MAX_DETS
from edge_ml_flywheel.evaluation.match import (
    IOU_THRESHOLDS,
    AreaRange,
    MatchCache,
)

# COCO's 101-point interpolation, 0.00 to 1.00 in steps of 0.01, spelled the way
# `pycocotools` spells it. Read-only for `IOU_THRESHOLDS`' reason.
RECALL_THRESHOLDS: Final[NDArray[np.float64]] = np.linspace(
    0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
)
RECALL_THRESHOLDS.setflags(write=False)


def image_rows(rows: Iterable[int]) -> NDArray[np.int64]:
    """A row list in the form the functions below take.

    Exists so a caller with a Python list -- a slice query's result, say -- has
    one obvious way to produce an empty one: `np.array([])` is float64 and fails
    as an index, which is a confusing error a long way from its cause.
    """
    return np.fromiter(rows, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Scope:
    """Which number is being asked for, as one value rather than three arguments.

    Area range, detection cap and IoU threshold travel together through every
    function here and into `bootstrap`, and a comparison is only a comparison if
    both sides used the same three. Bundling them means a caller passes one
    object to both sides instead of repeating a triple and getting one of them
    wrong on the second call.

    The defaults are the headline metric: every box, COCO's cap, and the mean
    over all ten IoU thresholds.
    """

    area: AreaRange = AreaRange.ALL
    max_dets: int = MAX_DETS
    iou: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_dets <= MAX_DETS:
            raise ValueError(
                f"max_dets must be between 1 and the cached {MAX_DETS}: {self.max_dets}"
            )
        _ = self.threshold  # refuse an unscored IoU here, before any work is done

    @property
    def threshold(self) -> int | None:
        """Which row of the cached arrays this IoU is, or `None` for all of them.

        Matched approximately and then refused loudly. `IOU_THRESHOLDS` comes out
        of `linspace`, so 0.5 and 0.75 are exact and 0.9 is not -- and a caller
        asking for a threshold nobody scored at should be told, not silently
        given the mean over ten.
        """
        if self.iou is None:
            return None
        matches = np.flatnonzero(np.isclose(IOU_THRESHOLDS, self.iou))
        if matches.size != 1:
            raise ValueError(
                f"nothing was scored at IoU {self.iou}. Scored at: {IOU_THRESHOLDS.tolist()}"
            )
        return int(matches[0])


# The headline metric, as a shared instance. `Scope` is frozen, so one object can
# be the default everywhere rather than each signature constructing its own.
OVERALL: Final = Scope()


def precision_curve(
    cache: MatchCache,
    rows: NDArray[np.int64],
    category_id: int,
    scope: Scope = OVERALL,
) -> NDArray[np.float64] | None:
    """Interpolated precision for one category, shaped `(iou thresholds, 101)`.

    `None` when the category has no ground truth over these rows -- the cell
    `accumulate()` leaves at `-1` and `summarize()` then drops. `None` rather than
    an array of `-1` so a caller cannot average it in by forgetting to filter.

    A category with ground truth and no detections is not that case: it scores
    zero, which is a real measurement and comes back as an array of zeros.
    """
    blocks = cache.blocks(category_id, scope.area, rows)
    n_truth = cache.truth_count(blocks)
    if n_truth == 0:
        return None

    detections = cache.detection_rows(blocks, scope.max_dets)
    # Descending by score, stable. The gather order is the tie-break, which is
    # what makes this equal `accumulate()` rather than merely close to it.
    detections = detections[np.argsort(-cache.scores[detections], kind="mergesort")]

    matched = cache.matched[:, detections]
    ignored = cache.ignored[:, detections]
    # An ignored detection is neither a hit nor a miss: it landed outside the area
    # range and this slice does not judge it.
    hits = np.cumsum(matched & ~ignored, axis=1, dtype=np.float64)
    misses = np.cumsum(~matched & ~ignored, axis=1, dtype=np.float64)

    recall = hits / n_truth
    # `np.spacing(1)` guards the leading entry where a first ignored detection
    # leaves both sums at zero. `pycocotools`' own term, kept so the two agree in
    # the last bit.
    precision = hits / (misses + hits + np.spacing(1))
    # The envelope: precision at recall r is the best precision at any recall of
    # r or more, which is a suffix maximum along the detection axis.
    envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]

    curve = np.zeros((len(IOU_THRESHOLDS), len(RECALL_THRESHOLDS)), dtype=np.float64)
    found = detections.size
    for threshold in range(len(IOU_THRESHOLDS)):
        # Recall is non-decreasing, so this is the first detection reaching each
        # recall threshold. One past the end means that recall was never reached,
        # and the zero already in `curve` stands.
        at = np.searchsorted(recall[threshold], RECALL_THRESHOLDS, side="left")
        reached = at < found
        curve[threshold, reached] = envelope[threshold, at[reached]]
    return curve


def per_class_average_precision(
    cache: MatchCache, rows: NDArray[np.int64], scope: Scope = OVERALL
) -> dict[str, float]:
    """AP per class name, omitting classes with no ground truth over these rows.

    Omitted rather than reported as zero or as `None`, for `precision_curve`'s
    reason: the regression report lists what was measured, and a class absent
    from a slice was not measured on it.
    """
    threshold = scope.threshold
    scores: dict[str, float] = {}
    for name in cache.classes.names:
        curve = precision_curve(cache, rows, cache.classes.category_id(name), scope)
        if curve is None:
            continue
        scores[name] = float(curve.mean() if threshold is None else curve[threshold].mean())
    return scores


def average_precision(cache: MatchCache, rows: NDArray[np.int64], scope: Scope = OVERALL) -> float:
    """The headline number: mean AP over the classes present in these rows.

    `Scope(iou=None)` is the COCO mean over all ten thresholds, which is what the
    gate reads. `Scope(iou=0.5)` is the number everyone quotes.

    Averaged over the pooled precision entries rather than over the per-class APs.
    The two agree here because every present class contributes the same number of
    entries, and this is the order `summarize()` does it in.
    """
    threshold = scope.threshold
    curves = [
        curve
        for name in cache.classes.names
        if (curve := precision_curve(cache, rows, cache.classes.category_id(name), scope))
        is not None
    ]
    if not curves:
        raise ValueError(
            f"no class in this class set has a {scope.area.value} ground-truth box over these "
            f"{rows.size} images, so there is no AP to report"
        )
    stacked = np.stack(curves)
    return float(stacked.mean() if threshold is None else stacked[:, threshold].mean())

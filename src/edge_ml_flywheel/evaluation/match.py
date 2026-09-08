"""One scoring pass over the eval cohort, kept in the form a resample reads.

`COCOeval.evaluate()` performs the greedy IoU assignment per image and
`COCOeval.accumulate()` folds the result into AP over every image it was given.
This module keeps the first and discards the second: the arrays between the two
steps are the only durable artifact, and every later question -- overall AP, a
per-slice score, one of a thousand bootstrap resamples -- is an accumulation over
a different list of images off the same cache (design section 4.2).

**What is stored is what accumulation reads, and nothing else.** Accumulation
needs four facts per image, per category, per area range: whether each detection
matched a box, whether each detection is ignored by the area filter, each
detection's score, and how many non-ignored boxes were there to find. It never
reads *which* box a detection matched, so `dtMatches` collapses from annotation
IDs to a boolean and the cache becomes bit arrays and floats rather than a
box-to-box mapping. That is what fits 5,000 images in a single `.npz` that the
bootstrap can hold in memory and read a thousand times
(`conventions.eval_matches_key`).

**Blocks are ragged and stored flat.** One image has twelve detections and the
next has none, so the natural layout -- an array indexed by image -- would be
either padded to `MAX_DETS` or an object array. Instead every detection in the
pass lives on one axis, and `starts` says where each block begins.
`detection_rows` turns a list of blocks into indices into that axis, which is
what makes a resample a numpy gather instead of a Python loop.

**Block order is `COCOeval`'s own**: category, then area range, then image row.
Two properties follow, and accumulation depends on both. The blocks of one
(category, area range) group are contiguous and in image-row order, so gathering
an ascending image list reproduces the concatenation order `accumulate()` would
have built -- which is what makes the stable sort by score break ties the same
way, and so what makes equality with `accumulate()` on the identity resample a
meaningful test rather than an approximate one.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from edge_ml_flywheel.conventions import ClassSet, ClassSetVersion, ImageId, class_set
from edge_ml_flywheel.evaluation.coco import ImageIndex

# COCO's ten thresholds, 0.50 to 0.95 in steps of 0.05, spelled the way
# `pycocotools` spells them rather than as a literal list: the headline metric is
# the mean over these, so a value differing in the last bit is a number that no
# longer compares to a published COCO result.
#
# Read-only because a `Final` numpy array is still a mutable buffer, and this one
# is handed to `COCOeval.params`, persisted into every cache, and checked on load.
IOU_THRESHOLDS: Final[NDArray[np.float64]] = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)
IOU_THRESHOLDS.setflags(write=False)

# The cap the cache is built at, and the largest one anything can ask for later.
# `COCOeval` applies it inside the per-image step, after sorting by score, so a
# block holds at most this many detections and a smaller cap is a truncation of
# each block rather than a re-score (see `detection_rows`).
MAX_DETS: Final = 100


class AreaRange(StrEnum):
    """COCO's four box-area filters, evaluated in one pass.

    Declaration order is block order in the cache, so it is as permanent as
    `ClassSet`'s tuple order for the same reason: a stored block index means a
    different area range if this is reordered, and nothing raises.

    `SMALL` is the slice design section 4.4 gates on, and it is defined in native
    1280x720 pixels rather than at the model's 416 px because that is the frame
    every coordinate reaching `coco` is already in.
    """

    ALL = "all"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# `pycocotools`' own bounds, in square pixels. The endpoints are its literals --
# `1e5**2` for an upper bound that excludes nothing and `1e10` for the last one --
# because a bound stated differently silently reassigns boxes near 96x96 to
# another slice.
AREA_BOUNDS: Final[dict[AreaRange, tuple[float, float]]] = {
    AreaRange.ALL: (0.0, 1e5**2),
    AreaRange.SMALL: (0.0, 32.0**2),
    AreaRange.MEDIUM: (32.0**2, 96.0**2),
    AreaRange.LARGE: (96.0**2, 1e10),
}


@dataclass(frozen=True, slots=True, eq=False)
class MatchCache:
    """One model-seed's scoring pass. Data only: no model, no boxes, no I/O.

    `eq=False` because the fields are numpy arrays and a generated `__eq__` would
    raise on the first comparison rather than answer. Equality of two caches is
    not a question anything asks; equality of the AP read off them is, and that is
    a float.

    `class_set_version` rather than a `ClassSet`, because a category ID in
    `matched` is a position in that version's tuple and the version is the fact
    that makes it readable. Passing the set itself would let a caller score under
    one set and record another.

    The arrays:

    - `matched`, `ignored` -- `(len(IOU_THRESHOLDS), n_detections)` booleans.
    - `scores` -- `(n_detections,)`, float64 and descending within each block.
      Float64 rather than float32 to keep the global sort order identical to
      `accumulate()`'s, since a rounded tie reorders two detections and moves AP.
    - `starts` -- `(n_blocks + 1,)`, the flat offset each block begins at.
    - `n_truth` -- `(n_blocks,)`, non-ignored ground-truth boxes in that block.
      A block scores nothing without it: it is AP's denominator, and it cannot be
      recovered from the detections, which say nothing about what was missed.
    """

    index: ImageIndex
    class_set_version: ClassSetVersion
    matched: NDArray[np.bool_]
    ignored: NDArray[np.bool_]
    scores: NDArray[np.float64]
    starts: NDArray[np.int64]
    n_truth: NDArray[np.int64]

    def __post_init__(self) -> None:
        classes = self.classes  # raises on a version nobody wrote down
        expected_blocks = len(classes.names) * len(AreaRange) * len(self.index)
        if self.n_truth.shape != (expected_blocks,):
            raise ValueError(
                f"cache holds {self.n_truth.shape} truth counts, and "
                f"{len(classes.names)} classes over {len(AreaRange)} area ranges and "
                f"{len(self.index)} images is {expected_blocks} blocks"
            )
        if self.starts.shape != (expected_blocks + 1,):
            raise ValueError(
                f"{expected_blocks} blocks need {expected_blocks + 1} offsets, not "
                f"{self.starts.shape}"
            )
        if self.starts[0] != 0:
            raise ValueError(f"the first block starts at {self.starts[0]}, not 0")
        lengths = np.diff(self.starts)
        if np.any(lengths < 0):
            raise ValueError("block offsets are not non-decreasing")
        if np.any(lengths > MAX_DETS):
            raise ValueError(f"a block holds more than {MAX_DETS} detections")

        total = int(self.starts[-1])
        thresholds = len(IOU_THRESHOLDS)
        for name, array in (("matched", self.matched), ("ignored", self.ignored)):
            if array.shape != (thresholds, total):
                raise ValueError(
                    f"{name} is {array.shape}, and {total} detections at {thresholds} IoU "
                    f"thresholds is {(thresholds, total)}"
                )
        if self.scores.shape != (total,):
            raise ValueError(f"scores is {self.scores.shape}, not {(total,)}")

    @property
    def classes(self) -> ClassSet:
        return class_set(self.class_set_version)

    def blocks(
        self, category_id: int, area: AreaRange, image_rows: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Block indices for one category and area range over these image rows.

        `image_rows` are `ImageIndex` positions and may repeat: a bootstrap
        resample is a draw with replacement, and an image drawn twice must count
        twice, which here is the same block index appearing twice.

        Ascending and complete is the identity resample, and it is deliberately
        not special-cased -- the whole point of the layout is that the overall
        metric and one resample take the same path.
        """
        names = self.classes.names
        if not 1 <= category_id <= len(names):
            raise ValueError(f"category ID {category_id} is outside a class set of {len(names)}")
        images = len(self.index)
        if image_rows.size and (image_rows.min() < 0 or image_rows.max() >= images):
            raise ValueError(f"an image row is outside an index of {images} images")
        # Category IDs are 1-based, and blocks are laid out category-major.
        group = (category_id - 1) * len(AreaRange) + list(AreaRange).index(area)
        return np.asarray(group * images + image_rows, dtype=np.int64)

    def detection_rows(
        self, blocks: NDArray[np.int64], max_dets: int = MAX_DETS
    ) -> NDArray[np.int64]:
        """Indices into `scores`, `matched` and `ignored` for these blocks, in order.

        The gather that makes a resample cheap. Blocks are concatenated in the
        order given, so a caller controls the tie-break order of the sort by score
        that follows.

        `max_dets` truncates each block to its highest-scoring detections, which
        is valid only because a block is already descending by score -- the cap is
        `COCOeval`'s own semantics, applied to the cache rather than frozen into
        it (`coco.detections` declines to apply it earlier for the same reason).
        """
        if not 1 <= max_dets <= MAX_DETS:
            raise ValueError(f"max_dets must be between 1 and the cached {MAX_DETS}: {max_dets}")
        lengths = np.minimum(self.starts[blocks + 1] - self.starts[blocks], max_dets)
        # Standard ragged gather: repeat each block's start once per detection it
        # contributes, then add that detection's rank within its block. The rank
        # is a global arange minus the running total of the blocks before it.
        starts = np.repeat(self.starts[blocks], lengths)
        ends = np.cumsum(lengths)
        rank = np.arange(int(ends[-1]) if ends.size else 0) - np.repeat(ends - lengths, lengths)
        return np.asarray(starts + rank, dtype=np.int64)

    def truth_count(self, blocks: NDArray[np.int64]) -> int:
        """Non-ignored ground-truth boxes across these blocks. AP's denominator.

        Summed with repeats, like `detection_rows` gathers with them: a resample
        that draws an image twice has twice as many boxes to find.
        """
        return int(self.n_truth[blocks].sum())


def score(
    truth: COCO,
    results: COCO,
    class_set_version: ClassSetVersion,
    index: ImageIndex,
) -> MatchCache:
    """Run the per-image assignment once and keep the arrays.

    `truth` and `results` come from `coco.as_coco` and `coco.as_coco_results`, so
    both are already restricted to `index`'s images and this version's categories.

    Every IoU threshold and every area range in one pass, because `evaluate()`
    computes the IoU matrix per image and category once and reuses it across
    thresholds; scoring per slice instead would recompute that matrix four times
    for answers that are already in hand.
    """
    classes = class_set(class_set_version)
    category_ids = [classes.category_id(name) for name in classes.names]

    evaluator = COCOeval(truth, results, iouType="bbox")
    evaluator.params.imgIds = list(index.numeric_ids)
    evaluator.params.catIds = category_ids
    evaluator.params.iouThrs = np.array(IOU_THRESHOLDS)
    evaluator.params.areaRng = [list(AREA_BOUNDS[area]) for area in AreaRange]
    evaluator.params.areaRngLbl = [area.value for area in AreaRange]
    evaluator.params.maxDets = [MAX_DETS]
    evaluator.evaluate()

    # `evaluate()` rewrites both lists through `np.unique`, so the block layout
    # depends on what came back rather than on what went in. Both are already
    # sorted and distinct here, which makes this an assertion about `ImageIndex`
    # and `ClassSet` holding -- not a reordering to accommodate.
    if tuple(int(image) for image in evaluator.params.imgIds) != index.numeric_ids:
        raise ValueError(
            "COCOeval reordered the image IDs, so block offsets would not address them"
        )
    if [int(category) for category in evaluator.params.catIds] != category_ids:
        raise ValueError("COCOeval reordered the category IDs, so block offsets name other classes")

    blocks = len(category_ids) * len(AreaRange) * len(index)
    matched_blocks: list[NDArray[np.bool_]] = []
    ignored_blocks: list[NDArray[np.bool_]] = []
    score_blocks: list[NDArray[np.float64]] = []
    lengths = np.zeros(blocks, dtype=np.int64)
    n_truth = np.zeros(blocks, dtype=np.int64)

    for position, entry in enumerate(evaluator.evalImgs):
        # `evaluateImg` returns `None` for an image with neither a box nor a
        # detection in this category, which the typeshed stub does not model. That
        # is an empty block and not a gap: the image is still in the index, and a
        # resample that draws it contributes nothing to this category's AP.
        if entry is None:
            continue
        # `dtMatches` holds the matched box's annotation ID, and `ground_truth`
        # numbers annotations from 1, so zero is the unmatched sentinel.
        matched_blocks.append(np.asarray(entry["dtMatches"]) > 0)
        ignored_blocks.append(np.asarray(entry["dtIgnore"]) > 0)
        score_blocks.append(np.asarray(entry["dtScores"], dtype=np.float64))
        lengths[position] = len(entry["dtScores"])
        n_truth[position] = int(np.count_nonzero(np.asarray(entry["gtIgnore"]) == 0))

    thresholds = len(IOU_THRESHOLDS)
    starts = np.zeros(blocks + 1, dtype=np.int64)
    np.cumsum(lengths, out=starts[1:])

    return MatchCache(
        index=index,
        class_set_version=class_set_version,
        # `hstack` on an empty list raises, and a model that detected nothing is a
        # gate failure to report rather than an exception (`coco.as_coco_results`
        # makes the same allowance).
        matched=(
            np.hstack(matched_blocks)
            if matched_blocks
            else np.zeros((thresholds, 0), dtype=np.bool_)
        ),
        ignored=(
            np.hstack(ignored_blocks)
            if ignored_blocks
            else np.zeros((thresholds, 0), dtype=np.bool_)
        ),
        scores=(np.concatenate(score_blocks) if score_blocks else np.zeros(0, dtype=np.float64)),
        starts=starts,
        n_truth=n_truth,
    )


# Archive member names, spelled once. `np.load` returns a mapping, so a renamed
# field would otherwise be a `KeyError` on a file written weeks earlier by a
# version of this module that agreed with itself.
_IMAGE_IDS: Final = "image_ids"
_CLASS_SET_VERSION: Final = "class_set_version"
_IOU_THRESHOLDS: Final = "iou_thresholds"
_MATCHED: Final = "matched"
_IGNORED: Final = "ignored"
_SCORES: Final = "scores"
_STARTS: Final = "starts"
_N_TRUTH: Final = "n_truth"


def save(cache: MatchCache, path: Path) -> None:
    """Write the `.npz` that goes to `conventions.eval_matches_key`.

    Compressed, because `matched` and `ignored` are one byte per boolean in memory
    and mostly zero at the high IoU thresholds.

    The image IDs and `class_set_version` travel with the arrays, so a reader
    needs the file and nothing else: an index rebuilt from a cohort query six
    weeks later would be a second derivation of the mapping the integers were
    written under. `IOU_THRESHOLDS` travels for a different reason -- it is a
    constant, so it is checked on load rather than used.
    """
    members: dict[str, NDArray[Any]] = {
        _IMAGE_IDS: np.array(cache.index.image_ids),
        _CLASS_SET_VERSION: np.array(cache.class_set_version),
        _IOU_THRESHOLDS: np.array(IOU_THRESHOLDS),
        _MATCHED: cache.matched,
        _IGNORED: cache.ignored,
        _SCORES: cache.scores,
        _STARTS: cache.starts,
        _N_TRUTH: cache.n_truth,
    }
    # `savez_compressed` takes its member names as keyword arguments, and the one
    # keyword it declares itself -- `allow_pickle` -- makes a `**` unpacking of
    # arrays unmatchable to a checker. Naming the members inline instead would
    # spell each of them twice, which is the failure `_IMAGE_IDS` and the rest
    # exist to prevent.
    np.savez_compressed(path, **members)  # type: ignore[arg-type]


def load(path: Path) -> MatchCache:
    """Read a cache back. Refuses one scored at other IoU thresholds.

    A threshold set that has moved since the file was written would otherwise be
    read as the current one: the arrays are indexed by threshold position, so the
    mismatch is silent and every AP off the file is a mean over the wrong ten
    numbers. Refusing beats correcting -- the pass cannot be re-derived from the
    cache, only re-run.
    """
    with np.load(path) as archive:
        thresholds = archive[_IOU_THRESHOLDS]
        if not np.array_equal(thresholds, IOU_THRESHOLDS):
            raise ValueError(
                f"{path.name} was scored at IoU thresholds {thresholds.tolist()}, and this "
                f"module reads {IOU_THRESHOLDS.tolist()}"
            )
        return MatchCache(
            index=ImageIndex.of(ImageId(str(image)) for image in archive[_IMAGE_IDS]),
            class_set_version=ClassSetVersion(int(archive[_CLASS_SET_VERSION])),
            matched=archive[_MATCHED],
            ignored=archive[_IGNORED],
            scores=archive[_SCORES],
            starts=archive[_STARTS],
            n_truth=archive[_N_TRUTH],
        )

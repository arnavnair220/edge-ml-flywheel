"""One uncertainty score per pool image, from what the champion detected in it.

Pure functions over detections. Nothing here loads a model, opens a socket or
reads a label -- the champion's inference is somebody else's step, and what
arrives is the list of boxes it produced. That is what keeps the ranking testable
without a trained detector, and it is also what keeps selection inside the label
wall: an image's score is a function of predictions alone, so no part of scoring
could read ground truth even if it wanted to.

**Per object, not per image.** An image-level maximum is decided by its single
worst box, which ranks a frame carrying one ambiguous detection above a frame the
model is uniformly unsure of (design section 2). The mean asks the question the
budget is actually spending against: how hard is this frame, on average, for the
champion.

**Two ways an image can have no in-band detections, and they are opposites.** A
frame of thirty cars at 0.99 is the champion's best case, and a frame it found
nothing at all in is its blind spot. Both leave the mean undefined, so both need
an answer rather than a crash, and giving them the same answer would put the
easiest images in the batch next to the most interesting ones. They are scored at
opposite ends, and the blind spots are counted in the selection report -- ranking
them top is a choice no cycle has yet reported on, and the count is what turns
the first bad batch into a number.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from edge_ml_flywheel.conventions import ClassSet, ImageId
from edge_ml_flywheel.evaluation.coco import Detection

# The confidence range a detection has to fall in to count toward the mean.
#
# The upper edge is the one doing the work. A detection at 0.99 is not evidence
# of anything the champion finds hard, and averaging it in dilutes a frame's
# score in proportion to how many easy objects happen to share the frame -- which
# would rank a quiet road above a busy one for being quiet. The lower edge is
# near-symmetric and mostly inert: whatever runs inference has already applied its
# own confidence threshold, so detections below it do not arrive.
#
# Constants rather than fields on a config object, for `gates.thresholds`'
# reason: these are design parameters with one setting for the life of the
# project, and a caller free to widen the band is a caller who can shop for a
# ranking. A cycle that shows the band is wrong changes them here, once.
BAND_LOW: Final = 0.05
BAND_HIGH: Final = 0.95

# What an image with no detections at all scores. The top of the range the
# formula below can produce, so a blind spot outranks every frame the champion
# did see something in. It ties with a perfectly ambiguous image -- every box at
# exactly 0.5 -- which floating point makes a case that does not arise, and which
# `selection.select` breaks on image ID anyway rather than leaving to chance.
BLIND_SPOT: Final = 1.0

# What an image scores when it has detections and none of them are in the band.
# The champion was decisive about everything it saw, which is the opposite
# statement from having seen nothing, and the two must not collapse together.
DECISIVE: Final = 0.0


@dataclass(frozen=True, slots=True)
class Predictions:
    """What the champion produced over the pool, and the vocabulary it produced it in.

    The two travel together because neither is interpretable alone. A category
    string means nothing without the class set that defines it, and the class
    set is what makes a class the champion detected *nothing* of visible -- an
    absent count and a zero count are the same bytes otherwise.

    The category check is the reason this is a class rather than two arguments.
    `conventions` records that the legacy archive spells three categories
    differently from the `det_20` release, and that a mismatch is silent: the
    class scores zero AP every cycle and nothing raises. Here it would be
    quieter still -- a batch ranked on detections nobody could name -- so it is
    refused where the two facts first meet.

    An image with no entry is not an error. It is the blind spot, and scoring it
    is the point.
    """

    classes: ClassSet
    of_image: Mapping[ImageId, Sequence[Detection]]

    def __post_init__(self) -> None:
        known = set(self.classes.names)
        unknown = sorted(
            {
                detection.category
                for detections in self.of_image.values()
                for detection in detections
                if detection.category not in known
            }
        )
        if unknown:
            raise ValueError(
                f"the champion detected categories outside the class set it is scored under: "
                f"{unknown}. Declared: {list(self.classes.names)}"
            )

    def for_image(self, image_id: ImageId) -> Sequence[Detection]:
        return self.of_image.get(image_id, ())


def uncertainty(confidence: float) -> float:
    """How far one detection sits from a decision, on a 0-to-1 scale.

    Peaks at 0.5 and falls to zero at both ends. A box at 0.5 is the one the
    champion has no opinion about, and a label for it buys the most; a box at
    0.02 is a near-certain *rejection* and is as informative as one at 0.98,
    which is why this is distance from the boundary rather than `1 - confidence`.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"a confidence outside [0, 1] is not a confidence: {confidence}")
    return 1.0 - abs(2.0 * confidence - 1.0)


def image_score(detections: Sequence[Detection]) -> float:
    """Mean uncertainty over one image's in-band detections.

    The two empty cases are told apart here rather than by the caller, because
    the caller only has the same list to tell them apart with -- and one of them
    is the top of the ranking while the other is the bottom.
    """
    in_band = [
        uncertainty(detection.score)
        for detection in detections
        if BAND_LOW <= detection.score <= BAND_HIGH
    ]
    if in_band:
        return sum(in_band) / len(in_band)
    return DECISIVE if detections else BLIND_SPOT


def score_pool(pool: Iterable[ImageId], predictions: Predictions) -> dict[ImageId, float]:
    """Score every image in the pool, whether or not inference produced boxes for it.

    Driven by the pool rather than by `detected`, so an image the champion found
    nothing in is scored as a blind spot instead of quietly dropping out of the
    ranking. Silently missing from a ranking is the failure mode this whole
    module exists to avoid: the frames the model handles worst are exactly the
    ones that produce no detections to rank by.

    A detection for an image outside the pool is refused rather than ignored. It
    means inference ran over the wrong image set -- the eval cohort, most
    consequentially -- and a selector working from a set it was not given is a
    bug worth stopping for, not one worth filtering away.
    """
    scores = {image_id: image_score(predictions.for_image(image_id)) for image_id in pool}

    stray = sorted(set(predictions.of_image) - set(scores))
    if stray:
        raise ValueError(
            f"{len(stray)} image(s) have detections but are not in the pool being scored: "
            f"{stray[:5]}"
        )
    if not scores:
        raise ValueError("an empty pool has nothing to rank")
    return scores

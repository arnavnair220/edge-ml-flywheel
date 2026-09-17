"""Turning a raw ONNX output into detections, without a runtime to produce one.

`training.quantization`'s arrangement, for the same cause: the part of the device
path worth testing is the part that reads a tensor, so it lives where the suite
can reach it with no onnxruntime installed and no ARM to install it on. `replay`
holds the session, the images and the publish, and imports this.

**The device does its own decoding because the export left NMS out.** The
quantizer has no int8 form for it (design section 4.3, and `training.export`), so
the graph emits raw predictions and whatever runs it applies the suppression.
In the cloud that is Ultralytics; on the device this is, and the two have to
agree closely enough that a confidence reported from a device is comparable to
the same frame's confidence from the offline pass.

**Only the scores are kept downstream.** The boxes are computed anyway and
discarded by the caller, because suppression *is* box arithmetic -- there is no
cheaper way to find out that two predictions are one object. What that means for
this module is that a coordinate error here surfaces as a detection count rather
than as a wrong box, which is the harder failure to see and the reason the
geometry is tested rather than eyeballed.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

# Boxes overlapping more than this are the same object. Ultralytics' own default,
# matched deliberately: the cloud pass and the device pass should differ by the
# silicon and the precision, not by a suppression threshold one of them chose.
IOU_THRESHOLD: Final = 0.7

# The most predictions one frame may keep, after suppression. Ultralytics'
# default again, and a ceiling rather than an expectation -- a BDD100K frame
# carries eighteen boxes on average, so this fires on a frame the model has
# fragmented rather than on a busy one.
MAX_DETECTIONS: Final = 300

# What a YOLO output row spends on geometry before the class scores start:
# centre x, centre y, width, height.
_BOX_FIELDS: Final = 4

# Batch, channel, anchor. One frame at a time -- batch size 1 is what design
# section 4.3 measures and what an edge device does -- so the batch axis is
# present and is always one.
_OUTPUT_RANK: Final = 3


def decode(
    output: NDArray[np.float32], confidence_floor: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.intp]]:
    """Split one frame's raw output into boxes, scores and classes.

    The tensor arrives as `(1, 4 + classes, anchors)` -- geometry and class
    scores down the rows, one column per anchor -- which is the layout every
    Ultralytics detection export writes. It is transposed here rather than at the
    call site because the row-major form is the one every operation below wants,
    and a caller that transposed it itself would be a caller that has to know
    this layout.

    The score of a prediction is its highest class score and its class is that
    score's position. There is no separate objectness term in this head, which is
    the v8-and-later shape: reading one would be reading a class score and
    multiplying by it twice.

    Boxes come back as corners in the network's own input pixels, which is the
    letterboxed frame rather than the source image. Nothing here undoes that,
    deliberately -- suppression is scale-invariant, and the caller keeps the
    scores, so a rescale would be arithmetic performed to be thrown away.
    """
    if output.ndim != _OUTPUT_RANK or output.shape[0] != 1:
        raise ValueError(
            f"expected one frame's output as (1, 4 + classes, anchors), got {output.shape}"
        )
    if output.shape[1] <= _BOX_FIELDS:
        raise ValueError(
            f"an output with {output.shape[1]} rows carries geometry and no class scores, so it "
            f"is not a detection head this project exports"
        )
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError(
            f"a confidence floor outside [0, 1] admits nothing or everything: {confidence_floor}"
        )

    predictions = output[0].T
    centres = predictions[:, :_BOX_FIELDS]
    class_scores = predictions[:, _BOX_FIELDS:]

    scores = class_scores.max(axis=1)
    classes = class_scores.argmax(axis=1)

    kept = scores >= confidence_floor
    return _corners(centres[kept]), scores[kept].astype(np.float32), classes[kept]


def _corners(centres: NDArray[np.float32]) -> NDArray[np.float32]:
    """Centre-and-size to opposite corners, in the order `DetectionRow` states.

    Same order as every other box in this project, so a device's geometry and a
    scoring job's are the same four numbers meaning the same four things.
    """
    half = centres[:, 2:] / 2.0
    return np.concatenate([centres[:, :2] - half, centres[:, :2] + half], axis=1).astype(np.float32)


def suppress(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    classes: NDArray[np.intp],
    iou_threshold: float = IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> NDArray[np.intp]:
    """Which predictions survive, as indices into the arrays given.

    Indices rather than filtered arrays, so a caller keeping only the scores does
    not pay to have the boxes gathered as well.

    **Suppression is per class, done in one pass.** Each box is offset by its
    class into a region of the plane no other class occupies, which makes two
    predictions of different classes non-overlapping by construction and lets one
    global sweep do what a loop over classes would. The offset is derived from
    the coordinates present rather than fixed, so it cannot be too small for a
    frame whose boxes happen to be large.

    Greedy and highest-score-first, which is what every implementation this is
    meant to agree with does. A frame yielding more than `max_detections`
    survivors is truncated by score, not refused: the tail of a fragmented frame
    is the least confident part of it, and a device that dropped the frame
    entirely would report a gap that reads as a lost message.
    """
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError(
            f"{len(boxes)} boxes, {len(scores)} scores and {len(classes)} classes are not one "
            f"frame's predictions"
        )
    if not 0.0 < iou_threshold < 1.0:
        raise ValueError(
            f"an IoU threshold of {iou_threshold} either suppresses every box or none of them"
        )
    if len(boxes) == 0:
        return np.empty(0, dtype=np.intp)

    offset = (classes * (float(boxes.max()) + 1.0)).astype(np.float32)
    shifted = boxes + offset[:, None]

    areas = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    order = scores.argsort()[::-1].astype(np.intp)

    keep: list[np.intp] = []
    while order.size > 0 and len(keep) < max_detections:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break

        rest = order[1:]
        overlap = _intersections(shifted[best], shifted[rest])
        union = areas[best] + areas[rest] - overlap
        # A zero union is a degenerate pair rather than a perfect overlap, and
        # dividing by it would make the whole row NaN -- which compares false
        # against the threshold and keeps a box that should have gone.
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.intp)


def _intersections(box: NDArray[np.float32], others: NDArray[np.float32]) -> NDArray[np.float32]:
    """Overlapping area between one box and many, clipped at zero.

    The clip is what makes a non-overlapping pair contribute nothing instead of a
    negative area, which would make the union larger than either box and the IoU
    quietly too small.
    """
    left = np.maximum(box[0], others[:, 0])
    top = np.maximum(box[1], others[:, 1])
    right = np.minimum(box[2], others[:, 2])
    bottom = np.minimum(box[3], others[:, 3])
    width = np.maximum(0.0, right - left)
    height = np.maximum(0.0, bottom - top)
    return np.asarray(width * height, dtype=np.float32)


def scores_of(
    output: NDArray[np.float32],
    confidence_floor: float,
    iou_threshold: float = IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> tuple[float, ...]:
    """One frame's surviving confidences, which is all the telemetry carries.

    The whole decode-and-suppress path behind one call, because the device wants
    exactly this and nothing else, and because a caller assembling the two steps
    itself is a caller that can pass the boxes of one frame with the scores of
    another.

    Descending, so a record is readable without being sorted and two frames'
    records compare without either being rearranged first.
    """
    boxes, scores, classes = decode(output, confidence_floor)
    kept = suppress(boxes, scores, classes, iou_threshold, max_detections)
    return tuple(sorted((float(score) for score in scores[kept]), reverse=True))

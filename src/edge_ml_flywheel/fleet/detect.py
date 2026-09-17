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

**The boxes are kept.** This is the pass the cycle's ranking is computed from, so
what leaves here is `DetectionRow`s in source-image pixels -- the same seven
facts the cloud pass writes, in the same units, into the same parquet schema. A
coordinate error therefore surfaces as a wrong box in a file someone can open,
which is the failure this module's geometry tests are for.

**Rescaling is this module's job and nothing else's.** The tensor comes back in
the letterboxed square the network was fed, and `DetectionRow` is specified in
`NATIVE_IMAGE_SIZE` pixels. `rescale` is the one place that conversion happens on
the device, mirroring the single rescale the cloud path performs.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from edge_ml_flywheel.conventions import CLASS_SET, DetectionRow, ImageId

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


@dataclass(frozen=True, slots=True)
class Letterbox:
    """How one source image was fitted into the network's square, as numbers.

    The padding geometry written down once, because two operations need it and
    they must not disagree: `replay.letterbox` builds the canvas from it, and
    `rescale` undoes it. Derived rather than passed, so a device cannot rescale
    by a factor other than the one it padded with.

    `width` and `height` are the pasted image's size in network pixels, and they
    are rounded -- so the honest inverse divides by `width / source_width` rather
    than by the unrounded `scale` that produced them. At 416 px the two differ by
    a fraction of a pixel, which is smaller than anything the metric can see, but
    the rounded form is the one that actually happened.
    """

    source_width: int
    source_height: int
    size: int

    def __post_init__(self) -> None:
        if self.source_width < 1 or self.source_height < 1:
            raise ValueError(
                f"an image of {self.source_width}x{self.source_height} px is not an image"
            )
        if self.size < 1:
            raise ValueError(f"an input of {self.size} px is not a square to fit into")

    @property
    def scale(self) -> float:
        """The fit, before rounding. What the resize is computed from."""
        return min(self.size / self.source_width, self.size / self.source_height)

    @property
    def width(self) -> int:
        return max(1, round(self.source_width * self.scale))

    @property
    def height(self) -> int:
        return max(1, round(self.source_height * self.scale))

    @property
    def pad_x(self) -> int:
        """Where the pasted image starts, which is where a box's origin is."""
        return (self.size - self.width) // 2

    @property
    def pad_y(self) -> int:
        return (self.size - self.height) // 2


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


def rescale(boxes: NDArray[np.float32], fit: Letterbox) -> NDArray[np.float32]:
    """Boxes from the padded square back into source-image pixels.

    The inverse of the paste `replay.letterbox` performs: subtract where the
    image was placed, then divide by how much it was shrunk. Both axes are
    divided by their own factor rather than by one shared `scale`, because the
    pasted size is rounded independently on each.

    Clipped to the source frame. A prediction can extend into the grey padding --
    the model has no idea the padding is not road -- and a corner outside the
    image is a coordinate no ground truth can ever sit at. Clipping is what makes
    the geometry comparable to the cloud pass, which clips for the same reason.
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    scale_x = fit.width / fit.source_width
    scale_y = fit.height / fit.source_height

    moved = boxes.astype(np.float32).copy()
    moved[:, [0, 2]] = (moved[:, [0, 2]] - fit.pad_x) / scale_x
    moved[:, [1, 3]] = (moved[:, [1, 3]] - fit.pad_y) / scale_y

    moved[:, [0, 2]] = moved[:, [0, 2]].clip(0.0, fit.source_width)
    moved[:, [1, 3]] = moved[:, [1, 3]].clip(0.0, fit.source_height)
    return moved


def detections_of(
    output: NDArray[np.float32],
    image_id: ImageId,
    fit: Letterbox,
    confidence_floor: float,
) -> tuple[DetectionRow, ...]:
    """One frame's predictions, as the rows a detections file holds.

    The whole device-side path behind one call -- decode, suppress, rescale,
    name -- because the caller wants exactly this and because a caller assembling
    the steps itself is a caller that can pass the boxes of one frame with the
    scores of another.

    The suppression settings are this module's constants rather than arguments.
    They exist to agree with the cloud pass, so a caller free to vary them is a
    caller who can produce a ranking from a different suppression than the one
    every other cycle used; a test exercising the thresholds calls `suppress`.

    Descending by confidence, so a file is readable without being sorted and two
    frames' rows compare without either being rearranged first.

    **A frame with no surviving prediction returns no rows, and that is not the
    same as the frame being absent.** `selection.score` reads emptiness as a
    blind spot, which is the top of its ranking rather than the bottom, so the
    frame's presence is carried by the sample manifest and the telemetry row
    rather than by a placeholder box here.

    Boxes that clip away to nothing are dropped. A prediction lying entirely in
    the padding has no source-image area, and `DetectionRow` refuses corners that
    enclose none -- correctly, since a box of zero area is not something a metric
    can match against.
    """
    # The head's width against the class set, checked before a name is looked up
    # by position. A graph exported for a different vocabulary would otherwise
    # write plausible rows under the wrong categories, which is a ranking nobody
    # can tell is wrong from the file.
    predicted = output.shape[1] - _BOX_FIELDS
    if predicted != len(CLASS_SET.names):
        raise ValueError(
            f"a head predicting {predicted} classes is not this project's {len(CLASS_SET.names)}: "
            f"{list(CLASS_SET.names)}"
        )

    boxes, scores, classes = decode(output, confidence_floor)
    kept = suppress(boxes, scores, classes)
    if kept.size == 0:
        return ()

    order = kept[scores[kept].argsort()[::-1]]
    corners = rescale(boxes[order], fit)

    rows: list[DetectionRow] = []
    for (x1, y1, x2, y2), score, category in zip(
        corners, scores[order], classes[order], strict=True
    ):
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            DetectionRow(
                image_id=image_id,
                category=CLASS_SET.names[int(category)],
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
            )
        )
    return tuple(rows)

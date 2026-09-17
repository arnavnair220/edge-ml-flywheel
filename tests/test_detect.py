"""Tests for the device's decode and suppression, with no runtime in sight.

`test_quantization`'s situation: the part of the device path worth testing is the
part that reads a tensor, and it lives where the suite can reach it without
onnxruntime installed or ARM silicon to install it on.

Tensors are built by hand at the shape an Ultralytics detection export writes --
`(1, 4 + classes, anchors)`, geometry down the rows -- because that layout is
what the module is an assertion about. A fixture captured from a real export
would test the same thing and would stop being reproducible the moment the export
changed.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from edge_ml_flywheel.fleet import detect

CLASSES = 9


def an_output(
    *predictions: tuple[tuple[float, float, float, float], int, float],
) -> NDArray[np.float32]:
    """A raw output carrying the given boxes, each as (centre box, class, score).

    Built transposed, the way the graph emits it, so the module's own transpose
    is exercised rather than bypassed.
    """
    rows = np.zeros((4 + CLASSES, max(1, len(predictions))), dtype=np.float32)
    for anchor, (box, category, score) in enumerate(predictions):
        rows[:4, anchor] = box
        rows[4 + category, anchor] = score
    return rows[None]


class TestDecode:
    def test_a_prediction_comes_back_as_corners(self) -> None:
        """Centre and size in, opposite corners out, in the order `DetectionRow`
        states them -- so a device's geometry and a scoring job's are the same
        four numbers meaning the same four things."""
        boxes, scores, classes = detect.decode(an_output(((50, 50, 20, 10), 2, 0.9)), 0.1)

        assert boxes.tolist() == [[40.0, 45.0, 60.0, 55.0]]
        assert scores.tolist() == [pytest.approx(0.9)]
        assert classes.tolist() == [2]

    def test_the_score_is_the_highest_class_and_the_class_is_its_position(self) -> None:
        """There is no separate objectness term in this head, so reading one
        would be reading a class score and multiplying by it twice."""
        output = an_output(((50, 50, 20, 20), 0, 0.3))
        output[0, 4 + 5, 0] = 0.8

        _, scores, classes = detect.decode(output, 0.1)

        assert scores.tolist() == [pytest.approx(0.8)]
        assert classes.tolist() == [5]

    def test_predictions_below_the_floor_are_dropped(self) -> None:
        boxes, _, _ = detect.decode(
            an_output(((50, 50, 20, 20), 0, 0.9), ((10, 10, 4, 4), 1, 0.01)), 0.5
        )

        assert len(boxes) == 1

    def test_a_floor_of_zero_keeps_everything(self) -> None:
        boxes, _, _ = detect.decode(an_output(((50, 50, 20, 20), 0, 0.0)), 0.0)

        assert len(boxes) == 1

    def test_a_batch_of_more_than_one_frame_is_refused(self) -> None:
        """Batch size 1 is what design section 4.3 measures and what an edge
        device does, so a second frame in the tensor means the caller changed."""
        with pytest.raises(ValueError, match="one frame's output"):
            detect.decode(np.zeros((2, 4 + CLASSES, 1), dtype=np.float32), 0.1)

    def test_an_output_with_no_class_scores_is_refused(self) -> None:
        with pytest.raises(ValueError, match="geometry and no class scores"):
            detect.decode(np.zeros((1, 4, 10), dtype=np.float32), 0.1)

    def test_a_floor_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="admits nothing or everything"):
            detect.decode(an_output(((50, 50, 20, 20), 0, 0.9)), 1.5)


class TestSuppression:
    def test_two_predictions_of_one_object_become_one(self) -> None:
        boxes, scores, classes = detect.decode(
            an_output(((50, 50, 20, 20), 0, 0.9), ((51, 51, 20, 20), 0, 0.7)), 0.1
        )
        kept = detect.suppress(boxes, scores, classes)

        assert len(kept) == 1
        assert scores[kept].tolist() == [pytest.approx(0.9)]

    def test_the_highest_score_is_the_one_kept(self) -> None:
        """Greedy and highest-first, which is what every implementation this is
        meant to agree with does."""
        boxes, scores, classes = detect.decode(
            an_output(((51, 51, 20, 20), 0, 0.4), ((50, 50, 20, 20), 0, 0.95)), 0.1
        )
        kept = detect.suppress(boxes, scores, classes)

        assert scores[kept].tolist() == [pytest.approx(0.95)]

    def test_two_classes_in_one_place_both_survive(self) -> None:
        """Suppression is per class. A car and a truck predicted over the same
        pixels are two predictions, and folding them together would report one
        detection where the model made two."""
        boxes, scores, classes = detect.decode(
            an_output(((50, 50, 20, 20), 0, 0.9), ((50, 50, 20, 20), 1, 0.8)), 0.1
        )

        assert len(detect.suppress(boxes, scores, classes)) == 2

    def test_the_class_offset_is_large_enough_for_the_boxes_present(self) -> None:
        """Derived from the coordinates rather than fixed, so it cannot be too
        small for a frame whose boxes happen to be large."""
        boxes, scores, classes = detect.decode(
            an_output(((900, 900, 700, 700), 0, 0.9), ((900, 900, 700, 700), 1, 0.8)), 0.1
        )

        assert len(detect.suppress(boxes, scores, classes)) == 2

    def test_boxes_far_apart_both_survive(self) -> None:
        boxes, scores, classes = detect.decode(
            an_output(((10, 10, 8, 8), 0, 0.9), ((300, 300, 8, 8), 0, 0.8)), 0.1
        )

        assert len(detect.suppress(boxes, scores, classes)) == 2

    def test_a_frame_with_no_prediction_suppresses_to_nothing(self) -> None:
        boxes, scores, classes = detect.decode(an_output(((50, 50, 20, 20), 0, 0.01)), 0.5)

        assert len(detect.suppress(boxes, scores, classes)) == 0

    def test_a_fragmented_frame_is_truncated_by_score_rather_than_dropped(self) -> None:
        """The tail of a fragmented frame is its least confident part, and a
        device that dropped the frame would report a gap that reads as a lost
        message."""
        spread = an_output(
            *(((float(20 * n), 10.0, 8.0, 8.0), 0, 0.5 + n / 1000) for n in range(10))
        )
        boxes, scores, classes = detect.decode(spread, 0.1)
        kept = detect.suppress(boxes, scores, classes, max_detections=3)

        assert len(kept) == 3
        assert scores[kept].tolist() == sorted(scores.tolist(), reverse=True)[:3]

    def test_mismatched_arrays_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not one frame's predictions"):
            detect.suppress(
                np.zeros((2, 4), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(2, dtype=np.intp),
            )

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 2.0])
    def test_an_iou_threshold_that_decides_nothing_is_refused(self, value: float) -> None:
        boxes, scores, classes = detect.decode(an_output(((50, 50, 20, 20), 0, 0.9)), 0.1)

        with pytest.raises(ValueError, match="suppresses every box or none"):
            detect.suppress(boxes, scores, classes, iou_threshold=value)


class TestScoresOf:
    def test_the_whole_path_returns_confidences_descending(self) -> None:
        """A record readable without being sorted, and two frames' records that
        compare without either being rearranged first."""
        output = an_output(
            ((50, 50, 20, 20), 0, 0.4), ((300, 300, 20, 20), 1, 0.9), ((10, 10, 8, 8), 2, 0.7)
        )

        assert detect.scores_of(output, 0.1) == pytest.approx((0.9, 0.7, 0.4))

    def test_a_frame_the_model_found_nothing_in_returns_nothing(self) -> None:
        assert detect.scores_of(an_output(((50, 50, 20, 20), 0, 0.01)), 0.5) == ()

    def test_duplicates_are_already_suppressed(self) -> None:
        """The boxes are computed and discarded, because suppression is box
        arithmetic -- there is no cheaper way to find out that two predictions
        are one object."""
        output = an_output(((50, 50, 20, 20), 0, 0.9), ((50, 51, 20, 20), 0, 0.85))

        assert detect.scores_of(output, 0.1) == pytest.approx((0.9,))

    def test_the_defaults_are_ultralytics_own(self) -> None:
        """Matched deliberately: the cloud pass and the device pass should differ
        by the silicon and the precision, not by a threshold one of them chose."""
        assert detect.IOU_THRESHOLD == 0.7
        assert detect.MAX_DETECTIONS == 300

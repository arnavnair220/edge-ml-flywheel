"""Unit tests for the detections file: the row, the codec, and the two shapes
both consumers take it in.

This file is the join between the scoring job and everything after it, and every
failure it can have is quiet. A transposed corner is a box in the wrong place; a
dropped score makes AP's ranking arbitrary; an absent file instead of an empty
one reads as a model that detected nothing, which is a verdict the quality gate
acts on. None of those raise on their own.
"""

from pathlib import Path

import pytest

from edge_ml_flywheel.conventions import (
    Cohort,
    Cycle,
    DetectionRow,
    ImageId,
    RunId,
    Seed,
    detections_key,
    new_model_version,
)
from edge_ml_flywheel.scoring import detections

RUN = RunId("20260812t143355z-v1-uncertainty")
VERSION = new_model_version(RUN, Cycle(1))


def an_image_id(index: int) -> ImageId:
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


def a_row(
    image: int = 1,
    category: str = "car",
    score: float = 0.9,
    **corners: float,
) -> DetectionRow:
    box = {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 70.0} | corners
    return DetectionRow(image_id=an_image_id(image), category=category, score=score, **box)


class TestTheRow:
    def test_it_carries_the_corners_and_the_score(self) -> None:
        """The fields of `evaluation.coco.Detection` plus the image, in the same
        frame, so reading one back is a construction and not a conversion."""
        row = a_row()
        assert (row.x1, row.y1, row.x2, row.y2) == (10.0, 20.0, 110.0, 70.0)
        assert row.score == 0.9

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_a_confidence_outside_zero_to_one_is_refused(self, score: float) -> None:
        with pytest.raises(ValueError, match="not a confidence"):
            a_row(score=score)

    def test_reversed_corners_are_refused(self) -> None:
        """A reversed pair converts to a negative width, which COCO's area filter
        reads as a box below every threshold rather than a malformed one -- so it
        would drop out of the small-object slice silently instead of raising."""
        with pytest.raises(ValueError, match="positive area"):
            a_row(x1=200.0, x2=100.0)

    def test_a_box_with_no_area_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive area"):
            a_row(y1=50.0, y2=50.0)

    def test_an_id_that_is_not_a_bdd_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a BDD100K image ID"):
            DetectionRow(
                image_id=ImageId("nope"),
                category="car",
                x1=1.0,
                y1=1.0,
                x2=2.0,
                y2=2.0,
                score=0.5,
            )


class TestTheCodec:
    def test_it_round_trips(self, tmp_path: Path) -> None:
        rows = [a_row(image=1), a_row(image=2, category="bus", score=0.25)]
        path = tmp_path / "part-00000.parquet"

        assert detections.write(rows, path) == 2
        assert list(detections.read(path)) == rows

    def test_a_model_that_saw_nothing_still_writes_a_file(self, tmp_path: Path) -> None:
        """Design section 4.2 hard fails a challenger whose classes collapse to
        zero detections, and the gate can only report that if the absence arrives
        as a file with no rows. A missing object is indistinguishable from a job
        that died before writing."""
        path = tmp_path / "part-00000.parquet"

        assert detections.write([], path) == 0
        assert path.is_file()
        assert list(detections.read(path)) == []

    def test_it_writes_more_rows_than_fit_in_one_row_group(self, tmp_path: Path) -> None:
        """The pool is 62,000 images at up to `MAX_DETS` each, so the writer
        flushes rather than holding the cohort in memory. The boundary is where a
        chunked writer gets a row count wrong."""
        rows = [a_row(image=index) for index in range(detections.ROW_GROUP + 3)]
        path = tmp_path / "part-00000.parquet"

        assert detections.write(rows, path) == len(rows)
        assert sum(1 for _ in detections.read(path)) == len(rows)

    def test_it_consumes_a_generator_without_a_list(self, tmp_path: Path) -> None:
        """The container hands it inference results as they come, which is what
        keeps peak memory a buffer rather than every box in the cohort."""
        path = tmp_path / "part-00000.parquet"
        assert detections.write((a_row(image=index) for index in range(5)), path) == 5

    def test_collect_reads_every_part(self, tmp_path: Path) -> None:
        detections.write([a_row(image=1)], tmp_path / "part-00000.parquet")
        detections.write([a_row(image=2)], tmp_path / "part-00001.parquet")

        assert len(detections.collect(tmp_path)) == 2


class TestTheShapeConsumersTake:
    def test_it_groups_by_image(self) -> None:
        rows = [a_row(image=1), a_row(image=1, category="bus"), a_row(image=2)]
        grouped = detections.group(rows)

        assert set(grouped) == {an_image_id(1), an_image_id(2)}
        assert len(grouped[an_image_id(1)]) == 2

    def test_the_score_survives_the_conversion(self) -> None:
        """AP sweeps a threshold over these, so a constant score makes the
        ranking arbitrary and the metric meaningless."""
        grouped = detections.group([a_row(score=0.42)])
        assert grouped[an_image_id(1)][0].score == 0.42

    def test_an_image_with_no_detections_is_simply_absent(self) -> None:
        """Which is what both consumers expect: `coco.detections` documents that
        an image may be missing, and `score.score_pool` is driven by the pool so
        that a blind spot is scored rather than dropped."""
        assert an_image_id(9) not in detections.group([a_row(image=1)])


class TestTheKey:
    def test_it_is_under_the_cycle_that_produced_the_model(self) -> None:
        """Not the cycle being decided: a champion is re-compared every cycle
        without being re-scored, so keying by the deciding cycle would write a
        fresh copy of an unchanged answer every time."""
        key = detections_key(new_model_version(RUN, Cycle(3)), Seed(1), Cohort.EVAL)
        assert key.startswith(f"run_id={RUN}/cycle=003/")
        assert key.endswith("cohort=eval/part-00000.parquet")

    def test_no_key_exists_for_a_cohort_nothing_scores(self) -> None:
        """The gate produces the path rather than being a check performed before
        building one, so there is no argument that names this file."""
        for cohort in (Cohort.BOOTSTRAP, Cohort.RESERVE):
            with pytest.raises(ValueError, match="not a scored cohort"):
                detections_key(VERSION, Seed(1), cohort)

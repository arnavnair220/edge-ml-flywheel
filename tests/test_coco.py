"""The BDD-to-COCO conversion, and what `pycocotools` does with the result.

Boxes are placed by hand at readable coordinates, so a failure names the broken
conversion rather than reporting that a number moved.

The round trip runs through `pycocotools` rather than through a local inverse. An
inverse would establish only that two halves of one module agree; loading the
document into `COCO` and reading the annotations back establishes the module's
actual claim, that `pycocotools` accepts these shapes.
"""

import pytest

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    NATIVE_IMAGE_SIZE,
    ClassSet,
    ImageId,
)
from edge_ml_flywheel.evaluation.coco import (
    Detection,
    ImageIndex,
    detections,
    ground_truth,
)
from edge_ml_flywheel.evaluation.match import as_coco, as_coco_results
from edge_ml_flywheel.ingest.labels import Box

# Well-formed BDD100K IDs, declared out of sorted order so every test exercises
# `ImageIndex.of`'s sort rather than the literal order here.
IMAGE_B = ImageId("ffffffff-00000002")
IMAGE_A = ImageId("00000000-00000001")

CLASSES = CLASS_SET


def a_box(
    category: str = "car", x1: float = 10, y1: float = 20, w: float = 100, h: float = 50
) -> Box:
    """A box by corner and extent, which is the readable form."""
    return Box(category=category, x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)


def a_detection(box: Box | None = None, score: float = 0.9) -> Detection:
    """A detection is a box with a score, so it is built from one."""
    box = a_box() if box is None else box
    return Detection(category=box.category, x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, score=score)


# --- ImageIndex ---------------------------------------------------------------


class TestImageIndex:
    def test_sorts_and_numbers_from_zero(self):
        index = ImageIndex.of([IMAGE_B, IMAGE_A])

        assert index.image_ids == (IMAGE_A, IMAGE_B)
        assert index.numeric(IMAGE_A) == 0
        assert index.numeric(IMAGE_B) == 1
        assert index.numeric_ids == (0, 1)
        assert len(index) == 2

    def test_caller_order_does_not_change_the_ids(self):
        assert ImageIndex.of([IMAGE_A, IMAGE_B]) == ImageIndex.of([IMAGE_B, IMAGE_A])

    def test_numeric_and_image_id_are_inverses(self):
        index = ImageIndex.of([IMAGE_B, IMAGE_A])

        for image_id in index.image_ids:
            assert index.image_id(index.numeric(image_id)) == image_id

    def test_refuses_a_repeat_rather_than_deduplicating(self):
        with pytest.raises(ValueError, match="repeats an image"):
            ImageIndex.of([IMAGE_A, IMAGE_B, IMAGE_A])

    def test_refuses_an_empty_set(self):
        with pytest.raises(ValueError, match="no images"):
            ImageIndex.of([])

    def test_refuses_an_id_it_does_not_hold(self):
        index = ImageIndex.of([IMAGE_A])

        with pytest.raises(ValueError, match="not in this eval index"):
            index.numeric(IMAGE_B)

    @pytest.mark.parametrize("numeric", [-1, 1, 99])
    def test_refuses_a_number_outside_the_index(self, numeric: int):
        with pytest.raises(ValueError, match="outside an index"):
            ImageIndex.of([IMAGE_A]).image_id(numeric)


# --- Ground truth -------------------------------------------------------------


class TestGroundTruth:
    def test_corners_become_x_y_width_height(self):
        index = ImageIndex.of([IMAGE_A])
        document = ground_truth({IMAGE_A: [a_box(x1=10, y1=20, w=100, h=50)]}, CLASSES, index)

        (annotation,) = document["annotations"]
        assert annotation["bbox"] == [10, 20, 100, 50]
        assert annotation["area"] == 5_000
        assert annotation["iscrowd"] == 0

    def test_category_ids_come_from_the_class_set(self):
        index = ImageIndex.of([IMAGE_A])
        boxes = [a_box(category=name) for name in CLASSES.names]
        document = ground_truth({IMAGE_A: boxes}, CLASSES, index)

        assert [annotation["category_id"] for annotation in document["annotations"]] == [
            CLASSES.category_id(name) for name in CLASSES.names
        ]
        assert document["categories"] == [
            {"id": position, "name": name} for position, name in enumerate(CLASSES.names, start=1)
        ]

    def test_annotation_ids_are_unique(self):
        index = ImageIndex.of([IMAGE_A, IMAGE_B])
        document = ground_truth(
            {IMAGE_A: [a_box(), a_box(y1=300)], IMAGE_B: [a_box()]}, CLASSES, index
        )

        identifiers = [annotation["id"] for annotation in document["annotations"]]
        assert identifiers == [1, 2, 3]

    def test_every_indexed_image_appears_at_native_resolution(self):
        index = ImageIndex.of([IMAGE_A, IMAGE_B])
        document = ground_truth({IMAGE_A: [a_box()], IMAGE_B: []}, CLASSES, index)

        width, height = NATIVE_IMAGE_SIZE
        assert [image["id"] for image in document["images"]] == [0, 1]
        assert all(
            (image["width"], image["height"]) == (width, height) for image in document["images"]
        )

    def test_a_category_outside_the_class_set_is_dropped(self):
        """`train` is in the archive and in no class set (design section 3)."""
        index = ImageIndex.of([IMAGE_A])
        document = ground_truth(
            {IMAGE_A: [a_box(category="car"), a_box(category="train")]}, CLASSES, index
        )

        assert len(document["annotations"]) == 1
        assert document["annotations"][0]["category_id"] == CLASSES.category_id("car")

    def test_an_image_with_no_boxes_must_be_present_not_absent(self):
        index = ImageIndex.of([IMAGE_A, IMAGE_B])

        with pytest.raises(ValueError, match="no labels entry"):
            ground_truth({IMAGE_A: [a_box()]}, CLASSES, index)

    def test_refuses_labels_for_an_unindexed_image(self):
        index = ImageIndex.of([IMAGE_A])

        with pytest.raises(ValueError, match="not in the eval index"):
            ground_truth({IMAGE_A: [], IMAGE_B: [a_box()]}, CLASSES, index)

    def test_refuses_a_box_enclosing_no_area(self):
        index = ImageIndex.of([IMAGE_A])
        backwards = Box(category="car", x1=200, y1=20, x2=100, y2=70)

        with pytest.raises(ValueError, match="encloses no area"):
            ground_truth({IMAGE_A: [backwards]}, CLASSES, index)


# --- Detections ---------------------------------------------------------------


class TestDetections:
    def test_carries_the_score_and_converts_the_box(self):
        index = ImageIndex.of([IMAGE_A])
        records = detections({IMAGE_A: [a_detection(score=0.75)]}, CLASSES, index)

        assert records == [
            {"image_id": 0, "category_id": 1, "bbox": [10, 20, 100, 50], "score": 0.75}
        ]

    def test_an_image_with_no_predictions_may_be_absent(self):
        index = ImageIndex.of([IMAGE_A, IMAGE_B])

        assert detections({IMAGE_B: [a_detection()]}, CLASSES, index) == [
            {"image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 50], "score": 0.9}
        ]

    def test_a_class_outside_the_set_raises_rather_than_being_dropped(self):
        index = ImageIndex.of([IMAGE_A])

        with pytest.raises(ValueError, match="not in the class set"):
            detections({IMAGE_A: [a_detection(a_box(category="train"))]}, CLASSES, index)

    def test_refuses_a_prediction_for_an_unindexed_image(self):
        index = ImageIndex.of([IMAGE_A])

        with pytest.raises(ValueError, match="not in the eval index"):
            detections({IMAGE_B: [a_detection()]}, CLASSES, index)

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_refuses_a_score_outside_zero_to_one(self, score: float):
        index = ImageIndex.of([IMAGE_A])

        with pytest.raises(ValueError, match="outside"):
            detections({IMAGE_A: [a_detection(score=score)]}, CLASSES, index)


# --- What pycocotools makes of it ---------------------------------------------


class TestRoundTrip:
    def test_pycocotools_reads_back_the_boxes_that_went_in(self):
        index = ImageIndex.of([IMAGE_B, IMAGE_A])
        labels = {
            IMAGE_A: [a_box(category="car", x1=10, y1=20, w=100, h=50)],
            IMAGE_B: [
                a_box(category="person", x1=400, y1=300, w=30, h=80),
                a_box(category="bus", x1=600, y1=100, w=200, h=200),
            ],
        }

        coco = as_coco(ground_truth(labels, CLASSES, index))

        assert sorted(coco.getImgIds()) == [0, 1]
        assert sorted(coco.getCatIds()) == list(range(1, len(CLASSES.names) + 1))

        for image_id, boxes in labels.items():
            loaded = coco.loadAnns(coco.getAnnIds(imgIds=[index.numeric(image_id)]))
            assert [annotation["bbox"] for annotation in loaded] == [
                [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1] for box in boxes
            ]
            assert [annotation["category_id"] for annotation in loaded] == [
                CLASSES.category_id(box.category) for box in boxes
            ]
            assert [annotation["area"] for annotation in loaded] == [box.area for box in boxes]

    def test_areas_land_in_the_expected_coco_size_bands(self):
        """The small-object slice is these bands at native resolution."""
        index = ImageIndex.of([IMAGE_A])
        small = a_box(x1=10, y1=20, w=30, h=30)  # 900 px, under 32x32
        medium = a_box(x1=100, y1=100, w=60, h=60)  # 3,600 px
        large = a_box(x1=300, y1=200, w=200, h=200)  # 40,000 px, over 96x96

        coco = as_coco(ground_truth({IMAGE_A: [small, medium, large]}, CLASSES, index))
        areas = [annotation["area"] for annotation in coco.loadAnns(coco.getAnnIds())]

        assert areas == [900, 3_600, 40_000]
        assert areas[0] < 32**2 <= areas[1] < 96**2 <= areas[2]

    def test_load_res_accepts_the_detection_records(self):
        index = ImageIndex.of([IMAGE_A, IMAGE_B])
        truth = as_coco(ground_truth({IMAGE_A: [a_box()], IMAGE_B: []}, CLASSES, index))
        records = detections(
            {IMAGE_A: [a_detection(score=0.8)], IMAGE_B: [a_detection(a_box(category="bus"))]},
            CLASSES,
            index,
        )

        results = as_coco_results(truth, records)
        loaded = results.loadAnns(results.getAnnIds())

        assert [annotation["score"] for annotation in loaded] == [0.8, 0.9]
        assert sorted(results.getImgIds()) == [0, 1]

    def test_a_model_that_detects_nothing_does_not_crash_load_res(self):
        """`COCO.loadRes` reads `anns[0]`, so the empty list is its own path."""
        index = ImageIndex.of([IMAGE_A])
        truth = as_coco(ground_truth({IMAGE_A: [a_box()]}, CLASSES, index))

        results = as_coco_results(truth, detections({}, CLASSES, index))

        assert results.getAnnIds() == []
        assert results.getImgIds() == [0]
        assert sorted(results.getCatIds()) == list(range(1, len(CLASSES.names) + 1))


# --- The class set the IDs come from ------------------------------------------


class TestClassSet:
    def test_ids_are_one_based_positions(self):
        assert CLASS_SET.category_ids["car"] == 1
        assert CLASS_SET.category_name(1) == "car"
        assert CLASS_SET.category_id("person") == 4

    def test_the_project_set_is_the_nine_the_design_names(self):
        """The order is `_provenance/integrity.json`'s, not a preference. A
        category ID is a position in this tuple and is stored in every cached
        match array, so reordering it silently renames every class."""
        assert CLASS_SET.names == (
            "car",
            "traffic sign",
            "traffic light",
            "person",
            "truck",
            "bus",
            "bike",
            "rider",
            "motor",
        )

    def test_train_is_in_the_archive_and_not_in_the_class_set(self):
        assert "train" not in CLASS_SET.names

    def test_refuses_a_det_20_spelling_the_archive_has_no_boxes_for(self):
        """Uncaught, this scores 0.0 AP every cycle and raises nothing."""
        with pytest.raises(ValueError, match="no boxes for"):
            ClassSet(names=("car", "pedestrian"))

    def test_refuses_a_repeated_class(self):
        with pytest.raises(ValueError, match="repeats a class"):
            ClassSet(names=("car", "car"))

    def test_refuses_an_empty_class_set(self):
        with pytest.raises(ValueError, match="not a class set"):
            ClassSet(names=())

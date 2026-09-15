"""Unit tests for the YOLO conversion and the image manifest.

Both are places where a wrong answer trains a model rather than raising. A box
normalized against the wrong frame is a box in the wrong place, a class ID off by
one trains every car as a traffic sign, and a manifest that names an image the
labels do not cover is a job that trains on less than the cycle paid for. None of
those fail; they all produce a model.
"""

import json
from pathlib import Path

import pytest

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    NATIVE_IMAGE_SIZE,
    ImageId,
)
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.training import dataset, images

NINE = CLASS_SET
WIDTH, HEIGHT = NATIVE_IMAGE_SIZE


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


def a_box(category: str = "car", x1: float = 100.0, x2: float = 200.0) -> Box:
    return Box(category=category, x1=x1, y1=100.0, x2=x2, y2=200.0)


def an_image_dir(root: Path, image_ids: list[ImageId]) -> Path:
    """A channel directory of empty JPEGs.

    Empty because nothing in the conversion decodes an image: coordinates are
    normalized against `NATIVE_IMAGE_SIZE`, which ingest already refused anything
    else against.
    """
    channel = root / "images-channel"
    channel.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        (channel / f"{image_id}.jpg").write_bytes(b"")
    return channel


def label_rows(root: Path, image_id: ImageId) -> list[str]:
    text = (root / dataset.LABELS_DIR / f"{image_id}.txt").read_text(encoding="utf-8")
    return text.splitlines()


class TestCoordinates:
    def test_a_box_becomes_a_centre_and_an_extent_over_the_native_frame(
        self, tmp_path: Path
    ) -> None:
        """`car` is class 0, and the corners become centre and extent as
        fractions of 1280x720 -- the frame every box in this project is stated
        in, rather than a size read off the JPEG."""
        image_id = an_image_id(1)
        root = tmp_path / "dataset"
        dataset.write(
            root,
            an_image_dir(tmp_path, [image_id]),
            {image_id: [Box("car", 100.0, 100.0, 200.0, 200.0)]},
            NINE,
        )

        assert label_rows(root, image_id) == [
            f"0 {150 / WIDTH:.6f} {150 / HEIGHT:.6f} {100 / WIDTH:.6f} {100 / HEIGHT:.6f}"
        ]

    def test_class_ids_are_zero_based_where_the_class_set_is_one_based(self) -> None:
        """COCO counts categories from 1 and YOLO from 0. The subtraction happens
        once, in the conversion, so this is where it is checked."""
        assert NINE.category_id("traffic sign") == 2

    def test_the_class_id_matches_the_position_in_data_yaml(self, tmp_path: Path) -> None:
        image_id = an_image_id(2)
        root = tmp_path / "dataset"
        dataset.write(
            root, an_image_dir(tmp_path, [image_id]), {image_id: [a_box("traffic sign")]}, NINE
        )

        assert label_rows(root, image_id)[0].split()[0] == "1"
        assert "  1: traffic sign" in (root / dataset.DATA_YAML).read_text(encoding="utf-8")


class TestWhatIsDropped:
    def test_a_category_outside_the_class_set_is_dropped_and_counted(self, tmp_path: Path) -> None:
        """`train` is in the archive and in neither class set, at 151 boxes
        archive-wide. A class the model is not trained on must not be one it is
        scored on."""
        image_id = an_image_id(3)
        root = tmp_path / "dataset"
        stats = dataset.write(
            root,
            an_image_dir(tmp_path, [image_id]),
            {image_id: [a_box("car"), a_box("train")]},
            NINE,
        )

        assert stats.dropped_out_of_class == 1
        assert stats.boxes == 1
        assert len(label_rows(root, image_id)) == 1

    def test_a_box_over_the_edge_is_clipped_and_counted(self, tmp_path: Path) -> None:
        """The archive states corners past the frame on objects entering or
        leaving it. Dropping them instead would teach the model that a
        half-visible car is background."""
        image_id = an_image_id(4)
        root = tmp_path / "dataset"
        stats = dataset.write(
            root,
            an_image_dir(tmp_path, [image_id]),
            {image_id: [Box("car", -40.0, 100.0, 60.0, 200.0)]},
            NINE,
        )

        assert stats.clipped == 1
        assert stats.boxes == 1
        # Clipped at 0, so the box is 0 to 60 rather than -40 to 60.
        assert label_rows(root, image_id)[0].split()[1] == f"{30 / WIDTH:.6f}"

    def test_a_box_entirely_outside_the_frame_is_dropped_as_degenerate(
        self, tmp_path: Path
    ) -> None:
        image_id = an_image_id(5)
        root = tmp_path / "dataset"
        stats = dataset.write(
            root,
            an_image_dir(tmp_path, [image_id]),
            {image_id: [Box("car", -80.0, 100.0, -20.0, 200.0)]},
            NINE,
        )

        assert stats.dropped_degenerate == 1
        assert stats.boxes == 0

    def test_an_image_with_no_boxes_is_kept_with_an_empty_label_file(self, tmp_path: Path) -> None:
        """YOLO's own spelling of a background frame. "There is nothing here" is
        an answer the model has to learn, and it is a legitimate label the oracle
        charges for."""
        image_id = an_image_id(6)
        root = tmp_path / "dataset"
        stats = dataset.write(root, an_image_dir(tmp_path, [image_id]), {image_id: []}, NINE)

        assert stats.images == 1
        assert stats.empty_images == 1
        assert (root / dataset.LABELS_DIR / f"{image_id}.txt").read_text(encoding="utf-8") == ""


class TestTheChannelAndTheLabelsMustAgree:
    def test_an_image_with_no_label_is_refused(self, tmp_path: Path) -> None:
        """The manifest named something this run has not bought."""
        labeled, unlabeled = an_image_id(7), an_image_id(8)
        with pytest.raises(ValueError, match="have no label"):
            dataset.write(
                tmp_path / "dataset",
                an_image_dir(tmp_path, [labeled, unlabeled]),
                {labeled: [a_box()]},
                NINE,
            )

    def test_a_label_with_no_image_is_refused(self, tmp_path: Path) -> None:
        """The manifest was written from a different label set than the one that
        arrived, so the model would train on fewer images than the cycle paid
        for -- and nothing downstream would say so."""
        present, absent = an_image_id(9), an_image_id(10)
        with pytest.raises(ValueError, match="did not arrive"):
            dataset.write(
                tmp_path / "dataset",
                an_image_dir(tmp_path, [present]),
                {present: [a_box()], absent: [a_box()]},
                NINE,
            )


class TestTheImageManifest:
    def test_it_leads_with_one_prefix_and_then_names_keys_under_it(self) -> None:
        document = images.document("bucket", [an_image_id(2), an_image_id(1)])

        assert document[0] == {"prefix": "s3://bucket/raw/images/100k/train/"}
        assert document[1:] == [f"{an_image_id(1)}.jpg", f"{an_image_id(2)}.jpg"]

    def test_it_is_sorted_and_deduplicated(self) -> None:
        """Sorted so two runs over one labeled set write the same file, and
        deduplicated because an image named twice is downloaded twice."""
        repeated = [an_image_id(3), an_image_id(1), an_image_id(3)]
        assert images.document("bucket", repeated)[1:] == [
            f"{an_image_id(1)}.jpg",
            f"{an_image_id(3)}.jpg",
        ]

    def test_a_cycle_with_no_images_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no images to train on"):
            images.document("bucket", [])

    def test_it_round_trips_through_json(self, tmp_path: Path) -> None:
        path = tmp_path / "images.manifest"
        named = images.write(path, "bucket", [an_image_id(1), an_image_id(2)])

        assert named == 2
        assert json.loads(path.read_text(encoding="utf-8"))[0]["prefix"].startswith("s3://bucket/")

    def test_the_cap_takes_the_front_of_the_sort(self) -> None:
        """Arbitrary the way a sample would be, without a second seed in a
        project where `seed` already means one thing."""
        image_ids = [an_image_id(index) for index in range(10)]
        assert images.capped(image_ids, 3) == tuple(sorted(image_ids))[:3]

    def test_zero_means_the_whole_labeled_set(self) -> None:
        """Everything is the real default, and any positive default would be a
        cap nobody chose silently shrinking a real run."""
        image_ids = [an_image_id(index) for index in range(4)]
        assert images.capped(image_ids, 0) == tuple(sorted(image_ids))

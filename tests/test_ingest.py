"""Unit tests for the ingest module, on synthetic data in a temp directory.

The real archives are 5.8 GB and live behind an hour-long CodeBuild run, so the
tests that matter here are the ones that would otherwise only fail at minute
forty: a label document shaped slightly differently than expected, a file that
lands at the wrong key, a schema that has drifted from `ManifestRow`.

`verification/test_real_extract.py` is the other half, and it runs against the
real extract. This file is about the parsing and the layout; that one is about
the data.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from PIL import Image

from edge_ml_flywheel.conventions import (
    LABEL_SOURCE,
    NATIVE_IMAGE_SIZE,
    ImageId,
    ManifestRow,
    Scene,
    Split,
    TimeOfDay,
    Weather,
    columns,
    manifest_key,
    raw_image_key,
    raw_label_key,
)
from edge_ml_flywheel.ingest import manifest as manifest_module
from edge_ml_flywheel.ingest.__main__ import _listed_keys, _parser, _require_host, _staged_keys
from edge_ml_flywheel.ingest.images import inspect_image
from edge_ml_flywheel.ingest.labels import parse_label
from edge_ml_flywheel.ingest.provenance import verify_archive
from edge_ml_flywheel.ingest.source import IMAGES, LABELS
from edge_ml_flywheel.ingest.stage import stage

IMAGE = ImageId("0000f77c-6257be58")
OTHER = ImageId("00054602-3bf57337")

BUILDSPEC = Path(__file__).resolve().parent.parent / "buildspecs" / "ingest.yml"

# Every step the CLI offers. A literal rather than a read of the parser's own
# choices, so adding a subcommand is a decision about whether the buildspec
# should call it rather than a line that updates itself.
SUBCOMMANDS = frozenset(
    {"url", "verify-archives", "stage", "manifest", "provenance", "verify-upload"}
)

# `url` is called twice, once per archive.
BUILDSPEC_INVOCATIONS = 7

_INVOCATION = re.compile(r"python -m edge_ml_flywheel\.ingest\s+(.+)")


def _placeholder(token: str) -> str:
    """A shell expansion stands in for its result, which argparse never sees.

    `"$WORK/extract"` is a path to argparse either way, and nothing at parse
    time touches the filesystem.
    """
    return "placeholder" if token.startswith("$") else token


def _buildspec_commands() -> tuple[list[str], ...]:
    """Every `python -m edge_ml_flywheel.ingest ...` the buildspec runs.

    Only YAML list items, so a command quoted in a comment is not mistaken for
    one that executes. The trailing paren of a `$(...)` capture is dropped
    before splitting, since the shell removes it too.
    """
    found: list[list[str]] = []
    for line in BUILDSPEC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        match = _INVOCATION.search(stripped)
        if match is None:
            continue
        tail = match.group(1).strip().removesuffix(")")
        found.append([_placeholder(token) for token in shlex.split(tail)])
    return tuple(found)


BUILDSPEC_COMMANDS = _buildspec_commands()


def a_label(**overrides: Any) -> dict[str, Any]:
    """A valid legacy Scalabel document, with named parts replaced."""
    document: dict[str, Any] = {
        "name": f"{IMAGE}.jpg",
        "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
        "frames": [
            {
                "timestamp": 10000,
                "objects": [
                    {
                        "category": "car",
                        "id": 0,
                        "attributes": {"occluded": False, "truncated": False},
                        "box2d": {"x1": 100.0, "y1": 200.0, "x2": 110.0, "y2": 220.0},
                    },
                    {
                        "category": "traffic light",
                        "id": 1,
                        "box2d": {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 5.0},
                    },
                    {"category": "area/drivable", "id": 2, "poly2d": [[0, 0], [1, 1]]},
                    {"category": "lane/road curb", "id": 3, "poly2d": [[2, 2], [3, 3]]},
                ],
            }
        ],
    }
    return document | overrides


def an_image(path: Path, size: tuple[int, int] = (64, 32), *, blank: bool = False) -> None:
    """A real JPEG on disk. Gradient unless a blank frame is what is wanted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if blank:
        image = Image.new("RGB", size, (0, 0, 0))
    else:
        image = Image.linear_gradient("L").convert("RGB").resize(size)
    image.save(path, "JPEG")


# --- source -------------------------------------------------------------------


class TestSource:
    def test_url_is_built_from_the_host(self) -> None:
        assert LABELS.url("example.test") == "http://example.test/bdd100k/bdd100k_labels.zip"

    def test_the_images_archive_publishes_no_digest(self) -> None:
        # Pinned because it is the reason the image-ID set check exists. If a
        # digest ever appears upstream, this test is the prompt to use it.
        assert IMAGES.sha256 is None

    def test_the_labels_digest_is_a_lowercase_hex_sha256(self) -> None:
        assert LABELS.sha256 is not None
        assert len(LABELS.sha256) == 64
        assert LABELS.sha256 == LABELS.sha256.lower()


class TestVerifyArchive:
    def test_a_wrong_size_is_refused_before_the_digest_is_read(self, tmp_path: Path) -> None:
        truncated = tmp_path / LABELS.name
        truncated.write_bytes(b"not the archive")
        with pytest.raises(ValueError, match="expected 189,638,612 bytes"):
            verify_archive(LABELS, truncated)


# --- labels -------------------------------------------------------------------


class TestParseLabel:
    def test_reads_the_three_attributes_into_their_vocabularies(self) -> None:
        parsed = parse_label(a_label(), IMAGE)
        assert (parsed.weather, parsed.scene, parsed.timeofday) == (
            Weather.CLEAR,
            Scene.HIGHWAY,
            TimeOfDay.DAYTIME,
        )

    def test_areas_are_native_pixels_in_document_order(self) -> None:
        assert parse_label(a_label(), IMAGE).box_areas == (200.0, 20.0)

    def test_polygons_are_not_boxes(self) -> None:
        # The area/* and lane/* entries carry poly2d rather than box2d, which is
        # the structural fact the parser selects on instead of a class list.
        parsed = parse_label(a_label(), IMAGE)
        assert parsed.box_categories == ("car", "traffic light")

    def test_an_image_with_nothing_to_detect(self) -> None:
        document = a_label(frames=[{"timestamp": 0, "objects": []}])
        parsed = parse_label(document, IMAGE)
        assert parsed.box_areas == ()
        assert parsed.degenerate_boxes == 0

    @pytest.mark.parametrize(
        ("x2", "y2"),
        [(100.0, 220.0), (110.0, 200.0), (90.0, 220.0)],
        ids=["zero-width", "zero-height", "corners-reversed"],
    )
    def test_a_box_enclosing_no_area_is_dropped_and_counted(self, x2: float, y2: float) -> None:
        document = a_label(
            frames=[
                {
                    "timestamp": 0,
                    "objects": [
                        {"category": "car", "box2d": {"x1": 100.0, "y1": 200.0, "x2": x2, "y2": y2}}
                    ],
                }
            ]
        )
        parsed = parse_label(document, IMAGE)
        assert parsed.box_areas == ()
        assert parsed.degenerate_boxes == 1

    def test_a_missing_name_is_tolerated(self) -> None:
        document = a_label()
        del document["name"]
        assert parse_label(document, IMAGE).image_id == IMAGE

    def test_a_name_for_another_image_is_not(self) -> None:
        with pytest.raises(ValueError, match="names a different image"):
            parse_label(a_label(name=f"{OTHER}.jpg"), IMAGE)

    @pytest.mark.parametrize("attribute", ["weather", "scene", "timeofday"])
    def test_rejects_a_missing_attribute(self, attribute: str) -> None:
        attributes = dict(a_label()["attributes"])
        del attributes[attribute]
        with pytest.raises(ValueError, match=f"attribute {attribute!r} is missing"):
            parse_label(a_label(attributes=attributes), IMAGE)

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("weather", "drizzle"),
            ("scene", "gas station"),
            ("timeofday", "dawn-dusk"),
        ],
        ids=["unknown", "the-plural-dropped", "the-slash-tidied"],
    )
    def test_rejects_a_value_outside_the_vocabulary(self, attribute: str, value: str) -> None:
        # The vocabularies were counted over all 80,000 images, so this means the
        # host is serving different data. Carrying the value through would write a
        # tag no eval slice can match, and an empty slice reads as a clean pass.
        attributes = dict(a_label()["attributes"]) | {attribute: value}
        with pytest.raises(ValueError, match=f"attribute {attribute!r} is {value!r}"):
            parse_label(a_label(attributes=attributes), IMAGE)

    @pytest.mark.parametrize(
        ("frames", "message"),
        [
            ([], "expected exactly one frame, found 0"),
            ([{"objects": []}, {"objects": []}], "expected exactly one frame, found 2"),
            ("nope", "expected exactly one frame, found no"),
        ],
        ids=["none", "two", "not-a-list"],
    )
    def test_rejects_anything_but_one_frame(self, frames: Any, message: str) -> None:
        # Two frames would mean a tracking archive wearing the wrong name, which
        # this host has already shipped twice, or every box counted twice.
        with pytest.raises(ValueError, match=message):
            parse_label(a_label(frames=frames), IMAGE)

    def test_rejects_a_box_with_no_category(self) -> None:
        document = a_label(
            frames=[
                {
                    "objects": [
                        {"box2d": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="no category"):
            parse_label(document, IMAGE)

    def test_rejects_a_non_numeric_corner(self) -> None:
        document = a_label(
            frames=[
                {
                    "objects": [
                        {"category": "car", "box2d": {"x1": 0.0, "y1": 0.0, "x2": "1", "y2": 1.0}},
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="no numeric 'x2'"):
            parse_label(document, IMAGE)

    def test_rejects_a_document_that_is_not_an_object(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            parse_label([1, 2, 3], IMAGE)


# --- images -------------------------------------------------------------------


class TestInspectImage:
    def test_reports_the_digest_and_the_size(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jpg"
        an_image(path, NATIVE_IMAGE_SIZE)
        facts = inspect_image(path)
        assert facts.decode_error is None
        assert facts.size == NATIVE_IMAGE_SIZE
        assert len(facts.sha256) == 64

    def test_a_gradient_is_not_blank(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jpg"
        an_image(path)
        assert inspect_image(path).is_blank is False

    def test_a_single_luminance_frame_is_blank(self, tmp_path: Path) -> None:
        path = tmp_path / "black.jpg"
        an_image(path, blank=True)
        assert inspect_image(path).is_blank is True

    def test_an_undecodable_file_still_has_a_digest(self, tmp_path: Path) -> None:
        # The point of not raising: a bad file that the report can name is worth
        # more than an exception that loses which file it was.
        path = tmp_path / "broken.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0 not actually a jpeg")
        facts = inspect_image(path)
        assert facts.decode_error is not None
        assert facts.size is None
        assert len(facts.sha256) == 64


# --- stage --------------------------------------------------------------------


# What the archives really unzip to, measured from a failed run: no top-level
# directory at all, and both modalities merged into one `100k/<split>/` tree.
# The wrapped form is what a repackaged mirror would ship. Staging reads only
# the trailing three components, so both work -- and the first ingest attempt
# died precisely because the flat form was assumed not to exist.
LAYOUTS = ("", "bdd100k")


def an_extract(root: Path, split: str, image_id: str, top: str = "") -> None:
    """One image and its label, at the layout the archives unzip to."""
    base = root / top if top else root
    an_image(base / "100k" / split / f"{image_id}.jpg")
    label = base / "100k" / split / f"{image_id}.json"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(json.dumps(a_label(name=f"{image_id}.jpg")), encoding="utf-8")


class TestStage:
    @pytest.mark.parametrize("top", LAYOUTS, ids=["flat", "wrapped"])
    def test_files_land_at_the_keys_conventions_builds(self, tmp_path: Path, top: str) -> None:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "train", IMAGE, top)

        stage(extract, staged)

        assert (staged / raw_image_key(IMAGE, Split.TRAIN)).is_file()
        assert (staged / raw_label_key(IMAGE, Split.TRAIN)).is_file()

    def test_reports_both_id_sets_per_split(self, tmp_path: Path) -> None:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "train", IMAGE)
        an_extract(extract, "val", OTHER)

        pool = stage(extract, staged)

        assert pool.images[Split.TRAIN] == {IMAGE}
        assert pool.images[Split.VAL] == {OTHER}
        assert pool.labels == pool.images
        assert (pool.image_count, pool.label_count) == (2, 2)

    @pytest.mark.parametrize("top", LAYOUTS, ids=["flat", "wrapped"])
    def test_a_test_split_file_is_a_hard_failure(self, tmp_path: Path, top: str) -> None:
        # The unzip exclusion should mean this never happens. It did happen --
        # the pattern matched nothing and the build extracted the withheld
        # split -- and this is the guard that stopped it. Not downgradable.
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "test", IMAGE, top)

        with pytest.raises(ValueError, match="withheld test-split file"):
            stage(extract, staged)

    def test_files_outside_the_pool_layout_are_reported_not_moved(self, tmp_path: Path) -> None:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "train", IMAGE)
        (extract / "notes.json").write_text("{}", encoding="utf-8")

        pool = stage(extract, staged)

        assert pool.ignored == ("notes.json",)
        assert (extract / "notes.json").is_file()

    def test_an_unknown_split_is_refused(self, tmp_path: Path) -> None:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "trainval", IMAGE)

        with pytest.raises(ValueError, match="unknown split 'trainval'"):
            stage(extract, staged)


# --- manifest -----------------------------------------------------------------


class TestManifestSchema:
    def test_parquet_columns_are_the_schema_of_record(self) -> None:
        # The one place the column list is written twice -- parquet needs types
        # and a dataclass field does not carry one -- so it is pinned rather
        # than trusted.
        assert tuple(manifest_module.SCHEMA.names) == columns(ManifestRow)

    def test_box_areas_is_a_list_of_doubles(self) -> None:
        assert manifest_module.SCHEMA.field("box_areas").type == pa.list_(pa.float64())


class TestManifestBuild:
    @pytest.fixture
    def built(self, tmp_path: Path) -> tuple[Path, manifest_module.ManifestBuild]:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "train", IMAGE)
        an_extract(extract, "val", OTHER)
        stage(extract, staged)
        return staged, manifest_module.write(staged)

    def test_one_row_per_image(self, built: tuple[Path, manifest_module.ManifestBuild]) -> None:
        _, build = built
        assert [row.image_id for row in build.rows] == [IMAGE, OTHER]

    def test_rows_carry_the_boxes_and_the_label_source(
        self, built: tuple[Path, manifest_module.ManifestBuild]
    ) -> None:
        _, build = built
        row = build.rows[0]
        assert row.n_boxes == 2
        assert row.box_areas == (200.0, 20.0)
        assert row.label_source == LABEL_SOURCE

    def test_the_vocabulary_is_measured_rather_than_assumed(
        self, built: tuple[Path, manifest_module.ManifestBuild]
    ) -> None:
        _, build = built
        assert build.box_categories == {"car": 2, "traffic light": 2}

    def test_a_wrong_resolution_is_a_finding_not_an_exception(self, tmp_path: Path) -> None:
        extract, staged = tmp_path / "extract", tmp_path / "stage"
        an_extract(extract, "train", IMAGE)
        stage(extract, staged)

        build = manifest_module.build(staged)

        assert [finding.problem for finding in build.findings] == ["resolution"]
        assert build.rows[0].image_id == IMAGE

    def test_the_parquet_round_trips(
        self, built: tuple[Path, manifest_module.ManifestBuild]
    ) -> None:
        staged, build = built
        table = manifest_module.read_parquet(staged / manifest_key())
        assert table.num_rows == len(build.rows)
        assert tuple(table.column_names) == columns(ManifestRow)
        assert table.column("box_areas").to_pylist()[0] == [200.0, 20.0]

    def test_the_integrity_report_is_written_beside_the_data(
        self, built: tuple[Path, manifest_module.ManifestBuild]
    ) -> None:
        staged, _ = built
        report = json.loads((staged / manifest_module.INTEGRITY_KEY).read_text(encoding="utf-8"))
        assert report["rows"] == 2
        assert report["degenerate_boxes"] == 0

    def test_staged_ids_reads_both_modalities(
        self, built: tuple[Path, manifest_module.ManifestBuild]
    ) -> None:
        staged, _ = built
        ids = manifest_module.staged_ids(staged)
        assert ids["images"]["train"] == {IMAGE}
        assert ids["labels"]["val"] == {OTHER}


# --- verify-upload ------------------------------------------------------------


class TestUploadVerification:
    def test_staged_keys_are_relative_posix_paths(self, tmp_path: Path) -> None:
        an_image(tmp_path / raw_image_key(IMAGE, Split.TRAIN))
        assert _staged_keys(tmp_path) == {raw_image_key(IMAGE, Split.TRAIN)}

    def test_listing_lines_become_keys(self, tmp_path: Path) -> None:
        listing = tmp_path / "uploaded.txt"
        listing.write_text(
            "2026-08-15 12:00:00      58123 raw/images/100k/train/a.jpg\n"
            "2026-08-15 12:00:01       1024 raw/labels/scalabel/train/a.json\n",
            encoding="utf-8",
        )
        assert _listed_keys(listing) == {
            "raw/images/100k/train/a.jpg",
            "raw/labels/scalabel/train/a.json",
        }

    def test_directory_placeholders_are_not_objects(self, tmp_path: Path) -> None:
        listing = tmp_path / "uploaded.txt"
        listing.write_text(
            "                           PRE raw/\n"
            "2026-08-15 12:00:00          0 raw/images/\n"
            "2026-08-15 12:00:00      58123 raw/images/100k/train/a.jpg\n",
            encoding="utf-8",
        )
        assert _listed_keys(listing) == {"raw/images/100k/train/a.jpg"}


# --- the command line the buildspec actually types ----------------------------


class TestBuildspecInvocations:
    """The buildspec is the only caller of this CLI, so nothing else checks it.

    `--host` was declared on the top-level parser, where argparse accepts it
    only *before* the subcommand. The buildspec wrote it after, every unit test
    called the functions directly, and the disagreement surfaced as a failed
    CodeBuild run. Same class of failure the whole module exists to prevent --
    two spellings of one interface -- one layer up from the S3 keys.

    Parsing the buildspec rather than restating its commands here is the point:
    a copy of the invocations in this file could drift from the file that runs.
    """

    def test_the_commands_were_found(self) -> None:
        """A regex that matched nothing would make every other test vacuous."""
        assert len(BUILDSPEC_COMMANDS) == BUILDSPEC_INVOCATIONS

    @pytest.mark.parametrize("argv", BUILDSPEC_COMMANDS, ids=lambda argv: argv[0])
    def test_every_invocation_parses(self, argv: list[str]) -> None:
        assert _parser().parse_args(argv).command == argv[0]

    def test_every_subcommand_is_reachable_from_the_buildspec(self) -> None:
        """A step nobody runs is a step that rots unnoticed."""
        assert {argv[0] for argv in BUILDSPEC_COMMANDS} == SUBCOMMANDS

    @pytest.mark.parametrize("command", ["url", "verify-archives"])
    def test_host_is_accepted_after_the_subcommand(self, command: str) -> None:
        """The exact spelling that failed, pinned in the order a person types."""
        rest = ["images"] if command == "url" else ["--work-dir", "/tmp/w"]
        parsed = _parser().parse_args([command, *rest, "--host", "example.test"])
        assert parsed.host == "example.test"

    def test_host_falls_back_to_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`verify-archives` passes no `--host`; the project sets BDD100K_HOST."""
        monkeypatch.setenv("BDD100K_HOST", "from.env")
        parsed = _parser().parse_args(["verify-archives", "--work-dir", "/tmp/w"])
        assert parsed.host == "from.env"

    def test_a_missing_host_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(SystemExit):
            _require_host("")

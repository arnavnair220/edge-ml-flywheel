"""Parsing one BDD100K legacy (Scalabel) label document.

One JSON per image, shaped like this:

```json
{
  "name": "0000f77c-6257be58.jpg",
  "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
  "frames": [{"timestamp": 10000, "objects": [
      {"category": "car", "box2d": {"x1": 380.0, "y1": 404.8, "x2": 402.3, "y2": 416.7}},
      {"category": "area/drivable", "poly2d": [...]},
      {"category": "lane/road curb", "poly2d": [...]}
  ]}]
}
```

**A box is an object carrying `box2d`, not an object whose category is on a
list.** The 10 detection categories are exactly the ones with boxes -- the
`area/*` and `lane/*` entries are polygons -- so structure and category agree,
and picking structure means no hardcoded vocabulary that a rename upstream turns
into a silent drop. Legacy calls them `person`/`motor`/`bike` where `det_20`
says `pedestrian`/`motorcycle`/`bicycle`, which is exactly the kind of list
worth not having written down twice. The categories that *did* carry boxes are
counted and recorded in the integrity report, so the vocabulary is measured
rather than asserted.

`weather`, `scene` and `timeofday` are the deliberate opposite: measured once and
then written down, as the vocabularies in `conventions`. The difference is what
reads them. No predicate is written against a box category here -- naming the
classes is `class_set_version`'s job downstream -- whereas every eval slice and
the selection mix record is written against the three tags, and a predicate
is exactly what an enum protects.

**Everything else is strict and raises.** A missing attribute, a tag outside its
vocabulary, a second frame, a label naming a different image: each of those means
the archive is not what this code was written against, and the one thing worse
than failing an hour-long ingest is completing it with a manifest nobody can
trust.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Final

from edge_ml_flywheel.conventions import ImageId, Scene, TimeOfDay, Weather

_CORNERS: Final = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True, slots=True)
class Box:
    """One detection box, in the archive's native 1280x720 pixels.

    Corners rather than an area, because there are two readers and they need
    different things from the same object. The manifest keeps areas only -- a
    box's position tells it nothing about the image -- while the oracle sells
    the boxes themselves, and a geometry reconstructed from an area is not the
    label. Keeping the corners means one parser serves both instead of the
    archive being read twice by two modules that can come to disagree about it.
    """

    category: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    """One label document, read once and served to both of its readers.

    `boxes` is the whole of what the archive says about the objects in an image.
    The manifest projects it down to areas and a category count; the oracle
    stores it as-is, since a box is what a cycle pays for.

    `degenerate_boxes` counts boxes dropped for enclosing no area. A zero-width
    box is not a box, and `ManifestRow` refuses a non-positive area, so they
    cannot simply be carried -- but dropping without counting would make an
    archive-wide oddity invisible.
    """

    image_id: ImageId
    weather: Weather
    scene: Scene
    timeofday: TimeOfDay
    boxes: tuple[Box, ...]
    degenerate_boxes: int

    @property
    def box_areas(self) -> tuple[float, ...]:
        """The manifest's projection. Parallel to `box_categories` by index."""
        return tuple(box.area for box in self.boxes)

    @property
    def box_categories(self) -> tuple[str, ...]:
        """Counted into the vocabulary report and stored nowhere per image.

        Naming the classes is `class_set_version`'s business downstream, not a
        fact the archive settles.
        """
        return tuple(box.category for box in self.boxes)


def _attributes(document: dict[str, Any], image_id: ImageId) -> dict[str, Any]:
    attributes = document.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError(f"{image_id}: label has no attributes object")
    return attributes


def _tag[Tag: StrEnum](
    attributes: dict[str, Any], key: str, vocabulary: type[Tag], image_id: ImageId
) -> Tag:
    """Read one of the three tag attributes into its vocabulary.

    Required to be present, because the archive writes an explicit `undefined`
    where a value is unknown -- so an absent key is a malformed document rather
    than a missing observation, and `undefined` is a member rather than a null.

    Unlisted values raise too. The vocabularies in `conventions` were counted
    over all 80,000 images, so a value outside one does not mean a member was
    overlooked; it means the host is serving data this code was not written
    against, which is worth failing an hour-long ingest over. Carrying it through
    would put a tag in the manifest that no eval slice and no mix row can
    ever match, and an empty slice charts as a flat line rather than as a gap.

    Generic over the vocabulary so each of the three returns its own type. A
    common `StrEnum` return would type-check a weather value into the scene
    column.
    """
    value = attributes.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{image_id}: attribute {key!r} is missing or not a string")
    try:
        return vocabulary(value)
    except ValueError:
        listed = ", ".join(member.value for member in vocabulary)
        raise ValueError(
            f"{image_id}: attribute {key!r} is {value!r}, which is not one of: {listed}"
        ) from None


def _objects(document: dict[str, Any], image_id: ImageId) -> list[Any]:
    """The one frame's objects.

    Exactly one frame is asserted rather than assumed. These are still images,
    so a second frame would mean either a tracking archive wearing the wrong
    name -- which is a mistake this host has already made twice -- or every box
    counted as many times as there are frames.
    """
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != 1:
        count = len(frames) if isinstance(frames, list) else "no"
        raise ValueError(f"{image_id}: expected exactly one frame, found {count}")

    frame = frames[0]
    if not isinstance(frame, dict):
        raise ValueError(f"{image_id}: frame is not a JSON object")

    objects = frame.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{image_id}: frame has no objects list")
    return objects


def _box(box: dict[str, Any], category: str, image_id: ImageId) -> Box:
    """One `box2d` object, corners carried through unrepaired.

    Native 1280x720 pixels, and the corners are taken in the order the archive
    states them rather than normalized so `x1 < x2`. A box whose corners are the
    wrong way round is not a box drawn backwards, it is a box we cannot
    interpret; sorting the corners here would launder that into a plausible
    rectangle. `Box.area` stays signed for the same reason, and `parse_label`
    drops a non-positive one as degenerate rather than fixing it.
    """
    for corner in _CORNERS:
        if not isinstance(box.get(corner), int | float):
            raise ValueError(f"{image_id}: box2d has no numeric {corner!r}")

    return Box(
        category=category,
        x1=float(box["x1"]),
        y1=float(box["y1"]),
        x2=float(box["x2"]),
        y2=float(box["y2"]),
    )


def parse_label(document: Any, image_id: ImageId) -> ParsedLabel:
    """Read one label document. Raises on anything unexpected."""
    if not isinstance(document, dict):
        raise ValueError(f"{image_id}: label is not a JSON object")

    # Checked when present rather than trusted: the file's own path is what
    # decides the image ID, so a `name` disagreeing with it means the archive
    # was assembled wrong and the boxes belong to some other picture.
    name = document.get("name")
    if isinstance(name, str) and PurePosixPath(name).stem != image_id:
        raise ValueError(f"{image_id}: label names a different image: {name!r}")

    attributes = _attributes(document, image_id)

    boxes: list[Box] = []
    degenerate = 0

    for obj in _objects(document, image_id):
        if not isinstance(obj, dict):
            raise ValueError(f"{image_id}: object is not a JSON object")

        box = obj.get("box2d")
        if box is None:
            continue  # an area/* or lane/* polygon
        if not isinstance(box, dict):
            raise ValueError(f"{image_id}: box2d is not a JSON object")

        category = obj.get("category")
        if not isinstance(category, str):
            raise ValueError(f"{image_id}: boxed object has no category")

        parsed = _box(box, category, image_id)
        if parsed.area <= 0:
            degenerate += 1
            continue

        boxes.append(parsed)

    return ParsedLabel(
        image_id=image_id,
        weather=_tag(attributes, "weather", Weather, image_id),
        scene=_tag(attributes, "scene", Scene, image_id),
        timeofday=_tag(attributes, "timeofday", TimeOfDay, image_id),
        boxes=tuple(boxes),
        degenerate_boxes=degenerate,
    )

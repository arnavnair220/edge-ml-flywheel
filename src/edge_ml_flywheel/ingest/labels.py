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
rather than asserted -- the same treatment `weather`, `scene` and `timeofday`
already get.

**Everything else is strict and raises.** A missing attribute, a second frame, a
label naming a different image: each of those means the archive is not what this
code was written against, and the one thing worse than failing an hour-long
ingest is completing it with a manifest nobody can trust.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from edge_ml_flywheel.conventions import ImageId

# The three the wave design runs on. Every one is required to be present: the
# archive uses an explicit "undefined" where a value is unknown, so an absent
# key is a malformed document rather than a missing observation.
ATTRIBUTES: Final = ("weather", "scene", "timeofday")

_CORNERS: Final = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    """What one label document contributes to a manifest row.

    `box_categories` is parallel to `box_areas` and exists only to be counted
    into the vocabulary report; nothing downstream stores it per image, because
    the class set is `class_set_version`'s business and not a fact about the
    archive.

    `degenerate_boxes` counts boxes dropped for enclosing no area. A zero-width
    box is not a box, and `ManifestRow` refuses a non-positive area, so they
    cannot simply be carried -- but dropping without counting would make an
    archive-wide oddity invisible.
    """

    image_id: ImageId
    weather: str
    scene: str
    timeofday: str
    box_areas: tuple[float, ...]
    box_categories: tuple[str, ...]
    degenerate_boxes: int


def _attributes(document: dict[str, Any], image_id: ImageId) -> dict[str, str]:
    attributes = document.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError(f"{image_id}: label has no attributes object")

    values: dict[str, str] = {}
    for key in ATTRIBUTES:
        value = attributes.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{image_id}: attribute {key!r} is missing or not a string")
        values[key] = value
    return values


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


def _area(box: dict[str, Any], image_id: ImageId) -> float:
    for corner in _CORNERS:
        if not isinstance(box.get(corner), int | float):
            raise ValueError(f"{image_id}: box2d has no numeric {corner!r}")

    # Native 1280x720 pixels, and signed rather than absolute. A box whose
    # corners are the wrong way round is not a box drawn backwards, it is a box
    # we cannot interpret, and repairing it here would launder that into a
    # plausible number.
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


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

    areas: list[float] = []
    categories: list[str] = []
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

        area = _area(box, image_id)
        if area <= 0:
            degenerate += 1
            continue

        areas.append(area)
        categories.append(category)

    return ParsedLabel(
        image_id=image_id,
        weather=attributes["weather"],
        scene=attributes["scene"],
        timeofday=attributes["timeofday"],
        box_areas=tuple(areas),
        box_categories=tuple(categories),
        degenerate_boxes=degenerate,
    )

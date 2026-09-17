"""A list of images for a SageMaker channel, in `ManifestFile` form.

The labels are a key range and the images are not. They sit flat under a split's
image prefix among tens of thousands, and every set this project puts in front of
a model -- the cumulative labeled set, the eval cohort, the remaining pool -- is
a scattered subset of one of those prefixes. An `S3Prefix` channel would take the
whole prefix, so a manifest is the only way to name the subset.

**One format, two planes.** Training writes one of these and scoring writes one
per cohort, which is why `split` is an argument rather than the constant it was
while training was the only caller. It is not defaulted: a manifest is a list of
keys under one prefix, and a default is how the wrong prefix gets named by a
caller that had no opinion. `training` passes `Split.TRAIN` because bootstrap and
pool are both `train` cohorts and a training set never crosses a split; `scoring`
derives the split from the cohort, which is what makes its two manifests two
documents.

**The document is also the record of what ran.** It is written once at prepare
time, read by every seed of that cycle, and lives under the write-once cycle
prefix -- so the training set of cycle six is a file rather than a set someone
reconstructs later from a ledger and a partition. That is what makes `max_images`
a parameter here rather than in the container: a short skeleton run is a short
manifest, and the record still says exactly what was used.

**Both sides of the cap live here**, and that is the point. `capped` names the
images for the manifest; `capped_labels` drops the labels for everything the
manifest left out, which the container needs because labels arrive on whole
prefixes no cap can be expressed on. Two resolutions of one decision, from one
sort, in a module that runs on a laptop -- the container is a caller, so the
arithmetic is not stranded behind an `ultralytics` import.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    RAW_IMAGES_PREFIX,
    ImageId,
    Split,
    parse_image_id,
    raw_image_key,
    uri,
)
from edge_ml_flywheel.ingest.labels import Box

log = logging.getLogger(__name__)

# The prefix header plus at least one key. A manifest is a list whose first
# element names the prefix the rest are relative to, so a document of one entry
# names nothing.
_MIN_ENTRIES: Final = 2


def prefix_of(split: Split) -> str:
    """The one prefix every entry of a split's manifest sits below."""
    return f"{RAW_IMAGES_PREFIX}{split.value}/"


def document(bucket: str, image_ids: Sequence[ImageId], split: Split) -> list[Any]:
    """`[{"prefix": <s3 uri>}, <key>, <key>, ...]`, sorted and deduplicated.

    Each entry is derived from `raw_image_key` and then had the shared prefix
    removed, rather than being formatted here as `<id>.jpg`. The two spellings
    agree today; deriving means they cannot stop agreeing, which matters because
    a manifest naming keys that do not exist fails the job with a download error
    that says nothing about why.
    """
    prefix = prefix_of(split)
    ordered = sorted(set(image_ids))
    if not ordered:
        raise ValueError("a manifest naming no images is not a manifest")

    entries: list[Any] = [{"prefix": uri(bucket, prefix)}]
    for image_id in ordered:
        key = raw_image_key(image_id, split)
        if not key.startswith(prefix):
            raise ValueError(f"{key} is not under the one prefix a manifest can name: {prefix}")
        entries.append(key.removeprefix(prefix))
    return entries


def write(path: Path, bucket: str, image_ids: Sequence[ImageId], split: Split) -> int:
    """Write the manifest and return how many images it names."""
    entries = document(bucket, image_ids, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")

    named = len(entries) - 1
    log.info("manifest names %d images under %s", named, uri(bucket, prefix_of(split)))
    return named


def manifest_ids(path: Path) -> tuple[ImageId, ...]:
    """The images a written manifest names, read back out of the document.

    The inverse of `document`, and the evaluation job's statement of which images
    were scored. It cannot read that off the detections instead: a model that
    found nothing in a frame contributes no row, so the detections name the images
    with boxes rather than the images that were put in front of the model -- and
    the difference is every frame the challenger missed entirely, which is exactly
    what the metric has to count.

    The leading `{"prefix": ...}` entry is dropped rather than checked against a
    split. A manifest is a list of keys under one prefix whatever that prefix is,
    and this reader's caller already knows which cohort it asked for.
    """
    entries = json.loads(path.read_text(encoding="utf-8"))
    # A list of one is the prefix header and no images, which `document` refuses
    # to write in the first place -- so reaching here means the file was
    # truncated or produced by something else.
    if not isinstance(entries, list) or len(entries) < _MIN_ENTRIES:
        raise ValueError(f"{path.name} is not a manifest naming any image")

    found = tuple(parse_image_id(Path(str(entry)).stem) for entry in entries[1:])
    log.info("manifest %s names %d images", path.name, len(found))
    return found


def capped(image_ids: Sequence[ImageId], max_images: int) -> tuple[ImageId, ...]:
    """The first `max_images` of a labeled set, or all of them at 0.

    Sorted and taken from the front rather than sampled, because the cap exists
    to make the skeleton run cost cents and not to produce a representative
    subset -- and a seeded sample would be a second seed in a project where
    `seed` already means one thing. BDD100K image IDs are hashes, so the front of
    the sort is not a slice of anything: it is arbitrary in the way a sample
    would be, without the second parameter.
    """
    if max_images < 0:
        raise ValueError(f"max images cannot be negative: {max_images}")
    ordered = tuple(sorted(set(image_ids)))
    if max_images == 0 or max_images >= len(ordered):
        return ordered

    log.info("capping the training set at %d of %d labeled images", max_images, len(ordered))
    return ordered[:max_images]


def capped_labels(
    labeled: Mapping[ImageId, Sequence[Box]], max_images: int
) -> Mapping[ImageId, Sequence[Box]]:
    """The labels for exactly the images `capped` would name, or all of them at 0.

    The container's half of the cap. `prepare` applies `capped` to the labeled
    set and writes the result as the manifest; the labels themselves arrive on
    whole prefixes -- the bootstrap file and every purchase, appended to and
    never rewritten -- so the same cap has to be re-derived where they are read.

    Re-derived and not communicated, because both sides start from the same
    labeled set and `capped` is a sort and a slice: the two resolve the identical
    images without a second document to keep in step. If they ever did diverge,
    `dataset.write` refuses the pair rather than training on the overlap.
    """
    kept = set(capped(tuple(labeled), max_images))
    if len(kept) == len(labeled):
        return labeled
    return {image_id: boxes for image_id, boxes in labeled.items() if image_id in kept}

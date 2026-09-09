"""The list of images one cycle trains on, in SageMaker's `ManifestFile` form.

The labels are a key range and the images are not. They sit flat under the
`train` image prefix among all 70,000, and the cumulative labeled set is a
scattered subset of those, so an `S3Prefix` channel would take the whole prefix
and a manifest is the only way to name the subset.

**This document is also the record of what the challenger trained on.** It is
written once at prepare time, read by all five seeds, and lives under the
write-once cycle prefix -- so the training set of cycle six is a file rather than
a set someone reconstructs later from a ledger and a partition. That is what
makes `max_images` a parameter here rather than in the container: a short
skeleton run is a short manifest, and the record still says exactly what was
trained on.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    RAW_IMAGES_PREFIX,
    ImageId,
    Split,
    raw_image_key,
    uri,
)

log = logging.getLogger(__name__)

# The split every training image is drawn from. `bootstrap` and `pool` are both
# `train` cohorts, so a training set never crosses a split -- which is what lets
# every entry share one prefix, as the manifest format requires.
_SPLIT: Final = Split.TRAIN

_PREFIX: Final = f"{RAW_IMAGES_PREFIX}{_SPLIT.value}/"


def document(bucket: str, image_ids: Sequence[ImageId]) -> list[Any]:
    """`[{"prefix": <s3 uri>}, <key>, <key>, ...]`, sorted and deduplicated.

    Each entry is derived from `raw_image_key` and then had the shared prefix
    removed, rather than being formatted here as `<id>.jpg`. The two spellings
    agree today; deriving means they cannot stop agreeing, which matters because
    a manifest naming keys that do not exist fails the job with a download error
    that says nothing about why.
    """
    ordered = sorted(set(image_ids))
    if not ordered:
        raise ValueError("a cycle with no images to train on is not a cycle")

    entries: list[Any] = [{"prefix": uri(bucket, _PREFIX)}]
    for image_id in ordered:
        key = raw_image_key(image_id, _SPLIT)
        if not key.startswith(_PREFIX):
            raise ValueError(f"{key} is not under the one prefix a manifest can name: {_PREFIX}")
        entries.append(key.removeprefix(_PREFIX))
    return entries


def write(path: Path, bucket: str, image_ids: Sequence[ImageId]) -> int:
    """Write the manifest and return how many images it names."""
    entries = document(bucket, image_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")

    named = len(entries) - 1
    log.info("manifest names %d images under %s", named, uri(bucket, _PREFIX))
    return named


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

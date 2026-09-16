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
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from edge_ml_flywheel.conventions import (
    RAW_IMAGES_PREFIX,
    ImageId,
    Split,
    raw_image_key,
    uri,
)

log = logging.getLogger(__name__)


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

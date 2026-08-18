"""Move the extracted archives to the exact keys they will have in S3.

The staging directory is not a working copy of the data -- it is the bucket,
laid out locally. Every path under it is `stage_dir / <key>` for a key built by
`edge_ml_flywheel.conventions`, which is what lets the upload be a single
recursive copy with no key formatting anywhere in the buildspec. A key spelled
in a shell script is a key that drifts from the one the readers use, and the
failure is the silent kind: the writer succeeds and the reader finds nothing.

Moving rather than copying, because both trees are on the same filesystem and a
rename is free where a second copy of 4.6 GB is not.

**What this refuses, and what it merely reports.** A file under a `test`
directory is a hard failure with nothing staged, because the withheld split is
the leakage guard and a leakage guard that can be downgraded to a warning is not
one. Everything else -- a count that is off, an image with no label -- is
reported and left to the verification suite, which runs before any upload and is
where the archive's expected shape is written down.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import (
    ImageId,
    Split,
    parse_image_id,
    raw_image_key,
    raw_label_key,
)

# Both archives lay the pool out as `100k/<split>/<file>`, with no top-level
# directory, so the two modalities merge into one tree on extract. Only the
# trailing three components are read, so a repackaged mirror that wraps the
# pool in a directory stages identically.
_POOL_DIR: Final = "100k"

# Not a `Split` member, and that is the point: `Split` has no `TEST`, so the
# name of the thing being excluded has to be written here or nowhere.
WITHHELD_SPLIT: Final = "test"

_SPLIT_BY_NAME: Final = {split.value: split for split in Split}

# `100k/<split>/<file>`: the three trailing components a pool path must have.
_POOL_PATH_DEPTH: Final = 3

# `raw_image_key` and `raw_label_key` share a signature, which is what lets both
# modalities go through one loop instead of two near-identical ones.
type _KeyBuilder = Callable[[ImageId, Split], str]


@dataclass(frozen=True, slots=True)
class StagedPool:
    """What ended up staged, per modality and split.

    Two ID sets rather than one, because their *equality* is the check that
    validates the images archive -- the one with no published digest. A
    truncated image download shows up here and nowhere else.
    """

    images: Mapping[Split, frozenset[ImageId]]
    labels: Mapping[Split, frozenset[ImageId]]
    ignored: tuple[str, ...]

    @property
    def image_count(self) -> int:
        return sum(len(ids) for ids in self.images.values())

    @property
    def label_count(self) -> int:
        return sum(len(ids) for ids in self.labels.values())


def _split_of(path: Path) -> Split:
    """The split a `.../100k/<split>/<file>` path names.

    Raises on a test-split path and on any layout this code was not written
    against, rather than skipping either. A file quietly skipped here is a file
    missing from `raw/` that nothing downstream can tell apart from an image the
    archive never had.
    """
    parts = path.parts
    if len(parts) < _POOL_PATH_DEPTH or parts[-_POOL_PATH_DEPTH] != _POOL_DIR:
        raise ValueError(f"not a 100k/<split>/<file> path: {path}")

    name = parts[-2]
    if name == WITHHELD_SPLIT:
        raise ValueError(
            f"a withheld test-split file reached the extract, which the unzip "
            f"exclusion should have prevented: {path}"
        )

    split = _SPLIT_BY_NAME.get(name)
    if split is None:
        raise ValueError(f"unknown split {name!r} in {path}")
    return split


def stage(extract_dir: Path, stage_dir: Path) -> StagedPool:
    """Move every image and label into `stage_dir` at its S3 key."""
    images: dict[Split, set[ImageId]] = {split: set() for split in Split}
    labels: dict[Split, set[ImageId]] = {split: set() for split in Split}
    ignored: list[str] = []
    made: set[Path] = set()

    modalities: tuple[tuple[str, dict[Split, set[ImageId]], _KeyBuilder], ...] = (
        ("*.jpg", images, raw_image_key),
        ("*.json", labels, raw_label_key),
    )

    for pattern, found, build_key in modalities:
        for path in sorted(extract_dir.rglob(pattern)):
            # Anything the archives ship outside the pool layout -- a README, a
            # stray manifest -- is recorded and left where it is rather than
            # guessed at.
            if _POOL_DIR not in path.parts:
                ignored.append(path.relative_to(extract_dir).as_posix())
                continue

            split = _split_of(path)
            image_id = parse_image_id(path.stem)

            destination = stage_dir / build_key(image_id, split)
            if destination.parent not in made:
                destination.parent.mkdir(parents=True, exist_ok=True)
                made.add(destination.parent)

            path.replace(destination)
            found[split].add(image_id)

    return StagedPool(
        images={split: frozenset(ids) for split, ids in images.items()},
        labels={split: frozenset(ids) for split, ids in labels.items()},
        ignored=tuple(ignored),
    )

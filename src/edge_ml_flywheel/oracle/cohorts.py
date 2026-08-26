"""The cohort gate: what a run is allowed to buy, and the refusal of everything else.

The oracle reads ground truth straight out of `raw/labels/`, so nothing about the
storage layout distinguishes a `pool` label from an `eval` one -- they are files
in sibling prefixes, told apart only by the assignments parquet the partitioner
wrote. This module is where that distinction is enforced, and it is therefore the
single place the eval guarantee lives.

**Every purchase passes through `check_purchasable` before a key is built.** Not
after the read and not alongside it: the refusal happens while the request is
still a list of image IDs, so a rejected purchase never causes a label file to be
opened at all. The ordering is the difference between "we did not sell it" and
"we did not read it", and only the second is worth anything if the process is
later asked what it saw.

**A cohort index is a fact about a partition, not about a run.** Two runs over
one `partition_version` are gated by identical answers, which is why this is
loaded from `assignments/` rather than copied per run: the partition is already
frozen and checksummed, so a second copy would be a second thing to keep true.

**Refusals name the cohort, not just the ID.** "Image X is not purchasable" sends
whoever is debugging to look for a typo; "image X is in `eval`" says the
selector handed the oracle something it should never have seen, which is a
different bug in a different component.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import pyarrow.parquet as pq

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    Cohort,
    ImageId,
    PartitionVersion,
    Split,
    assignments_prefix,
    parse_image_id,
)

log = logging.getLogger(__name__)

# The one cohort a cycle may buy from. `bootstrap` is already owned, `eval` is
# the ruler, and `reserve` is deliberately inert -- so all three are refused, and
# for three different reasons that `check_purchasable` reports separately.
PURCHASABLE: Final = Cohort.POOL

# Which split a purchasable label is read from. Derived rather than written down:
# a key the oracle builds can then only ever address the split `COHORT_SPLIT`
# says `pool` draws from, so `raw/labels/scalabel/val/` -- where `eval` lives --
# is not a path this module is able to construct.
PURCHASABLE_SPLIT: Final = COHORT_SPLIT[PURCHASABLE]

_COLUMNS: Final = ("image_id", "cohort")

# How many offending IDs a refusal lists before summarizing. Enough to see a
# pattern -- one stray ID reads differently from nine hundred -- without putting
# a whole batch in an exception message.
_REPORTED: Final = 5


class NotPurchasableError(Exception):
    """A purchase named an image that is not in the pool.

    Its own type because the callers want different things from it. A retry of a
    purchase wants to distinguish this from a throttle or a network failure, and
    the selector -- the component that produced the list -- wants a failure loud
    enough that a run stops rather than quietly buying 999 of the 1,000 images it
    asked for.
    """


@dataclass(frozen=True, slots=True)
class Cohorts:
    """Which cohort each of the 80,000 images was drawn into.

    Read once and held: 80,000 short strings is a few megabytes, and the
    alternative is re-reading a parquet on every purchase to answer the same
    frozen question.
    """

    partition_version: PartitionVersion
    of_image: Mapping[ImageId, Cohort]

    @classmethod
    def read(cls, root: Path, partition_version: PartitionVersion) -> Self:
        """Load the assignments parquet from a tree laid out at its S3 keys.

        Every part file, because the writer is free to emit more than one and a
        reader that took only `part-00000` would silently gate against a subset
        -- refusing legitimate pool images and, far worse, failing to recognise
        an eval one.
        """
        directory = root / assignments_prefix(partition_version)
        parts = sorted(directory.glob("part-*.parquet"))
        if not parts:
            raise ValueError(f"no assignments parquet under {directory}")

        found: dict[ImageId, Cohort] = {}
        for part in parts:
            table = pq.read_table(part, columns=list(_COLUMNS))
            for value, cohort in zip(
                table.column("image_id").to_pylist(),
                table.column("cohort").to_pylist(),
                strict=True,
            ):
                image_id = parse_image_id(value)
                if image_id in found:
                    raise ValueError(f"{image_id} is assigned a cohort twice")
                found[image_id] = Cohort(cohort)

        log.info("cohort index for partition v%d: %d images", partition_version, len(found))
        return cls(partition_version=partition_version, of_image=found)

    def cohort_of(self, image_id: ImageId) -> Cohort | None:
        """`None` for an image the partition never assigned.

        Distinct from every cohort rather than folded into one, because an
        unknown ID is a different failure: a known ID in the wrong cohort means
        the selector picked badly, and an unknown one means it is not working
        from this partition at all.
        """
        return self.of_image.get(image_id)

    def in_cohort(self, cohort: Cohort) -> frozenset[ImageId]:
        return frozenset(image for image, drawn in self.of_image.items() if drawn is cohort)

    @property
    def pool(self) -> frozenset[ImageId]:
        return self.in_cohort(PURCHASABLE)


def refusals(cohorts: Cohorts, image_ids: Iterable[ImageId]) -> dict[ImageId, str]:
    """Every image in the request that is not purchasable, with why.

    Separate from the raise so the same judgement can be reported without
    stopping -- a dry run, or a log line counting what a selector proposed --
    and so the reason strings are testable without catching an exception.
    """
    found: dict[ImageId, str] = {}
    for image_id in image_ids:
        cohort = cohorts.cohort_of(image_id)
        if cohort is PURCHASABLE:
            continue
        found[image_id] = (
            f"in {cohort.value}"
            if cohort is not None
            else f"not assigned by partition v{cohorts.partition_version}"
        )
    return found


def check_purchasable(cohorts: Cohorts, image_ids: Sequence[ImageId]) -> None:
    """Refuse the whole batch if any image in it is not in the pool.

    All or nothing, deliberately. Dropping the offenders and selling the rest
    would charge the run for a batch it did not ask for, and would turn the loud
    version of this failure -- a selector that reached into `eval` -- into a
    cycle that bought slightly fewer labels than usual and reported nothing.

    Duplicates are refused too. A repeated ID is a single label billed twice,
    which the ledger cannot detect: the batch is a set of images to the oracle
    and a count to the budget, and this is the one place those two views meet.
    """
    if not image_ids:
        raise NotPurchasableError("a purchase of no images is not a purchase")

    counted = Counter(image_ids)
    duplicated = sorted(image for image, times in counted.items() if times > 1)
    if duplicated:
        raise NotPurchasableError(
            f"{len(duplicated)} image(s) appear more than once in one batch, which the budget "
            f"would charge for twice: {duplicated[:_REPORTED]}"
        )

    wrong = refusals(cohorts, image_ids)
    if wrong:
        listed = ", ".join(
            f"{image} ({reason})" for image, reason in sorted(wrong.items())[:_REPORTED]
        )
        raise NotPurchasableError(
            f"{len(wrong)} of {len(image_ids)} images are not in {PURCHASABLE.value}: {listed}"
            + (f", and {len(wrong) - _REPORTED} more" if len(wrong) > _REPORTED else "")
        )


def purchasable_split(cohorts: Cohorts, image_id: ImageId) -> Split:
    """The split a purchasable image's label is read from.

    Takes the image and re-checks it rather than returning a constant, so that a
    caller cannot get a usable path out of this module for an image the gate
    would have refused. The only split it ever returns is `PURCHASABLE_SPLIT`;
    what varies is whether it returns at all.
    """
    check_purchasable(cohorts, [image_id])
    return PURCHASABLE_SPLIT

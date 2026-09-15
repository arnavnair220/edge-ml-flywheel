"""What a cycle bought, recorded beside what it left behind.

The ranking is bought unfiltered, so this is the only thing standing between a
degenerate batch and a null result nobody can explain. A cycle that gains nothing
over the random control is either a ranking that does not work or a batch that
was a thousand copies of one scene, and the two call for opposite responses
(design section 2). One is visible here the cycle it happens; the other is not
visible anywhere else.

**The remaining pool is the reference, because no better one exists.** Condition
tags come from the vehicle, so they are known for images nobody has labeled -- but
a true population proportion is not. A batch whose mix tracks the pool's says
redundancy is not the problem; a batch collapsed onto one condition says it is.

**Every value in a vocabulary appears, including the zeros.** A count missing
from a mapping and a count of zero read identically in a chart -- as nothing --
and they are opposite facts. `foggy: 0` in a batch is the observation; `foggy`
absent is a report that cannot be distinguished from one where the column was
never computed.

**Predicted classes are a proxy, and are recorded as one.** The data gate fails a
cycle whose batch carries under ten new boxes of any class, and the labels stay
bought (design section 4.1). Selection cannot see that coming -- boxes are ground
truth and sit behind the label wall -- but the champion's own detections carry a
category, and they are the only advance signal there is. At the archive's
frequencies a random batch of a thousand clears that floor many times over, so
this is a tripwire rather than a routine constraint, and it is here to say which
way to move if it ever trips.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from edge_ml_flywheel.conventions import (
    ImageId,
    ManifestRow,
    TimeOfDay,
    Weather,
)
from edge_ml_flywheel.selection.score import Predictions


def _counts(values: Sequence[str], vocabulary: Sequence[str]) -> dict[str, int]:
    """Tally over a fixed vocabulary, so absence and zero stay distinguishable."""
    counted = Counter(values)
    return {member: counted[member] for member in vocabulary}


@dataclass(frozen=True, slots=True)
class Mix:
    """One image set's condition composition.

    `weather` and `timeofday` only. `scene` is a tag the manifest carries and the
    eval slices use, but the design's condition record names these two (design
    section 2) and a third column here would be a third thing to read without a
    question waiting on it.
    """

    images: int
    weather: Mapping[str, int]
    timeofday: Mapping[str, int]

    @classmethod
    def of(cls, image_ids: Sequence[ImageId], manifest: Mapping[ImageId, ManifestRow]) -> "Mix":
        missing = sorted(image_id for image_id in image_ids if image_id not in manifest)
        if missing:
            raise ValueError(
                f"{len(missing)} image(s) have no manifest row, so their conditions would count as "
                f"absent rather than as unknown: {missing[:5]}"
            )
        rows = [manifest[image_id] for image_id in image_ids]
        return cls(
            images=len(rows),
            weather=_counts(
                [row.weather.value for row in rows], [member.value for member in Weather]
            ),
            timeofday=_counts(
                [row.timeofday.value for row in rows], [member.value for member in TimeOfDay]
            ),
        )

    def document(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "weather": dict(self.weather),
            "timeofday": dict(self.timeofday),
        }


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """The record one cycle's selection leaves behind, written once per cycle.

    `blind_spots` is the count of images in the batch the champion produced no
    detections for. Those are ranked to the top of the uncertainty ordering
    deliberately -- a model that sees nothing is the case worth labeling -- but
    nothing yet says how many of them a batch should be allowed to be. A batch
    that is nine hundred blind spots and a batch that is nine are the same
    ranking rule with very different consequences, and this is the number that
    tells them apart before a training run does.
    """

    batch: Mix
    remaining: Mix
    predicted_classes: Mapping[str, int]
    blind_spots: int

    def document(self) -> dict[str, Any]:
        return {
            "batch": self.batch.document(),
            "remaining": self.remaining.document(),
            "predicted_classes": dict(self.predicted_classes),
            "blind_spots": self.blind_spots,
        }

    def as_json(self) -> str:
        return json.dumps(self.document(), indent=2, sort_keys=True)


def selection_report(
    batch: Sequence[ImageId],
    pool: Sequence[ImageId],
    manifest: Mapping[ImageId, ManifestRow],
    predictions: Predictions,
) -> SelectionReport:
    """Compose the record. `pool` is the pool as it stood *before* this batch.

    Taking the whole pool and subtracting, rather than taking the remainder
    ready-made, so the two halves of the comparison cannot be built from
    different sets -- which is the one way this report could mislead rather than
    simply be wrong.
    """
    chosen = set(batch)
    outside = sorted(chosen - set(pool))
    if outside:
        raise ValueError(
            f"{len(outside)} image(s) in the batch are not in the pool it was drawn from: "
            f"{outside[:5]}"
        )
    remaining = [image_id for image_id in pool if image_id not in chosen]

    detections = [
        detection.category for image_id in batch for detection in predictions.for_image(image_id)
    ]
    return SelectionReport(
        batch=Mix.of(list(batch), manifest),
        remaining=Mix.of(remaining, manifest),
        predicted_classes=_counts(detections, predictions.classes.names),
        blind_spots=sum(1 for image_id in batch if not predictions.for_image(image_id)),
    )

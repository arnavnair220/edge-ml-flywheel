"""The caller's side of selection: rank what the fleet reported, write what was
chosen.

`scoring.launch`'s shape for the step that consumes what a pass produced, and it
reuses `training.launch`'s account helpers rather than restating them -- the
session, the bucket names, the run registration and the S3 primitives are the
same facts about the same account.

**There is no job here.** Selection is arithmetic over a file that already
exists, so this runs in the control Lambda rather than on an instance: the pass
over the sample happened on the device, and what is left is a mean per image and
a sort.

**Nothing here reads a label.** The detections are predictions, the sample is a
list of image IDs, and the ranking is a function of the two. That is what keeps
selection inside the label wall by construction -- the control plane is denied
`raw/labels/` outright, and this step has no reason to want it.

**The model that ranks is the one the fleet is running.** Not by policy: the
detections come from a device, and the only model a device has is the deployed
champion. A cycle that promoted ranks with its challenger because that
challenger was deployed a state earlier; a cycle that rejected one ranks with the
champion still installed. Either way the labels stay bought and the ranking is a
statement about a model that really ran on real hardware.
"""

import json
import logging
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import boto3

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    Cohort,
    Cycle,
    ImageId,
    ModelVersion,
    Precision,
    RunId,
    Seed,
    detections_prefix,
    model_version_cycle,
    model_version_run_id,
    parse_image_id,
    replay_manifest_key,
    selection_ranking_key,
    uri,
)
from edge_ml_flywheel.scoring import detections
from edge_ml_flywheel.selection import ranking
from edge_ml_flywheel.selection.score import BAND_LOW, Predictions, score_pool
from edge_ml_flywheel.selection.select import by_uncertainty, select
from edge_ml_flywheel.training import launch as base

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Ranked:
    """What one cycle's selection decided, for the state machine to act on.

    `blind_spots` is how many images in the batch the model saw nothing in -- see
    `selection.score.BLIND_SPOT`. Returned rather than only logged because it is
    the number that says whether the ranking is working or buying a thousand
    frames the detector simply failed on, and the execution history is where a
    cycle's decisions are already recorded.

    `pool` is what was ranked, which is the remaining pool this cycle scored. The
    purchase subtracts the batch from it to say what is left.
    """

    version: ModelVersion
    cycle: Cycle
    pool: int
    batch: int
    blind_spots: int


def rank(
    aws: boto3.Session,
    version: ModelVersion,
    seed: Seed,
    cycle: Cycle | None = None,
    replace: bool = False,
) -> Ranked:
    """Rank this cycle's sample and write the file the purchase is charged against.

    **`version` is the model that scored, `cycle` is the cycle being decided, and
    they are not always the same cycle.** A cycle that promoted ranks with its
    own challenger and the two agree. A cycle that rejected one ranks with the
    champion still deployed, which was trained earlier -- so the detections are
    keyed by that older version under *this* cycle's prefix, and the ranking is
    written under this cycle. Omitting `cycle` means the model's own, which is
    the promoting case.

    The order is the order the failures are worth having in. The refusal to
    overwrite comes first, before anything is downloaded, so a cycle that has
    already selected costs a listing rather than a copy of its detections. The
    manifest is read before the detections, because it is the set being ranked and
    a missing one means the cycle never scored.

    The whole ordering and the batch come from two calls to one rule. `select`
    applies the guards that are about the call -- a budget has to buy something
    and cannot buy more than is left -- and `by_uncertainty` over the whole pool
    produces the order the file is written in. The batch is that order's prefix by
    construction rather than by a sort repeated here, which is the one way the
    record and the purchase could disagree about what was chosen.

    `replace` permits rather than supplies, for `training.launch.Preparation`'s
    reason and one more: this file is what the oracle charges against, so
    rewriting it after a purchase has been made leaves the ledger pointing at a
    batch the record no longer names.
    """
    run_id = model_version_run_id(version)
    if cycle is None:
        cycle = model_version_cycle(version)
    run = base.registration(aws, run_id)

    artifacts = base.buckets(aws).artifacts
    key = selection_ranking_key(run_id, cycle)
    if not replace and base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} already exists, and it is the record this cycle's purchase is "
            f"charged against. Pass --replace only if nothing has been bought against it."
        )

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        pool = _ranked_pool(aws, artifacts, run_id, cycle, work)
        predictions = _predictions(aws, version, seed, cycle, work)

        scores = score_pool(pool, predictions)
        order = by_uncertainty(pool, scores, len(pool))
        batch = select(pool, scores, run.label_budget_per_cycle)

        local = work / Path(key).name
        ranking.write(ranking.rows(order, scores, batch), local)
        aws.client("s3").upload_file(str(local), artifacts, key)

    log.info("wrote %s", uri(artifacts, key))
    return _report(version, cycle, pool, batch, predictions)


def _ranked_pool(
    aws: boto3.Session, artifacts: str, run_id: RunId, cycle: Cycle, work: Path
) -> tuple[ImageId, ...]:
    """The images this cycle ranks: the sample the device was given.

    The sample rather than the detections, and the difference is every blind
    spot. A frame the model found nothing in contributes no row to the parquet,
    so the detections name the images with boxes while this names the images that
    were put in front of the model -- and the frames in the gap are the ones
    `score_pool` ranks at the top.

    A JSON array rather than a SageMaker `ManifestFile`, because the reader that
    matters is the device: it resolves each ID through `raw_image_key` under its
    own grant, and the envelope a Processing channel wants would be an envelope
    the component has to strip.
    """
    key = replay_manifest_key(run_id, cycle)
    if not base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} does not exist, so nothing says which images this cycle "
            f"put in front of the fleet. Run score prepare for this cycle."
        )

    local = work / Path(key).name
    aws.client("s3").download_file(artifacts, key, str(local))
    document = json.loads(local.read_text())
    if not isinstance(document, list) or not document:
        raise SystemExit(f"{uri(artifacts, key)} is not a non-empty list of image IDs")
    return tuple(parse_image_id(str(value)) for value in document)


def _predictions(
    aws: boto3.Session,
    version: ModelVersion,
    seed: Seed,
    cycle: Cycle,
    work: Path,
) -> Predictions:
    """What the fleet saw in the sample, above the floor the ranking reads.

    `precision=INT8`, and that is the whole point of the arrangement: the only
    model that scores the pool is the quantized graph deployed to a device, so
    the uncertainty a label is bought on is the uncertainty of the model actually
    driving. The fp32 prefix holds `eval` alone.

    `BAND_LOW` is still passed as a floor even though the device already writes
    at it. Applying it here costs nothing on a file that has none below the line,
    and it keeps this reader correct against a file written by something that did
    emit the tail -- a re-run of the old cloud pass, or a device on an older
    component version. See `scoring.detections.read`.
    """
    artifacts = base.buckets(aws).artifacts
    prefix = detections_prefix(version, seed, Cohort.POOL, Precision.INT8, cycle)
    root = work / "detections"

    found = base.download_prefix(aws, artifacts, prefix, root)
    if not found:
        raise SystemExit(
            f"{uri(artifacts, prefix)} holds no detections, so no device has scored this cycle's "
            f"sample with seed {seed} of {version}. The fleet pass runs after promotion."
        )

    of_image = detections.grouped(root, BAND_LOW)
    log.info("%d pool images carry a detection at or above %.2f", len(of_image), BAND_LOW)
    return Predictions(classes=CLASS_SET, of_image=of_image)


def _report(
    version: ModelVersion,
    cycle: Cycle,
    pool: tuple[ImageId, ...],
    batch: tuple[ImageId, ...],
    predictions: Predictions,
) -> Ranked:
    """Count what the batch is made of, and log it.

    Logged rather than written, unlike the ranking. Both numbers here are proxies
    -- a blind-spot count, and the categories the model *predicted* rather than
    the ones the batch turns out to contain -- and the condition mix that would be
    the real record is a join of the ranking onto the image manifest. That is a
    query over two files already in the bucket rather than a third file, and the
    manifest is in a codec this runtime cannot open anyway. See
    `conventions.selection_report_key`.

    Every class appears, including the zeros, for `selection.mix`'s reason: a
    count missing from a log line and a count of zero read identically and mean
    opposite things.
    """
    blind_spots = sum(1 for image_id in batch if not predictions.for_image(image_id))
    counted = Counter(
        detection.category for image_id in batch for detection in predictions.for_image(image_id)
    )

    log.info(
        "%s cycle %d: bought %d of %d ranked, %d of them blind spots",
        version,
        cycle,
        len(batch),
        len(pool),
        blind_spots,
    )
    log.info(
        "predicted classes in the batch: %s",
        ", ".join(f"{name} {counted[name]:,}" for name in predictions.classes.names),
    )
    return Ranked(
        version=version,
        cycle=cycle,
        pool=len(pool),
        batch=len(batch),
        blind_spots=blind_spots,
    )

"""The caller's side of selection: score the pool, rank it, write what was chosen.

`scoring.launch`'s shape for the step that consumes what that job produced, and
it reuses `training.launch`'s account helpers rather than restating them -- the
session, the bucket names, the run registration and the S3 primitives are the
same facts about the same account.

**There is no job here.** Selection is arithmetic over a file the scoring job
already wrote, so this runs in the control Lambda rather than on an instance: the
expensive pass over 62,000 images happened upstream, and what is left is a mean
per image and a sort.

**Nothing here reads a label.** The detections are predictions, the manifest is a
list of image IDs, and the ranking is a function of the two. That is what keeps
selection inside the label wall by construction -- the control plane is denied
`raw/labels/` outright, and this step has no reason to want it.

**The model that ranks is the one this cycle just scored.** Not the champion: the
pool manifest this cycle wrote is exactly what that model was run over, so the
scores cover the set being ranked with nothing to reconcile. A rejected
challenger still ranks, which is the rule the rest of the cycle already follows
-- the labels stay bought and the champion stays put.
"""

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
    RunId,
    Seed,
    detections_prefix,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
    selection_ranking_key,
    uri,
)
from edge_ml_flywheel.scoring import detections
from edge_ml_flywheel.selection import ranking
from edge_ml_flywheel.selection.score import BAND_LOW, Predictions, score_pool
from edge_ml_flywheel.selection.select import by_uncertainty, select
from edge_ml_flywheel.training import images
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
    replace: bool = False,
) -> Ranked:
    """Rank this cycle's pool and write the file the purchase is charged against.

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
        predictions = _predictions(aws, artifacts, version, seed, work)

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
    """The images this cycle ranks, read off the manifest that was scored.

    The manifest rather than the detections, for `training.images.manifest_ids`'
    reason: a model that found nothing in a frame contributes no row, so the
    detections name the images with boxes and the manifest names the images that
    were put in front of the model. The difference is every blind spot, which is
    the top of the ranking.
    """
    key = scoring_manifest_key(run_id, cycle, Cohort.POOL)
    if not base.exists(aws, artifacts, key):
        raise SystemExit(
            f"{uri(artifacts, key)} does not exist, so nothing says which images this cycle "
            f"ranked. Run score prepare for this cycle."
        )

    local = work / Path(key).name
    aws.client("s3").download_file(artifacts, key, str(local))
    return images.manifest_ids(local)


def _predictions(
    aws: boto3.Session, artifacts: str, version: ModelVersion, seed: Seed, work: Path
) -> Predictions:
    """What the model saw in the pool, above the floor the ranking reads.

    `BAND_LOW` is passed as a floor rather than applied afterwards, because the
    file is millions of rows at the scoring job's 0.001 confidence floor and none
    below 0.05 change a score. See `scoring.detections.read`.
    """
    prefix = detections_prefix(version, seed, Cohort.POOL)
    root = work / "detections"

    found = base.download_prefix(aws, artifacts, prefix, root)
    if not found:
        raise SystemExit(
            f"{uri(artifacts, prefix)} holds no detections, so seed {seed} of {version} was never "
            f"scored over the pool. Run the scoring job for this seed."
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

"""The evaluation job itself, as it runs inside the container.

Started by `container/evaluate.py`, which is a three-line file at the root of the
source archive for `container/score.py`'s reason: a Processing job has no script
mode, so the file is there to be named by `job.container_entrypoint`.

The order of the steps is the order the failures are worth having in. The match
cache is built and written before anything is compared, so a job that dies in the
bootstrap still leaves the arrays -- which are the expensive half, and the half
every later cycle reads back rather than recomputing. The gates come last,
because a verdict is the one output that is worthless if any input to it was
wrong.

**This is where the two halves of the plane meet.** Scoring holds no label grant
and wrote down what the model emitted; this reads those detections and the eval
cohort's boxes together, which is the whole reason it is a second job under a
second role. Nothing here runs a model: there is no checkpoint on any channel,
so a detection that reaches the metric was produced by the scoring job under the
`git_commit` this same archive records.

**A failed gate is a completed job.** The verdict is the product, so a challenger
that does not clear the promotion rule is written to the report and the process
exits zero. Only a job that could not reach a verdict -- a missing channel, an
eval cohort that does not match the champion's -- fails.
"""

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from pycocotools.coco import COCO

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    Cohort,
    GateResult,
    ImageId,
    ModelVersion,
    Seed,
    eval_matches_key,
    eval_prefix,
    parse_model_version,
)
from edge_ml_flywheel.evaluation import coco
from edge_ml_flywheel.evaluation.bootstrap import PairedDelta, paired_delta
from edge_ml_flywheel.evaluation.coco import ImageIndex
from edge_ml_flywheel.evaluation.job import (
    CHAMPION_CHANNEL,
    EVAL_OUTPUT,
    GATES_OUTPUT,
    LABELS_CHANNEL,
    MANIFEST_CHANNEL,
    detections_channel,
    output_names,
)
from edge_ml_flywheel.evaluation.match import MatchCache, load, save, score
from edge_ml_flywheel.evaluation.metrics import (
    OVERALL,
    Scope,
    average_precision,
    image_rows,
    per_class_average_precision,
)
from edge_ml_flywheel.gates import DEFAULT, quality_gate
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.scoring import detections as detection_rows
from edge_ml_flywheel.scoring.job import INPUT_ROOT, OUTPUT_ROOT
from edge_ml_flywheel.training import images
from edge_ml_flywheel.training import labels as label_files

log = logging.getLogger("edge_ml_flywheel.evaluation")

# The IoU everyone quotes, reported beside the COCO mean over all ten. One extra
# read of the same cache, so it is free in the sense that matters: no second
# scoring pass and no second file.
AT_50: Final = Scope(iou=0.5)

# How many offending IDs a refusal lists before summarizing, matching the rest of
# the package.
_REPORTED: Final = 5


def _parser() -> argparse.ArgumentParser:
    """The flags `job.arguments` writes, underscored to match the other two."""
    parser = argparse.ArgumentParser(prog="evaluate.py")
    parser.add_argument("--version", required=True, help="The challenger being evaluated.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True, help="Every seed this cycle trained."
    )
    parser.add_argument(
        "--champion",
        default=None,
        help="The model to compare against. Absent on a run's first cycle.",
    )
    return parser


def channel(name: str) -> Path:
    return Path(INPUT_ROOT) / name


def deployed_seed(seeds: Sequence[Seed]) -> Seed:
    """The seed whose numbers describe the model that would ship.

    Seed 1 by convention (design section 4.2), and never the best-scoring one --
    picking on the eval set biases the number the gate then reports, and the
    leakage check cannot catch it because there is no ID overlap. `min` rather
    than a literal 1 so a run that trained some other set still names one seed
    rather than raising over a convention.
    """
    return min(seeds)


def scored_images() -> ImageIndex:
    """The eval images this cycle put in front of the model.

    Read off the manifest rather than off the detections, for `job.inputs`'
    reason: a frame the model missed entirely produces no detection row, so an
    index built from the detections would drop every miss and score the model on
    its hits alone.
    """
    found = sorted(channel(MANIFEST_CHANNEL).glob("*.manifest"))
    if len(found) != 1:
        raise SystemExit(
            f"the {MANIFEST_CHANNEL} channel carries {len(found)} manifests, and exactly one "
            f"document says which images were scored"
        )
    return ImageIndex.of(images.manifest_ids(found[0]))


def eval_boxes(index: ImageIndex) -> dict[ImageId, tuple[Box, ...]]:
    """The ground truth for exactly the scored images.

    Read through `training.labels`, which is the reader of record for a label
    file: the eval cohort's parquet carries the same two columns a purchase does,
    so one decoder serves both and a box read back here is the box the
    partitioner wrote.

    The cohort file covers all 5,000 eval images and the manifest may name fewer
    -- a skeleton run caps it -- so the set is narrowed here rather than in the
    index. Narrowing the other way would score the unscored images as frames the
    model detected nothing in, which is a real verdict about an imaginary pass.
    """
    root = channel(LABELS_CHANNEL)
    found = label_files.collect([root])
    if not found:
        raise SystemExit(
            f"the {LABELS_CHANNEL} channel carries no boxes, so there is nothing to match against"
        )

    missing = sorted(set(index.image_ids) - set(found))
    if missing:
        raise SystemExit(
            f"{len(missing)} of the {len(index)} scored images have no ground truth in the "
            f"{Cohort.EVAL.value} labels: {missing[:_REPORTED]}"
        )
    return {image_id: found[image_id] for image_id in index.image_ids}


def seed_cache(truth: COCO, index: ImageIndex, seed: Seed) -> MatchCache:
    """One seed's scoring pass, matched and kept in the form a resample reads."""
    root = channel(detections_channel(seed))
    rows = detection_rows.collect(root)
    if not rows:
        log.warning("seed %d detected nothing at all over %d images", seed, len(index))

    predictions = detection_rows.group(rows)
    results = coco.as_coco_results(truth, coco.detections(predictions, CLASS_SET, index))
    cache = score(truth, results, index)

    log.info(
        "seed %d: %d detections over %d images, cached as %d blocks",
        seed,
        len(rows),
        len(index),
        cache.n_truth.size,
    )
    return cache


def champion_caches(
    champion: ModelVersion, seeds: Sequence[Seed], index: ImageIndex
) -> dict[Seed, MatchCache]:
    """The champion's cached arrays, one per seed, read rather than recomputed.

    Addressed by `eval_matches_key` rather than globbed, so the seeds loaded are
    exactly the seeds the challenger trained: `bootstrap.paired_delta` refuses a
    comparison whose two sides carry different seed sets, and a glob would hand
    it whichever ones happened to be in the bucket.

    The cohort is checked here rather than left to the refusal downstream. Two
    caches over different eval sets is the one mismatch that produces arithmetic
    instead of an error at every step but the last, and naming it as a channel
    problem is what sends someone to `max_images` rather than to the bootstrap.
    """
    root = channel(CHAMPION_CHANNEL)
    prefix = eval_prefix(champion)

    caches: dict[Seed, MatchCache] = {}
    for seed in seeds:
        path = root / eval_matches_key(champion, seed).removeprefix(prefix)
        if not path.is_file():
            raise SystemExit(
                f"champion {champion} has no cached matches for seed {seed}. The challenger was "
                f"trained at seeds {sorted(seeds)}, and a paired delta compares matching seeds."
            )
        cache = load(path)
        if cache.index != index:
            raise SystemExit(
                f"champion {champion} seed {seed} was scored over {len(cache.index)} eval images "
                f"and this cycle scored {len(index)}. A delta between them is not a measurement "
                f"of the same test -- re-score this cycle over the cohort the champion saw."
            )
        caches[seed] = cache

    log.info("loaded %d champion caches from %s", len(caches), champion)
    return caches


def all_rows(index: ImageIndex) -> NDArray[np.int64]:
    """Every image in the cohort, as the row list `metrics` takes.

    The whole eval set is the gate's scope (design section 4.4): promotion turns
    on the overall metric, and a subset here would be a slice casting a vote.
    """
    return image_rows(range(len(index)))


def metrics_document(
    version: ModelVersion, index: ImageIndex, caches: Mapping[Seed, MatchCache]
) -> dict[str, Any]:
    """The `eval_metrics_key` document: what each seed scored, on its own.

    Per seed rather than averaged, because the average is the gate's business and
    this is the record the average was taken over. A cycle that trains one seed
    writes one entry, and the file reads the same either way.

    Slices are deliberately absent. The regression report groups these same
    arrays by the manifest's condition tags (design section 4.4), which is a join
    against a table this job is not handed -- and the cache beside this file is
    what makes that a later query rather than a second scoring pass.
    """
    rows = all_rows(index)
    return {
        "version": version,
        "images": len(index),
        "classes": list(CLASS_SET.names),
        "seeds": {
            str(seed): {
                "map": average_precision(cache, rows, OVERALL),
                "map50": average_precision(cache, rows, AT_50),
                "per_class": per_class_average_precision(cache, rows, OVERALL),
            }
            for seed, cache in sorted(caches.items())
        },
    }


def report_document(
    version: ModelVersion,
    champion: ModelVersion | None,
    seeds: Sequence[Seed],
    gates: Sequence[GateResult],
    delta: PairedDelta | None,
) -> dict[str, Any]:
    """The `gate_report_key` document: the verdict, and what produced it.

    The thresholds are recorded beside the result they produced. They are a
    constant, so this is not the record of a choice -- it is what makes the
    verdict checkable six weeks later without a git checkout, and it is the claim
    that models are gated against fixed, pre-declared numbers rather than
    numbers someone can shop for (`gates.thresholds`).

    `passed` is every reported gate green. Two of the four are unimplemented and
    simply absent, rather than recorded as passing: `edge` and `canary` read
    measurements nothing in this project produces yet, and a green verdict for a
    check that never ran is the one entry in this document that would be a lie.
    """
    return {
        "version": version,
        "champion": champion,
        "seeds": [int(seed) for seed in sorted(seeds)],
        "passed": all(gate.passed for gate in gates),
        "gates": [
            {"gate": str(gate.gate), "passed": gate.passed, "reason": gate.reason} for gate in gates
        ],
        "delta": (
            None
            if delta is None
            else {
                "observed": delta.observed,
                "lower": delta.lower,
                "upper": delta.upper,
                "resamples": delta.resamples,
                "confidence": delta.confidence,
            }
        ),
        "thresholds": {
            "min_new_images": DEFAULT.min_new_images,
            "min_instances_per_class": DEFAULT.min_instances_per_class,
            "min_mean_delta": DEFAULT.min_mean_delta,
        },
    }


def _write_json(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parser().parse_args(argv)

    version = parse_model_version(args.version)
    champion = parse_model_version(args.champion) if args.champion else None
    seeds = tuple(Seed(seed) for seed in sorted(set(args.seeds)))
    names = output_names(version, seeds)

    index = scored_images()
    log.info("evaluating %s at seeds %s over %d images", version, list(seeds), len(index))

    truth = coco.as_coco(coco.ground_truth(eval_boxes(index), CLASS_SET, index))

    # The caches are written as they are built, before anything is compared, so a
    # failure in the bootstrap leaves behind the expensive half of this job.
    caches = {seed: seed_cache(truth, index, seed) for seed in seeds}
    for seed, cache in caches.items():
        path = Path(OUTPUT_ROOT) / EVAL_OUTPUT / names[f"matches-{seed}"]
        path.parent.mkdir(parents=True, exist_ok=True)
        save(cache, path)
        log.info("cached seed %d matches at %s", seed, path)

    metrics = metrics_document(version, index, caches)
    _write_json(metrics, Path(OUTPUT_ROOT) / EVAL_OUTPUT / names["metrics"])

    delta = (
        paired_delta(champion_caches(champion, seeds, index), caches)
        if champion is not None
        else None
    )
    if delta is not None:
        log.info(
            "paired delta %+.4f, 95%% band [%+.4f, %+.4f]", delta.observed, delta.lower, delta.upper
        )
    else:
        log.info("no champion, so %s is this run's baseline rather than a challenger", version)

    # Seed 1's per-class scores, because the collapse check asks whether the
    # model being promoted produces output -- and exactly one artifact ships.
    # Recomputed off the cache rather than read back out of the document above,
    # so the gate reads floats rather than whatever survived a JSON round trip.
    shipped = deployed_seed(seeds)
    per_class = per_class_average_precision(caches[shipped], all_rows(index), OVERALL)
    verdict = quality_gate(delta, per_class, CLASS_SET)
    log.info(
        "%s gate: %s -- %s",
        verdict.gate,
        "pass" if verdict.passed else "FAIL",
        verdict.reason,
    )

    report = report_document(version, champion, seeds, [verdict], delta)
    _write_json(report, Path(OUTPUT_ROOT) / GATES_OUTPUT / names["report"])


if __name__ == "__main__":
    main()

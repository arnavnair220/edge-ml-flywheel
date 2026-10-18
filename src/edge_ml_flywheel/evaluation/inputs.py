"""What one evaluation job reads, fetched from the store onto local disk.

`training.inputs`' arrangement for the evaluation job. This module needs no GPU
stack and neither does the job, but the keys are still the part worth testing
on their own: they are the whole of the eval wall's reach.

**This is the one job that reads a ground-truth box for the eval cohort.** Its
only label input is the `cohort=eval` prefix of the partition. Bootstrap and the
purchases are never fetched -- they are the model's training set, and a job that
could read them could report a number measured over what the model learned from.

Every input, named by `conventions`:

- **the manifest** at `scoring_manifest_key` for `eval`: the statement of which
  images were put in front of the model.
- **the eval labels**, the whole `cohort=eval` prefix.
- **each seed's detections**, the fp32 pass at `detections_prefix`.
- **the int8 detections** of the deployed seed, when the edge gate runs.
- **the champion's cached arrays**, its whole `eval_prefix`, when there is one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import (
    Cohort,
    ModelVersion,
    PartitionVersion,
    Precision,
    Seed,
    cohort_labels_prefix,
    detections_prefix,
    eval_prefix,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
)
from edge_ml_flywheel.storage import ObjectStore, tree

# One short directory per input under the scratch root, each holding its objects
# relative to the input's prefix (see `storage.tree`).
MANIFEST: Final = "manifest"
LABELS: Final = "labels"
DETECTIONS: Final = "detections-seed"
INT8: Final = "detections-int8"
CHAMPION: Final = "champion"


@dataclass(frozen=True, slots=True)
class Comparison:
    """What one evaluation decides: a challenger's seeds against a champion.

    `champion` is `None` on a run's first cycle, whose model is the baseline.
    `gates_edge` is whether the deployed seed's int8 pass is judged.
    """

    version: ModelVersion
    seeds: tuple[Seed, ...]
    partition_version: PartitionVersion
    champion: ModelVersion | None = None
    gates_edge: bool = False

    @property
    def deployed_seed(self) -> Seed:
        """The lowest seed, matching `entrypoint.deployed_seed`."""
        return min(self.seeds)


@dataclass(frozen=True, slots=True)
class Staged:
    """Where each input landed. `None` for an input this cycle does not have."""

    manifest: Path
    labels: Path
    detections: dict[Seed, Path]
    int8: Path | None
    champion: Path | None


def stage(data: ObjectStore, artifacts: ObjectStore, comparison: Comparison, root: Path) -> Staged:
    """Fetch every input this evaluation reads, and say where each one is."""
    version = comparison.version
    run_id = model_version_run_id(version)
    key = scoring_manifest_key(run_id, model_version_cycle(version), Cohort.EVAL)
    manifest = root / MANIFEST / Path(key).name
    artifacts.download(key, manifest)

    labels = cohort_labels_prefix(comparison.partition_version, Cohort.EVAL)
    if tree.fetch_prefix(data, labels, root / LABELS, "eval label files") == 0:
        raise SystemExit(f"{labels} holds no labels, so there is nothing to match against")

    detections: dict[Seed, Path] = {}
    for seed in sorted(comparison.seeds):
        local = root / f"{DETECTIONS}-{seed}"
        prefix = detections_prefix(version, seed, Cohort.EVAL)
        tree.fetch_prefix(artifacts, prefix, local, f"seed {seed} detections")
        detections[seed] = local

    int8 = None
    if comparison.gates_edge:
        int8 = root / INT8
        prefix = detections_prefix(version, comparison.deployed_seed, Cohort.EVAL, Precision.INT8)
        tree.fetch_prefix(artifacts, prefix, int8, "int8 detections")

    champion = None
    if comparison.champion is not None:
        champion = root / CHAMPION
        tree.fetch_prefix(artifacts, eval_prefix(comparison.champion), champion, "champion caches")

    return Staged(
        manifest=manifest,
        labels=root / LABELS,
        detections=detections,
        int8=int8,
        champion=champion,
    )

"""What one scoring job reads, fetched from the store onto local disk.

`training.inputs`' arrangement for the scoring job: the keys are the part worth
testing, and `entrypoint` cannot be imported without a GPU stack.

Two inputs, both named by `conventions`:

- **the model**, at `model_artifact_key` for the precision being run: the
  checkpoint for fp32, the int8 graph that ships for int8.
- **each cohort's images**, the objects `scoring_manifest_key` names, refused
  unless the manifest names the split that cohort is drawn from.

**No label is read, and there is nowhere to read one from.** Neither input is a
label prefix, and the matching against ground truth happens in the evaluation
job under an identity that may hold it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    Cohort,
    ModelArtifact,
    ModelVersion,
    Precision,
    Seed,
    model_artifact_key,
    model_version_cycle,
    model_version_run_id,
    scoring_manifest_key,
)
from edge_ml_flywheel.storage import ObjectStore, tree
from edge_ml_flywheel.training import images

# The cohorts a scoring job covers: `eval`, the 5,000 frozen images every
# reported metric is computed over. The pool is scored on the device by the model
# deployed to it, which is what makes the selector's uncertainty the fleet's.
COHORTS: Final = frozenset({Cohort.EVAL})

# The model's directory under the scratch root. Each cohort's images land in a
# directory named by the cohort.
MODEL: Final = "model"


@dataclass(frozen=True, slots=True)
class Pass:
    """Which model one scoring job runs, at which precision."""

    version: ModelVersion
    seed: Seed
    precision: Precision = Precision.FP32


@dataclass(frozen=True, slots=True)
class Staged:
    """Where the model landed, and every cohort's images in a fixed order."""

    model: Path
    images: dict[Cohort, list[Path]]


def artifact_for(precision: Precision) -> ModelArtifact:
    """The file one pass scores with.

    The fp32 pass loads the checkpoint and the int8 pass loads the quantized
    graph that ships. Ultralytics runs either, and applies NMS itself over the
    ONNX graph's raw outputs exactly as it does over the checkpoint's -- which is
    why the export leaves NMS out.
    """
    return ModelArtifact.ONNX if precision is Precision.INT8 else ModelArtifact.TORCH


def stage(data: ObjectStore, artifacts: ObjectStore, scored: Pass, root: Path) -> Staged:
    """Fetch the model and every cohort's images, and say where each one is.

    The model first, so a missing or unreadable checkpoint costs one request
    rather than a copy of 5,000 images.
    """
    artifact = artifact_for(scored.precision)
    model = root / MODEL / artifact.value
    artifacts.download(model_artifact_key(scored.version, scored.seed, artifact), model)

    run_id = model_version_run_id(scored.version)
    cycle = model_version_cycle(scored.version)

    staged: dict[Cohort, list[Path]] = {}
    for cohort in sorted(COHORTS):
        split = COHORT_SPLIT[cohort]
        manifest = artifacts.get(scoring_manifest_key(run_id, cycle, cohort))
        keys = images.manifest_keys(manifest, split)
        prefix = images.prefix_of(split)
        tree.fetch(data, keys, root / cohort.value, prefix, f"{cohort.value} images")
        staged[cohort] = sorted(root / cohort.value / key.removeprefix(prefix) for key in keys)

    return Staged(model=model, images=staged)

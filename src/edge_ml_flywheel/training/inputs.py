"""What one training job reads, fetched from the store onto local disk.

Here rather than in `entrypoint`, which cannot be imported without a GPU stack:
which keys a job reads is the part worth testing, and it needs only a store.

Four inputs, every one named by `conventions` from the model version and the
partition version:

- **images**, the objects `training_manifest_key` names. The labeled set is a
  scattered subset of the `train` images, so it is named object by object.
- **bootstrap labels**, the whole `cohort=bootstrap` prefix of the partition.
- **purchases**, the whole of the run's purchase prefix, which is the rest of the
  labeled set. Empty at cycle 0, when nothing has been bought.
- **base weights**, the one checkpoint every seed of every cycle starts from.

No `eval` label is among them, and there is no argument that makes one appear:
the label prefixes are built from `Cohort.BOOTSTRAP` and the run, and the
images are refused unless the manifest names the `train` prefix.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from edge_ml_flywheel.conventions import (
    Cohort,
    ModelVersion,
    PartitionVersion,
    Split,
    base_weights_key,
    cohort_labels_prefix,
    model_version_cycle,
    model_version_run_id,
    purchases_run_prefix,
    training_manifest_key,
)
from edge_ml_flywheel.storage import ObjectStore, tree
from edge_ml_flywheel.training import images

log = logging.getLogger(__name__)

# One short directory per input under the scratch root, each holding its objects
# relative to the input's prefix (see `storage.tree`).
IMAGES: Final = "images"
BOOTSTRAP: Final = "bootstrap"
PURCHASES: Final = "purchases"
BASE: Final = "base"


@dataclass(frozen=True, slots=True)
class Staged:
    """Where each input landed."""

    images: Path
    labels: tuple[Path, ...]
    base: Path


def stage(
    data: ObjectStore,
    artifacts: ObjectStore,
    version: ModelVersion,
    partition_version: PartitionVersion,
    root: Path,
) -> Staged:
    """Fetch every input this job trains from, and say where each one is.

    The manifest is read before anything else is fetched, so a manifest naming
    the wrong split costs one request rather than a copy of the labels first.
    """
    run_id = model_version_run_id(version)
    cycle = model_version_cycle(version)

    keys = images.manifest_keys(artifacts.get(training_manifest_key(run_id, cycle)), Split.TRAIN)
    tree.fetch(data, keys, root / IMAGES, images.prefix_of(Split.TRAIN), "training images")

    bootstrap = cohort_labels_prefix(partition_version, Cohort.BOOTSTRAP)
    if tree.fetch_prefix(data, bootstrap, root / BOOTSTRAP, "bootstrap label files") == 0:
        raise SystemExit(
            f"{bootstrap} holds no labels. The partition writes them; draw partition "
            f"{partition_version} before training on it."
        )

    bought = tree.fetch_prefix(data, purchases_run_prefix(run_id), root / PURCHASES, "purchases")
    log.info("cycle %d trains on the bootstrap and %d purchase files", cycle, bought)

    base = root / BASE / Path(base_weights_key()).name
    artifacts.download(base_weights_key(), base)

    return Staged(
        images=root / IMAGES,
        labels=(root / BOOTSTRAP, root / PURCHASES),
        base=base,
    )

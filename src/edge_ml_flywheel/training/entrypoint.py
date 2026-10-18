"""The training job itself, as it runs inside the container.

Started by `container/train.py`, a three-line file at the root of the source
archive. Everything it does is here instead, where ruff and mypy see it and the
pure parts have tests.

The order of the steps is the order the failures are worth having in. Seeding
comes first, so nothing has drawn a random number before the seed is set. The
inputs are fetched from the store and inventoried before the work, so a job that
dies later still leaves the download measurement design section 11 asks every job
to record. The dataset is built and asserted before the model is loaded, so a
mismatch between the manifest and the labels costs seconds of GPU rather than an
epoch of it.

**What this job reads and writes.** Every input comes from the store, at keys
`training.inputs` derives from the model version and the partition version; the
bucket names come from configuration (`docs/storage.md`). The outputs are the
loose model files and their digests, at the keys
`conventions.model_artifact_key` names. Same bytes, hashed once, where they were
produced -- a digest computed anywhere else is a digest of a copy.

Two of those files are models. `model.pt` is the checkpoint, and `model.onnx` is
the int8 graph an ARM CPU can run, which is what the fleet loads and what the
manifest's digest names. `export` produces the second from the first, in this
job rather than a later one, so the artifact that ships and the weights it came
from are never separated by a step that could fail between them.
"""

import argparse
import hashlib
import logging
import os
import random
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch
from ultralytics import YOLO

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    ModelArtifact,
    PartitionVersion,
    Seed,
    model_artifact_key,
    parse_model_version,
    sha256sums_document,
    uri,
)
from edge_ml_flywheel.storage import ARTIFACTS_BUCKET, DATA_BUCKET, ObjectStore, bucket_name, tree
from edge_ml_flywheel.storage.s3 import S3Store
from edge_ml_flywheel.training import dataset, export, images, inputs, labels

log = logging.getLogger("edge_ml_flywheel.training")

# Where the inputs land, and where the conversion and the run happen. Local
# scratch on the job's own volume; nothing here survives the job except what is
# uploaded.
WORK: Final = Path("/tmp/training")

# Ultralytics writes `last.pt` and `best.pt`. `last.pt` is what ships, and the
# distinction is the same rule that makes seed 1 the deployed artifact rather
# than the best-scoring seed: selecting a checkpoint on a score is a choice made
# against a number this project measures somewhere else, on a cohort this job
# cannot see.
_RUN_NAME: Final = "train"
_CHECKPOINT: Final = "last.pt"

_CHUNK: Final = 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    """Every flag as `--<key> <value>`, underscored to match the other two jobs.

    `parse_known_args` in `main` rather than `parse_args`: a flag this job does
    not read is logged and ignored, so a workflow that passes one more parameter
    than this version knows does not fail a GPU job at startup.
    """
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--version", required=True, help="The model version this job produces.")
    parser.add_argument(
        "--partition_version",
        type=int,
        required=True,
        help="The partition the bootstrap labels are read from, off the run registration.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--freeze", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    # Required like the rest, rather than defaulted to 0. Every request carries
    # it, and a default here would silently train on every label the run has
    # bought if the hyperparameter were ever dropped -- which is the failure this
    # argument exists to end, not one to leave a quiet path back to.
    parser.add_argument("--max_images", type=int, required=True)
    return parser


def inventory(roots: Sequence[Path]) -> None:
    """Log what arrived under each input, in files and in bytes.

    This is the measurement design section 11 leaves open: fetching the inputs
    costs a time driven by object count rather than bytes, and whether 8,000
    images rising to 16,000 makes that copy slow enough to pack into shards is a
    number rather than an argument.
    """
    for root in roots:
        if not root.is_dir():
            log.info("%s: absent", root)
            continue
        files = [path for path in root.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        log.info("%s: %d files, %.1f MB", root, len(files), size / 1024 / 1024)


def seed_everything(seed: Seed) -> None:
    """Fix initialization and augmentation order, which is what pairing shares.

    Seeded, not bit-exact, and the difference is stated rather than papered over
    (design section 3): GPU kernels leave residual non-determinism that no
    setting fully removes, and its contribution to a paired delta is far below
    the spread between two seeds of the same recipe.

    `PYTHONHASHSEED` is set for the record and takes effect only in subprocesses,
    since the interpreter read it before this line ran. Nothing in the training
    path depends on string hash order; it is here so a dataloader worker starts
    from the same place its parent did.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest(path: Path) -> str:
    """sha256 over the file, streamed. The model is megabytes, not gigabytes,
    but the loop costs nothing and this function is the one the device's
    verification is checked against."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_dataset(staged: inputs.Staged, max_images: int = 0) -> Path:
    """Staged inputs in, a YOLO dataset directory out.

    `max_images` caps the labels to the same images the manifest names, and the
    arithmetic is `images.capped_labels` rather than anything written here: the
    labels arrive on whole prefixes that no cap can be expressed on, so the
    number has to be re-derived where they are read, and this module cannot be
    imported without a GPU stack. Nothing here decides *which* images -- it
    re-derives what the manifest already says, and `dataset.write` still refuses
    the pair if the two ever disagree.
    """
    started = time.monotonic()
    collected = labels.collect(staged.labels)
    log.info(
        "labeled set: %d images and %d boxes before the class filter",
        len(collected),
        labels.box_count(collected),
    )

    labeled = images.capped_labels(collected, max_images)
    if len(labeled) != len(collected):
        log.info("capped the labels to %d images, matching the manifest", len(labeled))

    root = WORK / "dataset"
    dataset.write(root, staged.images, labeled, CLASS_SET)
    log.info("dataset built in %.1fs", time.monotonic() - started)
    return root


def train(root: Path, base: Path, args: argparse.Namespace) -> Path:
    """Fine-tune from the COCO base and return the checkpoint that ships.

    `val=False` because this project's metric is the eval plane's, over the
    frozen 5,000-image cohort, computed from cached match arrays and compared
    against the champion's. A number YOLO prints against the training images is
    not that metric and would invite being read as though it were.

    The base is a file in the bucket rather than a download, so the same bytes
    start every seed of every cycle, and its digest is logged for the run to be
    able to prove it.
    """
    log.info("base weights %s, sha256 %s", base.name, digest(base))

    started = time.monotonic()
    model = YOLO(str(base))
    model.train(
        data=str(root / dataset.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        freeze=args.freeze,
        seed=args.seed,
        deterministic=True,
        val=False,
        plots=False,
        project=str(WORK),
        name=_RUN_NAME,
        exist_ok=True,
    )
    log.info("trained %d epochs in %.1f minutes", args.epochs, (time.monotonic() - started) / 60)

    checkpoint = WORK / _RUN_NAME / "weights" / _CHECKPOINT
    if not checkpoint.is_file():
        raise SystemExit(f"training finished without writing {checkpoint}")
    return checkpoint


def publish(checkpoint: Path, root: Path, args: argparse.Namespace, artifacts: ObjectStore) -> None:
    """Put the model where all of its readers look, and prove what landed.

    Two artifacts, not one. `model.pt` is what the scoring job loads; `model.onnx`
    is the int8 graph the fleet runs, and the digest the manifest carries. Both are
    hashed here, where the bytes were produced -- a digest taken anywhere else is
    a digest of a copy.

    Every seed exports, not only the one that ships. The alternative makes
    `artifact_sha256` mean the ONNX digest for seed 1 and the checkpoint digest
    for every other seed, which is a field with two meanings and a branch in this
    function to produce them.

    The read-back is the partitioner's and the registration's arrangement: the
    job does not report success on the strength of an upload that returned. A
    truncated upload is exactly the failure the device's digest check is meant to
    catch at the far end, and catching it here costs one GET per artifact.
    """
    version = parse_model_version(args.version)
    seed = Seed(args.seed)

    model_dir = WORK / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    local = {ModelArtifact.TORCH: model_dir / ModelArtifact.TORCH.value}
    shutil.copy2(checkpoint, local[ModelArtifact.TORCH])
    local[ModelArtifact.ONNX] = export.deployable(
        checkpoint,
        export.calibration_frames(root / dataset.IMAGES_DIR, seed),
        model_dir / ModelArtifact.ONNX.value,
        args.image_size,
        WORK,
    )

    digests = {artifact: digest(path) for artifact, path in local.items()}

    # One file listing both artifacts rather than one file each: that is what the
    # `sha256sum -c` format is for, and `registry.launch` reads the line it wants
    # by name.
    local_sha = model_dir / ModelArtifact.SHA256.value
    local_sha.write_text(sha256sums_document(digests), encoding="utf-8")

    for artifact, path in (*local.items(), (ModelArtifact.SHA256, local_sha)):
        key = model_artifact_key(version, seed, artifact)
        artifacts.upload(path, key)
        log.info("wrote %s", uri(artifacts.bucket, key))

    for artifact, sha256 in digests.items():
        landed = WORK / f"readback-{artifact.value}"
        artifacts.download(model_artifact_key(version, seed, artifact), landed)
        if digest(landed) != sha256:
            raise SystemExit(
                f"{artifact.value} read back from the store does not match what was uploaded: "
                f"{sha256} written, {digest(landed)} read"
            )
        landed.unlink()
        log.info("model %s seed %d, %s sha256 %s", version, seed, artifact.value, sha256)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args, unknown = _parser().parse_known_args(argv)
    if unknown:
        log.info("ignoring arguments this job does not read: %s", unknown)

    seed_everything(Seed(args.seed))

    data = S3Store.connect(bucket_name(DATA_BUCKET), pool=tree.WORKERS)
    artifacts = S3Store.connect(bucket_name(ARTIFACTS_BUCKET))

    started = time.monotonic()
    staged = inputs.stage(
        data,
        artifacts,
        parse_model_version(args.version),
        PartitionVersion(args.partition_version),
        WORK / "input",
    )
    log.info("inputs fetched in %.1fs", time.monotonic() - started)
    inventory([staged.images, *staged.labels])

    root = build_dataset(staged, args.max_images)
    publish(train(root, staged.base, args), root, args, artifacts)


if __name__ == "__main__":
    main()

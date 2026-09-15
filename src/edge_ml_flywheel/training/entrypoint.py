"""The training job itself, as it runs inside the container.

Started by `container/train.py`, which is a three-line file at the root of the
source archive because SageMaker requires the entry point there. Everything it
does is here instead, where ruff and mypy see it and the pure parts have tests.

The order of the steps is the order the failures are worth having in. Seeding
comes first, so nothing has drawn a random number before the seed is set. The
channel inventory comes before the work, so a job that dies later still leaves
the download measurement design section 11 asks every job to record. The dataset
is built and asserted before the model is loaded, so a mismatch between the
manifest and the labels costs seconds of GPU rather than an epoch of it.

**What this job writes, and why it writes it twice.** SageMaker collects
`/opt/ml/model` into a `model.tar.gz`, which is the form the batch transform job
and the model registry consume. The device and the model manifest want the loose
file and its digest at the keys `conventions.model_artifact_key` names. Same
bytes, hashed once, where they were produced -- a digest computed anywhere else
is a digest of a copy.
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

import boto3
import numpy as np
import torch
from ultralytics import YOLO

from edge_ml_flywheel.conventions import (
    CLASS_SET,
    ModelArtifact,
    Seed,
    model_artifact_key,
    parse_model_version,
    uri,
)
from edge_ml_flywheel.training import dataset, labels
from edge_ml_flywheel.training.job import (
    BASE_CHANNEL,
    BOOTSTRAP_CHANNEL,
    CHANNEL_ROOT,
    IMAGES_CHANNEL,
    MODEL_DIR,
    PURCHASES_CHANNEL,
)

log = logging.getLogger("edge_ml_flywheel.training")

# Where the conversion and the run happen. Local scratch on the job's own volume;
# nothing here survives the job except what is uploaded.
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
    """SageMaker passes every hyperparameter as `--<key> <value>`, verbatim.

    So the flags are spelled with underscores rather than hyphens: the key in
    `job.hyperparameters` is the flag, and a hyphenated flag here would be a job
    that starts, downloads its channels and then exits on an unrecognized
    argument. `parse_known_args` covers the other direction -- the toolkit's own
    `sagemaker_*` parameters are meant to be filtered out before the script is
    called, and a job should not die if one arrives anyway.
    """
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--version", required=True, help="The model version this job produces.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--freeze", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--artifacts_bucket", required=True)
    return parser


def channel(name: str) -> Path:
    return Path(CHANNEL_ROOT) / name


def inventory(names: Sequence[str]) -> None:
    """Log what arrived on each channel, in files and in bytes.

    This is the measurement design section 11 leaves open: `File` mode copies the
    channel before the first step at a cost driven by object count rather than
    bytes, and whether 8,000 images rising to 16,000 makes that copy slow enough
    to pack into shards is a number rather than an argument. The download
    *duration* is a SageMaker secondary status and is reported by the launcher;
    what only the container can say is how many objects that duration was for.
    """
    for name in names:
        root = channel(name)
        if not root.is_dir():
            log.info("channel %s: absent", name)
            continue
        files = [path for path in root.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        log.info("channel %s: %d files, %.1f MB", name, len(files), size / 1024 / 1024)


def seed_everything(seed: Seed) -> None:
    """Fix initialization and augmentation order, which is what pairing shares.

    Seeded, not bit-exact, and the difference is stated rather than papered over
    (design section 3): GPU kernels leave residual non-determinism that no
    setting fully removes, and its contribution to a paired delta is far below
    the seed spread the five seeds already average over.

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


def build_dataset() -> Path:
    """Channels in, a YOLO dataset directory out."""
    started = time.monotonic()
    labeled = labels.collect([channel(BOOTSTRAP_CHANNEL), channel(PURCHASES_CHANNEL)])
    log.info(
        "labeled set: %d images and %d boxes before the class filter",
        len(labeled),
        labels.box_count(labeled),
    )

    root = WORK / "dataset"
    dataset.write(root, channel(IMAGES_CHANNEL), labeled, CLASS_SET)
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


def publish(checkpoint: Path, args: argparse.Namespace) -> None:
    """Put the model where both of its readers look, and prove what landed.

    The read-back is the partitioner's and the registration's arrangement: the
    job does not report success on the strength of a `put_object` that returned.
    A truncated upload is exactly the failure the device's digest check is meant
    to catch at the far end, and catching it here costs one GET.
    """
    version = parse_model_version(args.version)
    seed = Seed(args.seed)
    bucket = args.artifacts_bucket

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model = model_dir / ModelArtifact.TORCH.value
    shutil.copy2(checkpoint, local_model)

    sha256 = digest(local_model)
    # `sha256sum -c` format, so the device verifies with the tool it already has.
    local_sha = model_dir / ModelArtifact.SHA256.value
    local_sha.write_text(f"{sha256}  {ModelArtifact.TORCH.value}\n", encoding="utf-8")

    client = boto3.client("s3")
    for artifact, path in ((ModelArtifact.TORCH, local_model), (ModelArtifact.SHA256, local_sha)):
        key = model_artifact_key(version, seed, artifact)
        client.upload_file(str(path), bucket, key)
        log.info("wrote %s", uri(bucket, key))

    landed = WORK / "readback.pt"
    torch_key = model_artifact_key(version, seed, ModelArtifact.TORCH)
    client.download_file(bucket, torch_key, str(landed))

    if digest(landed) != sha256:
        raise SystemExit(
            f"the model read back from S3 does not match what was uploaded: {sha256} written, "
            f"{digest(landed)} read"
        )
    landed.unlink()

    log.info("model %s seed %d, sha256 %s", version, seed, sha256)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args, unknown = _parser().parse_known_args(argv)
    if unknown:
        log.info("ignoring arguments this job does not read: %s", unknown)

    seed_everything(Seed(args.seed))
    inventory([IMAGES_CHANNEL, BOOTSTRAP_CHANNEL, PURCHASES_CHANNEL, BASE_CHANNEL])

    root = build_dataset()

    base = next(iter(sorted(channel(BASE_CHANNEL).glob("*.pt"))), None)
    if base is None:
        raise SystemExit(f"the {BASE_CHANNEL} channel carries no checkpoint to fine-tune from")

    publish(train(root, base, args), args)


if __name__ == "__main__":
    main()

# Plane 2 — Training

Fine-tunes YOLO11n on one cycle's labeled set as a SageMaker training job, one job per seed. See the
[architecture overview](00-overview.md) for the plane's position in the loop.

---

## Environment

| Component | Value |
|---|---|
| Container | `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` |
| Added packages | `ultralytics==8.4.146`, `pyarrow>=18` |
| Instance | `ml.g4dn.xlarge`, managed spot, one instance |
| Entry point | `container/train.py`, which calls `training.entrypoint` |
| Max runtime | 2 hours, 3 hours including the spot wait |

A spot interrupt restarts the job. There is no `CheckpointConfig` to resume from.

---

## Recipe

| Knob | Value |
|---|---|
| `epochs` | Per job |
| `image_size` | 416 |
| `batch` | 16 |
| `freeze` | 10 (YOLO11n's backbone) |
| `workers` | 4 |

Validation is off; the shipped checkpoint is `last.pt`. Python, NumPy and Torch are seeded at process
start.

---

## Channels

| Channel | Source | Type |
|---|---|---|
| `images` | `training_manifest_key(run_id, cycle)` | `ManifestFile` |
| `bootstrap` | `labels/cohort=bootstrap/` | `S3Prefix` |
| `purchases` | `derived/purchases/run_id=<id>/` | `S3Prefix` |
| `base` | `base/yolo11n.pt` | `S3Prefix` |

`purchases` is absent at cycle 0. All four are `File` mode.

---

## Dataset conversion

`training.labels` reads the bootstrap file and every purchase into one labeled set; a repeated image
raises. `training.dataset` writes YOLO's layout:

- Coordinates normalize against `NATIVE_IMAGE_SIZE` (1280x720).
- A box outside the class set is dropped.
- A box over the frame edge is clipped; one left with no area is dropped.
- An image with no boxes is kept, with an empty label file.

All counts are logged. The conversion raises if the image channel and the label set disagree in
either direction.

---

## Permissions

The training role reads `raw/images/100k/train/`, `labels/cohort=bootstrap/` and
`derived/purchases/`. It is denied `raw/labels/*` in its own policy and by the bucket policy, and
writes only `run_id=*/cycle=*/models/*`. See [infra/training.tf](../infra/training.tf).

---

## Artifacts

| Artifact | Location |
|---|---|
| `model.tar.gz` | `seed=<n>/_sagemaker/` |
| `model.pt` | `model_artifact_key(version, seed, TORCH)` |
| `model.sha256` | `model_artifact_key(version, seed, SHA256)` |

The job reads its upload back and compares digests before reporting success.

---

## Commands

```
python -m edge_ml_flywheel.training stage-base --weights yolo11n.pt
python -m edge_ml_flywheel.training prepare --run-id <id> --cycle 0 --max-images 300
python -m edge_ml_flywheel.training launch --run-id <id> --cycle 0 --seed 1 --epochs 1 --wait
```

`stage-base` runs once for the project. `prepare` runs once per cycle and writes the image manifest
and source archive; it refuses to overwrite either without `--replace`. `launch` runs once per seed.
The partition and class set come from the run registration.

---

## Incomplete

The overview's plane 2 covers five seeds and an int8 ONNX export. Neither is built: `launch` starts
one seed at a time, and the exported artifact is `model.pt` only.

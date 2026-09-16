# Plane 2 — Training

Fine-tunes YOLO11n on one cycle's labeled set as a SageMaker training job, one job per seed. See the
[architecture overview](00-overview.md) for the plane's position in the loop.

---

## Environment

| Component | Value |
|---|---|
| Container | `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` |
| Added packages | `ultralytics==8.4.146`, `pyarrow>=18` |
| Instance | `ml.g4dn.xlarge`, on demand, one instance |
| Entry point | `container/train.py`, which calls `training.entrypoint` |
| Max runtime | 2 hours |

On demand rather than managed spot. A cycle trains one seed, so a spot interrupt would discard the
cycle's training and a wait for capacity would block every step after it; the saving is under a
dollar per job. A retry restarts the job from the beginning: there is no `CheckpointConfig` to
resume from.

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

A cycle trains one seed, seed 1 — the one that ships. The seed is a per-job argument rather than a
recipe knob, so a run that trains more adds jobs over the same prepared cycle: `--seeds 1 2 3` on
`run start`, or a second `launch`. Each extra seed costs another GPU hour and needs instance quota
to cover it.

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
| `model.onnx` | `model_artifact_key(version, seed, ONNX)` |
| `model.sha256` | `model_artifact_key(version, seed, SHA256)` |

`model.pt` is the checkpoint the scoring job loads. `model.onnx` is the int8 graph the fleet runs,
and the digest the model manifest carries. Every seed exports one, so `artifact_sha256` names the
same kind of file for the seed that ships and for the seeds retained beside it.

`model.sha256` lists both digests in `sha256sum -c` format, one line per artifact, so a device
verifies its download with the tool it already has. The job reads every upload back and compares
digests before reporting success.

---

## Export

The deployed artifact is produced in this job rather than a later one, so the weights and the file
that ships are never separated by a step that can fail between them.

| Step | Setting |
|---|---|
| ONNX export | opset 17, static shapes, NMS outside the graph |
| Quantization | `onnxruntime` static, QDQ format, convolutions only |
| Weight scales | per channel |
| Detection head | excluded, left at full precision |
| Calibration | 256 training images, letterboxed as inference letterboxes them |

Ultralytics' own int8 path targets TensorRT and OpenVINO, so the export is fp32 and the
quantization is a second pass over it.

Per-channel scales and the excluded head are the starting configuration, not a remedy applied after
a failure. One scale shared across a convolution's output channels quantizes the narrow ones to a
constant; the head's convolutions emit box coordinates directly, where a rounded value displaces a
box past the IoU threshold rather than blurring a feature. Neither is a finding worth a cycle of GPU
time, and the head is a small share of a model whose parameters are in its backbone.

The head is named by module path: Ultralytics exports each node under `/model.<n>/`, numbered in
definition order with `Detect` last, so the highest index present is the head. A graph the rule
cannot read produces no artifact — quantizing the head silently is the failure it exists to prevent.

Calibration frames are drawn from the cycle's training images. `eval` frames would fold the cohort
the model is scored on into how the model is built, and the training role cannot read them in any
case.

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

The archive `prepare` writes carries the cycle's three entry points, `train.py`, `score.py` and
`evaluate.py`. The scoring and evaluation jobs unpack the same object, so the whole cycle runs from
one tree.

---

## Incomplete

The quantized model's accuracy is not yet measured against fp32. Two of the edge gate's four
thresholds need no device — artifact size and accuracy within 2% relative — and until the second is
checked, the export is known to produce a small artifact and not known to preserve the model. That
measurement is plane 3's, over `eval`.

# Plane 3 — Evaluation

Matches a cycle's detections against the eval cohort's ground truth, applies the quality gate, and
caches the per-image match arrays every later statistic reads. One SageMaker Processing job per
cycle. See the [architecture overview](00-overview.md) for the plane's position in the loop.

---

## Environment

| Component | Value |
|---|---|
| Container | `pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker` |
| Added packages | `pycocotools==2.0.11`, `pyarrow>=18` |
| Instance | `ml.m5.xlarge`, one instance |
| Entry point | `container/evaluate.py`, which calls `evaluation.entrypoint` |
| Max runtime | 1 hour |

The CPU build of the tag `training.job` pins, since the job loads no model. A Processing job has no
script mode, so the container command unpacks the cycle's source archive, installs its requirements
and runs the entry point.

One job per cycle rather than one per seed, because the paired delta is a mean over same-seed
differences.

---

## Channels

| Channel | Source | Type |
|---|---|---|
| `code` | `training_code_key(run_id, cycle)` | `S3Prefix` |
| `manifest` | `scoring_manifest_key(run_id, cycle, eval)` | `S3Prefix` |
| `labels` | `labels/cohort=eval/` | `S3Prefix` |
| `detections-seed-<n>` | `detections_prefix(version, seed, eval)` | `S3Prefix` |
| `champion` | `eval_prefix(champion)` | `S3Prefix` |

`champion` is absent on a run's first cycle. All are `File` mode.

`manifest` supplies the image IDs that were scored, not the images. SageMaker caps a Processing job
at ten inputs, which puts the ceiling at six seeds.

---

## Steps

1. Read the manifest and index the images it names.
2. Read the eval boxes for those images.
3. Per seed, match the detections against the boxes and write `matches.npz`.
4. Write `metrics.json`.
5. With a champion, load its cached arrays and run the paired bootstrap.
6. Apply the quality gate and write the gate report.

The caches are written before the comparison, so a job that fails in the bootstrap keeps them. A
failed gate is a completed job; only a job that cannot reach a verdict fails.

---

## Gate

The quality gate passes when the mean paired delta is at least +0.005 and the lower end of the 95%
band over 1,000 resamples clears zero. A class scoring zero AP is a hard failure.

A run's first cycle has no champion, so the delta is absent and the verdict is the collapse check
alone.

The edge gate passes when the int8 artifact is at most 25 MB and retains at least 95% of the fp32
model's mAP. Both are properties of a file and a number, so the gate is complete without a device;
p95 latency and cold start are reported off the fleet rather than gated (design §4.3).

The 95% allowance is deliberately looser than design §4.3's 2%. A broken export is a 30% loss or a
model that detects nothing, and a threshold tight enough to reject a working artifact would stop
the loop over a number the fleet would never notice. What quantization actually cost is in the
verdict's reason either way.

Its input is a second pass over `eval` with the quantized graph, run by the scoring job under
`precision=int8`. That pass is the scoring role's because it loads a model, and this job's role
holds no model grant at all — so the artifact's size arrives as an argument rather than being read
off the object here.

A cycle whose training job predates the export has no int8 artifact, and reports no edge verdict
rather than half of one. `canary` is unimplemented and absent from the report rather than recorded
as passing.

---

## Artifacts

| Artifact | Location |
|---|---|
| `matches.npz` | `eval_matches_key(version, seed)` |
| `metrics.json` | `eval_metrics_key(version)` |
| `report.json` | `gate_report_key(run_id, cycle)` |

Both eval artifacts are filed under the cycle that produced the model, so a champion is re-compared
without being re-scored. `metrics.json` carries overall mAP, mAP@0.5 and per-class AP per seed. The
report carries each verdict with its reason, the delta and its band, and the thresholds applied.

---

## Permissions

The evaluation role is the single ARN on `eval_label_reader_arns`. It reads `labels/cohort=eval/`,
the cycle's code, manifest and detections, and a champion's cached arrays. It is denied
`raw/labels/`, `labels/cohort=bootstrap/` and `derived/purchases/` in its own policy, and writes
`eval/*` and `gates/*` only. See [infra/evaluation.tf](../infra/evaluation.tf).

---

## Incomplete

The overview's plane 3 covers a per-slice regression report. It is not built: `metrics.json` carries
overall and per-class AP only.

# Stage 3 — Evaluation

Runs the model over the eval cohort, matches the detections against its ground truth, applies three
of the four gates, and caches the per-image match arrays every later statistic reads. See the
[architecture overview](00-overview.md) for the stage's position in the loop.

Two SageMaker Processing jobs per cycle. **Scoring** produces detections, once per seed, and holds no
label grant. **Matching** compares them against ground truth, once per cycle, and is the single
principal admitted to `labels/cohort=eval/`. The split is what lets the scoring role be denied every
label prefix outright rather than trusted to stay away from one.

**This stage covers `eval` and nothing else.** The pool is scored on the device, by the deployed
model, as part of the fleet round trip: this pass measures a model, that one ranks a sample. What the
two share is a detections schema and a reader, not a job. See
[05-fleet-and-deployment.md](05-fleet-and-deployment.md).

---

## Scoring

| Component | Value |
|---|---|
| Container | `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` |
| Instance | `ml.g4dn.xlarge`, on demand — `ml.m5.xlarge` for the int8 pass |
| Entry point | `container/score.py`, which calls `scoring.entrypoint` |
| Max runtime | 1 hour |

A Processing job rather than a Batch Transform. Both produce the same detections; a transform needs a
`Model` resource, an inference handler and one output object per input object — 5,000 objects and a
compaction step. A Processing job loads the model once, walks the channel and writes the file.

One job per seed, one cohort inside it. `eval` is 5,000 images, so the pass is minutes of GPU and an
hour is a generous ceiling.

| Knob | Value |
|---|---|
| `image_size` | `training.job.IMAGE_SIZE` |
| `confidence_floor` | 0.001 |
| `max_detections` | `evaluation.match.MAX_DETS` |
| `precision` | `fp32`, or `int8` for the edge gate's pass |

`confidence_floor` is COCO's convention and far below anything a person would call a detection. AP is
the area under a curve swept by lowering a threshold, so the low-confidence tail is most of what the
metric integrates; raising this floor would truncate the curve and quietly lower every number the
project reports.

**The device's floor is `BAND_LOW`, not this one, and the ranking is unchanged by the difference.**
The tail exists here because AP integrates it. Nothing integrates the pool: `image_score` keeps only
detections at or above `BAND_LOW`, and decides blind-spot against decisive on that filtered list, so
a row at 0.02 contributes to no score either way. Suppression is greedy and highest-first, so a box
below the band can be suppressed but can never suppress one above it — the surviving in-band rows are
the same set at either floor. What the device gains by not writing the tail is a file an order of
magnitude smaller and a bounded working set on a two-core instance.

A frame with no in-band detection is therefore absent from the device's file rather than present with
weak rows, and it still scores as a blind spot: `score_pool` is driven by the sample manifest, not by
the file's keys, so a frame nobody detected anything in is scored rather than dropped.

The int8 pass runs on CPU, and not as a saving: int8 is a CPU format. ONNX Runtime's CUDA provider
has no kernel for most of what `quantize_static` emits and falls back to float, which would measure a
model the device will never run. It covers `eval`, and it is what the edge gate compares against the
fp32 pass to say what quantization cost.

Detections land at `detections_prefix(version, seed, cohort)` with `cohort=eval`. The device writes
the pool's detections to the same family of keys under `precision=int8`, so the two passes share a
schema and a reader and differ only in which machine produced them. The job carries no `VpcConfig`,
no label channel of any kind, and no `max_images` — the images scored are exactly those named in
`scoring_manifest_key`. See [infra/scoring.tf](../infra/scoring.tf).

---

## Matching

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

## Gates

Three of the four gates are applied here: `data`, `quality` and `edge`. The predicates and their
thresholds are in [gates.md](gates.md); what belongs to this job is where they run and what they are
given.

The quality gate reads the `PairedDelta` this job computed and the per-class AP it wrote. The edge
gate reads the artifact's size and the two mAP scores, the int8 one coming from a second pass over
`eval` with the quantized graph, run by the scoring job under `precision=int8`. That pass is the
scoring role's because it loads a model, and this job's role holds no model grant at all — so the
artifact's size arrives as an argument rather than being read off the object here.

A cycle whose training job predates the export has no int8 artifact, and reports no edge verdict
rather than half of one. `canary` cannot be asked here at all: it needs a device, so it is absent
from the report rather than recorded as passing.

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

The scoring role reads images, the cycle's code and manifests, and the model artifact, and writes
detections. It holds no label grant of any kind — not even to the labels the run already owns — so
the denial is structural rather than a rule it is trusted to follow. See
[infra/scoring.tf](../infra/scoring.tf).

---

## Incomplete

The overview's stage 3 covers a per-slice regression report. It is not built: `metrics.json` carries
overall and per-class AP only.

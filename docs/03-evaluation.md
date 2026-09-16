# Plane 3 — Evaluation

Scores each model once over `eval` and the pool, then answers every later question from the cached
result. Two SageMaker Processing jobs per cycle: one that runs the model and reads no label, and one
that reads the eval boxes and runs no model. See the [architecture overview](00-overview.md) for the
plane's position in the loop.

---

## The split, and why there are two jobs

| | Scoring | Evaluation |
|---|---|---|
| Runs | The model, over 67,000 images | Nothing. Numpy over cached arrays |
| Reads | Images and a checkpoint | Detections and the eval cohort's boxes |
| May read a label | No, denied every prefix | `labels/cohort=eval/` only |
| Writes | Detections | The match cache, `metrics.json`, the gate report |
| Per | Seed | Cycle |
| Instance | `ml.g4dn.xlarge` | `ml.m5.xlarge` |

Matching a detection against a box needs both in one process. Putting that in the job that also
holds the model would put the images, the checkpoint and the answer key inside one identity's reach.
Splitting it means the scoring role can be denied every label prefix outright rather than trusted to
leave them alone, and the evaluation role never sees a checkpoint. The second role is the only
principal in the account on `eval_label_reader_arns`.

The unit of work differs for the same kind of reason. A seed is an independent pass over the images,
so scoring fans out per seed; the paired delta is a mean over same-seed differences, so evaluation
cannot be divided at all.

---

## Environment

| Component | Value |
|---|---|
| Container | `pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker` |
| Added packages | `pycocotools==2.0.11`, `pyarrow>=18` |
| Instance | `ml.m5.xlarge`, one instance |
| Entry point | `container/evaluate.py`, which calls `evaluation.entrypoint` |
| Max runtime | 1 hour |

The CPU build of the tag the training job pins. Nothing here loads a model, so the GPU image would
be gigabytes of CUDA pulled for drivers nothing opens; the framework version matches so that the
pyarrow reading a detections file is the one that wrote it.

A Processing job has no script mode, so the container command unpacks the cycle's source archive,
installs `requirements.txt` and runs the named file. That archive is the object the training job
ran, so the code that gated a model is the tree that trained and scored it.

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

The manifest is read for the image IDs it names, not for the images. It is the statement of what was
put in front of the model: a frame the model missed entirely contributes no detection row, so an
index built from the detections alone would drop every miss and score the model on its hits.

SageMaker caps a Processing job at ten inputs, which puts the ceiling at six seeds with a champion.
A cycle trains one.

---

## What the job does

1. Read the manifest, and build the image index from it.
2. Read the eval boxes for exactly those images.
3. Per seed: read the detections, match them against the boxes, write `matches.npz`.
4. Compose `metrics.json`.
5. With a champion: load its cached arrays and run the paired bootstrap.
6. Apply the quality gate and write the gate report.

The caches are written before anything is compared, so a job that dies in the bootstrap still leaves
the expensive half. A failed gate is a completed job: the verdict is the product, and only a job that
could not reach one fails.

---

## Artifacts

| Artifact | Location |
|---|---|
| `matches.npz` | `eval_matches_key(version, seed)` |
| `metrics.json` | `eval_metrics_key(version)` |
| `report.json` | `gate_report_key(run_id, cycle)` |

Both eval artifacts are filed under the cycle that produced the model rather than the cycle being
decided. The eval cohort is frozen, so a champion's scores are a function of the model alone and its
cached arrays stay valid where they were first written — which is what lets a champion be
re-compared every cycle without being re-scored.

`metrics.json` carries overall mAP, mAP@0.5 and per-class AP, per seed. The gate report carries each
verdict with its reason, the delta and its band, and the thresholds that were applied.

---

## The gate

One gate runs here: quality. It requires the mean paired delta to clear +0.005 **and** the lower end
of the 95% band over 1,000 resamples to clear zero, and it hard fails if any class scores zero AP.
The data gate is not run — it checks the batch a cycle bought, and nothing buys anything until the
purchase step lands. `edge` and `canary` read measurements the project does not yet produce, and are
absent from the report rather than recorded as passing.

**A run's first model has no delta.** There is no champion to compare against, so the two statistical
conditions have nothing to test and the verdict is the collapse check alone, with a reason that says
so. Reporting no verdict instead would leave `gates_passed` false for the only model that can become
the first champion.

**Slices are not computed.** The regression report groups these same cached arrays by the manifest's
condition tags, which is a join against a table this job is not handed. The cache beside `metrics.json`
is what makes that a later query rather than a second scoring pass.

---

## Permissions

The evaluation role reads `labels/cohort=eval/`, the cycle's code, manifest and detections, and a
champion's cached arrays. It is denied `raw/labels/`, `labels/cohort=bootstrap/` and
`derived/purchases/` in its own policy: those are the model's training set, and a job that could read
them could report a number measured over what the model learned from. It writes `eval/*` and
`gates/*` and nothing else. See [infra/evaluation.tf](../infra/evaluation.tf).

---

## Incomplete

The overview's plane 3 covers the per-slice regression report, and plane 4 covers the data gate. The
first needs the manifest's tags on a channel; the second needs a purchase to check. Neither is built.

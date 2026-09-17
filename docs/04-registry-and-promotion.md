# Stage 4 — Registry and promotion

Records what a cycle produced and what the gates decided about it, then advances the champion
pointer if they passed. One control step and four Step Functions states; no job and no compute. See
the [architecture overview](00-overview.md) for the stage's position in the loop.

---

## Manifest

Written to `model_manifest_key(version)` before anything is registered. A model with no manifest
cannot be promoted.

| Field | Source |
|---|---|
| `version` | the model version the cycle trained |
| `created_at` | registration time |
| `git_commit` | the run registration |
| `partition_version`, `recipe_version` | the run registration |
| `cohorts_trained_on` | `bootstrap`, and `pool` once anything has been bought |
| `labels_spent` | images under `derived/purchases/<run_id>/` |
| `deployed_seed` | the lowest seed the cycle trained |
| `artifact_sha256` | the `model.onnx` line of each seed's `model.sha256` |
| `gates` | the report at `gate_report_key(run_id, cycle)` |

The document is read back and re-parsed after the write, so a field that does not survive the round
trip fails the cycle that wrote it rather than the cycle that tried to promote it.

Run and cycle are not fields. Both are inside `version`, and a document restating them would be one
two readers could disagree about.

`ModelManifest.disagreements` checks the manifest against its run before the write. It cannot fail
today — see [Incomplete](#incomplete).

---

## Registry

| Element | Value |
|---|---|
| Group | the `run_id`, one per run |
| Version | one per cycle |
| `ModelApprovalStatus` | `Approved` or `Rejected`, from `ModelManifest.gates_passed` |
| `ModelCard` | the manifest's fields, restated |
| `ModelMetrics` | a pointer to `eval_metrics_key(version)` |
| `CustomerMetadataProperties` | the manifest's URI, and the fields a listing filters on |

A rejected challenger is registered too, since the rejection log is a deliverable.
`PendingManualApproval` is never used: the gates are the approver.

One group per run rather than per project, because a partition or recipe change forces a new run and
a re-baselined champion. The group is the run alone and carries no prefix — SageMaker caps an entity
name at 63 characters and the longest run ID a slug permits is 49.

The card carries what the manifest carries. Both exist because a device verifies a digest before
loading and cannot call an AWS API to read a card, so the file is the copy that matters; the card's
custom details are the manifest's metadata verbatim, so the two cannot drift.

The control function builds the request and the state machine makes the call, as with the three job
requests.

---

## Promotion

A conditional `UpdateItem` setting `champion_version` on the run's `fleet_config` item, reached only
when every gate passed.

The condition is `attribute_not_exists(champion_version) OR champion_version < :version`. A version
sorts lexicographically by its padded cycle within one run, so promotion only ever moves forward.

`ClaimCycle` already returns that item with `ALL_OLD`, so the champion arrives with the cycle number
and nothing reads it twice. A run's first cycle has no such attribute and is evaluated as its own
baseline.

---

## Permissions

| Principal | Step | Grant |
|---|---|---|
| Control function | manifest | read `gates/*` and `models/*`, write `models/version=*/manifest.json` |
| State machine | registry | `CreateModelPackageGroup`, `CreateModelPackage`, `DescribeModelPackage`, `AddTags` |
| State machine | promotion | `UpdateItem` on `fleet_config`, shared with the cycle claim |

The control function holds no SageMaker grant, so the function that builds a registration cannot
make one. `UpdateModelPackage` is granted to nobody: an approval status is the verdict a gate
reached, and a role that could revise one could approve a model the gates rejected. See
[infra/cycle.tf](../infra/cycle.tf).

---

## Commands

```
python -m edge_ml_flywheel.run show --run-id <id>
aws sagemaker list-model-packages --model-package-group-name <run_id>
```

`show` prints the registration beside the control item, which carries the current champion.

---

## Incomplete

`git_commit` is the commit the run was registered at, not the tree that trained the model. It is the
only commit available to a function with no checkout; recording the deployed tree's own SHA is a
Lambda environment variable set at deploy time.

`ModelManifest.disagreements` cannot fail. The versions it compares are read from the same
registration it checks against, so it becomes live only once a version reaches this step from the
job that ran under it rather than from the item that job was configured from.

`artifact_sha256` is the digest of `model.onnx`, the int8 graph the device loads — a digest of
anything else verifies nothing the device does. It is read out of `model.sha256` by filename rather
than by position, so a third artifact appearing in that file cannot change which line the manifest
records. A seed that published no ONNX digest is refused registration.

The overview's stage 4 covers a candidate → shadow → canary → champion → archived progression. The
approval status and the champion pointer exist; the intermediate states land with the features that
need them. See [stage 5](05-fleet-and-deployment.md).

Promotion writes no `desired_version`, and nothing else does either. Stage 5 makes the Greengrass
deployment the record of what a device should be running, so `fleet_config` holds no device item.

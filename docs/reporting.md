# Reporting

Where a finished run's record is, and what it holds. See the
[architecture overview](00-overview.md) for reporting's position beside the five stages.

Reporting is not a stage. [reporting/](../src/edge_ml_flywheel/reporting/) reduces artifacts the
stages already wrote into one document per run, and decides nothing: every figure in it was measured
by the job that recorded it, so summarizing a run twice cannot reach two answers unless one of those
artifacts changed.

---

## Where it is

```
s3://<project>-artifacts-<account_id>/run_id=<run_id>/summary.json
```

`conventions.run_summary_key` is the one spelling of that key. Under the run prefix rather than a
cycle's, because a row per cycle is a statement about the run.

```bash
aws s3 cp s3://<project>-artifacts-<account_id>/run_id=<run_id>/summary.json -
```

## When it is written

Once, by the cycle state machine's `Summarize` state, after the last cycle and before `Done`. It is
the only control step that is not part of a cycle.

Written at the end rather than maintained as the run goes, so that a missing value means one thing. A
summary rewritten every cycle would carry a newest row whose `p95_ms` and canary verdict are absent
for most of that cycle's life, leaving `null` to mean both *not measured* and *not measured yet*. At
the end of a run it means only the first.

## What a row holds

| Field | Source | Notes |
|---|---|---|
| `cycle` | the model version | Position in the run, from 0 |
| `version` | `model_manifest_key` | The model this cycle trained |
| `labels_spent` | the manifest | Cumulative, so it is what this model trained on rather than what its own batch cost |
| `gates` | the manifest, plus the canary | Each verdict with its reason. `data`, `quality` and `edge` come from the gate report the evaluation job wrote; `canary` is appended for a version that reached a device |
| `delta` | `gate_report_key` | `observed`, `lower`, `upper` — the paired mAP difference against the champion and its bootstrap band. `null` for the baseline cycle, which had no champion to be compared against |
| `deployed` | carried | The version the device was running when the cycle ended |
| `p95_ms` | fleet telemetry | Nearest-rank p95 of the device's per-frame latency. `null` for a challenger no device replayed, which at the end of a run means one that never shipped |

`deployed` is the one field not read straight off an artifact. It is a carry: a cycle whose gates all
passed put its own model on the device, and a cycle that failed any of them left whatever was already
there. That covers both ways a cycle can fail — a challenger rejected before it shipped, and a
rollout undone by the canary — because the device ends both on the previous version.

## Reading an outcome off a row

There is no `decision` field: a stored word would be a second opinion able to disagree with the
verdicts it came from. What a cycle did is read off `gates`.

| `gates` | Outcome |
|---|---|
| Every one passed | **Promoted** — this cycle's version became the champion |
| `canary` failed, the rest passed | **Rolled back** — the challenger shipped and the device's pass refused it |
| Any other one failed | **Rejected** — the challenger never left the cloud |
| Empty | **No verdict** — no check ran, which is a different fact from a model that failed one |

`CycleSummary.failed` is that reading in code.

---

## Example

Invented rows, to show the shape. **This is not a run that happened** — for real numbers, fetch the
object.

```json
{
  "schema": 1,
  "run_id": "20260921-143000-example",
  "cycles": [
    {
      "cycle": 0,
      "version": "20260921-143000-example-c000",
      "labels_spent": 8000,
      "gates": [
        { "gate": "data", "passed": true, "reason": "8000 new images, every class above 50" },
        { "gate": "edge", "passed": true, "reason": "int8 within 1.2% relative of fp32, 9.8 MB" }
      ],
      "delta": null,
      "deployed": "20260921-143000-example-c000",
      "p95_ms": 41.0
    },
    {
      "cycle": 1,
      "version": "20260921-143000-example-c001",
      "labels_spent": 9000,
      "gates": [
        { "gate": "data", "passed": true, "reason": "1000 new images, every class above 50" },
        { "gate": "quality", "passed": true, "reason": "delta +0.011, band [+0.005, +0.017]" },
        { "gate": "edge", "passed": true, "reason": "int8 within 0.9% relative of fp32, 9.8 MB" },
        { "gate": "canary", "passed": true, "reason": "p95 43 ms, digest verified, 1 start" }
      ],
      "delta": { "observed": 0.011, "lower": 0.005, "upper": 0.017 },
      "deployed": "20260921-143000-example-c001",
      "p95_ms": 43.0
    },
    {
      "cycle": 2,
      "version": "20260921-143000-example-c002",
      "labels_spent": 10000,
      "gates": [
        { "gate": "data", "passed": true, "reason": "1000 new images, every class above 50" },
        { "gate": "quality", "passed": false, "reason": "delta +0.002 below the +0.005 floor" },
        { "gate": "edge", "passed": true, "reason": "int8 within 1.1% relative of fp32, 9.8 MB" }
      ],
      "delta": { "observed": 0.002, "lower": -0.004, "upper": 0.008 },
      "deployed": "20260921-143000-example-c001",
      "p95_ms": null
    }
  ]
}
```

Read as a table, those three rows are a promotion, a promotion that survived its canary, and a
rejection that kept its labels — cycle 2's `labels_spent` is spent, its `deployed` is still
cycle 1's model, and the next challenger trains on the larger set.

---

## Incomplete

The charts are not built. When they are, they land here as committed static images beside this
document.

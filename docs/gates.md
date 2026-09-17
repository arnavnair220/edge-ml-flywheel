# Gates

Four pass/fail checks with pre-declared thresholds, and the reason behind each. See the
[architecture overview](00-overview.md) for their position in the loop.

Gates are not a stage. [gates/](../src/edge_ml_flywheel/gates/) is a library of pure functions with
no infrastructure of its own: nothing in it trains, scores, reads S3 or decodes an image. The inputs
are a purchase, a partition and numbers the evaluation job has already computed, so a promotion
decision is reproducible from a saved cycle and testable without a model.

---

## The four

| Gate | Asked | Input | What a failure decides |
|---|---|---|---|
| `data` | In the evaluation job | The purchase and the partition | The challenger is not registered |
| `quality` | In the evaluation job | A `PairedDelta` and per-class AP | The challenger is not promoted |
| `edge` | In the evaluation job | Artifact size and the two mAP scores | The challenger is not promoted |
| `canary` | After deployment | A `ReplayReport` from the device | The rollout is rolled back |

The first three are computed from a cycle's own artifacts and decide whether the challenger is
registered and promoted. `canary` is computed from what a device reported after the promoted artifact
reached it, so it is asked afterwards and decides whether the rollout stands. Its input is a
reduction of telemetry rather than an AWS call, so it stays a pure function like the rest.

---

## Two properties across all four

**A verdict is never recorded without its reason.** `GateResult` carries both. The rejection log is
what the project is built to produce (design §5), and a bare `False` six weeks later is a fact
nobody can act on.

**A gate reports every condition that failed, not the first.** The next attempt costs a training run,
so short-circuiting turns one diagnosis into two cycles.

---

## Thresholds

[gates/thresholds.py](../src/edge_ml_flywheel/gates/thresholds.py). One frozen dataclass, so the
numbers are readable before a cycle runs and recordable beside the verdict they produced. A threshold
written into a predicate body is a promotion rule that exists only as code.

| Threshold | Value | Gate |
|---|---|---|
| `min_new_images` | 250 | data |
| `min_instances_per_class` | 10 | data |
| `min_mean_delta` | +0.005 | quality |
| `max_quantization_loss` | 0.05 | edge |
| `max_artifact_bytes` | 25 MB | edge |
| `max_throughput_drop` | 0.10 | canary |

Defaults rather than a config file. These are design parameters with one setting for the life of the
project; a run that changes them is measuring something else, which `RunRegistration` already
expresses. Each is validated on construction against a value that would admit everything — a floor of
zero images passes a cycle that bought nothing.

Two numbers are deliberately elsewhere. The confidence level is `bootstrap`'s constant rather than an
argument, so a caller cannot shop for a level that promotes. The label budget is on the run
registration, which is the write-once item that makes it fixed for the run.

---

## Data

Four conditions over what was bought, before anything is asked about the model.

| Condition | Rule |
|---|---|
| Leakage | Every purchased image is in the pool |
| Volume | The batch is at least `min_new_images` |
| Budget | The batch is within the run's per-cycle budget |
| Coverage | Every class has at least `min_instances_per_class` boxes in the batch |

Leakage is hard, with no override and no threshold to soften it. An `eval` image in the training set
invalidates every number the run has produced, and it does so silently — the metric goes up, which is
what a leak looks like.

It is checked twice on purpose. `oracle.cohorts` already refuses a batch reaching into `eval`, and
this re-runs the identical judgement on what the ledger says was actually bought. The oracle's check
runs on the list it was handed; this one runs on the labels that came back. A bug that lets the two
disagree is the bug worth catching, and the design calls this the single most valuable check in the
system (§4.1). It reuses `refusals` rather than restating it, so there is one implementation of what
`eval` means.

Coverage counts boxes bought *this* cycle, so it asks whether the batch teaches every class
something. Coverage of the cumulative training set is not in question: the bootstrap is 8,000 random
images.

Integrity checks are absent. Design §4.1 also lists corrupt files, wrong resolution and blank frames;
those decode a thousand JPEGs, and ingest already validated resolution and recorded a sha256 per
image. Distribution shift is not checked and never will be here — a concentrated snow purchase is the
selector working. The batch's composition is charted against the pool's base rates and gated on by
nothing.

---

## Quality

Three conditions over numbers the evaluation job has already computed.

| Condition | Rule |
|---|---|
| Uplift | Mean paired delta is at least `min_mean_delta` |
| Band | The lower end of the 95% resampling band clears zero |
| Collapse | No class scores zero AP |

Both statistical conditions are required, not either (design §4.2). The design predicts the two are
close to redundant and says to confirm it against real numbers, which is only possible if both are
evaluated and both are reported. That arithmetic was worked out over a seed spread, and a cycle now
trains one seed: the band resamples images rather than seeds, so the floor carries more of the
decision than the prediction assumed.

The band is the condition that makes this more than a demo. Most pipelines promote on a raw metric
bump sitting inside the noise. Requiring the interval to clear zero means sometimes correctly
refusing to promote, and a chart of honest rejections is the evidence the project is for.

Slices cast no vote. Per-slice scores are computed every cycle and charted, never gated (design
§4.4). Acquisition is condition-blind, so a cycle makes no per-condition bet for a per-condition gate
to settle. The collapse check is not a slice test: it asks whether the model produces output at all.

A run's first cycle has no champion, so the delta is absent and the verdict is the collapse check
alone. Reporting no verdict would leave `gates_passed` false for the only model that can become the
first champion, which is a run that cannot start.

---

## Edge

Two conditions, both properties of a file and a number. Neither needs a device, so the gate is
complete in the cloud.

| Condition | Rule |
|---|---|
| Size | The int8 artifact is at most `max_artifact_bytes` |
| Accuracy | It retains at least 95% of the fp32 model's mAP |

The accuracy check is relative, not absolute. A challenger that improved and one that regressed both
quantize about as well, so the question is what the conversion cost *this* model rather than where it
landed.

The 5% allowance is looser than design §4.3's 2%, which was written before anything had been
quantized. What the project needs from this gate is that a broken export cannot ship, and a broken
export is not a 3% model — it is a 30% one, or a graph that detects nothing. A threshold tight enough
to reject a slightly lossy but working artifact would stop the loop over a number the fleet would
never notice, and what quantization actually cost is in the reason either way. Tighten it once
several cycles have said what the real spread is.

The size ceiling is not a number the export is near — a quantized YOLO11n is a few megabytes — which
is the point: it catches a cycle that shipped the fp32 graph under the int8 filename.

Speed is reported, not gated (design §4.3). Gating on latency would mean a promotion decision that
cannot be made until the model has already been deployed.

---

## Canary

Three conditions over one device's replay, all operational rather than statistical.

| Condition | Rule |
|---|---|
| Digest | The device loaded the artifact the manifest names |
| Completion | The component started once and the replay finished |
| Throughput | Frame rate is within `max_throughput_drop` of the champion's |

A detector is deterministic, so there is no run-to-run spread on one device and any "within N
standard deviations" test over it is vacuous (design §4.5). What can genuinely fail is the plumbing.

The digest is the check the other two are worthless without. Greengrass verifies its own copy against
its own recipe on download, which is a closed loop that cannot catch a recipe built over the wrong
object; this is the independent half. A mismatch is a rollout of a model no gate ever saw.

Throughput is the only comparison here and the only check needing a champion. It catches a graph that
reached the device intact and runs at a fraction of the speed — which every cloud pass misses, since
the int8 pass scores on a cloud CPU and what ARM silicon makes of the same operators is not a
question that hardware answers.

---

## Incomplete

Design §4.5 lists five canary conditions and three are implemented. Memory flatness over two replay
hours needs the hours; the distance between two confidence distributions needs a champion replaying
the same frames at the same time, which is shadow mode and a second component. They are absent rather
than approximated, because a leak test over four minutes and a distribution distance over one sample
pass by construction and then report themselves as evidence.

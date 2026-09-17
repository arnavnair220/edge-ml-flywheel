# Stage 5 — Fleet and deployment

Publishes a promoted model as a Greengrass component, deploys it to one Graviton device, and runs the
cycle's pass over the unlabeled pool there. One Terraform file, one Python package and a CLI; no
daemon. See the [architecture overview](00-overview.md) for the stage's position in the loop.

**Inference over unseen frames happens on the device, not in the cloud.** The pool is the footage the
fleet drives through, so the model that scores it is the quantized artifact actually deployed, on the
hardware actually running it. What comes back is predictions and latencies, never pixels, which is
the direction traffic runs in a real fleet.

Greengrass performs the agent loop the design describes — verify the digest, install, roll back on a
failed install. This stage supplies what it has no opinion about: which frames the device scores,
what it reports, and whether that report is good enough to leave the model deployed.

---

## Device

| Element | Value |
|---|---|
| Instance | `t4g.small`, one, in a thing group of one |
| Image | Amazon Linux 2023 arm64, from the public SSM parameter |
| Network | Public subnet, public IP, no inbound rule, no NAT gateway |
| Access | Session Manager |
| Nucleus | Greengrass v2 `2.14.3`, installed at first boot as a systemd unit |
| Interpreter | `/opt/edge-ml-flywheel/venv`, built at first boot from `infra/device-requirements.txt` |

`t4g.small` rather than the design's `t4g.nano`: a JVM nucleus needs 256 MB before the scoring
component's onnxruntime session, so 512 MB is short.

**The instance runs for the length of a run, not the length of a canary.** Every cycle blocks on it,
so it is started when a run opens and stopped when the run ends, at roughly $16/month against a $40
alarm — the instance at $12.30, its public IPv4 at $3.65, and the volume at $1.60. Stopping it
between runs releases the address and leaves the volume alone. `Deploy` refuses a cycle whose device
is stopped, rather than entering the wait and timing out an hour later with nothing to show.

The networking itself is free. An EC2 instance must sit in a subnet, so the VPC is not optional, but
a VPC, subnet, internet gateway, route table and security group carry no charge. What the public
address replaces is a NAT gateway at roughly $32/month, or interface endpoints at $7.30 each.

The thing group holds one device and is still the deployment target, so a second device joins without
any deployment code changing. A second device would split the sample rather than score it twice; the
draw is per cycle and the split is not built.

---

## Component

One component per run, one version per cycle, carrying the model and the code that exercises it.

| Element | Value |
|---|---|
| Name | `edge-ml-flywheel.<run_id>` |
| Version | `0.<cycle>.0` |
| Artifacts | the seed's `model.onnx`, the cycle's `replay.json`, the commit's `replay.zip` |
| Platform | `linux/aarch64` |
| Lifecycle | `Run`, exiting zero when the pool pass finishes |

A component version must be semver and a model version is not one, so the run goes in the name and
the cycle in the version. The cycle is the one number in the project that is **not** zero-padded:
semver forbids a leading zero and compares those identifiers numerically.

Two runs therefore cannot collide at one cycle. A component version is immutable once created, so
that collision would be permanent.

**The version is the cycle that deployed it, not the cycle that trained the model.** Every cycle
publishes one, including a cycle that rejected its challenger and is redeploying the standing
champion: the recipe names that cycle's sample, so the frames differ even when the model does not,
and an immutable version cannot be amended to say so. A champion deployed across three cycles is
therefore three component versions over one `model.onnx`.

That makes the number in `0.<cycle>.0` a poor answer to "what is this device running", so the
deployment carries the model version in its configuration and every read takes it from there.

The design's separate replay component is folded in. At one device the split costs a mechanism for
discovering which version is deployed and buys a redeploy of one without the other, which never
happens.

---

## The pool pass

`POOL_SAMPLE` frames after 50 warmup, batch size 1, at the image size the model was exported against.
Ten thousand frames, which is the number that keeps a cycle's wait under an hour at the device's
measured per-frame cost; it is a constant to raise once several cycles have reported their own
throughput, not a figure the design derives.

| Property | Value |
|---|---|
| Frames | `POOL_SAMPLE`, 10,000 |
| Warmup | 50, excluded from every reported latency, scored like any other frame |
| Drawn from | the remaining pool: the cohort minus everything bought to date |
| Seeded by | run and cycle |
| Precision | int8, the artifact the component holds |
| Confidence floor | `BAND_LOW`, not the cloud pass's 0.001 |

The floor is the one knob that differs from the cloud pass, and the ranking is identical either way.
`image_score` keeps only detections at or above `BAND_LOW` and decides blind-spot against decisive on
that filtered list, so a row at 0.02 feeds into no score; suppression is greedy and highest-first, so
a box below the band can be suppressed but never suppresses one above it. What the device buys by not
writing the tail is a file an order of magnitude smaller and a bounded working set on two cores.

Warmup frames are scored and not timed. Their detections belong in the file — a frame left out of the
ranking for having been drawn first would be a hole in the sample nothing recorded — while their
latencies describe a cold session rather than the device.

Frames are drawn at random from the remaining pool rather than from the top of a ranking, because
this pass *is* what produces the ranking — there is nothing to be at the top of yet. Unbought rows
only, since a bought image has labels and is in the training set rather than the catalogue.

The draw is seeded by the run and the cycle, so a redeploy after a rollback scores identical frames
and the second pass is comparable to the first. The list is written to
`replay_manifest_key(run_id, cycle)` before the deployment and shipped as a component artifact. It is
the per-cycle record of sampled image IDs design §7.2 requires, and what ties every later number —
a telemetry latency, a ranking row, a purchase — back to one frame.

Ten thousand of a shrinking 62,000 is the sampled fraction, so a cycle ranks the part of the pool it
saw and buys 1,000 out of that. The alternative is the whole remaining pool every cycle, which is
hours of ARM inference for a ranking whose top 1,000 is decided by its tail. It is also the arrangement
a fleet has anyway: a device ranks the footage it drove, not the catalogue.

---

## What comes back

Two paths, because the two outputs have different failure tolerances.

| Output | Path | Lands at |
|---|---|---|
| Predictions | the device writes one parquet to S3 | `detections_key(version, seed, pool, int8)` |
| Latencies and the summary | the device publishes to its IoT topic | `fleet/run_id=<id>/dt=<date>/` |

**The predictions go to S3 because the ranking is bought from them.** MQTT delivery is lossy by
design and the telemetry format already admits it — a summary claiming 10,000 frames beside batches
carrying 9,400 is a recognized outcome. A ranking is not something to compute off a channel that can
silently drop its middle. The file is written once, whole, and the pass's completeness is carried by
the task token rather than by a count: a device that died partway writes no parquet, publishes no
summary and resumes nothing, so the cycle times out instead of ranking a fraction of a sample.

The predictions land in the same key family the cloud pass writes, under `cohort=pool` and
`precision=int8` — the same `DetectionRow` schema, one parquet, one part. The key carries the
deploying cycle rather than the model's own: each cycle draws its own sample, so one champion
deployed across three cycles produces three different files, and keying them by the model's cycle
would have the second overwrite the ranking the first purchase was charged against.

`Precision` means something different per cohort. For `eval` the fp32 pass is the measurement and
int8 is the comparison the edge gate reads; for `pool` int8 is the only pass there is, because the
only model that scores the pool is the one on the device.

Telemetry carries the operational half. Frames carry `image_id` and `inference_ms`; the confidences
are in the parquet, so publishing them twice would be a bigger message saying the same thing.

| Record | Carries |
|---|---|
| `frames` | up to `TELEMETRY_BATCH` frames, each `image_id` and `inference_ms` |
| `replay` | the digest the device hashed, cold start, starts, frames scored, and the detections object it wrote |

The summary is published last, so its presence means the pass finished rather than that it began. At
100 frames per batch a cycle is a hundred objects and a summary, which is still a pandas read and
still not a Firehose — the landing prefix is the one a delivery stream would later write to
unchanged.

The topic's third and fourth segments are the run and the device, and the rule's S3 key template
addresses them by position. That layout is a contract with [infra/fleet.tf](../infra/fleet.tf) and is
tested as one.

---

## Returning to the cycle

`FleetScore` is a `waitForTaskToken` state, and the device is what resumes it. The device calls
`SendTaskSuccess` after its parquet is durable and its summary is published — in that order, so a
resumed execution cannot read a file that is not there.

**The token rides the deployment, not the artifacts.** A component version is published before the
cycle reaches the state that issues a token and is immutable once it exists, so the token cannot be
an artifact of it. It arrives instead as a configuration update on the deployment, which is created
after the token exists and is the only half of the pair that can carry a value that late. That is
also why `Deploy` and `FleetScore` are two states: one publishes, the other deploys with the token
and waits.

**The token is the completion signal, and nothing else is.** A pass that dies partway calls nothing,
so the cycle cannot proceed on a partial file — which is why `select` does not have to count the
device's rows against the draw. A failure the device can see is `SendTaskFailure` with the cause, so
the execution stops in seconds rather than in hours. Silence is reserved for a device that cannot
report at all, and the state's two-hour timeout is what catches that: the execution fails with the
wait exhausted, the champion stays deployed, and nothing is bought.

The token rather than an IoT rule invoking a Lambda that resumes the execution, because the device
already knows the two things the cycle is waiting on. A rule would be a second place for the "is it
finished" judgement to live, and it would have to re-derive from a message what the device knows
directly.

---

## Canary gate

Asked inside the cycle, immediately after the pass the device just finished, because the execution
is already waiting on hardware and the frames are already scored. Three checks over one device's
pass. The predicates are in [gates.md](gates.md); what belongs here is what the device supplies them.

| Check | Fails when |
|---|---|
| Digest | the device's hash of the loaded file differs from `ModelManifest.artifact_sha256` |
| Completion | the component started more than once, or fewer frames arrived than the summary claimed |
| Throughput | frames per second fell more than 10% below the champion's |

A run's first deployment has no champion and passes the throughput check as its own baseline.

p95 latency and cold start are reported in the verdict's reason, never gated (design §4.3). Cold
start is the session plus the first inference, excluding interpreter startup, which costs the same
for every version. Latency is measured over 10,000 frames, so the p95 is a percentile rather than an
estimate of one.

**A failed canary and a failed pass are different things.** Throughput is a property of the rollout:
the model ran correctly and slowly, the deployment is rolled back, and the cycle ranks and buys from
what it scored. Digest and completion are properties of the scores: the bytes that ran are not the
bytes the gates were reported over, or the file is short. Those roll back *and* refuse the purchase,
because a ranking from an unidentified model is not a ranking. That is the one case in the project
where a cycle spends nothing, and it is recorded with its reason like any other refusal.

---

## Deployment and rollback

A deployment names one component version on the thing group, with `failureHandlingPolicy: ROLLBACK`.
A device that cannot install or start the new component returns to the one it was running.

Deployment happens at `Promote`, before the pool pass that depends on it. A cycle that promoted
deploys its challenger; a cycle that did not leaves the standing deployment alone and scores with the
champion. Either way the pool is ranked by whatever is deployed, which is the definition the
selector already carries.

**The Greengrass deployment is the record of intent.** The design left this open between
`fleet_config` and the deployment; this stage settles it on the deployment. No `desired_version` is
written and `fleet_config` holds no device item.

**A rollback is the same call naming the previous version.** There is no separate rollback path, so
the path a rollback takes is the one every cycle has already exercised.

---

## Permissions

| Principal | Grant |
|---|---|
| Device certificate | connect as itself, publish under its own name, exchange itself for a session |
| Token exchange role | read train-split images and its three artifacts, write its own detections prefix, publish its own telemetry, resume its own execution |
| IoT rule role | write one telemetry prefix, read nothing |
| Instance profile | read its own two SSM parameters, and Session Manager |

**A device sits inside the label wall exactly as a training job does.** The token exchange role
carries an explicit deny over every label prefix, and the data bucket's allowlist refuses it besides.
Its image grant is the `train` split alone, since `eval` is drawn from `val`. What it writes is
predictions: a box and a confidence are functions of an image and a model, so the write grant opens
no path to ground truth. See [infra/fleet.tf](../infra/fleet.tf).

`states:SendTaskSuccess` and `states:SendTaskFailure` are the two grants the round trip adds, and
they are not scoped to an execution — a task token is the capability, and holding one is what
authorizes resuming the state that issued it.

---

## Commands

```
python -m edge_ml_flywheel.fleet deploy   --run-id <id> --version <version> --git-commit <sha> [--cycle <n>]
python -m edge_ml_flywheel.fleet canary   --run-id <id> --version <version> --champion <version>
python -m edge_ml_flywheel.fleet rollback --run-id <id> --to <version>
python -m edge_ml_flywheel.fleet status   --run-id <id>
```

**The cycle performs all of this itself.** These are the hand-operated copy, for the cases a state
machine is the wrong tool for: re-running a pass after a device failure, restoring a version once the
execution that deployed it has ended, and looking at what the fleet is doing. Each goes through the
same functions the control steps call, so an operator and the state machine cannot form two opinions
about one rollout.

`--cycle` is how a deployment by hand says which cycle it is for, when that is not the cycle the
model was trained in. `canary` exits non-zero on a failed verdict, so a shell can chain a rollback
behind it, and prints `ranks` beside `passed` — a slow model is rolled back and its detections are
still bought from. `status` prints what the deployment names beside what the device reported.

A pass run this way carries no task token and resumes nothing, which is correct: the execution that
issued one has long since timed out.

---

## Incomplete

The promotion state machine is candidate → champion → archived. The intermediate shadow and canary
states land with the features that need them.

The device's own ranking is not charted yet. `fleet.telemetry.image_ids` joins onto
`selection_ranking_key` by image ID, and what the chart would show is the distribution of latency
against uncertainty — where the model is slow *and* unsure. It lands with the charts.

`starts` is a counter in the component's work directory, which Greengrass preserves across a restart
and across a new component version. It is keyed by **cycle**, not by model version: a cycle that
rejects its challenger redeploys the champion, so one model version can be three cycles' deployments,
and a counter keyed by it would report a clean install as a restart and fail the completion check for
something that never happened. A cycle deploys exactly once, which is what makes it the key.

Nothing in the suite covers it, because `fleet.replay` imports onnxruntime and the tests run without
it — the same line `detect` sits on the testable side of.

A run's first cycle must promote to have anything to score with. The gates make that near-certain —
with no champion the quality gate is the collapse check alone — but a first cycle that fails it ends
with nothing deployed and buys nothing, rather than falling back to a cloud pass that would rank by a
model the fleet never ran.

---

## Deferred

Two of design §4.5's five conditions are absent rather than approximated: a leak test over four
minutes and a distribution distance over one sample both pass by construction. See
[gates.md](gates.md).

| Deferred | Needs |
|---|---|
| Staged rollout, one device then two then the group | a second device |
| Splitting the sample across devices | a second device |
| Shadow mode | a second component scoring the same frames |
| Population stability index between confidence distributions | shadow mode |
| Memory flatness over two scoring hours | a pass long enough to leak |
| Scoring the whole remaining pool each cycle | throughput the measurements do not yet support |
| Firehose, parquet conversion and an Athena table | telemetry too large for pandas |
| Fleet provisioning | hardware that enrolls itself |

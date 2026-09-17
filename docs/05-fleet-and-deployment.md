# Stage 5 — Fleet and deployment

Publishes a promoted model as a Greengrass component, deploys it to one Graviton device, and reads
back what the device saw. One Terraform file, one Python package and a CLI; no state machine and no
daemon. See the [architecture overview](00-overview.md) for the stage's position in the loop.

Greengrass performs the agent loop the design describes — verify the digest, install, roll back on a
failed install. This stage supplies what it has no opinion about: which frames the device replays,
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

`t4g.small` rather than the design's `t4g.nano`: a JVM nucleus needs 256 MB before the replay
component's onnxruntime session, so 512 MB is short.

**Stop the instance between cycles.** It is only needed while a cycle is canaried, and running it
costs roughly $16/month against a $40 alarm — the instance at $12.30, its public IPv4 at $3.65, and
the volume at $1.60. Stopping it releases the address and leaves the volume alone.

The networking itself is free. An EC2 instance must sit in a subnet, so the VPC is not optional, but
a VPC, subnet, internet gateway, route table and security group carry no charge. What the public
address replaces is a NAT gateway at roughly $32/month, or interface endpoints at $7.30 each.

The thing group holds one device and is still the deployment target, so a second device joins without
any deployment code changing.

Identity is created by Terraform rather than by fleet provisioning, which would put an IoT identity
outside state. The certificate reaches the instance through Parameter Store, read once at first boot;
the private key is in Terraform state, which is the accepted cost.

---

## Component

One component per run, one version per cycle, carrying the model and the code that exercises it.

| Element | Value |
|---|---|
| Name | `edge-ml-flywheel.<run_id>` |
| Version | `0.<cycle>.0` |
| Artifacts | the seed's `model.onnx`, the cycle's `replay.json`, the commit's `replay.zip` |
| Platform | `linux/aarch64` |
| Lifecycle | `Run`, exiting zero when the replay finishes |

A component version must be semver and a model version is not one, so the run goes in the name and
the cycle in the version. The cycle is the one number in the project that is **not** zero-padded:
semver forbids a leading zero and compares those identifiers numerically.

Two runs therefore cannot collide at one cycle. A component version is immutable once created, so
that collision would be permanent.

The design's separate replay component is folded in. At one device the split costs a mechanism for
discovering which version is deployed and buys a redeploy of one without the other, which never
happens.

---

## Replay

500 pool frames after 50 warmup, batch size 1, at the size and confidence floor the cloud pass used.

Frames are drawn at random from the rows the cycle's own purchase left behind. Random rather than the
top of the ranking, or every reported confidence would come from the low-confidence tail. Unbought
rows only, because the bought ones have labels by the time the model is deployed.

The draw is seeded by the run and the cycle, so a redeploy after a rollback replays identical frames.
The list is written to `replay_manifest_key(run_id, cycle)` and shipped as a component artifact. It
is the per-cycle record of sampled image IDs design §7.2 requires, and what ties a telemetry record
back to a scoring decision.

---

## Telemetry

The device publishes to `edge-ml-flywheel/fleet/<run_id>/<thing>`. An IoT rule writes each message to
`fleet/run_id=<id>/dt=<date>/` in the telemetry bucket. Nothing on the device writes a file.

| Record | Carries |
|---|---|
| `frames` | up to 100 frames, each `image_id`, `inference_ms`, and the confidence of each detection |
| `replay` | the digest the device hashed, cold start, starts, and how many frames it replayed |

The summary is published last, so its presence means the run finished rather than that it began.
Frames are batched at 100, which makes a replay six objects rather than five hundred.

The topic's third and fourth segments are the run and the device, and the rule's S3 key template
addresses them by position. That layout is a contract with [infra/fleet.tf](../infra/fleet.tf) and is
tested as one.

No Firehose. One device publishes six objects per cycle, the reader is pandas, and a delivery stream
would buffer five minutes to write one small object. The landing prefix is the one a Firehose would
later write to unchanged.

---

## Canary gate

Read after promotion, and the one gate that cannot be asked in the cloud: it needs an artifact that
has reached a device. What it decides is whether the rollout stands. The predicates are in
[gates.md](gates.md); what belongs here is what the device supplies to them.

| Check | Fails when |
|---|---|
| Digest | the device's hash of the loaded file differs from `ModelManifest.artifact_sha256` |
| Completion | the component started more than once, or fewer frames arrived than the summary claimed |
| Throughput | frames per second fell more than 10% below the champion's |

A run's first deployment has no champion and passes the throughput check as its own baseline.

p95 latency and cold start are reported in the verdict's reason, never gated (design §4.3). Cold
start is the session plus the first inference, excluding interpreter startup, which costs the same
for every version.

---

## Deployment and rollback

A deployment names one component version on the thing group, with `failureHandlingPolicy: ROLLBACK`.
A device that cannot install or start the new component returns to the one it was running.

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
| Token exchange role | read train-split images and its three artifacts, publish its own telemetry |
| IoT rule role | write one telemetry prefix, read nothing |
| Instance profile | read its own two SSM parameters, and Session Manager |

**A device sits inside the label wall exactly as a training job does.** The token exchange role
carries an explicit deny over every label prefix, and the data bucket's allowlist refuses it besides.
Its image grant is the `train` split alone, since `eval` is drawn from `val`. See
[infra/fleet.tf](../infra/fleet.tf).

---

## Commands

```
python -m edge_ml_flywheel.fleet deploy   --run-id <id> --version <version> --git-commit <sha>
python -m edge_ml_flywheel.fleet canary   --run-id <id> --version <version> --champion <version>
python -m edge_ml_flywheel.fleet rollback --run-id <id> --to <version>
python -m edge_ml_flywheel.fleet status   --run-id <id>
```

`canary` exits non-zero on a failed verdict, so a shell can chain a rollback behind it. `status`
prints what the deployment names beside what the device reported.

Deploy after the cycle reaches `Purchase`, not at `Promote`. `Select` writes the ranking the replay
sample is drawn from and runs after `Promote`; deploying inside that window is refused with that
explanation rather than with a missing key.

---

## Incomplete

Deployment is driven by a CLI rather than by the cycle state machine. A canary is asked after the
execution has ended, so folding it in would mean a state machine waiting on hardware every cycle.
See [control.md](control.md).

The promotion state machine is candidate → champion → archived. The intermediate shadow and canary
states land with the features that need them.

The fleet's own ranking over the frames it replayed is not reported yet.
`fleet.telemetry.image_ids` is this stage's half of that join; the other half is a query over
`selection_ranking_key`, and it lands with the charts when reporting is built.

`starts` is a counter in the component's work directory, which Greengrass preserves across a restart.
It is keyed by model version, so it counts this cycle's restarts and not the previous cycle's.

---

## Deferred

Two of design §4.5's five conditions are absent rather than approximated: a leak test over four
minutes and a distribution distance over one sample both pass by construction. See
[gates.md](gates.md).

| Deferred | Needs |
|---|---|
| Staged rollout, one device then two then the group | a second device |
| Shadow mode | a second component scoring the same frames |
| Population stability index between confidence distributions | shadow mode |
| Memory flatness over two replay hours | a replay long enough to leak |
| Firehose, parquet conversion and an Athena table | telemetry too large for pandas |
| Fleet provisioning | hardware that enrolls itself |

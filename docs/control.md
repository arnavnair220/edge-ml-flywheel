# Control

One Step Functions state machine, and the only place a cycle's order is written down. See the
[architecture overview](00-overview.md) for its position in the loop.

**One execution is a whole run.** The machine describes a single cycle and then loops: `MoreCycles`
returns to `ClaimCycle` while the cap is unspent and the pool is not empty, so a run of eight cycles
is one execution with eight passes through the same states, not eight executions.

Control is not a stage a cycle passes through. It sequences the five that are, owns their retries
and branching, and holds the lock that stops two cycles overlapping.

---

## The state machine

[infra/cycle.asl.json](../infra/cycle.asl.json), twenty-two states in JSONata. Every SageMaker call
the loop makes is in this file, so the execution history is the record of what ran.

| State | Type | What it does |
|---|---|---|
| `ClaimCycle` | DynamoDB `UpdateItem` | Claims the next cycle number and returns the champion pointer |
| `Prepare` | Lambda | Writes the cycle's image manifest and source archive |
| `Train` | Map over seeds | `TrainingRequest` builds, `TrainSeed` runs `createTrainingJob.sync` |
| `ScorePrepare` | Lambda | Writes the eval manifest and the pool sample the device will score |
| `Score` | Map over seeds | `ScoringRequest` builds, `ScoreSeed` runs `createProcessingJob.sync` over `eval` |
| `ScoreQuantizedRequest` | Lambda | Builds the int8 pass over `eval` |
| `ScoreQuantized` | Processing `.sync` | Runs it |
| `EvaluateRequest` | Lambda | Builds the match-and-compare job over every seed at once |
| `Evaluate` | Processing `.sync` | Runs it; the data, quality and edge gates are applied inside |
| `Register` | Lambda | Writes the model manifest, returns the verdict and the package request |
| `OpenTheRegistryGroup` | SDK | One package group per run, opened by the first cycle |
| `RegisterTheVersion` | SDK | Records the verdict as a model package |
| `Passed` | Choice | Promote, or straight to deployment |
| `Promote` | DynamoDB `UpdateItem` | Flips the champion pointer |
| `Deploy` | Lambda | Publishes the component version and creates the deployment, or confirms the standing one |
| `FleetScore` | Lambda `.waitForTaskToken` | Hands the device the token and waits for its pass over the pool sample |
| `Canary` | Lambda | Reduces the telemetry, applies the canary gate, rolls back a failure |
| `Ranked` | Choice | Buy from what the device scored, or end the cycle with the reason it cannot |
| `Select` | Lambda | Ranks the device's detections and writes what the cycle chose to buy |
| `Purchase` | Lambda (oracle) | Charges the ledger and files the labels |
| `MoreCycles` | Choice | Round again, or stop |
| `Done` | Succeed | Ends the run |

The five states from `Promote` to `Select` are one round trip to the fleet: deploy whatever is now
champion, let the device score a sample of the unlabeled pool with it, read that pass twice — once as
the canary gate and once as the ranking — and buy from it. Inference over unseen frames is the
device's work rather than a second cohort of the cloud job, which is why `Score` covers `eval` alone.
See [05-fleet-and-deployment.md](05-fleet-and-deployment.md).

---

## The single-flight lock

`ClaimCycle` is one conditional `UpdateItem` that adds one to `next_cycle` on the run's control item
and returns the item as it was. The number handed back is the cycle this execution owns and no other
execution can own.

The condition carries the cap as well, so a run that has spent its cycles is refused here rather
than by a check upstream that has to remember to run. `ALL_OLD` rather than `UPDATED_OLD` because the
cap and the champion pointer are read off the same call, so nothing reads the item twice.

A read followed by a write would be the same code with a race in it, and that race — a double-fired
tick opening a second cycle against one budget — is the case worth refusing. See
[run/control.py](../src/edge_ml_flywheel/run/control.py).

---

## The control function

[control/handler.py](../src/edge_ml_flywheel/control/handler.py). One Lambda, one entry point,
dispatching on a `step` name: `prepare`, `train_request`, `score_prepare`, `score_request`,
`evaluate_request`, `register`, `deploy`, `fleet_score`, `canary`, `select`.

One function rather than ten. The steps share a session, the bucket names and the run
registration; none is hot, large or differently privileged. Ten functions would be ten packages,
ten log groups and ten roles for a dispatch that is a dictionary lookup. The step name is in the
ASL, so a typo is a failed execution naming the step it could not find.

`register` and `select` are the two steps that write something read back later — the manifest, which
is the precondition of promoting, and the ranking, which the purchase is charged against. The rest
produce inputs to a job about to run, or act on the device.

**`fleet_score` hands over a token and returns.** It writes the task token into the cycle's replay
manifest, confirms the device is running, and exits; the execution stays in the state until the
device calls `SendTaskSuccess`. A Lambda that waited for the pass would be a Lambda timing out at
fifteen minutes against a pass measured in tens of them.

**`deploy` and `canary` are the two steps that act on hardware.** They are the same three calls the
fleet CLI makes — publish a component version, create a deployment, reduce the telemetry — and they
are in this function rather than in a fleet Lambda of their own because the alternative is a second
package holding one import.

**The `*_request` steps build and do not call.** A Lambda that started a training job would either
wait ninety minutes for it or hand the polling back to the state machine anyway, and `.sync` already
owns waiting, retrying and stopping the job if the execution aborts. So the step returns the request,
and the Lambda's role holds no SageMaker grant and cannot pass the training role.

**The purchase is a different function.** It is the eighth step of a cycle and it runs under its own
role, because the control function is denied `raw/labels/` outright and the oracle is the single
principal that must read exactly those files. See
[oracle/handler.py](../src/edge_ml_flywheel/oracle/handler.py) and
[01-data-and-labels.md](01-data-and-labels.md).

---

## Branching

`Passed` is the short-circuit, and it is narrower than it sounds. A failed gate leaves the champion
in place, but the cycle carries on through the fleet round trip to `Select` and `Purchase` — the
labels stay bought, and the next challenger simply has more to learn from. **No verdict on a model's
quality costs the run its purchase.** What the cycle skips on a rejection is `Promote` and a new
deployment, so the champion is what scores the pool that cycle.

`Ranked` is the one branch that can end a cycle having bought nothing, and what decides it is whether
the detections can be trusted rather than whether the model was good. A canary that failed on
throughput still ranks: the model ran correctly and slowly, the rollout is rolled back, and the
detections are the detections. A canary that failed on the digest or on a short file does not,
because a ranking computed from an unidentified model is not a ranking. The cycle ends with that
reason recorded rather than buying 1,000 labels off it. See
[05-fleet-and-deployment.md](05-fleet-and-deployment.md).

`MoreCycles` goes round while the cap is unspent and the pool is not empty. The cap is checked here
and again in `ClaimCycle`, deliberately: this decides whether to go round, that decides whether going
round is permitted, and only one of them is a lock.

`OpenTheRegistryGroup` catches every error and continues. Every cycle after the first finds the group
already there, and what SageMaker calls that failure is a validation error whose message is not a
contract worth matching on.

---

## Retries

| States | Retries |
|---|---|
| `TrainSeed`, `ScoreSeed`, `ScoreQuantized`, `Evaluate` | `States.TaskFailed`, 2 attempts, 60 s, backoff 2 |
| Every Lambda state | Service, SDK-client and throttling faults, 3 attempts, 5 s, backoff 2 |
| `ClaimCycle`, `Promote` | DynamoDB throttling and internal errors, 4 attempts, 2 s, backoff 2 |
| `RegisterTheVersion` | SageMaker throttling and SDK-client faults, 3 attempts, 5 s, backoff 2 |
| `FleetScore` | `TimeoutSeconds` 7,200, no retry |

Jobs are retried because they read frozen inputs and overwrite their own outputs, so a restart is
safe.

`FleetScore` has a timeout and no retry, which is the opposite arrangement. The timeout is the only
thing standing between a device that cannot report and an execution waiting forever, and two hours is
several times the measured pass against a device that is meant to be running before the cycle starts.
A retry would re-deploy and re-score on the same hardware that just failed to answer, so the pass is
re-run by hand once the device is understood.

Nothing retries a refusal. `ControlError` and `OracleError` are the shape of failure that fails
again — an unregistered run, a manifest that already exists, an exhausted budget — so a second
attempt spends the execution's time to report the same thing. `ClaimCycle` is the same rule at the
storage layer: a failed condition means the cap is spent or another execution holds the run, and
neither changes on a retry.

---

## Permissions

Two roles. The control function reads the run registration and the project bucket, and holds no
SageMaker grant and no `iam:PassRole` — it cannot start a job even by mistake. The fleet steps add
the Greengrass create-component and create-deployment grants, `iot:DescribeEndpoint` so the recipe
can name the address the device publishes to, `ec2:DescribeInstances` on the device so `Deploy` can
refuse a stopped one, and read access to the telemetry prefix. It still cannot pass a role or start a
job.

The state machine role holds the SageMaker create, describe and stop grants for training jobs,
processing jobs and model packages, `dynamodb:UpdateItem` on the fleet config table,
`lambda:InvokeFunction` on the two functions, and `iam:PassRole` scoped to the training, scoring and
evaluation roles. See [infra/cycle.tf](../infra/cycle.tf).

The device resumes the execution under its own token exchange role, which holds
`states:SendTaskSuccess` and `states:SendTaskFailure` and nothing else about Step Functions. It
cannot start, stop or inspect an execution — a task token is the capability, and the only execution
it can affect is the one that handed it one. See
[05-fleet-and-deployment.md](05-fleet-and-deployment.md).

---

## Commands

The CLI registers a run and starts it in one step:

```
python -m edge_ml_flywheel.run start --epochs <n> [--seeds 1] [--max-images <n>]
```

It prints the run ID on its own line before the execution ARN, so a failure to start still leaves
the operator holding the name of the run that now exists. `register` mints a run without starting
it, for the case where it has to exist first.

The execution input the CLI builds:

```json
{"run_id": "<id>", "epochs": 30, "seeds": [1]}
```

`max_images`, `replace` and `instance_type` are optional. The last three apply to training and
scoring alike, since both are capped and re-prepared together.

---

## Incomplete

**The execution waits on hardware.** A cycle cannot finish while the device is down, because nothing
else scores the pool. The two hours in `FleetScore` bound how long a run stalls before it fails and
says so; they do not let it continue without the fleet. There is no cloud fallback, by design: a
ranking from a model the fleet never ran is what this arrangement exists to prevent.

A run's first cycle must promote to have a model on the device. See
[05-fleet-and-deployment.md](05-fleet-and-deployment.md).

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

[infra/cycle.asl.json](../infra/cycle.asl.json), eighteen states in JSONata. Every SageMaker call
the loop makes is in this file, so the execution history is the record of what ran.

| State | Type | What it does |
|---|---|---|
| `ClaimCycle` | DynamoDB `UpdateItem` | Claims the next cycle number and returns the champion pointer |
| `Prepare` | Lambda | Writes the cycle's image manifest and source archive |
| `Train` | Map over seeds | `TrainingRequest` builds, `TrainSeed` runs `createTrainingJob.sync` |
| `ScorePrepare` | Lambda | Writes the two manifests naming what this cycle scores |
| `Score` | Map over seeds | `ScoringRequest` builds, `ScoreSeed` runs `createProcessingJob.sync` |
| `ScoreQuantizedRequest` | Lambda | Builds the int8 pass over `eval` |
| `ScoreQuantized` | Processing `.sync` | Runs it |
| `EvaluateRequest` | Lambda | Builds the match-and-compare job over every seed at once |
| `Evaluate` | Processing `.sync` | Runs it; the data, quality and edge gates are applied inside |
| `Register` | Lambda | Writes the model manifest, returns the verdict and the package request |
| `OpenTheRegistryGroup` | SDK | One package group per run, opened by the first cycle |
| `RegisterTheVersion` | SDK | Records the verdict as a model package |
| `Passed` | Choice | Promote, or straight to selection |
| `Promote` | DynamoDB `UpdateItem` | Flips the champion pointer |
| `Select` | Lambda | Ranks the pool and writes what the cycle chose to buy |
| `Purchase` | Lambda (oracle) | Charges the ledger and files the labels |
| `MoreCycles` | Choice | Round again, or stop |
| `Done` | Succeed | Ends the run |

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
`evaluate_request`, `register`, `select`.

One function rather than seven. The steps share a session, the bucket names and the run
registration; none is hot, large or differently privileged. Seven functions would be seven packages,
seven log groups and seven roles for a dispatch that is a dictionary lookup. The step name is in the
ASL, so a typo is a failed execution naming the step it could not find.

`register` and `select` are the two steps that write something read back later — the manifest, which
is the precondition of promoting, and the ranking, which the purchase is charged against. The other
five produce inputs to a job about to run.

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
in place, but the cycle carries on to `Select` and `Purchase` — the labels stay bought, and the next
challenger simply has more to learn from. There is no state in which a rejection costs the run its
purchase.

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

Jobs are retried because they read frozen inputs and overwrite their own outputs, so a restart is
safe.

Nothing retries a refusal. `ControlError` and `OracleError` are the shape of failure that fails
again — an unregistered run, a manifest that already exists, an exhausted budget — so a second
attempt spends the execution's time to report the same thing. `ClaimCycle` is the same rule at the
storage layer: a failed condition means the cap is spent or another execution holds the run, and
neither changes on a retry.

---

## Permissions

Two roles. The control function reads the run registration and the project bucket, and holds no
SageMaker grant and no `iam:PassRole` — it cannot start a job even by mistake.

The state machine role holds the SageMaker create, describe and stop grants for training jobs,
processing jobs and model packages, `dynamodb:UpdateItem` on the fleet config table,
`lambda:InvokeFunction` on the two functions, and `iam:PassRole` scoped to the training, scoring and
evaluation roles. See [infra/cycle.tf](../infra/cycle.tf).

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

The canary gate is not a state. It is asked after the execution has ended, because a device has to
have replayed first; folding it in would mean a state machine waiting on hardware every cycle. See
[gates.md](gates.md) and [05-fleet-and-deployment.md](05-fleet-and-deployment.md).

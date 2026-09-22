# Running it

Bring-up, start to finish. See the [architecture overview](00-overview.md) for what any of these
steps is for.

Running it is not a stage. Everything below is either a one-time setup step or a command against
infrastructure the stages own, and none of it decides anything a stage has not already decided.

---

## What you need

| Requirement | Detail |
|---|---|
| AWS account | One account, one region. `aws_region` defaults to `us-east-1`; a run that crosses regions pays for the crossing |
| SageMaker quota | `ml.g4dn.xlarge` on **both** training and processing job usage, at least 1 of each |
| Local tooling | Python 3.12, `uv`, Terraform 1.10 or later, the AWS CLI |
| Data license | BDD100K is UC Regents, non-commercial research use. Accept it before ingest downloads anything |

The two SageMaker quotas are counted in separate buckets and do not share. Training is a training
job and scoring is a processing job, so a training allowance of 1 and a processing allowance of 0
trains a model and then fails the cycle at `Score`. Check both before starting a run:

```bash
aws service-quotas get-service-quota --service-code sagemaker --quota-code L-2F1EB012  # processing
```

---

## The shape of it

Four steps, in order. The first three happen once per account; the fourth is what you repeat.

| Step | What it does | Specified in |
|---|---|---|
| `terraform apply` | Creates every bucket, table, role, job, state machine and the device | below |
| ingest build | Downloads BDD100K, verifies it, and writes 80,000 rows to the data bucket | [01-data-and-labels.md](01-data-and-labels.md) |
| partition build | Draws bootstrap, pool, eval and reserve once, and freezes the labels the first and third are owed | [01-data-and-labels.md](01-data-and-labels.md) |
| start a run | Registers a `run_id` and starts one execution, which is the whole run rather than one cycle | [control.md](control.md) |

This file is the order those go in, plus the setup that happens before any of them. Each step's
command, and what it is overridable by, stays in the document that specifies the step.

---

## Locally

Nothing here touches AWS. The test suite runs against the pure-logic layer with no credentials,
which is most of what there is to check before an apply.

```bash
uv sync --frozen --group dev --group ingest
uv run pytest
uv run ruff check . && uv run mypy src
```

`--frozen` installs strictly from `uv.lock`, so the environment matches what CI resolved. The
`ingest` group carries `pyarrow` and `pillow`, which the ingest and fleet modules import.

---

## Bootstrap, outside Terraform

Three things exist before Terraform does, because they cannot be created by the thing they guard.

- **The state bucket.** Versioned, encrypted, public access blocked, TLS-only bucket policy. It
  cannot hold its own state.
- **A budget alarm.** First, before any resource exists. Every later cost decision assumes it.
- **A region-lock SCP**, if the account sits under an organization. Optional, and cheaper than
  discovering a resource in the wrong region.

Then fill in the two config files from their examples:

```bash
cd infra
cp backend.hcl.example backend.hcl            # the state bucket you just made
cp terraform.tfvars.example terraform.tfvars  # account ID, GitHub owner and repo, their numeric IDs
terraform init -backend-config=backend.hcl
terraform apply
```

Both files are gitignored. The numeric GitHub IDs are the ones stamped into the OIDC subject claim,
and `terraform.tfvars.example` gives the `gh api` call that reads them.

The first apply runs locally under credentials that can create IAM, because it is the apply that
creates the OIDC provider CI will later assume. After it, one manual step Terraform cannot perform:
authorize the CodeConnections GitHub App in the console, once. `terraform output
ingest_connection_status` reads `PENDING` until you do and `AVAILABLE` after.

---

## GitHub OIDC

Every later apply runs in CI, with no long-lived keys anywhere. The first apply created the provider
and two roles; CI needs to be told where they are. Set four **repository variables**:

| Variable | Value |
|---|---|
| `AWS_ACCOUNT_ID` | The account |
| `TF_STATE_BUCKET` | The state bucket |
| `AWS_PLAN_ROLE_ARN` | `terraform output plan_role_arn` |
| `AWS_APPLY_ROLE_ARN` | `terraform output apply_role_arn` |

From then on a pull request plans and a merge to `main` applies, under
[.github/workflows/terraform.yml](../.github/workflows/terraform.yml). The plan role can read and
the apply role can write; neither is a user and neither has a key.

---

## The data, once per account

Two CodeBuild jobs, ingest then partition. Both are idempotent, both take tens of minutes, and
neither has a webhook — a run of ingest re-downloads 5.7 GB and rewrites `raw/`, so they are started
by hand.

Ingest downloads both BDD100K archives, verifies them against published digests, drops the withheld
test split at extraction, and writes the images, labels and an 80,000-row manifest to the data
bucket. It runs in a container that is destroyed afterwards, so the only copy of the withheld labels
is the one the bucket policy describes. Partition then draws `bootstrap`, `pool`,
`eval` and `reserve` under one `partition_version` and freezes the labels `bootstrap` and `eval` are
owed.

Both invocations are in [01-data-and-labels.md](01-data-and-labels.md), with what each build is
overridable by.

---

## A run

A run is one Step Functions execution. It loops over cycles itself and leaves when the cycle cap or
the pool is spent, so there is nothing to tick and nothing to call again. `run start` registers the
run and starts it in one step, and prints the `run_id` every later step takes.

The command and the execution input it builds are in [control.md](control.md). The CodeBuild project
that registers a run without starting it is in [01-data-and-labels.md](01-data-and-labels.md).

Check the processing quota before starting one. A run that begins without it trains a model and then
fails at `Score`, having paid for the training job.

---

## The device

Terraform creates the instance, its IoT thing, certificate and policy, and the thing group
deployments target. First boot provisions Greengrass against that identity with `--provision false`,
so nothing exists outside Terraform state.

```bash
aws ssm start-session --target "$(terraform -chdir=infra output -raw device_instance_id)"
```

The setup log is at `/var/log/edge-ml-flywheel-setup.log`, and the nucleus logs under
`/greengrass/v2/logs/`. Stop the instance between runs — it is most of the standing cost:

```bash
aws ec2 stop-instances --instance-ids "$(terraform -chdir=infra output -raw device_instance_id)"
```

Deployments are made by the state machine, not by hand. The hand-operated commands — reading what the
fleet is running, and rolling a version back — are in
[05-fleet-and-deployment.md](05-fleet-and-deployment.md).

---

## Reading what happened

A run writes one summary when it ends, and every figure in it was measured by the job that recorded
it. [reporting.md](reporting.md) gives the key and the fields.

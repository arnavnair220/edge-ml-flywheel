# Storage

Every job reads and writes objects through one interface, `edge_ml_flywheel.storage.ObjectStore`,
implemented once against the S3-compatible API. The same code runs against AWS S3, MinIO and GCS;
the provider is selected by configuration.

---

## Interface

An `ObjectStore` is bound to one bucket and exposes six operations. The set is the operations the
jobs perform; nothing else is part of the contract.

| Operation | Behavior |
|---|---|
| `get(key) -> bytes` | Returns the whole object. Raises `NoSuchKeyError` if the key is absent. |
| `put(key, body)` | Writes `body` to `key`, replacing any existing object. |
| `download(key, path)` | Writes the object to a local file, creating parent directories. Raises `NoSuchKeyError` if the key is absent. |
| `upload(path, key)` | Writes a local file to `key`. Multipart for large files. |
| `keys(prefix)` | Yields every key under `prefix`, across all listing pages. |
| `exists(key) -> bool` | True only if this exact key exists. A key that merely prefixes another is not reported. |

`bucket` names the bound bucket, for log lines and error messages.

`NoSuchKeyError` is raised only for an absent key. A denied request propagates as the provider's
error, so an access failure is never reported as a missing object.

`exists` is implemented as a listing rather than a `HEAD`. A `HEAD` on an absent key returns 403
rather than 404 to an identity without list permission, which would make the answer depend on the
grant rather than on the bucket.

## Implementation

`edge_ml_flywheel.storage.s3.S3Store` is the only implementation. `S3Store.connect(bucket)` builds
a client from `StorageConfig.from_env()`. A caller that shares one store across a thread pool
passes `pool=` with its thread count, so threads do not queue for connections.

All clients retry in the SDK's `standard` mode, up to five attempts. When an endpoint is
configured, the client also:

- uses path-style addressing, which MinIO requires; and
- sends checksums only when an operation requires one, because GCS interoperability rejects the
  SDK's default checksum headers.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `STORAGE_ENDPOINT_URL` | No | The S3-compatible endpoint. Unset for AWS S3. |
| `STORAGE_REGION` | No | The region passed to the client. Unset to use the SDK's default chain. |
| `STORAGE_DATA_BUCKET` | Per job | The data bucket: raw images and labels, manifest, partitions, purchases. |
| `STORAGE_ARTIFACTS_BUCKET` | Per job | Models, rankings, detections, reports, staged code. |
| `STORAGE_TELEMETRY_BUCKET` | Per job | Device telemetry. |

A job requires only the bucket variables for the buckets it opens, and exits naming any that are
unset. Bucket names are never derived from a cloud account; the infrastructure that creates the
buckets supplies their names.

Credentials are not configured here. The SDK's default chain resolves them:

| Environment | Credentials | `STORAGE_ENDPOINT_URL` |
|---|---|---|
| AWS (EKS) | Workload identity (IRSA) | unset |
| GCP (GKE) | HMAC key for the job's service account | `https://storage.googleapis.com` |
| Local (kind) | MinIO access key, as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` | the MinIO service URL |

The local object store is MinIO, run from `cgr.dev/chainguard/minio` pinned by digest. MinIO is
used because it enforces per-prefix bucket policies, which the local check of the `eval` label
boundary requires.
| Device | Greengrass token exchange role | unset |

## Job inputs and outputs

Every job reads its inputs from the store and writes its outputs to it. No job reads from a
fixed local path. At startup a job fetches its inputs into a scratch directory, one subdirectory
per input, with each object placed relative to its input's prefix. Each output is uploaded to the
key `conventions` names for it as soon as it is written. Keys are derived from the job's
arguments, never passed as paths.

`storage.tree.fetch` copies many small objects with a thread pool and attempts every key before
reporting failures, so a missing object and a denied prefix are distinguishable by count.

| Job | Arguments that name its keys | Reads | Writes |
|---|---|---|---|
| Training | `--version`, `--partition_version` | Images named by `training_manifest_key`; `cohort=bootstrap` labels; the run's purchases; base weights | `model.pt`, `model.onnx` and `sha256sums` at `model_artifact_key` |
| Scoring | `--version`, `--seed`, `--precision` | The model at `model_artifact_key` for the precision; images named by `scoring_manifest_key` | Detections at `detections_key` |
| Evaluation | `--version`, `--seeds`, `--partition_version`, `--champion`, `--artifact_bytes` | The `eval` manifest; `cohort=eval` labels; each seed's detections; int8 detections when judged; the champion's `eval_prefix` | Match caches and metrics under `eval_prefix`; the gate report at `gate_report_key` |
| Partition | `--partition-version` | Label documents for the labeled cohorts | Nothing; the build uploads its staged tree |

The input modules (`training.inputs`, `scoring.inputs`, `evaluation.inputs`) hold the key
derivation and import no GPU stack, so each job's reach into the store is tested directly.

A manifest's entries are checked before anything is fetched: the prefix must be the split the
job reads (`train` for training, the cohort's split for scoring), and every entry must be a file
name. A manifest that names another split, or an entry such as `../`, is refused.

## Access boundaries

The store applies no access rule of its own. Two mechanisms enforce what a job may read:

- **Identity.** Each job runs under its own credentials, and the bucket or the provider's policy
  decides what those credentials may read. Only the evaluation job's identity may read
  `labels/cohort=eval/`.
- **Key construction.** Label keys are built only by `oracle.labels.label_key` and
  `partition.cohort_labels.label_key`, neither of which can produce a key under the split `eval` is
  drawn from. A refused image produces no request. Training and scoring fetch no `eval` label
  prefix under any arguments; evaluation fetches only `cohort=eval`.

## Verification

`tests/test_job_inputs.py` checks each job's fetched inputs against a bucket that also holds the
labels the job is denied. `tests/test_evaluation_job.py` runs the evaluation job end to end
against the store.

`tests/test_storage.py` runs one contract suite against every backend:

| Backend | When it runs |
|---|---|
| moto (in-process S3) | Always. |
| MinIO (container) | When a Docker daemon is reachable; skipped otherwise. |
| AWS S3 | When `STORAGE_TEST_AWS_BUCKET` names a bucket, with credentials and region from the environment. Each test writes under its own prefix and removes what it wrote. |

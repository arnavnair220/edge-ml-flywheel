# Storage

Every job reads and writes objects through one interface, `edge_ml_flywheel.storage.ObjectStore`,
implemented once over [fsspec](https://filesystem-spec.readthedocs.io/). The same code runs against
AWS S3 and MinIO through `s3fs`, and against GCS through `gcsfs`; the provider is selected by
configuration.

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

`NoSuchKeyError` is raised only for an absent key, which fsspec reports as `FileNotFoundError`. A
denied request propagates as the filesystem's error, `PermissionError` for `s3fs`, so an access
failure is never reported as a missing object.

`keys` sends `prefix` to the provider as the listing's own prefix, so it is a key prefix and need
not end at a `/`.

`exists` is implemented as a listing rather than a `HEAD`. A `HEAD` on an absent key returns 403
rather than 404 to an identity without list permission, which would make the answer depend on the
grant rather than on the bucket.

## Implementation

`edge_ml_flywheel.storage.filesystem.FsspecStore` is the only implementation. It holds a bucket
name and an fsspec filesystem, and addresses each object as `<bucket>/<key>`.
`FsspecStore.connect(bucket)` builds the filesystem from `StorageConfig.from_env()`. A caller that
shares one store across a thread pool passes `pool=` with its thread count, so threads do not
queue for connections.

For the `s3` protocol, the filesystem:

- retries in botocore's `standard` mode, up to five attempts;
- keeps no listings cache, so a key written after a listing is seen by the next `keys` or
  `exists`; and
- uses path-style addressing when an endpoint is configured, which MinIO requires.

`FsspecStore.from_session(aws, bucket)` builds an `s3` store from a boto3 session's credentials
and region, for old-stack callers that hold a session rather than storage configuration.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `STORAGE_PROTOCOL` | No | The fsspec protocol: `s3` for AWS S3 and MinIO, `gs` for GCS, `memory` for tests. Defaults to `s3`. |
| `STORAGE_ENDPOINT_URL` | No | The S3 endpoint, for `s3` only. Unset for AWS S3. |
| `STORAGE_REGION` | No | The region, for `s3` only. Unset to use the SDK's default chain. |
| `STORAGE_DATA_BUCKET` | Per job | The data bucket: raw images and labels, manifest, partitions, purchases. |
| `STORAGE_ARTIFACTS_BUCKET` | Per job | Models, rankings, detections, reports, staged code. |
| `STORAGE_TELEMETRY_BUCKET` | Per job | Device telemetry. |

A job requires only the bucket variables for the buckets it opens, and exits naming any that are
unset. Bucket names are never derived from a cloud account; the infrastructure that creates the
buckets supplies their names.

Credentials are not configured here. Each filesystem's default chain resolves them:

| Environment | `STORAGE_PROTOCOL` | Credentials | `STORAGE_ENDPOINT_URL` |
|---|---|---|---|
| AWS (EKS) | `s3` | Workload identity (IRSA) | unset |
| GCP (GKE) | `gs` | Workload Identity, through `gcsfs` | unset |
| Local (kind) | `s3` | MinIO access key, as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` | `http://minio.minio:9000` |
| Device | `s3` | Greengrass token exchange role | unset |

GCS is reached through its own API rather than its S3 interoperability layer, so a GKE job needs
no HMAC key: its Kubernetes service account is bound to a Google service account, and `gcsfs`
picks up that identity. `gcsfs` is added to the `storage` dependency group when a job first runs
on GCS.

The local object store is MinIO, run from `cgr.dev/chainguard/minio` pinned by digest. MinIO is
used because it enforces per-prefix bucket policies, which the local check of the `eval` label
boundary requires.

## Job inputs and outputs

Every job reads its inputs from the store and writes its outputs to it. No job reads from a
fixed local path. At startup a job fetches its inputs into a scratch directory, one subdirectory
per input, with each object placed relative to its input's prefix. Each output is uploaded to the
key `conventions` names for it as soon as it is written. Keys are derived from the job's
arguments, never passed as paths.

`storage.tree.fetch` copies many small objects with a thread pool and attempts every key before
reporting failures, so a missing object and a denied prefix are distinguishable by count.
`storage.tree.upload` is the same in the other direction: every file under a local directory is
written to the given prefix plus its relative path, so it writes nothing outside that prefix.

| Job | Arguments that name its keys | Reads | Writes |
|---|---|---|---|
| Training | `--version`, `--partition_version` | Images named by `training_manifest_key`; `cohort=bootstrap` labels; the run's purchases; base weights | `model.pt`, `model.onnx` and `sha256sums` at `model_artifact_key` |
| Scoring | `--version`, `--seed`, `--precision` | The model at `model_artifact_key` for the precision; images named by `scoring_manifest_key` | Detections at `detections_key` |
| Evaluation | `--version`, `--seeds`, `--partition_version`, `--champion`, `--artifact_bytes` | The `eval` manifest; `cohort=eval` labels; each seed's detections; int8 detections when judged; the champion's `eval_prefix` | Match caches and metrics under `eval_prefix`; the gate report at `gate_report_key` |
| Ingest | `--stage-dir` | Nothing; it lists `raw/` and `derived/` to verify the upload | Every staged file, at its path relative to the stage directory |
| Partition | `--partition-version` | The manifest under `MANIFEST_PREFIX`; the version's `_partition.json`, absent on a first run; label documents for the labeled cohorts | The staged partition prefix only, then reads `_partition.json` back to verify it |

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
against the store. Both use fsspec's in-memory filesystem, through the `memory_storage` fixture
in `tests/conftest.py`, which empties it before and after each test.

`tests/test_storage.py` runs one contract suite against every backend:

| Backend | When it runs |
|---|---|
| fsspec memory | Always. |
| moto (threaded server on loopback) | Always. `s3fs` sends requests through aiohttp, which moto's in-process mock does not intercept. |
| MinIO (container) | When a Docker daemon is reachable; skipped otherwise. |
| AWS S3 | When `STORAGE_TEST_AWS_BUCKET` names a bucket, with credentials and region from the environment. Each test writes under its own prefix and removes what it wrote. |

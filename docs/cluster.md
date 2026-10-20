# Clusters

Two Kubernetes clusters meet one contract: a local cluster on kind, and EKS on AWS. Everything
above the cluster schedules the same way on either. The kind cluster checks wiring only; its GPU
node has no GPU.

---

## Contract

| Requirement | kind | EKS |
|---|---|---|
| Services pool, labeled `pool=services` | One worker node | One managed node group |
| GPU pool, labeled `pool=gpu`, tainted `nvidia.com/gpu=present:NoSchedule` | One worker node, no GPU | A Karpenter NodePool, empty when idle |
| Images | Local registry at `localhost:5001` | Public GHCR, pulled directly |
| Bring-up and teardown | `make kind-up`, `make kind-down` | `make eks-up`, `make eks-down` |
| Check | `make kind-check` | `make eks-check` |

A GPU job selects `pool=gpu` and tolerates the `nvidia.com/gpu` taint. Nothing else schedules on
the GPU pool.

---

## Local cluster (kind)

### Nodes

| Node | Label | Taint |
|---|---|---|
| control plane | none | kind's default control-plane taint |
| services worker | `pool=services` | none |
| GPU worker | `pool=gpu` | `nvidia.com/gpu=present:NoSchedule` |

The cluster shape is `clusters/kind/kind-config.yaml`.

### Registry

A `registry:3` container named `kind-registry`, published at `localhost:5001` on the host and
attached to the `kind` Docker network. `docker-bake.hcl` tags images for it by default, so
`docker buildx bake --push` publishes to it. Each node's containerd resolves `localhost:5001` to
the container, so a manifest names an image the same way on the host and in the cluster.

The `local-registry-hosting` ConfigMap in `kube-public` publishes the registry address, following
[kind's local registry setup](https://kind.sigs.k8s.io/docs/user/local-registry/).

### In-cluster state

`clusters/kind/apps/` holds everything the cluster runs, as one Kustomize root. `kind-up` applies
it from the working tree with `kubectl apply -k`, so a saved change reaches the cluster on the
next `kind-up`, with no commit or push. Each component is a subdirectory with its own
`kustomization.yaml`, listed under `resources` in `clusters/kind/apps/kustomization.yaml`.

| Component | Directory |
|---|---|
| MinIO | `clusters/kind/apps/minio/` |

Secrets are not kept in the repository. `kind-up` creates them directly, before the apply.

### MinIO

MinIO is the cluster's S3 endpoint, at `http://minio.minio:9000`. It runs from
`cgr.dev/chainguard/minio` pinned by digest; see [storage](storage.md).

| Resource | Name | Spec |
|---|---|---|
| Namespace | `minio` | |
| Deployment | `minio` | One replica on `pool=services`, `Recreate` strategy, readiness probe on `/minio/health/ready` |
| PersistentVolumeClaim | `minio` | 10Gi, storage class `standard`, `ReadWriteOnce` |
| Service | `minio` | Port 9000 |
| Secret | `minio-root` | `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`; created by `kind-up`, not in the repository |
| Bucket | `edge-ml-flywheel` | Created by `kind-up` |

`kind-up` applies `minio/namespace.yaml` first, then creates `minio-root` if it is absent, with
user `admin` and a random 48-character hex password. An existing Secret is kept, so the password
is stable for the life of the cluster. To read it:

```sh
kubectl --context kind-edge-ml-flywheel -n minio get secret minio-root \
  -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 -d
```

After the rollout, `kind-up` runs the Job `minio-bucket` from `hack/kind-bucket-job.yaml`, which
runs `mc mb --ignore-existing` with `cgr.dev/chainguard/minio-client` pinned by digest. The Job is
deleted when it succeeds and left in place when it fails.

Objects live on the services worker's disk. They survive a restart of the pod, not `kind-down`.

### Versions

| Component | Version | Pinned in |
|---|---|---|
| kind | v0.33.0 | `Makefile` |
| Node image | `kindest/node:v1.37.0`, pinned by digest | `Makefile` |
| Registry | `registry:3`, pinned by digest | `Makefile` |
| MinIO server | `cgr.dev/chainguard/minio@sha256:709099319add48dfb44aa596a7070d3ddd8dd04d7ae881765719aec29df9e510` | `clusters/kind/apps/minio/deployment.yaml`, the same digest as `tests/test_storage.py` |
| MinIO client | `cgr.dev/chainguard/minio-client@sha256:7d0cf8f3de478cd4d2a8df4efcff9563091e298c8916122bbbbedb2ce08edcad` | `hack/kind-bucket-job.yaml`, `hack/kind-check-minio-job.yaml` |

Chainguard's free tier tags only `latest`, so both MinIO images are pinned by digest alone.

kind is downloaded to `bin/` on first use and checked against the release's published sha256, so
no installed kind is used. The node image is built for its kind release; the two move together.

### Commands

| Command | Effect |
|---|---|
| `make kind-up` | Starts the registry, creates the cluster, context `kind-edge-ml-flywheel`, creates the MinIO Secret, applies `clusters/kind/apps/`, waits for MinIO to roll out and creates the bucket |
| `make kind-check` | Builds and pushes the `run` image and runs it as a Job on the GPU worker with `--help`, then runs `kind-check-minio`, which writes one object under `kind-check/` in the bucket and reads it back |
| `make kind-down` | Deletes the cluster and the registry |

Each command skips or reapplies what is already done, so it is safe to repeat. `kind-check` fails
unless each Job completes within five minutes and the object read back matches the one written.
On failure it leaves the failing Job in place for inspection. Images in the registry and objects
in MinIO do not survive `kind-down`.

### Requirements

Docker, `kubectl`, GNU `make`, `bash`, `curl` and `openssl`. On Windows, run `make` from Git Bash.

---

## EKS cluster

The Terraform root is `clusters/eks/`. It is separate from `infra/` and keeps its state under its own key,
`eks/terraform.tfstate`, in the same state bucket. Bringing the cluster up or down never touches
the data bucket or any other `infra/` resource.

### Nodes

| Pool | Provided by | Label | Taint | Size |
|---|---|---|---|---|
| services | EKS managed node group | `pool=services` | none | Fixed, `services_node_count` nodes of `services_instance_types` |
| gpu | Karpenter NodePool `gpu` | `pool=gpu` | `nvidia.com/gpu=present:NoSchedule` | 0 to `gpu_max_nodes` nodes of `gpu_instance_types` |

Karpenter launches a gpu node when a pod that tolerates the taint is pending, and removes it one
minute after its last pod ends. The gpu pool is empty when no GPU job runs. Its nodes run
on-demand, on the Amazon Linux 2023 NVIDIA AMI, with a `gpu_root_volume_gib` root volume. The NVIDIA
device plugin runs on `pool=gpu` nodes only.

The Karpenter controller runs on the services pool. The NodePool and its EC2NodeClass are the
`clusters/eks/charts/gpu-pool` chart.

### Network

| Property | Value |
|---|---|
| Subnets | Public only, in three availability zones |
| Outbound | Each node's public IPv4 address; no NAT gateway |
| Inbound | Only from the control plane and other nodes, through the node security group |
| Images | Pulled directly from public GHCR; no registry cache, no pull secret |
| API endpoint | Public, IAM-authenticated, limited to `endpoint_public_access_cidrs` |

Karpenter selects the subnets and the node security group by the tag
`karpenter.sh/discovery=<cluster>`, so gpu nodes get the same network rules as services nodes.

### Identity

The cluster has an IAM OIDC provider for IAM roles for service accounts (IRSA). The per-job roles
that trust it are not part of this stack. The Karpenter controller authenticates through EKS Pod
Identity. The identity that runs `terraform apply` is a cluster administrator.

### Outputs

| Output | Value |
|---|---|
| `cluster_name` | The cluster name |
| `kubeconfig_command` | The `aws eks update-kubeconfig` command that adds context `eks-<cluster>` |
| `data_bucket` | The existing data bucket, read and not managed |
| `oidc_provider_arn` | The IRSA OIDC provider |
| `oidc_issuer_url` | The cluster's OIDC issuer URL |

### Variables

Set in `clusters/eks/terraform.tfvars`. Only `account_id` is required.

| Variable | Default |
|---|---|
| `account_id` | none |
| `kubernetes_version` | `1.36` |
| `services_instance_types` | `["t3.medium"]` |
| `services_node_count` | `2` |
| `gpu_instance_types` | `["g4dn.xlarge"]` |
| `gpu_max_nodes` | `1` |
| `gpu_root_volume_gib` | `100` |
| `endpoint_public_access_cidrs` | `["0.0.0.0/0"]` |

### Versions

| Component | Version | Pinned in |
|---|---|---|
| Kubernetes | 1.36 | `clusters/eks/variables.tf` |
| `terraform-aws-modules/eks` and its `karpenter` submodule | 21.29.0 | `clusters/eks/main.tf` |
| `terraform-aws-modules/vpc` | 6.7.3 | `clusters/eks/main.tf` |
| Karpenter chart | 1.14.1 | `clusters/eks/variables.tf` |
| NVIDIA device plugin chart | 0.20.1 | `clusters/eks/variables.tf` |
| Providers | exact versions | `clusters/eks/.terraform.lock.hcl` |

A Kubernetes minor is supported by Karpenter only from a given release onward, per
[Karpenter's compatibility matrix](https://karpenter.sh/docs/upgrading/compatibility/). The two
are raised together.

### Commands

Each runs with AWS profile `edgeml` in `us-east-1`.

| Command | Effect |
|---|---|
| `make eks-up` | Applies `clusters/eks/` and adds context `eks-edge-ml-flywheel` to the kubeconfig |
| `make eks-check` | Runs the `run` image from GHCR with `--help` as a Job on each pool, checks the gpu node's label and taint, and waits for the gpu pool to return to zero nodes |
| `make eks-down` | Deletes the gpu pool, waits for every Karpenter instance to terminate, then destroys `clusters/eks/`, the network last |

`eks-up` and `eks-down` are safe to repeat. `eks-check` fails unless both Jobs complete and the gpu
pool scales to zero within ten minutes, and on failure leaves the Jobs in place for inspection.
The image it runs is `EKS_CHECK_IMAGE` in the `Makefile`.

`EKS_GPU_INSTANCE_TYPES` overrides `gpu_instance_types` for `eks-up`, as a Terraform list. Until the
account has EC2 GPU quota, the check runs with a CPU type in the gpu pool:

```sh
make eks-up EKS_GPU_INSTANCE_TYPES='["t3.medium"]'
make eks-check
make eks-down
```

### Teardown

Karpenter's instances are not in Terraform state. `eks-down` therefore deletes the NodePool and
EC2NodeClass first, which makes Karpenter terminate its instances and remove the instance profile it
created, and waits until no instance tagged `karpenter.sh/nodepool` remains. It then destroys the
cluster and its nodes, deletes any unattached network interface the VPC CNI tagged with the cluster's
name, and destroys the network last. A terminated node can leave such an interface behind, and it
holds the subnet and node security group until deleted. `eks-down` fails if any instance in the
cluster or the cluster itself remains.

After `eks-down`, nothing billable remains except storage. The cluster keeps no customer-managed
KMS key and no control-plane log group, so neither outlives it.

### Billing

| Resource | Billed while |
|---|---|
| EKS control plane | The cluster exists |
| Services nodes and their volumes | The cluster exists |
| gpu nodes and their volumes | A GPU job runs, plus one minute |
| Public IPv4 addresses | Their node exists |

### Requirements

Terraform 1.10 or later, the AWS CLI, `kubectl`, GNU `make` and `bash`. Before the first `eks-up`,
copy `clusters/eks/backend.hcl.example` to `clusters/eks/backend.hcl` and
`clusters/eks/terraform.tfvars.example` to `clusters/eks/terraform.tfvars`, and fill both in. On Windows, run `make` from Git Bash.

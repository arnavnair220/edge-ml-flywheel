# Local cluster

A local Kubernetes cluster on kind, with a local image registry. It provides the same node pools
as the cloud clusters, so jobs schedule locally as they do in the cloud. It checks wiring only:
its GPU node has no GPU.

---

## Nodes

| Node | Label | Taint |
|---|---|---|
| control plane | none | kind's default control-plane taint |
| services worker | `pool=services` | none |
| GPU worker | `pool=gpu` | `nvidia.com/gpu=present:NoSchedule` |

A GPU job selects `pool=gpu` and tolerates the `nvidia.com/gpu` taint. Nothing else schedules on
the GPU worker. The cluster shape is `kind/kind-config.yaml`.

## Registry

A `registry:3` container named `kind-registry`, published at `localhost:5001` on the host and
attached to the `kind` Docker network. `docker-bake.hcl` tags images for it by default, so
`docker buildx bake --push` publishes to it. Each node's containerd resolves `localhost:5001` to
the container, so a manifest names an image the same way on the host and in the cluster.

The `local-registry-hosting` ConfigMap in `kube-public` publishes the registry address, following
[kind's local registry setup](https://kind.sigs.k8s.io/docs/user/local-registry/).

## Versions

Pinned in the `Makefile`.

| Component | Version |
|---|---|
| kind | v0.33.0 |
| Node image | `kindest/node:v1.37.0`, pinned by digest |
| Registry | `registry:3`, pinned by digest |

kind is downloaded to `bin/` on first use and checked against the release's published sha256, so
no installed kind is used. The node image is built for its kind release; the two move together.

## Commands

| Command | Effect |
|---|---|
| `make cluster-up` | Starts the registry and creates the cluster, context `kind-edge-ml-flywheel` |
| `make cluster-check` | Builds and pushes the `run` image, runs it as a Job on the GPU worker with `--help` |
| `make cluster-down` | Deletes the cluster and the registry |

Each command skips what is already done, so it is safe to repeat. `cluster-check` fails unless the
Job completes, and on failure leaves the Job in place for inspection. Images in the registry do not
survive `cluster-down`.

## Requirements

Docker, `kubectl`, GNU `make`, `bash` and `curl`. On Windows, run `make` from Git Bash.

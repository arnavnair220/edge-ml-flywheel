# Images

Every job runs from its own container image. Images are built from this repository, published to
GitHub Container Registry (GHCR), and pulled by every cluster from there. A cloud's own registry,
where one is used, is a pull-through cache in front of GHCR and never the source of an image.

---

## Image set

One image per job entry point. Each Dockerfile is `images/<name>.Dockerfile`, built with the
repository root as context. `docker-bake.hcl` lists the images and is the only list: CI and local
builds both read it.

| Image | Entry point | Accelerator | Dependency groups |
|---|---|---|---|
| `train` | `container/train.py` | GPU | `training`, plus torch and `container/requirements.txt` |
| `score` | `container/score.py` | GPU | `training`, plus torch and `container/requirements.txt` |
| `evaluate` | `container/evaluate.py` | CPU | `training` |
| `ingest` | `python -m edge_ml_flywheel.ingest` | CPU | `ingest`, `run` |
| `partition` | `python -m edge_ml_flywheel.partition` | CPU | `ingest`, `run` |
| `run` | `python -m edge_ml_flywheel.run` | CPU | `run` |
| `fleet` | `python -m edge_ml_flywheel.fleet` | CPU | `training` |

The entry point is the image's `ENTRYPOINT`, so a job's arguments are the container's arguments.
Every image also carries the project dependencies. `boto3`, which the object store uses, comes
from the `run` or `training` group.

`evaluate` is CPU only. It computes metrics from cached match arrays and loads no model.

## Build

Every image is built in two stages.

1. **Build stage.** `uv sync --frozen --no-dev --no-editable` installs the project and the
   image's groups from `uv.lock` into `/opt/venv`. Locked dependencies are installed before the
   source is copied, so a source-only change rebuilds one layer.
2. **Runtime stage.** `/opt/venv` and the job's entry point are copied onto a clean base. No
   build tools, uv, or source tree are present.

`train` and `score` split this differently, because their dependencies include several gigabytes
of CUDA libraries. A `deps` stage installs only the dependencies, and a separate `wheel` stage
builds the project with `uv build --wheel`. The runtime stage copies `/opt/venv` from `deps`, then
installs the wheel with `--no-deps` as its own layer. A source-only change rebuilds that small
layer and reuses the dependency layer.

| Property | Value |
|---|---|
| Base | `python:3.12-slim-bookworm`, pinned by digest |
| uv | `ghcr.io/astral-sh/uv:0.12.10`, pinned by digest, build stage only |
| User | `app`, uid 10001, non-root |
| Working directory | `/home/app` |
| Job scratch | `/tmp`, as each job's `WORK` constant names |

Base images are pinned by digest, so a rebuild of the same commit uses the same bases. Moving a
base is an edit to the digest in each Dockerfile.

### GPU images

torch is not in `uv.lock`. `train` and `score` install it after the locked sync:

| Package | Version | Index |
|---|---|---|
| `torch` | 2.9.0 | `https://download.pytorch.org/whl/cu130` |
| `torchvision` | 0.24.0 | `https://download.pytorch.org/whl/cu130` |

`container/requirements.txt` is then installed on top. The torch wheels carry their own CUDA
13.0 runtime, so the base needs no CUDA install; the node supplies only the GPU driver. CUDA
13.0 supports Turing (T4) and newer.

The final sync runs with `--inexact`, so it restores locked versions without removing torch or
the requirements packages. The runtime stage adds `libgl1` and `libglib2.0-0`, which OpenCV
loads, and creates `/home/app/.config/Ultralytics` for the settings file ultralytics writes.

A GPU image is about 5.5 GB unpacked, most of it the CUDA libraries the torch wheels carry.

`train` and `score` share this environment, so a model is scored by the stack that trained it.

## Smoke check

An image passes when its container starts and its entry point prints usage:

```bash
docker run --rm <image> --help
```

This proves the image imports its job and every module the job imports, including torch and
ultralytics in the GPU images. It requires no GPU, credentials, or storage.

## Tags

An image reference is `${REGISTRY}/<name>:${TAG}`, from two variables in `docker-bake.hcl`.

| Variable | Default | CI |
|---|---|---|
| `REGISTRY` | `localhost:5001/edge-ml-flywheel` | `ghcr.io/arnavnair220/edge-ml-flywheel` |
| `TAG` | `dev` | the full commit SHA |

There is no `latest` tag; a manifest names a published image by SHA, so a tag always resolves to
the same build.

## CI

`.github/workflows/images.yml` reads the image list from `docker-bake.hcl` and builds each image
on its own runner, in parallel. The GPU images do not fit on one runner's disk together.

| Event | Build | Smoke check | Push to GHCR |
|---|---|---|---|
| Pull request | yes | yes | no |
| Push to `refactor/kubernetes-platform` | yes | yes | yes |

CI authenticates to GHCR with the workflow's `GITHUB_TOKEN`; no cloud credential is involved.
Layers are cached in the GitHub Actions cache, one scope per image.

GHCR packages are public, matching the repository, so clusters pull without a pull secret. GHCR
creates a package as private on its first push; each package is set to public once, in the
package's settings on GitHub.

## Local builds

The default `REGISTRY` is the local registry kind pulls from, at `localhost:5001`.

```bash
docker buildx bake                                  # build every image
docker buildx bake ingest run                       # build named images only
TAG=$(git rev-parse HEAD) docker buildx bake --push # build and push every image, tagged by commit
docker buildx bake --load run                       # build into the local Docker image store
docker run --rm localhost:5001/edge-ml-flywheel/run:dev --help
```

A cluster pulls merged images from GHCR. The local registry holds builds of code that is not yet
merged.
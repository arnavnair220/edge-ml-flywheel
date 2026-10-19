# Developer commands. Each target calls a script in hack/. See docs/cluster.md.

# kind and its node image are released together. Bump both from the same kind
# release notes, and keep the node image pinned by digest, as those notes require.
KIND_VERSION    := v0.33.0
KIND_NODE_IMAGE := kindest/node:v1.37.0@sha256:a1ed56cfb0e7b93589bdf97c8cd566405a265939e3620fc4f5de89adff580ae5
REGISTRY_IMAGE  := registry:3@sha256:ddf754342cfc8acc51a56d5d0ab6af06826461864460636d8bd5c546dab2a7b8

KIND_CLUSTER  := edge-ml-flywheel
REGISTRY_NAME := kind-registry
# docker-bake.hcl tags images for this port.
REGISTRY_PORT := 5001

ifeq ($(OS),Windows_NT)
EXE := .exe
endif
KIND := bin/kind-$(KIND_VERSION)$(EXE)

EKS_CLUSTER := edge-ml-flywheel
EKS_PROFILE := edgeml
EKS_REGION  := us-east-1
# A run image GHCR already holds. Images are pushed only when their inputs
# change, so not every commit has one.
EKS_CHECK_IMAGE := ghcr.io/arnavnair220/edge-ml-flywheel/run:b317d300eaee7ce3ca75487a72650d12bcc57b07
# Overrides the gpu pool's instance types, as a Terraform list:
#   make eks-up EKS_GPU_INSTANCE_TYPES='["t3.medium"]'
EKS_GPU_INSTANCE_TYPES ?=

SHELL := bash
export KIND KIND_NODE_IMAGE REGISTRY_IMAGE KIND_CLUSTER REGISTRY_NAME REGISTRY_PORT
export EKS_CLUSTER EKS_CHECK_IMAGE

eks-up eks-check eks-down: export AWS_PROFILE := $(EKS_PROFILE)
eks-up eks-check eks-down: export AWS_REGION := $(EKS_REGION)
eks-up eks-check eks-down: export TF_VAR_cluster_name := $(EKS_CLUSTER)
ifneq ($(EKS_GPU_INSTANCE_TYPES),)
eks-up: export TF_VAR_gpu_instance_types := $(EKS_GPU_INSTANCE_TYPES)
endif

.PHONY: kind-up kind-check kind-down eks-up eks-check eks-down

$(KIND):
	bash hack/kind-install.sh $(KIND_VERSION) $@

kind-up: $(KIND)
	bash hack/kind-up.sh

kind-check: $(KIND)
	bash hack/kind-check.sh

kind-down: $(KIND)
	bash hack/kind-down.sh

eks-up:
	bash hack/eks-up.sh

eks-check:
	bash hack/eks-check.sh

eks-down:
	bash hack/eks-down.sh

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

SHELL := bash
export KIND KIND_NODE_IMAGE REGISTRY_IMAGE KIND_CLUSTER REGISTRY_NAME REGISTRY_PORT

.PHONY: cluster-up cluster-check cluster-down

$(KIND):
	bash hack/kind-install.sh $(KIND_VERSION) $@

cluster-up: $(KIND)
	bash hack/kind-up.sh

cluster-check: $(KIND)
	bash hack/kind-check.sh

cluster-down: $(KIND)
	bash hack/kind-down.sh

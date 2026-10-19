#!/usr/bin/env bash
# Delete the kind cluster and its local registry. Safe to run again.
# Run through `make kind-down`, which sets the variables below.
set -euo pipefail

: "${KIND:?}" "${KIND_CLUSTER:?}" "${REGISTRY_NAME:?}"

if "$KIND" get clusters | grep -qx "$KIND_CLUSTER"; then
  "$KIND" delete cluster --name "$KIND_CLUSTER"
fi

if docker inspect "$REGISTRY_NAME" >/dev/null 2>&1; then
  docker rm -f "$REGISTRY_NAME"
fi

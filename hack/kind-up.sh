#!/usr/bin/env bash
# Bring up the kind cluster and its local registry. Safe to run again: each step
# skips what already exists. Follows https://kind.sigs.k8s.io/docs/user/local-registry/
# Run through `make kind-up`, which sets the variables below.
set -euo pipefail
# Git Bash would rewrite the /etc/... paths passed to docker exec into Windows paths.
export MSYS_NO_PATHCONV=1

: "${KIND:?}" "${KIND_NODE_IMAGE:?}" "${REGISTRY_IMAGE:?}"
: "${KIND_CLUSTER:?}" "${REGISTRY_NAME:?}" "${REGISTRY_PORT:?}"
cd "$(dirname "$0")/.."

# 1. The registry container, published on the host at localhost:$REGISTRY_PORT.
state=$(docker inspect -f '{{.State.Running}}' "$REGISTRY_NAME" 2>/dev/null || true)
if [ -z "$state" ]; then
  docker run -d --restart=always -p "127.0.0.1:${REGISTRY_PORT}:5000" --network bridge \
    --name "$REGISTRY_NAME" "$REGISTRY_IMAGE"
elif [ "$state" = false ]; then
  docker start "$REGISTRY_NAME"
fi

# 2. The cluster.
if ! "$KIND" get clusters | grep -qx "$KIND_CLUSTER"; then
  "$KIND" create cluster --name "$KIND_CLUSTER" --image "$KIND_NODE_IMAGE" \
    --config clusters/kind/kind-config.yaml --wait 120s
fi

# 3. Point each node's containerd at the registry. Inside a node, localhost is the
#    node itself, so localhost:$REGISTRY_PORT is aliased to the registry container.
#    Node images from kind v0.27.0 on already read this directory, so the
#    cluster config needs no containerd patch.
registry_dir="/etc/containerd/certs.d/localhost:${REGISTRY_PORT}"
for node in $("$KIND" get nodes --name "$KIND_CLUSTER"); do
  docker exec "$node" mkdir -p "$registry_dir"
  printf '[host."http://%s:5000"]\n' "$REGISTRY_NAME" |
    docker exec -i "$node" cp /dev/stdin "$registry_dir/hosts.toml"
done

# 4. Put the registry on the kind network, so nodes can reach it by name.
if [ "$(docker inspect -f '{{json .NetworkSettings.Networks.kind}}' "$REGISTRY_NAME")" = null ]; then
  docker network connect kind "$REGISTRY_NAME"
fi

# 5. Publish the registry's address for in-cluster tools (KEP-1755).
kubectl --context "kind-${KIND_CLUSTER}" apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REGISTRY_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

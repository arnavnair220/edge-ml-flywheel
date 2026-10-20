#!/usr/bin/env bash
# Bring up the kind cluster and its local registry, then apply clusters/kind/apps/
# and create MinIO's bucket. Safe to run again: each step skips or reapplies what
# already exists. The registry follows
# https://kind.sigs.k8s.io/docs/user/local-registry/
# Run through `make kind-up`, which sets the variables below.
set -euo pipefail
# Git Bash would rewrite the /etc/... paths passed to docker exec into Windows paths.
export MSYS_NO_PATHCONV=1

: "${KIND:?}" "${KIND_NODE_IMAGE:?}" "${REGISTRY_IMAGE:?}"
: "${KIND_CLUSTER:?}" "${REGISTRY_NAME:?}" "${REGISTRY_PORT:?}"
cd "$(dirname "$0")/.."
kube=(kubectl --context "kind-${KIND_CLUSTER}")

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
"${kube[@]}" apply -f - <<EOF
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

# 6. MinIO's root credentials, kept out of git. The password is generated once;
#    a later run keeps the Secret it finds. The namespace comes first, so the
#    Secret has somewhere to go.
"${kube[@]}" apply -f clusters/kind/apps/minio/namespace.yaml
if ! "${kube[@]}" -n minio get secret minio-root >/dev/null 2>&1; then
  "${kube[@]}" -n minio create secret generic minio-root \
    --from-literal=MINIO_ROOT_USER=admin \
    --from-literal=MINIO_ROOT_PASSWORD="$(openssl rand -hex 24)"
fi

# 7. Everything the cluster runs, from the working tree.
"${kube[@]}" apply -k clusters/kind/apps
"${kube[@]}" -n minio rollout status deployment/minio --timeout 300s

# 8. The bucket. The Job is rerun each time and removed once it succeeds; on
#    failure it is left in place for inspection.
bucket_job=job/minio-bucket
"${kube[@]}" -n minio delete "$bucket_job" --ignore-not-found --wait
"${kube[@]}" apply -f hack/kind-bucket-job.yaml
if ! "${kube[@]}" -n minio wait "$bucket_job" --for=condition=complete --timeout 120s; then
  "${kube[@]}" -n minio logs "$bucket_job" || true
  echo "kind-up failed: $bucket_job did not complete; left in place for inspection" >&2
  exit 1
fi
"${kube[@]}" -n minio delete "$bucket_job" --wait

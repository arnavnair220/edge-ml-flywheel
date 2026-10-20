#!/usr/bin/env bash
# Check the kind cluster end to end: build the run image, push it to the local
# registry, and run it on the gpu node with --help; then check that Argo CD's root
# Application is synced and healthy and its test ConfigMap is on the cluster.
# Exits non-zero unless both hold. Run through `make kind-check`, after
# `make kind-up`.
set -euo pipefail

: "${KIND_CLUSTER:?}"
cd "$(dirname "$0")/.."
kube=(kubectl --context "kind-${KIND_CLUSTER}")
job=job/kind-check

docker buildx bake --push run

"${kube[@]}" delete "$job" --ignore-not-found --wait
"${kube[@]}" apply -f hack/kind-check-job.yaml

# Wait for either outcome. kubectl wait takes one condition, so poll the status.
result=
for _ in $(seq 1 60); do
  result=$("${kube[@]}" get "$job" -o jsonpath='{.status.succeeded}{.status.failed}')
  [ -n "$result" ] && break
  sleep 5
done

"${kube[@]}" logs "$job" || true
if [ "$("${kube[@]}" get "$job" -o jsonpath='{.status.succeeded}')" != 1 ]; then
  "${kube[@]}" describe "$job"
  echo "cluster check failed: $job did not complete; left in place for inspection" >&2
  exit 1
fi

"${kube[@]}" delete "$job" --wait

# Argo CD: the root Application reports Synced and Healthy, and the ConfigMap it
# syncs from clusters/apps/ exists. Argo CD polls the repo every three minutes.
app=application/root
status=
for _ in $(seq 1 60); do
  status=$("${kube[@]}" -n argocd get "$app" -o jsonpath='{.status.sync.status}/{.status.health.status}' || true)
  [ "$status" = Synced/Healthy ] && break
  sleep 5
done

if [ "$status" != Synced/Healthy ] ||
  ! "${kube[@]}" -n default get configmap/argocd-sync-check >/dev/null; then
  "${kube[@]}" -n argocd describe "$app"
  echo "cluster check failed: $app is $status, want Synced/Healthy" >&2
  exit 1
fi

echo "cluster check passed"

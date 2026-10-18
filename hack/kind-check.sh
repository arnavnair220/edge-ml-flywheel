#!/usr/bin/env bash
# Check the kind cluster end to end: build the run image, push it to the local
# registry, and run it on the gpu node with --help. Exits non-zero unless the Job
# completes. Run through `make cluster-check`, after `make cluster-up`.
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
echo "cluster check passed"

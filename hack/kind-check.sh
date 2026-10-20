#!/usr/bin/env bash
# Check the kind cluster end to end: build the run image, push it to the local
# registry, and run it on the gpu node with --help; then write one object to
# MinIO's bucket and read it back. Exits non-zero unless both hold. Run through
# `make kind-check`, after `make kind-up`.
set -euo pipefail

: "${KIND_CLUSTER:?}"
cd "$(dirname "$0")/.."
kube=(kubectl --context "kind-${KIND_CLUSTER}")

# Run a Job from a manifest and wait for it to succeed. On failure, describe it,
# leave it in place for inspection and exit.
run_job() {
  local namespace=$1 job=$2 manifest=$3 result=
  "${kube[@]}" -n "$namespace" delete "$job" --ignore-not-found --wait
  "${kube[@]}" apply -f "$manifest"

  # Wait for either outcome. kubectl wait takes one condition, so poll the status.
  for _ in $(seq 1 60); do
    result=$("${kube[@]}" -n "$namespace" get "$job" -o jsonpath='{.status.succeeded}{.status.failed}')
    [ -n "$result" ] && break
    sleep 5
  done

  "${kube[@]}" -n "$namespace" logs "$job" --all-containers || true
  if [ "$("${kube[@]}" -n "$namespace" get "$job" -o jsonpath='{.status.succeeded}')" != 1 ]; then
    "${kube[@]}" -n "$namespace" describe "$job"
    echo "cluster check failed: $job did not complete; left in place for inspection" >&2
    exit 1
  fi
}

docker buildx bake --push run

run_job default job/kind-check hack/kind-check-job.yaml
"${kube[@]}" -n default delete job/kind-check --wait

# MinIO: the object `read` prints is the one `write` put.
storage_job=job/kind-check-minio
run_job minio "$storage_job" hack/kind-check-minio-job.yaml
want=$("${kube[@]}" -n minio get configmap/kind-check-minio -o jsonpath='{.data.object}')
got=$("${kube[@]}" -n minio logs "$storage_job" -c read)
if [ "$got" != "$want" ]; then
  echo "cluster check failed: $storage_job read back '$got', want '$want'; left in place for inspection" >&2
  exit 1
fi
"${kube[@]}" delete -f hack/kind-check-minio-job.yaml --wait

echo "cluster check passed"

#!/usr/bin/env bash
# Check the EKS cluster end to end: run $EKS_CHECK_IMAGE with --help as a Job on
# each pool, confirm the gpu node Karpenter launched carries the pool's label and
# taint, then wait for the gpu pool to scale back to zero. Exits non-zero on any
# failure, and leaves failed Jobs in place for inspection.
# Run through `make eks-check`, after `make eks-up`.
set -euo pipefail

: "${EKS_CLUSTER:?}" "${EKS_CHECK_IMAGE:?}"
kube=(kubectl --context "eks-${EKS_CLUSTER}")
pools=(services gpu)

fail() {
  echo "cluster check failed: $*" >&2
  exit 1
}

# A Job pinned to one pool. Only the gpu Job tolerates the gpu taint.
manifest() {
  local pool=$1
  cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: eks-check-${pool}
spec:
  backoffLimit: 0
  # Long enough for Karpenter to launch a gpu node and the node to pull the image.
  activeDeadlineSeconds: 900
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        pool: ${pool}
EOF
  if [ "$pool" = gpu ]; then
    cat <<EOF
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
EOF
  fi
  cat <<EOF
      containers:
        - name: run
          image: ${EKS_CHECK_IMAGE}
          args: ["--help"]
EOF
}

# Wait for either outcome. kubectl wait takes one condition, so poll the status.
wait_job() {
  local job=job/eks-check-$1 result=
  for _ in $(seq 1 190); do
    result=$("${kube[@]}" get "$job" -o jsonpath='{.status.succeeded}{.status.failed}')
    [ -n "$result" ] && break
    sleep 5
  done
  "${kube[@]}" logs "$job" || true
  if [ "$("${kube[@]}" get "$job" -o jsonpath='{.status.succeeded}')" != 1 ]; then
    "${kube[@]}" describe "$job"
    fail "$job did not complete; left in place for inspection"
  fi
}

for pool in "${pools[@]}"; do
  "${kube[@]}" delete "job/eks-check-${pool}" --ignore-not-found --wait
done
for pool in "${pools[@]}"; do
  manifest "$pool"
  echo ---
done | "${kube[@]}" apply -f -

for pool in "${pools[@]}"; do
  wait_job "$pool"
done

# Karpenter removes an empty node a minute after its last pod ends, so inspect
# the gpu node now, before it goes.
node=$("${kube[@]}" get pods -l job-name=eks-check-gpu -o jsonpath='{.items[0].spec.nodeName}')
labels=$("${kube[@]}" get node "$node" -o jsonpath='{.metadata.labels.pool} {.metadata.labels.karpenter\.sh/nodepool}')
[ "$labels" = "gpu gpu" ] || fail "node $node is not a Karpenter gpu-pool node (pool, nodepool: $labels)"
"${kube[@]}" get node "$node" \
  -o jsonpath='{range .spec.taints[*]}{.key}={.value}:{.effect}{"\n"}{end}' |
  grep -qx 'nvidia.com/gpu=present:NoSchedule' || fail "node $node lacks the nvidia.com/gpu taint"
echo "gpu node $node: label pool=gpu, taint nvidia.com/gpu=present:NoSchedule"

for pool in "${pools[@]}"; do
  "${kube[@]}" delete "job/eks-check-${pool}" --wait
done

echo "waiting for the gpu pool to scale to zero"
for _ in $(seq 1 120); do
  [ -z "$("${kube[@]}" get nodeclaims -o name)" ] && break
  sleep 5
done
[ -z "$("${kube[@]}" get nodeclaims -o name)" ] || fail "gpu pool did not scale to zero within 10 minutes"

echo "cluster check passed"

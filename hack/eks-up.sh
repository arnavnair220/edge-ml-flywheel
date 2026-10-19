#!/usr/bin/env bash
# Bring up the EKS cluster and add it to the kubeconfig as context
# eks-$EKS_CLUSTER. Safe to run again: apply changes only what differs.
# Run through `make eks-up`, which sets the variables below. The cluster bills
# until `make eks-down`.
set -euo pipefail

: "${AWS_PROFILE:?}" "${AWS_REGION:?}" "${EKS_CLUSTER:?}"
cd "$(dirname "$0")/.."
tf=(terraform -chdir=clusters/eks)

"${tf[@]}" init -input=false -backend-config=backend.hcl
"${tf[@]}" apply -input=false -auto-approve

aws eks update-kubeconfig --region "$AWS_REGION" --name "$EKS_CLUSTER" \
  --alias "eks-${EKS_CLUSTER}" --profile "$AWS_PROFILE"
kubectl --context "eks-${EKS_CLUSTER}" get nodes -L pool

#!/usr/bin/env bash
# Tear down the EKS cluster, leaving nothing billable but storage. Safe to run
# again. Run through `make eks-down`, which sets the variables below.
#
# Karpenter's nodes are not in Terraform state. Deleting the NodePool and
# EC2NodeClass first makes Karpenter terminate its nodes and remove the instance
# profile it created. Destroying first would leave those nodes running, and the
# destroy would stall on the subnets and security group they hold.
set -euo pipefail

: "${AWS_PROFILE:?}" "${AWS_REGION:?}" "${EKS_CLUSTER:?}"
cd "$(dirname "$0")/.."
tf=(terraform -chdir=clusters/eks)
kube=(kubectl --context "eks-${EKS_CLUSTER}")

# Instances in the cluster that are not yet terminated. Karpenter's nodes and the
# node group's both carry the cluster's ownership tag.
live_instances() {
  aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:kubernetes.io/cluster/${EKS_CLUSTER},Values=owned" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    "$@" --query 'Reservations[].Instances[].InstanceId' --output text
}

# 1. Remove the gpu pool and wait for Karpenter to terminate its nodes.
if aws eks describe-cluster --region "$AWS_REGION" --name "$EKS_CLUSTER" >/dev/null 2>&1; then
  aws eks update-kubeconfig --region "$AWS_REGION" --name "$EKS_CLUSTER" \
    --alias "eks-${EKS_CLUSTER}" --profile "$AWS_PROFILE" >/dev/null
  if "${kube[@]}" get crd nodepools.karpenter.sh >/dev/null 2>&1; then
    "${kube[@]}" delete nodepools --all --wait=false
    # Each EC2NodeClass's finalizer holds it until its nodes have terminated.
    "${kube[@]}" delete ec2nodeclasses --all --timeout=15m
  fi
fi

echo "waiting for Karpenter's instances to terminate"
for _ in $(seq 1 90); do
  [ -z "$(live_instances "Name=tag-key,Values=karpenter.sh/nodepool")" ] && break
  sleep 10
done
left=$(live_instances "Name=tag-key,Values=karpenter.sh/nodepool")
if [ -n "$left" ]; then
  echo "Karpenter instances still running: $left; not destroying" >&2
  exit 1
fi

# 2. The cluster and everything that depends on it: its node group, add-ons,
# Helm releases and Karpenter's resources. Security groups and the network do
# not depend on the cluster, so they stay until step 4.
"${tf[@]}" init -input=false -backend-config=backend.hcl
"${tf[@]}" destroy -input=false -auto-approve -target=module.eks.aws_eks_cluster.this

# 3. Network interfaces the VPC CNI attached to nodes are sometimes left behind,
# unattached, when a node terminates. They hold the subnet and the node security
# group, so the network cannot be destroyed until they are deleted. Only
# unattached interfaces the CNI tagged with this cluster's name are deleted.
for eni in $(aws ec2 describe-network-interfaces --region "$AWS_REGION" \
  --filters "Name=tag:cluster.k8s.amazonaws.com/name,Values=${EKS_CLUSTER}" \
  "Name=status,Values=available" \
  --query 'NetworkInterfaces[].NetworkInterfaceId' --output text); do
  echo "deleting leftover network interface $eni"
  aws ec2 delete-network-interface --region "$AWS_REGION" --network-interface-id "$eni"
done

# 4. Everything else Terraform created.
"${tf[@]}" destroy -input=false -auto-approve

# 5. Confirm nothing billable is left.
left=$(live_instances)
if [ -n "$left" ]; then
  echo "instances still running after destroy: $left" >&2
  exit 1
fi
if aws eks describe-cluster --region "$AWS_REGION" --name "$EKS_CLUSTER" >/dev/null 2>&1; then
  echo "cluster $EKS_CLUSTER still exists after destroy" >&2
  exit 1
fi
echo "cluster $EKS_CLUSTER is down; no instances remain"

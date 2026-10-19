output "cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "kubeconfig_command" {
  description = "Adds the cluster to the local kubeconfig as context eks-<cluster>. Add --profile to pick the account."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name} --alias eks-${module.eks.cluster_name}"
}

output "data_bucket" {
  description = "The existing data bucket, owned by infra/."
  value       = data.aws_s3_bucket.data.id
}

output "oidc_provider_arn" {
  description = "IAM OIDC provider for IRSA. Per-job roles trust this provider."
  value       = module.eks.oidc_provider_arn
}

output "oidc_issuer_url" {
  description = "The cluster's OIDC issuer URL."
  value       = module.eks.cluster_oidc_issuer_url
}

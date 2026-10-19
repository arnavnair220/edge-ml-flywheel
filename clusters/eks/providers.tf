provider "aws" {
  region = var.aws_region

  # Refuse to run against any account but the project account, so a stale
  # profile cannot create the cluster in another account.
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
      Repo      = var.repo
      Stack     = "eks"
    }
  }
}

# Authenticates as the caller through the AWS CLI, so the CLI must be installed
# and AWS_PROFILE must name the same account as the aws provider.
provider "helm" {
  kubernetes = {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec = {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.aws_region]
    }
  }
}

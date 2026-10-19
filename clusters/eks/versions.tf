terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.59"
    }

    # Installs Karpenter, the gpu NodePool and the NVIDIA device plugin.
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.0"
    }
  }

  # The same state bucket as infra/, under its own key, so bringing the cluster
  # up or down never reads or writes the old stack's state. use_lockfile is
  # Terraform's native S3 locking.
  #
  # bucket is absent because a backend block cannot interpolate variables.
  # Supply it at init time:
  #   terraform init -backend-config=backend.hcl
  backend "s3" {
    key          = "eks/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

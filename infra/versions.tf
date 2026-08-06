terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # State lives in a bucket created by hand during bootstrap, because a bucket
  # cannot hold its own state. use_lockfile is Terraform's native S3 locking and
  # replaces the deprecated DynamoDB lock table.
  backend "s3" {
    bucket       = "edge-ml-flywheel-tfstate-339741260621"
    key          = "core/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

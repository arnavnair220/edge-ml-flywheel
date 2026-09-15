terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }

    # Zips the control function's deployment package and the layer beside it.
    # The package is pure Python, so there is nothing to build and a directory
    # is the whole input -- which is what keeps a Lambda in this stack from
    # needing a build step CI has to keep in step with.
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
  }

  # State lives in a bucket created by hand during bootstrap, because a bucket
  # cannot hold its own state. use_lockfile is Terraform's native S3 locking and
  # replaces the deprecated DynamoDB lock table.
  #
  # bucket is deliberately absent: a backend block cannot interpolate variables,
  # and the name is unique per deployment. Supply it at init time instead:
  #   terraform init -backend-config=backend.hcl
  # CI passes -backend-config="bucket=${{ vars.TF_STATE_BUCKET }}".
  backend "s3" {
    key          = "core/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

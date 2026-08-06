variable "project" {
  description = "Project name, used as a tag and as a prefix on resource names."
  type        = string
  default     = "edge-ml-flywheel"
}

variable "account_id" {
  description = "The dedicated project account. Not the personal or management account."
  type        = string
  default     = "339741260621"
}

variable "aws_region" {
  description = "Single region for the whole project. An SCP denies everything else."
  type        = string
  default     = "us-east-1"
}

variable "github_owner" {
  description = "GitHub account that owns the repo."
  type        = string
  default     = "arnavnair220"
}

variable "github_repo" {
  description = "Repo allowed to assume the CI roles."
  type        = string
  default     = "edge-ml-flywheel"
}

variable "state_bucket" {
  description = "Terraform state bucket, created by hand during bootstrap."
  type        = string
  default     = "edge-ml-flywheel-tfstate-339741260621"
}

variable "state_key" {
  description = "State object key. Must match the backend block in versions.tf."
  type        = string
  default     = "core/terraform.tfstate"
}

# Variables with no default are deployment identity: they must be supplied, so a
# clone fails loudly instead of silently pointing at someone else's account.
# Copy terraform.tfvars.example to terraform.tfvars and fill it in. In CI they
# arrive as TF_VAR_* environment variables.

variable "account_id" {
  description = "The dedicated project account. Not a personal or management account."
  type        = string
}

variable "state_bucket" {
  description = "Terraform state bucket, created by hand during bootstrap. Globally unique."
  type        = string
}

variable "github_owner" {
  description = "GitHub account that owns the repo."
  type        = string
}

variable "github_repo" {
  description = "Repo allowed to assume the CI roles."
  type        = string
}

# GitHub stamps immutable numeric IDs into the OIDC subject claim so that a
# renamed or recreated repo cannot inherit these roles. Read the live values with
#   gh api repos/OWNER/REPO/actions/oidc/customization/sub
# In CI these come free from the github context, no configuration needed.

variable "github_owner_id" {
  description = "Immutable numeric ID of the GitHub account."
  type        = string
}

variable "github_repo_id" {
  description = "Immutable numeric ID of the repository."
  type        = string
}

# Below here the defaults are properties of the project itself, not of whoever
# is deploying it, so they are safe to keep.

variable "project" {
  description = "Project name, used as a tag and as a prefix on resource names."
  type        = string
  default     = "edge-ml-flywheel"
}

variable "aws_region" {
  description = "Single region for the whole project. An SCP denies everything else."
  type        = string
  default     = "us-east-1"
}

variable "state_key" {
  description = "State object key. Must match the backend block in versions.tf."
  type        = string
  default     = "core/terraform.tfstate"
}

# pyarrow for the control function, which reads the label parquet to recover the
# image IDs a cycle's manifest names. It is a platform wheel, so it cannot come
# out of `archive_file` over a source directory the way the pure-Python package
# does, and building one would put a compile step in a stack that has none.
#
# AWS publishes it inside the SDK for pandas layer, and version 20 is release
# 3.14.0 on Python 3.12 -- the same interpreter `pyproject.toml` pins.
#
# The version is pinned rather than floating for the container tag's reason: the
# layer is half of what the control function is. It is also the half this account
# cannot enumerate, since `lambda:ListLayerVersions` is not granted on a layer
# owned by someone else, so a newer one is found by asking for it:
#
#   aws lambda get-layer-version --profile edgeml \
#     --layer-name arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python312 \
#     --version-number <n> --query Description
#
# A version that does not exist fails the apply with a not-found on this ARN,
# which is a one-line fix rather than a silent breakage.
variable "pyarrow_layer_arn" {
  description = "Managed layer supplying pyarrow to the control function. Bump the trailing version if the apply cannot find it."
  type        = string
  default     = "arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python312:20"
}

# The UC Berkeley host serving both BDD100K archives. A variable rather than a
# constant in the buildspec so that a move can be answered with a per-build
# override instead of a commit: `dl.yf.io` resolves to this same address and
# serves byte-identical files. Trusting DNS or an IP costs nothing extra here --
# the transport is plain HTTP either way, and the labels sha256 plus the
# image-ID set check are what actually establish that the right bytes arrived.
variable "bdd100k_host" {
  description = "Host serving the BDD100K archives. Plain HTTP, no login."
  type        = string
  default     = "128.32.162.150"
}

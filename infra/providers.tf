provider "aws" {
  region = var.aws_region

  # Guardrail: refuse to run against any account but the project account, so a
  # stale profile cannot apply this stack into the personal account.
  allowed_account_ids = [var.account_id]

  # Satisfies the "tag every resource with the project name" rule without having
  # to remember it on each resource.
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
      Repo      = "${var.github_owner}/${var.github_repo}"
    }
  }
}

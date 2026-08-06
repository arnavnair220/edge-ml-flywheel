# GitHub Actions authenticates to AWS by presenting a short-lived signed token
# instead of storing an access key. Nothing secret is kept in GitHub.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # AWS no longer verifies this thumbprint for GitHub's provider (it validates
  # against a trusted CA instead), but the API still requires the field.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  oidc_host = "token.actions.githubusercontent.com"

  # Exact subject strings GitHub puts in the token. Deliberately not wildcards:
  # a loose match here would let other repositories assume these roles.
  sub_pull_request = "repo:${var.github_owner}/${var.github_repo}:pull_request"
  sub_main_branch  = "repo:${var.github_owner}/${var.github_repo}:ref:refs/heads/main"

  state_bucket_arn = "arn:aws:s3:::${var.state_bucket}"
  state_lock_arn   = "arn:aws:s3:::${var.state_bucket}/${var.state_key}.tflock"
  state_object_arn = "arn:aws:s3:::${var.state_bucket}/${var.state_key}"
}

# ---------------------------------------------------------------------------
# Plan role: assumed by pull requests. Reads everything, changes nothing.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "plan_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = [local.sub_pull_request]
    }
  }
}

resource "aws_iam_role" "plan" {
  name                 = "${var.project}-gha-plan"
  description          = "Assumed by pull requests to run terraform plan. Read-only."
  assume_role_policy   = data.aws_iam_policy_document.plan_trust.json
  max_session_duration = 3600
}

resource "aws_iam_role_policy_attachment" "plan_readonly" {
  role       = aws_iam_role.plan.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

# ReadOnlyAccess cannot write, but plan still needs to take and release the
# state lock. Scoped to the lock object only, so a plan cannot overwrite state.
data "aws_iam_policy_document" "plan_state_lock" {
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket_arn]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.state_object_arn]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [local.state_lock_arn]
  }
}

resource "aws_iam_role_policy" "plan_state_lock" {
  name   = "state-lock"
  role   = aws_iam_role.plan.id
  policy = data.aws_iam_policy_document.plan_state_lock.json
}

# ---------------------------------------------------------------------------
# Apply role: assumed only by pushes to main. Can change infrastructure.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "apply_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = [local.sub_main_branch]
    }
  }
}

resource "aws_iam_role" "apply" {
  name                 = "${var.project}-gha-apply"
  description          = "Assumed by pushes to main to run terraform apply."
  assume_role_policy   = data.aws_iam_policy_document.apply_trust.json
  max_session_duration = 3600
}

# Broad on purpose: this account is isolated, region-locked by SCP, and budget
# alarmed, so blast radius is already bounded. The label wall that the project
# depends on belongs in an SCP, not here, because this role can edit IAM.
resource "aws_iam_role_policy_attachment" "apply_admin" {
  role       = aws_iam_role.apply.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

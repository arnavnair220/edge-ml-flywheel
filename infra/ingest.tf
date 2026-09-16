# The ingest identity: the one principal allowed to write `raw/`, and the one
# allowed to read a withheld label out of it.
#
# The role lands before the CodeBuild project that assumes it, because the data
# bucket policy is written in terms of its ARN. Ingest itself runs outside any
# VPC -- no subnets, no gateways, nothing left running -- which is what keeps the
# no-NAT rule out of tension with a job that has to reach the public internet.

locals {
  ingest_project_name = "${var.project}-ingest"
  ingest_project_arn  = "arn:aws:codebuild:${var.aws_region}:${var.account_id}:project/${local.ingest_project_name}"

  ingest_role_name = "${var.project}-ingest"
  ingest_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.ingest_role_name}"

  # The two allowlists the data bucket policy denies against. Separate lists
  # because they answer different questions and grow at different times: Phase 3's
  # oracle Lambda becomes a label reader and never a raw writer, and a future
  # re-ingest under a new `label_source` is a raw writer.
  #
  # Composed from the account and the role name rather than read off
  # `aws_iam_role.ingest.arn`, which is the difference between a plan that prints
  # the label wall and a plan that prints "(known after apply)". These two
  # statements are the ones a reviewer most needs to read *before* they take
  # effect, and a role ARN is fully determined by the account and the name -- so
  # the only thing the indirection bought was hiding them. Both spellings derive
  # from a role name, so an ARN here cannot name a role this stack does not
  # create.
  #
  # The partitioner is a label reader and not a raw writer, which is the shape
  # these two lists exist to express. It writes the boxes for `bootstrap` and
  # `eval` under its own partition prefix, and reading `raw/labels/` is the only
  # way to obtain them: cohort is a column in the assignments parquet, so no
  # narrower prefix names a cohort's labels. That widens the wall from one
  # principal to two, which is the floor rather than a concession -- something has
  # to read a raw label to write those files, and a separate job for it would hold
  # the identical grant behind an additional role and buildspec.
  #
  # What keeps the widening bounded is not IAM. `cohort_labels_prefix` raises on
  # any cohort outside `LABELED_COHORTS`, so no key this role can construct
  # addresses a `pool` label, and the partitioner asserts each file holds exactly
  # its cohort's IDs before writing either one.
  # The oracle is the third and last label reader, and the only one that reads a
  # *withheld* label. Ingest writes the archive and the partitioner copies out the
  # two cohorts the draw labels; this is the one principal that opens a `pool`
  # document, which is the read the whole budget exists to meter. Its own policy
  # denies it the `val` half of the tree, so the widening here is to `train/` in
  # practice -- see `oracle.tf`.
  raw_writer_arns   = [local.ingest_role_arn]
  label_reader_arns = [local.ingest_role_arn, local.partition_role_arn, local.oracle_role_arn]
}

# Conditioned on the calling project, not just the service. Without
# `aws:SourceArn`, any CodeBuild project in any account that could name this role
# would be trusted by it -- the confused-deputy shape. The ARN is composed from
# the name rather than referenced off the resource so that this role can exist
# before the project does.
data "aws_iam_policy_document" "ingest_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [local.ingest_project_arn]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = local.ingest_role_name
  description        = "BDD100K ingest. The only principal that may write raw/ or read a withheld label."
  assume_role_policy = data.aws_iam_policy_document.ingest_trust.json
}

data "aws_iam_policy_document" "ingest" {
  # Listing is scoped by prefix rather than granted over the bucket, so ingest
  # cannot enumerate the artifacts of a run it has nothing to do with. The two
  # prefixes are the only ones it touches: it writes `raw/`, and it writes the
  # manifest under `derived/`.
  statement {
    sid       = "ListOwnPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket", "s3:ListBucketMultipartUploads"]
    resources = [local.bucket_arns["data"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["raw/*", "derived/*", "raw/", "derived/"]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"]]
  }

  # `AbortMultipartUpload` is here because a 5.7 GB archive is uploaded in parts
  # and a retry that cannot abort its predecessor leaves parts that are billed
  # forever and appear in no listing. The lifecycle rule in `storage.tf` is the
  # backstop; being able to clean up in-band is the fix.
  statement {
    sid    = "WriteRawAndDerived"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = [
      "${local.bucket_arns["data"]}/raw/*",
      "${local.bucket_arns["data"]}/derived/*",
    ]
  }

  # Deliberately no DynamoDB grant at all. Ingest held a write on a table of
  # withheld labels while that table was the mechanism keeping `eval` out of
  # reach; the oracle now enforces that in code against the assignments, and the
  # table is gone (see `tables.tf`). Ingest writes the archive to `raw/` and
  # nothing else.

  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]

    resources = [
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.ingest_project_name}",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.ingest_project_name}:*",
    ]
  }

  # Cloning the repo is CodeBuild acting as this role, so the permission to
  # borrow the connection's short-lived token belongs here rather than on the
  # project.
  #
  # Unlike the label wall above, this ARN cannot be composed from names: the
  # connection carries a server-generated UUID, so it is genuinely "known after
  # apply" and the indirection is forced rather than chosen.
  #
  # Both service prefixes, because CodeConnections is the rename of CodeStar
  # Connections and the authorization for a given call is not guaranteed to have
  # followed the rename everywhere. A denial here surfaces as a build that fails
  # before the first log line, which is the least diagnosable place to be short a
  # permission.
  statement {
    sid    = "UseTheGitHubConnection"
    effect = "Allow"

    actions = [
      "codeconnections:GetConnection",
      "codeconnections:GetConnectionToken",
      "codeconnections:UseConnection",
      "codestar-connections:GetConnection",
      "codestar-connections:GetConnectionToken",
      "codestar-connections:UseConnection",
    ]

    resources = [
      aws_codeconnections_connection.github.arn,
      replace(aws_codeconnections_connection.github.arn, ":codeconnections:", ":codestar-connections:"),
    ]
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "ingest"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

# ---------------------------------------------------------------------------
# How the build gets the code
# ---------------------------------------------------------------------------

# A connection rather than a personal access token, because the repo is private
# and a PAT would be the long-lived key this project has none of. AWS holds and
# rotates the credential; nothing secret is stored in the account or in git.
#
# **This resource is created in `PENDING` and does nothing until the GitHub App
# authorization is completed by hand in the console**, which is why it joins the
# state bucket and the BDD100K license as documented bootstrap. Terraform cannot
# perform an OAuth handshake, so the alternative is not automation, it is a
# secret in a variable. `terraform output ingest_connection_status` reports
# whether the handshake has happened.
resource "aws_codeconnections_connection" "github" {
  name          = "${var.project}-github"
  provider_type = "GitHub"
}

# Account-and-region scoped, not per project: one credential per source
# provider. Named as a separate resource rather than inline on the project
# because that is what the API models, and because a second project later
# reuses this one rather than declaring its own.
resource "aws_codebuild_source_credential" "github" {
  auth_type   = "CODECONNECTIONS"
  server_type = "GITHUB"
  token       = aws_codeconnections_connection.github.arn
}

# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

# Created here rather than left to CodeBuild so that retention is set. Ingest
# logs are the record of where 160,000 objects came from, and CloudWatch's
# default is to keep them forever and bill for it.
resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/aws/codebuild/${local.ingest_project_name}"
  retention_in_days = 90
}

# The whole point of this resource is what it does *not* declare.
#
# **No `vpc_config`.** A CodeBuild project outside a VPC has ordinary outbound
# internet, the way a laptop does, so the 5.7 GB pull from the Berkeley host
# needs no NAT gateway -- which at ~$32/month would be twice the entire project
# budget and would exist solely to serve a job that runs a handful of times.
# This is the one component that has to reach the public internet, and it is
# also the one that never touches a subnet.
#
# **No webhook.** Ingest is started by hand:
#   aws codebuild start-build --project-name edge-ml-flywheel-ingest
# A webhook would re-download 5.7 GB and rewrite `raw/` on every push to main.
#
# **No artifacts.** Everything this build produces it writes to S3 itself,
# through keys built by `edge_ml_flywheel.conventions`. A CodeBuild artifact
# would be a second, differently-named copy in a bucket nothing else reads.
resource "aws_codebuild_project" "ingest" {
  name          = local.ingest_project_name
  description   = "BDD100K ingest. Started by hand, outside any VPC, nothing left running."
  service_role  = aws_iam_role.ingest.arn
  build_timeout = 180

  # A ceiling, not an estimate. Measured work is roughly 30-75 minutes -- the
  # download dominates and the Berkeley host sets that pace -- and SMALL is
  # ~$0.005/min, so the ceiling costs at most ~$0.90 on a build that has hung.
  # Set below the 8-hour maximum precisely so a hang is bounded.

  source {
    type      = "GITHUB"
    location  = "https://github.com/${var.github_owner}/${var.github_repo}.git"
    buildspec = "buildspecs/ingest.yml"

    # Nothing in the build reads history, and a shallow clone of a repo this
    # size is the difference between a second and several.
    git_clone_depth = 1

    # This is a data job, not CI. A commit status reading "build failed" against
    # a green commit because the Berkeley host was down would be actively
    # misleading.
    report_build_status = false
  }

  # The default when `start-build` is given no `--source-version`. Overridable
  # per build, which is the practical reason this project uses a connection at
  # all: iterating on an hour-long job by pushing to main is not a loop anyone
  # should have to run.
  source_version = "refs/heads/main"

  artifacts {
    type = "NO_ARTIFACTS"
  }

  # A cache would hold the uv-resolved environment, which takes seconds against
  # a download measured in tens of minutes, and would then need its own S3
  # location and lifecycle rule.
  cache {
    type = "NO_CACHE"
  }

  environment {
    type         = "LINUX_CONTAINER"
    compute_type = "BUILD_GENERAL1_SMALL"

    # 2 vCPU and 3 GB is enough because nothing here holds the dataset in
    # memory: images are hashed and inspected one at a time, and the manifest is
    # 80,000 rows of scalars plus box areas. The binding resource is the 64 GB
    # build volume, against ~11 GB of archives plus extract.
    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = false

    environment_variable {
      name  = "DATA_BUCKET"
      value = local.bucket_names["data"]
    }

    # Overridable per build with `--environment-variables-override`. The
    # download page's buttons point at this raw IP; `dl.yf.io` resolves to the
    # same host and serves byte-identical files, so this is the knob to turn if
    # either name stops answering.
    environment_variable {
      name  = "BDD100K_HOST"
      value = var.bdd100k_host
    }
  }

  logs_config {
    cloudwatch_logs {
      status     = "ENABLED"
      group_name = aws_cloudwatch_log_group.ingest.name
    }

    # S3 build logs would be a third copy of the same lines, in a bucket whose
    # lifecycle rules were written for telemetry.
    s3_logs {
      status = "DISABLED"
    }
  }

  # The credential is account-level and is not referenced by the project, so
  # Terraform cannot infer the ordering. A project created first clones with no
  # credential and fails.
  depends_on = [aws_codebuild_source_credential.github]
}

# The partition identity: reads the manifest, writes one partition version, and
# can reach nothing else in the data bucket.
#
# A separate role rather than a reuse of ingest's, which can already write every
# `derived/` prefix. Ingest is the only principal permitted to write `raw/` or
# read a withheld label, and those two grants are the whole label wall; a job
# that needs neither should not hold them because it happens to write to the same
# bucket. This role is deliberately absent from `raw_writer_arns` and
# `label_reader_arns`, so the data bucket policy denies it both, and its own
# policy never mentions `raw/` at all.

locals {
  partition_project_name = "${var.project}-partition"
  partition_project_arn  = "arn:aws:codebuild:${var.aws_region}:${var.account_id}:project/${local.partition_project_name}"

  partition_role_name = "${var.project}-partition"
  partition_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.partition_role_name}"

  # The manifest is the input and one partition version is the output. Written as
  # prefixes here and as `MANIFEST_PREFIX` / `partition_prefix` in the package,
  # which are the same two strings: the buildspec prints them out of the package
  # rather than spelling them, so a drift between these and the code is a build
  # that copies nothing rather than a policy that permits the wrong prefix.
  manifest_objects  = "${local.bucket_arns["data"]}/derived/manifest/*"
  partition_objects = "${local.bucket_arns["data"]}/derived/partition_version=*"
}

# Conditioned on the calling project for `ingest_trust`'s reason: without
# `aws:SourceArn` any CodeBuild project that could name this role would be
# trusted by it.
data "aws_iam_policy_document" "partition_trust" {
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
      values   = [local.partition_project_arn]
    }
  }
}

resource "aws_iam_role" "partition" {
  name               = local.partition_role_name
  description        = "Cohort assignment. Reads derived/manifest/, writes one partition version."
  assume_role_policy = data.aws_iam_policy_document.partition_trust.json
}

data "aws_iam_policy_document" "partition" {
  # Scoped to `derived/`, so this role cannot enumerate `raw/` even to discover
  # what is there. The trailing-slash entries are the prefixes themselves, which
  # a `--recursive` copy lists before it reads.
  statement {
    sid       = "ListDerived"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["data"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["derived/*", "derived/"]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"]]
  }

  # Read-only on the manifest, which is ingest's output and immutable to this
  # job. A partitioner that could rewrite image facts could make its own
  # assignment consistent with a manifest nobody else agrees with.
  statement {
    sid       = "ReadTheManifest"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.manifest_objects]
  }

  # The read is what the redraw guard needs: the buildspec fetches any
  # `_partition.json` already at this prefix and refuses to continue if its seed,
  # sizes or draw rule disagree with the code. No `DeleteObject` and no multipart
  # actions -- the two objects are 715 KB and 400 bytes, so a single PUT each, and
  # a re-drawn version is byte-identical by construction rather than something to
  # clear out first.
  statement {
    sid       = "WriteOnePartitionVersion"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.partition_objects]
  }

  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]

    resources = [
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.partition_project_name}",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.partition_project_name}:*",
    ]
  }

  # Cloning the repo is CodeBuild acting as this role. The connection itself is
  # account-level and shared with ingest -- one credential per source provider --
  # so this is the second consumer of the resource declared in `ingest.tf`, not a
  # second connection.
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

resource "aws_iam_role_policy" "partition" {
  name   = "partition"
  role   = aws_iam_role.partition.id
  policy = data.aws_iam_policy_document.partition.json
}

# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "partition" {
  name              = "/aws/codebuild/${local.partition_project_name}"
  retention_in_days = 90
}

# Same shape as the ingest project and for the same reasons -- no `vpc_config`,
# no webhook, no artifacts -- with one difference worth naming: this job reaches
# nothing on the public internet except GitHub and PyPI, so it is outside a VPC
# for convenience rather than out of necessity.
#
# Started by hand:
#   aws codebuild start-build --project-name edge-ml-flywheel-partition
#
# A webhook would redraw the partition on every push to main. The draw is
# deterministic, so that would be harmless today and would silently stop being
# harmless the first time a version's seed was edited -- which is the case the
# redraw guard in the buildspec exists to refuse.
resource "aws_codebuild_project" "partition" {
  name          = local.partition_project_name
  description   = "Cohort assignment for one partition_version. Started by hand."
  service_role  = aws_iam_role.partition.arn
  build_timeout = 20

  # A ceiling on a hang, not an estimate. The work is a 5 MB download, 80,000
  # digests and a 715 KB upload -- under two minutes including the uv sync.

  source {
    type            = "GITHUB"
    location        = "https://github.com/${var.github_owner}/${var.github_repo}.git"
    buildspec       = "buildspecs/partition.yml"
    git_clone_depth = 1

    # A data job, not CI. A red commit status because the bucket was misread
    # would be misleading about the commit.
    report_build_status = false
  }

  source_version = "refs/heads/main"

  artifacts {
    type = "NO_ARTIFACTS"
  }

  cache {
    type = "NO_CACHE"
  }

  environment {
    type         = "LINUX_CONTAINER"
    compute_type = "BUILD_GENERAL1_SMALL"

    # 2 vCPU and 3 GB against a job that holds 80,000 image IDs and their tickets
    # in memory -- tens of megabytes. The manifest's box column is never read.
    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = false

    environment_variable {
      name  = "DATA_BUCKET"
      value = local.bucket_names["data"]
    }

    # Which version to draw, overridable per build with
    # `--environment-variables-override`. A value not in `conventions.PARTITIONS`
    # is refused by the CLI's own choices before anything is read, so this
    # duplicating a number that lives in code cannot produce a wrong partition --
    # only a build that stops.
    environment_variable {
      name  = "PARTITION_VERSION"
      value = "0"
    }
  }

  logs_config {
    cloudwatch_logs {
      status     = "ENABLED"
      group_name = aws_cloudwatch_log_group.partition.name
    }

    s3_logs {
      status = "DISABLED"
    }
  }

  depends_on = [aws_codebuild_source_credential.github]
}

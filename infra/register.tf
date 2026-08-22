# The registration identity: mints a run and claims its name in `runs`. It can
# write one item to one table and reach nothing else in the account.
#
# A CodeBuild project for a job that makes a single `PutItem`, which is worth
# justifying. `git_commit` is a required field of the registration and it is
# written once, so a SHA read off a working tree with uncommitted changes names
# a commit that did not produce the run and can never be corrected. CodeBuild
# resolves the source itself and exports it as
# `CODEBUILD_RESOLVED_SOURCE_VERSION`, which makes the field true by
# construction rather than by whoever started the run being careful. The
# CloudWatch log is the second reason: a run's first act leaves a record even
# though the item it wrote says nothing about who wrote it.
#
# This role is deliberately absent from `raw_writer_arns` and
# `label_reader_arns`, and holds no S3 grant at all. Registering a run is not a
# reason to be able to read the dataset.

locals {
  register_project_name = "${var.project}-register"
  register_project_arn  = "arn:aws:codebuild:${var.aws_region}:${var.account_id}:project/${local.register_project_name}"

  register_role_name = "${var.project}-register"
}

# Conditioned on the calling project for `ingest_trust`'s reason: without
# `aws:SourceArn` any CodeBuild project that could name this role would be
# trusted by it.
data "aws_iam_policy_document" "register_trust" {
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
      values   = [local.register_project_arn]
    }
  }
}

resource "aws_iam_role" "register" {
  name               = local.register_role_name
  description        = "Run registration. Writes one item to the runs table and holds nothing else."
  assume_role_policy = data.aws_iam_policy_document.register_trust.json
}

data "aws_iam_policy_document" "register" {
  # `PutItem` and `GetItem`, on one table, and nothing else -- no `UpdateItem`,
  # no `DeleteItem`, no `Query`, no `Scan`. The registration is write-once, and
  # the difference between "the collision was refused" and "the earlier run's
  # config was overwritten" is exactly the absence of `UpdateItem` here. The
  # conditional expression is the guard against a *second* write; this is the
  # guard against a *different kind* of write.
  #
  # `GetItem` is the read-back: the build confirms what landed rather than
  # trusting a call that returned without raising, which is the same shape as
  # the partitioner re-reading its uploaded `_partition.json`.
  statement {
    sid    = "RegisterOneRun"
    effect = "Allow"

    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
    ]

    resources = [aws_dynamodb_table.runs.arn]
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
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.register_project_name}",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.register_project_name}:*",
    ]
  }

  # Cloning the repo is CodeBuild acting as this role. The connection is
  # account-level and shared with ingest and the partitioner -- one credential
  # per source provider -- so this is a third consumer of the resource declared
  # in `ingest.tf`, not a third connection.
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

resource "aws_iam_role_policy" "register" {
  name   = "register"
  role   = aws_iam_role.register.id
  policy = data.aws_iam_policy_document.register.json
}

# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

# Longer retention than the data jobs, and the reason is the run rather than the
# build: this log is where a run's slug, selector, budget and version trio were
# chosen, and a run's cycles are measured in weeks. An ingest log stops being
# interesting once the bucket is verified.
resource "aws_cloudwatch_log_group" "register" {
  name              = "/aws/codebuild/${local.register_project_name}"
  retention_in_days = 365
}

# Started by hand, with every field of the registration overridden per build:
#
#   aws codebuild start-build --project-name edge-ml-flywheel-register \
#     --environment-variables-override \
#       name=RUN_SLUG,value=v1-uncertainty,type=PLAINTEXT \
#       name=SELECTOR,value=uncertainty,type=PLAINTEXT \
#       name=RUN_NOTE,value="first real loop",type=PLAINTEXT
#
# No webhook, and this is the one project where that is not a preference. Ingest
# and the partitioner are idempotent -- a redundant build re-derives the same
# bytes -- and this one mints a new `run_id` on every invocation. A push to main
# would start a run.
resource "aws_codebuild_project" "register" {
  name          = local.register_project_name
  description   = "Mints a run_id and claims it in the runs table. Started by hand."
  service_role  = aws_iam_role.register.arn
  build_timeout = 10

  # A ceiling on a hang. The work is a uv sync and two DynamoDB calls.

  source {
    type            = "GITHUB"
    location        = "https://github.com/${var.github_owner}/${var.github_repo}.git"
    buildspec       = "buildspecs/register.yml"
    git_clone_depth = 1

    # An operator action, not CI. A red commit status because a slug collided
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

    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = false

    # Defaults that describe the loop as designed. `RUN_SLUG` and `RUN_NOTE`
    # have none on purpose -- the buildspec fails when either is empty, because
    # a default slug is how two runs come to be named the same thing and a
    # default note is how the field stops meaning anything.
    environment_variable {
      name  = "SELECTOR"
      value = "uncertainty"
    }

    # 1,000 labels a cycle against a 62,000-image pool: 1.6% of what was scored,
    # which is the selectivity a ranking needs to diverge from a random draw.
    # Fixed for the life of a run once registered.
    environment_variable {
      name  = "LABEL_BUDGET"
      value = "1000"
    }

    # The drawn partition. A version not in `conventions.PARTITIONS` is refused
    # by the CLI's own choices before anything is written.
    environment_variable {
      name  = "PARTITION_VERSION"
      value = "0"
    }

    # No registry backs these two yet, unlike the partition version. They are
    # the numbers a model manifest is later checked against, so they start at 1
    # and move when the class set or the training recipe does.
    environment_variable {
      name  = "CLASS_SET_VERSION"
      value = "1"
    }

    environment_variable {
      name  = "RECIPE_VERSION"
      value = "1"
    }
  }

  logs_config {
    cloudwatch_logs {
      status     = "ENABLED"
      group_name = aws_cloudwatch_log_group.register.name
    }

    s3_logs {
      status = "DISABLED"
    }
  }

  depends_on = [aws_codebuild_source_credential.github]
}

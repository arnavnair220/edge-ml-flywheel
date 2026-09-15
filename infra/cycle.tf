# The control plane: the state machine that runs a cycle, the one Lambda it
# calls, and the two roles between them.
#
# **The control flow is in `cycle.asl.json` and nowhere else** (design section
# 5). This file creates that definition and grants what it needs; it decides no
# sequencing of its own, so a change to the loop is a change to one JSON file
# that a reader can follow top to bottom.
#
# **The two identities are split by what they may do, not by tidiness.** The
# state machine starts training jobs and passes them the training role; the
# Lambda writes a manifest and reads a registration. Neither can do the other's
# work: the Lambda holds no SageMaker grant and no `iam:PassRole`, so the
# function that builds a `CreateTrainingJob` request physically cannot create
# one, and the request is only ever executed by the state machine, in the open,
# where the execution history records it.
#
# Both are absent from `raw_writer_arns` and `label_reader_arns`. Orchestrating a
# cycle is not a reason to be able to read a withheld label, and the control
# plane's own reads stop at the labels a run already owns -- which the Lambda
# needs only to list the image IDs the manifest names.

locals {
  control_function_name = "${var.project}-control"
  cycle_machine_name    = "${var.project}-cycle"

  # The same interpreter `pyproject.toml` pins and the training container ships,
  # which is what makes the package that runs here and the package that runs
  # there one package.
  control_runtime = "python3.12"

  # Mirrors of the key builders in `conventions`, as in `training.tf`.
  #
  # `training_manifest_key` and `training_code_key`, which are the two objects
  # `prepare` writes. Same prefix the training role reads them from, which is the
  # point: one cycle hands its seeds exactly what this wrote.
  control_training_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/training/*"

  # `cohort_labels_prefix(v, BOOTSTRAP)` and `purchases_run_prefix(run)`. Read
  # only to recover the image IDs the cumulative labeled set covers, which is
  # what the manifest names. `cohort=eval/` is absent here and denied to every
  # principal by the bucket, so the manifest cannot name an eval frame even if
  # this role were wrong about which prefix it was reading.
  control_label_objects = [
    "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=bootstrap/*",
    "${local.bucket_arns["data"]}/derived/purchases/*",
  ]
}

# ---------------------------------------------------------------------------
# The Lambda
# ---------------------------------------------------------------------------

# The deployment package: `src/`, so the zip root holds `edge_ml_flywheel/` and
# the handler resolves as a module path. Built by Terraform rather than by a
# build step, because the package is pure Python -- there is nothing to compile
# and no platform wheel to match, which is exactly what makes zipping a directory
# sufficient here and insufficient for pyarrow below.
data "archive_file" "control" {
  type        = "zip"
  source_dir  = "${path.module}/../src"
  output_path = "${path.module}/build/control.zip"

  # Bytecode is a function of an interpreter that is not the Lambda's, and
  # shipping it invites a stale `.pyc` shadowing a module that changed.
  excludes = ["**/__pycache__/**"]
}

# `container/train.py` and `requirements.txt`, in a layer.
#
# They are in the deployment for one reason: `prepare` builds the source archive
# the training container unpacks, and SageMaker's script mode requires both of
# them at the root of that archive. A Lambda has no git checkout to build it
# from, so the tree has to arrive at deploy time -- and a Lambda package has
# exactly one source directory, while these two files live outside `src/`. A
# layer is the one place a Lambda can be handed files from a second directory,
# so they land at `/opt` and `control.handler` tars them together with the
# package from `/var/task`.
data "archive_file" "entry_point" {
  type        = "zip"
  source_dir  = "${path.module}/../container"
  output_path = "${path.module}/build/entrypoint.zip"
}

resource "aws_lambda_layer_version" "entry_point" {
  layer_name          = "${var.project}-entrypoint"
  description         = "container/train.py and requirements.txt, which script mode requires at the root of the training source archive."
  filename            = data.archive_file.entry_point.output_path
  source_code_hash    = data.archive_file.entry_point.output_base64sha256
  compatible_runtimes = [local.control_runtime]
}

data "aws_iam_policy_document" "control_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "control" {
  name               = local.control_function_name
  description        = "The control plane's Lambda. Writes a cycle's training inputs and reads a run registration, and holds no SageMaker grant."
  assume_role_policy = data.aws_iam_policy_document.control_trust.json
}

data "aws_iam_policy_document" "control" {
  # The same deny the training role carries, for the same reason and with less
  # excuse: this role has no business near `raw/labels/` at all. The bucket
  # policy refuses it already, as an allowlist this role is not on; the explicit
  # deny is what beats an allow someone grants later without reading storage.tf.
  statement {
    sid    = "NeverReadAWithheldLabel"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [local.raw_label_objects]
  }

  # `prepare` reads the labeled set to recover the image IDs it names, and the
  # listing is scoped to the two prefixes it reads for `training.tf`'s reason: a
  # listing is how a walk of the label tree finds documents the assignments
  # refuse to name.
  statement {
    sid       = "ListTheLabelPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["data"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "derived/partition_version=*/labels/cohort=bootstrap/*",
        "derived/purchases/*",
      ]
    }
  }

  statement {
    sid       = "ReadTheLabelsThisRunOwns"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = local.control_label_objects
  }

  # The manifest and the source archive. `PutObject` and no delete against a
  # write-once bucket, and `ListBucket` because `prepare` refuses to overwrite a
  # cycle that has already been prepared -- that refusal is a listing, and
  # without this grant it would read as "not there yet" and overwrite the record
  # of what a challenger trained on.
  statement {
    sid       = "WriteTheCycleTrainingInputs"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.control_training_objects]
  }

  statement {
    sid       = "ListTheCyclePrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["run_id=*/cycle=*/training/*"]
    }
  }

  # `GetItem` and nothing else. The partition and class set a job trains under
  # come off the registration, and this role reading them is the mechanism that
  # keeps them off the state machine's input -- where they would be a flag, and a
  # flag is how a job comes to train under a class set its run never declared.
  statement {
    sid       = "ReadTheRunRegistration"
    effect    = "Allow"
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.runs.arn]
  }

  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]

    resources = ["${aws_cloudwatch_log_group.control.arn}:*"]
  }
}

resource "aws_iam_role_policy" "control" {
  name   = "control"
  role   = aws_iam_role.control.id
  policy = data.aws_iam_policy_document.control.json
}

# Created here rather than left to the service, so retention is set and so the
# role's grant above can name one group instead of the account's logs.
resource "aws_cloudwatch_log_group" "control" {
  name              = "/aws/lambda/${local.control_function_name}"
  retention_in_days = 365
}

resource "aws_lambda_function" "control" {
  function_name = local.control_function_name
  description   = "Two steps of a cycle that are Python: writing the image manifest, and building one seed's training request."
  role          = aws_iam_role.control.arn
  handler       = "edge_ml_flywheel.control.handler.handler"
  runtime       = local.control_runtime
  architectures = ["x86_64"]

  filename         = data.archive_file.control.output_path
  source_code_hash = data.archive_file.control.output_base64sha256

  # `prepare` downloads a run's label parquet, reads the image IDs out of it and
  # uploads a manifest. Minutes at the outside, and the ceiling is a bound on a
  # hang rather than an estimate.
  timeout = 300

  # pyarrow reading a few megabytes of parquet. Lambda scales CPU with memory, so
  # this is as much about the read finishing quickly as about fitting.
  memory_size = 1024

  # Two layers, for two things the deployment package cannot carry. The managed
  # one supplies pyarrow, which is a platform wheel and so cannot come out of
  # `archive_file` over a source directory; ours carries the two script-mode root
  # files, which live outside `src/`.
  layers = [
    var.pyarrow_layer_arn,
    aws_lambda_layer_version.entry_point.arn,
  ]

  environment {
    variables = {
      # Read by `training.job.environment`'s callers and by boto3. Set
      # explicitly rather than relied on, since the Lambda runtime's own region
      # variable is not one this package names.
      AWS_DEFAULT_REGION = var.aws_region
    }
  }

  depends_on = [aws_cloudwatch_log_group.control]
}

# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "cycle_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
  }
}

resource "aws_iam_role" "cycle" {
  name               = local.cycle_machine_name
  description        = "The cycle state machine. Claims a cycle, invokes the control function, and starts training jobs as the training role."
  assume_role_policy = data.aws_iam_policy_document.cycle_trust.json
}

data "aws_iam_policy_document" "cycle" {
  statement {
    sid       = "InvokeTheControlFunction"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.control.arn]
  }

  # `UpdateItem` and `GetItem` on one table. No `PutItem` and no `DeleteItem`:
  # the run's control item is created by the registration and advanced by this
  # role, and the difference between "the cap was reached" and "the counter was
  # reset to zero mid-run" is exactly the absence of a put here.
  statement {
    sid    = "ClaimACycle"
    effect = "Allow"

    actions = [
      "dynamodb:UpdateItem",
      "dynamodb:GetItem",
    ]

    resources = [aws_dynamodb_table.fleet_config.arn]
  }

  # `.sync` needs all four: create the job, poll it, stop it if the execution is
  # aborted, and read the tags it was created with.
  statement {
    sid    = "RunTrainingJobs"
    effect = "Allow"

    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
    ]

    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:training-job/*",
    ]
  }

  # The one grant that makes the label wall a question about this role. Handing
  # SageMaker the training role is how a job gets any data access at all, so the
  # condition pins the service it can be handed to: without it, a principal that
  # could start a training job could pass this role to anything that accepts one.
  statement {
    sid       = "PassTheTrainingRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.training.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # `.sync` is implemented with a managed EventBridge rule that tells Step
  # Functions when the job reaches a terminal state, and the rule is created by
  # this role on first use. The name is the service's, not ours.
  statement {
    sid    = "ManageTheSyncCompletionRule"
    effect = "Allow"

    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule",
    ]

    resources = [
      "arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
    ]
  }

  # Step Functions delivers its own execution logs, which is a different set of
  # actions from writing them and is not resource-scopable: the delivery API
  # takes no resource, so `*` here is the service's shape rather than a widened
  # grant. What it can reach is still one group, because the log destination is
  # fixed on the state machine below.
  statement {
    sid    = "DeliverExecutionLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogDelivery",
      "logs:GetLogDelivery",
      "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries",
      "logs:PutResourcePolicy",
      "logs:DescribeResourcePolicies",
      "logs:DescribeLogGroups",
    ]

    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "cycle" {
  name   = "cycle"
  role   = aws_iam_role.cycle.id
  policy = data.aws_iam_policy_document.cycle.json
}

# A year, for `register`'s reason rather than a build log's: this is where a
# cycle's decisions are recorded -- which cycle was claimed, which seeds ran,
# which gate refused -- and a run's cycles are measured in weeks.
resource "aws_cloudwatch_log_group" "cycle" {
  name              = "/aws/vendedlogs/states/${local.cycle_machine_name}"
  retention_in_days = 365
}

# Standard rather than Express. An execution outlives a training job by design,
# Express caps at five minutes, and the execution history is the record of what a
# cycle did.
resource "aws_sfn_state_machine" "cycle" {
  name     = local.cycle_machine_name
  role_arn = aws_iam_role.cycle.arn
  type     = "STANDARD"

  # The two values the definition cannot know: what the function is called and
  # what the table is called. Everything else in that file is control flow, which
  # is why it is a file rather than a heredoc in here.
  definition = templatefile("${path.module}/cycle.asl.json", {
    control_function_arn = aws_lambda_function.control.arn
    fleet_config_table   = aws_dynamodb_table.fleet_config.name
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.cycle.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
}

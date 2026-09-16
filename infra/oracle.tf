# The oracle: the one principal in the account that may read a withheld label,
# and the function that charges for it.
#
# **This is a second Lambda over the same deployment package, and the second
# identity is the whole reason it exists.** Every other Python step of a cycle
# runs in the control function, which is denied `raw/labels/` outright -- see
# `cycle.tf`. The purchase cannot: it reads the boxes for a thousand pool images
# and files them under the run. One function holding both roles would mean the
# step that writes a training manifest could also read the 62,000 labels the
# oracle exists to sell, and every gate would still pass.
#
# So the split is not tidiness. It is the label wall drawn between two functions
# rather than trusted inside one, and it costs a role, a log group and twenty
# lines of Terraform pointing at an archive that already exists.
#
# **The `val` deny is the second half of the guarantee.** `oracle.cohorts` is the
# mechanism: the gate runs before a key is built, the key builder routes through
# it, and no function in the package returns a key under the split `eval` is
# drawn from. That is a property of the code. The deny below is a property of the
# account, and it is expressible only because `pool` draws from `train` and
# `eval` from `val` -- so one prefix separates them, and it costs this role no
# access it is entitled to. Contamination then needs a code bug and an IAM
# misconfiguration rather than either one alone.

locals {
  oracle_function_name = "${var.project}-oracle"

  oracle_role_name = "${var.project}-oracle"
  oracle_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.oracle_role_name}"

  # `raw_label_key(image_id, Split.VAL)`. The half of the label tree this role
  # must never read: `eval` and `reserve` are the only cohorts drawn from `val`,
  # and neither is ever sold. Denied rather than simply unallowed, because the
  # grant below is the whole of `raw/labels/` and an explicit deny is what beats
  # it.
  oracle_val_label_objects = "${local.bucket_arns["data"]}/raw/labels/*/val/*"

  # `raw_label_key(image_id, Split.TRAIN)`. Where a pool label lives, and the one
  # read this role exists to make. Wildcarded at `label_source` for
  # `storage.tf`'s reason: a future format lands inside the same statements
  # rather than outside them by default.
  oracle_train_label_objects = "${local.bucket_arns["data"]}/raw/labels/*/train/*"

  # `assignments_prefix(v)`, which is the cohort gate itself. Two columns and no
  # box: an assignment says which cohort an image is in, which is the fact the
  # eval wall is built on rather than a thing the wall keeps out.
  oracle_assignment_objects = "${local.bucket_arns["data"]}/derived/partition_version=*/assignments/*"

  # `purchase_labels_key(run_id, cycle)`. The only thing this role writes, and
  # deliberately not the whole of `derived/`: an oracle produces purchases, so it
  # has no reason to be able to overwrite a partition, a manifest or another run's
  # labels.
  oracle_purchase_objects = "${local.bucket_arns["data"]}/derived/purchases/*"

  # `selection_ranking_key(run_id, cycle)`. Read-only, and it is the batch: the
  # oracle takes its image IDs from the record of how they were chosen rather
  # than from an execution input, so a retry names the same set. A role that could
  # write here could choose its own batch after the fact.
  oracle_selection_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/selection/*"
}

data "aws_iam_policy_document" "oracle_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "oracle" {
  name               = local.oracle_role_name
  description        = "The label oracle. The one principal that may read a withheld pool label, and it may not read a val one."
  assume_role_policy = data.aws_iam_policy_document.oracle_trust.json
}

data "aws_iam_policy_document" "oracle" {
  # The deny first, because it is the statement this role exists to be read for.
  # `eval` is drawn from `val`, and a model trained on the exam scores better on
  # exactly the set it is measured against while every gate passes -- which is
  # the one failure in this project that is silent and unrecoverable.
  #
  # `EvalLabelsAreScoringOnly` in `storage.tf` says this about the partitioner's
  # copy of the same boxes. This says it about the archive they were copied from.
  statement {
    sid    = "NeverReadAValLabel"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [local.oracle_val_label_objects]
  }

  # The read the oracle exists to make, and the reason this role is on
  # `label_reader_arns`. `train/` holds `bootstrap` and `pool` as siblings --
  # cohort is a column in the assignments parquet and not a component of any key
  # -- so there is no narrower prefix, and `oracle.cohorts` is what refuses the
  # bootstrap images this grant can technically reach.
  statement {
    sid       = "ReadTheLabelsItSells"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.oracle_train_label_objects]
  }

  # No `ListBucket` over `raw/`, deliberately. A listing is how a walk of the
  # label tree finds the 62,000 documents the assignments refuse to name, and this
  # role never needs one: every key it opens is built from an image ID the gate
  # has already admitted. It can read a label it can name and cannot discover one.
  statement {
    sid       = "ListWhatItReadsAndWrites"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["data"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "derived/partition_version=*/assignments/*",
        "derived/purchases/*",
      ]
    }
  }

  statement {
    sid       = "ListTheSelectionPrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["run_id=*/cycle=*/selection/*"]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"], local.bucket_arns["artifacts"]]
  }

  statement {
    sid       = "ReadTheCohortGateAndTheBatch"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.oracle_assignment_objects, local.oracle_selection_objects]
  }

  # `PutObject` and no delete. A purchase is not write-once the way an artifact
  # is -- a replayed charge rewrites the same boxes over the same key, which is
  # what makes the label write safe to repeat after a crash between the charge and
  # the file. `GetObject` is what the next cycle's `prepare` reads, and this role
  # keeps it so a purchase can be confirmed by the function that made it.
  statement {
    sid    = "WriteTheLabelsItSold"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
    ]

    resources = [local.oracle_purchase_objects]
  }

  # The charge. One `TransactWriteItems` across two tables, each carrying its own
  # condition: the audit item refuses the second charge for a batch already
  # bought, and the ledger update refuses the overspend.
  #
  # `PutItem` on the audit log and no update or delete, because an audit item is
  # evidence that a charge happened and a role that could revise one could erase
  # a purchase the budget saw. `UpdateItem` on the ledger and no put, because the
  # item is created by the same expression that first debits it -- a `PutItem`
  # here could reset a spent cycle's remaining balance to its full cap.
  statement {
    sid    = "ChargeForABatch"
    effect = "Allow"

    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
    ]

    resources = [aws_dynamodb_table.audit_log.arn]
  }

  statement {
    sid    = "DebitTheLedger"
    effect = "Allow"

    actions = [
      "dynamodb:UpdateItem",
      "dynamodb:GetItem",
    ]

    resources = [aws_dynamodb_table.label_budget.arn]
  }

  # `GetItem` and nothing else, matching the control role. The budget a cycle is
  # capped at and the partition its cohorts are gated by both come off the
  # registration, which is what keeps them off the state machine's input -- where
  # they would be a flag, and a flag is how a run comes to spend a budget it never
  # declared.
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

    resources = ["${aws_cloudwatch_log_group.oracle.arn}:*"]
  }
}

resource "aws_iam_role_policy" "oracle" {
  name   = "oracle"
  role   = aws_iam_role.oracle.id
  policy = data.aws_iam_policy_document.oracle.json
}

# A year, matching the control function's. This log is where a run's spend is
# narrated -- which batch, how many labels, what was left -- and cost per label is
# a headline deliverable measured over weeks.
resource "aws_cloudwatch_log_group" "oracle" {
  name              = "/aws/lambda/${local.oracle_function_name}"
  retention_in_days = 365
}

# The same package the control function runs, at a different handler. One tree,
# two identities: `data.archive_file.control` is the whole of `src/`, so the
# oracle's modules are already in it and a second build would be a second thing to
# keep in step with the first.
#
# No entry-point layer, unlike the control function. That layer carries the three
# files script mode and the Processing commands name at the root of the source
# archive, and the oracle builds no archive -- it reads labels and writes parquet.
resource "aws_lambda_function" "oracle" {
  function_name = local.oracle_function_name
  description   = "The label oracle. Gates a batch against the partition, charges the ledger, and files the boxes it sold."
  role          = aws_iam_role.oracle.arn
  handler       = "edge_ml_flywheel.oracle.handler.handler"
  runtime       = local.control_runtime
  architectures = ["x86_64"]

  filename         = data.archive_file.control.output_path
  source_code_hash = data.archive_file.control.output_base64sha256

  # A thousand `GetObject` calls, one per image, plus a transaction and an upload.
  # Serial and small, so this is minutes at the outside and the ceiling is a bound
  # on a hang rather than an estimate.
  timeout = 600

  # pyarrow reading a ranking of 62,000 rows and writing a thousand labels.
  # Lambda scales CPU with memory, and the thousand sequential reads are the wall
  # clock here rather than the parquet.
  memory_size = 1024

  # pyarrow, for the ranking and the purchase file. The same managed layer the
  # control function takes it from, which is why both write snappy.
  layers = [var.pyarrow_layer_arn]

  depends_on = [aws_cloudwatch_log_group.oracle]
}

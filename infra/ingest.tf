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
  # because they answer different questions and grow at different times: Phase 4's
  # oracle Lambda becomes a label reader and never a raw writer, and a future
  # re-ingest under a new `label_source` is a raw writer.
  #
  # Composed from the account and the role name rather than read off
  # `aws_iam_role.ingest.arn`, which is the difference between a plan that prints
  # the label wall and a plan that prints "(known after apply)". These two
  # statements are the ones a reviewer most needs to read *before* they take
  # effect, and a role ARN is fully determined by the account and the name -- so
  # the only thing the indirection bought was hiding them. Both spellings derive
  # from `ingest_role_name`, so the ARN cannot name a role this stack does not
  # create.
  raw_writer_arns   = [local.ingest_role_arn]
  label_reader_arns = [local.ingest_role_arn]
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

  # Loading the withheld labels is the last Phase 1 step and the only write this
  # table ever takes. Deliberately no read: the loader has no reason to query what
  # it just wrote, and leaving the read out means the oracle is the only principal
  # in the account that can get a label out of DynamoDB, matching the bucket-level
  # wall on `raw/labels/`.
  statement {
    sid    = "LoadWithheldLabels"
    effect = "Allow"

    actions = [
      "dynamodb:PutItem",
      "dynamodb:BatchWriteItem",
      "dynamodb:DescribeTable",
    ]

    resources = [aws_dynamodb_table.oracle_labels.arn]
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
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.ingest_project_name}",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/codebuild/${local.ingest_project_name}:*",
    ]
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "ingest"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

# The evaluation identity: the role a SageMaker Processing job runs as when it
# matches a cycle's detections against ground truth and applies the gates.
#
# **This is the one principal in the account admitted to `labels/cohort=eval/`.**
# It is the single ARN on `eval_label_reader_arns` in `storage.tf`, which is what
# turns that bucket deny from "nobody" into an allowlist of one. Everything about
# the split between this role and `scoring.tf`'s exists to make that list short:
# scoring is the job that reads the `val` images and it holds no label grant of
# any kind, and this is the job that holds ground truth and never sees a
# checkpoint. Neither can do the other's work, so a reader asking what could
# possibly have contaminated the eval finds two identities and one answer each.
#
# **It reads the eval boxes and no other label.** `raw/labels/`,
# `cohort=bootstrap/` and `derived/purchases/` are denied below. The first is the
# withheld pool and the wall the oracle exists behind; the other two are the
# labels a run already owns, which are training's input and are no part of
# deciding whether the model that trained on them got better. An evaluation job
# that could read its own training set could compute a number over it, and the
# one number this project reports is over images nothing trained on.
#
# **It loads no model, so it needs no checkpoint.** `models/*` is absent from
# every statement here, which is the arrangement the three jobs share: training
# writes a checkpoint, scoring reads it, and this one reads neither -- it is
# handed what the scoring job wrote down.
#
# No `VpcConfig` on the jobs that assume it, for `training.tf`'s reason.

locals {
  evaluation_role_name = "${var.project}-evaluation"
  evaluation_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.evaluation_role_name}"

  # Mirrors of the key builders in `conventions`, as in `training.tf` and
  # `scoring.tf`. Wildcarded at the components a job varies by and fixed
  # everywhere else.

  # `cohort_labels_prefix(v, EVAL)`. The grant this role exists for, and the one
  # no other principal in the account holds. Wildcarded over `partition_version=`
  # so a re-partition is covered by the statement that already exists, matching
  # `storage.tf`'s `eval_label_objects`.
  eval_label_reader_objects = "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=eval/*"

  # `training_code_key`, `scoring_manifest_key` and `detections_prefix`: the
  # source archive this job unpacks, the document naming which images were
  # scored, and the boxes the scoring job wrote. The archive is the same object
  # the other two jobs ran, so the code that gated a model is the tree that
  # trained and scored it.
  evaluation_input_objects = [
    "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/training/*",
    "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/scoring/*",
    "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/detections/*",
  ]

  # `eval_prefix`. Read and written by the same statement below, and the overlap
  # is the design rather than a widening: a cycle writes its challenger's cached
  # match arrays here and reads the champion's from the cycle that produced them,
  # because the eval cohort is frozen and a champion is re-compared without being
  # re-scored (design section 7). A prefix per cycle and a write-once bucket are
  # what keep the two from being the same object.
  eval_cache_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/eval/*"

  # `gate_report_prefix`. The verdict, and the only other thing this role writes.
  # Deliberately not the cycle prefix: an evaluation job produces a report, so it
  # has no reason to be able to overwrite a model, a manifest or a selection
  # record filed under the same cycle.
  gate_report_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/gates/*"
}

# Conditioned on the account rather than the job, for `training_trust`'s reason.
data "aws_iam_policy_document" "evaluation_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
  }
}

resource "aws_iam_role" "evaluation" {
  name               = local.evaluation_role_name
  description        = "SageMaker evaluation jobs. The one principal admitted to the eval labels, and denied every other label in the account."
  assume_role_policy = data.aws_iam_policy_document.evaluation_trust.json
}

data "aws_iam_policy_document" "evaluation" {
  # The deny first, because a role that may read one label prefix is exactly the
  # role worth stating the other three about. `cohort=eval/` is absent from this
  # list and that absence is the whole grant.
  #
  # `raw/labels/*` covers the withheld pool, which this role has no more business
  # with than training does. The other two are the labels a run already owns: an
  # evaluation job that could read its own training set could report a number
  # measured over it, and that failure looks like a very good model rather than
  # like a bug.
  statement {
    sid    = "ReadNoLabelButTheEvalCohort"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [
      local.raw_label_objects,
      "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=bootstrap/*",
      "${local.bucket_arns["data"]}/derived/purchases/*",
    ]
  }

  # Every channel this job reads is an `S3Prefix`, which SageMaker enumerates
  # before it copies, so both buckets need a listing. Scoped by prefix for
  # `training.tf`'s reason.
  statement {
    sid       = "ListTheLabelChannel"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["data"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["derived/partition_version=*/labels/cohort=eval/*"]
    }
  }

  statement {
    sid       = "ListTheArtifactChannelPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "run_id=*/cycle=*/training/*",
        "run_id=*/cycle=*/scoring/*",
        "run_id=*/cycle=*/detections/*",
        "run_id=*/cycle=*/eval/*",
        "run_id=*/cycle=*/gates/*",
      ]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"], local.bucket_arns["artifacts"]]
  }

  # The grant the role exists for. One prefix, one cohort, read-only: the boxes
  # are frozen by `LabelsAreFrozenExceptThePartitioner`, so this role could not
  # rewrite them even if a statement here said it could.
  statement {
    sid       = "ReadTheEvalGroundTruth"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.eval_label_reader_objects]
  }

  statement {
    sid       = "ReadItsOwnInputs"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = local.evaluation_input_objects
  }

  # `PutObject` and no delete against a write-once bucket, matching the other two
  # job roles. `GetObject` on the cache prefix is a real read and not the upload
  # artifact it is in `scoring.tf`: this is how the champion's arrays arrive, and
  # it is what makes a re-comparison a download rather than a second scoring pass.
  statement {
    sid    = "WriteItsCachesAndReadTheChampions"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = [local.eval_cache_objects]
  }

  statement {
    sid    = "WriteTheGateReport"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = [local.gate_report_objects]
  }

  # The same group the scoring jobs stream to, since SageMaker writes all
  # Processing output to one group of its own. Retention is set there, in
  # `scoring.tf`, rather than twice.
  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]

    resources = [
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/sagemaker/ProcessingJobs",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/sagemaker/ProcessingJobs:*",
    ]
  }
}

resource "aws_iam_role_policy" "evaluation" {
  name   = "evaluation"
  role   = aws_iam_role.evaluation.id
  policy = data.aws_iam_policy_document.evaluation.json
}

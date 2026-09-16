# The scoring identity: the role a SageMaker Processing job runs as when it runs
# the model over the eval cohort and the remaining pool.
#
# **This role may not read a box, of any cohort, from anywhere.** That is the
# whole shape of it, and it is a stronger statement than the training role makes.
# Training is denied the *withheld* labels and granted the ones its run owns,
# because it has to learn from something. Scoring has to learn from nothing: it
# loads a checkpoint, decodes JPEGs, and writes down what came back. Ground truth
# enters the cycle one step later, in the job that matches these detections
# against it, and that job is a different identity with a different policy.
#
# The division is worth the extra role. Scoring is the one job in the cycle that
# reads the `val` images -- eval is drawn from that split -- so it is also the one
# whose blast radius includes the frames every cycle is measured on. Giving it
# both those images and their boxes would put the entire evaluation inside one
# identity's reach, and the deny below is what says it is not.
#
# No `VpcConfig` on the jobs that assume it, for `training.tf`'s reason: a
# Processing job outside a VPC reaches S3 over the service's own network, so the
# no-NAT rule and a job that downloads 67,000 images are not in tension.

locals {
  scoring_role_name = "${var.project}-scoring"
  scoring_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.scoring_role_name}"

  # Mirrors of the key builders in `conventions`, as in `training.tf`. Wildcarded
  # at the components a job varies by -- run, cycle, version, seed -- and fixed
  # everywhere else.

  # `raw_image_key` for both splits. The one grant that is wider than training's,
  # and it is wider in pixels rather than in labels. See `raw_images_prefix`.
  scoring_image_objects = "${local.bucket_arns["data"]}/${local.raw_images_prefix}*"

  # `scoring_manifest_key` and `training_code_key`. The manifests naming what to
  # score, and the source archive the container unpacks -- which is the same
  # object the training job ran, so the code that scored a model is the tree that
  # trained it.
  scoring_input_objects = [
    "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/scoring/*",
    "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/training/*",
  ]

  # `model_artifact_key`. Read-only, and narrower than it looks: the training role
  # writes this prefix and this one reads it, so a model is produced by one
  # identity and consumed by another. A scoring job that could write here could
  # replace the checkpoint it was asked to score.
  scoring_model_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/models/*"

  # `detections_prefix`. The only thing this role may write anywhere, and
  # deliberately not the cycle prefix: a scoring job produces detections, so it
  # has no reason to be able to overwrite a gate report, a selection record or a
  # model filed under the same cycle.
  detection_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/detections/*"
}

# Conditioned on the account rather than the job, for `training_trust`'s reason:
# a job's ARN is only known once it has been created, so there is no
# `aws:SourceArn` to pin that a job could satisfy.
data "aws_iam_policy_document" "scoring_trust" {
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

resource "aws_iam_role" "scoring" {
  name               = local.scoring_role_name
  description        = "SageMaker scoring jobs. Reads images and a checkpoint, writes detections, and can read no label anywhere."
  assume_role_policy = data.aws_iam_policy_document.scoring_trust.json
}

data "aws_iam_policy_document" "scoring" {
  # The deny first, because it is the statement this role exists to be read for,
  # and it names every prefix in the account that holds a box rather than only
  # the withheld ones. Three of the four are already refused without it --
  # `WithheldLabelsAreOracleOnly` and `EvalLabelsAreScoringOnly` are allowlists
  # this role is not on, and no allow below addresses `cohort=bootstrap/` or
  # `purchases/` -- so this is not what makes the property true today. It is what
  # keeps it true: an allowlist is a list someone can be added to, an explicit
  # deny beats any allow granted later, and a reader asking whether the scoring
  # job can see ground truth reads this policy before they read `storage.tf`.
  #
  # `EvalLabelsAreScoringOnly` is named for the plane rather than for this role,
  # and the distinction is the design: the scoring *plane* reads those boxes, in
  # the evaluation job that matches detections against them, and this identity is
  # the half of the plane that must not.
  statement {
    sid    = "NeverReadALabel"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [
      local.raw_label_objects,
      local.cohort_label_objects,
      "${local.bucket_arns["data"]}/derived/purchases/*",
    ]
  }

  # A `ManifestFile` channel names its objects, so the images need no listing.
  # The prefix channels are the code archive and the checkpoint, both of which
  # SageMaker enumerates before it copies.
  statement {
    sid       = "ListTheChannelPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "run_id=*/cycle=*/scoring/*",
        "run_id=*/cycle=*/training/*",
        "run_id=*/cycle=*/models/*",
        "run_id=*/cycle=*/detections/*",
      ]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"], local.bucket_arns["artifacts"]]
  }

  statement {
    sid       = "ReadTheImagesItScores"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.scoring_image_objects]
  }

  statement {
    sid       = "ReadItsOwnInputs"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = concat(local.scoring_input_objects, [local.scoring_model_objects])
  }

  # `PutObject` and no delete against a write-once bucket, matching the training
  # role. `GetObject` is not the read-back here that it is there -- SageMaker
  # uploads a Processing output channel itself, so the job never sees the object
  # it produced -- and it is granted anyway because the upload is a multipart PUT
  # the service completes on the role's behalf.
  statement {
    sid    = "WriteItsDetections"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = [local.detection_objects]
  }

  # SageMaker streams the job's stdout here under a stream named for the job. A
  # different group from the training jobs', because the service writes Processing
  # output to its own, and retention is set on it below for the same reason.
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

resource "aws_iam_role_policy" "scoring" {
  name   = "scoring"
  role   = aws_iam_role.scoring.id
  policy = data.aws_iam_policy_document.scoring.json
}

# One group for every scoring job the project ever runs, created here rather than
# left to SageMaker so that retention is set. The numbers worth keeping are in
# here: how long the 67,000-object channel took to download, which is the
# measurement design section 11 leaves open at eight times the training job's
# object count, and how many detections each cohort produced -- a count that
# collapsing toward zero is the quality gate's hard failure arriving early.
resource "aws_cloudwatch_log_group" "scoring" {
  name              = "/aws/sagemaker/ProcessingJobs"
  retention_in_days = 365
}

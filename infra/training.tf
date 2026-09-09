# The training identity: the role a SageMaker training job runs as, and the one
# place the label wall has to hold under a job that genuinely needs to read the
# dataset.
#
# Everything it can read is a label it already owns or an image. The 8,000
# `bootstrap` boxes come from `labels/cohort=bootstrap/` and the bought ones from
# `derived/purchases/`, which is why this role is absent from `label_reader_arns`
# and denied `raw/labels/` in the policy below as well as by the bucket. The
# grant it would otherwise need is `raw/labels/scalabel/train/`, and that prefix
# is the pool: the same read that hands training the 8,000 it owns would hand it
# the 62,000 the oracle exists to sell, and every gate would still pass.
#
# No `VpcConfig` on the jobs that assume it. A training job outside a VPC reaches
# S3 over the service's own network, so the no-NAT rule and a job that has to
# download 8,000 images are not in tension -- the same arrangement ingest runs
# under.

locals {
  training_role_name = "${var.project}-training"
  training_role_arn  = "arn:aws:iam::${var.account_id}:role/${local.training_role_name}"

  # Mirrors of the key builders in `conventions`, one per thing this role reads
  # or writes. Wildcarded at the components a job varies by -- partition version,
  # run, cycle, seed -- and fixed everywhere else, so the statements below say
  # what a training job touches rather than which bucket it is pointed at.
  train_image_objects = "${local.bucket_arns["data"]}/${local.raw_train_images_prefix}*"

  # `cohort_labels_prefix(v, BOOTSTRAP)`. Deliberately not the whole `labels/`
  # subtree: `cohort=eval/` is the second copy of the ground truth every cycle is
  # scored against, and the asymmetry between these two prefixes is the reason
  # the partitioner writes them separately at all.
  bootstrap_label_objects = "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=bootstrap/*"

  # `purchase_labels_prefix(run_id, cycle)`. Every cycle up to this one, which is
  # what makes the cumulative labeled set a key range rather than a set something
  # has to reassemble. Not scoped to one run: a run is a value in a key here, not
  # a role, and narrowing it would mean an IAM role per run.
  purchase_label_objects = "${local.bucket_arns["data"]}/derived/purchases/*"

  # `training_manifest_key` and the source tarball beside it -- the two objects a
  # job is handed, both under the write-once cycle prefix.
  training_input_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/training/*"

  # `model_artifact_key`. The only thing this role may write anywhere, and
  # narrower than the run prefix on purpose: a training job produces model
  # artifacts, so it has no reason to be able to overwrite a gate report or a
  # selection record filed under the same cycle.
  model_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/models/*"

  # `base_weights_key`. The COCO-pretrained YOLO11n every seed of every cycle
  # starts from, staged once and read-only from then on. Outside the run prefixes
  # because it is a function of the recipe rather than of any run.
  base_weights_objects = "${local.bucket_arns["artifacts"]}/base/*"
}

# Conditioned on the account and not on the job, unlike the CodeBuild roles. A
# training job's ARN is only known once the job has been created, so there is no
# `aws:SourceArn` to pin here that a job could satisfy -- what the condition can
# still say is that the SageMaker calling this role is SageMaker in this account,
# which is the cross-account confused-deputy shape the CodeBuild roles use
# `SourceArn` to close.
data "aws_iam_policy_document" "training_trust" {
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

resource "aws_iam_role" "training" {
  name               = local.training_role_name
  description        = "SageMaker training jobs. Reads train images and the labels a run already owns, and no withheld label."
  assume_role_policy = data.aws_iam_policy_document.training_trust.json
}

data "aws_iam_policy_document" "training" {
  # The deny first, because it is the statement this role exists to be read for.
  #
  # Restating the bucket policy rather than relying on it, and both are wanted:
  # `WithheldLabelsAreOracleOnly` is an allowlist, so this role is refused by it
  # on the day it is created and would go on being refused if this statement were
  # deleted -- but an allowlist is a list someone can be added to, and a reader
  # checking whether training can reach a withheld label reads the role's own
  # policy first. An explicit deny also beats any allow granted later by someone
  # who did not read `storage.tf`.
  #
  # `cohort=eval/` has no statement here and needs none: `EvalLabelsAreScoringOnly`
  # denies it to every principal in the account, and the allow below names
  # `cohort=bootstrap/` alone, so no grant this role holds addresses it.
  statement {
    sid    = "NeverReadAWithheldLabel"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [local.raw_label_objects]
  }

  # A `ManifestFile` channel names its objects, so the images need no listing.
  # The three prefixes here are the `S3Prefix` channels, which SageMaker
  # enumerates before it copies. Scoped by prefix rather than granted over the
  # bucket for `ingest`'s reason, and one prefix tighter than the reads below
  # allow: a listing is how a walk of the label tree finds the 62,000 documents
  # the assignments refuse to name, so the prefixes it is granted on are the ones
  # a channel is actually pointed at.
  statement {
    sid       = "ListTheChannelPrefixes"
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
    sid       = "ListTheArtifactChannelPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "base/*",
        "run_id=*/cycle=*/training/*",
        "run_id=*/cycle=*/models/*",
      ]
    }
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"], local.bucket_arns["artifacts"]]
  }

  # `train/` and not `raw/images/100k/`. The `val` images are what `eval` and
  # `reserve` are drawn from, and while an image carries no boxes and leaks no
  # label, a training set that could include one is a training set that could
  # include an eval frame -- which is the failure the manifest channel and this
  # prefix agree on rather than either one holding alone.
  statement {
    sid       = "ReadTrainImages"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.train_image_objects]
  }

  # The whole of what this role may learn about the objects in a picture: the
  # cohort it was given for free, and the batches its run has paid for.
  statement {
    sid       = "ReadTheLabelsThisRunOwns"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.bootstrap_label_objects, local.purchase_label_objects]
  }

  # The image manifest, the source tarball, and the base weights. Read-only on
  # all three: the code a job runs and the checkpoint it starts from are inputs
  # recorded before it starts, and a job that could rewrite either could produce
  # a model whose recorded provenance is not what trained it.
  statement {
    sid       = "ReadItsOwnInputs"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.training_input_objects, local.base_weights_objects]
  }

  # `PutObject` and no delete, against a write-once bucket. `GetObject` is the
  # read-back: the job hashes what it uploaded rather than trusting a call that
  # returned, which is the arrangement the partitioner and the registration both
  # run under.
  statement {
    sid    = "WriteItsModelArtifacts"
    effect = "Allow"

    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = [local.model_objects]
  }

  # SageMaker streams the job's stdout here under a stream named for the job, so
  # the group is the service's and the retention below is what this stack sets on
  # it. No ECR grant beside it: a prebuilt SageMaker image is pulled with the
  # service's own credentials, so a grant for it would permit nothing that
  # happens.
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
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/sagemaker/TrainingJobs",
      "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/sagemaker/TrainingJobs:*",
    ]
  }
}

resource "aws_iam_role_policy" "training" {
  name   = "training"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}

# One group for every training job the project ever runs, created here rather
# than left to SageMaker so that retention is set. Forty seeds a month at a few
# hundred log lines each is nothing to store and something to keep: the channel
# download time that settles the sharding question is a line in here, and so is
# the digest of the base weights each job started from.
resource "aws_cloudwatch_log_group" "training" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 365
}

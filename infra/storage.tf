# The three project buckets. Names, prefixes and the rules below are the
# Terraform half of `edge_ml_flywheel.conventions` -- the Python half builds keys,
# this half creates the containers and enforces the properties the keys assume.
# The two are kept in step by the `<project>-<purpose>-<account_id>` pattern and
# nothing else, so a rename is a change in both files or neither.
#
# Split by immutability rule, writer and lifecycle rather than by tidiness.
# Buckets cost nothing; a lifecycle rule that matches one prefix too many costs
# the data.

locals {
  # `versioned` is the only per-bucket difference uniform enough to table.
  # Telemetry is unversioned because Firehose writes a large number of small
  # objects and every retained version is billed forever, and because a
  # telemetry object is never overwritten -- versioning would protect against a
  # mistake that cannot happen while charging for the privilege.
  buckets = {
    data      = { versioned = true }
    artifacts = { versioned = true }
    telemetry = { versioned = false }
  }

  bucket_names = { for purpose, _ in local.buckets : purpose => "${var.project}-${purpose}-${var.account_id}" }
  bucket_arns  = { for purpose, name in local.bucket_names : purpose => "arn:aws:s3:::${name}" }

  versioned_buckets = { for purpose, cfg in local.buckets : purpose => cfg if cfg.versioned }

  # Mirrors of the prefix constants in `conventions`. Wildcarded one level wider
  # than the Python constants on purpose: `raw/labels/*` rather than
  # `raw/labels/scalabel/*`, so a future `label_source` lands inside the same
  # guarantee instead of outside it by default.
  raw_objects        = "${local.bucket_arns["data"]}/raw/*"
  raw_label_objects  = "${local.bucket_arns["data"]}/raw/labels/*"
  athena_results_arn = "${local.bucket_arns["telemetry"]}/athena-results/*"

  # Wildcarded over `partition_version=` so a re-partition is covered by the
  # statement that already exists rather than by an edit nobody makes. Mirrors
  # `cohort_labels_prefix(version, Cohort.EVAL)` in `conventions`.
  eval_label_objects = "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=eval/*"

  # Empty, and the emptiness is the current state rather than a placeholder: no
  # role that exists today has a reason to read the eval labels. The scoring
  # plane arrives in Phase 3 and adds its ARN here, which is the same one-line
  # diff that would admit it to `label_reader_arns`.
  eval_label_reader_arns = []
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = local.bucket_names[each.key]
}

# ACLs are a legacy access-control path that sits beside bucket policies and can
# grant what a policy never mentions. Disabling them makes the bucket policy the
# only statement about who may read an object, which is what makes the label wall
# below readable as the whole answer.
resource "aws_s3_bucket_ownership_controls" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# SSE-S3 rather than a KMS customer key. S3 encrypts with SSE-S3 by default, so
# this block is here to state the intent and fail a plan if someone changes it --
# not to add protection. A CMK would bill per request against 160,000 ingest PUTs
# and every training read, buying key rotation this project has no requirement
# for.
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = local.versioned_buckets
  bucket   = aws_s3_bucket.this[each.key].id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

# An interrupted multipart upload leaves parts that are billed forever and appear
# in no object listing, so nothing ever reminds you they exist. The 5.7 GB ingest
# is exactly the kind of upload that gets interrupted.
resource "aws_s3_bucket_lifecycle_configuration" "versioned" {
  for_each = local.versioned_buckets
  bucket   = aws_s3_bucket.this[each.key].id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # `raw/` and the artifacts bucket are write-once, so a noncurrent version only
  # exists because something was overwritten or deleted that should not have
  # been. Versioning is here to make that recoverable, not archival: thirty days
  # is long enough to notice and short enough that a botched re-ingest does not
  # keep a second copy of 4.6 GB indefinitely.
  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "telemetry" {
  bucket = aws_s3_bucket.this["telemetry"].id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # Athena writes a result object for every query, including the failed ones, and
  # never cleans up after itself. Pure scratch: the query is in the history and
  # re-running it is free at this data volume.
  rule {
    id     = "expire-athena-results"
    status = "Enabled"

    filter {
      prefix = "athena-results/"
    }

    expiration {
      days = 30
    }
  }

  # Deliberately no expiry on `fleet/`. It is the bucket's real content and the
  # one thing here a retention rule could actually destroy: the champion-over-time
  # and drift charts are read at the end of the project and want the earliest
  # cycles' frames, so any rule short enough to bound cost would delete exactly
  # the data that makes the curve a curve. At parquet volumes this is cents a month.
}

# ---------------------------------------------------------------------------
# Bucket policies
# ---------------------------------------------------------------------------

# Every bucket refuses plaintext HTTP. The same statement the hand-bootstrapped
# state bucket carries, for the same reason: a presigned URL or a stray SDK
# config is the realistic way an object moves in the clear, and neither is
# something a reviewer can spot.
data "aws_iam_policy_document" "tls_only" {
  for_each = local.buckets

  statement {
    sid       = "DenyPlaintextTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [local.bucket_arns[each.key], "${local.bucket_arns[each.key]}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

# The data bucket carries two more denies, and they are the two statements the
# project's whole leakage claim rests on.
#
# Both are written as an allowlist -- `StringNotLike` over a list of principal
# ARNs -- rather than as a deny naming the roles that must not pass. The
# difference decides what happens to a role that does not exist yet: the training
# role arrives in Phase 3, and under an allowlist it is denied on the day it is
# created, with no step in Phase 3 that has to remember. A deny naming the
# training role would be correct on the day it was written and silently
# incomplete for every role added afterwards.
#
# Scoped to object actions only, never bucket actions, which is what keeps this
# recoverable: the apply role can still replace this policy, so a mistake here is
# a Terraform diff rather than a support ticket. That is also the only path back
# to a label object for a human operator, deliberately -- an admin exemption
# would make the boundary a convention, while a Terraform change to remove the
# deny is reviewable, auditable and in git.
data "aws_iam_policy_document" "data_bucket" {
  source_policy_documents = [data.aws_iam_policy_document.tls_only["data"].json]

  # `raw/` is immutable after ingest. Every reprocessing reads from it, which is
  # what makes "confidence scores are regenerable offline" true rather than
  # aspirational, and what lets a re-partition be a new prefix instead of a
  # migration. Overwriting by the ingest role itself is still allowed -- a
  # re-ingest is a legitimate operation, and versioning is what makes it
  # reversible.
  statement {
    sid    = "RawIsWriteOnceExceptIngest"
    effect = "Deny"

    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
    ]

    resources = [local.raw_objects]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "StringNotLike"
      variable = "aws:PrincipalArn"
      values   = local.raw_writer_arns
    }
  }

  # The label wall. The oracle is where a label is *sold*, and the budget is only
  # real if the labels cannot be read any other way -- a training job that can GET
  # these 80,000 JSON files bypasses the oracle, the ledger, and the entire
  # cost-per-label deliverable, while every gate still passes.
  #
  # This allowlist is the outer boundary and not the whole guarantee. Which
  # cohorts the oracle may sell from is a line no policy can draw, since cohort
  # lives in the assignments parquet rather than in any key; that is enforced in
  # `edge_ml_flywheel.oracle.cohorts`. This keeps everyone else out.
  #
  # Denied here at the bucket as well as in each role's own policy. An explicit
  # deny in a bucket policy beats any allow, including one granted later in an
  # inline policy by someone who did not read this file.
  statement {
    sid    = "WithheldLabelsAreOracleOnly"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [local.raw_label_objects]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "StringNotLike"
      variable = "aws:PrincipalArn"
      values   = local.label_reader_arns
    }
  }

  # The second copy of the ground truth. `labels/cohort=eval/` carries the boxes
  # for the 5,000 images every cycle is scored on, outside the prefix the
  # statement above is scoped to.
  #
  # A different failure from the one the label wall catches. Eval labels are
  # never withheld and never charged, so reading one bypasses neither the oracle
  # nor the budget; what it costs is the eval. `ModelRecord` already refuses
  # `eval` in `cohorts_trained_on`, but that is the training job attesting to its
  # own inputs -- the class of guarantee this bucket policy exists to replace.
  #
  # Denied to every principal, and written ahead of the roles it constrains for
  # the reason the statements above are allowlists. Reads only: the file does not
  # exist yet and something has to create it, so freezing it is the separate
  # statement noted at the end of this file.
  statement {
    sid    = "EvalLabelsAreScoringOnly"
    effect = "Deny"

    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]

    resources = [local.eval_label_objects]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    # Absent while the allowlist is empty. An unconditional deny and a
    # `StringNotLike` over no ARNs mean the same thing, but the second is a
    # malformed policy, so the condition appears with its first reader.
    dynamic "condition" {
      for_each = length(local.eval_label_reader_arns) > 0 ? [1] : []

      content {
        test     = "StringNotLike"
        variable = "aws:PrincipalArn"
        values   = local.eval_label_reader_arns
      }
    }
  }
}

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.this["data"].id
  policy = data.aws_iam_policy_document.data_bucket.json

  # A policy that names no public principal is still refused while the public
  # access block is being evaluated, so order the two explicitly rather than
  # relying on Terraform to guess.
  depends_on = [aws_s3_bucket_public_access_block.this]
}

resource "aws_s3_bucket_policy" "other" {
  for_each = { for purpose, cfg in local.buckets : purpose => cfg if purpose != "data" }

  bucket     = aws_s3_bucket.this[each.key].id
  policy     = data.aws_iam_policy_document.tls_only[each.key].json
  depends_on = [aws_s3_bucket_public_access_block.this]
}

# Freezing the eval cohort is a second deny on `labels/cohort=eval/`, on writes
# rather than on the reads `EvalLabelsAreScoringOnly` already refuses. Both are
# expressible only because cohort is a path component rather than a column.
#
# It waits for a reason the read deny does not share: the statement has to name
# the writer, and nothing writes these labels yet. Until then the prefix is
# readable by no one and writable by whatever holds `derived/`.

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
  #
  # `assignments_prefix(v)` joins them for `score_prepare`, which subtracts what
  # a run has bought from the pool the partition drew. Two columns and no box: an
  # assignment says which cohort an image is in, which is the fact the eval wall
  # is built on rather than a thing the wall keeps out.
  control_label_objects = [
    "${local.bucket_arns["data"]}/derived/partition_version=*/labels/cohort=bootstrap/*",
    "${local.bucket_arns["data"]}/derived/partition_version=*/assignments/*",
    "${local.bucket_arns["data"]}/derived/purchases/*",
  ]

  # `scoring_manifest_key`. Written by `score_prepare` and read by the scoring
  # role, the same arrangement the training prefix has one step earlier: one
  # cycle hands its seeds exactly what this wrote.
  control_scoring_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/scoring/*"

  # What `register` reads to build a manifest and what it writes when it has one.
  #
  # The gate report is the verdict, written by the evaluation job; the model
  # prefix carries each seed's `model.sha256` and SageMaker's own `model.tar.gz`,
  # which are the digest the manifest records and the object the registry points
  # at. Both are reads of a prefix this role already lists.
  control_gate_objects  = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/gates/*"
  control_model_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/models/*"

  # `detections_prefix(version, seed, POOL)`, which is what `select` ranks. The
  # one grant in this policy over something a model produced rather than something
  # a cycle was configured with, and it is admissible for the reason the whole
  # selection plane is: a detection is a prediction. Ground truth enters the cycle
  # in the evaluation job, under a role this one is not.
  #
  # The eval cohort's detections are inside the same wildcard and are read by
  # nothing here -- narrowing to `cohort=pool/` would name a path component the
  # key builder puts last, which the prefix above already reaches.
  control_detection_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/detections/*"

  # `model_manifest_key`, and deliberately narrower than the prefix above. The
  # training role writes everything else under `models/`; this role writes the
  # one document in it that is not a model, which is what keeps "the job that
  # produced the artifact did not also write the claims about it" true.
  control_manifest_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/models/version=*/manifest.json"

  # `selection_ranking_key`. Written by `select` and read by the oracle, the same
  # arrangement the training and scoring prefixes have earlier in the cycle: one
  # step writes what the next is handed. This role writes it and the oracle only
  # reads it, which is what keeps the batch a record of how images were chosen
  # rather than something the function that charges for them can author.
  control_selection_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/selection/*"

  # `base_weights_key()`. Written by `launch.ensure_base` on the first cycle of a
  # fresh account and read by nothing here -- the training role is what reads it,
  # as a channel. One object rather than the prefix, so this grant cannot become
  # a way to put a second checkpoint beside the one the recipe names.
  control_base_object = "${local.bucket_arns["artifacts"]}/base/yolo11n.pt"

  # `run_summary_key`. One object per run and the only key in this policy above
  # a cycle prefix, because the document is a statement about the run rather than
  # about one of its turns. Named exactly rather than as `run_id=*/*`, which would
  # reach every cycle prefix under it and make a reporting step able to write
  # anything a cycle wrote.
  control_summary_objects = "${local.bucket_arns["artifacts"]}/run_id=*/summary.json"
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

# `container/`'s three entry points and `requirements.txt`, in a layer.
#
# They are in the deployment for one reason: `prepare` builds the source archive
# the cycle's three jobs unpack, and each of them is named at the root of that
# archive -- `train.py` because script mode requires it there, `score.py` and
# `evaluate.py` because `scoring.job.container_entrypoint` names their paths in a
# shell command. A Lambda has no git checkout to build the tree from, so it has
# to arrive at deploy time -- and a Lambda package has exactly one source
# directory, while these files live outside `src/`. A layer is the one place a
# Lambda can be handed files from a second directory, so they land at `/opt` and
# `control.handler` tars them together with the package from `/var/task`.
data "archive_file" "entry_point" {
  type        = "zip"
  source_dir  = "${path.module}/../container"
  output_path = "${path.module}/build/entrypoint.zip"
}

resource "aws_lambda_layer_version" "entry_point" {
  layer_name          = "${var.project}-entrypoint"
  description         = "The three container entry points and requirements.txt, which belong at the root of the cycle's source archive."
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
        "derived/partition_version=*/assignments/*",
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

  # The manifests and the source archive, both cycles' worth. `PutObject` and no
  # delete against a write-once bucket, and `ListBucket` below because both
  # prepare steps refuse to overwrite a cycle that has already been prepared --
  # that refusal is a listing, and without the grant it would read as "not there
  # yet" and overwrite the record of what a challenger trained on or ranked.
  statement {
    sid       = "WriteTheCycleJobInputs"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.control_training_objects, local.control_scoring_objects]
  }

  # The ranking, which is the one thing this role writes that is not an input to a
  # job about to run. `GetObject` beside it is the refusal to overwrite: `select`
  # checks whether a cycle has already ranked, and a denied listing would read as
  # "not there yet" and rewrite the record a purchase was charged against.
  statement {
    sid       = "WriteTheSelectionRanking"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.control_selection_objects]
  }

  # The COCO base, staged by `prepare` when the bucket has none so that a fresh
  # account needs no setup command. `PutObject` and no delete, against a bucket
  # that is write-once: the first cycle creates it and every later cycle finds it
  # and makes no request at all.
  statement {
    sid       = "StageTheBaseWeightsOnce"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = [local.control_base_object]
  }

  # What `register` reads: the verdict the evaluation job wrote, and the digests
  # and tarball the training job left beside each seed's model. Reads only -- the
  # gate report is evidence about a decision already taken, and a control plane
  # that could rewrite one could change a verdict after the fact.
  statement {
    sid       = "ReadTheVerdictAndTheArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.control_gate_objects, local.control_model_objects]
  }

  # What `select` ranks. Predictions rather than boxes, which is what makes this
  # grant compatible with a role denied every label prefix in the account.
  statement {
    sid       = "ReadThePoolDetections"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.control_detection_objects]
  }

  # The manifest, which is the one object this role writes under `models/`.
  # `GetObject` is the read-back: the step does not report a registered model on
  # the strength of a `put_object` that returned, the same arrangement the
  # partitioner and the training job's upload use.
  statement {
    sid       = "WriteTheModelManifest"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.control_manifest_objects]
  }

  # The run summary, which `summarize` writes once the loop has left. `GetObject`
  # is the read-back, as with the manifest: this document is the deliverable and
  # is not reported as written on the strength of a call that returned. It is the
  # only write this role makes outside a cycle prefix, and the only artifact in
  # the project a later step reads back out of the bucket to check itself.
  statement {
    sid       = "WriteTheRunSummary"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = [local.control_summary_objects]
  }

  statement {
    sid       = "ListTheCyclePrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arns["artifacts"]]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      # `base/` alongside the cycle prefixes, because `ensure_base` decides
      # whether to fetch by listing and a denied listing reads as "not there yet"
      # -- which would re-stage the base on every cycle. `eval/*` is listed and
      # never read: `evaluate_request` checks that the inputs it is about to point
      # a job at exist, which is a listing, and it opens none of the files. That
      # one matters most -- it holds the cached match arrays, so the control plane
      # confirms a champion has them and is in no position to read one.
      #
      # `detections/*` was on the same footing until `select` landed, and is now
      # the one prefix here this role genuinely reads: the pool's boxes are what
      # the uncertainty ranking is computed from. They are predictions rather than
      # ground truth, which is why selection can run in the plane that is denied
      # every label prefix in the account.
      #
      # `models/*` is read for `register`: SageMaker files its `model.tar.gz`
      # under a directory named for the training job, so the key the registry
      # points at is found rather than built.
      #
      # `gates/*` is listed because `register` asks whether the report is there
      # before opening it, and `base.exists` asks that with a listing. The read
      # itself is already granted above -- the verdict is the whole input to the
      # step, since a rejection is recorded with its reason exactly as a
      # promotion is. Missing here, the first cycle to reach a verdict failed on
      # the permission rather than on the verdict, with the report sitting in the
      # bucket.
      values = [
        "run_id=*/cycle=*/training/*",
        "run_id=*/cycle=*/scoring/*",
        "run_id=*/cycle=*/models/*",
        "run_id=*/cycle=*/detections/*",
        "run_id=*/cycle=*/selection/*",
        "run_id=*/cycle=*/eval/*",
        "run_id=*/cycle=*/gates/*",
        "run_id=*/cycle=*/fleet/*",
        "base/*",
      ]
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

  # The fleet round trip: `deploy` publishes the cycle's component version,
  # `fleet_score` deploys it with the task token, and `canary` reads back what
  # the device did with it and rolls a failed rollout back.
  #
  # `CreateComponentVersion` is not resource-scoped because the component being
  # created does not exist to be named yet -- which is the API's shape, not a
  # widening. The deployment calls are the ones that carry a target, and it is
  # the thing group below rather than the account's.
  statement {
    sid    = "PublishTheCycleComponent"
    effect = "Allow"

    actions = [
      "greengrass:CreateComponentVersion",
      "greengrass:ListDeployments",
      "greengrass:GetDeployment",
    ]

    resources = ["*"]
  }

  # `CreateDeployment` authorizes against two resources, not one: the deployment
  # it is about to create, and the target it names. Scoping this to the thing
  # group alone read as the tighter policy and was in fact no policy at all --
  # the call is refused on `deployments:*` before the target is ever considered.
  # The deployment ARN cannot be narrowed, because the ID is minted by the call
  # being authorized; the target is the scope that does the work here, and it is
  # still this project's thing group rather than the account's.
  statement {
    sid     = "DeployToTheFleet"
    effect  = "Allow"
    actions = ["greengrass:CreateDeployment"]

    resources = [
      "arn:aws:greengrass:${var.aws_region}:${var.account_id}:deployments:*",
      "arn:aws:iot:${var.aws_region}:${var.account_id}:thinggroup/${var.project}-devices",
    ]
  }

  # The recipe names the address the device publishes its telemetry to, and
  # `fleet.deploy.iot_endpoint` resolves it here so that the device is handed one
  # rather than discovering its own. That makes the lookup part of building a
  # component version, which is why the grant sits with the Greengrass calls
  # rather than with the telemetry read below.
  #
  # `iot:DescribeEndpoint` takes no resource -- it returns the one ATS address
  # the account has -- so `*` is the API's shape rather than a widening.
  statement {
    sid       = "FindWhereTheDevicePublishes"
    effect    = "Allow"
    actions   = ["iot:DescribeEndpoint"]
    resources = ["*"]
  }

  # `deploy` refuses a cycle whose device is stopped rather than deploying into
  # a two-hour wait that can only time out. A describe and nothing else: this
  # role cannot start the instance, which is deliberate -- a stopped device is an
  # operator's decision about cost, and a control plane that silently started one
  # would spend money nobody asked it to.
  statement {
    sid       = "SeeWhetherTheDeviceIsRunning"
    effect    = "Allow"
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"]
  }

  # The device's own report, which is what the canary gate is computed from.
  # Read-only over the one prefix the IoT rule writes: the verdict is evidence
  # about a rollout, and a plane that could rewrite the evidence could pass a
  # gate after the fact.
  statement {
    sid       = "ReadWhatTheDeviceReported"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [local.bucket_arns["telemetry"], "${local.bucket_arns["telemetry"]}/fleet/*"]
  }

  # The package the component runs, addressed by commit. Two cycles built from
  # one tree write identical bytes to one key, so this is a put with no delete
  # against an object that is content-addressed by construction.
  #
  # The read is not this function's: `CreateComponentVersion` hashes every
  # artifact the recipe names, and it does that under the caller's identity. So
  # the grant that stages the archive is also the grant that lets the component
  # be published from it, and a put alone fails at the publish with the artifact
  # reported as inaccessible rather than as unreadable. The recipe's other two
  # artifacts -- the model and the frame list -- are already readable under the
  # statements that write them, which is why this was the one that showed.
  statement {
    sid       = "StageTheDeviceCode"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${local.bucket_arns["artifacts"]}/fleet/code/*"]
  }

  # The sample the recipe names as an artifact. Written by `score_prepare` and
  # read back by `deploy`, which refuses a deployment whose frames were never
  # drawn rather than letting a device start and find none.
  statement {
    sid       = "WriteAndReadTheFleetSample"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/fleet/*"]
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
  description   = "The steps of a cycle that are Python: writing the training and scoring manifests, and building each job's request."
  role          = aws_iam_role.control.arn
  handler       = "edge_ml_flywheel.control.handler.handler"
  runtime       = local.control_runtime
  architectures = ["x86_64"]

  filename         = data.archive_file.control.output_path
  source_code_hash = data.archive_file.control.output_base64sha256

  # The ceiling is `select`'s now rather than `prepare`'s. That step downloads the
  # pool's detections -- millions of rows across 62,000 images -- filters them and
  # sorts the result, where `prepare` reads a few megabytes of labels. Still a
  # bound on a hang rather than an estimate, and the first real cycle is what
  # measures it.
  timeout = 900

  # Sized for `select` as well. The detections arrive as an Arrow table and leave
  # as one `Detection` per surviving box, and both are live while the file is
  # being read: a pool pass at the scoring job's 0.001 confidence floor is roughly
  # a million rows above `BAND_LOW`. Lambda scales CPU with memory, and the parse
  # loop over those rows is the wall clock here.
  memory_size = 2048

  # The detections are downloaded to `/tmp` before they are read, and the default
  # is 512 MB. The pool's parquet is tens of megabytes compressed, so this is
  # headroom rather than a measured requirement -- but running out of it is a
  # cycle that fails after the expensive job rather than before it.
  ephemeral_storage {
    size = 2048
  }

  # Two layers, for two things the deployment package cannot carry. The managed
  # one supplies pyarrow, which is a platform wheel and so cannot come out of
  # `archive_file` over a source directory; ours carries the two script-mode root
  # files, which live outside `src/`.
  layers = [
    var.pyarrow_layer_arn,
    aws_lambda_layer_version.entry_point.arn,
  ]

  # No `environment` block, and the absence is the fix for a failed apply rather
  # than an omission. The region is what this function would have needed one for
  # -- `job.Target` carries it and boto3 resolves it -- and the Lambda runtime
  # already exports `AWS_REGION` and `AWS_DEFAULT_REGION` itself. Both are
  # reserved keys that `CreateFunction` refuses to have set, so passing the
  # region explicitly is not belt and braces: it is the one way to make this
  # function uncreatable.

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
  description        = "The cycle state machine. Claims a cycle, invokes the control function, and starts the training, scoring and evaluation jobs as their own roles."
  assume_role_policy = data.aws_iam_policy_document.cycle_trust.json
}

data "aws_iam_policy_document" "cycle" {
  # Both functions, because a cycle's Python runs in two identities. The control
  # function does every step that does not read a withheld label, and the oracle
  # does the one that does -- see `oracle.tf`. The state machine is what calls
  # each, so it is the one principal that invokes both.
  statement {
    sid       = "InvokeTheCycleFunctions"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.control.arn, aws_lambda_function.oracle.arn]
  }

  # `UpdateItem` and `GetItem` on one table. No `PutItem` and no `DeleteItem`:
  # the run's control item is created by the registration and advanced by this
  # role, and the difference between "the cap was reached" and "the counter was
  # reset to zero mid-run" is exactly the absence of a put here.
  #
  # Two updates use it and they write different attributes of the same item: the
  # claim advances `next_cycle`, and the promotion advances `champion_version`.
  # One grant rather than two because DynamoDB scopes an action to a table and
  # not to an attribute -- what keeps each write to its own field is the update
  # expression, and each carries a condition that refuses the other's mistake.
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

  # The same four `.sync` needs, for the two Processing jobs a cycle runs --
  # scoring and evaluation. One statement rather than two, because they are the
  # same resource type and SageMaker has no way to name one Processing job and
  # not another before either exists; what tells them apart is the role each is
  # passed, above. A separate statement from the training one, because the
  # resource *is* a different type and collapsing them would mean granting
  # training actions on processing jobs and the reverse -- which is not a wider
  # grant that matters, but is a policy that stops saying which job each action
  # is for.
  statement {
    sid    = "RunProcessingJobs"
    effect = "Allow"

    actions = [
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
    ]

    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:processing-job/*",
    ]
  }

  # The registry. One group per run, opened by the first cycle that reaches the
  # registration, and one version per cycle written into it with the gate's
  # verdict as its approval status.
  #
  # No `UpdateModelPackage` and no delete. An approval status is the verdict a
  # gate reached, and a role that could revise one could approve a model the
  # gates rejected -- which is the single fact this whole plane exists to record
  # faithfully. A promotion changes which version a device is pointed at, not
  # what the registry says happened.
  #
  # `AddTags` because the request carries tags, and SageMaker treats tagging on
  # create as its own action. Wildcarded resources for `RunProcessingJobs`'
  # reason: neither the group nor the version exists before the call that names
  # it, so there is nothing narrower to name.
  statement {
    sid    = "RegisterModelVersions"
    effect = "Allow"

    actions = [
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:AddTags",
    ]

    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:model-package-group/*",
      "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:model-package/*",
    ]
  }

  # The grant that makes the label wall a question about this role. Handing
  # SageMaker one of these roles is how a job gets any data access at all, so the
  # condition pins the service they can be handed to: without it, a principal that
  # could start a job could pass any of them to anything that accepts one.
  #
  # Three roles and not one, and that is the point of there being three. This role
  # can start a job that reads the labels a run owns, a job that reads no label at
  # all, and a job that reads the eval boxes and nothing else -- and which of those
  # a given job is follows from which role it was passed, not from what its
  # container happens to do with the grant.
  statement {
    sid     = "PassTheJobRoles"
    effect  = "Allow"
    actions = ["iam:PassRole"]

    resources = [
      aws_iam_role.training.arn,
      aws_iam_role.scoring.arn,
      aws_iam_role.evaluation.arn,
    ]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # `.sync` is implemented with a managed EventBridge rule that tells Step
  # Functions when the job reaches a terminal state, and the rule is created by
  # this role on first use. One rule per job type, and both names are the
  # service's, not ours.
  statement {
    sid    = "ManageTheSyncCompletionRules"
    effect = "Allow"

    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule",
    ]

    resources = [
      "arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
      "arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule",
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
#
# `terraform validate` does not read the definition and `plan` does not either --
# the JSONata, the `.sync` integration and the shape of every Retry are checked
# by the Step Functions API when it creates the machine, which is to say at
# apply, which is to say on main. Checking it earlier is one read-only call
# against a rendered copy:
#
#   aws stepfunctions validate-state-machine-definition --profile edgeml \
#     --type STANDARD --definition file://<the file with its two ${...} filled in>
resource "aws_sfn_state_machine" "cycle" {
  name     = local.cycle_machine_name
  role_arn = aws_iam_role.cycle.arn
  type     = "STANDARD"

  # The two values the definition cannot know: what the function is called and
  # what the table is called. Everything else in that file is control flow, which
  # is why it is a file rather than a heredoc in here.
  definition = templatefile("${path.module}/cycle.asl.json", {
    control_function_arn = aws_lambda_function.control.arn
    oracle_function_arn  = aws_lambda_function.oracle.arn
    fleet_config_table   = aws_dynamodb_table.fleet_config.name
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.cycle.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
}

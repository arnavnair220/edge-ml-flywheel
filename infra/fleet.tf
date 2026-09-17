# The edge plane: one Graviton device running Greengrass, its identity, and the
# rule that lands what it says in the telemetry bucket.
#
# **This is the first resource in the stack that stays running, and the first
# that sits in a subnet.** Both are departures from rules the rest of the project
# holds to, so both are stated here rather than discovered by a reviewer.
# `ingest.tf` and `training.tf` have no `vpc_config` because a job outside a VPC
# reaches S3 over the service's own network and needs no NAT gateway; a device
# cannot be outside one, because it is a machine rather than a job. So it goes in
# a public subnet with a public IP and a security group admitting nothing, which
# gives it the outbound reach an IoT endpoint needs at the cost of a route table
# and no NAT gateway at all. Cost discipline is intact: a VPC, a subnet, an
# internet gateway and a route table are free, and the only meter running is the
# instance.
#
# **One device.** Design section 7 describes two to five in a thing group, and
# the group here holds one. It is still a group rather than a thing ARN, because
# a second device joins a group without any of the deployment code changing --
# targeting the thing directly would make growing the fleet a change to every
# command in `edge_ml_flywheel.fleet`.
#
# **Terraform owns no component and no deployment.** A component version is a
# function of the model being deployed, so it is minted per cycle by
# `edge_ml_flywheel.fleet.deploy` and lives in no state file. What is here is
# everything with a life longer than a cycle: the device, the identity it
# presents, the role its components borrow, and the pipe its telemetry falls
# down.
#
# **`fleet_config` holds no device item and the design's `desired_version` is
# never written.** The Greengrass deployment is the record of intent (design
# section 6, settled here) -- the service stores it, revises it, and reports what
# each device made of it, so a DynamoDB copy would be one fact written twice by
# two calls with nothing to say which is right when they disagree. See the
# fleet_config section of `edge_ml_flywheel.conventions`.

locals {
  # One name for the device's whole identity: the security group, the IoT policy
  # and the role alias are three views of the same thing, and naming them apart
  # would be three strings to correlate in a console.
  fleet_name = "${var.project}-fleet"

  fleet_role_name          = "${var.project}-fleet"
  telemetry_rule_role_name = "${var.project}-telemetry-rule"

  # The thing group `fleet.__main__.THING_GROUP` composes its target ARN from.
  # Spelled in both places rather than passed, which is the arrangement
  # `storage.tf` describes: the Python half builds the address and this half
  # creates the thing, kept in step by the `<project>-<suffix>` pattern and
  # nothing else.
  thing_group_name = "${var.project}-devices"
  device_name      = "${var.project}-device-1"

  greengrass_root = "/greengrass/v2"
  device_venv     = "/opt/${var.project}/venv"

  # Mirrors of the key builders in `conventions`, wildcarded one level wider than
  # the Python constants for `storage.tf`'s reason.
  #
  # `fleet_model_objects` is every seed's ONNX under every cycle. Wider than the
  # deployed seed because Greengrass downloads whatever a recipe names, and a
  # recipe names one seed's file -- narrowing this to `seed=1` would make a
  # cycle that deployed another seed fail on the device rather than at the
  # `CreateComponentVersion` that named it.
  fleet_model_objects  = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/models/*"
  fleet_frames_objects = "${local.bucket_arns["artifacts"]}/run_id=*/cycle=*/fleet/*"

  # `replay_code_key`. The commit is a directory below this, so the wildcard
  # covers every tree the device has ever been asked to run.
  fleet_code_objects = "${local.bucket_arns["artifacts"]}/fleet/code/*"

  # `raw_image_key(image_id, COHORT_SPLIT[Cohort.POOL])`. The train split alone,
  # which is where the pool lives -- the `val` images are what `eval` is drawn
  # from, and a device has no business reaching one.
  fleet_image_objects = "${local.bucket_arns["data"]}/${local.raw_train_images_prefix}*"

  # `telemetry_prefix`, one level wider. The IoT rule writes here; nothing on the
  # device does, which is the point of routing through IoT Core at all.
  fleet_telemetry_objects = "${local.bucket_arns["telemetry"]}/fleet/*"
}

# The two endpoints a nucleus config names. Data is where a component publishes
# and credentials is where the certificate is exchanged for a session -- two
# different hostnames doing two different jobs, and a config naming one of them
# twice is a device that installs and can never authenticate.
data "aws_iot_endpoint" "data" {
  endpoint_type = "iot:Data-ATS"
}

data "aws_iot_endpoint" "credentials" {
  endpoint_type = "iot:CredentialProvider"
}

# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

resource "aws_vpc" "fleet" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = local.fleet_name }
}

resource "aws_internet_gateway" "fleet" {
  vpc_id = aws_vpc.fleet.id

  tags = { Name = local.fleet_name }
}

# Public, and the public address is the whole point: it is what replaces a NAT
# gateway. A device in a private subnet would need one to reach IoT Core, S3 and
# the package repositories, at roughly $32/month against a project budget of $40.
resource "aws_subnet" "fleet" {
  vpc_id                  = aws_vpc.fleet.id
  cidr_block              = "10.0.0.0/24"
  map_public_ip_on_launch = true

  tags = { Name = local.fleet_name }
}

resource "aws_route_table" "fleet" {
  vpc_id = aws_vpc.fleet.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.fleet.id
  }

  tags = { Name = local.fleet_name }
}

resource "aws_route_table_association" "fleet" {
  subnet_id      = aws_subnet.fleet.id
  route_table_id = aws_route_table.fleet.id
}

# No `ingress` block at all, which is a stronger statement than an empty one: a
# security group with no inbound rule admits nothing, and there is nothing to
# admit. Greengrass connects outward to IoT Core and holds the connection open,
# so a deployment reaches this device without anything reaching in. The operator
# gets in through Session Manager, which is also outbound -- see the instance
# profile below.
resource "aws_security_group" "fleet" {
  name        = local.fleet_name
  description = "Greengrass device. Nothing inbound; outbound to IoT Core, S3 and package repositories."
  vpc_id      = aws_vpc.fleet.id

  egress {
    description = "Everything outbound. The device pulls its own artifacts and pushes its own telemetry."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = local.fleet_name }
}

# ---------------------------------------------------------------------------
# The device's identity
# ---------------------------------------------------------------------------

# Created here rather than by fleet provisioning, which is the textbook answer
# and is deliberately skipped -- `github-oidc.tf`'s Identity Center argument
# applied to devices. A provisioning template is for hardware that enrolls itself
# at a scale nobody wants to pre-register, and this is one instance Terraform
# already creates. What it would buy is the thing this stack most wants to avoid:
# an IoT identity outside state that a destroy cannot remove.
resource "aws_iot_thing" "device" {
  name = local.device_name
}

resource "aws_iot_thing_group" "devices" {
  name = local.thing_group_name
}

resource "aws_iot_thing_group_membership" "device" {
  thing_name       = aws_iot_thing.device.name
  thing_group_name = aws_iot_thing_group.devices.name
}

# The private key is returned by this call and is therefore in Terraform state,
# which is the accepted cost of Terraform creating the certificate at all. State
# lives in the bootstrapped bucket with versioning, SSE and a TLS-only policy,
# and is readable by the two OIDC roles and nothing else. The alternative is a
# key generated on the device, which needs the device to be able to register
# itself -- an instance profile that may create IoT identities, which is a
# larger grant than this is a risk.
resource "aws_iot_certificate" "device" {
  active = true
}

resource "aws_iot_thing_principal_attachment" "device" {
  thing     = aws_iot_thing.device.name
  principal = aws_iot_certificate.device.arn
}

# What the certificate itself may do, which is a narrower question than what the
# token exchange role below may do. This policy covers the MQTT connection the
# nucleus holds open and the credential exchange; everything a component does
# with AWS goes through the role.
#
# `iot:Connect` is scoped to a client ID equal to the thing name. Without that
# condition any holder of this certificate could connect as any client and
# silently disconnect the real device -- IoT Core allows one connection per
# client ID, so the two would take turns knocking each other offline.
data "aws_iam_policy_document" "device_certificate" {
  statement {
    sid     = "ConnectAsItself"
    effect  = "Allow"
    actions = ["iot:Connect"]

    # The bare thing name and a suffixed form of it. The nucleus connects as the
    # thing, and its own subsystems take client IDs of `<thing>-<suffix>` -- so
    # the bare resource alone is a device that installs, authenticates, and then
    # cannot open the connection a deployment arrives over. Still bounded to this
    # device's name, which is the property that matters: IoT Core allows one
    # connection per client ID, so a certificate free to connect as any client
    # could knock the real device offline and take turns with it.
    resources = [
      "arn:aws:iot:${var.aws_region}:${var.account_id}:client/$${iot:Connection.Thing.ThingName}",
      "arn:aws:iot:${var.aws_region}:${var.account_id}:client/$${iot:Connection.Thing.ThingName}-*",
    ]
  }

  # The topic a replay publishes to, and only below this run-and-device prefix.
  # Mirrors `telemetry_topic(run_id, thing)`: the wildcard is the run, and the
  # device names itself in the last segment through the same policy variable the
  # connection is scoped by -- so this certificate cannot publish under another
  # device's name.
  statement {
    sid       = "PublishItsOwnTelemetry"
    effect    = "Allow"
    actions   = ["iot:Publish"]
    resources = ["arn:aws:iot:${var.aws_region}:${var.account_id}:topic/${var.project}/fleet/*/$${iot:Connection.Thing.ThingName}"]
  }

  # How a component gets AWS credentials at all: the nucleus presents this
  # certificate to the credentials endpoint and receives a session for the role
  # the alias names. There is no long-lived key on the device.
  statement {
    sid       = "ExchangeItsCertificateForASession"
    effect    = "Allow"
    actions   = ["iot:AssumeRoleWithCertificate"]
    resources = [aws_iot_role_alias.device.arn]
  }

  # The nucleus reports deployment status and component state over these, which
  # is what makes a failed install visible in the console rather than only in a
  # log on the device. Greengrass's own topics, so the resource is the service's
  # namespace rather than this project's.
  statement {
    sid    = "ReportWhatItMadeOfADeployment"
    effect = "Allow"

    actions = [
      "iot:Publish",
      "iot:Subscribe",
      "iot:Receive",
    ]

    resources = [
      "arn:aws:iot:${var.aws_region}:${var.account_id}:topic/$aws/things/$${iot:Connection.Thing.ThingName}/*",
      "arn:aws:iot:${var.aws_region}:${var.account_id}:topicfilter/$aws/things/$${iot:Connection.Thing.ThingName}/*",
      "arn:aws:iot:${var.aws_region}:${var.account_id}:topic/$aws/greengrass/*",
      "arn:aws:iot:${var.aws_region}:${var.account_id}:topicfilter/$aws/greengrass/*",
    ]
  }

  # The four data-plane calls a core device makes to find and fetch a deployment:
  # which groups it is in, what the deployment says, which component versions
  # satisfy it, and where the artifacts are. Read-only, and `*` because these
  # APIs take no resource narrower than the account.
  #
  # `VerifyClientDeviceIdentity` and `PutCertificateAuthorities` are deliberately
  # absent. They are for client-device auth -- other things connecting *through*
  # this device -- which this fleet does not do, and the usual permissive
  # `greengrass:*` would grant them along with everything else.
  statement {
    sid    = "FindTheDeploymentsMeantForIt"
    effect = "Allow"

    actions = [
      "greengrass:ListThingGroupsForCoreDevice",
      "greengrass:GetDeploymentConfiguration",
      "greengrass:ResolveComponentCandidates",
      "greengrass:GetComponentVersionArtifact",
    ]

    resources = ["*"]
  }
}

resource "aws_iot_policy" "device" {
  name   = local.fleet_name
  policy = data.aws_iam_policy_document.device_certificate.json
}

resource "aws_iot_policy_attachment" "device" {
  policy = aws_iot_policy.device.name
  target = aws_iot_certificate.device.arn
}

resource "aws_iot_role_alias" "device" {
  alias    = local.fleet_name
  role_arn = aws_iam_role.fleet.arn
}

# ---------------------------------------------------------------------------
# The token exchange role: what a component may reach
# ---------------------------------------------------------------------------

# No `aws:SourceArn`, unlike the CodeBuild and SageMaker trusts: the credentials
# provider does not supply one. `aws:SourceAccount` is what this can be
# conditioned on, and the certificate and role alias above are what actually
# bound who may assume it.
data "aws_iam_policy_document" "fleet_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["credentials.iot.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
  }
}

resource "aws_iam_role" "fleet" {
  name               = local.fleet_role_name
  description        = "Greengrass components on the device. Reads pool images, its own artifacts and nothing else; can read no label anywhere."
  assume_role_policy = data.aws_iam_policy_document.fleet_trust.json
}

data "aws_iam_policy_document" "fleet" {
  # The deny first, because it is the statement this role exists to be read for.
  #
  # A device is inside the label wall exactly as a training job is (design
  # section 7). It replays the unlabeled pool, and the whole claim of the project
  # is that a label can only be obtained by paying the oracle -- a device that
  # could GET these files would bypass the ledger while every gate still passed.
  #
  # Restating the bucket policy rather than relying on it, for `training.tf`'s
  # reason: `WithheldLabelsAreOracleOnly` is an allowlist, so this role is
  # refused by it on the day it is created and would go on being refused if this
  # statement were deleted. But an allowlist is a list someone can be added to,
  # and a reader asking whether a device can reach a label reads the device's own
  # policy first.
  statement {
    sid       = "NeverReadALabel"
    effect    = "Deny"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = [local.raw_label_objects, local.cohort_label_objects]
  }

  # The frames it replays, and only the train split. `eval` is drawn from `val`,
  # so a device that could read that prefix would be one whose blast radius
  # covers the images every cycle is measured on. Which frames inside the split
  # it actually reads is the list shipped as a component artifact, not this.
  statement {
    sid       = "ReadTheFramesItReplays"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [local.fleet_image_objects]
  }

  # The three component artifacts Greengrass downloads on its behalf: the int8
  # graph, the sampled frame list, and the package that runs them. Read-only and
  # narrower than it looks -- the training job writes the model, the deploy step
  # writes the other two, and this identity writes none of them. A device that
  # could write here could replace the artifact whose digest the canary checks.
  statement {
    sid    = "ReadItsOwnArtifacts"
    effect = "Allow"

    actions = ["s3:GetObject"]

    resources = [
      local.fleet_model_objects,
      local.fleet_frames_objects,
      local.fleet_code_objects,
    ]
  }

  statement {
    sid       = "ResolveBucketRegion"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.bucket_arns["data"], local.bucket_arns["artifacts"]]
  }

  # What a replay actually reports. Scoped to this project's topic root and to
  # this device's own name inside it, the same two bounds the certificate policy
  # applies -- stated twice because a component's credentials and the nucleus's
  # MQTT connection are two different paths to the same topic.
  statement {
    sid       = "PublishItsOwnTelemetry"
    effect    = "Allow"
    actions   = ["iot:Publish"]
    resources = ["arn:aws:iot:${var.aws_region}:${var.account_id}:topic/${var.project}/fleet/*/${local.device_name}"]
  }

  # The nucleus's own logs, which is where a deployment that failed on the device
  # is diagnosed from. Created by Greengrass rather than declared here, unlike
  # every other group in this stack, because the group name carries the component
  # name and a component is minted per cycle.
  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]

    resources = ["arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/greengrass/*"]
  }
}

resource "aws_iam_role_policy" "fleet" {
  name   = "fleet"
  role   = aws_iam_role.fleet.id
  policy = data.aws_iam_policy_document.fleet.json
}

# ---------------------------------------------------------------------------
# Telemetry to S3
# ---------------------------------------------------------------------------

# **No Firehose, and the absence is a decision rather than a gap.** Design
# section 7 routes detections through Kinesis Firehose, which converts them to
# parquet and lands them partitioned. That is the right shape at a volume this
# fleet does not have: one device publishes a handful of batched messages per
# cycle, so Firehose would buffer for five minutes to write one small object,
# cost a delivery stream to keep alive, and need a Glue table to convert
# against. The rule writes the object directly, the reader is pandas, and the
# landing layout is the one `telemetry_prefix` already describes -- so the
# Firehose is a later insert under the same prefix rather than a rewrite.
#
# The key template is the contract `conventions.telemetry_topic` is written
# against. `topic(3)` is the run and `topic(4)` is the device, which is why that
# function's segments are fixed and tested. `newuuid()` rather than a timestamp
# alone: two messages in one millisecond would otherwise be one object, and the
# loss would look exactly like a batch that never arrived.
resource "aws_iot_topic_rule" "telemetry" {
  name        = replace("${var.project}_telemetry", "-", "_")
  description = "Lands replay telemetry under fleet/run_id=<id>/dt=<date>/ in the telemetry bucket."
  enabled     = true

  # `SELECT *` because a record is already exactly what a reader wants -- the
  # device composes the document and this moves it. A SQL projection here would
  # be a second definition of the format, in a language the test suite cannot
  # reach.
  sql         = "SELECT * FROM '${var.project}/fleet/+/+'"
  sql_version = "2016-03-23"

  s3 {
    role_arn    = aws_iam_role.telemetry_rule.arn
    bucket_name = aws_s3_bucket.this["telemetry"].bucket
    key         = "fleet/run_id=$${topic(3)}/dt=$${parse_time(\"yyyy-MM-dd\", timestamp())}/$${topic(4)}-$${newuuid()}.json"
  }

  # A rule that cannot write its destination fails silently otherwise: the
  # message is accepted, the object never appears, and the canary reports a
  # replay that lost every batch. This turns that into a log line.
  error_action {
    cloudwatch_logs {
      role_arn       = aws_iam_role.telemetry_rule.arn
      log_group_name = aws_cloudwatch_log_group.telemetry.name
    }
  }
}

# 365 days, matching the groups that narrate a run rather than the ones that
# narrate a build: a rule error is evidence about a cycle's telemetry, and the
# cycle it belongs to may be read months later.
resource "aws_cloudwatch_log_group" "telemetry" {
  name              = "/aws/iot/${var.project}-telemetry"
  retention_in_days = 365
}

data "aws_iam_policy_document" "telemetry_rule_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["iot.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
  }
}

data "aws_iam_policy_document" "telemetry_rule" {
  # Write-only, and no read. The rule's job is to land an object; a role that
  # could also read the prefix would be a second identity holding the fleet's
  # whole history for no operation it performs.
  statement {
    sid       = "LandOneTelemetryObject"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = [local.fleet_telemetry_objects]
  }

  statement {
    sid    = "ReportItsOwnFailures"
    effect = "Allow"

    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]

    resources = ["${aws_cloudwatch_log_group.telemetry.arn}:*"]
  }
}

resource "aws_iam_role" "telemetry_rule" {
  name               = local.telemetry_rule_role_name
  description        = "The IoT rule that lands replay telemetry. Writes one prefix and reads nothing."
  assume_role_policy = data.aws_iam_policy_document.telemetry_rule_trust.json
}

resource "aws_iam_role_policy" "telemetry_rule" {
  name   = "telemetry-rule"
  role   = aws_iam_role.telemetry_rule.id
  policy = data.aws_iam_policy_document.telemetry_rule.json
}

# ---------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------

# The certificate reaches the device through Parameter Store rather than through
# user data. User data is readable by anything on the instance through IMDS and
# by anyone holding `ec2:DescribeInstanceAttribute`, so a private key there would
# be a private key in two places, neither of them a secret store.
#
# `SecureString` uses the account's AWS-managed SSM key, which is free and
# rotated by AWS. A customer key would bill per request for a parameter read once
# per boot.
resource "aws_ssm_parameter" "device_certificate" {
  name        = "/${var.project}/fleet/${local.device_name}/certificate"
  description = "The device's IoT certificate, fetched once at first boot."
  type        = "String"
  value       = aws_iot_certificate.device.certificate_pem
}

resource "aws_ssm_parameter" "device_private_key" {
  name        = "/${var.project}/fleet/${local.device_name}/private-key"
  description = "The device's IoT private key, fetched once at first boot."
  type        = "SecureString"
  value       = aws_iot_certificate.device.private_key
}

# Amazon Linux 2023 on arm64, resolved from the public SSM parameter AWS keeps
# current. A hardcoded AMI ID would pin this to one region and one build, and
# would go stale silently -- a launch against a deregistered image fails at apply
# with a message about an AMI rather than about anything this project did.
data "aws_ssm_parameter" "al2023_arm64" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

data "aws_iam_policy_document" "device_instance_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "device_instance" {
  # The two parameters this device's own boot reads, named individually. A
  # wildcard over `/<project>/fleet/*` would let one device read the next one's
  # key, which is the whole reason each has its own certificate.
  statement {
    sid    = "ReadItsOwnIdentityOnce"
    effect = "Allow"

    actions = ["ssm:GetParameter"]

    resources = [
      aws_ssm_parameter.device_certificate.arn,
      aws_ssm_parameter.device_private_key.arn,
    ]
  }
}

resource "aws_iam_role" "device_instance" {
  name               = "${var.project}-device"
  description        = "The EC2 instance itself, as distinct from the components on it. Reads its own certificate and nothing else."
  assume_role_policy = data.aws_iam_policy_document.device_instance_trust.json
}

resource "aws_iam_role_policy" "device_instance" {
  name   = "device"
  role   = aws_iam_role.device_instance.id
  policy = data.aws_iam_policy_document.device_instance.json
}

# A managed policy, unlike everything else in this stack, and the exception is
# worth naming: this is what gives Session Manager its outbound agent channel,
# which is the only way onto a device whose security group admits nothing.
# Hand-writing the equivalent means tracking a list of `ssmmessages` and
# `ec2messages` actions that AWS revises, in exchange for a boundary Session
# Manager already draws.
resource "aws_iam_role_policy_attachment" "device_instance_ssm" {
  role       = aws_iam_role.device_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "device" {
  name = "${var.project}-device"
  role = aws_iam_role.device_instance.name
}

resource "aws_instance" "device" {
  ami                    = data.aws_ssm_parameter.al2023_arm64.value
  instance_type          = var.device_instance_type
  subnet_id              = aws_subnet.fleet.id
  vpc_security_group_ids = [aws_security_group.fleet.id]
  iam_instance_profile   = aws_iam_instance_profile.device.name

  # IMDSv2 required. The instance profile below is small, but a device on a
  # public IP running a component that fetches URLs is exactly the shape that
  # turns a request-forgery bug into stolen credentials, and IMDSv1 is what makes
  # that a single GET.
  metadata_options {
    http_tokens                 = "required"
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    # The nucleus, a JRE, onnxruntime's wheels and a cycle's artifacts. AL2023's
    # default is 8 GB, which the JRE and the wheels alone make tight.
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }

  # **A change to the boot script replaces the instance.** Left at the default,
  # an edited `user_data` is written to the instance attribute and never run
  # again -- the script runs once, at first boot -- so a plan would show a change,
  # an apply would report success, and the device would go on running what it
  # was built with. That is the worst of the three possible behaviours: the
  # config and the machine disagree and nothing says so. Replacement is honest,
  # and it costs a few minutes on a device that is stopped between cycles anyway.
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/device.sh", {
    thing_name            = aws_iot_thing.device.name
    region                = var.aws_region
    role_alias            = aws_iot_role_alias.device.alias
    data_endpoint         = data.aws_iot_endpoint.data.endpoint_address
    credentials_endpoint  = data.aws_iot_endpoint.credentials.endpoint_address
    certificate_parameter = aws_ssm_parameter.device_certificate.name
    private_key_parameter = aws_ssm_parameter.device_private_key.name
    greengrass_root       = local.greengrass_root
    venv                  = local.device_venv
    nucleus_url           = var.greengrass_nucleus_url
    nucleus_version       = var.greengrass_nucleus_version
    requirements          = file("${path.module}/device-requirements.txt")
  })

  # The certificate has to be attached to the thing and the policy attached to
  # the certificate before a nucleus using them can authenticate. Terraform sees
  # no dependency, because the instance references the parameters rather than the
  # attachments -- so a first boot would race the identity into existence and
  # fail in a way that only a log on the device explains.
  depends_on = [
    aws_iot_policy_attachment.device,
    aws_iot_thing_principal_attachment.device,
    aws_route_table_association.fleet,
  ]

  tags = { Name = local.device_name }
}

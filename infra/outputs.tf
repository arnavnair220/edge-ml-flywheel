output "oidc_provider_arn" {
  description = "GitHub Actions OIDC identity provider."
  value       = aws_iam_openid_connect_provider.github.arn
}

output "plan_role_arn" {
  description = "Role assumed by pull requests. Set as AWS_PLAN_ROLE_ARN in CI."
  value       = aws_iam_role.plan.arn
}

output "apply_role_arn" {
  description = "Role assumed by pushes to main. Set as AWS_APPLY_ROLE_ARN in CI."
  value       = aws_iam_role.apply.arn
}

# Names, not ARNs, because every consumer is an SDK call that takes a name. They
# are outputs rather than something a script derives so that `terraform output`
# is the one place to look, and a caller that reads them cannot disagree with the
# stack about what exists.
output "buckets" {
  description = "The three project buckets, keyed by purpose. Mirrors conventions.Buckets."
  value       = local.bucket_names
}

output "tables" {
  description = "The five DynamoDB tables, keyed by the conventions.Table member."
  value = {
    runs         = aws_dynamodb_table.runs.name
    label_budget = aws_dynamodb_table.label_budget.name
    fleet_config = aws_dynamodb_table.fleet_config.name
    audit_log    = aws_dynamodb_table.audit_log.name
    run_locks    = aws_dynamodb_table.run_locks.name
  }
}

output "ingest_role_arn" {
  description = "The only principal permitted to write raw/ or read a withheld label."
  value       = aws_iam_role.ingest.arn
}

output "ingest_project_name" {
  description = "Start a run with: aws codebuild start-build --project-name <this>."
  value       = aws_codebuild_project.ingest.name
}

output "partition_role_arn" {
  description = "Cohort assignment. Reads derived/manifest/, writes one partition version, no raw/."
  value       = aws_iam_role.partition.arn
}

output "partition_project_name" {
  description = "Start a draw with: aws codebuild start-build --project-name <this>."
  value       = aws_codebuild_project.partition.name
}

output "register_role_arn" {
  description = "Run registration. PutItem and GetItem on the runs table, and no S3 at all."
  value       = aws_iam_role.register.arn
}

output "register_project_name" {
  description = "Start a run with: aws codebuild start-build --project-name <this>, RUN_SLUG and RUN_NOTE overridden."
  value       = aws_codebuild_project.register.name
}

output "training_role_arn" {
  description = "SageMaker training jobs run as this. Pass it as RoleArn; it is denied every withheld label."
  value       = aws_iam_role.training.arn
}

output "scoring_role_arn" {
  description = "SageMaker scoring jobs run as this. It reads images and a checkpoint, and can read no label anywhere."
  value       = aws_iam_role.scoring.arn
}

output "evaluation_role_arn" {
  description = "SageMaker evaluation jobs run as this. The one principal in the account admitted to labels/cohort=eval/."
  value       = aws_iam_role.evaluation.arn
}

# An ARN rather than a name, unlike the buckets and tables: the one consumer is
# `aws stepfunctions start-execution --state-machine-arn`, which takes an ARN.
output "cycle_state_machine_arn" {
  description = "Run a cycle with: aws stepfunctions start-execution --state-machine-arn <this> --input '{\"run_id\":\"...\",\"epochs\":1,\"seeds\":[1],\"max_images\":300}'."
  value       = aws_sfn_state_machine.cycle.arn
}

output "control_function_name" {
  description = "The control plane's Lambda. Writes a cycle's job inputs and builds the training, scoring and evaluation requests."
  value       = aws_lambda_function.control.function_name
}

# PENDING until the GitHub App authorization is completed by hand in the
# console. An output rather than a note in a doc, because it is the one piece of
# this stack that applying cannot finish, and a build against a pending
# connection fails before it produces a log line to diagnose.
output "ingest_connection_status" {
  description = "AVAILABLE once the GitHub connection is authorized in the console. PENDING before."
  value       = aws_codeconnections_connection.github.connection_status
}

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
  description = "The six DynamoDB tables, keyed by the conventions.Table member."
  value = {
    runs          = aws_dynamodb_table.runs.name
    oracle_labels = aws_dynamodb_table.oracle_labels.name
    label_budget  = aws_dynamodb_table.label_budget.name
    fleet_config  = aws_dynamodb_table.fleet_config.name
    audit_log     = aws_dynamodb_table.audit_log.name
    run_locks     = aws_dynamodb_table.run_locks.name
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

# PENDING until the GitHub App authorization is completed by hand in the
# console. An output rather than a note in a doc, because it is the one piece of
# this stack that applying cannot finish, and a build against a pending
# connection fails before it produces a log line to diagnose.
output "ingest_connection_status" {
  description = "AVAILABLE once the GitHub connection is authorized in the console. PENDING before."
  value       = aws_codeconnections_connection.github.connection_status
}

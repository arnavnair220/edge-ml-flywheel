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

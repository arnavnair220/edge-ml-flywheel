# The EKS cluster and its network. See docs/cluster.md for the contract it meets.

data "aws_availability_zones" "available" {
  state = "available"

  # use1-az3 cannot host an EKS control plane. Zone IDs are the same in every
  # account; zone names are not.
  exclude_zone_ids = ["use1-az3"]

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# The data bucket belongs to infra/. It is read, never managed, so destroying
# this stack cannot touch it.
data "aws_s3_bucket" "data" {
  bucket = "${var.project}-data-${var.account_id}"
}

locals {
  vpc_cidr = "10.42.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)

  # Karpenter finds the gpu nodes' subnets and security group by this tag.
  discovery_tags = { "karpenter.sh/discovery" = var.cluster_name }
}

# Public subnets only. Nodes get a public address and reach GHCR directly, so
# there is no NAT gateway. Inbound traffic is limited by the node security group
# below; the default security group is left with no rules.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.7.3"

  name = var.cluster_name
  cidr = local.vpc_cidr

  azs            = local.azs
  public_subnets = [for i, _ in local.azs : cidrsubnet(local.vpc_cidr, 4, i)]

  map_public_ip_on_launch = true
  enable_nat_gateway      = false

  public_subnet_tags = local.discovery_tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.29.0"

  name               = var.cluster_name
  kubernetes_version = var.kubernetes_version

  endpoint_public_access       = true
  endpoint_public_access_cidrs = var.endpoint_public_access_cidrs

  # The identity that runs apply administers the cluster, so Terraform can
  # install the Helm releases in addons.tf.
  enable_cluster_creator_admin_permissions = true

  # EKS encrypts secrets with an AWS-owned key by default. A customer key would
  # be left pending deletion after every teardown.
  encryption_config = null
  create_kms_key    = false

  # No control-plane logs, so teardown leaves no log group behind.
  enabled_log_types           = []
  create_cloudwatch_log_group = false

  addons = {
    coredns = {}
    eks-pod-identity-agent = {
      before_compute = true
    }
    kube-proxy = {}
    vpc-cni = {
      before_compute = true
    }
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnets

  eks_managed_node_groups = {
    services = {
      instance_types = var.services_instance_types

      min_size     = var.services_node_count
      max_size     = var.services_node_count
      desired_size = var.services_node_count

      labels = { pool = "services" }
    }
  }

  # The node security group admits only the control plane and other nodes. Only
  # this group carries the discovery tag, so Karpenter's nodes get the same rules.
  node_security_group_tags = local.discovery_tags
}

# Karpenter's controller role, the gpu nodes' role and access entry, and the
# interruption queue. The controller authenticates through EKS Pod Identity.
module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "21.29.0"

  cluster_name = module.eks.cluster_name

  # The EC2NodeClass names this role, so it has a fixed name.
  node_iam_role_use_name_prefix   = false
  node_iam_role_name              = "${var.cluster_name}-karpenter-node"
  create_pod_identity_association = true

  # The controller policy is longer than a managed policy's 6,144-character
  # limit, so it is attached inline, where the limit is 10,240.
  enable_inline_policy = true
}

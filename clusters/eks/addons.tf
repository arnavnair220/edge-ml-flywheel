# In-cluster components the cluster contract needs: Karpenter, its gpu NodePool,
# and the NVIDIA device plugin. Each depends on the whole eks module, so on
# destroy it is removed while the services nodes are still running.

# Pulled anonymously from ECR Public. A registry login would store a credential
# through the host's Docker credential helper, which fails on some Windows hosts.
resource "helm_release" "karpenter" {
  name       = "karpenter"
  namespace  = "kube-system"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = var.karpenter_version

  values = [yamlencode({
    # One replica: the controller only acts when a gpu job is pending.
    replicas     = 1
    nodeSelector = { pool = "services" }
    settings = {
      clusterName       = module.eks.cluster_name
      clusterEndpoint   = module.eks.cluster_endpoint
      interruptionQueue = module.karpenter.queue_name
    }
  })]

  depends_on = [module.eks]
}

# The EC2NodeClass and NodePool for the gpu pool, from charts/gpu-pool. A chart
# rather than raw manifests, so Terraform can pass the variables in and the CRDs
# need not exist at plan time.
resource "helm_release" "gpu_pool" {
  name      = "gpu-pool"
  namespace = "kube-system"
  chart     = "${path.module}/charts/gpu-pool"

  values = [yamlencode({
    clusterName    = module.eks.cluster_name
    nodeRole       = module.karpenter.node_iam_role_name
    instanceTypes  = var.gpu_instance_types
    maxNodes       = var.gpu_max_nodes
    rootVolumeSize = "${var.gpu_root_volume_gib}Gi"
    tags = {
      Project   = var.project
      ManagedBy = "karpenter"
      Repo      = var.repo
      Stack     = "eks"
    }
  })]

  depends_on = [helm_release.karpenter]
}

resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  namespace  = "kube-system"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = var.nvidia_device_plugin_version

  values = [yamlencode({
    nodeSelector = { pool = "gpu" }
    # The chart's default affinity requires labels from Node Feature Discovery,
    # which this cluster does not run. The node selector replaces it.
    affinity = null
    # Idle rather than crash on a gpu-pool node with no GPU, as when the pool is
    # set to a CPU instance type for the cluster check.
    failOnInitError = false
  })]

  depends_on = [module.eks]
}

# account_id has no default: it is deployment identity and must be supplied, so a
# clone fails loudly instead of pointing at someone else's account. Copy
# terraform.tfvars.example to terraform.tfvars and fill it in.

variable "account_id" {
  description = "The dedicated project account. Not a personal or management account."
  type        = string
}

variable "project" {
  description = "Project name, used as a tag and in the data bucket's name."
  type        = string
  default     = "edge-ml-flywheel"
}

variable "repo" {
  description = "Repository that owns this stack, recorded as a tag."
  type        = string
  default     = "arnavnair220/edge-ml-flywheel"
}

variable "aws_region" {
  description = "Single region for the whole project. An SCP denies everything else."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name. The Makefile sets it from EKS_CLUSTER."
  type        = string
  default     = "edge-ml-flywheel"
}

# Karpenter supports a Kubernetes minor only from a given release onward. Check
# the compatibility matrix at https://karpenter.sh/docs/upgrading/compatibility/
# before raising this, and raise karpenter_version with it.
variable "kubernetes_version" {
  description = "EKS Kubernetes minor version."
  type        = string
  default     = "1.36"
}

variable "endpoint_public_access_cidrs" {
  description = "CIDRs allowed to reach the public Kubernetes API endpoint. Access is still IAM-authenticated."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "services_instance_types" {
  description = "Instance types for the services node group."
  type        = list(string)
  default     = ["t3.medium"]
}

variable "services_node_count" {
  description = "Fixed size of the services node group. Karpenter's controller runs here."
  type        = number
  default     = 2
}

variable "gpu_instance_types" {
  description = "Instance types Karpenter may launch for the gpu pool."
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "gpu_max_nodes" {
  description = "Most nodes the gpu pool may run at once."
  type        = number
  default     = 1
}

variable "gpu_root_volume_gib" {
  description = "Root volume size of gpu nodes. Holds the pulled job images."
  type        = number
  default     = 100
}

variable "karpenter_version" {
  description = "Karpenter Helm chart version."
  type        = string
  default     = "1.14.1"
}

variable "nvidia_device_plugin_version" {
  description = "NVIDIA device plugin Helm chart version."
  type        = string
  default     = "0.20.1"
}

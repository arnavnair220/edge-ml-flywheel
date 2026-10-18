# The job images, built with `docker buildx bake`. CI and local builds both
# read this file, so it is the only list of images. See docs/images.md.

variable "REGISTRY" {
  default = "localhost:5001/edge-ml-flywheel"
}

# The commit SHA in CI. Set it locally to tag a build by commit.
variable "TAG" {
  default = "dev"
}

variable "IMAGES" {
  default = ["train", "score", "evaluate", "ingest", "partition", "run", "fleet"]
}

group "default" {
  targets = IMAGES
}

target "image" {
  name       = image
  matrix     = { image = IMAGES }
  context    = "."
  dockerfile = "images/${image}.Dockerfile"
  tags       = ["${REGISTRY}/${image}:${TAG}"]
  labels = {
    "org.opencontainers.image.source" = "https://github.com/arnavnair220/edge-ml-flywheel"
  }
}

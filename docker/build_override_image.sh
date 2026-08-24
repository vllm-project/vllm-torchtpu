#!/bin/bash
set -e

# Usage Examples:
#
# 1. Build a dev variant overriding with local wheels:
#    ./docker/build_override_image.sh -b vllm-torchtpu-local:dev -t vllm-torchtpu-local:dev-override
#
# 2. Build a prod variant overriding with local wheels:
#    ./docker/build_override_image.sh -b vllm-torchtpu-local:prod -t vllm-torchtpu-local:prod-override

# Default values
IMAGE_TAG="vllm-torchtpu-local-override"
BASE_IMAGE="vllm-torchtpu-local"
OVERRIDE_SOURCE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -b|--base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    -o|--override-source)
      OVERRIDE_SOURCE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "===> Building vllm-torchtpu local-wheel-override variant image..."
echo "Variant Image Tag: $IMAGE_TAG"
echo "Base Image: $BASE_IMAGE"
echo "Override Source: $OVERRIDE_SOURCE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

# Run docker build with BuildKit
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile.override" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg OVERRIDE_SOURCE="${OVERRIDE_SOURCE}" \
  "${REPO_ROOT}"

echo "===> Done!"

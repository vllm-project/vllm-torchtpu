#!/bin/bash
set -e

# Usage Examples:
#
# 1. Build with default values (prod variant):
#    ./docker/build_image.sh
#
# 2. Build with custom image tag and base image:
#    ./docker/build_image.sh -t my-vllm-image -b my-base-image
#
# 3. Build with local vLLM source:
#    ./docker/build_image.sh -s /path/to/vllm/source

# Default values
IMAGE_TAG="vllm-torchtpu-local"
BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"
VLLM_SOURCE=""
TARGET="prod"

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
    -s|--vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "===> Building vllm-torchtpu image..."
echo "Image Tag: $IMAGE_TAG"
echo "Base Image: $BASE_IMAGE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

if [ -z "${ACCESS_TOKEN:-}" ]; then
  if command -v gcloud >/dev/null 2>&1; then
    ACCESS_TOKEN="$(gcloud auth print-access-token)"
    export ACCESS_TOKEN
  elif command -v curl >/dev/null 2>&1; then
    METADATA_TOKEN_JSON="$(curl -sf -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
      || true)"
    ACCESS_TOKEN="$(printf '%s' "${METADATA_TOKEN_JSON}" | \
      sed -nE 's/.*"access_token"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')"
    if [ -n "${ACCESS_TOKEN}" ]; then
      export ACCESS_TOKEN
    fi
  fi
fi

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "Missing ACCESS_TOKEN and gcloud is unavailable. Private registry authentication will fail."
  exit 1
fi

# Handle local vllm source directory
CLEANUP_VLLM=false
if [ -n "$VLLM_SOURCE" ] && [ -d "$VLLM_SOURCE" ]; then
  echo "===> Copying local vllm source from $VLLM_SOURCE..."
  rm -rf "${REPO_ROOT}/vllm"
  cp -r "$VLLM_SOURCE" "${REPO_ROOT}/vllm"
  CLEANUP_VLLM=true
else
  echo "===> Creating empty vllm directory for build context..."
  mkdir -p "${REPO_ROOT}/vllm"
  touch "${REPO_ROOT}/vllm/.dummy"
  CLEANUP_VLLM=true
fi

echo "Build Target: $TARGET"
DOCKER_ARGS=(
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  --target "${TARGET}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg VLLM_SOURCE="${VLLM_SOURCE}" \
)

if [ -n "${ACCESS_TOKEN:-}" ]; then
  DOCKER_ARGS+=(--secret "id=gcloud_token,env=ACCESS_TOKEN")
fi

# Run docker build
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"

if [ "$CLEANUP_VLLM" = true ]; then
  echo "===> Cleaning up copied vllm source..."
  rm -rf "${REPO_ROOT}/vllm"
fi

echo "===> Done!"

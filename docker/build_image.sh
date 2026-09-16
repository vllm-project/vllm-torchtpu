#!/bin/bash
set -e

usage() {
  cat <<USAGE
Usage: ./docker/build_image.sh [options]

Builds the vllm-torchtpu image from the repository root as the docker context.

Options:
  -t, --image-tag TAG      Tag for the built image (default: vllm-torchtpu-local)
  -s, --vllm-source DIR    Local vLLM source tree to install in editable mode
      --target TARGET      Dockerfile stage to build: prod, dev or ci (default: prod)
  -h, --help               Show this help and exit

Examples:
  ./docker/build_image.sh
  ./docker/build_image.sh -t my-vllm-image
  ./docker/build_image.sh -s /path/to/vllm/source

The Dockerfile copies <repo>/vllm into the image. The script stages that
directory itself (a copy of --vllm-source, or an empty directory) and removes it
when it exits. If <repo>/vllm already exists with content, the script refuses
to touch it; pass it with --vllm-source to use it in place.
USAGE
}

# Default values
IMAGE_TAG="vllm-torchtpu-local"
VLLM_SOURCE=""
VLLM_COMMIT_HASH=""
TARGET="prod"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -s|--vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    -c|--vllm-commit-hash)
      VLLM_COMMIT_HASH="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "===> Building vllm-torchtpu image..."
echo "Image Tag: $IMAGE_TAG"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"
STAGED_VLLM="${REPO_ROOT}/vllm"

if [ -n "$VLLM_SOURCE" ] && [ ! -d "$VLLM_SOURCE" ]; then
  echo "vllm source directory not found: $VLLM_SOURCE" >&2
  exit 1
fi

# Enforce VLLM_COMMIT_HASH to prevent silently building from unpinned main
if [ -z "$VLLM_COMMIT_HASH" ]; then
  # Extract the pinned vLLM commit directly from pyproject.toml
  VLLM_COMMIT_HASH="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"

  if [ -n "$VLLM_COMMIT_HASH" ]; then
    echo "-c/--vllm-commit-hash not provided. Defaulting to pyproject.toml pin: $VLLM_COMMIT_HASH"
  else
    echo "ERROR: VLLM_COMMIT_HASH could not be parsed from pyproject.toml." >&2
    exit 1
  fi
fi


echo "vLLM Commit Hash: $VLLM_COMMIT_HASH"

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

# The staged directory is removed only when this script created it. The EXIT
# trap also runs after the INT and TERM handlers, so an interrupted or failed
# docker build leaves nothing behind.
CLEANUP_VLLM=false
cleanup() {
  if [ "$CLEANUP_VLLM" = true ]; then
    echo "===> Cleaning up staged vllm source..."
    rm -rf "${STAGED_VLLM}"
    CLEANUP_VLLM=false
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The Dockerfile only needs the directory to exist, so the placeholder is an
# empty directory, and an empty directory is the only thing the script reuses.
is_empty_dir() {
  [ -d "$1" ] && [ -z "$(find "$1" -mindepth 1 -print -quit)" ]
}

# Handle local vllm source directory
if [ -n "$VLLM_SOURCE" ] && [ "$(cd "$VLLM_SOURCE" && pwd)" = "${STAGED_VLLM}" ]; then
  echo "===> Using in-tree vllm source at ${STAGED_VLLM} in place..."
elif [ -e "${STAGED_VLLM}" ] && ! is_empty_dir "${STAGED_VLLM}"; then
  echo "Refusing to overwrite existing ${STAGED_VLLM}." >&2
  echo "Move it away, or build from it with: $0 -s ${STAGED_VLLM}" >&2
  exit 1
elif [ -n "$VLLM_SOURCE" ]; then
  echo "===> Copying local vllm source from $VLLM_SOURCE..."
  CLEANUP_VLLM=true
  rm -rf "${STAGED_VLLM}"
  cp -r "$VLLM_SOURCE" "${STAGED_VLLM}"
else
  echo "===> Creating empty vllm directory for build context..."
  CLEANUP_VLLM=true
  mkdir -p "${STAGED_VLLM}"
fi

echo "Build Target: $TARGET"
DOCKER_ARGS=(
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  --target "${TARGET}" \
  --build-arg VLLM_SOURCE="${VLLM_SOURCE}" \
  --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
)

if [ -n "${ACCESS_TOKEN:-}" ]; then
  DOCKER_ARGS+=(--secret "id=gcloud_token,env=ACCESS_TOKEN")
fi

# Run docker build
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"

cleanup
echo "===> Done!"

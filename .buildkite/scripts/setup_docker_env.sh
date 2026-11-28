#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

TARGET="${1:-dev}"
IMAGE_SUFFIX=""
if [ "${TARGET}" = "omni" ]; then
  IMAGE_SUFFIX="-omni"
elif [ "${TARGET}" != "dev" ]; then
  echo "[FATAL] Unsupported target '${TARGET}'; expected dev or omni." >&2
  exit 1
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
# Resolve the target vLLM commit SHA from build metadata. bootstrap.sh and
# bootstrap_dev.sh publish it; a pipeline that uploads pipeline_build.yml
# directly (e.g. a standalone image-build pipeline) has no bootstrap step, so
# fall back to VLLM_COMMIT_HASH_OVERRIDE or the vLLM pin in pyproject.toml.
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/configs/pipeline_config.sh"
  if [ -n "${VLLM_COMMIT_HASH_OVERRIDE:-}" ]; then
    VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH_OVERRIDE}"
    VLLM_COMMIT_SOURCE="VLLM_COMMIT_HASH_OVERRIDE"
  else
    VLLM_COMMIT_HASH="$(get_vllm_commit_hash)"
    VLLM_COMMIT_SOURCE="pyproject.toml"
  fi
  echo "[WARN] VLLM_COMMIT_HASH metadata is empty (no bootstrap step);" \
    "using ${VLLM_COMMIT_HASH} from ${VLLM_COMMIT_SOURCE}." >&2
  # Publish it so later steps of this build resolve the same commit.
  buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
fi

BASE_TAG="${COMMIT_HASH}-${VLLM_COMMIT_HASH}${IMAGE_SUFFIX}"
BASE_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}"

# CI_IMAGE=<image> (the dev target only): publish that image as this build's
# and build nothing, as K3_ENGINE_IMAGE does for the Kimi-K3 engine image. The
# image's code is what the steps run, so give one built from a commit whose
# image would be the same - a rerun of a pipeline-only change, say.
if [[ -n "${CI_IMAGE:-}" && -z "${IMAGE_SUFFIX}" ]]; then
  if ! docker manifest inspect "${CI_IMAGE}" >/dev/null 2>&1; then
    echo "[FATAL] CI_IMAGE=${CI_IMAGE} is not in the registry." >&2
    exit 1
  fi
  echo "--- Image given: ${CI_IMAGE}"
  buildkite-agent meta-data set "CI_IMAGE_TAG" "${CI_IMAGE}"
  echo "--- Done setup_docker_env.sh"
  exit 0
fi

# A single-leg build accepts at most one wheel version override.
if [[ -n "${TORCH_TPU_VERSION_OVERRIDE:-}" && -n "${TPU_SYNC_TORCH_VERSION_OVERRIDE:-}" ]]; then
  echo "[FATAL] Both TORCH_TPU_VERSION_OVERRIDE and TPU_SYNC_TORCH_VERSION_OVERRIDE are set; set only one version override per build." >&2
  exit 1
fi

# bootstrap.py (and bootstrap_dev.sh) resolves, normalizes, and validates the
# canary leg once and publishes only 'torch-tpu' or 'tpu-sync' to meta-data.
CANARY_LEG="$(buildkite-agent meta-data get "TORCHTPU_NIGHTLY_LEG" --default "")"
if [ -z "${CANARY_LEG}" ]; then
  if [[ -n "${TORCH_TPU_VERSION_OVERRIDE:-}" || -n "${TPU_SYNC_TORCH_VERSION_OVERRIDE:-}" ]]; then
    echo "[FATAL] Wheel version override is set without TORCHTPU_NIGHTLY_LEG metadata; run via bootstrap.py." >&2
    exit 1
  fi
fi

# A rerun of a commit - a retried build, or a second build of the same commit -
# finds its image in the registry already: publish it and skip the build, as
# build_k3_engine_image.sh does for the engine image. The tag names this commit
# and the vLLM ref the image is built from, and the steps pull it by that tag
# either way. Skipping also keeps the rerun off the builder's disk and off the
# package registries, either of which can fail a build with nothing new to
# build. CI_IMAGE_REBUILD=1 builds anyway. The nightly canary below layers
# fresh wheels over the base image, so it always builds.
if [[ -z "${CANARY_LEG}" && "${CI_IMAGE_REBUILD:-0}" != "1" ]] \
    && docker manifest inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
  echo "--- ${BASE_IMAGE_TAG} is already in the registry; reusing it"
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${BASE_IMAGE_TAG}"
  echo "--- Done setup_docker_env.sh"
  exit 0
fi

if [ -z "${CANARY_LEG}" ]; then
  echo "--- Building Docker Image: ${BASE_IMAGE_TAG} (target: ${TARGET})"
  # Run build_image.sh and pass VLLM_COMMIT_HASH directly as a build argument
  ./docker/build_image.sh --target "${TARGET}" -t "${BASE_IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"

  # Publish image tag metadata for downstream test steps
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${BASE_IMAGE_TAG}"

  echo "--- Pushing Docker Image to Registry"
  docker push "${BASE_IMAGE_TAG}"

  echo "--- Cleaning up local built image"
  docker rmi "${BASE_IMAGE_TAG}" || true

  echo "--- Done setup_docker_env.sh"
  exit 0
fi

# ==============================================================================
# Nightly Canary Build: Single-Leg Layered Override
# ==============================================================================
case "${CANARY_LEG}" in
  torch-tpu)
    if [ -n "${TPU_SYNC_TORCH_VERSION_OVERRIDE:-}" ]; then
      echo "[FATAL] TPU_SYNC_TORCH_VERSION_OVERRIDE is set, but selected canary leg is '${CANARY_LEG}' (expected tpu-sync)." >&2
      exit 1
    fi
    OVERRIDE_PACKAGE="torch-tpu"
    WHEEL_PREFIX="torch_tpu"
    TAG_PREFIX="torch-tpu"
    EXTRA_TAG_PREFIX=""
    OVERRIDE_VERSION="${TORCH_TPU_VERSION_OVERRIDE:-}"
    VERSION_META_KEY="TORCH_TPU_VERSION"
    ;;
  tpu-sync)
    if [ -n "${TORCH_TPU_VERSION_OVERRIDE:-}" ]; then
      echo "[FATAL] TORCH_TPU_VERSION_OVERRIDE is set, but selected canary leg is '${CANARY_LEG}' (expected torch-tpu)." >&2
      exit 1
    fi
    OVERRIDE_PACKAGE="tpu-sync-torch"
    WHEEL_PREFIX="tpu_sync_torch"
    TAG_PREFIX="tpu-sync"
    EXTRA_TAG_PREFIX="tpu-sync-torch"
    OVERRIDE_VERSION="${TPU_SYNC_TORCH_VERSION_OVERRIDE:-}"
    VERSION_META_KEY="TPU_SYNC_TORCH_VERSION"
    ;;
  *)
    echo "[FATAL] Unsupported TORCHTPU_NIGHTLY_LEG '${CANARY_LEG}'; expected torch-tpu or tpu-sync." >&2
    exit 1
    ;;
esac

echo "--- Ensuring Base Docker Image is available: ${BASE_IMAGE_TAG}"
if ! docker pull "${BASE_IMAGE_TAG}" 2>/dev/null; then
  echo "Base image not found in registry. Building base image: ${BASE_IMAGE_TAG}"
  ./docker/build_image.sh --target "${TARGET}" -t "${BASE_IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"
  echo "Pushing base image to registry..."
  docker push "${BASE_IMAGE_TAG}"
else
  echo "Reusing prebuilt base image from registry: ${BASE_IMAGE_TAG}"
fi

echo "--- Resolving ${OVERRIDE_PACKAGE} nightly wheel for Python 3.12 (leg: ${CANARY_LEG})"
REGISTRY_URL="https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple"

# Acquire registry access token from VM metadata server if not already provided
if [ -z "${ACCESS_TOKEN:-}" ]; then
  if command -v curl >/dev/null 2>&1; then
    METADATA_TOKEN_JSON="$(curl -sf -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
      || true)"
    ACCESS_TOKEN="$(printf '%s' "${METADATA_TOKEN_JSON}" | \
      sed -nE 's/.*"access_token"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')"
  fi
  if [ -z "${ACCESS_TOKEN:-}" ] && command -v gcloud >/dev/null 2>&1; then
    ACCESS_TOKEN="$(gcloud auth print-access-token 2>/dev/null || true)"
  fi
  if [ -n "${ACCESS_TOKEN:-}" ]; then
    export ACCESS_TOKEN
  fi
fi

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "[FATAL] Canary leg '${CANARY_LEG}' is enabled, but missing ACCESS_TOKEN and failed to obtain one via metadata server." >&2
  exit 1
fi

resolve_latest_cp312_wheel_version() {
  local dist_pkg="$1"
  local wheel_prefix="$2"
  local raw_html version

  raw_html="$(curl -f -s -H "Authorization: Bearer ${ACCESS_TOKEN}" "${REGISTRY_URL}/${dist_pkg}/")" || {
    echo "[FATAL] Failed to fetch ${dist_pkg} package index from ${REGISTRY_URL}/${dist_pkg}/" >&2
    exit 1
  }
  # PEP 427 wheel names separate fields with '-'; match up to -cp312 without
  # '-' inside the version segment so -cp312-cp312- is not partially kept.
  version="$({ echo "${raw_html}" | grep -oE "${wheel_prefix}-[0-9a-zA-Z.+_]+-cp312" || true; } | sed "s/^${wheel_prefix}-//;s/-cp312\$//" | sort -V | tail -n 1)"
  if [ -z "${version}" ]; then
    echo "[FATAL] Canary leg '${CANARY_LEG}' is enabled, but failed to resolve a cp312 wheel for ${dist_pkg}!" >&2
    exit 1
  fi
  echo "${version}"
}

if [ -z "${OVERRIDE_VERSION}" ]; then
  OVERRIDE_VERSION="$(resolve_latest_cp312_wheel_version "${OVERRIDE_PACKAGE}" "${WHEEL_PREFIX}")" || exit 1
fi

echo "Resolved ${OVERRIDE_PACKAGE} version: ${OVERRIDE_VERSION}"

# Docker tag names forbid '+'; sanitize '+' to '_' for image tagging
OVERRIDE_TAG_VERSION="${OVERRIDE_VERSION//+/_}"

# Image tag names follow the ${BASE_TAG}-${TAG_PREFIX}_${VERSION} convention.
OVERRIDE_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}-${TAG_PREFIX}_${OVERRIDE_TAG_VERSION}"
DOCKER_TAG_ARGS=(-t "${OVERRIDE_IMAGE_TAG}")
IMAGES_TO_PUSH=("${OVERRIDE_IMAGE_TAG}")
if [ -n "${EXTRA_TAG_PREFIX}" ]; then
  EXTRA_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}-${EXTRA_TAG_PREFIX}_${OVERRIDE_TAG_VERSION}"
  DOCKER_TAG_ARGS+=(-t "${EXTRA_IMAGE_TAG}")
  IMAGES_TO_PUSH+=("${EXTRA_IMAGE_TAG}")
fi

echo "--- Building Delta Image (${OVERRIDE_PACKAGE}): ${OVERRIDE_IMAGE_TAG}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f "docker/Dockerfile.override" \
  "${DOCKER_TAG_ARGS[@]}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
  --build-arg OVERRIDE_PACKAGE="${OVERRIDE_PACKAGE}" \
  --build-arg OVERRIDE_VERSION="${OVERRIDE_VERSION}" \
  --secret "id=gcloud_token,env=ACCESS_TOKEN" \
  .

echo "--- Pushing Delta Image (${OVERRIDE_PACKAGE}) to Registry"
for image_tag in "${IMAGES_TO_PUSH[@]}"; do
  docker push "${image_tag}"
done

# Publish image tag metadata for downstream test steps without overwriting
# non-omni keys when building TARGET=omni.
buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${OVERRIDE_IMAGE_TAG}"
buildkite-agent meta-data set "${VERSION_META_KEY}" "${OVERRIDE_VERSION}"

buildkite-agent annotate \
  ":rocket: Canary testing ${OVERRIDE_PACKAGE} \`${OVERRIDE_VERSION}\`" \
  --style "info" \
  --context "canary-${TAG_PREFIX}" || true

echo "--- Cleaning up local built images"
docker rmi "${IMAGES_TO_PUSH[@]}" || true

echo "--- Done setup_docker_env.sh"

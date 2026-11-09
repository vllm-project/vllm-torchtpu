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
# Resolve the target vLLM commit SHA from build metadata
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH metadata is empty; bootstrap.sh did not set it." >&2
  exit 1
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

# A rerun of a commit - a retried build, or a second build of the same commit -
# finds its image in the registry already: publish it and skip the build, as
# build_k3_engine_image.sh does for the engine image. The tag names this commit
# and the vLLM ref the image is built from, and the steps pull it by that tag
# either way. Skipping also keeps the rerun off the builder's disk and off the
# package registries, either of which can fail a build with nothing new to
# build. CI_IMAGE_REBUILD=1 builds anyway. The nightly canary below layers
# fresh wheels over the base image, so it always builds.
if [[ "${TORCHTPU_NIGHTLY:-0}" != "1" && "${CI_IMAGE_REBUILD:-0}" != "1" ]] \
    && docker manifest inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
  echo "--- ${BASE_IMAGE_TAG} is already in the registry; reusing it"
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX:-}" "${BASE_IMAGE_TAG}"
  echo "--- Done setup_docker_env.sh"
  exit 0
fi

if [[ "${TORCHTPU_NIGHTLY:-0}" != "1" ]]; then
  echo "--- Building Docker Image: ${BASE_IMAGE_TAG} (target: ${TARGET})"
  # Run build_image.sh and pass VLLM_COMMIT_HASH directly as a build argument
  ./docker/build_image.sh --target "${TARGET}" -t "${BASE_IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"

  # Publish image tag metadata for downstream test steps
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX:-}" "${BASE_IMAGE_TAG}"

  echo "--- Pushing Docker Image to Registry"
  docker push "${BASE_IMAGE_TAG}"

  echo "--- Cleaning up local built image"
  docker rmi "${BASE_IMAGE_TAG}" || true

  echo "--- Done setup_docker_env.sh"
  exit 0
fi

# ==============================================================================
# Nightly Canary Build: Two-Stage Layered Override (Option 2)
# ==============================================================================
echo "--- Ensuring Base Docker Image is available: ${BASE_IMAGE_TAG}"
if ! docker pull "${BASE_IMAGE_TAG}" 2>/dev/null; then
  echo "Base image not found in registry. Building base image: ${BASE_IMAGE_TAG}"
  ./docker/build_image.sh --target "${TARGET}" -t "${BASE_IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"
  echo "Pushing base image to registry..."
  docker push "${BASE_IMAGE_TAG}"
else
  echo "Reusing prebuilt base image from registry: ${BASE_IMAGE_TAG}"
fi

echo "--- Resolving latest torch-tpu and tpu-sync-torch nightly wheels for Python 3.12"
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
  echo "[FATAL] TORCHTPU_NIGHTLY is enabled, but missing ACCESS_TOKEN and failed to obtain one via metadata server." >&2
  exit 1
fi

TORCH_TPU_VERSION="${TORCH_TPU_VERSION_OVERRIDE:-}"
TPU_SYNC_TORCH_VERSION="${TPU_SYNC_TORCH_VERSION_OVERRIDE:-}"

if [ -z "${TORCH_TPU_VERSION}" ]; then
  RAW_HTML_TORCH_TPU="$(curl -f -s -H "Authorization: Bearer ${ACCESS_TOKEN}" "${REGISTRY_URL}/torch-tpu/")" || {
    echo "[FATAL] Failed to fetch torch-tpu package index from ${REGISTRY_URL}/torch-tpu/" >&2
    exit 1
  }
  TORCH_TPU_VERSION="$(echo "${RAW_HTML_TORCH_TPU}" | grep -oE 'torch_tpu-[0-9a-zA-Z.+_-]+-cp312' | sed 's/torch_tpu-//;s/-cp312//' | sort -V | tail -n 1)"
fi

if [ -z "${TPU_SYNC_TORCH_VERSION}" ]; then
  RAW_HTML_TPU_SYNC="$(curl -f -s -H "Authorization: Bearer ${ACCESS_TOKEN}" "${REGISTRY_URL}/tpu-sync-torch/")" || {
    echo "[FATAL] Failed to fetch tpu-sync-torch package index from ${REGISTRY_URL}/tpu-sync-torch/" >&2
    exit 1
  }
  TPU_SYNC_TORCH_VERSION="$(echo "${RAW_HTML_TPU_SYNC}" | grep -oE 'tpu_sync_torch-[0-9a-zA-Z.+_-]+-cp312' | sed 's/tpu_sync_torch-//;s/-cp312//' | sort -V | tail -n 1)"
fi

if [ -z "${TORCH_TPU_VERSION}" ]; then
  echo "[FATAL] TORCHTPU_NIGHTLY is enabled, but failed to resolve a cp312 wheel for torch-tpu!" >&2
  exit 1
fi

if [ -z "${TPU_SYNC_TORCH_VERSION}" ]; then
  echo "[FATAL] TORCHTPU_NIGHTLY is enabled, but failed to resolve a cp312 wheel for tpu-sync-torch!" >&2
  exit 1
fi

echo "Resolved torch-tpu version: ${TORCH_TPU_VERSION}"
echo "Resolved tpu-sync-torch version: ${TPU_SYNC_TORCH_VERSION}"

# Docker tag names forbid '+'; sanitize '+' to '_' for image tagging
TORCH_TPU_TAG_VERSION="${TORCH_TPU_VERSION//+/_}"
TPU_SYNC_TAG_VERSION="${TPU_SYNC_TORCH_VERSION//+/_}"

# Image tag names following ${BASE_TAG}-${DEP_REPO}_${VERSION} convention
TORCH_TPU_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}-torch-tpu_${TORCH_TPU_TAG_VERSION}"
TPU_SYNC_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}-tpu-sync_${TPU_SYNC_TAG_VERSION}"
TPU_SYNC_FULL_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}-tpu-sync-torch_${TPU_SYNC_TAG_VERSION}"

# Leg 1: Build delta image overriding torch-tpu
echo "--- Building Delta Image for Leg 1 (torch-tpu): ${TORCH_TPU_IMAGE_TAG}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f "docker/Dockerfile.override" \
  -t "${TORCH_TPU_IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
  --build-arg OVERRIDE_PACKAGE="torch-tpu" \
  --build-arg OVERRIDE_VERSION="${TORCH_TPU_VERSION}" \
  --secret "id=gcloud_token,env=ACCESS_TOKEN" \
  .

echo "--- Pushing Delta Image for Leg 1 to Registry"
docker push "${TORCH_TPU_IMAGE_TAG}"

# Leg 2: Build delta image overriding tpu-sync-torch
echo "--- Building Delta Image for Leg 2 (tpu-sync-torch): ${TPU_SYNC_IMAGE_TAG}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f "docker/Dockerfile.override" \
  -t "${TPU_SYNC_IMAGE_TAG}" \
  -t "${TPU_SYNC_FULL_IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
  --build-arg OVERRIDE_PACKAGE="tpu-sync-torch" \
  --build-arg OVERRIDE_VERSION="${TPU_SYNC_TORCH_VERSION}" \
  --secret "id=gcloud_token,env=ACCESS_TOKEN" \
  .

echo "--- Pushing Delta Image for Leg 2 to Registry"
docker push "${TPU_SYNC_IMAGE_TAG}"
docker push "${TPU_SYNC_FULL_IMAGE_TAG}"

# Determine primary CI image based on TORCHTPU_NIGHTLY_LEG if specified
PRIMARY_IMAGE_TAG="${TORCH_TPU_IMAGE_TAG}"
if [[ "${TORCHTPU_NIGHTLY_LEG:-}" =~ ^(tpu-sync|tpu_sync|tpu-sync-torch)$ ]]; then
  PRIMARY_IMAGE_TAG="${TPU_SYNC_IMAGE_TAG}"
fi

# Publish image tag metadata for downstream test steps
buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX:-}" "${PRIMARY_IMAGE_TAG}"
buildkite-agent meta-data set "CI_IMAGE_TAG_TORCH_TPU${IMAGE_SUFFIX:-}" "${TORCH_TPU_IMAGE_TAG}"
buildkite-agent meta-data set "CI_IMAGE_TAG_TPU_SYNC${IMAGE_SUFFIX:-}" "${TPU_SYNC_IMAGE_TAG}"
buildkite-agent meta-data set "CI_IMAGE_TAG_TPU_SYNC_TORCH${IMAGE_SUFFIX:-}" "${TPU_SYNC_IMAGE_TAG}"
if [ -n "${IMAGE_SUFFIX:-}" ]; then
  # Also set default keys if not already present
  buildkite-agent meta-data set "CI_IMAGE_TAG_TORCH_TPU" "${TORCH_TPU_IMAGE_TAG}" || true
  buildkite-agent meta-data set "CI_IMAGE_TAG_TPU_SYNC" "${TPU_SYNC_IMAGE_TAG}" || true
  buildkite-agent meta-data set "CI_IMAGE_TAG_TPU_SYNC_TORCH" "${TPU_SYNC_IMAGE_TAG}" || true
fi
buildkite-agent meta-data set "TORCH_TPU_VERSION" "${TORCH_TPU_VERSION}"
buildkite-agent meta-data set "TPU_SYNC_TORCH_VERSION" "${TPU_SYNC_TORCH_VERSION}"

buildkite-agent annotate ":rocket: Canary testing torch-tpu \`${TORCH_TPU_VERSION}\` and tpu-sync-torch \`${TPU_SYNC_TORCH_VERSION}\`" --style "info" || true

echo "--- Cleaning up local built images"
docker rmi "${TORCH_TPU_IMAGE_TAG}" "${TPU_SYNC_IMAGE_TAG}" "${TPU_SYNC_FULL_IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"

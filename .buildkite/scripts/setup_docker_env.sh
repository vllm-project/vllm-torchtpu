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
# finds its image in the registry already: publish it and skip the build. The tag names this commit
# and the vLLM ref the image is built from, and the steps pull it by that tag
# either way. Skipping also keeps the rerun off the builder's disk and off the
# package registries, either of which can fail a build with nothing new to
# build. CI_IMAGE_REBUILD=1 builds anyway. TORCHTPU_NIGHTLY=1 is refused
# below rather than handed a reused image.
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

# main also has a TORCHTPU_NIGHTLY canary that layers fresh torch-tpu and
# tpu-sync-torch wheels over the base image. It needs main's
# docker/Dockerfile.override; this branch keeps its own, which takes different
# build args, so the canary is not built here.
echo "[FATAL] TORCHTPU_NIGHTLY=1: the nightly canary build is not supported on this branch." >&2
exit 1

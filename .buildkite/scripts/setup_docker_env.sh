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

# shellcheck source=.buildkite/scripts/ci_image.sh
source "${SCRIPT_DIR}/ci_image.sh"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
# Resolve the target vLLM commit SHA from build metadata
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH metadata is empty; bootstrap.sh did not set it." >&2
  exit 1
fi

# Include vLLM commit SHA in the image tag for registry isolation
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}-${VLLM_COMMIT_HASH}"

echo "--- Building Docker Image: ${IMAGE_TAG}"
# Run build_image.sh and pass VLLM_COMMIT_HASH directly as a build argument
./docker/build_image.sh --target dev -t "${IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"

# Publish image tag metadata for downstream test steps
buildkite-agent meta-data set "CI_IMAGE_TAG" "${IMAGE_TAG}"

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

# A mirror push that fails is fatal, not a warning: the mirror exists for pulls
# written down outside this tree, and those have no way to notice that what
# they resolve is a week old. The tag is already pushed to IMAGE_REPO by here,
# so a failure leaves the lane's own image intact and only this step red.
MIRROR_TAGS=()
# shellcheck disable=SC2086  # the list is space separated on purpose
for MIRROR_REPO in ${CI_IMAGE_MIRROR_REPOS}; do
  MIRROR_TAG="${MIRROR_REPO}:${COMMIT_HASH}-${VLLM_COMMIT_HASH}"
  MIRROR_TAGS+=("${MIRROR_TAG}")
  echo "--- Mirroring Docker Image: ${MIRROR_TAG}"
  docker tag "${IMAGE_TAG}" "${MIRROR_TAG}"
  docker push "${MIRROR_TAG}"
done

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" ${MIRROR_TAGS[@]+"${MIRROR_TAGS[@]}"} || true

echo "--- Done setup_docker_env.sh"

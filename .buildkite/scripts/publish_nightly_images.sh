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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

REGISTRY="${NIGHTLY_REGISTRY:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu}"
DATE_TAG="nightly-$(date +%Y%m%d)"
LATEST_TAG="latest"

cd "${REPO_ROOT}"

# Build and push variants in order of dependency (ci -> dev, prod) so BuildKit layer cache is reused
TARGETS=("ci" "dev" "prod")

for TARGET in "${TARGETS[@]}"; do
  IMAGE_NAME="${REGISTRY}/torchtpu-vllm-${TARGET}"
  DATE_IMAGE="${IMAGE_NAME}:${DATE_TAG}"
  LATEST_IMAGE="${IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Building Nightly Image (${TARGET}): ${DATE_IMAGE}"
  ./docker/build_image.sh --target "${TARGET}" -t "${DATE_IMAGE}"

  echo "--- :docker: Tagging as ${LATEST_IMAGE}"
  docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

  echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Artifact Registry"
  docker push "${DATE_IMAGE}"
  docker push "${LATEST_IMAGE}"

  echo "--- Cleaning up locally built image tags for ${TARGET}"
  docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" || true
done

echo "--- Successfully built and published all nightly images!"

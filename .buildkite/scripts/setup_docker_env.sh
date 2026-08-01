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

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}"

echo "--- Building Docker Image: ${IMAGE_TAG}"
# Run the existing build_image.sh script with target dev
./docker/build_image.sh --target dev -t "${IMAGE_TAG}"

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"

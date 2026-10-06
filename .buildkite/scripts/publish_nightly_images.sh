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

# The branch decides the tags. main pushes nightly-YYYYMMDD and moves latest.
# release/X.Y.Z pushes YYYYMMDD-X.Y.Z and moves X.Y.Z, so a release nightly
# never touches latest. The Buildkite pipeline runs this script from whichever
# branch its schedule names, so a release branch cut from main after this
# change needs only a new schedule, not a code change.
BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
DATE="$(date +%Y%m%d)"
case "${BRANCH}" in
  main)
    DATE_TAG="nightly-${DATE}"
    LATEST_TAG="latest"
    ;;
  release/*)
    VERSION="${BRANCH#release/}"
    # Starting with a digit keeps it from colliding with latest or nightly-*.
    if [[ ! "${VERSION}" =~ ^[0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "ERROR: branch '${BRANCH}' does not name a version usable as an image tag; expected release/X.Y.Z." >&2
      exit 1
    fi
    DATE_TAG="${DATE}-${VERSION}"
    LATEST_TAG="${VERSION}"
    ;;
  *)
    echo "ERROR: nightly images are published from main or release/X.Y.Z, not '${BRANCH}'." >&2
    exit 1
    ;;
esac

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

REGISTRY="${NIGHTLY_REGISTRY:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu}"

cd "${REPO_ROOT}"

# Build and push variants in order of dependency (ci -> dev, prod) so BuildKit layer cache is reused
TARGETS=("ci" "dev" "prod")

for TARGET in "${TARGETS[@]}"; do
  IMAGE_NAME="${REGISTRY}/vllm-torchtpu-${TARGET}"
  DATE_IMAGE="${IMAGE_NAME}:${DATE_TAG}"
  LATEST_IMAGE="${IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Building Nightly Image (${TARGET}): ${DATE_IMAGE}"
  ./docker/build_image.sh --target "${TARGET}" -t "${DATE_IMAGE}"

  echo "--- :docker: Tagging as ${LATEST_IMAGE}"
  docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

  # Step 1: Tag and push extra legacy torchtpu-vllm images for backward compatibility
  # so existing consumers continue working smoothly until migration is finalized.
  EXTRA_IMAGE_NAME="${REGISTRY}/torchtpu-vllm-${TARGET}"
  EXTRA_DATE_IMAGE="${EXTRA_IMAGE_NAME}:${DATE_TAG}"
  EXTRA_LATEST_IMAGE="${EXTRA_IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Tagging extra legacy images for backward compatibility (${EXTRA_DATE_IMAGE}, ${EXTRA_LATEST_IMAGE})"
  docker tag "${DATE_IMAGE}" "${EXTRA_DATE_IMAGE}"
  docker tag "${DATE_IMAGE}" "${EXTRA_LATEST_IMAGE}"

  echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Artifact Registry"
  docker push "${DATE_IMAGE}"
  docker push "${LATEST_IMAGE}"

  echo "--- :docker: Pushing extra images ${EXTRA_DATE_IMAGE} and ${EXTRA_LATEST_IMAGE} to Artifact Registry"
  docker push "${EXTRA_DATE_IMAGE}"
  docker push "${EXTRA_LATEST_IMAGE}"

  echo "--- Cleaning up locally built image tags for ${TARGET}"
  docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" "${EXTRA_DATE_IMAGE}" "${EXTRA_LATEST_IMAGE}" || true
done

echo "--- Successfully built and published all nightly images!"

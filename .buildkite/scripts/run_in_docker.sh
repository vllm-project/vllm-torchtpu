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

if [ "$#" -lt 1 ]; then
  echo "ERROR: Usage: $0 <command_to_run_in_docker...>"
  exit 1
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}"

echo "--- Pulling Docker Image: ${IMAGE_TAG}"
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "${IMAGE_TAG}"

# Ensure cache directory exists on the host
mkdir -p /mnt/disks/persist/models/hub

# Ensure results directory exists on the persistent disk (always mountable)
rm -rf /mnt/disks/persist/perf_eval_results
mkdir -p /mnt/disks/persist/perf_eval_results
chmod 777 /mnt/disks/persist/perf_eval_results 2>/dev/null || true

# Ensure a clean results directory exists in the workspace
rm -rf perf_eval_results
mkdir -p perf_eval_results

# Set HF_TOKEN from GCP Secret Manager if not already in /etc/environment
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

# Disable exit on error temporarily to copy results back even if the test fails
set +e

CONTAINER_NAME="vllm-torchtpu-ci"
echo "--- Cleaning up any existing container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

trap 'docker kill "${CONTAINER_NAME}" 2>/dev/null || true' EXIT INT TERM

echo "--- Running command in Docker container"
docker run --rm --name "${CONTAINER_NAME}" --privileged --net=host --shm-size=16g --device /dev/fuse \
  -v /mnt/disks/persist/models:/local_hf_cache \
  -v /mnt/disks/persist/perf_eval_results:/perf_eval_results \
  -e HF_HOME=/local_hf_cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" \
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \
  -e FORCE_COLOR="1" \
  -e UV_NO_CACHE="1" \
  -e TQDM_MININTERVAL="30" \
  -e MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-}" \
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}" \
  -e EVALPLUS_DATASETS="${EVALPLUS_DATASETS:-}" \
  -e EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}" \
  "${IMAGE_TAG}" \
  bash -c "
    umask 000
    rm -rf /perf_eval_results/*
    mkdir -p /tmp/torch_tpu_cache
    ln -sfn /tmp/torch_tpu_cache /dev/shm/torch_tpu_cache
    $*
  "
DOCKER_EXIT_CODE=$?
set -e

echo "--- Copying test results back to workspace for artifact upload"
cp -r /mnt/disks/persist/perf_eval_results/* perf_eval_results/ 2>/dev/null || true

exit $DOCKER_EXIT_CODE

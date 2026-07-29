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

# Set HF_TOKEN from GCP Secret Manager if not already in /etc/environment
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi

# Sourcing /etc/environment early ensures TPU_VERSION, HF_TOKEN, etc., are available immediately
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

if [ "$#" -lt 1 ]; then
  echo "ERROR: Usage: $0 <command_to_run_in_docker...>"
  exit 1
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

echo "--- Pulling Docker Image: ${IMAGE_TAG}"
docker pull "${IMAGE_TAG}"

# Ensure cache directory exists on the host
mkdir -p /mnt/disks/persist/models

# Ensure results directory exists on the persistent disk (always mountable)
rm -rf /mnt/disks/persist/perf_eval_results
mkdir -p /mnt/disks/persist/perf_eval_results
chmod 777 /mnt/disks/persist/perf_eval_results 2>/dev/null || true

# Ensure a clean results directory exists in the workspace
rm -rf perf_eval_results
mkdir -p perf_eval_results

# Disable exit on error temporarily to copy results back even if the test fails
set +e

CONTAINER_NAME="vllm-torchtpu-ci"
echo "--- Cleaning up any existing container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

trap 'docker kill "${CONTAINER_NAME}" 2>/dev/null || true' EXIT INT TERM

# Buildkite and test suite related environment variables and volume mounts
export BUILDKITE_PARALLEL_JOB="${BUILDKITE_PARALLEL_JOB:-0}"
export BUILDKITE_PARALLEL_JOB_COUNT="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"

TEST_SUITE_VARS=()
while IFS='=' read -r key _; do
  if [[ "$key" == BUILDKITE_* ]]; then
    TEST_SUITE_VARS+=(-e "$key")
  fi
done < <(env)
if [ -n "${BUILDKITE_OIDC_TOKEN_PATH:-}" ]; then
  TEST_SUITE_VARS+=(-v "$(dirname "${BUILDKITE_OIDC_TOKEN_PATH}"):$(dirname "${BUILDKITE_OIDC_TOKEN_PATH}")")
fi

# Spanner eval upload tracking & metadata variables (from PR #14)
SPANNER_EVAL_VARS=(
  -e CREATED_BY="${CREATED_BY:-}"
  -e GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
  -e NIGHTLY="${NIGHTLY:-}"
  -e RUN_TYPE="${RUN_TYPE:-}"
  -e TPU_NAME="${TPU_NAME:-}"
)

echo "--- Running command in Docker container"
docker run --rm --name "${CONTAINER_NAME}" --privileged --net=host --shm-size=16g --device /dev/fuse \
  -w /root/torchtpu-vllm \
  -v /mnt/disks/persist/models:/local_hf_cache \
  -v /mnt/disks/persist/perf_eval_results:/perf_eval_results \
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}" \
  -e EVALPLUS_DATASETS="${EVALPLUS_DATASETS:-}" \
  -e EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}" \
  -e FORCE_COLOR="1" \
  -e HF_HOME=/local_hf_cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e RUN_EVALPLUS="${RUN_EVALPLUS:-}" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" \
  -e TQDM_MININTERVAL="30" \
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \
  -e UV_NO_CACHE="1" \
  ${MODEL_PATH:+-e MODEL_PATH="${MODEL_PATH}"} \
  ${P4D2_BIND_HOST:+-e P4D2_BIND_HOST="${P4D2_BIND_HOST}"} \
  ${PROXY_PORT:+-e PROXY_PORT="${PROXY_PORT}"} \
  ${RUN_ROOT:+-e RUN_ROOT="${RUN_ROOT}"} \
  ${SERVED_MODEL_NAME:+-e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"} \
  ${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:+-e TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}"} \
  "${TEST_SUITE_VARS[@]}" \
  "${SPANNER_EVAL_VARS[@]}" \
  "${IMAGE_TAG}" \
  bash -c '
    umask 000
    rm -rf /perf_eval_results/*
    "$@"
  ' -- "$@"
DOCKER_EXIT_CODE=$?
set -e

echo "--- Copying test results back to workspace for artifact upload"
cp -r /mnt/disks/persist/perf_eval_results/* perf_eval_results/ 2>/dev/null || true
# Clean up any broken symlinks on the host to prevent buildkite-agent upload failures
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

echo "[INFO] Docker finished with exit code ${DOCKER_EXIT_CODE}."

echo "--- Cleaning up pulled Docker image"
docker rmi "${IMAGE_TAG}" || true

exit $DOCKER_EXIT_CODE

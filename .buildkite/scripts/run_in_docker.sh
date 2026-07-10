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
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "${IMAGE_TAG}"

# Ensure cache directory exists on the host
mkdir -p /mnt/disks/persist/models/hub

# ==========================================
# 1. Cache Setup (TorchTPU Compilations)
# ==========================================
GCS_CACHE_BASE="gs://ullm-ci-cache/torchtpu_cache"
echo "[INFO] Probing TorchTPU version from docker image..."
TORCH_TPU_VERSION=$(docker run --rm --net=host "${IMAGE_TAG}" python3 -c "import importlib.metadata; print(importlib.metadata.version('torch-tpu'))")
if [ -z "${TORCH_TPU_VERSION}" ] || [ "${TORCH_TPU_VERSION}" = "unknown" ]; then
  echo "[ERROR] Failed to detect TorchTPU version from docker image ${IMAGE_TAG}."
  exit 1
fi
echo "[INFO] Detected TorchTPU Version: ${TORCH_TPU_VERSION}"

# Centralized GCS cache path (avoiding duplicate 'tpu' prefix)
CACHE_NAMESPACE="torchtpu${TORCH_TPU_VERSION}_${TPU_VERSION:-tpu6e}"
FINAL_CACHE_PATH="${GCS_CACHE_BASE}/${CACHE_NAMESPACE}"

LOCAL_TORCHTPU_CACHE_DIR="/mnt/disks/persist/torchtpu_cache/${CACHE_NAMESPACE}"

if ! mkdir -p "$LOCAL_TORCHTPU_CACHE_DIR"; then
  echo "[ERROR] Failed to create $LOCAL_TORCHTPU_CACHE_DIR on persistent disk."
  exit 1
fi
# Use Docker (running as root) to fix permissions without requiring sudo on the host
docker run --rm -v "$LOCAL_TORCHTPU_CACHE_DIR":"$LOCAL_TORCHTPU_CACHE_DIR" "${IMAGE_TAG}" chmod -R 777 "$LOCAL_TORCHTPU_CACHE_DIR" 2>/dev/null || true
echo "[INFO] Pulling TorchTPU Cache from GCS to local directory..."
gcloud storage rsync \
  --recursive \
  --no-clobber \
  --delete-unmatched-destination-objects \
  --exclude=".*_.gstmp$" \
  --no-user-output-enabled \
  "$FINAL_CACHE_PATH" "$LOCAL_TORCHTPU_CACHE_DIR" || \
  echo "[WARN] Failed to pull TorchTPU Cache from GCS. Proceeding with cold start."

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

echo "--- Running command in Docker container"
docker run --rm --name "${CONTAINER_NAME}" --privileged --net=host --shm-size=16g --device /dev/fuse \
  -v /mnt/disks/persist/models:/local_hf_cache \
  -v /mnt/disks/persist/perf_eval_results:/perf_eval_results \
  -v "$LOCAL_TORCHTPU_CACHE_DIR":"$LOCAL_TORCHTPU_CACHE_DIR" \
  -e HF_HOME=/local_hf_cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_CACHE_ROOT="$LOCAL_TORCHTPU_CACHE_DIR" \
  -e VLLM_XLA_CACHE_PATH="$LOCAL_TORCHTPU_CACHE_DIR" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" \
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \
  -e FORCE_COLOR="1" \
  -e UV_NO_CACHE="1" \
  -e TQDM_MININTERVAL="30" \
  -e MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-}" \
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}" \
  -e EVALPLUS_DATASETS="${EVALPLUS_DATASETS:-}" \
  -e EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}" \
  -e TPU_NAME="${TPU_NAME:-}" \
  -e BUILDKITE_COMMIT="${BUILDKITE_COMMIT:-}" \
  -e BUILDKITE_AGENT_META_DATA_QUEUE="${BUILDKITE_AGENT_META_DATA_QUEUE:-}" \
  -e BUILDKITE_AGENT_NAME="${BUILDKITE_AGENT_NAME:-}" \
  -e BUILDKITE_BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER:-}" \
  -e GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}" \
  -e RUN_TYPE="${RUN_TYPE:-}" \
  -e CREATED_BY="${CREATED_BY:-}" \
  -e NIGHTLY="${NIGHTLY:-}" \
  ${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:+-e TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}"} \
  ${RUN_ROOT:+-e RUN_ROOT="${RUN_ROOT}"} \
  ${MODEL_PATH:+-e MODEL_PATH="${MODEL_PATH}"} \
  ${SERVED_MODEL_NAME:+-e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"} \
  ${P4D2_BIND_HOST:+-e P4D2_BIND_HOST="${P4D2_BIND_HOST}"} \
  ${PROXY_PORT:+-e PROXY_PORT="${PROXY_PORT}"} \
  ${RAGGED_GATED_DELTA_RULE_IMPL:+-e RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL}"} \
  "${IMAGE_TAG}" \
  bash -c "
    umask 000
    rm -rf /perf_eval_results/*
    $*
  "
DOCKER_EXIT_CODE=$?
set -e

echo "--- Copying test results back to workspace for artifact upload"
cp -r /mnt/disks/persist/perf_eval_results/* perf_eval_results/ 2>/dev/null || true
# Clean up any broken symlinks on the host to prevent buildkite-agent upload failures
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

echo "[INFO] Docker finished with exit code ${DOCKER_EXIT_CODE}."
# Use Docker (running as root) to fix permissions without requiring sudo on the host
docker run --rm -v "$LOCAL_TORCHTPU_CACHE_DIR":"$LOCAL_TORCHTPU_CACHE_DIR" "${IMAGE_TAG}" chmod -R 777 "$LOCAL_TORCHTPU_CACHE_DIR" 2>/dev/null || true

if [ $DOCKER_EXIT_CODE -eq 0 ]; then
  echo "[INFO] Syncing local TorchTPU Cache back to GCS..."
  gcloud storage rsync \
    --recursive \
    --no-clobber \
    --exclude=".*_.gstmp$" \
    --no-user-output-enabled \
    "$LOCAL_TORCHTPU_CACHE_DIR" "$FINAL_CACHE_PATH" || \
    echo "[WARN] Failed to sync TorchTPU Cache back to GCS."
else
  echo "[WARN] Docker exited with non-zero code ${DOCKER_EXIT_CODE}. Skipping syncing local TorchTPU Cache back to GCS to avoid potential cache corruption."
fi

echo "--- Cleaning up pulled Docker image"
docker rmi "${IMAGE_TAG}" || true

exit $DOCKER_EXIT_CODE

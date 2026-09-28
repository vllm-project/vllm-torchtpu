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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
# Point Test Steps to the Metadata-Driven Image Tag
IMAGE_TAG=""
if command -v buildkite-agent &> /dev/null; then
  IMAGE_TAG="$(buildkite-agent meta-data get "CI_IMAGE_TAG" --default "" 2>/dev/null || true)"
fi

# Fallback if metadata is not found: construct the dual-tag <repo>-<vllm>
if [ -z "${IMAGE_TAG:-}" ]; then
  if [ "${BUILDKITE:-false}" == "true" ]; then
    echo "ERROR: CI_IMAGE_TAG metadata is missing in Buildkite CI environment." >&2
    exit 1
  fi
  VLLM_REF="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"
  if [ -z "${VLLM_REF}" ]; then
    echo "ERROR: Could not parse the vLLM pin from ${REPO_ROOT}/pyproject.toml" >&2
    exit 1
  fi
  IMAGE_TAG="${IMAGE_REPO}:${BUILDKITE_COMMIT:-latest}-${VLLM_REF}"
fi

# Buildkite expands the last `---` group whenever a log contains no `+++` group
# at all. Every group below that is pure housekeeping is therefore `~~~`
# (collapsed *and* de-emphasized, so consecutive ones fold into a single
# expander row), and the only `+++` we ever emit is the failure summary at the
# very end. Without that, a red job opened on `docker rmi` output.
echo "~~~ Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

# -q: the layer-by-layer pull progress is ~600 lines of a ~2000-line job log.
echo "~~~ Pulling Docker Image: ${IMAGE_TAG}"
docker pull -q "${IMAGE_TAG}"

HOST_CACHE_ROOT="/mnt/disks/persist"
if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  # CPU agents have no persistent disk mounted under /mnt/disks.
  HOST_CACHE_ROOT="${HOME}/.cache/vllm-torchtpu-ci"
fi

# Ensure cache directory exists on the host
mkdir -p "${HOST_CACHE_ROOT}/models"

# The container runs as root, so reset_results_dir.sh handles results the
# agent cannot remove.
bash "${SCRIPT_DIR}/reset_results_dir.sh" "${HOST_CACHE_ROOT}/perf_eval_results" "${IMAGE_TAG}"

# Ensure a clean results directory exists in the workspace
rm -rf perf_eval_results
mkdir -p perf_eval_results

# Disable exit on error temporarily to copy results back even if the test fails
set +e

CONTAINER_NAME="vllm-torchtpu-ci"
echo "~~~ Cleaning up any existing container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Container output is teed here so the failure summary can quote it without
# making the reader scroll the collapsed run group.
RUN_LOG="$(mktemp)"

trap 'docker kill "${CONTAINER_NAME}" 2>/dev/null || true; rm -f "${RUN_LOG}"' EXIT INT TERM

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
# Steps that shard one test file across parallel jobs set PYTEST_ADDOPTS so the
# marker filter also applies to the `pytest --collect-only` bktec runs to
# enumerate examples, not just to the test command. Passed through by name
# (`-e VAR`) rather than by value: the value contains spaces and quotes, which
# the unquoted `${VAR:+-e VAR=...}` expansions below would word-split.
if [ -n "${PYTEST_ADDOPTS:-}" ]; then
  TEST_SUITE_VARS+=(-e PYTEST_ADDOPTS)
fi

# BigQuery eval upload tracking & metadata variables
BQ_EVAL_VARS=(
  -e BQ_PROJECT_ID="${BQ_PROJECT_ID:-}"
  -e BQ_TABLE="${BQ_TABLE:-}"
  -e CREATED_BY="${CREATED_BY:-}"
  -e GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
  -e NIGHTLY="${NIGHTLY:-}"
  -e RUN_TYPE="${RUN_TYPE:-}"
  -e TPU_NAME="${TPU_NAME:-}"
)

DEVICE_ARGS=(--privileged --device /dev/fuse)
if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  DEVICE_ARGS=(-e JAX_PLATFORMS=cpu)
fi

echo "--- Running command in Docker container"
docker run --rm --name "${CONTAINER_NAME}" "${DEVICE_ARGS[@]}" --net=host --shm-size=64g \
  -w /root/torchtpu-vllm \
  -v "${HOST_CACHE_ROOT}/models:/local_hf_cache" \
  -v "${HOST_CACHE_ROOT}/perf_eval_results:/perf_eval_results" \
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}" \
  -e FORCE_COLOR="1" \
  -e HF_HOME=/local_hf_cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e RUN_CODE_EVAL="${RUN_CODE_EVAL:-}" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU="0.0.0" \
  -e TQDM_MININTERVAL="30" \
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \
  -e UV_NO_CACHE="1" \
  ${CAPTURE_PROFILE:+-e CAPTURE_PROFILE="${CAPTURE_PROFILE}"} \
  ${CONTAINER_TMPDIR:+-e TMPDIR="${CONTAINER_TMPDIR}" -e RAY_TMPDIR=/tmp} \
  ${EVAL_SOFT_FAIL:+-e EVAL_SOFT_FAIL="${EVAL_SOFT_FAIL}"} \
  ${MODEL_PATH:+-e MODEL_PATH="${MODEL_PATH}"} \
  ${P4D2_BIND_HOST:+-e P4D2_BIND_HOST="${P4D2_BIND_HOST}"} \
  ${PROXY_PORT:+-e PROXY_PORT="${PROXY_PORT}"} \
  ${RUN_ROOT:+-e RUN_ROOT="${RUN_ROOT}"} \
  ${SERVED_MODEL_NAME:+-e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"} \
  ${TPU_ACCELERATOR_TYPE:+-e TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE}"} \
  ${TPU_MULTIHOST_BACKEND:+-e TPU_MULTIHOST_BACKEND="${TPU_MULTIHOST_BACKEND}"} \
  ${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:+-e TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}"} \
  ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}"} \
  "${TEST_SUITE_VARS[@]}" \
  "${BQ_EVAL_VARS[@]}" \
  "${IMAGE_TAG}" \
  bash -c '
    umask 000
    rm -rf /perf_eval_results/*
    "$@"
  ' -- "$@" 2>&1 | tee "${RUN_LOG}"
DOCKER_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "~~~ Copying test results back to workspace for artifact upload"
cp -r "${HOST_CACHE_ROOT}/perf_eval_results/"* perf_eval_results/ 2>/dev/null || true
# Clean up any broken symlinks on the host to prevent buildkite-agent upload failures
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

echo "[INFO] Docker finished with exit code ${DOCKER_EXIT_CODE}."

echo "~~~ Cleaning up pulled Docker image"
docker rmi "${IMAGE_TAG}" >/dev/null 2>&1 || true

# Must be the last thing printed: `+++` is expanded by default, and its mere
# presence stops Buildkite from expanding the trailing cleanup group instead.
if [ "${DOCKER_EXIT_CODE}" -ne 0 ]; then
  echo "+++ :boom: ${BUILDKITE_LABEL:-Command} failed (exit ${DOCKER_EXIT_CODE})"
  FATAL_LINES="$(grep -m5 -E '\b[A-Za-z_]*(Error|Exception): |\[Errno [0-9]+\]|_FAIL:|^FAILED ' "${RUN_LOG}" || true)"
  if [ -n "${FATAL_LINES}" ]; then
    echo "First errors in the container output:"
    echo "${FATAL_LINES}"
    echo
  fi
  echo "Last 40 lines of container output:"
  tail -n 40 "${RUN_LOG}"
fi

exit "$DOCKER_EXIT_CODE"

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

# Multi-host Disaggregated Serving E2E on TPU (e.g., tpu_v7x_16_queue: 2 hosts x 8 chips).
# Node 0 (Head): Prefill role (PCP8, TP1) + Proxy server + Correctness smoke test.
# Node 1 (Worker): Decode role (DP8, TP1).

set -euo pipefail

# ---------------------------------------------------------------------------
# Bootstrap environment & credentials
# ---------------------------------------------------------------------------
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
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

# ---------------------------------------------------------------------------
# SSH Setup
# ---------------------------------------------------------------------------
SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then
  echo "--- Auto-generating SSH key for passwordless auth"
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

ssh_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2029
    if ssh "${SSH_OPTS[@]}" "$@"; then
      return 0
    fi
    echo "ssh attempt ${attempt}/5 to ${1##*@} failed; retrying in $((attempt * 10))s"
    sleep $((attempt * 10))
  done
  return 1
}

# ---------------------------------------------------------------------------
# Discover slice IPs from GCP metadata
# ---------------------------------------------------------------------------
if [ -z "${WORKER_IPS:-}" ]; then
  ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"
  if [ -z "$TPU_NAME" ] || [ -z "$ZONE" ]; then
    echo "ERROR: could not determine TPU_NAME/ZONE from metadata; set WORKER_IPS manually."
    exit 1
  fi
  echo "TPU_NAME=$TPU_NAME ZONE=$ZONE"
  ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")
  ALL_IPS="${ALL_IPS//;/ }"
  ALL_IPS="${ALL_IPS//,/ }"
  # shellcheck disable=SC2206
  ALL_IPS_ARRAY=($ALL_IPS)
  HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ALL_IPS_ARRAY[0]}}"
  WORKER_IPS_LIST=("${ALL_IPS_ARRAY[@]:1}")
  WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
fi
HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(hostname -I | awk '{print $1}')}"
echo "Head IP: ${HEAD_INTERNAL_IP}  Worker IPs: ${WORKER_IPS}"
IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"

if [ "${#WORKER_IPS_ARRAY[@]}" -lt 1 ]; then
  echo "ERROR: Multi-host disaggregation requires at least 1 worker host. Found 0."
  exit 1
fi
WORKER_IP="${WORKER_IPS_ARRAY[0]}"

# ---------------------------------------------------------------------------
# Results dir on host (no /mnt/disks on these agents; boot disk only)
# ---------------------------------------------------------------------------
PERSIST_ROOT="${HOME}/persist"
sudo -n rm -rf "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || rm -rf "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
mkdir -p "${PERSIST_ROOT}/perf_eval_results"
chmod 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
HOST_HF_HOME="${HOME}/hf_home"
mkdir -p "${HOST_HF_HOME}"

rm -rf perf_eval_results 2>/dev/null || true
mkdir -p perf_eval_results

# ---------------------------------------------------------------------------
# Cleanup handlers
# ---------------------------------------------------------------------------
cleanup() {
  echo "--- Cleaning up disagg containers and persist directories"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "docker rm -f disagg-worker >/dev/null 2>&1 || true; sudo -n rm -rf ~/persist/perf_eval_results >/dev/null 2>&1 || true" || true
  docker rm -f disagg-head >/dev/null 2>&1 || true
  sudo -n rm -rf "${PERSIST_ROOT}/perf_eval_results" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "--- Pre-start container, disk, and image cleanup"
cleanup
sudo -n rm -rf "${PERSIST_ROOT}/perf_eval_results" "${HOST_HF_HOME}/hub" 2>/dev/null || rm -rf "${PERSIST_ROOT}/perf_eval_results" "${HOST_HF_HOME}/hub" 2>/dev/null || true
mkdir -p "${PERSIST_ROOT}/perf_eval_results" "${HOST_HF_HOME}"
chmod 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
bash "${SCRIPT_DIR}/cleanup_docker.sh" || true

ssh_retry "${SSH_USER}@${WORKER_IP}" "mkdir -p ~/persist/perf_eval_results ~/hf_home; docker rm -f disagg-worker >/dev/null 2>&1 || true; sudo -n rm -rf ~/persist/perf_eval_results/* ~/hf_home/hub/* >/dev/null 2>&1 || rm -rf ~/persist/perf_eval_results/* ~/hf_home/hub/* >/dev/null 2>&1 || true"
ssh_retry "${SSH_USER}@${WORKER_IP}" "bash -s" < "${SCRIPT_DIR}/cleanup_docker.sh" || true

# ---------------------------------------------------------------------------
# Environment variables for Docker containers
# ---------------------------------------------------------------------------
TEST_SUITE_VARS=()
while IFS='=' read -r key _; do
  if [[ "$key" == BUILDKITE_* ]]; then
    TEST_SUITE_VARS+=(-e "$key")
  fi
done < <(env)
if [ -n "${BUILDKITE_OIDC_TOKEN_PATH:-}" ]; then
  TEST_SUITE_VARS+=(-v "$(dirname "${BUILDKITE_OIDC_TOKEN_PATH}"):$(dirname "${BUILDKITE_OIDC_TOKEN_PATH}")")
fi

BQ_EVAL_VARS=(
  -e BQ_PROJECT_ID="${BQ_PROJECT_ID:-}"
  -e BQ_TABLE="${BQ_TABLE:-}"
  -e CREATED_BY="${CREATED_BY:-}"
  -e GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
  -e NIGHTLY="${NIGHTLY:-}"
  -e RUN_TYPE="${RUN_TYPE:-}"
  -e TPU_NAME="${TPU_NAME:-}"
)

CONTAINER_ENV_COMMON=(
  -e VLLM_DISABLE_COMPILE_CACHE=1
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e FORCE_COLOR="1"
  -e TQDM_MININTERVAL="30"
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU="0.0.0"
  -e TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
  -e PREFILL_HOST="${HEAD_INTERNAL_IP}"
  -e DECODE_HOST="${WORKER_IP}"
  -e PREFILL_PORT="${PREFILL_PORT:-8400}"
  -e DECODE_PORT="${DECODE_PORT:-9400}"
  -e PROXY_PORT="${PROXY_PORT:-8000}"
  -e PREFILL_PCP="${PREFILL_PCP:-8}"
  -e DECODE_DP="${DECODE_DP:-8}"
  -e PREFILL_TP="${PREFILL_TP:-1}"
  -e DECODE_TP="${DECODE_TP:-1}"
  -e PREFILL_CTRL_PORT="${PREFILL_CTRL_PORT:-27000}"
  -e DECODE_CTRL_PORT="${DECODE_CTRL_PORT:-28000}"
  -e PREFILL_TPU_KV_TRANSFER_PORT="${PREFILL_TPU_KV_TRANSFER_PORT:-9100}"
  -e DECODE_TPU_KV_TRANSFER_PORT="${DECODE_TPU_KV_TRANSFER_PORT:-9200}"
  -e TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-${PREFILL_PCP:-8}}"
  -e PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-128}"
  -e STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
  -e RUN_ROOT="/perf_eval_results/qwen35_p8d8_disagg_ci"
  ${MODEL_PATH:+-e MODEL_PATH="${MODEL_PATH}"}
  ${SERVED_MODEL_NAME:+-e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"}
  ${GPU_MEMORY_UTILIZATION:+-e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"}
  ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN="${MAX_MODEL_LEN}"}
  ${MAX_NUM_BATCHED_TOKENS:+-e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"}
  ${MAX_NUM_SEQS:+-e MAX_NUM_SEQS="${MAX_NUM_SEQS}"}
  ${PREFILL_COMPILE_SIZES:+-e PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES}"}
  ${DECODE_COMPILE_SIZES:+-e DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256,4096}"}
  ${NUM_GPU_BLOCKS_OVERRIDE:+-e NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE}"}
  ${RUN_PREFIX_CACHE_E2E_DIVERGENCE:+-e RUN_PREFIX_CACHE_E2E_DIVERGENCE="${RUN_PREFIX_CACHE_E2E_DIVERGENCE}"}
)

# ---------------------------------------------------------------------------
# Start Worker container (ROLE=decode)
# ---------------------------------------------------------------------------
echo "--- Starting disagg-worker (ROLE=decode) on ${WORKER_IP}"
ssh_retry "${SSH_USER}@${WORKER_IP}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker pull ${IMAGE_TAG} || true"

# shellcheck disable=SC2029
ssh_retry "${SSH_USER}@${WORKER_IP}" "docker run -d --name disagg-worker --privileged --net=host --shm-size=128g --device /dev/fuse -w /root/torchtpu-vllm -v \${HOME}/hf_home:/root/.cache/huggingface -v \${HOME}/persist/perf_eval_results:/perf_eval_results -e ROLE=decode ${CONTAINER_ENV_COMMON[*]} ${IMAGE_TAG} bash -c 'umask 000; rm -rf /perf_eval_results/*; bash ./scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh'"

# ---------------------------------------------------------------------------
# Start Head container (ROLE=head)
# ---------------------------------------------------------------------------
echo "--- Starting disagg-head (ROLE=head) on ${HEAD_INTERNAL_IP}"
docker pull "${IMAGE_TAG}" || true

set +e
docker run --name disagg-head --privileged --net=host --shm-size=128g --device /dev/fuse \
  -w /root/torchtpu-vllm \
  -v "${HOST_HF_HOME}:/root/.cache/huggingface" \
  -v "${PERSIST_ROOT}/perf_eval_results:/perf_eval_results" \
  -e ROLE=head \
  "${CONTAINER_ENV_COMMON[@]}" \
  "${TEST_SUITE_VARS[@]}" \
  "${BQ_EVAL_VARS[@]}" \
  "${IMAGE_TAG}" \
  bash -c 'umask 000; rm -rf /perf_eval_results/*; bash ./scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh'
HEAD_EXIT_CODE=$?
set -e

# ---------------------------------------------------------------------------
# Collect logs and artifacts
# ---------------------------------------------------------------------------
echo "--- Collecting server logs"
docker logs disagg-head --tail 2000 > "${PERSIST_ROOT}/perf_eval_results/head.log" 2>&1 || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "docker logs disagg-worker --tail 2000" \
  > "${PERSIST_ROOT}/perf_eval_results/worker_decode.log" 2>&1 || true
sudo -n chmod -R 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "sudo -n chmod -R 777 ~/persist/perf_eval_results 2>/dev/null || true; tar -czf - -C ~/persist/perf_eval_results . 2>/dev/null" \
  | tar -xzf - -C "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
sudo -n chmod -R 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true

echo "--- Copying results back for artifact upload"
cp -r "${PERSIST_ROOT}/perf_eval_results"/* perf_eval_results/ 2>/dev/null || true
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

# Clean up persist directories to prevent permissions conflicts in subsequent jobs
sudo -n rm -rf "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "sudo -n rm -rf ~/persist/perf_eval_results 2>/dev/null || true" || true

echo "--- Disaggregation multi-host finished with exit code ${HEAD_EXIT_CODE}"
exit "${HEAD_EXIT_CODE}"

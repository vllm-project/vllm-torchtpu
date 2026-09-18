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

# Run a command or benchmark across a multi-host TPU slice
# (e.g. tpu_v7x_16_queue = 2 hosts x 8 chips).
#
# Supports two distributed execution backends:
#   1. ray (default): Starts Ray head on head node and Ray worker on each
#      worker host via scripts/multihost/run_cluster.sh with
#      TPU_MULTIHOST_BACKEND=ray, then executes the given command in the
#      head container.
#   2. mp: Starts independent `vllm serve` processes directly on head and
#      worker hosts via scripts/multihost/run_cluster_mp.sh with
#      TPU_MULTIHOST_BACKEND=mp (src/vllm_torchtpu/distributed/tpu_mp_multihost.py).
#
# Usage:
#   run_multihost.sh [--backend ray] <command_to_run_in_head_container...>
#   run_multihost.sh --backend mp <config_name> [extra run_eval_flow.sh args...]
# shellcheck disable=SC2034,SC2317,SC1091
set -euo pipefail

# Parse --backend flag if provided
BACKEND="${TPU_MULTIHOST_BACKEND:-ray}"
if [ "${1:-}" = "--backend" ]; then
  BACKEND="$2"
  shift 2
fi

if [ "$#" -lt 1 ]; then
  if [ "$BACKEND" = "mp" ]; then
    echo "ERROR: Usage: $0 [--backend mp] <config_name> [extra run_eval_flow.sh args...]"
  else
    echo "ERROR: Usage: $0 [--backend ray] <command_to_run_in_head_container...>"
  fi
  exit 1
fi

# Set HF_TOKEN from GCP Secret Manager if not already in /etc/environment
# (same bootstrap as run_in_docker.sh).
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_DIR}"
RUN_CLUSTER="${REPO_DIR}/scripts/multihost/run_cluster.sh"
RUN_CLUSTER_MP="${REPO_DIR}/scripts/multihost/run_cluster_mp.sh"

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

SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then
  echo "~~~ Auto-generating SSH key for passwordless auth"
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

# Worker-host ssh sessions intermittently drop ("client_loop: send
# disconnect: Broken pipe", exit 255) — retry setup commands with backoff.
ssh_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2029  # callers pass fully-built remote commands
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
NUM_HOSTS=$(( ${#WORKER_IPS_ARRAY[@]} + 1 ))

# ---------------------------------------------------------------------------
# Results dir on host (no /mnt/disks on these agents; boot disk only)
# ---------------------------------------------------------------------------
PERSIST_ROOT="${HOME}/persist"
# Containers write here as root, so never assume the agent can rm the leftovers
# of whatever ran last -- see reset_results_dir.sh.
bash "${SCRIPT_DIR}/reset_results_dir.sh" "${PERSIST_ROOT}/perf_eval_results" "${IMAGE_TAG}"
HOST_HF_HOME="${HOME}/hf_home"   # tokenizer/config only; weights stream from GCS
mkdir -p "${HOST_HF_HOME}"
rm -rf perf_eval_results
mkdir -p perf_eval_results

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
# Buildkite expands the last `---` group whenever a log contains no `+++` group
# at all. Housekeeping groups are therefore `~~~` (collapsed and de-emphasized,
# so consecutive ones fold into one expander row) and the only `+++` we emit is
# the failure summary at the very end -- otherwise a red job opens on cleanup
# output rather than on the error.
cleanup() {
  echo "~~~ Cleaning up ${BACKEND}-backend containers"
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker rm -f node >/dev/null 2>&1 || true" || true
  done
  docker rm -f node >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

# Free disk before pulling this commit's image (100G boot disk only).
echo "~~~ Cleaning up old Docker images"
bash "${SCRIPT_DIR}/cleanup_docker.sh" || true

# Worker hosts pull the same ~9G image every build but never pruned it, so they
# filled up while the head stayed healthy: tpu7x-16-ci-1 worker 1 reached 98%
# with six stale images on it. Prune them the same way we prune the head.
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "bash -s" < "${SCRIPT_DIR}/cleanup_docker.sh" || true
done


# The backends tee the command they run in the head container to MULTIHOST_RUN_LOG
# so the failure summary below can quote it.
export MULTIHOST_RUN_LOG="${MULTIHOST_RUN_LOG:-perf_eval_results/multihost.log}"

# Test suite and BigQuery tracking env vars
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
  -e TPU_MULTIHOST_BACKEND="${BACKEND}"
  -e TPU_SKIP_MDS_QUERY=1
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}"
  -e FORCE_COLOR="1"
  -e TQDM_MININTERVAL="30"
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU="0.0.0"
  ${USE_MOE_SPARSE_CORE:+-e USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE}"}
  ${TPU_ACCELERATOR_TYPE:+-e TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE}"}
  ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}"}
)

CONTAINER_ENV_HEAD=(
  "${CONTAINER_ENV_COMMON[@]}"
  ${MODEL_URI:+-e MODEL_URI="${MODEL_URI}"}
  -v "${PERSIST_ROOT}/perf_eval_results:/perf_eval_results"
  "${TEST_SUITE_VARS[@]}"
  "${BQ_EVAL_VARS[@]}"
)

echo "~~~ Pre-start scorched-earth cleanup"
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
if [ "$BACKEND" = "ray" ]; then
  sudo -n pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || true
fi
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true$( [ "$BACKEND" = "ray" ] && echo "; sudo -n pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || true" )" || true
done

if [ "$BACKEND" = "ray" ]; then
  # shellcheck source=.buildkite/scripts/run_multihost_ray.sh
  source "${SCRIPT_DIR}/run_multihost_ray.sh"
  run_ray_multihost "$@"
  EXIT_CODE=$?
elif [ "$BACKEND" = "mp" ]; then
  # shellcheck source=.buildkite/scripts/run_multihost_mp.sh
  source "${SCRIPT_DIR}/run_multihost_mp.sh"
  run_mp_multihost "$@"
  EXIT_CODE=$?
else
  echo "ERROR: Unknown backend: ${BACKEND}. Supported backends: ray, mp"
  exit 1
fi

echo "~~~ Copying results back for artifact upload"
cp -r "${PERSIST_ROOT}/perf_eval_results"/* perf_eval_results/ 2>/dev/null || true
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

# Must be the last thing printed: `+++` is expanded by default, and its mere
# presence stops Buildkite from expanding the trailing cleanup group instead.
#
# An empty run log means the backend died during cluster bring-up, before the
# command ever ran. There is nothing to summarise, so skip the `+++` on purpose
# and let Buildkite's own fallback expand the last `---` group -- which in that
# case is the bring-up step that actually failed.
if [ "${EXIT_CODE}" -ne 0 ] && [ -n "${MULTIHOST_RUN_LOG:-}" ] && [ -s "${MULTIHOST_RUN_LOG}" ]; then
  echo "+++ :boom: ${BUILDKITE_LABEL:-Command} failed (exit ${EXIT_CODE})"
  FATAL_LINES="$(grep -m5 -E '\b[A-Za-z_]*(Error|Exception): |\[Errno [0-9]+\]|_FAIL:|^FAILED ' "${MULTIHOST_RUN_LOG}" || true)"
  if [ -n "${FATAL_LINES}" ]; then
    echo "First errors in the head-container output:"
    echo "${FATAL_LINES}"
    echo
  fi
  echo "Last 40 lines of head-container output:"
  tail -n 40 "${MULTIHOST_RUN_LOG}"
fi
if [[ "${MULTIHOST_RUN_LOG:-}" == /tmp/* ]]; then
  rm -f "${MULTIHOST_RUN_LOG}"
fi

exit "${EXIT_CODE}"

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

# Run a benchmarking config across a multi-host TPU slice using the `mp`
# distributed-executor backend (e.g. tpu_v7x_16_queue = 2 hosts x 8 chips),
# instead of run_multihost.sh's Ray cluster.
#
# There is no cluster daemon to join here: each host runs its own
# independent `vllm serve` process (head vs. `--headless
# --data-parallel-start-rank`), and vllm-torchtpu's mp-multihost bootstrap
# (src/vllm_torchtpu/distributed/tpu_mp_multihost.py) handles cross-host
# rendezvous. This regression-tests that path directly, rather than the Ray
# executor's separate implementation.
#
# The Buildkite agent runs on the slice's head host. This script:
#   1. Discovers the worker host IPs from GCP metadata (same mechanism as
#      run_multihost.sh).
#   2. Sources the named benchmarking config and derives each host's
#      `vllm serve` command (head owns DP ranks
#      [0, DATA_PARALLELISM_LOCAL), each worker owns the next contiguous
#      block).
#   3. Starts a container on each worker (via ssh) and the head container
#      locally, each running its own `vllm serve` directly (no Ray).
#   4. Waits for the head's /health, then runs run_eval_flow.sh against it
#      inside the head container (weights stream from MODEL_URI via
#      runai_streamer -- these hosts have no attached data disk).
#   5. Copies /perf_eval_results back for artifact upload.
#
# Usage: run_multihost_mp.sh <config_name> [extra run_eval_flow.sh args...]
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "ERROR: Usage: $0 <config_name> [extra run_eval_flow.sh args...]"
  exit 1
fi
CONFIG_NAME="$1"
shift
EXTRA_EVAL_FLOW_ARGS=("$@")

# Set HF_TOKEN from GCP Secret Manager if not already in /etc/environment
# (same bootstrap as run_in_docker.sh / run_multihost.sh).
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${VLLM_TORCHTPU_IMAGE_TAG:-${IMAGE_REPO}:${COMMIT_HASH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_CLUSTER_MP="${REPO_DIR}/scripts/multihost/run_cluster_mp.sh"
CONFIG_FILE="${REPO_DIR}/scripts/vllm/benchmarking/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

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
# Discover slice IPs from GCP metadata (identical to run_multihost.sh)
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
# Source the config to derive server flags
# ---------------------------------------------------------------------------
MODEL_URI="${MODEL_URI:-}"
# This host's DP share of DATA_PARALLELISM (e.g. 8 chips/host x 2 hosts =
# 16). Topology-specific, not benchmark-specific, so it's meant to come from
# the pipeline step's env rather than being hardcoded into a shared config
# -- preserve a pre-set value the same way MODEL_URI does, instead of
# clobbering it before the config below can (not) set it.
DATA_PARALLELISM_LOCAL="${DATA_PARALLELISM_LOCAL:-}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
GPU_MEMORY_UTILIZATION="0.95"
KV_CACHE_DTYPE="fp8"
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
ATTENTION_BACKEND="CUSTOM"
ENABLE_PREFIX_CACHING=false
EXTRA_SERVE_ARGS=""
SERVER_READY_WAIT_MIN=180
# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [ -z "$MODEL" ]; then
  echo "ERROR: Config must set MODEL"
  exit 1
fi
if [ -z "$DATA_PARALLELISM_LOCAL" ]; then
  echo "ERROR: Config must set DATA_PARALLELISM_LOCAL (this host's DP share) for multi-host mp launch"
  exit 1
fi
if [ "$((DATA_PARALLELISM_LOCAL * NUM_HOSTS))" -ne "$DATA_PARALLELISM" ]; then
  echo "ERROR: DATA_PARALLELISM_LOCAL ($DATA_PARALLELISM_LOCAL) x NUM_HOSTS ($NUM_HOSTS) != DATA_PARALLELISM ($DATA_PARALLELISM)"
  exit 1
fi

PORT="${PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"

# Same server-flag construction as run_benchmarks.sh's start_vllm_server, so
# the externally-launched server here matches what a single-host config run
# would have produced.
extra_args=""
if [ "$ENABLE_EP" = "true" ]; then
  extra_args="--enable-expert-parallel"
fi
if [ -n "$QUANTIZATION" ]; then
  extra_args="$extra_args --quantization $QUANTIZATION"
fi
extra_args="$extra_args --attention-backend $ATTENTION_BACKEND"
if [ -n "$EXTRA_SERVE_ARGS" ]; then
  extra_args="$extra_args $EXTRA_SERVE_ARGS"
fi
prefix_caching_flag="--no-enable-prefix-caching"
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
  prefix_caching_flag="--enable-prefix-caching"
fi
serve_target="$MODEL"
if [ -n "$MODEL_URI" ]; then
  serve_target="$MODEL_URI"
  extra_args="$extra_args --load-format runai_streamer --served-model-name $MODEL"
fi

# Built as an array (word-split from plain variable expansions, so JSON-ish
# values like EXTRA_SERVE_ARGS's --limit-mm-per-prompt {"image":0,"video":0}
# keep their literal quotes/braces here) rather than one flat string: the
# head/worker commands below cross an extra `bash -c "$CMD"` re-parse inside
# the container (run_cluster_mp.sh's entrypoint override), and re-parsing a
# flat string as fresh shell source would brace-expand the unquoted comma in
# that JSON blob into two separate args. printf %q re-quotes each array
# element so that final re-parse reconstructs it byte-for-byte instead.
# shellcheck disable=SC2206  # intentional word-splitting of extra_args
COMMON_SERVE_ARGS_ARR=(
  "${serve_target}"
  --tensor-parallel-size="${TENSOR_PARALLELISM}"
  --data-parallel-size="${DATA_PARALLELISM}"
  --data-parallel-size-local="${DATA_PARALLELISM_LOCAL}"
  --data-parallel-address="${HEAD_INTERNAL_IP}"
  --data-parallel-rpc-port="${DP_RPC_PORT}"
  --max-model-len="${MAX_MODEL_LEN}"
  --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs="${MAX_NUM_SEQS}"
  --async-scheduling
  "${prefix_caching_flag}"
  --gpu-memory-utilization="${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype="${KV_CACHE_DTYPE}"
  ${extra_args}
)

HEAD_SERVE_CMD=$(printf '%q ' vllm serve "${COMMON_SERVE_ARGS_ARR[@]}" --data-parallel-start-rank=0 --host 0.0.0.0 --port "${PORT}")

# --headless nodes start no API servers, so head-only serving flags a config
# may carry in EXTRA_SERVE_ARGS (e.g. --api-server-count=N, tuned for the
# head's API frontend count) are invalid there -- vllm's CLI rejects
# --api-server-count outright when --headless is also set. Strip it before
# building the worker command below.
WORKER_SERVE_ARGS_ARR=()
for arg in "${COMMON_SERVE_ARGS_ARR[@]}"; do
  case "$arg" in
    --api-server-count=*) continue ;;
  esac
  WORKER_SERVE_ARGS_ARR+=("$arg")
done

# ---------------------------------------------------------------------------
# Results dir on host (no /mnt/disks on these agents; boot disk only)
# ---------------------------------------------------------------------------
PERSIST_ROOT="${HOME}/persist"
rm -rf "${PERSIST_ROOT}/perf_eval_results"
mkdir -p "${PERSIST_ROOT}/perf_eval_results"
chmod 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
HOST_HF_HOME="${HOME}/hf_home"
mkdir -p "${HOST_HF_HOME}"
rm -rf perf_eval_results
mkdir -p perf_eval_results

# ---------------------------------------------------------------------------
# Cleanup on exit: stop containers on all hosts
# ---------------------------------------------------------------------------
cleanup() {
  echo "--- Cleaning up mp-backend containers"
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker rm -f node >/dev/null 2>&1 || true" || true
  done
  docker rm -f node >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

echo "--- Cleaning up old Docker images"
bash "${SCRIPT_DIR}/cleanup_docker.sh" || true

CONTAINER_ENV_COMMON=(
  -e VLLM_DISABLE_COMPILE_CACHE=1
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e TPU_MULTIHOST_BACKEND=mp
  -e TPU_SKIP_MDS_QUERY=1
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}"
  -e FORCE_COLOR="1"
  -e TQDM_MININTERVAL="30"
  -e SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"
  -e BUILDKITE_BRANCH="${BUILDKITE_BRANCH:-}"
  -e BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST:-}"
  ${USE_MOE_SPARSE_CORE:+-e USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE}"}
  ${TPU_ACCELERATOR_TYPE:+-e TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE}"}
  ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}"}
)

echo "--- Pre-start scorched-earth cleanup"
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true" || true
done

# ---------------------------------------------------------------------------
# Start worker containers over ssh (each owns the next contiguous DP block)
# ---------------------------------------------------------------------------
worker_idx=0
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  worker_idx=$((worker_idx + 1))
  start_rank=$((worker_idx * DATA_PARALLELISM_LOCAL))
  worker_serve_cmd=$(printf '%q ' vllm serve "${WORKER_SERVE_ARGS_ARR[@]}" --headless --data-parallel-start-rank="${start_rank}")
  echo "--- Starting mp worker ${worker_idx} on ${worker_ip} (data-parallel-start-rank=${start_rank})"
  ssh_retry "${SSH_USER}@${worker_ip}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true; mkdir -p ~/multihost ~/hf_home"
  base64 < "${RUN_CLUSTER_MP}" > /tmp/run_cluster_mp.b64
  ssh_retry "${SSH_USER}@${worker_ip}" "base64 -d > ~/multihost/run_cluster_mp.sh" < /tmp/run_cluster_mp.b64
  # shellcheck disable=SC2029
  (
    for _attempt in 1 2 3; do
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
        "docker rm -f node >/dev/null 2>&1 || true; bash ~/multihost/run_cluster_mp.sh '${IMAGE_TAG}' \"\$HOME/hf_home\" '${worker_serve_cmd}' ${CONTAINER_ENV_COMMON[*]}" \
        && break
      echo "worker ${worker_idx} ssh dropped (attempt ${_attempt}/3); restarting in 15s"
      sleep 15
    done
  ) &
done

# ---------------------------------------------------------------------------
# Start head container locally (run_cluster_mp.sh blocks; background it)
# ---------------------------------------------------------------------------
CONTAINER_ENV_HEAD=(
  "${CONTAINER_ENV_COMMON[@]}"
  ${MODEL_URI:+-e MODEL_URI="${MODEL_URI}"}
  -v "${PERSIST_ROOT}/perf_eval_results:/perf_eval_results"
)
echo "--- Starting mp head on ${HEAD_INTERNAL_IP}"
bash "${RUN_CLUSTER_MP}" "${IMAGE_TAG}" "${HOST_HF_HOME}" "${HEAD_SERVE_CMD}" "${CONTAINER_ENV_HEAD[@]}" &

# ---------------------------------------------------------------------------
# Wait for the head server to become healthy
# ---------------------------------------------------------------------------
echo "--- Waiting for head server /health"
deadline=$((SECONDS + SERVER_READY_WAIT_MIN * 60))
head_up=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if docker exec node curl -s -o /dev/null --connect-timeout 1 "http://localhost:${PORT}/health" 2>/dev/null; then
    head_up=1
    break
  fi
  sleep 10
done
if [ "$head_up" != "1" ]; then
  echo "ERROR: head server did not become healthy within ${SERVER_READY_WAIT_MIN} minutes"
  docker logs node --tail 200 2>&1 || true
  exit 1
fi
echo "Head server is healthy."

# ---------------------------------------------------------------------------
# Run the benchmark/eval flow inside the head container, against the
# already-running server (--host skips run_benchmarks.sh's own server
# launch, per its --host/START_SERVER handling).
# ---------------------------------------------------------------------------
set +e
echo "--- Running eval flow in head container for config=${CONFIG_NAME}"
docker exec -w /root/torchtpu-vllm node bash -c '
  umask 000
  rm -rf /perf_eval_results/*
  bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config "$1" --host localhost --port "$2" --results-dir /perf_eval_results/"$1" "${@:3}"
' -- "${CONFIG_NAME}" "${PORT}" "${EXTRA_EVAL_FLOW_ARGS[@]}"
EXIT_CODE=$?
set -e

echo "--- Copying results back for artifact upload"
cp -r "${PERSIST_ROOT}/perf_eval_results"/* perf_eval_results/ 2>/dev/null || true
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

echo "--- Collecting server logs"
docker logs node --tail 2000 > perf_eval_results/head_server.log 2>&1 || true
widx=0
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  widx=$((widx + 1))
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker logs node --tail 2000" \
    > "perf_eval_results/worker${widx}_server.log" 2>&1 || true
done

exit "${EXIT_CODE}"

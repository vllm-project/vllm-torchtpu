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

# Backend implementation for MP multi-host distributed execution.
# Can be sourced by run_multihost.sh or executed directly as a wrapper.
# shellcheck disable=SC2154

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If executed directly instead of sourced, delegate to run_multihost.sh --backend mp
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exec bash "${SCRIPT_DIR}/run_multihost.sh" --backend mp "$@"
fi

run_mp_multihost() {
  if [ "$#" -lt 1 ]; then
    echo "ERROR: Usage: run_multihost.sh --backend mp <config_name> [extra run_eval_flow.sh args...]"
    return 1
  fi
  CONFIG_NAME="$1"
  shift
  EXTRA_EVAL_FLOW_ARGS=("$@")

  CONFIG_FILE="${REPO_DIR}/scripts/vllm/benchmarking/configs/${CONFIG_NAME}.sh"
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    return 1
  fi

  MODEL_URI="${MODEL_URI:-}"
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
    return 1
  fi
  if [ -z "$DATA_PARALLELISM_LOCAL" ]; then
    echo "ERROR: Config must set DATA_PARALLELISM_LOCAL (this host's DP share) for multi-host mp launch"
    return 1
  fi
  if [ "$((DATA_PARALLELISM_LOCAL * NUM_HOSTS))" -ne "$DATA_PARALLELISM" ]; then
    echo "ERROR: DATA_PARALLELISM_LOCAL ($DATA_PARALLELISM_LOCAL) x NUM_HOSTS ($NUM_HOSTS) != DATA_PARALLELISM ($DATA_PARALLELISM)"
    return 1
  fi

  PORT="${PORT:-8000}"
  DP_RPC_PORT="${DP_RPC_PORT:-29500}"

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

  # shellcheck disable=SC2206
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

  WORKER_SERVE_ARGS_ARR=()
  for arg in "${COMMON_SERVE_ARGS_ARR[@]}"; do
    case "$arg" in
      --api-server-count=*) continue ;;
    esac
    WORKER_SERVE_ARGS_ARR+=("$arg")
  done

  # Start worker containers over ssh
  worker_idx=0
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    worker_idx=$((worker_idx + 1))
    start_rank=$((worker_idx * DATA_PARALLELISM_LOCAL))
    worker_serve_cmd=$(printf '%q ' vllm serve "${WORKER_SERVE_ARGS_ARR[@]}" --headless --data-parallel-start-rank="${start_rank}")
    echo "--- Starting mp worker ${worker_idx} on ${worker_ip} (data-parallel-start-rank=${start_rank})"
    ssh_retry "${SSH_USER}@${worker_ip}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true; mkdir -p ~/multihost ~/hf_home" || return 1
    base64 < "${RUN_CLUSTER_MP}" > /tmp/run_cluster_mp.b64 || return 1
    ssh_retry "${SSH_USER}@${worker_ip}" "base64 -d > ~/multihost/run_cluster_mp.sh" < /tmp/run_cluster_mp.b64 || return 1
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

  # Start head container locally
  echo "--- Starting mp head on ${HEAD_INTERNAL_IP}"
  bash "${RUN_CLUSTER_MP}" "${IMAGE_TAG}" "${HOST_HF_HOME}" "${HEAD_SERVE_CMD}" "${CONTAINER_ENV_HEAD[@]}" &

  # Wait for head server /health
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
    return 1
  fi
  echo "Head server is healthy."

  # Run the benchmark/eval flow inside the head container
  set +e
  echo "--- Running eval flow in head container for config=${CONFIG_NAME}"
  docker exec -w /root/torchtpu-vllm node bash -c '
    umask 000
    rm -rf /perf_eval_results/*
    bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config "$1" --host localhost --port "$2" --results-dir /perf_eval_results/"$1" "${@:3}"
  ' -- "${CONFIG_NAME}" "${PORT}" "${EXTRA_EVAL_FLOW_ARGS[@]}" 2>&1 | tee "${MULTIHOST_RUN_LOG}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  echo "~~~ Collecting server logs"
  docker logs node --tail 2000 > perf_eval_results/head_server.log 2>&1 || true
  widx=0
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    widx=$((widx + 1))
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker logs node --tail 2000" \
      > "perf_eval_results/worker${widx}_server.log" 2>&1 || true
  done

  return "${EXIT_CODE}"
}

#!/usr/bin/env bash
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

port_pids() {
  local port="$1"
  ss -ltnp 2>/dev/null \
    | awk -v port=":${port}" '$4 ~ port "$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | sort -u
}

stop_port_listeners() {
  local pids=()
  local port pid
  for port in "$@"; do
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(port_pids "${port}")
  done
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u)
  echo "Stopping existing listeners: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    local alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive+=("${pid}")
      fi
    done
    if [[ "${#alive[@]}" -eq 0 ]]; then
      return 0
    fi
    sleep 0.5
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
}

require_free_ports() {
  local busy=0
  local port pids
  for port in "$@"; do
    pids="$(port_pids "${port}" | tr '\n' ' ')"
    if [[ -n "${pids}" ]]; then
      echo "Port ${port} is busy: ${pids}" >&2
      busy=1
    fi
  done
  return "${busy}"
}

get_local_ip() {
  if [[ -n "${LOCAL_IP:-}" ]]; then
    echo "${LOCAL_IP}"
    return 0
  fi
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "${ip}" ]]; then
    echo "${ip}"
    return 0
  fi
  echo "127.0.0.1"
}

# ---------------------------------------------------------------------------
# Configuration and defaults
# ---------------------------------------------------------------------------
ROLE="${ROLE:-all}"
RUN_ROOT="${RUN_ROOT:-${HOME}/pd_disagg_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/qwen35_disagg_${ROLE}_${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"

LOCAL_ROUTABLE_IP="$(get_local_ip)"
HOST="${HOST:-${LOCAL_ROUTABLE_IP}}"

# Prefill configuration
PREFILL_HOST="${PREFILL_HOST:-${HOST}}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
PREFILL_CTRL_PORT="${PREFILL_CTRL_PORT:-27000}"
PREFILL_TPU_KV_TRANSFER_PORT="${PREFILL_TPU_KV_TRANSFER_PORT:-9100}"
PREFILL_PCP="${PREFILL_PCP:-8}"
PREFILL_TP="${PREFILL_TP:-1}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-128}"
PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-4096}"
PREFILL_NUM_GPU_BLOCKS_OVERRIDE="${PREFILL_NUM_GPU_BLOCKS_OVERRIDE:-${NUM_GPU_BLOCKS_OVERRIDE-}}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
PREFILL_SB_PORT_BASE="${PREFILL_SB_PORT_BASE:-29500}"
PREFILL_ADVERTISE_HOST="${PREFILL_ADVERTISE_HOST:-${TPU_RAIDEN_ADVERTISE_HOST:-${LOCAL_ROUTABLE_IP}}}"
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET="${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET:-0}"
default_prefill_speculative_config='{"method":"mtp","num_speculative_tokens":1}'
PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-${default_prefill_speculative_config}}"

# Decode configuration
DECODE_HOST="${DECODE_HOST:-${HOST}}"
DECODE_PORT="${DECODE_PORT:-9400}"
DECODE_CTRL_PORT="${DECODE_CTRL_PORT:-28000}"
DECODE_TPU_KV_TRANSFER_PORT="${DECODE_TPU_KV_TRANSFER_PORT:-9200}"
DECODE_DP="${DECODE_DP:-8}"
DECODE_TP="${DECODE_TP:-1}"
DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-4096}"
DECODE_NUM_GPU_BLOCKS_OVERRIDE="${DECODE_NUM_GPU_BLOCKS_OVERRIDE:-${NUM_GPU_BLOCKS_OVERRIDE-}}"
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256,4096}"
DECODE_SB_PORT_BASE="${DECODE_SB_PORT_BASE:-29510}"
DECODE_DP_MASTER_PORT="${DECODE_DP_MASTER_PORT:-29520}"
DECODE_ADVERTISE_HOST="${DECODE_ADVERTISE_HOST:-${TPU_RAIDEN_ADVERTISE_HOST:-${LOCAL_ROUTABLE_IP}}}"
if [[ "${ROLE}" == "all" && -z "${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET+x}" ]]; then
  DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET=$((PREFILL_TP * PREFILL_PCP))
else
  DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET="${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET:-0}"
fi
default_decode_speculative_config='{"method":"mtp","num_speculative_tokens":3}'
DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-${default_decode_speculative_config}}"

# Shared / common settings
KV_PORT="${KV_PORT:-14579}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-9600}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
TPU_KV_SHM_POOL_GB="${TPU_KV_SHM_POOL_GB:-8}"
TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-${PREFILL_PCP:-8}}"
RESTART_EXISTING="${RESTART_EXISTING:-1}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
LOAD_FORMAT="${LOAD_FORMAT:-}"

mkdir -p "${RUN_DIR}/logs"

# ---------------------------------------------------------------------------
# Python environment resolution
# ---------------------------------------------------------------------------
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Could not find an executable python3/python in PATH." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Validate parameters
# ---------------------------------------------------------------------------
case "${ROLE}" in
  prefill|decode|all) ;;
  *)
    echo "ERROR: Invalid ROLE: ${ROLE}. Must be 'prefill', 'decode', or 'all'." >&2
    exit 2
    ;;
esac

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH directory does not exist: ${MODEL_PATH}" >&2
    exit 2
  fi
fi

for cs_var in PREFILL_COMPILE_SIZES DECODE_COMPILE_SIZES; do
  cs_val="${!cs_var}"
  if [[ ! "${cs_val}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: ${cs_var} must be a comma-separated integer list: ${cs_val}" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# Port management and conflict resolution
# ---------------------------------------------------------------------------
managed_ports=()
prefill_sb_workers=$((PREFILL_TP * PREFILL_PCP))
decode_sb_workers=$((DECODE_TP * DECODE_DP))

prefill_sb_addresses=""
for ((i = 0; i < prefill_sb_workers; i++)); do
  prefill_sb_addresses+="${prefill_sb_addresses:+,}localhost:$((PREFILL_SB_PORT_BASE + i))"
done

decode_sb_addresses=""
for ((i = 0; i < decode_sb_workers; i++)); do
  decode_sb_addresses+="${decode_sb_addresses:+,}localhost:$((DECODE_SB_PORT_BASE + i))"
done

if [[ "${ROLE}" == "prefill" || "${ROLE}" == "all" ]]; then
  managed_ports+=("${PREFILL_PORT}" "${TPU_SIDE_CHANNEL_PORT}" "${KV_PORT}")
  for ((i = 0; i < prefill_sb_workers; i++)); do
    managed_ports+=("$((PREFILL_SB_PORT_BASE + i))")
    managed_ports+=("$((PREFILL_TPU_KV_TRANSFER_PORT + i))")
    managed_ports+=("$((PREFILL_CTRL_PORT + i))")
    managed_ports+=("$((PREFILL_CTRL_PORT + 100 + i))")
  done
fi

if [[ "${ROLE}" == "decode" || "${ROLE}" == "all" ]]; then
  managed_ports+=("${DECODE_PORT}" "${TPU_SIDE_CHANNEL_PORT}" "${KV_PORT}" "${DECODE_DP_MASTER_PORT}")
  for ((i = 0; i < decode_sb_workers; i++)); do
    managed_ports+=("$((DECODE_SB_PORT_BASE + i))")
    managed_ports+=("$((DECODE_TPU_KV_TRANSFER_PORT + i))")
    managed_ports+=("$((DECODE_CTRL_PORT + i))")
    managed_ports+=("$((DECODE_CTRL_PORT + 100 + i))")
  done
fi

if [[ "${#managed_ports[@]}" -gt 0 ]]; then
  mapfile -t managed_ports < <(printf '%s\n' "${managed_ports[@]}" | sort -u)
  if [[ "${RESTART_EXISTING}" == "1" ]]; then
    stop_port_listeners "${managed_ports[@]}"
  else
    require_free_ports "${managed_ports[@]}"
  fi
fi

# ---------------------------------------------------------------------------
# Compilation configs
# ---------------------------------------------------------------------------
prefill_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":['"${PREFILL_COMPILE_SIZES}"'],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
decode_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":['"${DECODE_COMPILE_SIZES}"'],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

# ---------------------------------------------------------------------------
# Build common server CLI arguments
# ---------------------------------------------------------------------------
common_args=(
  --model "${MODEL_PATH}"
  --trust-remote-code
  --seed 42
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype fp8
  --language-model-only
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --no-disable-hybrid-kv-cache-manager
  --attention-backend "${ATTENTION_BACKEND}"
  --mamba-cache-mode "${MAMBA_CACHE_MODE}"
  --enable-prompt-tokens-details
  --no-enable-log-requests
)

if [[ "${SERVED_MODEL_NAME}" != "${MODEL_PATH}" ]]; then
  common_args+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

if [[ -z "${LOAD_FORMAT}" && "${MODEL_PATH}" == gs://* ]]; then
  LOAD_FORMAT="runai_streamer"
fi
if [[ -n "${LOAD_FORMAT}" ]]; then
  common_args+=(--load-format "${LOAD_FORMAT}")
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  common_args+=(--enable-prefix-caching)
else
  common_args+=(--no-enable-prefix-caching)
fi

if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  common_args+=(--async-scheduling)
else
  common_args+=(--no-async-scheduling)
fi

p_kv='{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":'"${KV_PORT}"'}'
d_kv='{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":'"${KV_PORT}"'}'

prefill_namespace="prefill_${RUN_ID}"
decode_namespace="decode_${RUN_ID}"

plan_dump_dir="${RUN_DIR}/plan_dump"
mkdir -p "${plan_dump_dir}"

# ---------------------------------------------------------------------------
# Construct role CLI commands
# ---------------------------------------------------------------------------
prefill_args=(
  --host 0.0.0.0
  --port "${PREFILL_PORT}"
  "${common_args[@]}"
  --compilation-config "${prefill_compilation_config}"
  --tensor-parallel-size "${PREFILL_TP}"
  --prefill-context-parallel-size "${PREFILL_PCP}"
  --kv-transfer-config "${p_kv}"
)
if [[ "${PREFILL_PCP}" -gt 1 ]]; then
  prefill_args+=(--cp-kv-cache-interleave-size "${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE}")
fi
if [[ -n "${PREFILL_BLOCK_SIZE}" ]]; then
  prefill_args+=(--block-size "${PREFILL_BLOCK_SIZE}")
fi
if [[ -n "${PREFILL_NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  prefill_args+=(--num-gpu-blocks-override "${PREFILL_NUM_GPU_BLOCKS_OVERRIDE}")
fi
if [[ -n "${PREFILL_SPECULATIVE_CONFIG}" ]]; then
  prefill_args+=(--speculative-config "${PREFILL_SPECULATIVE_CONFIG}")
fi

decode_args=(
  --host 0.0.0.0
  --port "${DECODE_PORT}"
  "${common_args[@]}"
  --compilation-config "${decode_compilation_config}"
  --tensor-parallel-size "${DECODE_TP}"
  --prefill-context-parallel-size 1
  --kv-transfer-config "${d_kv}"
)
if [[ "${DECODE_DP}" -gt 1 ]]; then
  decode_args+=(--data-parallel-size "${DECODE_DP}" --data-parallel-size-local "${DECODE_DP}")
fi
if [[ -n "${DECODE_BLOCK_SIZE}" ]]; then
  decode_args+=(--block-size "${DECODE_BLOCK_SIZE}")
fi
if [[ -n "${DECODE_NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  decode_args+=(--num-gpu-blocks-override "${DECODE_NUM_GPU_BLOCKS_OVERRIDE}")
fi
if [[ -n "${DECODE_SPECULATIVE_CONFIG}" ]]; then
  decode_args+=(--speculative-config "${DECODE_SPECULATIVE_CONFIG}")
fi

cat >"${RUN_DIR}/launch_params.txt" <<EOF
ROLE=${ROLE}
RUN_DIR=${RUN_DIR}
RUN_ID=${RUN_ID}
PYTHON_BIN=${PYTHON_BIN}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
CONNECTOR=TPURaidenConnector
PREFILL_HOST=${PREFILL_HOST}
PREFILL_ADVERTISE_HOST=${PREFILL_ADVERTISE_HOST}
PREFILL_PORT=${PREFILL_PORT}
PREFILL_CTRL_PORT=${PREFILL_CTRL_PORT}
PREFILL_TPU_KV_TRANSFER_PORT=${PREFILL_TPU_KV_TRANSFER_PORT}
PREFILL_TP=${PREFILL_TP}
PREFILL_PCP=${PREFILL_PCP}
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE}
PREFILL_COMPILE_SIZES=${PREFILL_COMPILE_SIZES}
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET=${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}
DECODE_HOST=${DECODE_HOST}
DECODE_ADVERTISE_HOST=${DECODE_ADVERTISE_HOST}
DECODE_PORT=${DECODE_PORT}
DECODE_CTRL_PORT=${DECODE_CTRL_PORT}
DECODE_TPU_KV_TRANSFER_PORT=${DECODE_TPU_KV_TRANSFER_PORT}
DECODE_TP=${DECODE_TP}
DECODE_DP=${DECODE_DP}
DECODE_COMPILE_SIZES=${DECODE_COMPILE_SIZES}
DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET=${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}
DECODE_DP_MASTER_PORT=${DECODE_DP_MASTER_PORT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
ASYNC_SCHEDULING=${ASYNC_SCHEDULING}
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}
EOF

# ---------------------------------------------------------------------------
# Launch servers
# ---------------------------------------------------------------------------
common_env_exports=(
  "export JAX_PLATFORMS=tpu,cpu"
  "export PJRT_DEVICE=TPU"
  "export TPU_BACKEND_TYPE=jax"
  "export VLLM_XLA_CHECK_RECOMPILATION=0"
  "export XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  "export USE_MOE_SPARSE_CORE=${USE_MOE_SPARSE_CORE:-1}"
  "export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}"
  "export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1"
  "export TPU_RAIDEN_QWEN35_ADMISSION=1"
  "export TPU_KV_RESHARD_TRANSPORT=raiden"
  "export TPU_RAIDEN_TRANSFER_PARALLELISM=${TPU_RAIDEN_TRANSFER_PARALLELISM}"
  "export TPU_ENABLE_D2H_TRANSFER=1"
  "export TPU_HMA_D2H_COPY_IMPL=device_put"
  "export TPU_HMA_MIN_DONE_RECVING_BATCH=1"
  "export TPU_HMA_MAX_DONE_RECVING_BATCH=1"
  "export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1"
  "export TPU_HMA_MAX_INFLIGHT_PULLS=8"
  "export TPU_MAX_HOST_KV_BUFFER_SIZE=64"
  "export TPU_P2P_WAIT_PULL_TIMEOUT=600"
  "export TPU_KV_SHM_POOL_GB=${TPU_KV_SHM_POOL_GB}"
  "export TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}"
  "export LIBTPU_INIT_ARGS=\"\${LIBTPU_INIT_ARGS:-} --xla_tpu_scoped_vmem_limit_kib=65536\""
  "export ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
  "export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback"
  "export TPU_RAGGED_GATHER_IMPL=fallback"
  "export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0"
  "export TPU_GDN_CONV_STATE_TILE_PAD=1"
  "export TPU_RAIDEN_PLAN_DUMP_DIR=$(printf '%q' "${plan_dump_dir}")"
)

joined_common_env=$(IFS='; '; echo "${common_env_exports[*]}")

if [[ "${ROLE}" == "prefill" || "${ROLE}" == "all" ]]; then
  prefill_cmd="${joined_common_env}; export DEBUG_TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${prefill_namespace}'; export TPU_RAIDEN_JOB_NAME=prefill; export TPU_RAIDEN_ENGINE_ID='prefill-engine-${RUN_ID}'; export TPU_RAIDEN_RESHARD_IMPL=store; export TPU_RAIDEN_ADVERTISE_HOST='${PREFILL_ADVERTISE_HOST}'; export TPU_RAIDEN_RESHARD_PORT_BASE='${PREFILL_CTRL_PORT}'; export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE='$((PREFILL_CTRL_PORT + 100))'; export TPU_KV_TRANSFER_PORT=${PREFILL_TPU_KV_TRANSFER_PORT}; export TORCH_TPU_SLICEBUILDER_ADDRESSES='${prefill_sb_addresses}'; $(printf '%q' "${PYTHON_BIN}") -m vllm.entrypoints.openai.api_server $(printf '%q ' "${prefill_args[@]}")"
  printf '%s\n' "${prefill_cmd}" >"${RUN_DIR}/prefill_cmd.sh"
  echo "--- Starting prefill server (PCP=${PREFILL_PCP}, PORT=${PREFILL_PORT})"
  setsid bash -lc "${prefill_cmd}" >"${RUN_DIR}/logs/prefill.log" 2>&1 &
  echo $! >"${RUN_DIR}/prefill.pid"
  echo "Prefill PID: $(cat "${RUN_DIR}/prefill.pid")"
fi

if [[ "${ROLE}" == "decode" || "${ROLE}" == "all" ]]; then
  decode_cmd="${joined_common_env}; export DEBUG_TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${decode_namespace}'; export TPU_RAIDEN_JOB_NAME=decode; export TPU_RAIDEN_ENGINE_ID='decode-engine-${RUN_ID}'; export TPU_RAIDEN_RESHARD_IMPL=store; export TPU_RAIDEN_ADVERTISE_HOST='${DECODE_ADVERTISE_HOST}'; export TPU_RAIDEN_RESHARD_PORT_BASE='${DECODE_CTRL_PORT}'; export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE='$((DECODE_CTRL_PORT + 100))'; export TPU_KV_TRANSFER_PORT=${DECODE_TPU_KV_TRANSFER_PORT}; export TORCH_TPU_SLICEBUILDER_ADDRESSES='${decode_sb_addresses}'; export TORCH_TPU_DP_MASTER_PORT='${DECODE_DP_MASTER_PORT}'; $(printf '%q' "${PYTHON_BIN}") -m vllm.entrypoints.openai.api_server $(printf '%q ' "${decode_args[@]}")"
  printf '%s\n' "${decode_cmd}" >"${RUN_DIR}/decode_cmd.sh"
  echo "--- Starting decode server (DP=${DECODE_DP}, PORT=${DECODE_PORT})"
  setsid bash -lc "${decode_cmd}" >"${RUN_DIR}/logs/decode.log" 2>&1 &
  echo $! >"${RUN_DIR}/decode.pid"
  echo "Decode PID: $(cat "${RUN_DIR}/decode.pid")"
fi

# ---------------------------------------------------------------------------
# Health check polling
# ---------------------------------------------------------------------------
health_log="${RUN_DIR}/logs/health_poll.log"
deadline=$((SECONDS + STARTUP_TIMEOUT_S))

echo "--- Waiting for server health (timeout ${STARTUP_TIMEOUT_S}s)"
while [[ "${SECONDS}" -lt "${deadline}" ]]; do
  all_healthy=1

  if [[ "${ROLE}" == "prefill" || "${ROLE}" == "all" ]]; then
    p_pid="$(cat "${RUN_DIR}/prefill.pid" 2>/dev/null || true)"
    if [[ -z "${p_pid}" ]] || ! kill -0 "${p_pid}" 2>/dev/null; then
      echo "ERROR: Prefill server process died unexpectedly!" | tee -a "${health_log}"
      tail -n 50 "${RUN_DIR}/logs/prefill.log" || true
      exit 1
    fi
    p_code="$(curl -fsS -o /dev/null -w "%{http_code}" "http://${PREFILL_HOST}:${PREFILL_PORT}/health" 2>/dev/null || true)"
    if [[ "${p_code}" != "200" ]]; then
      all_healthy=0
    fi
  else
    p_code="SKIP"
  fi

  if [[ "${ROLE}" == "decode" || "${ROLE}" == "all" ]]; then
    d_pid="$(cat "${RUN_DIR}/decode.pid" 2>/dev/null || true)"
    if [[ -z "${d_pid}" ]] || ! kill -0 "${d_pid}" 2>/dev/null; then
      echo "ERROR: Decode server process died unexpectedly!" | tee -a "${health_log}"
      tail -n 50 "${RUN_DIR}/logs/decode.log" || true
      exit 1
    fi
    d_code="$(curl -fsS -o /dev/null -w "%{http_code}" "http://${DECODE_HOST}:${DECODE_PORT}/health" 2>/dev/null || true)"
    if [[ "${d_code}" != "200" ]]; then
      all_healthy=0
    fi
  else
    d_code="SKIP"
  fi

  echo "$(date +%H:%M:%S) P=${p_code} D=${d_code}" | tee -a "${health_log}"
  if [[ "${all_healthy}" -eq 1 ]]; then
    echo "All requested servers healthy!"
    break
  fi

  sleep 10
done

if [[ "${all_healthy:-0}" -ne 1 ]]; then
  echo "ERROR: Server(s) failed to reach healthy status within ${STARTUP_TIMEOUT_S}s" >&2
  if [[ -f "${RUN_DIR}/logs/prefill.log" ]]; then
    echo "=== Prefill Server Log Tail (last 100 lines) ==="
    tail -n 100 "${RUN_DIR}/logs/prefill.log" || true
  fi
  if [[ -f "${RUN_DIR}/logs/decode.log" ]]; then
    echo "=== Decode Server Log Tail (last 100 lines) ==="
    tail -n 100 "${RUN_DIR}/logs/decode.log" || true
  fi
  exit 1
fi

echo "RUN_DIR=${RUN_DIR}"

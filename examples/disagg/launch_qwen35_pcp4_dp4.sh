#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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


RUN_ROOT="${RUN_ROOT:-${HOME}/pd_disagg_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/qwen35_pd_v1_pcp4_dp4_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B-FP8}"

SERVE_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
KV_PORT="${KV_PORT:-14579}"
TPU_KV_TRANSFER_PORT="${TPU_KV_TRANSFER_PORT:-9100}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-9600}"

# PCP4 to DP4 serving parameters:
PREFILL_TP=1
DECODE_TP=1
PREFILL_PCP=4
DECODE_DP=4

# Logical Rank Offset mapping:
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET=0
DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET=4

# Tunable sizes:
BLOCK_SIZE_PREFILL=4096
BLOCK_SIZE_DECODE=4096
NUM_GPU_BLOCKS_OVERRIDE_PREFILL=16
NUM_GPU_BLOCKS_OVERRIDE_DECODE=72

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
RESTART_EXISTING="${RESTART_EXISTING:-1}"

mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUN_ROOT}/latest_qwen35_pd_v1_pcp4dp4"
echo "RUN_DIR=${RUN_DIR}"
echo "latest=${RUN_ROOT}/latest_qwen35_pd_v1_pcp4dp4"

managed_ports=("${PROXY_PORT}" "${PREFILL_PORT}" "${DECODE_PORT}" "${TPU_SIDE_CHANNEL_PORT}" 27000 28000)
for offset in 0 1 2 3; do
  managed_ports+=("$((TPU_KV_TRANSFER_PORT + offset))")
  managed_ports+=("$((9200 + offset))")
done
if [[ "${RESTART_EXISTING}" == "1" ]]; then
  stop_port_listeners "${managed_ports[@]}"
else
  require_free_ports "${managed_ports[@]}"
fi

prefill_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
decode_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[256],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

common_args=(
  --host "${SERVE_HOST}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
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
  --no-enable-prefix-caching
  --no-async-scheduling
)

# TPUConnector (v1) configurations:
p_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"}'
d_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'

prefill_namespace="prefill_pcp4_dp4_$(date +%Y%m%d_%H%M%S)"
decode_namespace="decode_pcp4_dp4_$(date +%Y%m%d_%H%M%S)"

cat >"${RUN_DIR}/launch_params.txt" <<EOF
RUN_DIR=${RUN_DIR}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
CONNECTOR=TPUConnector
CONNECTOR_MODULE=vllm_torchtpu.distributed.kv_transfer.tpu_connector
PREFILL_TP=${PREFILL_TP}
DECODE_TP=${DECODE_TP}
PREFILL_PCP=${PREFILL_PCP}
DECODE_DP=${DECODE_DP}
BLOCK_SIZE_PREFILL=${BLOCK_SIZE_PREFILL}
BLOCK_SIZE_DECODE=${BLOCK_SIZE_DECODE}
NUM_GPU_BLOCKS_OVERRIDE_PREFILL=${NUM_GPU_BLOCKS_OVERRIDE_PREFILL}
NUM_GPU_BLOCKS_OVERRIDE_DECODE=${NUM_GPU_BLOCKS_OVERRIDE_DECODE}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
BIND_HOST=${SERVE_HOST} PREFILL_PORT=${PREFILL_PORT} DECODE_PORT=${DECODE_PORT} PROXY_PORT=${PROXY_PORT}
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET=${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET} TPU_KV_TRANSFER_NAMESPACE=${prefill_namespace}
DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET=${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET} TPU_KV_TRANSFER_NAMESPACE=${decode_namespace}
TPU_KV_TRANSFER_PORT=${TPU_KV_TRANSFER_PORT}
TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
EOF

printf '%q ' "${common_args[@]}" >"${RUN_DIR}/vllm_args.quoted"

common_env=$(cat <<EOF
export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax;
export VLLM_TARGET_DEVICE=tpu;
export VLLM_XLA_CHECK_RECOMPILATION=0;
export USE_MOE_SPARSE_CORE=1;
export TPUMISC_HMA_CHECKSUM_TRACE=0;
export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;
unset VLLM_DISABLE_COMPILE_CACHE VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR;
export LIBTPU_INIT_ARGS="\${LIBTPU_INIT_ARGS:-} --xla_tpu_scoped_vmem_limit_kib=65536";
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768;
export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback;
export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0;
export TPU_KV_TRANSFER_PORT="${TPU_KV_TRANSFER_PORT}";
export TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT}";

# Golden byte-IR env variables:
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_KV_RESHARD_DST_PAGE_TOKENS=4096
export TPU_RAIDEN_TRANSFER_PARALLELISM=4
unset TPU_KV_RESHARD_TRANSFER_TAGS
export TPU_GDN_CONV_STATE_TILE_PAD=1
EOF
)

# Start Controllers helper env:
ctrl_common_env=$(cat <<EOF
EOF
)

prefill_args=(--port "${PREFILL_PORT}" "${common_args[@]}" --compilation-config "${prefill_compilation_config}" --block-size "${BLOCK_SIZE_PREFILL}" --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE_PREFILL}" --tensor-parallel-size "${PREFILL_TP}" --prefill-context-parallel-size "${PREFILL_PCP}" --cp-kv-cache-interleave-size 256 --kv-transfer-config "${p_kv}")

decode_args=(--port "${DECODE_PORT}" "${common_args[@]}" --compilation-config "${decode_compilation_config}" --block-size "${BLOCK_SIZE_DECODE}" --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE_DECODE}" --tensor-parallel-size "${DECODE_TP}" --prefill-context-parallel-size 1 --data-parallel-size "${DECODE_DP}" --data-parallel-size-local "${DECODE_DP}" --kv-transfer-config "${d_kv}")

prefill_cmd="${common_env}"$'\n'"export DEBUG_TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${prefill_namespace}'; export TPU_RAIDEN_JOB_NAME=prefill; export TPU_RAIDEN_ENGINE_ID=prefill-engine; export TPU_RAIDEN_CONTROLLER_ADDRESS=127.0.0.1:27000; python -m vllm.entrypoints.openai.api_server $(printf '%q ' "${prefill_args[@]}")"

decode_cmd="${common_env}"$'\n'"export DEBUG_TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${decode_namespace}'; export TPU_RAIDEN_JOB_NAME=decode; export TPU_RAIDEN_ENGINE_ID=decode-engine; export TPU_RAIDEN_CONTROLLER_ADDRESS=127.0.0.1:28000; export TPU_KV_TRANSFER_PORT=9200; python -m vllm.entrypoints.openai.api_server $(printf '%q ' "${decode_args[@]}")"

prefill_ctrl_cmd="${ctrl_common_env}"$'\n'"python ${script_dir}/run_raiden_controller.py --port 27000"

decode_ctrl_cmd="${ctrl_common_env}"$'\n'"python ${script_dir}/run_raiden_controller.py --port 28000"

printf '%s\n' "${prefill_ctrl_cmd}" >"${RUN_DIR}/prefill_ctrl_cmd.sh"
printf '%s\n' "${decode_ctrl_cmd}" >"${RUN_DIR}/decode_ctrl_cmd.sh"
printf '%s\n' "${prefill_cmd}" >"${RUN_DIR}/prefill_cmd.sh"
printf '%s\n' "${decode_cmd}" >"${RUN_DIR}/decode_cmd.sh"

# Launch prefill controller:
setsid bash -lc "${prefill_ctrl_cmd}" >"${RUN_DIR}/logs/prefill_controller.log" 2>&1 &
echo $! >"${RUN_DIR}/prefill_controller.pid"

# Launch decode controller:
setsid bash -lc "${decode_ctrl_cmd}" >"${RUN_DIR}/logs/decode_controller.log" 2>&1 &
echo $! >"${RUN_DIR}/decode_controller.pid"

sleep 2

# Launch prefill and decode API servers:
setsid bash -lc "${prefill_cmd}" >"${RUN_DIR}/logs/prefill.log" 2>&1 &
echo $! >"${RUN_DIR}/prefill.pid"
setsid bash -lc "${decode_cmd}" >"${RUN_DIR}/logs/decode.log" 2>&1 &
echo $! >"${RUN_DIR}/decode.pid"

echo "prefill controller pid $(cat "${RUN_DIR}/prefill_controller.pid")"
echo "decode controller pid $(cat "${RUN_DIR}/decode_controller.pid")"
echo "prefill pid $(cat "${RUN_DIR}/prefill.pid")"
echo "decode pid $(cat "${RUN_DIR}/decode.pid")"

health_log="${RUN_DIR}/logs/health_poll.log"
for ((attempt = 1; attempt <= 120; attempt++)); do
  p_code=$(curl -fsS -o /dev/null -w "%{http_code}" "http://${SERVE_HOST}:${PREFILL_PORT}/health" 2>/dev/null || true)
  d_code=$(curl -fsS -o /dev/null -w "%{http_code}" "http://${SERVE_HOST}:${DECODE_PORT}/health" 2>/dev/null || true)
  echo "$(date +%H:%M:%S) P=${p_code:-NA} D=${d_code:-NA}" | tee -a "${health_log}"
  if [[ "${p_code}" == "200" && "${d_code}" == "200" ]]; then
    break
  fi
  if ! kill -0 "$(cat "${RUN_DIR}/prefill.pid" 2>/dev/null)" 2>/dev/null \
     || ! kill -0 "$(cat "${RUN_DIR}/decode.pid" 2>/dev/null)" 2>/dev/null \
     || ! kill -0 "$(cat "${RUN_DIR}/prefill_controller.pid" 2>/dev/null)" 2>/dev/null \
     || ! kill -0 "$(cat "${RUN_DIR}/decode_controller.pid" 2>/dev/null)" 2>/dev/null; then
    if grep -E "Traceback|RuntimeError|ValueError|LLVM ERROR|registered memory region overlaps|strided KV pull failed" "${RUN_DIR}/logs/prefill.log" "${RUN_DIR}/logs/decode.log" >/tmp/pd_pcp4dp4_launch_err.$$ 2>/dev/null; then
      cat /tmp/pd_pcp4dp4_launch_err.$$
      rm -f /tmp/pd_pcp4dp4_launch_err.$$
      exit 1
    fi
    echo "Prefill, decode, or one of their controller processes died unexpectedly!"
    exit 1
  fi
  sleep 15
done

p_code=$(curl -fsS -o /dev/null -w "%{http_code}" "http://${SERVE_HOST}:${PREFILL_PORT}/health" 2>/dev/null || true)
d_code=$(curl -fsS -o /dev/null -w "%{http_code}" "http://${SERVE_HOST}:${DECODE_PORT}/health" 2>/dev/null || true)
if [[ "${p_code}" != "200" || "${d_code}" != "200" ]]; then
  echo "P/D did not become healthy: P=${p_code:-NA} D=${d_code:-NA}" >&2
  exit 1
fi

proxy_cmd=$(cat <<EOF
python examples/disagg/toy_proxy_server.py --host '${SERVE_HOST}' --port '${PROXY_PORT}' --prefiller-hosts '${SERVE_HOST}' --prefiller-ports '${PREFILL_PORT}' --decoder-hosts '${SERVE_HOST}' --decoder-ports '${DECODE_PORT}'
EOF
)
printf '%s\n' "${proxy_cmd}" >"${RUN_DIR}/proxy_cmd.sh"
setsid bash -lc "${proxy_cmd}" >"${RUN_DIR}/logs/proxy.log" 2>&1 &
echo $! >"${RUN_DIR}/proxy.pid"

sleep 2
python "${script_dir}/../../scripts/vllm/integration/smoke_prefix_cache_correctness.py" \
  --host "${SERVE_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --quick-probe-only \
  | tee "${RUN_DIR}/logs/proxy_probe.log"

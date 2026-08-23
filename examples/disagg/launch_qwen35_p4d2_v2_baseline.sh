#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    echo "${CONDA_SH}"
    return 0
  fi
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "/mnt/data/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Could not find conda.sh. Set CONDA_SH." >&2
  return 1
}

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

shell_quote() {
  printf '%q' "$1"
}

RUN_ROOT="${RUN_ROOT:-${HOME}/pd_disagg_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/qwen35_pd_v2_p4_d2_$(date +%Y%m%d_%H%M%S)_baseline_config}"
PY_ENV="${PY_ENV:-${HOME}/.cache/tpu-misc-pd-disagg/conda_env_py3.12}"
VLLM_SRC="${VLLM_SRC:-/mnt/data/vllm}"
TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.27.0}"
USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-0}"

SERVE_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
KV_PORT="${KV_PORT:-14579}"
PREFILL_TPU_KV_TRANSFER_PORT="${PREFILL_TPU_KV_TRANSFER_PORT:-9100}"
DECODE_TPU_KV_TRANSFER_PORT="${DECODE_TPU_KV_TRANSFER_PORT:-9200}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-9600}"
PREFILL_CTRL_PORT="${PREFILL_CTRL_PORT:-27000}"
DECODE_CTRL_PORT="${DECODE_CTRL_PORT:-28000}"

PREFILL_TP="${PREFILL_TP:-4}"
PREFILL_PCP="${PREFILL_PCP:-1}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-1}"
DECODE_TP="${DECODE_TP:-2}"
DECODE_DP="${DECODE_DP:-1}"
# Temporary same-host P4/D2 test hook. This is intentionally DEBUG-prefixed:
# it offsets TorchTPU's physical LOCAL_RANK binding for the decode server and
# is not a CUDA_VISIBLE_DEVICES-style remapping mechanism.
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET="${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET:-0}"
DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET="${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET:-4}"

PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-}"
DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-1536}"
prefill_block_size_label="${PREFILL_BLOCK_SIZE:-auto}"
decode_block_size_label="${DECODE_BLOCK_SIZE:-auto}"
# Preserve an explicitly empty value so the large-cache test can exercise
# HBM-based block profiling instead of forcing the 128-block CI baseline.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
COMPILE_SIZES="${COMPILE_SIZES:-16384}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-${COMPILE_SIZES}}"
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-${COMPILE_SIZES}}"
PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-}"
DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
TPU_KV_SHM_POOL_GB="${TPU_KV_SHM_POOL_GB:-8}"
TPUMISC_HMA_CHECKSUM_TRACE="${TPUMISC_HMA_CHECKSUM_TRACE:-0}"
USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-0}"
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
RESTART_EXISTING="${RESTART_EXISTING:-1}"

mkdir -p "${RUN_DIR}/logs"
if [[ "${USE_CURRENT_PY_ENV}" == "1" ]]; then
  conda_sh="__current_python_environment__"
else
  conda_sh="$(find_conda_sh)"
fi

if [[ ! -d "${VLLM_SRC}" ]]; then
  echo "VLLM_SRC does not exist: ${VLLM_SRC}" >&2
  exit 2
fi
if [[ ! -d "${TORCHTPU_VLLM_SRC}" ]]; then
  echo "TORCHTPU_VLLM_SRC does not exist: ${TORCHTPU_VLLM_SRC}" >&2
  exit 2
fi
if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
  fi
fi
for compile_sizes_name in COMPILE_SIZES PREFILL_COMPILE_SIZES DECODE_COMPILE_SIZES; do
  compile_sizes_value="${!compile_sizes_name}"
  if [[ ! "${compile_sizes_value}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "${compile_sizes_name} must be a comma-separated integer list: ${compile_sizes_value}" >&2
    exit 2
  fi
done

managed_ports=("${PROXY_PORT}" "${PREFILL_PORT}" "${DECODE_PORT}" "${TPU_SIDE_CHANNEL_PORT}")
for offset in 0 1 2 3; do
  managed_ports+=("$((PREFILL_TPU_KV_TRANSFER_PORT + offset))")
  managed_ports+=("$((DECODE_TPU_KV_TRANSFER_PORT + offset))")
  managed_ports+=("$((PREFILL_CTRL_PORT + offset))")
  managed_ports+=("$((PREFILL_CTRL_PORT + 100 + offset))")
  managed_ports+=("$((DECODE_CTRL_PORT + offset))")
  managed_ports+=("$((DECODE_CTRL_PORT + 100 + offset))")
done
mapfile -t managed_ports < <(printf '%s\n' "${managed_ports[@]}" | sort -u)
if [[ "${RESTART_EXISTING}" == "1" ]]; then
  stop_port_listeners "${managed_ports[@]}"
else
  require_free_ports "${managed_ports[@]}"
fi

if [[ "${USE_CURRENT_PY_ENV}" != "1" ]]; then
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${PY_ENV}"
fi
python_bin="$(command -v python || true)"
if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "Could not find an executable python in the selected environment." >&2
  exit 2
fi

version_log="${RUN_DIR}/logs/version_check.log"
{
  if [[ "${USE_CURRENT_PY_ENV}" == "1" ]]; then
    echo "python_env=current"
  else
    echo "python_env=${PY_ENV}"
  fi
  echo "python_bin=${python_bin}"
  echo "vllm_src=${VLLM_SRC}"
  echo "torchtpu_vllm_src=${TORCHTPU_VLLM_SRC}"
  git -C "${VLLM_SRC}" rev-parse HEAD 2>/dev/null \
    | sed 's/^/vllm_commit=/' || true
  git -C "${TORCHTPU_VLLM_SRC}" rev-parse HEAD 2>/dev/null \
    | sed 's/^/torchtpu_vllm_commit=/' || true
  export PYTHONPATH="${TORCHTPU_VLLM_SRC}/src:${TORCHTPU_VLLM_SRC}:${VLLM_SRC}:${PYTHONPATH:-}"
  "${python_bin}" - <<PY
import sys
import vllm
import vllm_torchtpu

expected = ${EXPECTED_VLLM_VERSION@Q}
print("vllm.__version__", vllm.__version__)
print("vllm.__file__", vllm.__file__)
print("vllm_torchtpu.__file__", vllm_torchtpu.__file__)
if vllm.__version__ != expected:
    raise SystemExit(f"vLLM version must be {expected}, got {vllm.__version__}")
PY
} >"${version_log}" 2>&1

prefill_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":['"${PREFILL_COMPILE_SIZES}"'],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
decode_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":['"${DECODE_COMPILE_SIZES}"'],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

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
)

if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  common_args+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
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

p_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"}'
d_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'

prefill_namespace="prefill_p4d2_baseline_$(date +%Y%m%d_%H%M%S)"
decode_namespace="decode_p4d2_baseline_$(date +%Y%m%d_%H%M%S)"

cat >"${RUN_DIR}/launch_params.txt" <<EOF
RUN_DIR=${RUN_DIR}
PY_ENV=${PY_ENV}
CONDA_SH=${conda_sh}
USE_CURRENT_PY_ENV=${USE_CURRENT_PY_ENV}
PYTHON_BIN=${python_bin}
VLLM_PYTHONPATH=${VLLM_SRC}
TORCHTPU_VLLM_PYTHONPATH=${TORCHTPU_VLLM_SRC}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION}
CONNECTOR=TPUConnector
CONNECTOR_MODULE=vllm_torchtpu.distributed.kv_transfer.tpu_connector
PREFILL_TP=${PREFILL_TP}
PREFILL_PCP=${PREFILL_PCP}
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE}
DECODE_TP=${DECODE_TP}
DECODE_DP=${DECODE_DP}
PREFILL_COMPILE_SIZES=${PREFILL_COMPILE_SIZES}
DECODE_COMPILE_SIZES=${DECODE_COMPILE_SIZES}
PREFILL_SPECULATIVE_CONFIG=${PREFILL_SPECULATIVE_CONFIG}
DECODE_SPECULATIVE_CONFIG=${DECODE_SPECULATIVE_CONFIG}
PREFILL_BLOCK_SIZE=${prefill_block_size_label}
DECODE_BLOCK_SIZE=${decode_block_size_label}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE}
ASYNC_SCHEDULING=${ASYNC_SCHEDULING}
BIND_HOST=${SERVE_HOST} PREFILL_PORT=${PREFILL_PORT} DECODE_PORT=${DECODE_PORT} PROXY_PORT=${PROXY_PORT}
PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET=${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET} TPU_KV_TRANSFER_NAMESPACE=${prefill_namespace}
DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET=${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET} TPU_KV_TRANSFER_NAMESPACE=${decode_namespace}
PREFILL_TPU_KV_TRANSFER_PORT=${PREFILL_TPU_KV_TRANSFER_PORT}
DECODE_TPU_KV_TRANSFER_PORT=${DECODE_TPU_KV_TRANSFER_PORT}
TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}
PREFILL_CTRL_PORT=${PREFILL_CTRL_PORT}
DECODE_CTRL_PORT=${DECODE_CTRL_PORT}
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
EOF

printf '%q ' "${common_args[@]}" >"${RUN_DIR}/vllm_args.quoted"

plan_dump_dir="${RUN_DIR}/plan_dump"
mkdir -p "${plan_dump_dir}"

if [[ "${USE_CURRENT_PY_ENV}" == "1" ]]; then
  activation_cmd=":"
else
  activation_cmd="source $(shell_quote "${conda_sh}");
conda activate $(shell_quote "${PY_ENV}");"
fi

common_env=$(cat <<EOF
${activation_cmd}
cd $(shell_quote "${TORCHTPU_VLLM_SRC}");
export PYTHONPATH=$(shell_quote "${TORCHTPU_VLLM_SRC}/src:${TORCHTPU_VLLM_SRC}:${VLLM_SRC}"):\${PYTHONPATH:-};
export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax;
export VLLM_XLA_CHECK_RECOMPILATION=0;
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE}";
export TPUMISC_HMA_CHECKSUM_TRACE="${TPUMISC_HMA_CHECKSUM_TRACE}";
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}";
export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;
unset VLLM_DISABLE_COMPILE_CACHE VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR;
export LIBTPU_INIT_ARGS="\${LIBTPU_INIT_ARGS:-} --xla_tpu_scoped_vmem_limit_kib=65536";
export TPU_ENABLE_D2H_TRANSFER=1 TPU_HMA_D2H_COPY_IMPL=device_put;
export TPU_HMA_MIN_DONE_RECVING_BATCH=1 TPU_HMA_MAX_DONE_RECVING_BATCH=1;
export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1 TPU_HMA_MAX_INFLIGHT_PULLS=8;
export TPU_MAX_HOST_KV_BUFFER_SIZE=64 TPU_P2P_WAIT_PULL_TIMEOUT=600;
export TPU_KV_SHM_POOL_GB="${TPU_KV_SHM_POOL_GB}";
export TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT}";
export ONEHOT_MOE_PERMUTE_THRESHOLD=1024;
export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback;
export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0;
export TPU_RAIDEN_PLAN_DUMP_DIR=$(shell_quote "${plan_dump_dir}");

# Raiden configurations:
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1;
export TPU_RAIDEN_QWEN35_ADMISSION=1;
export TPU_KV_RESHARD_TRANSPORT=raiden;
export TPU_RAIDEN_TRANSFER_PARALLELISM=4;
EOF
)

prefill_args=(--port "${PREFILL_PORT}" "${common_args[@]}" --compilation-config "${prefill_compilation_config}" --tensor-parallel-size "${PREFILL_TP}" --prefill-context-parallel-size "${PREFILL_PCP}" --kv-transfer-config "${p_kv}")
if [[ "${PREFILL_PCP}" -gt 1 ]]; then
  prefill_args+=(--cp-kv-cache-interleave-size "${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE}")
fi
decode_args=(--port "${DECODE_PORT}" "${common_args[@]}" --compilation-config "${decode_compilation_config}" --tensor-parallel-size "${DECODE_TP}" --prefill-context-parallel-size 1 --kv-transfer-config "${d_kv}")
if [[ "${DECODE_DP}" -gt 1 ]]; then
  decode_args+=(--data-parallel-size "${DECODE_DP}" --data-parallel-size-local "${DECODE_DP}")
fi
if [[ -n "${PREFILL_SPECULATIVE_CONFIG}" ]]; then
  prefill_args+=(--speculative-config "${PREFILL_SPECULATIVE_CONFIG}")
fi
if [[ -n "${DECODE_SPECULATIVE_CONFIG}" ]]; then
  decode_args+=(--speculative-config "${DECODE_SPECULATIVE_CONFIG}")
fi
if [[ -n "${PREFILL_BLOCK_SIZE}" ]]; then
  prefill_args+=(--block-size "${PREFILL_BLOCK_SIZE}")
fi
if [[ -n "${DECODE_BLOCK_SIZE}" ]]; then
  decode_args+=(--block-size "${DECODE_BLOCK_SIZE}")
fi
printf '%q ' "${prefill_args[@]}" >"${RUN_DIR}/prefill_vllm_args.quoted"
printf '%q ' "${decode_args[@]}" >"${RUN_DIR}/decode_vllm_args.quoted"

prefill_cmd="${common_env}"$'\n'"export DEBUG_TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${PREFILL_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${prefill_namespace}'; export TPU_RAIDEN_JOB_NAME=prefill; export TPU_RAIDEN_ENGINE_ID=prefill-engine; export TPU_RAIDEN_RESHARD_IMPL=store; export TPU_RAIDEN_ADVERTISE_HOST='${SERVE_HOST}'; export TPU_RAIDEN_RESHARD_PORT_BASE='${PREFILL_CTRL_PORT}'; export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE='$((PREFILL_CTRL_PORT + 100))'; export TPU_KV_TRANSFER_PORT=${PREFILL_TPU_KV_TRANSFER_PORT}; $(shell_quote "${python_bin}") -m vllm.entrypoints.openai.api_server $(printf '%q ' "${prefill_args[@]}")"
decode_cmd="${common_env}"$'\n'"export DEBUG_TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_LOCAL_RANK_OFFSET='${DECODE_DEBUG_TPU_LOCAL_RANK_OFFSET}'; export TPU_KV_TRANSFER_NAMESPACE='${decode_namespace}'; export TPU_RAIDEN_JOB_NAME=decode; export TPU_RAIDEN_ENGINE_ID=decode-engine; export TPU_RAIDEN_RESHARD_IMPL=store; export TPU_RAIDEN_ADVERTISE_HOST='${SERVE_HOST}'; export TPU_RAIDEN_RESHARD_PORT_BASE='${DECODE_CTRL_PORT}'; export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE='$((DECODE_CTRL_PORT + 100))'; export TPU_KV_TRANSFER_PORT=${DECODE_TPU_KV_TRANSFER_PORT}; $(shell_quote "${python_bin}") -m vllm.entrypoints.openai.api_server $(printf '%q ' "${decode_args[@]}")"

printf '%s\n' "${prefill_cmd}" >"${RUN_DIR}/prefill_cmd.sh"
printf '%s\n' "${decode_cmd}" >"${RUN_DIR}/decode_cmd.sh"

setsid bash -lc "${prefill_cmd}" >"${RUN_DIR}/logs/prefill.log" 2>&1 &
echo $! >"${RUN_DIR}/prefill.pid"
setsid bash -lc "${decode_cmd}" >"${RUN_DIR}/logs/decode.log" 2>&1 &
echo $! >"${RUN_DIR}/decode.pid"

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
     || ! kill -0 "$(cat "${RUN_DIR}/decode.pid" 2>/dev/null)" 2>/dev/null; then
    if grep -E "Traceback|RuntimeError|ValueError|LLVM ERROR|registered memory region overlaps|strided KV pull failed" "${RUN_DIR}/logs/prefill.log" "${RUN_DIR}/logs/decode.log" >/tmp/pd_p4d2_launch_err.$$ 2>/dev/null; then
      cat /tmp/pd_p4d2_launch_err.$$
      rm -f /tmp/pd_p4d2_launch_err.$$
      exit 1
    fi
    echo "Prefill, decode, or one of their processes died unexpectedly!"
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
${activation_cmd}
cd $(shell_quote "${TORCHTPU_VLLM_SRC}");
export PYTHONPATH=$(shell_quote "${TORCHTPU_VLLM_SRC}/src:${TORCHTPU_VLLM_SRC}:${VLLM_SRC}"):\${PYTHONPATH:-};
$(shell_quote "${python_bin}") examples/disagg/toy_proxy_server.py --host '${SERVE_HOST}' --port '${PROXY_PORT}' --prefiller-hosts '${SERVE_HOST}' --prefiller-ports '${PREFILL_PORT}' --decoder-hosts '${SERVE_HOST}' --decoder-ports '${DECODE_PORT}'
EOF
)
printf '%s\n' "${proxy_cmd}" >"${RUN_DIR}/proxy_cmd.sh"
setsid bash -lc "${proxy_cmd}" >"${RUN_DIR}/logs/proxy.log" 2>&1 &
echo $! >"${RUN_DIR}/proxy.pid"

sleep 2
"${python_bin}" "${TORCHTPU_VLLM_SRC}/scripts/vllm/integration/smoke_prefix_cache_correctness.py" \
  --host "${SERVE_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --quick-probe-only \
  | tee "${RUN_DIR}/logs/proxy_probe.log"

ln -sfn "${RUN_DIR}" "${RUN_ROOT}/latest_qwen35_pd_v2_p4d2_baseline"
echo "RUN_DIR=${RUN_DIR}"
echo "latest=${RUN_ROOT}/latest_qwen35_pd_v2_p4d2_baseline"

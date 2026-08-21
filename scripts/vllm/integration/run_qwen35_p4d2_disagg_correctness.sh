#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${QWEN35_A3B_FP8_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_p4d2_disagg_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
P4D2_BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8000}"
VLLM_SRC="${VLLM_SRC:-${repo_root}}"
TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
PREFILL_TP="${PREFILL_TP:-1}"
PREFILL_PCP="${PREFILL_PCP:-4}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
DECODE_TP="${DECODE_TP:-1}"
DECODE_DP="${DECODE_DP:-4}"
# Match the large-cache PCP prefill compilation geometry. An empty override
# makes vLLM derive the block count from available HBM.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256}"
default_prefill_speculative_config='{"method":"mtp","num_speculative_tokens":1}'
default_decode_speculative_config='{"method":"mtp","num_speculative_tokens":3}'
PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-${default_prefill_speculative_config}}"
DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-${default_decode_speculative_config}}"

cleanup() {
  local pid_file pid
  if [[ -d "${RUN_DIR}" ]]; then
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          kill -TERM -- "-${pid}" 2>/dev/null \
            || kill -TERM "${pid}" 2>/dev/null \
            || true
        fi
      fi
    done
    sleep 2
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          if kill -0 -- "-${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
          elif kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
          fi
        fi
      fi
    done
  fi
}
trap cleanup EXIT

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}" "${RUN_DIR}/logs"

export MODEL_PATH
export SERVED_MODEL_NAME
export RUN_ROOT
export RUN_DIR
export P4D2_BIND_HOST
export PROXY_PORT
export VLLM_SRC
export TORCHTPU_VLLM_SRC
export USE_CURRENT_PY_ENV
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
export ASYNC_SCHEDULING
export PREFILL_TP
export PREFILL_PCP
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE
export DECODE_TP
export DECODE_DP
export NUM_GPU_BLOCKS_OVERRIDE
export GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN
export MAX_NUM_SEQS
export PREFILL_COMPILE_SIZES
export DECODE_COMPILE_SIZES
export PREFILL_SPECULATIVE_CONFIG
export DECODE_SPECULATIVE_CONFIG

export P4D2_SHORT_REPEAT_LINES="${P4D2_SHORT_REPEAT_LINES:-32}"
export P4D2_SHORT_REPEAT_COUNT="${P4D2_SHORT_REPEAT_COUNT:-3}"
export P4D2_LONG_REPEAT_LINES="${P4D2_LONG_REPEAT_LINES:-96}"
export P4D2_LONG_REPEAT_COUNT="${P4D2_LONG_REPEAT_COUNT:-3}"
export P4D2_LONG_SHARED_LINES="${P4D2_LONG_SHARED_LINES:-72}"
export P4D2_LONG_SHARED_ROUNDS="${P4D2_LONG_SHARED_ROUNDS:-2}"
export P4D2_MIXED_LINES="${P4D2_MIXED_LINES:-72}"
export P4D2_MIXED_ROUNDS="${P4D2_MIXED_ROUNDS:-2}"
export P4D2_CONCURRENT_REQUESTS="${P4D2_CONCURRENT_REQUESTS:-1}"
export RUN_PREFIX_CACHE_E2E_DIVERGENCE="${RUN_PREFIX_CACHE_E2E_DIVERGENCE:-1}"

cd "${repo_root}"

bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh \
  2>&1 | tee "${RUN_DIR}/logs/launch.log"

python scripts/vllm/integration/smoke_prefix_cache_correctness.py \
  --host "${P4D2_BIND_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"

if [[ "${RUN_PREFIX_CACHE_E2E_DIVERGENCE}" == "1" ]]; then
  python scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py \
    --host "${P4D2_BIND_HOST}" \
    --port "${PROXY_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --max-tokens 1 \
    2>&1 | tee "${RUN_DIR}/logs/prefix_cache_e2e_divergence.log"
else
  echo "RUN_PREFIX_CACHE_E2E_DIVERGENCE=${RUN_PREFIX_CACHE_E2E_DIVERGENCE}; skip prefix-cache E2E divergence smoke"
fi

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
RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"

cleanup() {
  local pid_file pid
  if [[ -d "${RUN_DIR}" ]]; then
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]]; then
          kill -TERM "${pid}" 2>/dev/null || true
        fi
      fi
    done
    sleep 2
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
          kill -KILL "${pid}" 2>/dev/null || true
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
export RAGGED_GATED_DELTA_RULE_IMPL

export P4D2_SHORT_REPEAT_LINES="${P4D2_SHORT_REPEAT_LINES:-32}"
export P4D2_SHORT_REPEAT_COUNT="${P4D2_SHORT_REPEAT_COUNT:-3}"
export P4D2_LONG_REPEAT_LINES="${P4D2_LONG_REPEAT_LINES:-96}"
export P4D2_LONG_REPEAT_COUNT="${P4D2_LONG_REPEAT_COUNT:-3}"
export P4D2_LONG_SHARED_LINES="${P4D2_LONG_SHARED_LINES:-72}"
export P4D2_LONG_SHARED_ROUNDS="${P4D2_LONG_SHARED_ROUNDS:-2}"
export P4D2_MIXED_LINES="${P4D2_MIXED_LINES:-72}"
export P4D2_MIXED_ROUNDS="${P4D2_MIXED_ROUNDS:-2}"
export P4D2_CONCURRENT_REQUESTS="${P4D2_CONCURRENT_REQUESTS:-1}"

cd "${repo_root}"

bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh \
  2>&1 | tee "${RUN_DIR}/logs/launch.log"

python examples/disagg/smoke_qwen35_p4d2_v2_prefix_cache_correctness.py \
  --host "${P4D2_BIND_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"

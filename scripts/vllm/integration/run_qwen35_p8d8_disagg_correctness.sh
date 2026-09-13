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
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

ROLE="${ROLE:-all}"
MODEL_PATH="${MODEL_PATH:-${QWEN35_A3B_FP8_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_p8d8_disagg_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"

PREFILL_PCP="${PREFILL_PCP:-8}"
DECODE_DP="${DECODE_DP:-8}"
PREFILL_TP="${PREFILL_TP:-1}"
DECODE_TP="${DECODE_TP:-1}"

PREFILL_CTRL_PORT="${PREFILL_CTRL_PORT:-27000}"
DECODE_CTRL_PORT="${DECODE_CTRL_PORT:-28000}"
PREFILL_TPU_KV_TRANSFER_PORT="${PREFILL_TPU_KV_TRANSFER_PORT:-9100}"
DECODE_TPU_KV_TRANSFER_PORT="${DECODE_TPU_KV_TRANSFER_PORT:-9200}"

TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256,4096}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-128}"
RUN_PREFIX_CACHE_E2E_DIVERGENCE="${RUN_PREFIX_CACHE_E2E_DIVERGENCE:-1}"
MIN_MTP_ACCEPTANCE_RATE="${MIN_MTP_ACCEPTANCE_RATE:-0.65}"
TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-${PREFILL_PCP}}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"

mkdir -p "${RUN_ROOT}" "${RUN_DIR}/logs"

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Could not find an executable python3/python in PATH." >&2
  exit 2
fi

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
trap cleanup EXIT INT TERM

export MODEL_PATH
export SERVED_MODEL_NAME
export RUN_ROOT
export RUN_DIR
export PREFILL_HOST
export DECODE_HOST
export PREFILL_PORT
export DECODE_PORT
export PROXY_PORT
export PREFILL_PCP
export DECODE_DP
export PREFILL_TP
export DECODE_TP
export PREFILL_CTRL_PORT
export DECODE_CTRL_PORT
export PREFILL_TPU_KV_TRANSFER_PORT
export DECODE_TPU_KV_TRANSFER_PORT
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
export TPU_RAIDEN_TRANSFER_PARALLELISM
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE
export ASYNC_SCHEDULING
export GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN
export MAX_NUM_BATCHED_TOKENS
export MAX_NUM_SEQS
export PREFILL_COMPILE_SIZES
export DECODE_COMPILE_SIZES
export RUN_PREFIX_CACHE_E2E_DIVERGENCE
export STARTUP_TIMEOUT_S

export P4D2_SHORT_REPEAT_LINES="${P4D2_SHORT_REPEAT_LINES:-32}"
export P4D2_SHORT_REPEAT_COUNT="${P4D2_SHORT_REPEAT_COUNT:-3}"
export P4D2_LONG_REPEAT_LINES="${P4D2_LONG_REPEAT_LINES:-96}"
export P4D2_LONG_REPEAT_COUNT="${P4D2_LONG_REPEAT_COUNT:-3}"
export P4D2_LONG_SHARED_LINES="${P4D2_LONG_SHARED_LINES:-60}"
export P4D2_LONG_SHARED_ROUNDS="${P4D2_LONG_SHARED_ROUNDS:-3}"
export P4D2_MIXED_LINES="${P4D2_MIXED_LINES:-60}"
export P4D2_MIXED_ROUNDS="${P4D2_MIXED_ROUNDS:-2}"
export P4D2_CONCURRENT_REQUESTS="${P4D2_CONCURRENT_REQUESTS:-1}"
export PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL="${PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL:-0.2}"

cd "${repo_root}"

case "${ROLE}" in
  prefill)
    echo "--- Launching role=prefill (PCP=${PREFILL_PCP}, PORT=${PREFILL_PORT})"
    ROLE=prefill bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
      2>&1 | tee "${RUN_DIR}/logs/launch_prefill.log"
    prefill_pid="$(cat "${RUN_DIR}/prefill.pid" 2>/dev/null || true)"
    if [[ -n "${prefill_pid}" ]]; then
      echo "Prefill server running (PID: ${prefill_pid}). Waiting on process..."
      while kill -0 "${prefill_pid}" 2>/dev/null; do
        sleep 2
      done
    fi
    ;;

  decode)
    echo "--- Launching role=decode (DP=${DECODE_DP}, PORT=${DECODE_PORT})"
    ROLE=decode bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
      2>&1 | tee "${RUN_DIR}/logs/launch_decode.log"
    decode_pid="$(cat "${RUN_DIR}/decode.pid" 2>/dev/null || true)"
    if [[ -n "${decode_pid}" ]]; then
      echo "Decode server running (PID: ${decode_pid}). Waiting on process..."
      while kill -0 "${decode_pid}" 2>/dev/null; do
        sleep 2
      done
    fi
    ;;

  head|runner|all)
    if [[ "${ROLE}" == "head" ]]; then
      echo "--- Launching head prefill server locally"
      ROLE=prefill bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
        2>&1 | tee "${RUN_DIR}/logs/launch_head_prefill.log"
      target_prefill_host="${PREFILL_HOST:-127.0.0.1}"
      target_decode_host="${DECODE_HOST}"
    elif [[ "${ROLE}" == "all" ]]; then
      echo "--- Launching prefill and decode servers locally"
      ROLE=all PREFILL_HOST="127.0.0.1" DECODE_HOST="127.0.0.1" bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
        2>&1 | tee "${RUN_DIR}/logs/launch_all.log"
      target_prefill_host="127.0.0.1"
      target_decode_host="127.0.0.1"
    else
      # ROLE=runner (assumes servers are launched on remote endpoints)
      target_prefill_host="${PREFILL_HOST}"
      target_decode_host="${DECODE_HOST}"
    fi

    # Wait for prefill and decode to be healthy
    echo "--- Verifying backend server health: Prefill (${target_prefill_host}:${PREFILL_PORT}), Decode (${target_decode_host}:${DECODE_PORT})"
    deadline=$((SECONDS + STARTUP_TIMEOUT_S))
    while [[ "${SECONDS}" -lt "${deadline}" ]]; do
      p_ok=0
      d_ok=0
      if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://${target_prefill_host}:${PREFILL_PORT}/health" 2>/dev/null || true)" == "200" ]]; then
        p_ok=1
      fi
      if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://${target_decode_host}:${DECODE_PORT}/health" 2>/dev/null || true)" == "200" ]]; then
        d_ok=1
      fi
      if [[ "${p_ok}" -eq 1 && "${d_ok}" -eq 1 ]]; then
        echo "Prefill and Decode backends are healthy!"
        break
      fi
      sleep 5
    done

    if [[ "${p_ok}" -ne 1 || "${d_ok}" -ne 1 ]]; then
      echo "ERROR: Backend servers did not become healthy within ${STARTUP_TIMEOUT_S}s (Prefill: ${p_ok}, Decode: ${d_ok})" >&2
      exit 1
    fi

    # Start toy proxy server
    echo "--- Starting toy proxy server on port ${PROXY_PORT}"
    proxy_cmd=(
      "${PYTHON_BIN}"
      "${repo_root}/examples/disagg/toy_proxy_server.py"
      --host "0.0.0.0"
      --port "${PROXY_PORT}"
      --prefiller-hosts "${target_prefill_host}"
      --prefiller-ports "${PREFILL_PORT}"
      --decoder-hosts "${target_decode_host}"
      --decoder-ports "${DECODE_PORT}"
    )
    setsid "${proxy_cmd[@]}" >"${RUN_DIR}/logs/proxy.log" 2>&1 &
    echo $! >"${RUN_DIR}/proxy.pid"
    echo "Proxy PID: $(cat "${RUN_DIR}/proxy.pid")"

    # Wait for proxy /healthcheck
    echo "--- Waiting for proxy /healthcheck"
    proxy_deadline=$((SECONDS + 60))
    proxy_ok=0
    while [[ "${SECONDS}" -lt "${proxy_deadline}" ]]; do
      if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PROXY_PORT}/healthcheck" 2>/dev/null || true)" == "200" ]]; then
        proxy_ok=1
        break
      fi
      sleep 2
    done
    if [[ "${proxy_ok}" -ne 1 ]]; then
      echo "ERROR: Proxy failed to become ready within 60s" >&2
      tail -n 50 "${RUN_DIR}/logs/proxy.log" || true
      exit 1
    fi
    echo "Proxy is healthy and ready."

    # Run prefix cache correctness test
    echo "--- Running smoke_prefix_cache_correctness.py"
    set +e
    "${PYTHON_BIN}" "${repo_root}/scripts/vllm/integration/smoke_prefix_cache_correctness.py" \
      --host "127.0.0.1" \
      --port "${PROXY_PORT}" \
      --model "${SERVED_MODEL_NAME}" \
      2>&1 | tee "${RUN_DIR}/logs/correctness.log"
    CORRECTNESS_EXIT_CODE=$?

    # Run prefix cache divergence test
    if [[ "${RUN_PREFIX_CACHE_E2E_DIVERGENCE}" == "1" ]]; then
      echo "--- Running smoke_prefix_cache_e2e_divergence.py"
      "${PYTHON_BIN}" "${repo_root}/scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py" \
        --host "127.0.0.1" \
        --port "${PROXY_PORT}" \
        --model "${SERVED_MODEL_NAME}" \
        --max-tokens 1 \
        --logprob-atol "${PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL}" \
        2>&1 | tee "${RUN_DIR}/logs/prefix_cache_e2e_divergence.log"
      DIVERGENCE_EXIT_CODE=$?
    else
      echo "RUN_PREFIX_CACHE_E2E_DIVERGENCE=${RUN_PREFIX_CACHE_E2E_DIVERGENCE}; skipping prefix-cache E2E divergence smoke"
      DIVERGENCE_EXIT_CODE=0
    fi

    # --- MTP (speculative decoding) health --------------------------------
    MTP_EXIT_CODE=0
    if [[ -n "${DECODE_SPECULATIVE_CONFIG+x}" && -z "${DECODE_SPECULATIVE_CONFIG}" ]]; then
      echo "DECODE_SPECULATIVE_CONFIG is explicitly empty; skipping MTP acceptance assertions"
    else
      echo "--- Checking MTP acceptance on decode (${target_decode_host}:${DECODE_PORT})"
      read_spec_counter() {
        # $1 = host, $2 = port, $3 = counter name without the _total suffix.
        # An unreachable server or a missing counter reads as 0, which the
        # draft-token check below rejects.
        curl -fsS "http://$1:$2/metrics" 2>/dev/null |
          awk -v want="^$3_total" '$0 ~ want { sum += $2 } END { printf "%.0f", sum + 0 }'
      }

      decode_draft_tokens="$(read_spec_counter "${target_decode_host}" "${DECODE_PORT}" vllm:spec_decode_num_draft_tokens)"
      decode_accepted_tokens="$(read_spec_counter "${target_decode_host}" "${DECODE_PORT}" vllm:spec_decode_num_accepted_tokens)"
      echo "MTP_DECODE_DRAFT_TOKENS ${decode_draft_tokens}"
      echo "MTP_DECODE_ACCEPTED_TOKENS ${decode_accepted_tokens}"

      # Observability only: prefill drafts to warm its own draft-layer KV but
      # never verifies, so this is expected to stay at 0. Printed, not asserted.
      echo "MTP_PREFILL_DRAFT_TOKENS $(read_spec_counter "${target_prefill_host}" "${PREFILL_PORT}" vllm:spec_decode_num_draft_tokens)"

      if [[ "${decode_draft_tokens}" -le 0 ]]; then
        echo "ERROR: Decode engine proposed no draft tokens; MTP never ran" >&2
        MTP_EXIT_CODE=1
      else
        decode_acceptance_rate="$(awk -v a="${decode_accepted_tokens}" \
          -v d="${decode_draft_tokens}" 'BEGIN { printf "%.4f", a / d }')"
        echo "MTP_DECODE_ACCEPTANCE_RATE ${decode_acceptance_rate}"
        if awk -v r="${decode_acceptance_rate}" -v m="${MIN_MTP_ACCEPTANCE_RATE}" \
            'BEGIN { exit !(r < m) }'; then
          echo "ERROR: Expected MTP acceptance rate >= ${MIN_MTP_ACCEPTANCE_RATE} on the" \
            "decode engine, got ${decode_acceptance_rate}" >&2
          MTP_EXIT_CODE=1
        else
          echo "MTP_ACCEPTANCE_OK 1"
        fi
      fi
    fi
    set -e

    chmod -R 777 /perf_eval_results 2>/dev/null || true

    if [[ "${CORRECTNESS_EXIT_CODE}" -ne 0 || "${DIVERGENCE_EXIT_CODE}" -ne 0 \
        || "${MTP_EXIT_CODE}" -ne 0 ]]; then
      echo "ERROR: Disaggregation smoke tests failed (correctness=${CORRECTNESS_EXIT_CODE}, divergence=${DIVERGENCE_EXIT_CODE}, mtp=${MTP_EXIT_CODE})" >&2
      exit 1
    fi

    echo "Disaggregation correctness smoke tests completed successfully."
    ;;

  *)
    echo "ERROR: Unknown ROLE: ${ROLE}. Supported: prefill, decode, runner, head, all." >&2
    exit 2
    ;;
esac

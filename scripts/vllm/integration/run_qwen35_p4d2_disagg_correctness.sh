#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

export PREFILL_PCP="${PREFILL_PCP:-4}"
export DECODE_DP="${DECODE_DP:-2}"
# shellcheck source=scripts/vllm/integration/configs/qwen35_disagg.sh
source "${script_dir}/configs/qwen35_disagg.sh"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_p4d2_disagg_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
P4D2_BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
VLLM_SRC="${VLLM_SRC:-${repo_root}}"
TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
# 单机 CI 使用较小的 per-worker premapped pool，缩短 libtpu 启动预留时间。
TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-8589934592}"
# Retain the main-branch regression floor; remeasure for TP2/DP2 on TPU.
MIN_MTP_ACCEPTANCE_RATE="${MIN_MTP_ACCEPTANCE_RATE:-0.70}"

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

# Dynamically extract the exact vLLM version installed in the container
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')}"
export EXPECTED_VLLM_VERSION

export SERVED_MODEL_NAME
export RUN_ROOT
export RUN_DIR
export P4D2_BIND_HOST
export PREFILL_PORT
export DECODE_PORT
export PROXY_PORT
export VLLM_SRC
export TORCHTPU_VLLM_SRC
export USE_CURRENT_PY_ENV
export TPU_PREMAPPED_BUFFER_SIZE

export P4D2_SHORT_REPEAT_LINES="${P4D2_SHORT_REPEAT_LINES:-32}"
export P4D2_SHORT_REPEAT_COUNT="${P4D2_SHORT_REPEAT_COUNT:-3}"
export P4D2_LONG_REPEAT_LINES="${P4D2_LONG_REPEAT_LINES:-96}"
export P4D2_LONG_REPEAT_COUNT="${P4D2_LONG_REPEAT_COUNT:-3}"
# With the Qwen3.5 chat template this is about 8.7k prompt tokens.  After the
# speculative-decoding tail reservation, 768-token alignment lands at 7680
# (not a 3072-token scheduler boundary), while scheduler alignment lands at
# 3072.  This makes the request sensitive to physical-vs-scheduler alignment.
export P4D2_PREFIX_HIT_PROBE_LINES="${P4D2_PREFIX_HIT_PROBE_LINES:-272}"
export P4D2_PREFIX_HIT_PROBE_COUNT="${P4D2_PREFIX_HIT_PROBE_COUNT:-3}"
export P4D2_LONG_SHARED_LINES="${P4D2_LONG_SHARED_LINES:-72}"
export P4D2_LONG_SHARED_ROUNDS="${P4D2_LONG_SHARED_ROUNDS:-2}"
export P4D2_MIXED_LINES="${P4D2_MIXED_LINES:-72}"
export P4D2_MIXED_ROUNDS="${P4D2_MIXED_ROUNDS:-2}"
export P4D2_CONCURRENT_REQUESTS="${P4D2_CONCURRENT_REQUESTS:-1}"
export RUN_PREFIX_CACHE_E2E_DIVERGENCE="${RUN_PREFIX_CACHE_E2E_DIVERGENCE:-1}"

cd "${repo_root}"

bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh \
  2>&1 | tee "${RUN_DIR}/logs/launch.log"

if ! grep -Fqx \
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}" \
    "${RUN_DIR}/launch_params.txt"; then
  echo "Expected VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL} in launch_params.txt" >&2
  exit 1
fi
echo "PREFIX_CACHE_RETENTION_INTERVAL_OK ${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}"

if ! grep -Fqx \
    "TPU_RAIDEN_PREFIX_AWARE_LOAD=${TPU_RAIDEN_PREFIX_AWARE_LOAD}" \
    "${RUN_DIR}/launch_params.txt"; then
  echo "Expected TPU_RAIDEN_PREFIX_AWARE_LOAD=${TPU_RAIDEN_PREFIX_AWARE_LOAD} in launch_params.txt" >&2
  exit 1
fi
echo "RAIDEN_PREFIX_AWARE_LOAD_OK ${TPU_RAIDEN_PREFIX_AWARE_LOAD}"

expected_prefill_scheduler_block_size=$((PREFILL_BLOCK_SIZE * PREFILL_PCP))
expected_decode_scheduler_block_size="${DECODE_BLOCK_SIZE}"
prefill_resolution="scheduler_block_size=${expected_prefill_scheduler_block_size} hash_block_size=${expected_prefill_scheduler_block_size}"

if ! grep -Fq "${prefill_resolution}" "${RUN_DIR}/logs/prefill.log"; then
  echo "Expected Prefill ${prefill_resolution}, but it was not found in the server log" >&2
  exit 1
fi
if ! grep -Fq "Using KV cache block size: ${DECODE_BLOCK_SIZE}" \
    "${RUN_DIR}/logs/decode.log"; then
  echo "Expected Decode physical/scheduler block size ${DECODE_BLOCK_SIZE}, but it was not found in the server log" >&2
  exit 1
fi
echo "SCHEDULER_BLOCK_SIZE_OK prefill=${expected_prefill_scheduler_block_size} decode=${expected_decode_scheduler_block_size}"

read_prefill_prefix_cache_hits() {
  curl -fsS "http://${P4D2_BIND_HOST}:${PREFILL_PORT}/metrics" |
    awk '/^vllm:prefix_cache_hits_total/ {sum += $2} END {printf "%.0f", sum}'
}

if ((P4D2_PREFIX_HIT_PROBE_COUNT < 2)); then
  echo "P4D2_PREFIX_HIT_PROBE_COUNT must be at least 2" >&2
  exit 2
fi
# The runner starts fresh servers; fixed prompt namespaces make acceptance
# comparable across independent cold starts without reusing a live cache.
prefix_hit_probe_namespace="pcp4-dp4-retention-v1"
if [[ "${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}" == "0" ]]; then
  # Zero retention starts with no periodic Mamba checkpoints. Prime the full
  # attention cache once so the next request discovers the shared-prefix
  # junction and materializes the adaptive Mamba checkpoint. Keep this setup
  # request outside the measured interval: the existing assertion then still
  # requires every repeat after the adaptive request to hit one scheduler block.
  python scripts/vllm/integration/smoke_prefix_cache_correctness.py \
    --host "${P4D2_BIND_HOST}" \
    --port "${PROXY_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --namespace "${prefix_hit_probe_namespace}" \
    --prefix-hit-probe-only \
    --long-repeat-lines "${P4D2_PREFIX_HIT_PROBE_LINES}" \
    --long-repeat-count 1 \
    2>&1 | tee "${RUN_DIR}/logs/prefix_cache_retention_warmup.log"
  echo "PREFIX_CACHE_RETENTION_WARMUP_OK 1"
fi
prefill_prefix_cache_hits_before="$(read_prefill_prefix_cache_hits)"
python scripts/vllm/integration/smoke_prefix_cache_correctness.py \
  --host "${P4D2_BIND_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --namespace "${prefix_hit_probe_namespace}" \
  --prefix-hit-probe-only \
  --long-repeat-lines "${P4D2_PREFIX_HIT_PROBE_LINES}" \
  --long-repeat-count "${P4D2_PREFIX_HIT_PROBE_COUNT}" \
  2>&1 | tee "${RUN_DIR}/logs/prefix_cache_hit_probe.log"
prefill_prefix_cache_hits_after="$(read_prefill_prefix_cache_hits)"
prefill_prefix_cache_hits_delta=$((
  prefill_prefix_cache_hits_after - prefill_prefix_cache_hits_before
))
expected_prefill_prefix_cache_hits_delta=$((
  expected_prefill_scheduler_block_size * (P4D2_PREFIX_HIT_PROBE_COUNT - 1)
))
echo "PREFILL_PREFIX_CACHE_HITS_DELTA ${prefill_prefix_cache_hits_delta}"
if ((prefill_prefix_cache_hits_delta != expected_prefill_prefix_cache_hits_delta)); then
  echo "Expected Prefill local prefix-cache hit delta ${expected_prefill_prefix_cache_hits_delta}, got ${prefill_prefix_cache_hits_delta}" >&2
  exit 1
fi
echo "PREFILL_PREFIX_CACHE_HIT_OK 1"

python scripts/vllm/integration/smoke_prefix_cache_correctness.py \
  --host "${P4D2_BIND_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --namespace pcp4-dp4-correctness-v1 \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"

if [[ "${RUN_PREFIX_CACHE_E2E_DIVERGENCE}" == "1" ]]; then
  python scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py \
    --host "${P4D2_BIND_HOST}" \
    --port "${PROXY_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --max-tokens 1 \
    --run-id pcp4-dp4-v1 \
    --no-logprobs \
    2>&1 | tee "${RUN_DIR}/logs/prefix_cache_e2e_divergence.log"
else
  echo "RUN_PREFIX_CACHE_E2E_DIVERGENCE=${RUN_PREFIX_CACHE_E2E_DIVERGENCE}; skip prefix-cache E2E divergence smoke"
fi

# --- MTP (speculative decoding) health -------------------------------------
if [[ -z "${DECODE_SPECULATIVE_CONFIG}" ]]; then
  echo "DECODE_SPECULATIVE_CONFIG is empty; skip MTP acceptance assertions"
else
  read_spec_counter() {
    # $1 = port, $2 = counter name without the Prometheus _total suffix.
    # Missing counters read as 0, which the draft-token check below rejects.
    curl -fsS "http://${P4D2_BIND_HOST}:$1/metrics" |
      awk -v want="^$2_total" '$0 ~ want { sum += $2 } END { printf "%.0f", sum + 0 }'
  }

  decode_draft_tokens="$(read_spec_counter "${DECODE_PORT}" vllm:spec_decode_num_draft_tokens)"
  decode_accepted_tokens="$(read_spec_counter "${DECODE_PORT}" vllm:spec_decode_num_accepted_tokens)"
  echo "MTP_DECODE_DRAFT_TOKENS ${decode_draft_tokens}"
  echo "MTP_DECODE_ACCEPTED_TOKENS ${decode_accepted_tokens}"

  # Observability only: the prefill engine drafts to warm its own draft-layer
  # KV but never verifies (the proxy caps it at max_tokens=1), so this is
  # expected to stay at 0. Printed, not asserted.
  echo "MTP_PREFILL_DRAFT_TOKENS $(read_spec_counter "${PREFILL_PORT}" vllm:spec_decode_num_draft_tokens)"

  if ((decode_draft_tokens <= 0)); then
    echo "Decode engine proposed no draft tokens; MTP never ran" >&2
    exit 1
  fi

  decode_acceptance_rate="$(awk -v a="${decode_accepted_tokens}" \
    -v d="${decode_draft_tokens}" 'BEGIN { printf "%.4f", a / d }')"
  echo "MTP_DECODE_ACCEPTANCE_RATE ${decode_acceptance_rate}"

  if awk -v a="${decode_accepted_tokens}" -v d="${decode_draft_tokens}" \
      -v m="${MIN_MTP_ACCEPTANCE_RATE}" 'BEGIN { exit !(a / d < m) }'; then
    echo "Expected MTP acceptance rate >= ${MIN_MTP_ACCEPTANCE_RATE} on the" \
      "decode engine, got ${decode_acceptance_rate}" >&2
    exit 1
  fi
  echo "MTP_ACCEPTANCE_OK 1"
fi

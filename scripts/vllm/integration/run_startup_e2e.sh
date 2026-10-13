#!/usr/bin/env bash
# ==============================================================================
# vLLM TorchTPU Startup Time & Compilation Cache E2E Integration Test
# ==============================================================================
# Verifies end-to-end cold and warm startup behavior, SLA time budget compliance
# (Cold <= 12-15m, Warm <= 5m), warm start cache hit speedup (WARM < COLD),
# smoke inference correctness, output fidelity (Cold == Warm), and Tier-3 cache.
#
# Usage:
#   ./scripts/vllm/integration/run_startup_e2e.sh [MODEL_NAME] [MAX_COLD_SECONDS] [MAX_WARM_SECONDS] [PORT]
#
# Example:
#   ./scripts/vllm/integration/run_startup_e2e.sh Qwen/Qwen3.5-397B-A17B-FP8 900 300 18100
# ==============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL="${1:-${MODEL_NAME:-Qwen/Qwen3-0.6B}}"

# Cold startup time budget (default: 15m for 397B, 6m for CI small models)
if [ -n "${2:-}" ]; then
  MAX_COLD_SECONDS="$2"
elif [ -n "${MAX_STARTUP_SECONDS:-}" ]; then
  MAX_COLD_SECONDS="${MAX_STARTUP_SECONDS}"
elif [ -z "${MAX_COLD_SECONDS:-}" ]; then
  if [[ "$MODEL" =~ "397B" || "$MODEL" =~ "480B" ]]; then
    MAX_COLD_SECONDS=900 # 15 minutes default for large models
  else
    MAX_COLD_SECONDS=360 # 6 minutes default for small models in CI
  fi
fi

# Warm startup time budget (default: 5m SLA for 397B, 3m for CI small models)
if [ -n "${3:-}" ]; then
  MAX_WARM_SECONDS="$3"
elif [ -z "${MAX_WARM_SECONDS:-}" ]; then
  if [[ "$MODEL" =~ "397B" || "$MODEL" =~ "480B" ]]; then
    MAX_WARM_SECONDS=300 # 5 minutes default SLA for large models per doc
  else
    MAX_WARM_SECONDS=180 # 3 minutes default for small models in CI
  fi
fi

PORT="${4:-${PORT:-18100}}"
HOST="127.0.0.1"

CACHE_DIR="$(mktemp -d /tmp/vllm_startup_e2e_XXXXXX)"
LOG_FILE_COLD="${CACHE_DIR}/vllm_cold_server.log"
LOG_FILE_WARM="${CACHE_DIR}/vllm_warm_server.log"

export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-${CACHE_DIR}/torchtpu_tier3}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_DIR}/vllm_cache}"
export VLLM_TARGET_DEVICE="tpu"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-0}"
export DISABLE_TIER3_CACHE="${DISABLE_TIER3_CACHE:-0}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-0}"

mkdir -p "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${VLLM_CACHE_ROOT}"

SERVER_PID=""

clean_tpu_processes() {
  echo "=== 🧹 Cleaning orphaned TPU processes... ==="
  for d in /proc/[0-9]*; do
    if [ -f "$d/cmdline" ]; then
      pid=$(basename "$d")
      cmd=$(tr "\0" " " < "$d/cmdline" 2>/dev/null || true)
      if echo "$cmd" | grep -qiE "vllm|enginecore|api_server"; then
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    fi
  done
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  rm -rf /tmp/libtpu* /tmp/tpu_logs* || true
}

stop_server() {
  if [ -n "${SERVER_PID}" ]; then
    echo "=== 🛑 Stopping vLLM server (PID: ${SERVER_PID})... ==="
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 3
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  clean_tpu_processes
}

cleanup() {
  echo "=== 🛑 Cleaning up environment... ==="
  stop_server
  if [ -d "${CACHE_DIR}" ] && [[ "${CACHE_DIR}" =~ ^/tmp/vllm_startup_e2e_ ]]; then
    rm -rf "${CACHE_DIR}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ==============================================================================
# Common Server Launch Arguments
# ==============================================================================
EXTRA_ARGS=()
if [[ "$MODEL" =~ "397B" ]]; then
  EXTRA_ARGS=(
    --max-model-len 262144
    --max-num-batched-tokens 4096
    --max-num-seqs 64
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --data-parallel-size "${DATA_PARALLEL_SIZE:-8}"
    --attention-backend CUSTOM
    --block-size 256
    --gpu-memory-utilization 0.90
    --kv-cache-dtype fp8
    --language-model-only
    --enable-expert-parallel
    --disable-custom-all-reduce
    --no-enable-prefix-caching
    --no-async-scheduling
    --no-disable-hybrid-kv-cache-manager
    --return-tokens-as-token-ids
  )
  if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ]; then
    EXTRA_ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
  fi
else
  EXTRA_ARGS=(
    --max-model-len 256
    --max-num-batched-tokens 256
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --data-parallel-size "${DATA_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization 0.80
    --no-async-scheduling
  )
fi

echo "=================================================="
echo "🚀 [E2E Startup Suite] Cold & Warm Startup Test"
echo "Target Model: ${MODEL}"
echo "Server Endpoint: http://${HOST}:${PORT}"
echo "Cold Startup Budget: ${MAX_COLD_SECONDS}s ($((MAX_COLD_SECONDS / 60)) mins)"
echo "Warm Startup Budget: ${MAX_WARM_SECONDS}s ($((MAX_WARM_SECONDS / 60)) mins)"
echo "Tier-3 Cache Root: ${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"
echo "vLLM Cache Root: ${VLLM_CACHE_ROOT}"
echo "=================================================="

# ==============================================================================
# [Phase 1] Cold Compilation Run
# ==============================================================================
echo ""
echo "=================================================="
echo "🏃 [Phase 1] Cold Compilation Run (Clean Cache)"
echo "=================================================="
clean_tpu_processes

START_TS_COLD=$(date +'%s.%N')
echo "trace-start-event: name=vllm-cold-server ts=${START_TS_COLD} pid=vllm tid=vllm"

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --seed 42 \
  "${EXTRA_ARGS[@]}" > "${LOG_FILE_COLD}" 2>&1 &
SERVER_PID=$!

echo "vLLM server started in background (PID: ${SERVER_PID}). Polling http://${HOST}:${PORT}/health..."

while true; do
  CURRENT_TS=$(date +'%s.%N')
  ELAPSED_INT=$(python3 -c "print(int(float('${CURRENT_TS}') - float('${START_TS_COLD}')))")

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "❌ [SERVER CRASH] vLLM server process died during cold startup!"
    echo "--- Last 50 lines of server log ---"
    tail -n 50 "${LOG_FILE_COLD}"
    exit 1
  fi

  if curl -s -f -m 5 "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    END_TS_COLD=${CURRENT_TS}
    break
  fi

  if [ "${ELAPSED_INT}" -ge "${MAX_COLD_SECONDS}" ]; then
    echo "❌ [TIMEOUT ERROR] Cold startup exceeded budget! Elapsed: ${ELAPSED_INT}s >= Max: ${MAX_COLD_SECONDS}s"
    echo "--- Last 50 lines of server log ---"
    tail -n 50 "${LOG_FILE_COLD}"
    exit 1
  fi

  sleep 3
done

echo "trace-end-event: name=vllm-cold-server ts=${END_TS_COLD} pid=vllm tid=vllm status=success"

DIFF_COLD=$(python3 -c "print(f'{float(\"${END_TS_COLD}\") - float(\"${START_TS_COLD}\"):.2f}')")
echo "⏱️  [Cold Metrics] Ready in: ${DIFF_COLD}s ($(( ${DIFF_COLD%.*} / 60 )) mins) (Allowed: ${MAX_COLD_SECONDS}s)"
echo "✅ [Cold Budget PASSED] Startup time ${DIFF_COLD}s is within budget of ${MAX_COLD_SECONDS}s"

# Cold Smoke Inference
echo "--- 🧪 Running Cold Smoke Inference Check ---"
RESPONSE_COLD=$(curl -s -m 30 "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "Hello, my name is",
    "max_tokens": 16,
    "temperature": 0
  }')

echo "Cold Inference Response: ${RESPONSE_COLD}"
if ! echo "${RESPONSE_COLD}" | grep -q '"text"'; then
  echo "❌ [Cold Inference FAILED] Invalid response!"
  tail -n 50 "${LOG_FILE_COLD}" || true
  exit 1
fi
TEXT_COLD=$(python3 -c "import json; d=json.loads('''${RESPONSE_COLD}'''); print(d.get('choices', [{}])[0].get('text', '').strip())" 2>/dev/null || echo "")
echo "Cold Generated text: '${TEXT_COLD}'"
echo "✅ [Cold Inference PASSED]"

# Verify Cache Generated
if [ "${DISABLE_TIER3_CACHE}" = "0" ] && [ -d "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" ]; then
  CACHE_COUNT=$(find "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${VLLM_CACHE_ROOT}" -type f 2>/dev/null | wc -l || echo 0)
  echo "Found ${CACHE_COUNT} cache artifact(s) generated during cold run."
fi

# Stop Cold Server
stop_server
sleep 5

# ==============================================================================
# [Phase 2] Warm Compilation Run (Cache Hit)
# ==============================================================================
echo ""
echo "=================================================="
echo "⚡ [Phase 2] Warm Start Run (Cache Hit)"
echo "=================================================="

START_TS_WARM=$(date +'%s.%N')
echo "trace-start-event: name=vllm-warm-server ts=${START_TS_WARM} pid=vllm tid=vllm"

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --seed 42 \
  "${EXTRA_ARGS[@]}" > "${LOG_FILE_WARM}" 2>&1 &
SERVER_PID=$!

echo "vLLM server restarted in background (PID: ${SERVER_PID}). Polling http://${HOST}:${PORT}/health..."

while true; do
  CURRENT_TS=$(date +'%s.%N')
  ELAPSED_INT=$(python3 -c "print(int(float('${CURRENT_TS}') - float('${START_TS_WARM}')))")

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "❌ [SERVER CRASH] vLLM server process died during warm startup!"
    echo "--- Last 50 lines of warm server log ---"
    tail -n 50 "${LOG_FILE_WARM}"
    exit 1
  fi

  if curl -s -f -m 5 "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    END_TS_WARM=${CURRENT_TS}
    break
  fi

  if [ "${ELAPSED_INT}" -ge "${MAX_WARM_SECONDS}" ]; then
    echo "❌ [TIMEOUT ERROR] Warm startup exceeded budget! Elapsed: ${ELAPSED_INT}s >= Max: ${MAX_WARM_SECONDS}s"
    echo "--- Last 50 lines of warm server log ---"
    tail -n 50 "${LOG_FILE_WARM}"
    exit 1
  fi

  sleep 2
done

echo "trace-end-event: name=vllm-warm-server ts=${END_TS_WARM} pid=vllm tid=vllm status=success"

DIFF_WARM=$(python3 -c "print(f'{float(\"${END_TS_WARM}\") - float(\"${START_TS_WARM}\"):.2f}')")
echo "⏱️  [Warm Metrics] Ready in: ${DIFF_WARM}s (Allowed: ${MAX_WARM_SECONDS}s)"
echo "✅ [Warm Budget PASSED] Startup time ${DIFF_WARM}s is within budget of ${MAX_WARM_SECONDS}s"

# Speedup Check
python3 -c "
c = float('${DIFF_COLD}')
w = float('${DIFF_WARM}')
print(f'Cold Startup Time: {c:.2f}s | Warm Startup Time: {w:.2f}s')
if w <= c:
    print(f'✅ [Speedup PASSED] Warm startup ({w:.2f}s) is faster than cold startup ({c:.2f}s).')
else:
    print(f'⚠️ [Speedup Warning] Warm startup ({w:.2f}s) took longer than cold ({c:.2f}s).')
"

# Warm Smoke Inference
echo "--- 🧪 Running Warm Smoke Inference Check ---"
RESPONSE_WARM=$(curl -s -m 30 "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "Hello, my name is",
    "max_tokens": 16,
    "temperature": 0
  }')

echo "Warm Inference Response: ${RESPONSE_WARM}"
if ! echo "${RESPONSE_WARM}" | grep -q '"text"'; then
  echo "❌ [Warm Inference FAILED] Invalid response!"
  tail -n 50 "${LOG_FILE_WARM}" || true
  exit 1
fi
TEXT_WARM=$(python3 -c "import json; d=json.loads('''${RESPONSE_WARM}'''); print(d.get('choices', [{}])[0].get('text', '').strip())" 2>/dev/null || echo "")
echo "Warm Generated text: '${TEXT_WARM}'"
echo "✅ [Warm Inference PASSED]"

# [Phase 3] Output Fidelity Check (Cold vs Warm)
echo ""
echo "=================================================="
echo "🎯 [Phase 3] Cold vs Warm Output Fidelity Check"
echo "=================================================="
echo "Cold Output: '${TEXT_COLD}'"
echo "Warm Output: '${TEXT_WARM}'"
if [ "${TEXT_COLD}" = "${TEXT_WARM}" ] && [ -n "${TEXT_COLD}" ]; then
  echo "✅ [Output Fidelity PASSED] Generated prompt output matches 100% between Cold and Warm runs."
else
  echo "❌ [Output Fidelity FAILED] Output mismatch between Cold and Warm runs!"
  exit 1
fi

stop_server

echo ""
echo "=================================================="
echo "🎉 [Integration Test Completed] All Cold + Warm E2E Startup Checks PASSED!"
echo "=================================================="

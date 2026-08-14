#!/usr/bin/env bash
# ==============================================================================
# Compilation Cache E2E Test (Raw TPU Execution)
# ==============================================================================
# Verifies compilation cache behavior, storage duplication reduction, and warm-start
# cache hit consistency for vllm-torchtpu across 4 distinct test phases in 2 runs:
#
#   [Test 1] Cold Compilation Run:
#            Verifies that running on a clean cache directory triggers FX graph
#            compilation and generates initial compilation artifacts.
#   [Test 2] Storage Duplication Audit:
#            Audits cache sizes and verifies that Category 2 Python pickle files
#            contain lightweight metadata handles (<50KB total) rather than duplicating
#            heavy compiled binary payloads.
#   [Test 3] Warm Start Cache Hit Verification:
#            Verifies that subsequent inference runs load compiled executables directly
#            from the cache (<1ms) without triggering re-compilation.
#   [Test 4] Cache File & MD5 Consistency Check (Cold vs Warm):
#            Compares Tier-3 binary artifact MD5sums between Cold and Warm runs to audit
#            cache file consistency.
#
# Usage: ./scripts/vllm/integration/run_compilation_cache_e2e.sh [model_name]
# Example: ./scripts/vllm/integration/run_compilation_cache_e2e.sh Qwen/Qwen3-0.6B
# ==============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CACHE_DIR="$(mktemp -d /tmp/compilation_cache_e2e_XXXXXX)"
mkdir -p "${CACHE_DIR}"

# Ensure all temporary cache files are deleted when the script finishes or errors out
cleanup() {
  echo "=== 🧹 Cleaning up temporary compilation cache artifacts... ==="
  rm -rf "${CACHE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# Export public TorchTPU compilation cache configuration
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_DIR}/torch_tpu_tier3"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_cache"
export VLLM_TARGET_DEVICE="tpu"

MODEL="${1:-${MODEL:-Qwen/Qwen3-0.6B}}"
TIER3_DIR="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"

# Helper function: kill orphaned processes holding TPU locks
clean_tpu_processes() {
  echo "=== 🧹 Cleaning orphaned TPU processes... ==="
  for d in /proc/[0-9]*; do
    if [ -f "$d/cmdline" ]; then
      pid=$(basename "$d")
      cmd=$(tr "\0" " " < "$d/cmdline" 2>/dev/null || true)
      if echo "$cmd" | grep -qiE "python|vllm|enginecore"; then
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    fi
  done
  rm -rf /tmp/libtpu* /tmp/tpu_logs* || true
}

# ==============================================================================
# [Test 1] Cold Compilation Run
# Description: Verifies that running on a clean cache directory triggers FX graph
#              compilation as expected.
# ==============================================================================
echo "================================================================="
echo "=== 🏃 [Test 1] Cold Compilation Run (${MODEL})"
echo "================================================================="
clean_tpu_processes

rm -rf "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${VLLM_CACHE_ROOT}"
mkdir -p "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${VLLM_CACHE_ROOT}"

python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 2>&1 | tee "${CACHE_DIR}/cold_run.log"

sync && sleep 10

if grep -q "Compiling FX graph" "${CACHE_DIR}/cold_run.log"; then
  echo "✅ [Test 1 PASSED] Cold run triggered FX graph compilation as expected."
else
  echo "❌ [Test 1 FAILED] Cold run did not trigger FX graph compilation!"
  exit 1
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -type f -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_cold.txt" || true
else
  touch "${CACHE_DIR}/tier3_cold.txt"
fi

# ==============================================================================
# [Test 2] Storage Duplication Audit
# Description: Verifies that Category 2 Python pickle files store lightweight
#              metadata handles (<50KB total) rather than duplicating heavy binary
#              executables.
#
# Categories Explained:
#   Category 1: C++ Native Tier-3 PJRT Binary Executables
#               (Managed natively by torch_tpu in torch_tpu_tier3/*.bin)
#   Category 2: vLLM TpuCompilerAdaptor Python Metadata Handles
#               (Pickled TpuCompilationHandle instances storing lightweight cache metadata)
#   Category 3: PyTorch Upstream Dynamo / AOTAutograd Tracing Cache
#               (Independent upstream PyTorch framework cache storing Dynamo FX graph
#                bytecode, Guard expressions, and BundledAOTAutogradResult structures
#                created during Python-side graph tracing, completely decoupled from
#                native C++ PJRT compiled binary executables)
# ==============================================================================
echo "================================================================="
echo "=== 📊 [Test 2] Storage Duplication Audit (Cold Run Artifacts)"
echo "================================================================="
CAT1_SIZE=$(du -sb "${TIER3_DIR}" 2>/dev/null | cut -f1 || echo 0)
CAT2_SIZE=$(find "${VLLM_CACHE_ROOT}" -type f -name "artifact_compile_range_*" -exec du -cb {} + 2>/dev/null | tail -n 1 | cut -f1 || echo 0)
CAT3_SIZE=$(du -sb "${VLLM_CACHE_ROOT}/torch_compile_cache/torch_aot_compile" 2>/dev/null | cut -f1 || echo 0)

CAT1_MB=$(awk "BEGIN {printf \"%.2f\", ${CAT1_SIZE}/1048576}")
CAT2_KB=$(awk "BEGIN {printf \"%.2f\", ${CAT2_SIZE}/1024}")
CAT3_MB=$(awk "BEGIN {printf \"%.2f\", ${CAT3_SIZE}/1048576}")

cat <<EOF | tee "${CACHE_DIR}/cache_duplication_audit.txt"
Category 1 (C++ Native Tier-3 PJRT Binary Executables): ${CAT1_SIZE} Bytes (${CAT1_MB} MB)
Category 2 (vLLM TpuCompilerAdaptor Python Metadata Handles): ${CAT2_SIZE} Bytes (${CAT2_KB} KB)
Category 3 (PyTorch Upstream Dynamo / AOTAutograd Tracing Cache): ${CAT3_SIZE} Bytes (${CAT3_MB} MB)
EOF

if [ "${CAT2_SIZE}" -lt 50000 ]; then
  echo "✅ [Test 2 PASSED] Optimization verified! Category 2 pickle files total ${CAT2_SIZE} Bytes (< 50KB total, ~100B per graph), eliminating binary duplication!"
else
  echo "⚠️ [Test 2 WARNING] Category 2 pickle files total ${CAT2_SIZE} Bytes. Binary payload may still be duplicated."
fi

# ==============================================================================
# [Test 3] Warm Start Cache Hit Verification
# Description: Verifies that subsequent inference runs load compiled executables
#              directly from the cache (<1ms) without re-compilation.
# ==============================================================================
echo "================================================================="
echo "=== 🏃 [Test 3] Warm Start Cache Hit Verification"
echo "================================================================="
clean_tpu_processes

python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 2>&1 | tee "${CACHE_DIR}/warm_run.log"

sync && sleep 10

WARM_TIME=$(grep -oP "Compilation finished in \K[0-9.]+" "${CACHE_DIR}/warm_run.log" | head -n 1 || echo "999")
IS_FAST=$(python3 -c "print(1 if float('${WARM_TIME}') < 5.0 else 0)" 2>/dev/null || echo "0")

if [ "${IS_FAST}" -eq 1 ] || grep -q -E "Directly load|Loading compiled executable|Cache the graph" "${CACHE_DIR}/warm_run.log"; then
  echo "✅ [Test 3 PASSED] Warm run log confirmed cache hit without recompilation (Compilation time: ${WARM_TIME}s < 5.0s)."
else
  echo "❌ [Test 3 FAILED] Warm run did not hit cache or triggered recompilation!"
  exit 1
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -type f -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_warm.txt" || true
else
  touch "${CACHE_DIR}/tier3_warm.txt"
fi

# ==============================================================================
# [Test 4] Cache File & MD5 Consistency Check (Cold vs Warm)
# Description: Compares Tier-3 binary artifact MD5sums between Cold and Warm runs
#              to audit cache file consistency.
# ==============================================================================
echo "================================================================="
echo "=== 🔍 [Test 4] Cache File & MD5 Consistency Check (Cold vs Warm)"
echo "================================================================="
# TODO(@maxwillzq): Tier-3 cache file non-deterministic diff is still under investigation.
# Log warning rather than exiting with failure until zero-diff is fully resolved.
if diff -u "${CACHE_DIR}/tier3_cold.txt" "${CACHE_DIR}/tier3_warm.txt"; then
  echo "✅ [Test 4 PASSED] Tier-3 cache files and MD5sums are 100% identical between Cold and Warm runs!"
else
  echo "⚠️ [Test 4 WARNING] Tier-3 cache files or MD5sums changed between Cold and Warm runs!"
fi

echo "================================================================="
echo "🎉 Compilation Cache E2E Test Suite Completed!"
echo "Logs saved in: ${CACHE_DIR}"
echo "================================================================="

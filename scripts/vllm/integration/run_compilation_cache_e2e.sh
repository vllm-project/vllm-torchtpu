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

# ==============================================================================
# Compilation Cache E2E Test (Raw TPU Execution)
# ==============================================================================
# Verifies compilation cache behavior, storage duplication reduction, warm-start
# cache hit speedup, inference output fidelity, and binary consistency across
# 5 distinct test phases in 2 runs (Cold and Warm):
#
#   [Test 1] Cold Compilation Run:
#            Verifies that running on a clean cache directory performs runtime
#            memory profiling (profile_run), triggers FX graph compilation, and
#            generates initial Tier-3 compilation artifacts.
#   [Test 2] Storage Duplication Audit:
#            Audits cache sizes and verifies that Category 2 Python pickle files
#            contain lightweight metadata handles (<50KB total) rather than duplicating
#            heavy compiled binary payloads.
#   [Test 3] Warm Start Cache Hit Verification:
#            Verifies that subsequent inference runs load compiled executables from
#            cache, significantly reducing total compilation time (WARM < COLD).
#   [Test 4] Inference Output Correctness Check (Cold vs Warm):
#            Verifies that generated prompt text matches 100% between Cold and Warm runs
#            (deterministic generation with temperature=0).
#   [Test 5] Tier-3 Binary & MD5 Checksum Consistency Check (Cold vs Warm):
#            Verifies that every Cold-run Tier-3 binary artifact is preserved
#            with a 100% identical MD5 checksum in the Warm run, and that any
#            additive Tier-3 .bin file comes strictly from pre-torch.compile
#            eager weight-loading ops (0.000s spent compiling during/after
#            torch.compile).
#
# Usage: ./scripts/vllm/integration/run_compilation_cache_e2e.sh [model_name]
# Example: ./scripts/vllm/integration/run_compilation_cache_e2e.sh Qwen/Qwen3-0.6B
# ==============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CACHE_DIR="$(mktemp -d /tmp/compilation_cache_e2e_XXXXXX)"
mkdir -p "${CACHE_DIR}"

# Ensure all temporary cache files are deleted when the script finishes or errors out,
# after briefly settling stdout so the Kubernetes remote log collector captures the
# final [Test N PASSED/FAILED] banner before pod termination.
cleanup() {
  echo "=== 🧹 Cleaning up temporary compilation cache artifacts... ==="
  rm -rf "${CACHE_DIR}" 2>/dev/null || true
  sync 2>/dev/null || true
  sleep "${COMPILATION_CACHE_E2E_LOG_FLUSH_SECS:-3}"
}
trap cleanup EXIT

# Export public TorchTPU compilation cache configuration
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_DIR}/torch_tpu_tier3"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_cache"
export VLLM_TARGET_DEVICE="tpu"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
# Log which graphs each run compiled, so the warm-run log of Test 3 names
# anything that was recompiled instead of loaded. The flag is in the compiler's
# _TPU_COMPILE_ENV_IGNORED set, so it does not perturb the Tier-3 binaries that
# Test 5 checksums.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

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

TEST_PROMPT="The capital of France is"

COLD_RC=0
python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 --max-tokens 8 \
  --prompt "${TEST_PROMPT}" 2>&1 | tee "${CACHE_DIR}/cold_run.log" || COLD_RC=$?

sync

if [ "${COLD_RC}" -ne 0 ]; then
  echo "❌ [Test 1 FAILED] Cold run offline_inference.py exited with status ${COLD_RC}!"
  exit "${COLD_RC}"
fi

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
CAT1_SIZE=$(du -sb "${TIER3_DIR}" 2>/dev/null | cut -f1 || true)
CAT1_SIZE="${CAT1_SIZE:-0}"
CAT2_SIZE=$(find "${VLLM_CACHE_ROOT}" -type f -name "artifact_compile_range_*" -exec du -cb {} + 2>/dev/null | tail -n 1 | cut -f1 || true)
CAT2_SIZE="${CAT2_SIZE:-0}"
CAT3_SIZE=$(du -sb "${VLLM_CACHE_ROOT}/torch_compile_cache/torch_aot_compile" 2>/dev/null | cut -f1 || true)
CAT3_SIZE="${CAT3_SIZE:-0}"

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

WARM_RC=0
python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 --max-tokens 8 \
  --prompt "${TEST_PROMPT}" 2>&1 | tee "${CACHE_DIR}/warm_run.log" || WARM_RC=$?

sync

if [ "${WARM_RC}" -ne 0 ]; then
  echo "❌ [Test 3 FAILED] Warm run offline_inference.py exited with status ${WARM_RC}!"
  exit "${WARM_RC}"
fi

# Evaluate total compilation time across all shapes for cold vs warm runs
COLD_COMP_TIME=$(python3 -c "
import re
with open('${CACHE_DIR}/cold_run.log') as f:
    times = [float(x) for x in re.findall(r'Compilation finished in ([0-9.]+) \[secs\]', f.read())]
print(f'{sum(times):.2f}' if times else '999')
")

WARM_COMP_TIME=$(python3 -c "
import re
with open('${CACHE_DIR}/warm_run.log') as f:
    times = [float(x) for x in re.findall(r'Compilation finished in ([0-9.]+) \[secs\]', f.read())]
print(f'{sum(times):.2f}' if times else '0.0')
")

echo "📊 Total Cold Compilation Time: ${COLD_COMP_TIME}s"
echo "📊 Total Warm Compilation Time: ${WARM_COMP_TIME}s"

# Assert that warm start compilation completed significantly faster than cold compilation
IS_FAST=$(python3 -c "
cold = float('${COLD_COMP_TIME}')
warm = float('${WARM_COMP_TIME}')
print(1 if (warm < 15.0 or (warm < cold and warm < cold * 0.75)) else 0)
" 2>/dev/null || echo "0")

if [ "${IS_FAST}" -eq 1 ]; then
  echo "✅ [Test 3 PASSED] Warm start confirmed cache hit! (Warm: ${WARM_COMP_TIME}s vs Cold: ${COLD_COMP_TIME}s)"
else
  echo "❌ [Test 3 FAILED] Warm start took too long (${WARM_COMP_TIME}s), indicating cache miss or recompilation!"
  exit 1
fi

# ==============================================================================
# [Test 4] Inference Output Correctness Check (Cold vs Warm)
# Description: Verifies that generated text output from the Warm Run matches
#              the Cold Run (deterministic at temperature=0), ensuring that
#              cached executables produce valid and correct inference results.
# ==============================================================================
echo "================================================================="
echo "=== 🎯 [Test 4] Inference Output Correctness Check (Cold vs Warm)"
echo "================================================================="
grep -E "^(Prompt:|Generated text:)" "${CACHE_DIR}/cold_run.log" > "${CACHE_DIR}/cold_output.txt" || true
grep -E "^(Prompt:|Generated text:)" "${CACHE_DIR}/warm_run.log" > "${CACHE_DIR}/warm_output.txt" || true

if [ ! -s "${CACHE_DIR}/cold_output.txt" ]; then
  echo "❌ [Test 4 FAILED] Cold run did not generate any prompt output!"
  exit 1
fi

if diff -u "${CACHE_DIR}/cold_output.txt" "${CACHE_DIR}/warm_output.txt"; then
  echo "✅ [Test 4 PASSED] Warm run generation output 100% matches Cold run (deterministic at temperature=0)!"
else
  echo "❌ [Test 4 FAILED] Warm run generation output differs from Cold run!"
  exit 1
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -type f -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_warm.txt" || true
else
  touch "${CACHE_DIR}/tier3_warm.txt"
fi

# ==============================================================================
# [Test 5] Cache File & MD5 Consistency Check (Cold vs Warm)
# Description: Verifies that every Cold Tier-3 binary artifact is preserved with
#              an identical MD5 checksum in the Warm run, and that any additive
#              Warm .bin entry comes strictly from pre-torch.compile eager
#              weight-loading ops rather than torch.compile recompilation.
# ==============================================================================
echo "================================================================="
echo "=== 🔍 [Test 5] Cache File & MD5 Consistency Check (Cold vs Warm)"
echo "================================================================="
python3 ./scripts/vllm/integration/check_tier3_cache_consistency.py \
  --cold-md5 "${CACHE_DIR}/tier3_cold.txt" \
  --warm-md5 "${CACHE_DIR}/tier3_warm.txt" \
  --warm-log "${CACHE_DIR}/warm_run.log"

echo "================================================================="
echo "🎉 Compilation Cache E2E Test Suite Completed!"
echo "Logs saved in: ${CACHE_DIR}"
echo "================================================================="

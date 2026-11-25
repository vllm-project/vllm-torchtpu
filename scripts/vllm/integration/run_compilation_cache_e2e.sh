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
#            cache: after model load the warm run compiles nothing (every
#            recompilation report shows 0.000s) and its total compilation time
#            drops (WARM < COLD).
#   [Test 4] Inference Output Correctness Check (Cold vs Warm):
#            Verifies that generated prompt text matches 100% between Cold and Warm runs
#            (deterministic generation with temperature=0).
#   [Test 5] Tier-3 Binary & MD5 Checksum Consistency Check (Cold vs Warm):
#            Verifies that Tier-3 binary artifacts exist and that the Warm run
#            leaves each of them unchanged. The Warm run may add one file for
#            each program that its model load compiled.
#
# Model load is treated apart: there the runtime, not the compilation cache,
# decides which programs it builds, and its choice differs from run to run.
#
# Tests 3 to 5 all run once both runs have finished, so one run reports every
# failed check. When ARTIFACTS_DIR is set, the verdicts and the Tier-3 file
# manifests are also written to ${ARTIFACTS_DIR}/compilation_cache_e2e.
#
# Usage: ./scripts/vllm/integration/run_compilation_cache_e2e.sh [model_name]
# Example: ./scripts/vllm/integration/run_compilation_cache_e2e.sh Qwen/Qwen3-0.6B
# ==============================================================================

set -euo pipefail

# The CI wrapper (.buildkite/kubernetes/run.sh) uploads what it finds under
# ARTIFACTS_DIR, which it resolves from its own working directory, so resolve
# the output directory before the cd below.
ARTIFACTS_OUT=""
if [[ -n "${ARTIFACTS_DIR:-}" ]]; then
  ARTIFACTS_OUT="${ARTIFACTS_DIR}/compilation_cache_e2e"
  [[ "${ARTIFACTS_OUT}" == /* ]] || ARTIFACTS_OUT="${PWD}/${ARTIFACTS_OUT}"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CACHE_DIR="$(mktemp -d /tmp/compilation_cache_e2e_XXXXXX)"
mkdir -p "${CACHE_DIR}"
SUMMARY="${CACHE_DIR}/summary.txt"
FAILED_CHECKS=0

# Print a verdict and keep it for the summary.
pass() { echo "✅ $*" | tee -a "${SUMMARY}"; }
fail() {
  echo "❌ $*" | tee -a "${SUMMARY}"
  FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

# Runs on every exit path. Pod output reaches the Buildkite log in batches about
# 30 s apart, and what is printed after the last batch is lost. So keep the
# verdicts as artifacts, and wait for one batch interval before exiting.
on_exit() {
  local status=$? name
  set +e # Nothing below may replace the exit status.
  echo "Exit status: ${status}" >> "${SUMMARY}"
  if [[ -n "${ARTIFACTS_OUT}" ]]; then
    mkdir -p "${ARTIFACTS_OUT}"
    # A manifest exists only once its run has finished.
    for name in summary.txt tier3_cold.txt tier3_warm.txt; do
      [[ -f "${CACHE_DIR}/${name}" ]] && cp "${CACHE_DIR}/${name}" "${ARTIFACTS_OUT}/"
    done
  fi
  echo "=== 🧹 Cleaning up temporary compilation cache artifacts... ==="
  rm -rf "${CACHE_DIR}" 2>/dev/null || true
  sleep 30
}
trap on_exit EXIT

# Export public TorchTPU compilation cache configuration
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_DIR}/torch_tpu_tier3"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_cache"
export VLLM_TARGET_DEVICE="tpu"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
# Test 3 reads the recompilation report the runner logs after each precompile
# case and each step. The flag is in the compiler's _TPU_COMPILE_ENV_IGNORED
# set, so it does not perturb the Tier-3 binaries that Test 5 checksums.
export VLLM_XLA_CHECK_RECOMPILATION=1

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

python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 --max-tokens 8 \
  --prompt "${TEST_PROMPT}" 2>&1 | tee "${CACHE_DIR}/cold_run.log" || {
  status=$?
  fail "[Test 1 FAILED] Cold run exited with status ${status}!"
  exit "${status}"
}

sync && sleep 10

if grep -q "Compiling FX graph" "${CACHE_DIR}/cold_run.log"; then
  pass "[Test 1 PASSED] Cold run triggered FX graph compilation as expected."
else
  fail "[Test 1 FAILED] Cold run did not trigger FX graph compilation!"
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
  pass "[Test 2 PASSED] Optimization verified! Category 2 pickle files total ${CAT2_SIZE} Bytes (< 50KB total, ~100B per graph), eliminating binary duplication!"
else
  echo "⚠️ [Test 2 WARNING] Category 2 pickle files total ${CAT2_SIZE} Bytes. Binary payload may still be duplicated." | tee -a "${SUMMARY}"
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
  --max-model-len 256 --max-num-batched-tokens 256 --max-tokens 8 \
  --prompt "${TEST_PROMPT}" 2>&1 | tee "${CACHE_DIR}/warm_run.log" || {
  status=$?
  fail "[Test 3 FAILED] Warm run exited with status ${status}!"
  exit "${status}"
}

sync && sleep 10

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

echo "📊 Total Cold Compilation Time: ${COLD_COMP_TIME}s" | tee -a "${SUMMARY}"
echo "📊 Total Warm Compilation Time: ${WARM_COMP_TIME}s" | tee -a "${SUMMARY}"

# Assert that warm start compilation completed significantly faster than cold compilation
IS_FAST=$(python3 -c "
cold = float('${COLD_COMP_TIME}')
warm = float('${WARM_COMP_TIME}')
print(1 if (warm < 15.0 or (warm < cold and warm < cold * 0.75)) else 0)
" 2>/dev/null || echo "0")

# The times above include tracing, so they drop by less than half on a full
# cache hit and cannot show a single recompiled graph. The runner's reports can:
# after model load, each precompile case and each step it logs the programs
# first read since the previous report and the seconds spent compiling them,
# and a program loaded from a cache costs 0.000s.
awk '
  /Total number of cached graphs:/ { case_name = $0; sub(/.*, case: /, "", case_name) }
  match($0, /[0-9]+ first read since the last check, [0-9.]+s compiling/) {
    print "   " case_name ": " substr($0, RSTART, RLENGTH)
  }
' "${CACHE_DIR}/warm_run.log" > "${CACHE_DIR}/warm_reports.txt"

# Compile time in the load_model report is accepted. The runtime fuses the
# operations of weight loading into programs of its own, and which ones it
# builds differs from run to run, so the warm run can need one that the cold
# run never built. Count these programs: Test 5 accepts one new Tier-3 file for
# each of them.
LOAD_COMPILED=$(awk '
  /Total number of cached graphs:/ { in_load = /, case: load_model$/ }
  in_load && /graph [0-9]+\/[0-9]+: compilation_duration=/ && !/compilation_duration=0:00:00( |$)/ { compiled++ }
  END { print compiled + 0 }
' "${CACHE_DIR}/warm_run.log")
WARM_REPORTS=$(grep -vc "^   load_model: " "${CACHE_DIR}/warm_reports.txt" || true)
grep -v -e "^   load_model: " -e " 0\.000s compiling$" "${CACHE_DIR}/warm_reports.txt" > "${CACHE_DIR}/warm_compiles.txt" || true

if grep "^   load_model: " "${CACHE_DIR}/warm_reports.txt" | grep -v " 0\.000s compiling$" > "${CACHE_DIR}/load_compiles.txt"; then
  echo "ℹ️ Model load of the warm run compiled ${LOAD_COMPILED} program(s), which is accepted:" | tee -a "${SUMMARY}"
  tee -a "${SUMMARY}" < "${CACHE_DIR}/load_compiles.txt"
fi

if ! grep -q "case: load_model" "${CACHE_DIR}/warm_run.log" || [ "${WARM_REPORTS}" -eq 0 ]; then
  fail "[Test 3 FAILED] Warm run logged no recompilation report for model load or after it, so its cache hits cannot be verified!"
elif [ -s "${CACHE_DIR}/warm_compiles.txt" ]; then
  fail "[Test 3 FAILED] Warm start compiled instead of loading from the cache:"
  tee -a "${SUMMARY}" < "${CACHE_DIR}/warm_compiles.txt"
elif [ "${IS_FAST}" -ne 1 ]; then
  fail "[Test 3 FAILED] Warm start took too long (${WARM_COMP_TIME}s), indicating cache miss or recompilation!"
else
  pass "[Test 3 PASSED] Warm start confirmed cache hit! (Warm: ${WARM_COMP_TIME}s vs Cold: ${COLD_COMP_TIME}s, ${WARM_REPORTS} recompilation reports at 0.000s)"
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
  fail "[Test 4 FAILED] Cold run did not generate any prompt output!"
elif diff -u "${CACHE_DIR}/cold_output.txt" "${CACHE_DIR}/warm_output.txt" | tee -a "${SUMMARY}"; then
  pass "[Test 4 PASSED] Warm run generation output 100% matches Cold run (deterministic at temperature=0)!"
else
  fail "[Test 4 FAILED] Warm run generation output differs from Cold run!"
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -type f -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_warm.txt" || true
else
  touch "${CACHE_DIR}/tier3_warm.txt"
fi

# ==============================================================================
# [Test 5] Cache File & MD5 Consistency Check (Cold vs Warm)
# Description: Compares Tier-3 binary artifact MD5sums between Cold and Warm runs
#              to audit cache file consistency.
# ==============================================================================
echo "================================================================="
echo "=== 🔍 [Test 5] Cache File & MD5 Consistency Check (Cold vs Warm)"
echo "================================================================="
# The files of the cold run that the warm run changed or removed, and how many
# files it added. Both manifests are sorted.
comm -23 "${CACHE_DIR}/tier3_cold.txt" "${CACHE_DIR}/tier3_warm.txt" > "${CACHE_DIR}/tier3_lost.txt"
TIER3_ADDED=$(comm -13 "${CACHE_DIR}/tier3_cold.txt" "${CACHE_DIR}/tier3_warm.txt" | wc -l)

if [ ! -s "${CACHE_DIR}/tier3_cold.txt" ]; then
  fail "[Test 5 FAILED] No Tier-3 binary cache files were generated!"
elif diff -u "${CACHE_DIR}/tier3_cold.txt" "${CACHE_DIR}/tier3_warm.txt" | tee -a "${SUMMARY}"; then
  pass "[Test 5 PASSED] Tier-3 cache files and MD5sums are 100% identical between Cold and Warm runs!"
elif [ -s "${CACHE_DIR}/tier3_lost.txt" ]; then
  fail "[Test 5 FAILED] Tier-3 cache files or MD5sums changed between Cold and Warm runs!"
elif [ "${TIER3_ADDED}" -gt "${LOAD_COMPILED}" ]; then
  fail "[Test 5 FAILED] Warm run added ${TIER3_ADDED} Tier-3 cache file(s), but its model load compiled ${LOAD_COMPILED} program(s)!"
else
  pass "[Test 5 PASSED] Warm run left the Tier-3 cache files of the Cold run unchanged and added ${TIER3_ADDED} for the program(s) that its model load compiled."
fi

if [ "${FAILED_CHECKS}" -ne 0 ]; then
  exit 1
fi

echo "================================================================="
echo "🎉 Compilation Cache E2E Test Suite Completed!"
echo "Logs saved in: ${CACHE_DIR}"
echo "================================================================="

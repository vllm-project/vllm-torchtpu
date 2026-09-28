#!/bin/bash
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
#
# Block-major unified pool nightly: P/D disaggregation and the Raiden host
# offload tier on the merged pool, in one job on one v7x host.
#
# VLLM_TPU_BLOCK_MAJOR_KV=1 merges a hybrid model's per-region unified pools
# into one (num_blocks, P * split, *page) allocation. The rig brings up the
# qwen35-pcp4-dp4 P/D pair through examples/disagg/launch_qwen35_p4d2_v2_baseline.sh
# with the layout on: PCP4/TP1 prefill, DP4/TP1 decode, TPUMultiConnector with
# the Stage-3 TPURaidenConnector and TPURaidenOffloadingConnector on both
# engines, toy proxy in front. Speculative decoding is off: the merged pool
# refuses it.
#
# Gates, all functional:
#   * every engine log carries the merged-pool allocation line and the offload
#     contract line, so an engine that fell back to per-region pools fails;
#   * disagg: the known-answer prefix-cache smokes through the proxy, so a
#     Stage-3 transfer that lands bytes in the wrong region fails on the first
#     tokens;
#   * offload: smoke_block_major_offload.py through the proxy, gated on the
#     prefill engine's counters: cold misses, hbm hits locally, store hits
#     externally after POST /reset_prefix_cache, first characters match cold.
#     The decode engine only compiles decode-sized batches, so it is gated on
#     the log lines and on the Stage-3 loads the disagg smokes drive into it.
#
# Trailing arguments are forwarded to the offload smoke.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

fail() {
  echo "BLOCK_MAJOR_DISAGG_OFFLOAD_FAIL: $*"
  exit 1
}

# The launcher reads MODEL_PATH; the pipelines hand over MODEL.
MODEL_PATH="${MODEL_PATH:-${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_block_major_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
P4D2_BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"

# The qwen35-pcp4-dp4 geometry with speculative decoding and async scheduling
# off.
export VLLM_SRC="${VLLM_SRC:-${repo_root}}"
export TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
export USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export VLLM_TPU_BLOCK_MAJOR_KV=1
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
export PREFILL_SPECULATIVE_CONFIG=""
export DECODE_SPECULATIVE_CONFIG=""
export PREFILL_TP="${PREFILL_TP:-1}"
export PREFILL_PCP="${PREFILL_PCP:-4}"
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
export DECODE_TP="${DECODE_TP:-1}"
export DECODE_DP="${DECODE_DP:-4}"
export PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-768}"
export DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-2304}"
export ENABLE_PREFIX_CACHING=1
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
export TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-8589934592}"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
export DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256}"
# Mounts POST /reset_prefix_cache for the offload smoke.
export VLLM_SERVER_DEV_MODE=1
export MODEL_PATH SERVED_MODEL_NAME RUN_ROOT RUN_DIR P4D2_BIND_HOST
export PREFILL_PORT DECODE_PORT PROXY_PORT
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')}"
export EXPECTED_VLLM_VERSION

# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  local pid_file pid
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 5
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

mkdir -p "${RUN_DIR}/logs" || exit 1
cd "${repo_root}" || exit 1

bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh 2>&1 | tee "${RUN_DIR}/logs/launch.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "the P/D pair did not come up (see ${RUN_DIR}/logs)"

# The layout is invisible from the API; the engine logs are the only evidence
# that each engine allocated the merged pool and registered it with the
# offload tier.
for role in prefill decode; do
  log="${RUN_DIR}/logs/${role}.log"
  grep -m1 "Block-major unified pool:" "${log}" \
    || fail "the ${role} engine did not allocate the block-major unified pool"
  grep -m1 "Block-major unified pool contract:" "${log}" \
    || fail "the ${role} engine did not register the block-major offload contract"
  grep -m1 "KVCacheStore up" "${log}" \
    || fail "no Raiden store came up on the ${role} engine"
done

expected_prefill_block=$((PREFILL_BLOCK_SIZE * PREFILL_PCP))
grep -Fq "scheduler_block_size=${expected_prefill_block} hash_block_size=${expected_prefill_block}" \
  "${RUN_DIR}/logs/prefill.log" \
  || fail "prefill did not resolve a ${expected_prefill_block}-token scheduler block"
grep -Fq "Using KV cache block size: ${DECODE_BLOCK_SIZE}" "${RUN_DIR}/logs/decode.log" \
  || fail "decode did not resolve a ${DECODE_BLOCK_SIZE}-token block"

echo "===== disagg: known-answer prefix-cache smokes through the proxy ====="
python3 scripts/vllm/integration/smoke_prefix_cache_correctness.py \
  --host "${P4D2_BIND_HOST}" --port "${PROXY_PORT}" --model "${SERVED_MODEL_NAME}" \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "prefix-cache correctness smoke through the proxy failed"
python3 scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py \
  --host "${P4D2_BIND_HOST}" --port "${PROXY_PORT}" --model "${SERVED_MODEL_NAME}" \
  --max-tokens 1 --no-logprobs \
  2>&1 | tee "${RUN_DIR}/logs/prefix_cache_e2e_divergence.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "prefix-cache divergence smoke through the proxy failed"

# Lengths clear one whole block by a quarter block, so every pass hits at
# least one block and the tail keeps the prompts off a block boundary.
smoke_lengths() {
  local block="$1"
  echo "$((block + block / 4)),$((2 * block + block / 4)),$((3 * block + block / 4))"
}

# Requests go through the proxy: the Stage-3 producer pins a request's blocks
# until decode has pulled them, so a prompt sent to the prefill engine alone
# can never be reset out of HBM. The counters and the reset address the
# prefill engine.
echo "===== offload: seed and restore on the prefill engine (PCP${PREFILL_PCP}) ====="
python3 "${script_dir}/smoke_block_major_offload.py" \
  --base "http://${P4D2_BIND_HOST}:${PROXY_PORT}" \
  --metrics-base "http://${P4D2_BIND_HOST}:${PREFILL_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --hit-block-size "${expected_prefill_block}" \
  --lengths "$(smoke_lengths "${expected_prefill_block}")" "$@" \
  2>&1 | tee "${RUN_DIR}/logs/offload_prefill.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "offload smoke on the prefill engine failed"

# Post-mortem for triage; never gating.
for role in prefill decode; do
  echo "--- ${role}.log block-major and store lines"
  grep -E "Block-major unified pool|KVCacheStore up" "${RUN_DIR}/logs/${role}.log" | head -6 || true
done

echo "BLOCK_MAJOR_DISAGG_OFFLOAD_OK"

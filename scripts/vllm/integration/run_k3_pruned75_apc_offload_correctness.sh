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
# Kimi-K3 prefix caching + host KV offload, one v7x-8, one server boot.
#
# Model: mgoin/Kimi-K3-pruned75 -- the K3 architecture (93 layers, 24 MLA +
# 69 KDA) with three quarters of the experts removed, so it fits TP8 on one
# host. Prefix caching and offload only touch the KV cache and the KDA state,
# whose shapes do not depend on the expert count.
#
# Per request the smokes read usage.cached_tokens and the deltas of the
# server counters prefix_cache_hits_total (served from HBM) and
# external_prefix_cache_hits_total (served from the host tier). Every stage
# checks that the cache was used from the expected tier by the expected
# amount AND that the generated tokens equal the uncached run.
#
# Stage 1  cold/warm            smoke_prefix_cache_e2e_divergence.py
#   One ~4.2k-token prompt sent twice. Pass: the repeat hits every whole
#   block (6 x 648 = 3,888 tokens); token ids + logprobs equal the first run.
#
# Stage 2  state contamination  smoke_mamba_state_contamination.py
#   Two different prefixes X and Y of equal length, each sent once for a
#   reference, then X, Y, X, Y ... for 3 rounds, so a hit on X resumes into
#   a state block last written by Y. Pass: every X output equals X's
#   reference, same for Y.
#
# Stage 3  multi-turn           smoke_prefix_multiturn_and_eviction_recall.py multiturn
#   Three turns; turn N+1 = turn N prompt + the answer + a new question, each
#   also sent under a different cache_salt as the uncached reference. Pass:
#   turn N+1 hits every whole block of turn N's prompt, the reference hits
#   nothing, tokens equal.
#
# Counter check (in this script) after stages 1-3: HBM hits > 0, host hits
#   == 0.
#
# Stage 4  recall after reset   smoke_basic_offload_correctness.py
#   For 1-, 2- and 4-block prompts: cold -> immediate repeat (whole-prefix
#   HBM hit, host hits 0) -> POST /reset_prefix_cache (empties HBM only) ->
#   repeat (HBM hits 0, host hits == whole prefix). Tokens equal to cold.
#
# Stage 5  recall after eviction  smoke_prefix_multiturn_and_eviction_recall.py eviction
#   Same as stage 4 with no reset: cold -> HBM repeat -> 30 unrelated
#   prompts totalling 2x the HBM pool -> repeat. Pass: HBM hits < whole
#   prefix, host hits > 0, cached_tokens == HBM + host hits, tokens equal.
#
# Server geometry:
#   * --num-gpu-blocks-override 64: small enough that stage 5's flood evicts,
#     large enough that an immediate repeat is always an HBM hit.
#   * --max-num-seqs 1, synchronous scheduling: identical single-request
#     batch shapes on every pass.
#   * --max-num-batched-tokens == block size (648 at TP8): see the comment on
#     MAX_NUM_BATCHED_TOKENS below.
#
#   MODEL=/path/to/Kimi-K3-pruned75 ./run_k3_pruned75_apc_offload_correctness.sh
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A gs:// MODEL streams through runai (CI agents have no disk for ~443 GiB);
# a local directory or an HF id goes through the default loader.
MODEL="${MODEL:-gs://tpu-commons-ci/moonshootai/kimi/k3-pruned75}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mgoin/Kimi-K3-pruned75}"
LOAD_FORMAT_ARGS=(--safetensors-load-strategy=prefetch)
if [[ "${MODEL}" == gs://* ]]; then
  # memory_limit caps each rank's streamer buffer (kimi-k3-tp32-ep.sh).
  LOAD_FORMAT_ARGS=(--load-format runai_streamer
                    --model-loader-extra-config '{"memory_limit":17179869184}')
fi

PORT="${PORT:-8000}"
RUN_ROOT="${RUN_ROOT:-${TMPDIR:-/tmp}/k3_pruned75_apc_offload_ci}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
SERVER_LOG="${LOG_DIR}/server.log"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"
BASE="http://127.0.0.1:${PORT}"

TP_SIZE="${TP_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
# The engine raises the block size to fit one KDA state per block (256 -> 648
# at TP8), so hits land on multiples of 648 tokens; the smokes read the real
# value from the engine log. 8192 holds every prompt here (<= ~4.5k tokens).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# One prefill chunk = one KV block (the 648 the engine derives at TP8; checked
# after boot). Hybrid prefix caching cuts chunks at block boundaries, so with
# this a request resumed from a hit and the same request computed from scratch
# run the identical sequence of chunk shapes from the hit onwards, and token
# equality between them is a fair gate. With a larger budget it is not: the
# uncached run covers several blocks per chunk, the resumed one starts
# mid-way, and K3's greedy output moves with chunk geometry alone -- two
# uncached boots at 2048 vs 648 already disagree after ~30-60 tokens.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-648}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Decode is one token (max_num_seqs=1); a prefill chunk pads up to the budget.
COMPILE_SIZES="${COMPILE_SIZES:-1,${MAX_NUM_BATCHED_TOKENS}}"
# HBM budget in scheduler blocks (the engine would otherwise fit ~470). 64
# blocks of 648 tokens is ~41k tokens: an order of magnitude above the largest
# request here (~7 attention blocks plus its KDA state blocks), so an immediate
# repeat is always an HBM hit, and small enough that ~30 short filler prompts
# turn the pool over twice. The host store holds ~430 blocks at the default
# CPU_BYTES_TO_USE, far more than the whole run writes, so whatever leaves HBM
# is still resolvable on the host. Empty = let the engine size the pool, which
# makes the eviction stage fail by design.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-64}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-68719476736}"
# Clear of the DSv4 (27801) and GLM-5.2 (27901) rigs' ranges.
RAIDEN_CONTROLLER_PORT="${RAIDEN_CONTROLLER_PORT:-28001}"

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export PJRT_DEVICE=TPU
export PYTHONHASHSEED=0
export VLLM_NO_USAGE_STATS=1
# Mounts POST /reset_prefix_cache for smoke_basic_offload_correctness.py.
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-${STARTUP_TIMEOUT_S}}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S="${RAIDEN_EXPECTED_WORKERS_TIMEOUT_S:-900}"

KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"raiden_job_name":"k3-pruned75-offload-ci"}}'

# The serve line is kimi-k3-tp32-ep-agentx.sh's with prefix caching on
# (--block-size 256, --mamba-cache-mode align, unified pool), narrowed to TP8.
# max_num_seqs=1 and synchronous scheduling give every pass the same
# single-request batch geometry, so a cold/warm difference is the cache and
# nothing else.
serve() {
  local extra=()
  if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    extra+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
  fi
  exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    "${LOAD_FORMAT_ARGS[@]}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE}" \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --language-model-only \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs 1 \
    --no-async-scheduling \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --block-size "${BLOCK_SIZE}" \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --enable-prompt-tokens-details \
    --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES}]}" \
    --port "${PORT}" \
    --kv-transfer-config "${KVT}" \
    "${extra[@]}"
}

fail() {
  echo "K3_APC_OFFLOAD_FAIL: $*"
  exit 1
}

SERVER_PID=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
  # Retitled workers outlive the parent and keep /dev/vfio open. Match the
  # comm name: a -f pattern would also match this script's command line.
  pkill -TERM '^VLLM' 2>/dev/null || true
  sleep 3
  pkill -KILL '^VLLM' 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "${LOG_DIR}" || exit 1
echo "===== launching ${SERVED_MODEL_NAME} from ${MODEL} (TP${TP_SIZE} + EP) \
on port ${PORT} ====="
boot_start="${SECONDS}"
( serve ) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf --max-time 10 "${BASE}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${SERVER_LOG}" || true
    fail "server exited during startup (see ${SERVER_LOG})"
  fi
  [[ "${SECONDS}" -lt "${deadline}" ]] \
    || fail "server did not become healthy in ${STARTUP_TIMEOUT_S}s"
  sleep 15
done
echo "server is healthy after $((SECONDS - boot_start))s"

if [[ "${HOLD_SERVER:-0}" == "1" ]]; then
  echo "HOLD_SERVER=1: leaving the server up on ${BASE}; no smoke is run"
  wait "${SERVER_PID}"
  exit 0
fi

# Hits land on whole blocks of the size the engine settled on, not the one
# asked for; read it from the engine rather than hardcoding the TP8 value.
HIT_BLOCK_SIZE="${HIT_BLOCK_SIZE:-$(grep -oE "Using KV cache block size: [0-9]+" \
  "${SERVER_LOG}" | tail -1 | grep -oE "[0-9]+$")}"
[[ -n "${HIT_BLOCK_SIZE}" ]] || fail "the engine never logged its KV cache block size"
# The pool the workers actually allocated (leading dim of the MLA pages), so
# the flood is sized from the engine's number and not from what was asked for.
NUM_GPU_BLOCKS="$(grep -oE "attention shape=\([0-9]+" "${SERVER_LOG}" \
  | tail -1 | grep -oE "[0-9]+$")"
[[ -n "${NUM_GPU_BLOCKS}" ]] || fail "the engine never logged its KV pool shape"
if [[ "${HIT_BLOCK_SIZE}" -ne "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  fail "the engine chose a ${HIT_BLOCK_SIZE}-token block but prefill chunks are \
${MAX_NUM_BATCHED_TOKENS} tokens; rerun with MAX_NUM_BATCHED_TOKENS=${HIT_BLOCK_SIZE} \
(see the comment on that variable)"
fi
grep -m1 "KVCacheStore up" "${SERVER_LOG}" \
  || fail "the offload store never came up (see ${SERVER_LOG})"
echo "hit block size ${HIT_BLOCK_SIZE} tokens, HBM pool ${NUM_GPU_BLOCKS} blocks"

# Two tokens: engine readiness, and far below one block so it caches nothing.
curl -sf --max-time 600 -X POST "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
  >/dev/null || fail "warm-up request failed"

failed=()
stage() {
  local name="$1" start="${SECONDS}"
  shift
  echo
  echo "===== ${name} ====="
  python3 "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  local rc="${PIPESTATUS[0]}"
  echo "===== ${name}: exit ${rc} in $((SECONDS - start))s ====="
  [[ "${rc}" -eq 0 ]] || failed+=("${name}")
}

# ---- prefix caching, HBM tier ----------------------------------------------
# 120 lines is ~4.2k tokens: six whole blocks, so the warm pass resumes from
# the KDA checkpoint at the sixth boundary.
stage apc_divergence "${script_dir}/smoke_prefix_cache_e2e_divergence.py" \
  --host 127.0.0.1 --port "${PORT}" --model "${SERVED_MODEL_NAME}" \
  --attempts 3 --line-count 120
stage apc_state_contamination "${script_dir}/smoke_mamba_state_contamination.py" \
  --host 127.0.0.1 --port "${PORT}" --model "${SERVED_MODEL_NAME}" --rounds 3
stage apc_multiturn "${script_dir}/smoke_prefix_multiturn_and_eviction_recall.py" multiturn \
  --base "${BASE}" --model "${SERVED_MODEL_NAME}" \
  --hit-block-size "${HIT_BLOCK_SIZE}"

# None of the above may have been answered by the host tier: they are the
# HBM-only half, and an external hit there means HBM was too small to hold a
# prompt between two consecutive requests.
ext_hits="$(curl -sf "${BASE}/metrics" \
  | awk '/^vllm:external_prefix_cache_hits_total/ {s += $2} END {printf "%d", s}')"
local_hits="$(curl -sf "${BASE}/metrics" \
  | awk '/^vllm:prefix_cache_hits_total/ {s += $2} END {printf "%d", s}')"
echo "after the HBM half: prefix_cache_hits_total=${local_hits} \
external_prefix_cache_hits_total=${ext_hits}"
[[ "${local_hits}" -gt 0 ]] || failed+=("apc_never_hit")
[[ "${ext_hits}" -eq 0 ]] || failed+=("apc_half_served_by_host_tier")

# ---- host offload -----------------------------------------------------------
stage offload_reset_recall "${script_dir}/smoke_basic_offload_correctness.py" \
  --base "${BASE}" --model "${SERVED_MODEL_NAME}" --dp-rank -1 \
  --hit-block-size "${HIT_BLOCK_SIZE}"
stage offload_eviction_recall "${script_dir}/smoke_prefix_multiturn_and_eviction_recall.py" eviction \
  --base "${BASE}" --model "${SERVED_MODEL_NAME}" \
  --hit-block-size "${HIT_BLOCK_SIZE}" \
  --hbm-tokens "$((NUM_GPU_BLOCKS * HIT_BLOCK_SIZE))"

echo
echo "--- server.log offload/store activity"
grep "KV Transfer metrics" "${SERVER_LOG}" | tail -1 || true
awk '/KVCacheStore up/{seen=1} seen' "${SERVER_LOG}" \
  | grep -iE "store job|load job|\[kv-offload\]|insert rejected|drain timed out|(WARNING|ERROR).*(raiden|offload)" \
  | tail -20 || true

if [[ "${#failed[@]}" -gt 0 ]]; then
  fail "failed stages: ${failed[*]} (logs in ${LOG_DIR})"
fi
echo "K3_APC_OFFLOAD_OK"

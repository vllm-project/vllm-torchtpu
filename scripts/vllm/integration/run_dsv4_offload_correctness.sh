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
# DSv4 KV offload correctness rig: does a block stored to the Raiden host
# pool reload as the same bytes?
#
# The rig exists because a plain cold/warm repeat never reaches the offload
# store -- the HBM prefix cache answers first. VLLM_SERVER_DEV_MODE mounts
# POST /reset_prefix_cache, which empties the HBM cache while leaving the
# Raiden store intact (the connector's reset_cache() reports failure and
# never reaches the store, see raiden_connector.py). That turns the repeat
# into a real host->device load with no eviction-by-volume games.
#
# The rig always runs the full model (~27 min to ready): a truncated model
# has non-finite logits, so every generation is a degenerate constant and cold
# and store agree whatever the store returns.
#
# Self-contained, the shape every other rig in this directory has: launch ->
# /health -> warm up -> smoke -> verdict, with the engine torn down on exit.
# This is the CI entry point. Trailing arguments are forwarded to the smoke:
#
#   ./run_dsv4_offload_correctness.sh --settle-s 10
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compile artifacts run to 30-40G for a full boot and default to
# ~/.cache/vllm on the root filesystem. On the dev box that filesystem is too
# small and the run dies mid-compile with ENOSPC, which looks like a crash, so
# redirect everything to the data volume. Under CI the container has its own
# scratch and no /mnt/pd at all -- pointing at one there would fail on the
# first write -- so fall back to TMPDIR.
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
if [[ -z "${SCRATCH_ROOT}" ]]; then
  if [[ -d /mnt/pd ]]; then
    SCRATCH_ROOT=/mnt/pd/tmp
  else
    SCRATCH_ROOT="${TMPDIR:-/tmp}/dsv4-offload"
  fi
fi
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRATCH_ROOT}/vllm-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/torchinductor}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${VLLM_CACHE_ROOT}" || exit 1

export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 MOE_REQUANTIZE_BLOCK_SIZE=512
export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export TPU_ROPE_CACHE_ROW_MAJOR=1 TPU_MOE_HASH_TABLE_ROW_MAJOR=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0 VLLM_NO_USAGE_STATS=1
export TORCH_DIST_TIMEOUT=1200 VLLM_RPC_TIMEOUT=1200000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1200 VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONHASHSEED=0
export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S="${RAIDEN_EXPECTED_WORKERS_TIMEOUT_S:-900}"
# Mounts POST /reset_prefix_cache. Without it the smoke cannot reach the store.
export VLLM_SERVER_DEV_MODE=1

VLLM_BIN="${VLLM_BIN:-vllm}"

PORT="${PORT:-$(python3 -c 'import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()' 2>/dev/null || echo 8123)}"
MODEL="${MODEL:-/mnt/pd/DeepSeek-V4-Flash-local}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
# DP 8 / TP 1, with every request pinned to one rank (see serve() below).
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
DP_RANK="${DP_RANK:-0}"
# PER DP REPLICA, not per server: each rank builds its own KVCacheStore, so the
# host reserves DP_SIZE times this. 25 GiB x 8 = 200 GiB on a 944 GiB box.
# Keep the product well under host DRAM: asking for 64 GiB each reserves
# ~512 GiB, and the kernel OOM killer then takes out a worker, which surfaces
# minutes later as an unrelated-looking registration timeout on another rank.
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-26843545600}"
# Base port. Each rank binds RAIDEN_CONTROLLER_PORT + dp_rank_local, so DP 8
# claims 27801..27808.
RAIDEN_CONTROLLER_PORT="${RAIDEN_CONTROLLER_PORT:-27801}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/dsv4_offload_ci}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
# A full boot is ~27 min of weight load plus compile; leave headroom.
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"
BASE="http://127.0.0.1:${PORT}"

# The smoke is stdlib-only, but prefer the interpreter beside VLLM_BIN so a
# venv-pinned launch does not silently run the probe on the system python.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="python3"
  if [[ "${VLLM_BIN}" == */* && -x "$(dirname "${VLLM_BIN}")/python3" ]]; then
    PYTHON_BIN="$(dirname "${VLLM_BIN}")/python3"
  fi
fi

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"raiden_job_name":"dsv4-offload-probe"}}'

# max-num-seqs 1 gives every pass identical single-request batch geometry, so
# a cold/warm difference is the cache and nothing else.

serve() {
  exec "${VLLM_BIN}" serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --quantization deepseek_v4_fp8 \
    --trust-remote-code \
    --max-model-len=9216 \
    --max-num-batched-tokens=1024 \
    --max-num-seqs=1 \
    --kv-cache-dtype=fp8_e4m3 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization="${GPU_MEM_UTIL}" \
    --data-parallel-size="${DP_SIZE}" \
    --tensor-parallel-size="${TP_SIZE}" \
    --no-async-scheduling \
    --enable-expert-parallel \
    --port "${PORT}" \
    --kv-transfer-config "$KVT"
}

mkdir -p "${LOG_DIR}" || exit 1
SERVER_LOG="${LOG_DIR}/server.log"

# Print to stdout: Buildkite's main log view surfaces stdout, and the failure
# reason must be readable there without digging through artifacts.
fail() {
  echo "DSV4_OFFLOAD_RIG_FAIL: $*"
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
  # The TP8 workers are retitled, so they survive the parent and keep
  # /dev/vfio open -- the next launch then dies with the device busy. Match on
  # the comm name: a -f pattern would also match this script's own command
  # line and kill the caller mid-cleanup. This rig owns all eight chips, so
  # there is no other engine on the host for it to catch.
  pkill -TERM '^VLLM' 2>/dev/null || true
  sleep 3
  pkill -KILL '^VLLM' 2>/dev/null || true
}
trap cleanup EXIT

echo "===== launching DSv4 (DP${DP_SIZE}/TP${TP_SIZE}), requests pinned to \
dp rank ${DP_RANK}, on port ${PORT} ====="
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
echo "server is healthy"

# Engine readiness doubles as the offload store's deferred worker-registration
# gate. Two tokens stays far below the 1024-token offload block, so this warm
# up publishes nothing the smoke could later mistake for its own cold prompt.
until curl -sf --max-time 300 -X POST "${BASE}/v1/completions" \
    -H "Content-Type: application/json" \
    -H "X-data-parallel-rank: ${DP_RANK}" \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
    >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${SERVER_LOG}" || true
    fail "server exited during warm up (see ${SERVER_LOG})"
  fi
  [[ "${SECONDS}" -lt "${deadline}" ]] \
    || fail "engine not ready in ${STARTUP_TIMEOUT_S}s"
  echo "engine compiling, waiting 30s..."
  sleep 30
done
echo "engine is ready"

SMOKE_ARGS=(--base "${BASE}" --model "${SERVED_MODEL_NAME}"
            --dp-rank "${DP_RANK}")
SMOKE_ARGS+=("$@")

"${PYTHON_BIN}" "${script_dir}/smoke_basic_offload_correctness.py" \
  "${SMOKE_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/smoke.log"
smoke_rc="${PIPESTATUS[0]}"

# Post-mortem for triage; never gating.
#
# Matching on "offload|raiden" alone reports the boot, not the run: those two
# words appear in the TPU offloading patches, the per-rank connector
# construction and the registration handshake -- around fifty lines before the
# store is even up -- and then the KV-transfer metrics heartbeat repeats every
# ten seconds. A plain grep | tail -40 is that noise end to end, and the smoke's
# own activity never reaches the window. Two of the old keywords ("cannot
# store", "external hit") match no string the connector can emit at all.
#
# So: cut to the log after the store came up, and match the vocabulary the
# connector actually logs when something goes wrong -- "store job"/"load job"
# (both only ever appear on a failed or retried job), the "[kv-offload]"
# transport prefix, plus any WARNING/ERROR from those modules. Silence here is
# the good outcome and says so out loud, rather than looking like a broken grep.
echo "--- server.log offload/store activity"
if grep -q "KVCacheStore up" "${SERVER_LOG}"; then
  grep -m1 "KVCacheStore up" "${SERVER_LOG}" || true
  grep "KV Transfer metrics" "${SERVER_LOG}" | tail -1 || true
  pm_re="store job|load job|\[kv-offload\]|insert rejected"
  pm_re="${pm_re}|drain timed out|external cache reset|vanished"
  pm_re="${pm_re}|(WARNING|ERROR).*(raiden|offload)"
  post_mortem="$(awk '/KVCacheStore up/{seen=1} seen' "${SERVER_LOG}" \
    | grep -iE "${pm_re}" | grep -v "KV Transfer metrics" | tail -40)"
  if [[ -n "${post_mortem}" ]]; then
    echo "${post_mortem}"
  else
    echo "(no connector warnings or errors after the store came up)"
  fi
else
  echo "(the offload store never came up -- see ${SERVER_LOG})"
fi

[[ "${smoke_rc}" -eq 0 ]] || fail "smoke_basic_offload_correctness.py exited \
${smoke_rc} (see ${LOG_DIR}/smoke.log)"
echo "DSV4_OFFLOAD_RIG_OK"

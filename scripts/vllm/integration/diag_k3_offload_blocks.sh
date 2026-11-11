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
# Throwaway probe (dev pipeline only): offloaded blocks per K3-pruned75
# prompt. The serve line is run_k3_pruned75_gpc_tier1_eviction.sh's, with a
# plain ~200-block host pool: no registry, no store node, no sweep.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail() {
  echo "K3_OFFLOAD_BLOCKS_PROBE_FAIL: $*"
  exit 1
}

MODEL="${MODEL:-gs://tpu-commons-ci/moonshootai/kimi/k3-pruned75}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mgoin/Kimi-K3-pruned75}"
RUN_ROOT="${RUN_ROOT:-${TMPDIR:-/tmp}/k3_offload_blocks_probe}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
BIND_HOST=127.0.0.1
PORT="${PORT:-8200}"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-22000}"
BASE="http://${BIND_HOST}:${PORT}"
STARTUP_TIMEOUT_S=3600
MAX_NUM_BATCHED_TOKENS=680
# 200.5 blocks of ~167.2 MB; the pool floors to 200.
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-33523700000}"
PHASES="${PHASES:-4:40:1000,2:80:3000}"
mkdir -p "${LOG_DIR}"

SERVER_PID=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then kill -TERM "${SERVER_PID}" 2>/dev/null || true; fi
  sleep 5
  if [[ -n "${SERVER_PID}" ]]; then kill -KILL "${SERVER_PID}" 2>/dev/null || true; fi
  pkill -TERM '^VLLM' 2>/dev/null || true
  sleep 3
  pkill -KILL '^VLLM' 2>/dev/null || true
}
trap cleanup EXIT

KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${OFFLOAD_CONTROLLER_PORT}"'}}'

(
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export PJRT_DEVICE=TPU
  export PYTHONHASHSEED=0
  export VLLM_NO_USAGE_STATS=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_S}"
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=900
  exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --load-format runai_streamer \
    --model-loader-extra-config '{"memory_limit":17179869184}' \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --language-model-only \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --max-model-len 8192 \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs 1 \
    --no-async-scheduling \
    --gpu-memory-utilization 0.85 \
    --block-size 256 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --compilation-config "{\"compile_sizes\":[1,${MAX_NUM_BATCHED_TOKENS}]}" \
    --num-gpu-blocks-override 64 \
    --host "${BIND_HOST}" \
    --port "${PORT}" \
    --kv-transfer-config "${KVT}"
) >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf --max-time 10 "${BASE}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${LOG_DIR}/server.log" || true
    fail "server exited during startup"
  fi
  [[ "${SECONDS}" -lt "${deadline}" ]] || fail "server not healthy in time"
  sleep 15
done
curl -sf --max-time 600 -X POST "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
  >/dev/null || fail "warm-up request failed"

HIT_BLOCK_SIZE="$(grep -oE "Using KV cache block size: [0-9]+" \
  "${LOG_DIR}/server.log" | tail -1 | grep -oE "[0-9]+$")"
POOL_BLOCKS="$(sed -nE \
  's/.*KVCacheStore up: .*\(([0-9]+) offloaded blocks x [0-9]+\).*/\1/p' \
  "${LOG_DIR}/server.log" | head -1)"
[[ -n "${HIT_BLOCK_SIZE}" && -n "${POOL_BLOCKS}" ]] || fail "no block size or pool size in the log"
grep -m1 "num_blocks=.*num_groups=" "${LOG_DIR}/server.log" || true
echo "hit block ${HIT_BLOCK_SIZE} tokens; pool ${POOL_BLOCKS} offloaded blocks"

python3 "${script_dir}/diag_k3_offload_blocks.py" \
  --base "${BASE}" --model "${SERVED_MODEL_NAME}" \
  --hit-block-size "${HIT_BLOCK_SIZE}" --pool-blocks "${POOL_BLOCKS}" \
  --phases "${PHASES}" 2>&1 | tee "${LOG_DIR}/probe.log"

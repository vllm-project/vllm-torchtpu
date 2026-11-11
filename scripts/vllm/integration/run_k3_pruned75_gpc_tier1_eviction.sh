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
# GLOBAL DRAM POOL WITH BOTH TIERS FULL: blocks that tier 0 evicts into a full
# tier 1 must make tier 1 drop its own coldest blocks, not be refused.
#
#   vllm serve (Kimi-K3-pruned75, TP8)      global_registry_server
#     TPURaidenOffloadingConnector    <-->    directory + placement
#     tier 0: host DRAM pool (small)          ^
#         | demote under pressure             | spec + heartbeats
#         v                                   v
#   tier 1: kv_cache_host_store_node_main (--evict_tier=1, small budget,
#           evict sweep on)
#
# Both tiers hold only a few prompts, and the smoke streams far more than
# they hold together. Tier 0's sweep demotes its oldest blocks to tier 1.
# Once a demotion leaves tier 1 under its low watermark, that write wakes
# tier 1's own sweep, which drops tier 1's oldest blocks locally and from the
# registry so the next demotions still land. A node without that sweep keeps
# its first blocks forever and refuses every later demotion with
# RESOURCE_EXHAUSTED; tier 0 then drops the blocks instead.
#
# The node's sweep period defaults to an hour, longer than the run, so only
# the remote-write wake can run it. NODE_EVICT_SWEEP=false turns the sweep
# off; the smoke must then fail, which checks that it still catches the
# regression.
#
# The model and serve line are run_k3_pruned75_apc_offload_correctness.sh's;
# the registry and store node are run_qwen35_gpc_global_dram_pool.sh's.
#
#   MODEL=/path/to/Kimi-K3-pruned75 ./run_k3_pruned75_gpc_tier1_eviction.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print to stdout: Buildkite's main log view surfaces stdout, and the
# failure reason must be readable there without digging through artifacts.
fail() {
  echo "K3_GPC_TIER1_EVICT_FAIL: $*"
  exit 1
}

# A gs:// MODEL streams through runai (CI agents have no disk for ~443 GiB);
# a local directory or an HF id goes through the default loader.
MODEL="${MODEL:-gs://tpu-commons-ci/moonshootai/kimi/k3-pruned75}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mgoin/Kimi-K3-pruned75}"
LOAD_FORMAT_ARGS=(--safetensors-load-strategy=prefetch)
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer
                    --model-loader-extra-config '{"memory_limit":17179869184}')
fi

RUN_ROOT="${RUN_ROOT:-${TMPDIR:-/tmp}/k3_pruned75_gpc_tier1_eviction_ci}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

# First free port at or above $1, probed the way the servers bind it (see
# run_qwen35_gpc_global_dram_pool.sh).
pick_port() {
  python3 - "$1" <<'PY'
import socket
import sys

base = int(sys.argv[1])
dualstack = socket.has_dualstack_ipv6()
family = socket.AF_INET6 if dualstack else socket.AF_INET
for port in range(base, base + 100):
    try:
        probe = socket.create_server(("", port), family=family,
                                     dualstack_ipv6=dualstack)
    except OSError:
        continue
    probe.close()
    print(port)
    break
else:
    raise SystemExit(f"no port free in [{base}, {base + 100})")
PY
}

PORT="${PORT:-$(pick_port 8200)}" || fail "no free port for the vLLM server"
REGISTRY_PORT="${REGISTRY_PORT:-$(pick_port 22500)}" \
  || fail "no free port for the global registry"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-$(pick_port 22000)}" \
  || fail "no free port for the offload controller"
NODE_CONTROLLER_PORT="${NODE_CONTROLLER_PORT:-$(pick_port 21500)}" \
  || fail "no free port for the store node controller"
BASE="http://${BIND_HOST}:${PORT}"
KV_POOL_GROUP="${KV_POOL_GROUP:-k3-tier1-evict-ci}"

# Engine geometry: the K3 offload rig's, whose comments explain each value.
# The engine raises the 256-token block to 680 at TP8 to fit one KDA state;
# one prefill chunk is one block.
TP_SIZE="${TP_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-680}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
COMPILE_SIZES="${COMPILE_SIZES:-1,${MAX_NUM_BATCHED_TOKENS}}"
# HBM holds a dozen prompts; the smoke empties it before every recall.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-64}"

# Prompts are PROMPT_BLOCKS 680-token blocks long, but each takes
# PROMPT_STORE_BLOCKS offloaded blocks. K3 has four KV cache groups of
# 680-token blocks, MLA attention and three KDA groups. The connector stores
# the attention block of every chunk but only the last state of each KDA
# group, so a prompt of n blocks takes n + 3 (measured on this rig: 2 blocks
# take 5, 4 take 7). The runner checks this against the groups the engine
# logs.
PROMPTS="${PROMPTS:-30}"
PROMPT_BLOCKS="${PROMPT_BLOCKS:-4}"
PROMPT_STORE_BLOCKS="${PROMPT_STORE_BLOCKS:-7}"
# Tier sizing, in prompts. An offloaded block is the same in both pools: 24
# block arrays of 870400 bytes on each of the 8 ranks. Both pools floor
# their budget to whole blocks, so each budget is half a block over the
# count it aims for; the smoke reads the counts the boot logs report and
# fails if they leave it no margin.
#   tier 0: 10 prompts, swept from under 20% free back to 40% free.
#   tier 1: 14 prompts, swept from under 30% free back to 40% free. Its low
#           watermark leaves more free blocks than one tier-0 demotion
#           batch, so a batch lands before the woken sweep has run.
BLOCK_BYTES=$((24 * 870400 * 8))
TIER0_PROMPTS=10
TIER1_PROMPTS=14
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-$(((2 * TIER0_PROMPTS * PROMPT_STORE_BLOCKS + 1) * BLOCK_BYTES / 2))}"
NODE_DRAM_BUDGET_BYTES="${NODE_DRAM_BUDGET_BYTES:-$(((2 * TIER1_PROMPTS * PROMPT_STORE_BLOCKS + 1) * BLOCK_BYTES / 2))}"
TIER0_LOW_WATERMARK="${TIER0_LOW_WATERMARK:-0.2}"
TIER0_HIGH_WATERMARK="${TIER0_HIGH_WATERMARK:-0.4}"
TIER1_LOW_WATERMARK="${TIER1_LOW_WATERMARK:-0.3}"
TIER1_HIGH_WATERMARK="${TIER1_HIGH_WATERMARK:-0.4}"
NODE_EVICT_SWEEP="${NODE_EVICT_SWEEP:-true}"
NODE_EVICT_SWEEP_PERIOD="${NODE_EVICT_SWEEP_PERIOD:-1h}"

mkdir -p "${LOG_DIR}"
echo "ports: server=${PORT} registry=${REGISTRY_PORT}\
 offload controller=${OFFLOAD_CONTROLLER_PORT}\
 node controller=${NODE_CONTROLLER_PORT}"

RAIDEN_VERSION="$(python3 -c \
  'import importlib.metadata as m; print(m.version("tpu-sync-torch"))')" \
  || fail "tpu-sync-torch is not installed"
for cmd in global_registry_server kv_cache_host_store_node_main; do
  command -v "${cmd}" >/dev/null || fail \
    "${cmd} not found: tpu-sync-torch ${RAIDEN_VERSION} does not bundle the \
global DRAM pool control plane"
done
# The node's evict sweep and its flags came with the same raiden change.
node_help="$(kv_cache_host_store_node_main --helpfull 2>&1 || true)"
grep -q -- "--enable_evict_sweep" <<<"${node_help}" \
  || fail "kv_cache_host_store_node_main in tpu-sync-torch ${RAIDEN_VERSION} \
has no --enable_evict_sweep: the wheel predates the store node's evict sweep"
echo "control plane from the tpu-sync-torch ${RAIDEN_VERSION} wheel"

# ---------------------------------------------------------------------------
# Global registry, then the tier-1 store node. The node waits for the pool
# group's KVTransferSpec, which the serving store publishes when it comes up.
# ---------------------------------------------------------------------------
export GLOG_alsologtostderr=1
global_registry_server --port="${REGISTRY_PORT}" \
  >"${LOG_DIR}/registry.log" 2>&1 &
REGISTRY_PID=$!

SERVER_PID=""
NODE_PID=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  local pid
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  sleep 5
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -KILL "${pid}" 2>/dev/null || true; fi
  done
  # Retitled workers outlive the parent and keep /dev/vfio open. Match the
  # comm name: a -f pattern would also match this script's command line.
  pkill -TERM '^VLLM' 2>/dev/null || true
  sleep 3
  pkill -KILL '^VLLM' 2>/dev/null || true
}
trap cleanup EXIT

check_control_plane() {
  kill -0 "${REGISTRY_PID}" 2>/dev/null || {
    tail -50 "${LOG_DIR}/registry.log" || true
    fail "global_registry_server exited"
  }
  if [[ -n "${NODE_PID}" ]]; then
    kill -0 "${NODE_PID}" 2>/dev/null || {
      tail -50 "${LOG_DIR}/store_node.log" || true
      fail "kv_cache_host_store_node_main exited"
    }
  fi
}

wait_port() {
  local port="$1" what="$2" timeout="${3:-60}" waited=0
  until (exec 3<>"/dev/tcp/${BIND_HOST}/${port}") 2>/dev/null; do
    check_control_plane
    [[ "${waited}" -lt "${timeout}" ]] \
      || fail "${what} (${BIND_HOST}:${port}) not reachable after ${waited}s"
    sleep 1
    waited=$((waited + 1))
  done
  echo "${what} port ${port} reachable"
}

wait_port "${REGISTRY_PORT}" "global registry"

kv_cache_host_store_node_main \
  --job_name=k3-tier1-evict-store-node \
  --kv_pool_group="${KV_POOL_GROUP}" \
  --evict_tier=1 \
  --global_registry_address="${BIND_HOST}:${REGISTRY_PORT}" \
  --store_server_ip="${BIND_HOST}" \
  --raiden_controller_port="${NODE_CONTROLLER_PORT}" \
  --dram_budget_bytes="${NODE_DRAM_BUDGET_BYTES}" \
  --enable_evict_sweep="${NODE_EVICT_SWEEP}" \
  --evict_sweep_period="${NODE_EVICT_SWEEP_PERIOD}" \
  --evict_low_watermark="${TIER1_LOW_WATERMARK}" \
  --evict_high_watermark="${TIER1_HIGH_WATERMARK}" \
  >"${LOG_DIR}/store_node.log" 2>&1 &
NODE_PID=$!

# ---------------------------------------------------------------------------
# The serving replica: tier 0.
# ---------------------------------------------------------------------------
KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${OFFLOAD_CONTROLLER_PORT}"',"raiden_job_name":"k3-tier1-evict-serve","global_registry_address":"'"${BIND_HOST}"':'"${REGISTRY_PORT}"'","store_server_ip":"'"${BIND_HOST}"'","kv_pool_group":"'"${KV_POOL_GROUP}"'"}}'

(
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export PJRT_DEVICE=TPU
  # Registry lookups only hit across matching block-hash chains, and the
  # connector refuses a registry-enabled boot without a fixed seed.
  export PYTHONHASHSEED=0
  export VLLM_NO_USAGE_STATS=1
  # Mounts POST /reset_prefix_cache, which the smoke uses to empty HBM.
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-${STARTUP_TIMEOUT_S}}"
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S="${RAIDEN_EXPECTED_WORKERS_TIMEOUT_S:-900}"
  # Tier 0's monitor and demotion sweep, on short periods so a 30-prompt
  # stream turns both tiers over in minutes.
  export RAIDEN_ENABLE_STORE_MONITOR=1
  export RAIDEN_ENABLE_EVICT_SWEEP=1
  export RAIDEN_STORE_MONITOR_HEARTBEAT_S=2
  export RAIDEN_EVICT_SWEEP_PERIOD_S=2
  export RAIDEN_EVICT_LOW_WATERMARK="${TIER0_LOW_WATERMARK}"
  export RAIDEN_EVICT_HIGH_WATERMARK="${TIER0_HIGH_WATERMARK}"
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
    --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES}]}" \
    --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
    --host "${BIND_HOST}" \
    --port "${PORT}" \
    --kv-transfer-config "${KVT}"
) >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

boot_start="${SECONDS}"
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf --max-time 10 "${BASE}/health" >/dev/null; do
  check_control_plane
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${LOG_DIR}/server.log" || true
    fail "server exited during startup (see ${LOG_DIR}/server.log)"
  fi
  [[ "${SECONDS}" -lt "${deadline}" ]] \
    || fail "server did not become healthy in ${STARTUP_TIMEOUT_S}s"
  sleep 15
done
echo "server is healthy after $((SECONDS - boot_start))s"

# Engine readiness, and the offload store's worker-registration gate; two
# tokens, far below one block, so it caches nothing.
curl -sf --max-time 600 -X POST "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
  >/dev/null || fail "warm-up request failed"

HIT_BLOCK_SIZE="$(grep -oE "Using KV cache block size: [0-9]+" \
  "${LOG_DIR}/server.log" | tail -1 | grep -oE "[0-9]+$")" \
  || fail "the engine never logged its KV cache block size"
if [[ "${HIT_BLOCK_SIZE}" -ne "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  fail "the engine chose a ${HIT_BLOCK_SIZE}-token block but prefill chunks are \
${MAX_NUM_BATCHED_TOKENS} tokens; rerun with MAX_NUM_BATCHED_TOKENS=${HIT_BLOCK_SIZE}"
fi
read -r MAMBA_GROUPS OTHER_GROUPS < <(
  sed -nE 's/.*group\[([0-9]+)\]: spec=([A-Za-z]+).*/\1 \2/p' "${LOG_DIR}/server.log" \
    | sort -u \
    | awk '{ if ($2 == "MambaSpec") m++; else o++ } END { print m + 0, o + 0 }')
if [[ $((MAMBA_GROUPS + OTHER_GROUPS)) -gt 0 ]]; then
  store_blocks=$((PROMPT_BLOCKS * OTHER_GROUPS + MAMBA_GROUPS))
  [[ "${store_blocks}" -eq "${PROMPT_STORE_BLOCKS}" ]] || fail "the engine logs \
${OTHER_GROUPS} attention and ${MAMBA_GROUPS} KDA cache groups, so a prompt takes \
${store_blocks} offloaded blocks, not PROMPT_STORE_BLOCKS=${PROMPT_STORE_BLOCKS}"
else
  echo "the engine logged no KV cache groups; trusting PROMPT_STORE_BLOCKS=${PROMPT_STORE_BLOCKS}"
fi
TIER0_BLOCKS="$(sed -nE \
  's/.*KVCacheStore up: .*\(([0-9]+) offloaded blocks x [0-9]+\).*/\1/p' \
  "${LOG_DIR}/server.log" | head -1)"
[[ -n "${TIER0_BLOCKS}" ]] || fail "the offload store never came up (see ${LOG_DIR}/server.log)"

# The node sizes its pool once the serving store has published the spec.
wait_port "${NODE_CONTROLLER_PORT}" "store node controller" 180
TIER1_BLOCKS="$(sed -nE 's/.*KVCacheHostStoreNode up: ([0-9]+) host blocks.*/\1/p' \
  "${LOG_DIR}/store_node.log" | head -1)"
[[ -n "${TIER1_BLOCKS}" ]] || fail "the store node never logged its pool size"
if [[ "${NODE_EVICT_SWEEP}" == "true" ]] \
    && grep -q "Evict sweep disabled" "${LOG_DIR}/store_node.log"; then
  fail "the store node runs without its evict sweep: $(grep -m1 \
"Evict sweep disabled" "${LOG_DIR}/store_node.log")"
fi
echo "hit block ${HIT_BLOCK_SIZE} tokens; tier 0 ${TIER0_BLOCKS} blocks;\
 tier 1 ${TIER1_BLOCKS} blocks; node evict sweep ${NODE_EVICT_SWEEP}"

check_control_plane
smoke_rc=0
python3 "${script_dir}/smoke_gpc_tier1_eviction.py" \
  --base "${BASE}" \
  --model "${SERVED_MODEL_NAME}" \
  --hit-block-size "${HIT_BLOCK_SIZE}" \
  --prompts "${PROMPTS}" \
  --prompt-blocks "${PROMPT_BLOCKS}" \
  --prompt-store-blocks "${PROMPT_STORE_BLOCKS}" \
  --tier0-blocks "${TIER0_BLOCKS}" \
  --tier0-low "${TIER0_LOW_WATERMARK}" \
  --tier0-high "${TIER0_HIGH_WATERMARK}" \
  --tier1-blocks "${TIER1_BLOCKS}" \
  --tier1-low "${TIER1_LOW_WATERMARK}" \
  --tier1-high "${TIER1_HIGH_WATERMARK}" \
  2>&1 | tee "${LOG_DIR}/smoke.log" || smoke_rc=$?

# Post-mortem for triage; never gating. Raiden's INFO lines do not reach
# server.log, so only its sweep warnings show here.
grep -E "Evict sweep" "${LOG_DIR}/server.log" | tail -10 || true
echo "--- store_node.log tail"
tail -30 "${LOG_DIR}/store_node.log" || true
echo "--- registry.log tail"
tail -20 "${LOG_DIR}/registry.log" || true

[[ "${smoke_rc}" -eq 0 ]] || fail "smoke failed (logs in ${LOG_DIR})"
echo "K3_GPC_TIER1_EVICT_OK"

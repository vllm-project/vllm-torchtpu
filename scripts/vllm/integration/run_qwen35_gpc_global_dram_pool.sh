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
# KV OFFLOADING WITH A GLOBAL DRAM POOL: one serving replica's KV cache
# through every tier of raiden's global DRAM pool, on a single host.
#
# The rig is the smallest one that exercises every pool tier:
#
#   vllm serve (qwen3.5, PCP4/TP1)      global_registry_server
#     TPURaidenOffloadingConnector  <-->  directory + placement
#     host DRAM pool (small)              ^
#         | demote under pressure         | spec + heartbeats
#         v                               v
#   kv_cache_host_store_node_main (--kv_pool_group matches, --evict_tier=1)
#
# Serving traffic offloads KV to the replica's own host DRAM pool. The pool
# is deliberately sized to only a few prompts' worth of KV, so sustained
# traffic pushes it over its watermark and the store's monitor demotes the
# coldest offloaded blocks to the store node, which joined the same
# kv_pool_group one tier down. Cold prefixes must then still be servable:
# the lookup resolves them through the registry and read_remote pulls them
# back from the store node.
#
# qwen3.5 is the model on purpose: its hybrid attention+GDN KV layout is
# what the KVTransferSpec machinery exists for, so a spec mismatch between
# the serving pool and the store node cannot hide behind a homogeneous
# layout.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_gpc_dram_pool_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
# The test container runs with --net=host, so fixed ports collide with
# whatever an earlier run left on the agent (a timed-out job's container
# outlives its docker client and keeps its listeners). Shift all ports by
# a per-job value; kept under 47 so the bases never cross each other.
job_tag="${BUILDKITE_JOB_ID:-${BUILDKITE_BUILD_NUMBER:-0}}"
PORT_SHIFT="${PORT_SHIFT:-$(($(printf '%s' "${job_tag}" | cksum | cut -d' ' -f1) % 47))}"
PORT="${PORT:-$((8100 + PORT_SHIFT))}"
REGISTRY_PORT="${REGISTRY_PORT:-$((48500 + PORT_SHIFT))}"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-$((47800 + PORT_SHIFT))}"
NODE_CONTROLLER_PORT="${NODE_CONTROLLER_PORT:-$((47900 + PORT_SHIFT))}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

# One pool group covers the serving replica (tier 0, the implicit serving
# tier) and the store node (tier 1). The group is the spec-compatibility and
# demotion domain; it says nothing about placement.
KV_POOL_GROUP="${KV_POOL_GROUP:-gpc-ci-pool}"

# Engine geometry. PCP shards the KV of one sequence across ranks, so the
# published KVTransferSpec carries multiple shards and the store node must
# reproduce the sharded hybrid layout — the strictest spec coverage this
# host can buy. The hash key namespace covers all of this; keep it in
# sync with anything that reuses these caches.
PCP="${PCP:-4}"
CP_KV_CACHE_INTERLEAVE_SIZE="${CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

# Traffic shape: GPC_PROMPTS distinct prompts of GPC_PROMPT_TOKENS tokens.
# The connector offloads KV in whole BLOCK_SIZE-token blocks addressed by
# prefix hash keys, so each prompt offloads its OFFLOADED_BLOCKS_PER_PROMPT
# whole blocks and the trailing partial block always recomputes.
GPC_PROMPTS="${GPC_PROMPTS:-12}"
GPC_PROMPT_TOKENS="${GPC_PROMPT_TOKENS:-16448}"
GPC_SWEEP_WAIT_S="${GPC_SWEEP_WAIT_S:-15}"
OFFLOADED_BLOCKS_PER_PROMPT=$((GPC_PROMPT_TOKENS / BLOCK_SIZE))

# Tiered sizing, from the top down.
#   HBM:       14 unified-pool pages per rank. vLLM refuses to boot below
#              what one max-model-len request needs, and qwen3.5's hybrid
#              accounting puts that at ~11 pages (mamba state is charged
#              per sequence on top of the attention pages), so this is
#              close to the smallest HBM cache that serves at all: it
#              cannot hold a finished prompt's KV while the next one runs,
#              so every recall must go through the offload connector.
#   host pool: CPU_BYTES_TO_USE buys the pool in whole offloaded blocks
#              (floored), sized here to 28 blocks = 7 prompts' worth of
#              KV. POOL_PROMPTS_MIN/MAX below trips if the model's bytes
#              per block drift. Seeding GPC_PROMPTS prompts overflows the
#              watermark, and the store's monitor demotes the oldest
#              blocks to the store node while the smoke's third-newest
#              recall target stays above the demotion line.
#   store node: DRAM budget comfortably above everything demoted (16 GiB,
#              ~25 prompts' worth).
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-14}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-4780000000}"
POOL_PROMPTS_MIN="${POOL_PROMPTS_MIN:-6}"
POOL_PROMPTS_MAX="${POOL_PROMPTS_MAX:-8}"
NODE_DRAM_BUDGET_BYTES="${NODE_DRAM_BUDGET_BYTES:-17179869184}"

mkdir -p "${RUN_DIR}/logs"

# Print to stdout: Buildkite's main log view surfaces stdout, and the
# failure reason must be readable there without digging through artifacts.
fail() {
  echo "GPC_DRAM_POOL_FAIL: $*"
  exit 1
}

# ---------------------------------------------------------------------------
# Version preflight: the runtime must match pyproject.toml exactly.
# ---------------------------------------------------------------------------
RAIDEN_PIN="$(sed -nE 's/^[[:space:]]*"tpu-raiden-torch==([^"]+)".*/\1/p' \
  "${repo_root}/pyproject.toml" | head -1)"
[[ -n "${RAIDEN_PIN}" ]] \
  || fail "no tpu-raiden-torch== pin found in pyproject.toml"

python3 - "${RAIDEN_PIN}" <<'PY'
import importlib.metadata
import sys

raiden_pin = sys.argv[1]
raiden_installed = importlib.metadata.version("tpu-raiden-torch")
if raiden_installed != raiden_pin:
    raise SystemExit(
        f"installed tpu-raiden-torch {raiden_installed} != pyproject pin "
        f"{raiden_pin}: the CI image must install the pinned wheel")
print(f"tpu-raiden-torch {raiden_installed} matches pyproject pin")
PY

for cmd in global_registry_server kv_cache_host_store_node_main; do
  command -v "${cmd}" >/dev/null || fail \
    "${cmd} not found: tpu-raiden-torch==${RAIDEN_PIN} does not bundle the \
global DRAM pool control plane — bump the pin in pyproject.toml"
done
echo "control plane from the tpu-raiden-torch ${RAIDEN_PIN} wheel"

# ---------------------------------------------------------------------------
# Global registry: the directory and placement plane everything else joins.
# ---------------------------------------------------------------------------
export GLOG_alsologtostderr=1
global_registry_server --port="${REGISTRY_PORT}" \
  >"${RUN_DIR}/logs/registry.log" 2>&1 &
REGISTRY_PID=$!

SERVER_PID=""
NODE_PID=""
cleanup() {
  local pid
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  sleep 2
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -KILL "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT

check_control_plane() {
  kill -0 "${REGISTRY_PID}" 2>/dev/null || {
    tail -50 "${RUN_DIR}/logs/registry.log" || true
    fail "global_registry_server exited"
  }
  if [[ -n "${NODE_PID}" ]]; then
    kill -0 "${NODE_PID}" 2>/dev/null || {
      tail -50 "${RUN_DIR}/logs/store_node.log" || true
      fail "kv_cache_host_store_node_main exited"
    }
  fi
}

wait_port() {
  local port="$1" what="$2" timeout="${3:-60}" waited=0
  until (exec 3<>"/dev/tcp/${BIND_HOST}/${port}") 2>/dev/null; do
    exec 3>&- 2>/dev/null || true
    check_control_plane
    [[ "${waited}" -lt "${timeout}" ]] \
      || fail "${what} (${BIND_HOST}:${port}) not reachable after ${waited}s"
    sleep 1
    waited=$((waited + 1))
  done
  exec 3>&- 2>/dev/null || true
  echo "${what} port ${port} reachable"
}

wait_port "${REGISTRY_PORT}" "global registry"

# ---------------------------------------------------------------------------
# The store node. Started before the serving engine on purpose: the node
# needs no ordering help — it polls the registry for the pool group's
# KVTransferSpec (published only once the serving engine constructs its
# store) and only then sizes its pool, registers on tier 1, and starts
# heartbeating free capacity. Booting it first exercises exactly that
# spec-waiting path.
# ---------------------------------------------------------------------------
kv_cache_host_store_node_main \
  --job_name=gpc-ci-store-node \
  --kv_pool_group="${KV_POOL_GROUP}" \
  --evict_tier=1 \
  --global_registry_address="${BIND_HOST}:${REGISTRY_PORT}" \
  --store_server_ip="${BIND_HOST}" \
  --raiden_controller_port="${NODE_CONTROLLER_PORT}" \
  --dram_budget_bytes="${NODE_DRAM_BUDGET_BYTES}" \
  >"${RUN_DIR}/logs/store_node.log" 2>&1 &
NODE_PID=$!

# ---------------------------------------------------------------------------
# The serving replica.
# ---------------------------------------------------------------------------
COMPILATION_CONFIG='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

KV_TRANSFER_CONFIG='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${OFFLOAD_CONTROLLER_PORT}"',"raiden_job_name":"gpc-ci-serve","global_registry_address":"'"${BIND_HOST}"':'"${REGISTRY_PORT}"'","store_server_ip":"'"${BIND_HOST}"'","kv_pool_group":"'"${KV_POOL_GROUP}"'"}}'

(
  # Registry lookups only ever hit across engines whose block-hash chains
  # match, and vLLM seeds the chain from os.urandom when PYTHONHASHSEED is
  # unset; the connector refuses a registry-enabled boot without it.
  export PYTHONHASHSEED=0
  # The demotion side of the pool: the monitor heartbeats this store's free
  # capacity to the registry, and the sweep demotes cold blocks once free
  # blocks fall under the low watermark. The aggressive watermarks and
  # short periods are what let a twelve-prompt smoke drive the full
  # demote-and-recall cycle in seconds instead of hours.
  export RAIDEN_ENABLE_STORE_MONITOR=1
  export RAIDEN_ENABLE_EVICT_SWEEP=1
  export RAIDEN_STORE_MONITOR_HEARTBEAT_S=2
  export RAIDEN_EVICT_SWEEP_PERIOD_S=2
  export RAIDEN_EVICT_LOW_WATERMARK=0.2
  export RAIDEN_EVICT_HIGH_WATERMARK=0.4

  # The offload store chain (store-job build, save fence, admission) logs
  # only at DEBUG; rerun with VLLM_LOGGING_LEVEL=DEBUG when triaging it.
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

  export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax
  export VLLM_XLA_CHECK_RECOMPILATION=0 SKIP_JAX_PRECOMPILE=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export USE_MOE_SPARSE_CORE=0
  export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback
  export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
  export TPU_GDN_CONV_STATE_TILE_PAD=1
  export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_latency_hiding_scheduler=false"
  export VLLM_CACHE_ROOT="${RUN_DIR}/cache"
  export TORCHINDUCTOR_CACHE_DIR="${RUN_DIR}/cache/inductor"
  export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_S}"
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
  export TPU_RAIDEN_QWEN35_ADMISSION=1
  export TPU_RAIDEN_TRANSFER_PARALLELISM="${PCP}"
  export TPU_ENABLE_D2H_TRANSFER=1 TPU_HMA_D2H_COPY_IMPL=device_put
  export TPU_HMA_MIN_DONE_RECVING_BATCH=1 TPU_HMA_MAX_DONE_RECVING_BATCH=1
  export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1 TPU_HMA_MAX_INFLIGHT_PULLS=8
  export TPU_MAX_HOST_KV_BUFFER_SIZE=64

  # max_num_seqs=1 removes batching nondeterminism, so a cold/recalled
  # difference is the cache and nothing else.
  exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --seed 42 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs 1 \
    --no-disable-hybrid-kv-cache-manager \
    --attention-backend CUSTOM \
    --mamba-cache-mode align \
    --enable-prompt-tokens-details \
    --no-enable-log-requests \
    --compilation-config "${COMPILATION_CONFIG}" \
    --enable-prefix-caching \
    --no-async-scheduling \
    --block-size "${BLOCK_SIZE}" \
    --tensor-parallel-size 1 \
    --prefill-context-parallel-size "${PCP}" \
    --cp-kv-cache-interleave-size "${CP_KV_CACHE_INTERLEAVE_SIZE}" \
    --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
    --host "${BIND_HOST}" \
    --port "${PORT}" \
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
) >"${RUN_DIR}/logs/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "http://${BIND_HOST}:${PORT}/health" >/dev/null; do
  check_control_plane
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${RUN_DIR}/logs/server.log" || true
    fail "server exited during startup"
  fi
  [[ "${SECONDS}" -lt "${deadline}" ]] \
    || fail "server did not become healthy in ${STARTUP_TIMEOUT_S}s"
  sleep 15
done
echo "server is healthy"

# Engine readiness doubles as the offload store's deferred
# worker-registration gate; the short prompt stays far below one offloaded
# block so it perturbs no smoke-stage state.
until curl -sf --max-time 300 -X POST \
    "http://${BIND_HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
    >/dev/null; do
  check_control_plane
  [[ "${SECONDS}" -lt "${deadline}" ]] \
    || fail "engine not ready in ${STARTUP_TIMEOUT_S}s"
  echo "engine compiling, waiting 30s..."
  sleep 30
done
echo "engine is ready"

# Sizing tripwire: the smoke's tier arithmetic assumes the host pool holds
# POOL_PROMPTS_MIN..POOL_PROMPTS_MAX prompts' worth of KV. If the model's
# bytes per offloaded block drift, retune CPU_BYTES_TO_USE here rather
# than letting the recall stages silently turn vacuous. Checked only after
# engine-ready: the store (and this log line) only exists once every
# worker has registered.
pool_blocks="$(sed -nE \
  's/.*KVCacheStore up: .*\(([0-9]+) offloaded blocks x [0-9]+\).*/\1/p' \
  "${RUN_DIR}/logs/server.log" | head -1)"
[[ -n "${pool_blocks}" ]] \
  || fail "could not find the KVCacheStore capacity line in the server log"
pool_prompts=$((pool_blocks / OFFLOADED_BLOCKS_PER_PROMPT))
echo "host DRAM pool holds ${pool_prompts} prompts of KV\
 (${pool_blocks} offloaded blocks)"
if [[ "${pool_prompts}" -lt "${POOL_PROMPTS_MIN}" || \
      "${pool_prompts}" -gt "${POOL_PROMPTS_MAX}" ]]; then
  fail "host pool holds ${pool_prompts} prompts' worth of KV, outside \
[${POOL_PROMPTS_MIN}, ${POOL_PROMPTS_MAX}]: adjust CPU_BYTES_TO_USE so the \
${GPC_PROMPTS}-prompt smoke still overflows the pool without starving it"
fi

# The serving store publishes the pool group's spec when it comes up; the
# node polls the registry for it (1s doubling to 1min), builds its pool,
# and exposes its controller. A node stuck here means the spec handshake
# is broken.
wait_port "${NODE_CONTROLLER_PORT}" "store node controller" 180

check_control_plane
python3 "${script_dir}/smoke_gpc_global_dram_pool.py" \
  --base "http://${BIND_HOST}:${PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --prompts "${GPC_PROMPTS}" \
  --pool-prompts "${pool_prompts}" \
  --prompt-tokens "${GPC_PROMPT_TOKENS}" \
  --sweep-wait-s "${GPC_SWEEP_WAIT_S}" \
  2>&1 | tee "${RUN_DIR}/logs/smoke.log"

# Post-mortem for triage; never gating.
echo "--- server.log offload/store activity"
grep -iE "offloading|fence|store job|cannot store|hit .* offloaded\
|demot|sweep|placement" "${RUN_DIR}/logs/server.log" | tail -40 || true
echo "--- store_node.log tail"
tail -30 "${RUN_DIR}/logs/store_node.log" || true
echo "--- registry.log tail"
tail -30 "${RUN_DIR}/logs/registry.log" || true

echo "GPC_DRAM_POOL_OK"

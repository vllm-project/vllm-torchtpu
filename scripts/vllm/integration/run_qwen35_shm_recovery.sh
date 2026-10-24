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
# Qwen3.5 KV-offload SHM crash-recovery rig.
#
# This is run_qwen35_gpc_global_dram_pool.sh's PROVEN offload stack (global
# registry + host store node + serving replica, with the store-monitor / evict-
# sweep env and the global_registry_address/store_server_ip/kv_pool_group config
# that actually drive the offload SAVE path for this hybrid model) with two
# additions:
#
#   1. RAIDEN_SHM_KEY on the serving replica, so its host DRAM pool + metadata
#      live in named POSIX shm that survives a crash (the feature under test).
#   2. A three-phase crash/recover choreography instead of a single smoke.
#
# Offload SAVE (HBM -> host pool, D2H) happens per request as blocks are produced;
# it does NOT need HBM eviction. The populate smoke proves the round-trip by
# emptying HBM with POST /reset_prefix_cache (external store kept) and recalling
# the target, so the recall must resolve through the shm-backed host pool. The
# target (index N-3) stays pool-resident: the 24 GiB pool is far below its
# demotion watermark under this 12-prompt smoke, so nothing demotes to the store
# node (whose DRAM is not shm-backed and would not recover).
#
# Phases, each its own full stack, driven by smoke_shm_recovery.py over one
# --state file:
#   A populate       RAIDEN_SHM_KEY=K. Seed, then recall prompt N-3 and prove in
#                    process it is an EXTERNAL (host-pool) hit; record it. Then
#                    SIGKILL the whole stack, LEAVING /dev/shm intact.
#   B verify-recover RAIDEN_SHM_KEY=K, fresh stack. Recall prompt N-3: it must be
#                    an EXTERNAL hit with first tokens matching pre-crash -- the
#                    pool + metadata recovered from shm. (positive, P)
#   C verify-cold    a DIFFERENT RAIDEN_SHM_KEY, fresh stack. Recall prompt N-3:
#                    it must MISS, proving P's hit is attributable to the shared
#                    key and not to /dev/shm residue. (negative, N)
#
# CRITICAL: cross-restart recovery needs a stable block-hash chain; vLLM seeds
# it from os.urandom when PYTHONHASHSEED is unset, so it is pinned below.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

fail() {
  echo "QWEN35_SHM_RECOVERY_RIG_FAIL: $*"
  if [[ -n "${CUR_SERVER_LOG:-}" && -f "${CUR_SERVER_LOG}" ]]; then
    echo "----- offload/store activity (${CUR_SERVER_LOG}) -----"
    grep -aiE "KVCacheStore up|registered|offloading|fence|store job|cannot store|hit .* offloaded|demot|sweep|placement|enable_shm|shm-backed|ERROR|WARNING" \
      "${CUR_SERVER_LOG}" 2>/dev/null | tail -60 || true
  fi
  exit 1
}

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
LOAD_FORMAT_ARGS=()
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer)
fi
BIND_HOST="${BIND_HOST:-127.0.0.1}"
RUN_ROOT="${RUN_ROOT:-${CDK_OUTPUT_DIR:-/tmp}/qwen35_shm_recovery_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
STATE_FILE="${STATE_FILE:-${RUN_DIR}/shm_recovery_state.json}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

pick_port() {
  python3 - "$1" <<'PY'
import socket, sys
base = int(sys.argv[1])
for port in range(base, base + 200):
    try:
        s = socket.socket(); s.bind(("", port)); s.close(); print(port); break
    except OSError:
        continue
else:
    raise SystemExit("no free port")
PY
}
PORT="${PORT:-$(pick_port 8100)}" || fail "no free server port"
REGISTRY_PORT="${REGISTRY_PORT:-$(pick_port 21500)}" || fail "no registry port"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-$(pick_port 21000)}" \
  || fail "no offload controller port"
NODE_CONTROLLER_PORT="${NODE_CONTROLLER_PORT:-$(pick_port 20500)}" \
  || fail "no node controller port"

# One pool group covers the serving replica (tier 0) and the store node (tier 1).
KV_POOL_GROUP="${KV_POOL_GROUP:-shmrec-pool}"

# Engine geometry -- matched to the PROVEN-GREEN gpc-correctness config (run
# j-66f35cb7 / j-c7699527), which offloads and recalls with external hits using
# the SAME reset-driven proof this rig's smoke uses (seed, /reset_prefix_cache to
# empty HBM, recall -> external hit). A large HBM override + seqs>1 is what boots
# this hybrid model reliably (j-b6cb20f5 reached KVCacheStore-up with no OOM);
# starving HBM to force eviction is neither needed here nor OOM-safe.
PCP="${PCP:-8}"
CP_KV_CACHE_INTERLEAVE_SIZE="${CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
# 24576 (>18292-token prompts + gen) instead of gpc's 65536: max_model_len that
# large inflates compile/boot memory and slow-boots/OOM-kills the single-replica
# stack. It does not affect offload-save (that's block_size/override/cpu_bytes).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

GPC_PROMPTS="${GPC_PROMPTS:-12}"
GPC_PROMPT_TOKENS="${GPC_PROMPT_TOKENS:-18292}"
GPC_SWEEP_WAIT_S="${GPC_SWEEP_WAIT_S:-15}"
OFFLOADED_BLOCKS_PER_PROMPT=$((GPC_PROMPT_TOKENS / BLOCK_SIZE))
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-128}"
# 24 GiB, not gpc's 64 GiB: the whole pool is pre-allocated AND DMA-mapped
# (pinned RAM) at store construction, and pinning 64 GiB OOM-kills the pod there
# (j-651536c4, j-5cdfa83e -- died right at "blocking KVCacheStore construction").
# 24 GiB still holds far more than the 12 test prompts and pins ~2.7x less RAM.
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-24000000000}"
POOL_PROMPTS_MIN="${POOL_PROMPTS_MIN:-6}"
POOL_PROMPTS_MAX="${POOL_PROMPTS_MAX:-8}"
NODE_DRAM_BUDGET_BYTES="${NODE_DRAM_BUDGET_BYTES:-17179869184}"

# Recovery target: third-newest prompt (proven pool-resident in stage 2).
TARGET="${TARGET:-$((GPC_PROMPTS - 3))}"

RUN_ID="${RUN_ID:-$$_$(date +%s)}"
SHM_KEY="${SHM_KEY:-qwen35shmrec_${RUN_ID}}"
SHM_KEY_NEG="${SHM_KEY_NEG:-${SHM_KEY}_neg}"

if [[ -z "${PYTHON_BIN:-}" ]]; then PYTHON_BIN="python3"; fi

mkdir -p "${LOG_DIR}" || exit 1

COMPILATION_CONFIG='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

# Version preflight: the runtime must bundle the global-pool control plane.
RAIDEN_PIN="$(sed -nE 's/^[[:space:]]*"tpu-raiden-torch==([^"]+)".*/\1/p' \
  "${repo_root}/pyproject.toml" | head -1)"
for cmd in global_registry_server kv_cache_host_store_node_main; do
  command -v "${cmd}" >/dev/null || fail \
    "${cmd} not found: tpu-raiden-torch==${RAIDEN_PIN:-?} does not bundle the \
global DRAM pool control plane"
done

REGISTRY_PID=""
NODE_PID=""
SERVER_PID=""
CUR_SERVER_LOG=""

check_control_plane() {
  if [[ -n "${REGISTRY_PID}" ]] && ! kill -0 "${REGISTRY_PID}" 2>/dev/null; then
    tail -50 "${LOG_DIR}/registry.log" 2>/dev/null || true
    fail "global_registry_server exited"
  fi
  if [[ -n "${NODE_PID}" ]] && ! kill -0 "${NODE_PID}" 2>/dev/null; then
    tail -50 "${LOG_DIR}/store_node.log" 2>/dev/null || true
    fail "kv_cache_host_store_node_main exited"
  fi
}

wait_port() {  # $1=port $2=what $3=timeout
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

# Non-fatal port wait for the tier-1 store node. This rig never exercises tier-1
# (nothing demotes at these prompt counts): verify-recover reloads from the
# tier-0 shm pool and verify-cold only needs the serving replica. The 3rd
# store node's bring-up has been seen to hang in a transport-init retry loop
# (j-1e5bac6c, j-c2b53a38 both died here), so its readiness MUST NOT gate a
# phase. Warn and proceed on timeout / node exit; only a dead registry is fatal.
wait_port_soft() {  # $1=port $2=what $3=timeout
  local port="$1" what="$2" timeout="${3:-90}" waited=0
  until (exec 3<>"/dev/tcp/${BIND_HOST}/${port}") 2>/dev/null; do
    exec 3>&- 2>/dev/null || true
    if [[ -n "${REGISTRY_PID}" ]] && ! kill -0 "${REGISTRY_PID}" 2>/dev/null; then
      tail -50 "${LOG_DIR}/registry_"*.log 2>/dev/null || true
      fail "global_registry_server exited"
    fi
    if [[ -n "${NODE_PID}" ]] && ! kill -0 "${NODE_PID}" 2>/dev/null; then
      echo "${what}: store node process exited; proceeding without tier-1 (unused in this rig)"
      return 1
    fi
    if [[ "${waited}" -ge "${timeout}" ]]; then
      echo "${what} not reachable after ${waited}s; proceeding without tier-1 (unused in this rig)"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  exec 3>&- 2>/dev/null || true
  echo "${what} port ${port} reachable"
}

serve() {  # $1 = shm key for this stack's serving replica
  local shm_key="$1"
  export PYTHONHASHSEED=0
  export RAIDEN_SHM_KEY="${shm_key}"
  export GLOG_alsologtostderr=1
  export RAIDEN_ENABLE_STORE_MONITOR=1
  export RAIDEN_ENABLE_EVICT_SWEEP=1
  export RAIDEN_STORE_MONITOR_HEARTBEAT_S=2
  export RAIDEN_EVICT_SWEEP_PERIOD_S=2
  export RAIDEN_EVICT_LOW_WATERMARK=0.2
  export RAIDEN_EVICT_HIGH_WATERMARK=0.4
  export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax
  export VLLM_XLA_CHECK_RECOMPILATION=0 SKIP_JAX_PRECOMPILE=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export USE_MOE_SPARSE_CORE=0
  export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback
  export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
  export TPU_GDN_CONV_STATE_TILE_PAD=1
  export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_latency_hiding_scheduler=false"
  # The compile cache is 30-40 GB. It must NOT go on /dev/shm (tmpfs = RAM): that
  # competes with the 64 GB shm KV pool and OOM-kills the pod at store
  # construction (seen in j-5cdfa83e). Put it under RUN_DIR -> the GCS-backed
  # $CDK_OUTPUT_DIR when mounted (this is the config that reached store-up in
  # j-83bc96c7), i.e. off both RAM and the tiny container /tmp.
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${RUN_DIR}/cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUN_DIR}/cache/inductor}"
  mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
  export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_S}"
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
  export TPU_RAIDEN_QWEN35_ADMISSION=1
  export TPU_RAIDEN_TRANSFER_PARALLELISM="${PCP}"
  export TPU_ENABLE_D2H_TRANSFER=1 TPU_HMA_D2H_COPY_IMPL=device_put
  export TPU_HMA_MIN_DONE_RECVING_BATCH=1 TPU_HMA_MAX_DONE_RECVING_BATCH=1
  export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1 TPU_HMA_MAX_INFLIGHT_PULLS=8
  export TPU_MAX_HOST_KV_BUFFER_SIZE=64
  # DEBUG so the offload store/save chain (store-job build, fence, admission,
  # demotion) logs -- it is silent at INFO. Drop to INFO once offload is proven.
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  export VLLM_SERVER_DEV_MODE=1

  local kvt
  kvt='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${OFFLOAD_CONTROLLER_PORT}"',"raiden_job_name":"qwen35-shm-recovery","global_registry_address":"'"${BIND_HOST}"':'"${REGISTRY_PORT}"'","store_server_ip":"'"${BIND_HOST}"'","kv_pool_group":"'"${KV_POOL_GROUP}"'"}}'

  exec vllm serve "${MODEL}" \
    "${LOAD_FORMAT_ARGS[@]}" \
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
    --max-num-seqs 16 \
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
    --kv-transfer-config "${kvt}"
}

# Bring up registry -> store node -> serving replica, and block until the engine
# is ready, the pool is sized, and the node controller is up.
launch_stack() {  # $1 = phase label, $2 = shm key
  local label="$1" shm_key="$2"
  local server_log="${LOG_DIR}/server_${label}.log"
  CUR_SERVER_LOG="${server_log}"
  echo "===== [${label}] launching stack (registry+node+server), \
RAIDEN_SHM_KEY=${shm_key}, PCP${PCP}, port ${PORT} ====="

  ( export GLOG_alsologtostderr=1
    exec global_registry_server --port="${REGISTRY_PORT}" ) \
    >"${LOG_DIR}/registry_${label}.log" 2>&1 &
  REGISTRY_PID=$!
  wait_port "${REGISTRY_PORT}" "[${label}] global registry" 60

  ( exec kv_cache_host_store_node_main \
      --job_name=shmrec-store-node \
      --kv_pool_group="${KV_POOL_GROUP}" \
      --evict_tier=1 \
      --global_registry_address="${BIND_HOST}:${REGISTRY_PORT}" \
      --store_server_ip="${BIND_HOST}" \
      --raiden_controller_port="${NODE_CONTROLLER_PORT}" \
      --dram_budget_bytes="${NODE_DRAM_BUDGET_BYTES}" ) \
    >"${LOG_DIR}/store_node_${label}.log" 2>&1 &
  NODE_PID=$!

  # Tee the server's output to BOTH the log file and stdout, so the boot/crash
  # cause is captured in the CDK container log even without a GCS mount (server
  # logs otherwise go only to a pod-local file that is lost when the pod dies).
  ( serve "${shm_key}" ) > >(tee "${server_log}" | sed 's/^/[srvA] /') 2>&1 &
  SERVER_PID=$!

  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  until curl -sf --max-time 10 "http://${BIND_HOST}:${PORT}/health" >/dev/null; do
    check_control_plane
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${server_log}" || true
      fail "[${label}] server exited during startup (see ${server_log})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] \
      || fail "[${label}] server not healthy in ${STARTUP_TIMEOUT_S}s"
    sleep 15
  done
  echo "[${label}] server is healthy"

  until curl -sf --max-time 300 -X POST "http://${BIND_HOST}:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
      >/dev/null; do
    check_control_plane
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${server_log}" || true
      fail "[${label}] server exited during warm up (see ${server_log})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] \
      || fail "[${label}] engine not ready in ${STARTUP_TIMEOUT_S}s"
    echo "[${label}] engine compiling, waiting 30s..."
    sleep 30
  done
  echo "[${label}] engine is ready"

  local pool_blocks
  pool_blocks="$(sed -nE \
    's/.*KVCacheStore up: .*\(([0-9]+) offloaded blocks x [0-9]+\).*/\1/p' \
    "${server_log}" | head -1)"
  if [[ -n "${pool_blocks}" ]]; then
    echo "[${label}] host DRAM pool holds $((pool_blocks / OFFLOADED_BLOCKS_PER_PROMPT)) prompts (${pool_blocks} offloaded blocks)"
  fi
  wait_port_soft "${NODE_CONTROLLER_PORT}" "[${label}] store node controller" 90 || true
}

hard_kill_stack() {
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  # SIGKILL the retitled TP workers too, WITHOUT unlinking /dev/shm.
  pkill -KILL '^VLLM' 2>/dev/null || true
  SERVER_PID=""; NODE_PID=""; REGISTRY_PID=""
  for _ in $(seq 1 60); do
    pgrep '^VLLM' >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 3
}

list_shm() {
  local f
  for f in /dev/shm/"${1}"*; do
    [[ -e "${f}" ]] && printf "%s\n" "${f##*/}"
  done
}
unlink_shm() { local n; for n in $(list_shm "$1"); do rm -f "/dev/shm/${n}" 2>/dev/null || true; done; }

# shellcheck disable=SC2317
cleanup() { hard_kill_stack; unlink_shm "${SHM_KEY}"; unlink_shm "${SHM_KEY_NEG}"; }
trap cleanup EXIT

run_phase() {  # $1 = driver phase, $2 = log name
  local phase="$1" logname="$2"
  "${PYTHON_BIN}" "${script_dir}/smoke_shm_recovery.py" \
    --phase "${phase}" --state "${STATE_FILE}" \
    --base "http://${BIND_HOST}:${PORT}" --model "${SERVED_MODEL_NAME}" \
    --prompts "${GPC_PROMPTS}" --prompt-tokens "${GPC_PROMPT_TOKENS}" \
    --sweep-wait-s "${GPC_SWEEP_WAIT_S}" --target "${TARGET}" \
    2>&1 | tee "${LOG_DIR}/${logname}.log"
  return "${PIPESTATUS[0]}"
}

# Clean slate.
hard_kill_stack
unlink_shm "${SHM_KEY}"; unlink_shm "${SHM_KEY_NEG}"
rm -f "${STATE_FILE}" 2>/dev/null || true

# HAPPY_PATH=1: just prove offload works with a shm-backed host pool (one boot,
# no crash/recover). This is the de-risked "does the feature work at all" test:
# launch with RAIDEN_SHM_KEY, seed, and recall from the shm-backed pool with
# external hits + correct tokens. Set to 0 for the full crash-recovery ladder.
HAPPY_PATH="${HAPPY_PATH:-1}"

# ---- Phase A: populate (seed + recall) under SHM_KEY ----
launch_stack A "${SHM_KEY}"
run_phase populate populate \
  || fail "populate phase failed -- offload with shm-backed pool did not round-trip \
(see ${LOG_DIR}/populate.log)"
echo "--- shm segments under ${SHM_KEY} (shm-backed host pool):"
list_shm "${SHM_KEY}"

if [[ "${HAPPY_PATH}" == "1" ]]; then
  echo "QWEN35_SHM_RECOVERY_RIG_OK (happy path: offload round-tripped through the \
shm-backed host pool; crash/recover not exercised)"
  exit 0
fi

echo "--- crashing stack A (SIGKILL; /dev/shm left untouched) ---"
hard_kill_stack
echo "--- shm segments after the crash (must still be present):"
list_shm "${SHM_KEY}"

# ---- Phase B: recover under the SAME SHM_KEY ----
launch_stack B "${SHM_KEY}"
run_phase verify-recover recover \
  || fail "verify-recover failed: offloaded KV did NOT recover across the crash \
under the same key (see ${LOG_DIR}/recover.log)"
hard_kill_stack

# ---- Phase C (negative control): a DIFFERENT key must cold-start ----
launch_stack C "${SHM_KEY_NEG}"
run_phase verify-cold cold \
  || fail "verify-cold failed: a different key still hit, so the recovery signal \
is not attributable to the shared key (see ${LOG_DIR}/cold.log)"

echo "QWEN35_SHM_RECOVERY_RIG_OK"

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

# Validates Global Prefix Caching (GPC) across independent serving replicas
# in a multi-host TPU environment.
#
# Multi-Host Architecture & Design Invariants:
# - Standalone Full-Host Engines: Each host runs one standalone engine (PCP8/TP1)
#   matching the prefill geometry of the production GPC configuration. Both engines
#   register their local offload stores with the centralized global registry.
# - Cross-Replica KV Cache Sharing: Prefixes computed and cached in host DRAM by
#   Replica A are advertised via the global registry, resolving as remote hits
#   for Replica B and transferred peer-to-peer over the network via read_remote.
# - Single Replica Per Host Hardware Invariant: Sub-host chip partitions at nonzero
#   rank offsets enumerate TPU devices in physical coordinate order rather than logical
#   rank order. Out-of-mesh XLA collectives subsequently assemble sequence chunks out
#   of order, corrupting generation. Allocating the full 8-chip host per replica
#   ensures canonical rank-ordered device enumeration.
# - Multi-Role Coordination (GPC_ROLE):
#   - head: Launches Replica A next to the orchestrator-provided registry container,
#     warms up both engines, and executes the smoke validation suite against
#     Replicas A and B.
#   - worker: Launches Replica B in the background and serves incoming requests
#     until torn down by the head orchestrator.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPC_ROLE="${GPC_ROLE:-head}"
# Configure host IP mappings for the registry and peer-to-peer KV transfer endpoints.
GPC_HEAD_IP="${GPC_HEAD_IP:?set GPC_HEAD_IP to the head host IP}"
GPC_MY_IP="${GPC_MY_IP:?set GPC_MY_IP to this host IP}"
if [[ "${GPC_ROLE}" == "head" ]]; then
  GPC_PEER_IP="${GPC_PEER_IP:?set GPC_PEER_IP to the worker host IP}"
fi

# Stream model weights via runai_streamer when a GCS snapshot URI is provided,
# preventing local boot disk exhaustion on CI worker nodes.
MODEL="${GPC_MODEL_URI:-${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
LOAD_FORMAT_ARGS=()
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer)
fi
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_gpc_ci}"
LOG_DIR="${RUN_ROOT}/logs"
PORT_A="${PORT_A:-8100}"
PORT_B="${PORT_B:-8200}"
# Use listen ports strictly below the Linux ephemeral port range (32768-60999).
# If unassigned ephemeral source ports collide with the target port during early
# TCP health checks before the server binds, self-connection can lock the port
# and cause subsequent bind() calls to fail with EADDRINUSE.
REGISTRY_PORT="${REGISTRY_PORT:-28500}"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-27800}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

# Engine geometry configuration and cache namespace invariants:
# - Namespace Alignment: Cross-replica KV cache sharing requires identical offload
#   key namespaces (derived from model identity, KV cache dtype, block size,
#   world size, and context parallel geometry). Any divergence results
#   in silent cache misses.
# - Offload Span Sizing: Block size is set to 2048 (yielding an offload span of
#   16,384 tokens across 8 PCP ranks), keeping probe documents near 18k tokens;
#   past a 32k-token span this engine deterministically echoes the document body
#   instead of answering.
PCP="${PCP:-8}"
CP_KV_CACHE_INTERLEAVE_SIZE="${CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-68719476736}"

mkdir -p "${LOG_DIR}"

fail() {
  echo "GPC_CI_FAIL: $*" >&2
  exit 1
}

# Wait for the global registry port to accept connections before launching engines.
# Store registration during engine initialization is non-retrying; premature startup
# would cause permanent registration failure.
wait_for_registry() {
  local timeout_s="$1" waited=0
  until (exec 3<>"/dev/tcp/${GPC_HEAD_IP}/${REGISTRY_PORT}") 2>/dev/null; do
    if [[ "${waited}" -ge "${timeout_s}" ]]; then
      fail "registry ${GPC_HEAD_IP}:${REGISTRY_PORT} not reachable after \
${waited}s"
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "registry ${GPC_HEAD_IP}:${REGISTRY_PORT} reachable"
}

COMPILATION_CONFIG='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'

kv_transfer_config() {
  local job_name="$1"
  printf '%s' '{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${OFFLOAD_CONTROLLER_PORT}"',"raiden_job_name":"'"${job_name}"'","global_registry_address":"'"${GPC_HEAD_IP}"':'"${REGISTRY_PORT}"'","store_server_ip":"'"${GPC_MY_IP}"'"}}'
}

launch_replica() {
  local name="$1" port="$2" pid_var="$3"
  (
    # Pin PYTHONHASHSEED=0 to guarantee deterministic block-hash chains across
    # replicas. Without fixed seeding, vLLM generates unique per-process hash salts,
    # preventing cross-replica prefix cache key resolution.
    export PYTHONHASHSEED=0
    export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax
    export VLLM_XLA_CHECK_RECOMPILATION=0 SKIP_JAX_PRECOMPILE=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export USE_MOE_SPARSE_CORE=0 ONEHOT_MOE_PERMUTE_THRESHOLD=1024
    export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback
    export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
    export TPU_GDN_CONV_STATE_TILE_PAD=1
    export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_latency_hiding_scheduler=false"
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled
    # Disable persistent compilation caching to conserve ephemeral disk space
    # during single-invocation CI integration runs.
    export VLLM_DISABLE_COMPILE_CACHE=1
    export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_S}"
    export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
    export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
    export TPU_RAIDEN_QWEN35_ADMISSION=1
    export TPU_RAIDEN_TRANSFER_PARALLELISM="${PCP}"
    export TPU_ENABLE_D2H_TRANSFER=1 TPU_HMA_D2H_COPY_IMPL=device_put
    export TPU_HMA_MIN_DONE_RECVING_BATCH=1 TPU_HMA_MAX_DONE_RECVING_BATCH=1
    export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1 TPU_HMA_MAX_INFLIGHT_PULLS=8
    export TPU_MAX_HOST_KV_BUFFER_SIZE=64

    exec vllm serve "${MODEL}" \
      "${LOAD_FORMAT_ARGS[@]}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --seed 42 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --enable-expert-parallel \
      --disable-custom-all-reduce \
      --gpu-memory-utilization 0.9 \
      --kv-cache-dtype fp8 \
      --language-model-only \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --no-disable-hybrid-kv-cache-manager \
      --attention-backend CUSTOM \
      --mamba-cache-mode align \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --compilation-config "${COMPILATION_CONFIG}" \
      --enable-prefix-caching \
      --no-async-scheduling \
      --block-size "${BLOCK_SIZE}" \
      --prefill-context-parallel-size "${PCP}" \
      --cp-kv-cache-interleave-size "${CP_KV_CACHE_INTERLEAVE_SIZE}" \
      --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --kv-transfer-config "$(kv_transfer_config "gpc-ci-${name}")"
  ) >"${LOG_DIR}/server_${name}.log" 2>&1 &
  printf -v "${pid_var}" '%s' "$!"
}

wait_healthy() {
  local name="$1" url="$2" pid="$3" deadline="$4"
  until curl -sf --max-time 10 "${url}/health" >/dev/null; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      tail -80 "${LOG_DIR}/server_${name}.log" >&2 || true
      fail "replica ${name} exited during startup"
    fi
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      fail "replica ${name} did not become healthy in ${STARTUP_TIMEOUT_S}s"
    fi
    sleep 15
  done
  echo "replica ${name} is healthy"
}

# Warm up the engine and verify end-to-end offload store registration.
# The initial request triggers JIT compilation and confirms worker registration.
# A minimal prompt length (well below the offload span threshold) ensures no
# partial prefixes are published that could interfere with subsequent smoke validation.
warm_up() {
  local name="$1" url="$2" deadline="$3"
  until curl -sf --max-time 300 -X POST \
      "${url}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 1}" \
      >/dev/null; do
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      fail "replica ${name} engine not ready in ${STARTUP_TIMEOUT_S}s"
    fi
    echo "replica ${name} compiling, waiting 30s..."
    sleep 30
  done
  echo "replica ${name} engine is ready"
}

SERVER_PID=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${GPC_ROLE}" == "worker" ]]; then
  # Allow extended registry timeout to accommodate head image pull and initialization.
  wait_for_registry 1200
  echo "===== Starting replica B (${GPC_MY_IP}, full host, PCP${PCP}) ====="
  launch_replica b "${PORT_B}" SERVER_PID
  # Stream server logs to stdout so the orchestrator can collect them
  # through docker logs.
  tail -n +1 -F "${LOG_DIR}/server_b.log" &
  wait "${SERVER_PID}" || true
  fail "replica b exited"
fi

# ---------------------------------------------------------------------------
# Head role: Launch replica A, verify health across both hosts, and execute smoke gates.
# ---------------------------------------------------------------------------
wait_for_registry 300

echo "===== Starting replica A (${GPC_MY_IP}, full host, PCP${PCP}) ====="
launch_replica a "${PORT_A}" SERVER_PID

BASE_A="http://127.0.0.1:${PORT_A}"
BASE_B="http://${GPC_PEER_IP}:${PORT_B}"

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
wait_healthy a "${BASE_A}" "${SERVER_PID}" "${deadline}"
# Replica B runs on the remote worker host; supervise its readiness via HTTP health checks.
wait_healthy b "${BASE_B}" "" "${deadline}"

warm_up a "${BASE_A}" "${deadline}" &
WARM_A=$!
warm_up b "${BASE_B}" "${deadline}" &
WARM_B=$!
wait "${WARM_A}" || fail "replica a warm-up failed"
wait "${WARM_B}" || fail "replica b warm-up failed"

python3 "${script_dir}/smoke_gpc_global_offloading.py" \
  --base-a "${BASE_A}" \
  --base-b "${BASE_B}" \
  --model "${SERVED_MODEL_NAME}" \
  --min-prompt-tokens "$((BLOCK_SIZE * PCP + 1))" \
  2>&1 | tee "${LOG_DIR}/smoke.log"

echo "GPC_CI_OK"

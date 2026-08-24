#!/bin/bash
# GLOBAL PREFIX CACHE across two engine replicas: two-host CI version of
# recipes/experimental/torchtpu-vllm/disagg-offloading/
# qwen35-gpc-global-offloading.yml.
#
# Each host runs one full-host PCP8/TP1 standalone engine, the same shared
# geometry (pcp8_prefill) the recipe's two stacks hit each other through,
# and both offload stores join one global registry on the head host. A
# prefix computed and DRAM-cached by replica A is then a byte-compatible
# remote hit for replica B, pulled P2P over the network via read_remote.
# The recipe's disagg legs are dropped: cross-replica registry hits are
# the feature under test, and the disagg path has its own CI job.
#
# One replica per host is a correctness requirement: a PCP engine on a
# sub-host chip block at a nonzero rank offset enumerates its devices in
# physical-coordinate order, which disagrees with rank
# order on this machine type, and the XLA collectives outside the PCP
# Pallas mesh assemble sequence chunks by device order, garbling every
# generation that replica produces. A full-host engine starting at chip 0
# enumerates in rank order.
#
# Roles, selected by GPC_ROLE:
#   head    runs replica A and the smoke gates;
#           .buildkite/scripts/run_gpc_multihost.sh runs it in a container
#           on the slice's first host, next to the registry container it
#           also provides (the registry binary only binds the IPv6
#           wildcard, so it needs its own network namespace on these
#           IPv6-disabled hosts).
#   worker  runs replica B in the foreground until torn down; the same
#           orchestrator runs it in a container on the second host.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPC_ROLE="${GPC_ROLE:-head}"
# The head host's IP carries the registry and replica A's store; the
# worker needs both, the head needs its own for the store it advertises.
GPC_HEAD_IP="${GPC_HEAD_IP:?set GPC_HEAD_IP to the head host IP}"
GPC_MY_IP="${GPC_MY_IP:?set GPC_MY_IP to this host IP}"
if [[ "${GPC_ROLE}" == "head" ]]; then
  GPC_PEER_IP="${GPC_PEER_IP:?set GPC_PEER_IP to the worker host IP}"
fi

# GPC_MODEL_URI is a gs:// snapshot directory the orchestrator resolved;
# runai_streamer streams it so nothing lands on the small boot disks. A
# plain HuggingFace id still works for local runs.
MODEL="${GPC_MODEL_URI:-${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
LOAD_FORMAT_ARGS=()
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer)
fi
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_gpc_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
PORT_A="${PORT_A:-8100}"
PORT_B="${PORT_B:-8200}"
# Listen ports stay below the kernel's ephemeral range (32768+): the
# workers probe the controller's address on their own host before it has
# bound, and a probe whose random source port lands on the target port
# self-connects, occupying the port so the controller's bind then fails.
REGISTRY_PORT="${REGISTRY_PORT:-28500}"
OFFLOAD_CONTROLLER_PORT="${OFFLOAD_CONTROLLER_PORT:-27800}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

# Engine geometry, mirroring the recipe's pcp8_prefill pod except for the
# block size. Cross-replica hits require both replicas identical: the
# offload key namespace hashes model, kv dtype, block sizes, world size,
# and cp geometry, and any mismatch is a silent zero-hit run. The
# namespace only has to match between these two replicas, so the block
# size can differ from the recipe's 4096: at 4096 one offload span is
# 32768 tokens, and past that length this engine deterministically
# echoes the document body instead of answering. 2048 keeps the span at
# 16384 and the probe documents near 18k tokens, where generation is
# well behaved on this stack.
PCP="${PCP:-8}"
CP_KV_CACHE_INTERLEAVE_SIZE="${CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-68719476736}"

GPC_PUBLISH_WAIT_S="${GPC_PUBLISH_WAIT_S:-15}"

mkdir -p "${RUN_DIR}/logs"

fail() {
  echo "GPC_CI_FAIL: $*" >&2
  exit 1
}

# The registry never retries a failed RegisterStore, so both roles wait
# for its port before starting their engine. The registry itself runs in
# the orchestrator's own container; its death terminates this container
# from outside, so nothing here polls it after startup.
wait_for_registry() {
  local timeout_s="$1" waited=0
  until (exec 3<>"/dev/tcp/${GPC_HEAD_IP}/${REGISTRY_PORT}") 2>/dev/null; do
    exec 3>&- 2>/dev/null || true
    if [[ "${waited}" -ge "${timeout_s}" ]]; then
      fail "registry ${GPC_HEAD_IP}:${REGISTRY_PORT} not reachable after \
${waited}s"
    fi
    sleep 5
    waited=$((waited + 5))
  done
  exec 3>&- 2>/dev/null || true
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
    # Cross-replica sharing needs identical block-hash chains: vLLM seeds
    # the chain with os.urandom(32) when PYTHONHASHSEED is unset, so
    # unseeded replicas could never match each other's registry keys.
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
    # A one-run engine gets nothing from persisting compile caches, and
    # the agents have small boot disks.
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
  ) >"${RUN_DIR}/logs/server_${name}.log" 2>&1 &
  printf -v "${pid_var}" '%s' "$!"
}

wait_healthy() {
  local name="$1" url="$2" pid="$3" deadline="$4"
  until curl -sf --max-time 10 "${url}/health" >/dev/null; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      tail -80 "${RUN_DIR}/logs/server_${name}.log" >&2 || true
      fail "replica ${name} exited during startup"
    fi
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      fail "replica ${name} did not become healthy in ${STARTUP_TIMEOUT_S}s"
    fi
    sleep 15
  done
  echo "replica ${name} is healthy"
}

# A first request through each replica proves the engine serves and, since
# the offload store blocks until every worker registers, that registration
# completed too. The short prompt stays far below one offload span, so it
# publishes nothing the smoke stage could hit.
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

if [[ "${GPC_ROLE}" == "worker" ]]; then
  SERVER_PID_B=""
  # shellcheck disable=SC2317  # invoked via the EXIT trap
  cleanup_worker() {
    if [[ -n "${SERVER_PID_B}" ]]; then
      kill -TERM "${SERVER_PID_B}" 2>/dev/null || true
      sleep 2
      kill -KILL "${SERVER_PID_B}" 2>/dev/null || true
    fi
  }
  trap cleanup_worker EXIT

  # The head's container may still be pulling its image; give the
  # registry longer than a local startup would need.
  wait_for_registry 1200
  echo "===== Starting replica B (${GPC_MY_IP}, full host, PCP${PCP}) ====="
  launch_replica b "${PORT_B}" SERVER_PID_B
  # The engine log is the only view the orchestrator has into this
  # container, so mirror it to stdout for docker logs.
  tail -n +1 -F "${RUN_DIR}/logs/server_b.log" &
  wait "${SERVER_PID_B}" || true
  fail "replica b exited"
fi

# ---------------------------------------------------------------------------
# Head role: replica A, then the smoke gates against both replicas. The
# registry is the orchestrator's own container on this host.
# ---------------------------------------------------------------------------
SERVER_PID_A=""
cleanup_head() {
  if [[ -n "${SERVER_PID_A}" ]]; then
    kill -TERM "${SERVER_PID_A}" 2>/dev/null || true
    sleep 2
    kill -KILL "${SERVER_PID_A}" 2>/dev/null || true
  fi
}
trap cleanup_head EXIT

wait_for_registry 300

echo "===== Starting replica A (${GPC_MY_IP}, full host, PCP${PCP}) ====="
launch_replica a "${PORT_A}" SERVER_PID_A

BASE_A="http://127.0.0.1:${PORT_A}"
BASE_B="http://${GPC_PEER_IP}:${PORT_B}"

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
wait_healthy a "${BASE_A}" "${SERVER_PID_A}" "${deadline}"
# Replica B runs on the other host; its failures surface through the
# orchestrator's docker-logs collection rather than a local pid.
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
  --publish-wait-s "${GPC_PUBLISH_WAIT_S}" \
  2>&1 | tee "${RUN_DIR}/logs/smoke.log"

echo "GPC_CI_OK"

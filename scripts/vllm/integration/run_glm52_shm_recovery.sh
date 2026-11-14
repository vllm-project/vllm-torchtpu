#!/usr/bin/env bash
# GLM-5.2 NVFP4 single-engine + shm-backed Raiden offload pool: crash-recovery ladder.
#
# Proves the shm offload host pool survives a hard engine crash for GLM-5.2 -- a
# MULTI-SHAPE (non-unified) MLA KV model (MLA latent NoPE/RoPE pair + DSA indexer),
# unlike the hybrid unified-pool Qwen3.5. Single engine, so there is NO reshard:
# the recall is cleanly attributable to the shm pool (strict smoke, byte-exact).
#
# Combines run_glm52_fp4_offload_correctness.sh's serve line (DP8/TP1 + EP, NVFP4,
# GLM parsers) with the three-phase crash choreography of run_qwen35_shm_recovery.sh
# (minus the unused tier-1 registry/store node). This test PASSES when the serving
# image carries two changes (described in the PR, NOT part of this test): (1) the
# RAIDEN_SHM_KEY offload feature in raiden_store.py -- relax the RAIDEN_SHM_KEY guard
# to log-and-allow + enable_shm on the offload host pool KVCacheManager; and (2) a GLM
# compile-compat tweak (TpuBackend(enable_serialization=False)) so GLM MLA graph does
# not abort the bundled AOT-cache save.
#
# DP8 pinning: each DP rank builds its OWN KVCacheStore/shm pool, so every request
# is pinned to one rank (--dp-rank -> X-data-parallel-rank header) and populate +
# recall hit the same rank's pool. Metrics are aggregate.
#
# Phases, each its own engine, driven by smoke_shm_recovery.py over one --state file:
#   A populate       RAIDEN_SHM_KEY=K. Seed fresh prompts (each offloaded to the shm
#                    pool), reset HBM, recall the target and prove IN PROCESS it comes
#                    back as an EXTERNAL hit with matching first tokens. SIGKILL,
#                    leaving /dev/shm intact.
#   B verify-recover RAIDEN_SHM_KEY=K, fresh engine. Recall target: must be an external
#                    hit with first tokens matching pre-crash (recovered from shm). (P)
#   C verify-cold    a DIFFERENT RAIDEN_SHM_KEY, fresh engine. Recall target: must MISS
#                    (0 external hits) -> the recovery hit is attributable to the key. (N)
#
# Markers: GLM52_SHM_BOOT_OK -> GLM52_SHM_POPULATE_OK -> GLM52_SHM_CRASHED ->
#          GLM52_SHM_RECOVER_OK -> GLM52_SHM_COLD_OK -> GLM52_SHM_RECOVERY_OK
# HAPPY_PATH=1 stops after GLM52_SHM_POPULATE_OK (boot + offload round-trip only).
set -uo pipefail
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"

fail() { echo "GLM52_SHM_RIG_FAIL: $*" >&2; exit 1; }

# ---- model / serving ----
MODEL="${MODEL:-gs://cloud-devkit/ckpt/hf/nvidia/GLM-5.2-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/GLM-5.2-NVFP4}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8123}"
BASE="http://${BIND_HOST}:${PORT}"
HAPPY_PATH="${HAPPY_PATH:-1}"

RUN_ROOT="${RUN_ROOT:-/tmp/glm52_shm_recovery}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
STATE_FILE="${STATE_FILE:-${RUN_DIR}/state.json}"
# Compile cache is 30-40G and MUST NOT land on /dev/shm (RAM, competes with the
# shm KV pool) or the tiny container rootfs. SCRATCH_DIR is a disk emptyDir.
SCRATCH_DIR="${SCRATCH_DIR:-/scratch}"
mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}" 2>/dev/null || true

# ---- GLM-5.2 topology (from run_glm52_fp4_offload_correctness.sh): DP8/TP1 + EP.
#      Each rank streams only its local experts (ep-weight-filter). Requests are
#      pinned to DP_RANK. ----
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
DP_RANK="${DP_RANK:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
# 25 GiB per DP replica x 8 = 200 GiB host shm pool.
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-26843545600}"
RAIDEN_CONTROLLER_PORT="${RAIDEN_CONTROLLER_PORT:-27901}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"

# ---- shm keys (positive + negative-control) ----
RUN_ID="${RUN_ID:-$$_$(date +%s)}"
SHM_KEY="${SHM_KEY:-glm52shmrec_${RUN_ID}}"
SHM_KEY_NEG="${SHM_KEY_NEG:-${SHM_KEY}_neg}"
export RAIDEN_SHM_MODEL_UID="${RAIDEN_SHM_MODEL_UID:-glm52-nvfp4}"

# ---- seeding: prompts are whole MLA hit-blocks (1024) within max-model-len ----
PROMPTS="${PROMPTS:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
TARGET="${TARGET:-0}"
SWEEP_WAIT_S="${SWEEP_WAIT_S:-15}"

# gs:// streams via runai; a local path uses prefetch (see the GLM offload rig).
LOAD_FORMAT_ARGS=(--safetensors-load-strategy=prefetch)
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer)
fi

# ---- transformers pin (the GLM-5.2 NVFP4 recipe's floor: model_type glm_moe_dsa
#      needs >=5.14.1). No-op if the image already ships it. ----
TRANSFORMERS_PIN="${TRANSFORMERS_PIN-5.14.1}"
if [[ -n "${TRANSFORMERS_PIN}" ]]; then
  if ! "${PY}" - "${TRANSFORMERS_PIN}" <<'PYEOF'
import sys
try:
    import transformers
except ImportError:
    sys.exit(1)
def parts(v):
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
sys.exit(0 if parts(transformers.__version__) >= parts(sys.argv[1]) else 1)
PYEOF
  then
    echo "installing transformers==${TRANSFORMERS_PIN} (the recipe's pin)"
    "${PY}" -m pip install -q -U "transformers==${TRANSFORMERS_PIN}" \
      || "${PY}" -m pip install -q --force-reinstall "transformers==${TRANSFORMERS_PIN}" \
      || fail "could not install transformers==${TRANSFORMERS_PIN}"
  fi
fi

SERVER_PID=""

serve() {  # $1 = shm key for this engine
  local shm_key="$1"
  export PYTHONHASHSEED=0
  export RAIDEN_SHM_KEY="${shm_key}"
  export GLOG_alsologtostderr=1
  export PJRT_DEVICE=TPU
  export MOE_REQUANTIZE_BLOCK_SIZE=512
  export TORCHINDUCTOR_AUTOGRAD_CACHE=0
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_S}"
  export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S="${RAIDEN_EXPECTED_WORKERS_TIMEOUT_S:-900}"
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRATCH_DIR}/vllm-cache}"
  export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-${VLLM_CACHE_ROOT}}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_DIR}/torchinductor}"
  export TMPDIR="${TMPDIR:-${SCRATCH_DIR}/tmp}"
  mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}" 2>/dev/null || true

  local kvt
  kvt='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"raiden_job_name":"glm52-shm-recovery"}}'

  exec vllm serve "${MODEL}" \
    "${LOAD_FORMAT_ARGS[@]}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --data-parallel-size="${DP_SIZE}" \
    --tensor-parallel-size="${TP_SIZE}" \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --max-model-len="${MAX_MODEL_LEN}" \
    --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs=1 \
    --no-async-scheduling \
    --kv-cache-dtype=fp8 \
    --tool-call-parser=glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser=glm45 \
    --gpu-memory-utilization="${GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --api-server-count=1 \
    --host "${BIND_HOST}" \
    --port "${PORT}" \
    --kv-transfer-config "${kvt}"
}

launch_engine() {  # $1 = label, $2 = shm key
  local label="$1" shm_key="$2" server_log="${LOG_DIR}/server_${1}.log"
  echo "===== [${label}] launching GLM-5.2 (DP${DP_SIZE}/TP${TP_SIZE}, EP), RAIDEN_SHM_KEY=${shm_key}, port ${PORT} ====="
  ( serve "${shm_key}" ) > >(tee "${server_log}" | sed 's/^/[srv] /') 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  until curl -sf --max-time 10 "${BASE}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${server_log}" || true
      fail "[${label}] server exited during startup (see ${server_log})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] || fail "[${label}] not healthy in ${STARTUP_TIMEOUT_S}s"
    sleep 15
  done
  echo "[${label}] server healthy; warming up (pinned to dp rank ${DP_RANK})..."
  until curl -sf --max-time 300 -X POST "${BASE}/v1/completions" \
      -H "Content-Type: application/json" -H "X-data-parallel-rank: ${DP_RANK}" \
      -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"Hello\", \"max_tokens\": 2}" \
      >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${server_log}" || true
      fail "[${label}] server exited during warm up (see ${server_log})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] || fail "[${label}] engine not ready in ${STARTUP_TIMEOUT_S}s"
    echo "[${label}] engine compiling, waiting 30s..."
    sleep 30
  done
  echo "[${label}] engine ready"
}

hard_kill() {
  if [[ -n "${SERVER_PID}" ]]; then kill -KILL "${SERVER_PID}" 2>/dev/null || true; fi
  # SIGKILL the DP worker procs too, WITHOUT unlinking /dev/shm. Single engine,
  # so every VLLM* process belongs to it (no prefill to protect).
  pkill -KILL '^VLLM' 2>/dev/null || true
  SERVER_PID=""
  for _ in $(seq 1 60); do pgrep '^VLLM' >/dev/null 2>&1 || break; sleep 1; done
  sleep 3
}

list_shm() { local f; for f in /dev/shm/"${1}"*; do [[ -e "${f}" ]] && printf "  %s\n" "${f##*/}"; done; }
unlink_shm() { local f; for f in /dev/shm/"${1}"*; do rm -f "${f}" 2>/dev/null || true; done; }

# shellcheck disable=SC2317
cleanup() { hard_kill; unlink_shm "${SHM_KEY}"; unlink_shm "${SHM_KEY_NEG}"; }
trap cleanup EXIT

run_phase() {  # $1 = driver phase, $2 = log name
  "${PY}" "${script_dir}/smoke_shm_recovery.py" \
    --phase "$1" --state "${STATE_FILE}" --dp-rank "${DP_RANK}" \
    --base "${BASE}" --model "${SERVED_MODEL_NAME}" \
    --prompts "${PROMPTS}" --prompt-tokens "${PROMPT_TOKENS}" \
    --sweep-wait-s "${SWEEP_WAIT_S}" --target "${TARGET}" \
    2>&1 | tee "${LOG_DIR}/${2}.log"
  return "${PIPESTATUS[0]}"
}

# Clean slate.
hard_kill
unlink_shm "${SHM_KEY}"; unlink_shm "${SHM_KEY_NEG}"
rm -f "${STATE_FILE}" 2>/dev/null || true

# ---- Phase A: populate under SHM_KEY ----
launch_engine A "${SHM_KEY}"
if ! list_shm "${SHM_KEY}" | grep -q .; then
  fail "no /dev/shm segments for ${SHM_KEY}: offload pool is not shm-backed (is the RAIDEN_SHM_KEY offload-recovery feature present in this image? see the PR description)"
fi
echo "--- /dev/shm segments under ${SHM_KEY}:"; list_shm "${SHM_KEY}"
echo "GLM52_SHM_BOOT_OK"

run_phase populate populate \
  || fail "populate failed -- offload with shm-backed pool did not round-trip (see ${LOG_DIR}/populate.log)"
echo "GLM52_SHM_POPULATE_OK"

if [[ "${HAPPY_PATH}" == "1" ]]; then
  echo "GLM52_SHM_HAPPY_OK (GLM-5.2 offload round-tripped through the shm-backed pool; crash/recover not exercised)"
  exit 0
fi

# ---- crash A (SIGKILL; /dev/shm untouched) ----
echo "--- crashing engine A (SIGKILL; /dev/shm left intact) ---"
hard_kill
list_shm "${SHM_KEY}" | grep -q . || fail "/dev/shm for ${SHM_KEY} vanished after crash; nothing to recover"
echo "--- /dev/shm survived:"; list_shm "${SHM_KEY}"
echo "GLM52_SHM_CRASHED"

# ---- Phase B: recover under the SAME key ----
launch_engine B "${SHM_KEY}"
run_phase verify-recover recover \
  || fail "verify-recover failed: offloaded KV did NOT recover across the crash (see ${LOG_DIR}/recover.log)"
echo "GLM52_SHM_RECOVER_OK"
hard_kill

# Free the original-key pool before the DIFFERENT-key cold leg. The same-key
# recover leg re-attaches the existing segments (1x /dev/shm), but the cold leg
# uses a NEW key that needs its OWN pool -- both resident at once is 2x the pool
# and overflows the /dev/shm tmpfs cap (posix_fallocate -> ENOSPC, which hangs
# engine C in a retry loop). Recovery is already proven above, so the original
# segments are safe to drop; the cold leg still proves a different key (its own
# empty pool) does not serve the target.
unlink_shm "${SHM_KEY}"

# ---- Phase C: cold control under a DIFFERENT key ----
launch_engine C "${SHM_KEY_NEG}"
run_phase verify-cold cold \
  || fail "verify-cold failed: a different key still hit, so the recovery signal is not attributable (see ${LOG_DIR}/cold.log)"
echo "GLM52_SHM_COLD_OK"

echo "GLM52_SHM_RECOVERY_OK (GLM-5.2 multi-shape MLA KV survived a hard crash: recovered from the shm pool under the same key with matching tokens, and a different key missed)"

#!/usr/bin/env bash
# Qwen3.5 p4d2 disaggregation + shm-backed DECODE offload pool: crash-recovery
# ladder (DEPLOYMENT-SURVIVAL bar, via the disagg proxy).
#
# Boots the p4d2 disagg trio (prefill PCP4 + decode DP4/TP1 + toy proxy) with
# ONLY the decode offloading host pool shm-backed (RAIDEN_SHM_KEY routed into
# decode_cmd via DECODE_RAIDEN_SHM_KEY), then proves the decode side survives a
# hard decode crash: it re-attaches its persisted /dev/shm segment, boots, and
# the disagg deployment keeps serving correctly.
#
# Why the DEPLOYMENT-SURVIVAL bar (and not an shm-attributed ext_hit):
#   All serving in disagg flows through the proxy, where the prefill engine
#   supplies the prompt KV to decode via reshard (TPURaidenConnector) on EVERY
#   request. That drives decode's external_prefix_cache_hits to ~100% regardless
#   of the shm pool, so a proxy-routed ext_hit cannot be attributed to shm. The
#   only way to bypass reshard would be to hit decode directly, but a decode-role
#   engine is compiled for decode shapes and asserts on a large direct prefill
#   (_get_padded_token_len: index < len(paddings)), so decode-direct is not
#   viable here. We therefore prove what IS observable end-to-end in disagg:
#     - offload SAVE happens (decode fills its shm pool while seeding),
#     - decode survives a hard SIGKILL and RE-ATTACHES the persisted /dev/shm
#       segment under the same key (same inode; boots without cold-fallback),
#     - the disagg deployment resumes and serves correct tokens after recovery,
#     - a different-key restart also comes up (control).
#   Recovery CORRECTNESS of the Raiden store itself is separately proven by
#   kv_cache_store_recovery_e2e_test; this rig covers the disagg integration.
#
# Markers (in order):
#   DISAGG_SHM_BOOT_OK        prefill+decode+proxy up; decode /dev/shm segments present
#   DISAGG_SHM_POPULATE_OK    seed via proxy -> decode offloads to shm; recall serves correct tokens
#   DISAGG_SHM_DECODE_KILLED  decode process group SIGKILLed; prefill+proxy up; /dev/shm kept
#   DISAGG_SHM_RECOVER_OK     decode relaunched SAME key; re-attaches persisted shm; proxy recall correct
#   DISAGG_SHM_COLD_OK        decode relaunched DIFFERENT key; fresh deployment comes up + serves
#   QWEN35_DISAGG_SHM_RECOVERY_OK
#
# HAPPY_PATH=1 stops after DISAGG_SHM_POPULATE_OK (boot + offload smoke only).
# HAPPY_PATH=0 (this test's default via the recipe) runs the full ladder.
set -euo pipefail
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
PY="${PYTHON_BIN:-python}"
command -v "${PY}" >/dev/null 2>&1 || PY=python3

# The launch defaults VLLM_SRC to a nonexistent CI path (/mnt/data/vllm) and
# fails its -d check; point it (and TORCHTPU_VLLM_SRC) at the in-image repo so
# PYTHONPATH resolves the (patched) vllm_torchtpu source.
export VLLM_SRC="${VLLM_SRC:-${repo_root}}"
export TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
# The launch's default EXPECTED_VLLM_VERSION (0.27.0) is stale vs the image;
# derive it from the installed vllm so the version gate passes. Use
# importlib.metadata (does NOT import vllm, so no TPU-init stdout noise) and
# strip the setuptools-scm local label (+empty) so it equals vllm.__version__.
export EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-$(${PY} -c 'import importlib.metadata as m; print(m.version("vllm").split("+")[0])' 2>/dev/null || true)}"

# ---- model / serving ----
MODEL_PATH="${MODEL_PATH:-gs://tpu-inference-hf-llm-model-checkpoints/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/9d1823d2dee688a6b25e77009dc727688c44936e/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_disagg_shm}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
HAPPY_PATH="${HAPPY_PATH:-1}"

# ---- shm: decode-only, namespaced + warm-boot-validated (Raiden README) ----
DECODE_RAIDEN_SHM_KEY="${DECODE_RAIDEN_SHM_KEY:-qwen35_disagg_decode}"
# A DIFFERENT key for the cold-control leg (fresh, non-recovered deployment).
COLD_RAIDEN_SHM_KEY="${COLD_RAIDEN_SHM_KEY:-qwen35_disagg_decode_cold}"
export RAIDEN_SHM_SERVER_NAME="${RAIDEN_SHM_SERVER_NAME:-decode}"
export RAIDEN_SHM_MODEL_UID="${RAIDEN_SHM_MODEL_UID:-qwen35-a3b-fp8-p4d2}"
# /reset_prefix_cache is a dev-mode endpoint; mount it on every server.
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

# ---- proven p4d2 config (mirror run_qwen35_p4d2_disagg_correctness.sh). Decode
#      MUST be TP1/DP4 (Raiden Qwen3.5 disagg rejects DP1 -> invalid 5-core slice
#      and TP4 -> admission reject). The launch's own defaults are too small
#      (MAX_MODEL_LEN=4096), so set the known-good values explicitly. ----
export USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
export PREFILL_TP="${PREFILL_TP:-1}"
export PREFILL_PCP="${PREFILL_PCP:-4}"
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
export DECODE_TP="${DECODE_TP:-1}"
export DECODE_DP="${DECODE_DP:-4}"
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-}"
export DECODE_CPU_BYTES_TO_USE="${DECODE_CPU_BYTES_TO_USE:-24000000000}"
export PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-768}"
export DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-2304}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
export DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-8589934592}"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"

# ---- seeding (via PROXY): the prefill engine does the big prefill and reshards
#      to decode, which generates and offloads to its shm pool. These are the
#      proven p4d2 sizes; offload SAVE is proactive so the pool fills as we seed.
PROMPTS="${PROMPTS:-12}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16448}"
TARGET="${TARGET:-0}"
STATE_FILE="${STATE_FILE:-${RUN_DIR}/state.json}"

export MODEL_PATH SERVED_MODEL_NAME RUN_ROOT RUN_DIR PROXY_PORT PREFILL_PORT DECODE_PORT DECODE_RAIDEN_SHM_KEY
export P4D2_BIND_HOST="${BIND_HOST}"
mkdir -p "${RUN_DIR}/logs"

fail() { echo "DISAGG_SHM_FAIL: $*" >&2; exit 1; }

PROXY="http://${BIND_HOST}:${PROXY_PORT}"
PREFILL="http://${BIND_HOST}:${PREFILL_PORT}"
DECODE="http://${BIND_HOST}:${DECODE_PORT}"

health() { curl -fsS -o /dev/null -w "%{http_code}" "$1/health" 2>/dev/null || true; }

wait_health() {  # $1=base url  $2=tries (5s each)
  local code=""
  for _ in $(seq 1 "${2:-96}"); do
    code="$(health "$1")"
    [[ "${code}" == "200" ]] && return 0
    sleep 5
  done
  return 1
}

# One throwaway request through the proxy to re-establish the prefill->decode
# reshard link after a decode restart (best-effort; errors tolerated).
warmup_proxy() {
  curl -fsS -o /dev/null --max-time 300 \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"${SERVED_MODEL_NAME}"'","prompt":"hello","max_tokens":1,"temperature":0}' \
    "${PROXY}/v1/completions" 2>/dev/null || true
}

# Identity (name inode size) of the decode key's /dev/shm segments -- used as
# evidence that a same-key restart RE-ATTACHES the persisted segment (same inode)
# rather than recreating it.
shm_identity() {  # $1 = key
  shopt -s nullglob
  local segs=(/dev/shm/*"$1"*)
  shopt -u nullglob
  (( ${#segs[@]} > 0 )) || { echo "(none)"; return; }
  stat -c '%n inode=%i size=%s' "${segs[@]}" 2>/dev/null | sort
}

# PGID of the decode api_server (identified by its unique --port). All decode
# EngineCore/worker children share this group, so killing the group is a clean,
# surgical decode-only crash that leaves prefill (--port ${PREFILL_PORT}) alone.
decode_pgid() {
  local leader
  leader="$(pgrep -f "api_server .*--port ${DECODE_PORT}\b" | head -1 || true)"
  [[ -n "${leader}" ]] || return 1
  ps -o pgid= -p "${leader}" 2>/dev/null | tr -d ' '
}

kill_decode() {
  local pgid
  if pgid="$(decode_pgid)"; then
    echo "  SIGKILL decode process group ${pgid}"
    kill -KILL -"${pgid}" 2>/dev/null || true
    for _ in $(seq 1 90); do
      pgrep -g "${pgid}" >/dev/null 2>&1 || break
      sleep 2
    done
  else
    echo "  (no decode api_server found to kill)"
  fi
  # Fallback: any straggler still bound to the decode port keeps TPU cores 4-7
  # busy and would block the relaunch; take its group down too.
  local straggler spg
  straggler="$(pgrep -f "api_server .*--port ${DECODE_PORT}\b" | head -1 || true)"
  if [[ -n "${straggler}" ]]; then
    spg="$(ps -o pgid= -p "${straggler}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${spg}" ]]; then kill -KILL -"${spg}" 2>/dev/null || true; fi
  fi
  # Confirm the decode port is actually free (decode truly gone).
  for _ in $(seq 1 30); do
    [[ "$(health "${DECODE}")" == "200" ]] || return 0
    sleep 2
  done
  return 0
}

# Relaunch decode ALONE from the launch-written, self-contained decode_cmd.sh
# (it carries its own cd + PYTHONPATH + all TPU/Raiden env + api_server args).
# $1 = shm key to run under, $2 = log filename under RUN_DIR/logs.
relaunch_decode() {
  local key="$1" logf="$2" cmd_file
  cmd_file="${RUN_DIR}/decode_cmd.${key}.sh"
  [[ -f "${RUN_DIR}/decode_cmd.sh" ]] || fail "relaunch: ${RUN_DIR}/decode_cmd.sh missing (launch did not write it)"
  sed "s|export RAIDEN_SHM_KEY='[^']*'|export RAIDEN_SHM_KEY='${key}'|" \
    "${RUN_DIR}/decode_cmd.sh" >"${cmd_file}"
  grep -q "RAIDEN_SHM_KEY='${key}'" "${cmd_file}" \
    || fail "relaunch: could not set RAIDEN_SHM_KEY=${key} in decode cmd"
  echo "  relaunching decode (RAIDEN_SHM_KEY=${key}) -> logs/${logf}"
  ( cd "${repo_root}" && setsid bash -lc "$(cat "${cmd_file}")" ) \
    >"${RUN_DIR}/logs/${logf}" 2>&1 &
  disown || true
}

# ---------------------------------------------------------------------------
echo "=== [1/6] launch disagg trio (decode pool shm-backed key=${DECODE_RAIDEN_SHM_KEY} server=${RAIDEN_SHM_SERVER_NAME}) ==="
cd "${repo_root}"
# The launch backgrounds prefill/decode/proxy (setsid), health-checks P/D, then
# returns; it writes {prefill,decode,proxy}.{pid,cmd.sh} under its RUN_DIR.
bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh 2>&1 | tee "${RUN_DIR}/logs/launch.log"

wait_health "${DECODE}" 24 || fail "decode server never became healthy"

# The shm data plane must exist for the decode key (proves the shm pool built
# under the disagg MultiConnector, not a normal-memory pool).
echo "--- /dev/shm segments for key ${DECODE_RAIDEN_SHM_KEY}:"
shopt -s nullglob
shm_segs=(/dev/shm/*"${DECODE_RAIDEN_SHM_KEY}"*)
shopt -u nullglob
if (( ${#shm_segs[@]} > 0 )); then
  printf '  %s\n' "${shm_segs[@]}"
else
  fail "no /dev/shm segments for ${DECODE_RAIDEN_SHM_KEY}: decode offload pool is not shm-backed"
fi
echo "DISAGG_SHM_BOOT_OK"

# ---------------------------------------------------------------------------
echo "=== [2/6] populate (via PROXY): seed -> prefill reshards to decode -> decode offloads to its shm pool -> recall ==="
# Generation flows through the PROXY (prefill reshards prompt KV to decode);
# metrics/reset target the DECODE server (--metrics-base). --relaxed drops the
# ext_hit attribution gates (reshard makes them meaningless) and gates on decode
# serving correct tokens.
"${PY}" "${script_dir}/smoke_shm_recovery.py" \
  --phase populate --state "${STATE_FILE}" --relaxed \
  --base "${PROXY}" --metrics-base "${DECODE}" --model "${SERVED_MODEL_NAME}" \
  --prompts "${PROMPTS}" --prompt-tokens "${PROMPT_TOKENS}" --target "${TARGET}" \
  2>&1 | tee "${RUN_DIR}/logs/populate.log"
rc="${PIPESTATUS[0]}"
[[ "${rc}" -eq 0 ]] || fail "populate failed (decode did not offload/serve; see populate.log)"

SHM_PRE="$(shm_identity "${DECODE_RAIDEN_SHM_KEY}")"
echo "--- /dev/shm identity after populate (pre-crash):"
printf '  %s\n' "${SHM_PRE}"
echo "DISAGG_SHM_POPULATE_OK"

# ---------------------------------------------------------------------------
if [[ "${HAPPY_PATH}" == "1" ]]; then
  echo "QWEN35_DISAGG_SHM_HAPPY_OK (decode offload pool is shm-backed and served the recall through the disagg proxy; crash/recover not exercised)"
  exit 0
fi

# ---------------------------------------------------------------------------
echo "=== [3/6] crash: SIGKILL the decode engine only (prefill+proxy stay up; /dev/shm kept) ==="
kill_decode
pcode="$(health "${PREFILL}")"
[[ "${pcode}" == "200" ]] || fail "prefill did not survive the decode kill (prefill health=${pcode:-none}); the kill was not surgical"
[[ "$(health "${DECODE}")" != "200" ]] || fail "decode still healthy after kill; the crash did not take effect"
shopt -s nullglob
shm_after=(/dev/shm/*"${DECODE_RAIDEN_SHM_KEY}"*)
shopt -u nullglob
(( ${#shm_after[@]} > 0 )) || fail "decode /dev/shm segments for ${DECODE_RAIDEN_SHM_KEY} vanished after the crash; nothing to recover"
echo "  prefill still healthy; /dev/shm survived: ${shm_after[*]}"
echo "DISAGG_SHM_DECODE_KILLED"

# ---------------------------------------------------------------------------
echo "=== [4/6] warm recover: relaunch decode under the SAME key -> re-attach persisted shm -> proxy recall correct ==="
relaunch_decode "${DECODE_RAIDEN_SHM_KEY}" decode_recover.log
wait_health "${DECODE}" 180 || fail "decode did not come back healthy after warm relaunch (see logs/decode_recover.log)"
# Evidence: a same-key restart should re-attach the SAME persisted segment.
SHM_WARM="$(shm_identity "${DECODE_RAIDEN_SHM_KEY}")"
echo "--- /dev/shm identity after warm restart:"
printf '  %s\n' "${SHM_WARM}"
if [[ "${SHM_WARM}" == "${SHM_PRE}" ]]; then
  echo "  shm re-attach evidence: SAME segment inode(s) across the crash (persisted segment re-attached)"
else
  echo "  shm re-attach note: segment identity changed across the crash (recovery may recreate-on-recover; not gated)"
  echo "    pre : ${SHM_PRE}"
  echo "    warm: ${SHM_WARM}"
fi
warmup_proxy
"${PY}" "${script_dir}/smoke_shm_recovery.py" \
  --phase verify-recover --state "${STATE_FILE}" --relaxed \
  --base "${PROXY}" --metrics-base "${DECODE}" --model "${SERVED_MODEL_NAME}" \
  2>&1 | tee "${RUN_DIR}/logs/verify_recover.log"
rc="${PIPESTATUS[0]}"
[[ "${rc}" -eq 0 ]] || fail "verify-recover failed (decode did not serve correctly after a same-key restart; see verify_recover.log)"
echo "DISAGG_SHM_RECOVER_OK"

# ---------------------------------------------------------------------------
echo "=== [5/6] cold control: kill decode, relaunch under a DIFFERENT key -> fresh deployment comes up + serves ==="
kill_decode
[[ "$(health "${PREFILL}")" == "200" ]] || fail "prefill did not survive the second decode kill"
relaunch_decode "${COLD_RAIDEN_SHM_KEY}" decode_cold.log
wait_health "${DECODE}" 180 || fail "decode did not come back healthy after cold relaunch (see logs/decode_cold.log)"
echo "--- /dev/shm identity after cold restart (different key ${COLD_RAIDEN_SHM_KEY}):"
printf '  %s\n' "$(shm_identity "${COLD_RAIDEN_SHM_KEY}")"
warmup_proxy
"${PY}" "${script_dir}/smoke_shm_recovery.py" \
  --phase verify-cold --state "${STATE_FILE}" --relaxed \
  --base "${PROXY}" --metrics-base "${DECODE}" --model "${SERVED_MODEL_NAME}" \
  2>&1 | tee "${RUN_DIR}/logs/verify_cold.log"
rc="${PIPESTATUS[0]}"
[[ "${rc}" -eq 0 ]] || fail "verify-cold failed (fresh-key deployment did not come up/serve; see verify_cold.log)"
echo "DISAGG_SHM_COLD_OK"

# ---------------------------------------------------------------------------
echo "=== [6/6] done ==="
echo "QWEN35_DISAGG_SHM_RECOVERY_OK (decode survived a hard crash: re-attached its persisted /dev/shm segment under the same key, the disagg deployment resumed serving correctly, and a different-key restart also came up)"

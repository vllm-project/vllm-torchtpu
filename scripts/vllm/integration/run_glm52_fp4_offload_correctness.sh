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
# GLM-5.2 NVFP4 KV offload correctness rig: does a block stored to the Raiden
# host pool reload as the same bytes?
#
# The sibling of run_dsv4_offload_correctness.sh, same smoke, same three-arm
# gate; the model is the difference. GLM-5.2 is the other DSA/MLA MoE in this
# tree -- (NoPE, RoPE) latent pair plus a DSA indexer cache, i.e. the
# multi-shape KV family the offload store already carries DSv4 through -- so
# the value of running it is that it covers that path against a checkpoint
# whose experts are natively FP4 rather than requantized from FP8 at load.
#
# The rig exists because a plain cold/warm repeat never reaches the offload
# store -- the HBM prefix cache answers first. VLLM_SERVER_DEV_MODE mounts
# POST /reset_prefix_cache, which empties the HBM cache while leaving the
# Raiden store intact (the connector's reset_cache() reports failure and
# never reaches the store, see raiden_connector.py). That turns the repeat
# into a real host->device load with no eviction-by-volume games.
#
# The rig always runs the full model: a truncated model has non-finite logits,
# so every generation is a degenerate constant and cold and store agree
# whatever the store returns.
#
# Self-contained, the shape every other rig in this directory has: launch ->
# /health -> warm up -> smoke -> verdict, with the engine torn down on exit.
# This is the CI entry point. Trailing arguments are forwarded to the smoke:
#
#   ./run_glm52_fp4_offload_correctness.sh --settle-s 10
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
    SCRATCH_ROOT="${TMPDIR:-/tmp}/glm52-fp4-offload"
  fi
fi
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRATCH_ROOT}/vllm-cache}"
# Paired with VLLM_CACHE_ROOT the way the GLM-5.2 NVFP4 recipe pairs them, so
# a warm restart skips XLA compilation.
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-${VLLM_CACHE_ROOT}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/torchinductor}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${VLLM_CACHE_ROOT}" || exit 1

# Everything down to the next blank line is the GLM-5.2 NVFP4 recipe's
# environment (cloud-devkit-recipes: recipes/experimental/torchtpu-vllm/
# disagg/glm52-nvfp4-disagg-raiden-dp8.yml), minus what only its JobSet needs;
# what follows that is rig plumbing carried over from the DSv4 sibling and
# holds for any model.
#
# Block-512 requantization of the checkpoint's block-16 fp4 MoE weights
# (nvfp4.py): ~32x less scale data and no in-kernel dequant. The recipe sets
# this and nothing else about quantization -- in particular NOT
# MOE_REQUANTIZE_WEIGHT_DTYPE, which the DSv4 rig sets: that knob is read by
# the fp8 path only, where it is what turns an FP8 checkpoint into fp4
# weights. This checkpoint is already NVFP4 (hf_quant_config.json:
# quant_algo=NVFP4, group_size=16), so vLLM detects it as modelopt_fp4 and
# there is nothing to requantize the dtype of -- only the block to widen.
export MOE_REQUANTIZE_BLOCK_SIZE=512
export PJRT_DEVICE=TPU
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
# First-request stragglers must not kill the engine.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# Not carried over from the recipe, deliberately:
#   NEW_MODEL_DESIGN / MODEL_IMPL_TYPE / VLLM_PLUGINS -- no GLM-5.2 config in
#     cloud-devkit-recipes sets them (the FP8 v7x-16 recipe has MODEL_IMPL_TYPE
#     commented out), and the sibling qwen35 rigs run this image without them.
#   TPU_ROPE_CACHE_ROW_MAJOR / TPU_MOE_HASH_TABLE_ROW_MAJOR -- DSv4 knobs; no
#     GLM-5.2 recipe sets them.
#   TPU_TOPOLOGY / TPU_HOST_BOUNDS / MEGASCALE_* / TPU_PROCESS_* /
#     RAY_LOG_TO_STDERR -- the recipe blanks inherited multihost variables for
#     a 2x2x1 JobSet pod. This rig is one process on one v7x-8 host, and the
#     recipe's 2x2x1 topology would be the wrong shape to assert here.
#   TPU_KV_SHM_POOL_GB / TPU_KV_PIN_SHM / TPU_PREMAPPED_BUFFER_* -- tuning for
#     the disagg connector's shm staging pool, a different pool from the
#     offload store's cpu_bytes_to_use below.

# Rig plumbing, from run_dsv4_offload_correctness.sh: the compile and timeout
# settings the offload connector has been exercised under, a fixed hash seed,
# and the dev-mode endpoint the smoke needs. None of it is model-specific.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0 VLLM_NO_USAGE_STATS=1
export TORCH_DIST_TIMEOUT=1200 VLLM_RPC_TIMEOUT=1200000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1200 VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
# 5400s is the recipe's value, and matches this checkpoint: at ~465 GB the
# weight load alone outlasts the DSv4 rig's hour.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-5400}"
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
MODEL="${MODEL:-nvidia/GLM-5.2-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/GLM-5.2-NVFP4}"
# How the 47 shards reach the engine, keyed off the MODEL scheme the way
# run_qwen35_gpc_correctness.sh does it.
#
# gs:// streams through runai rather than landing on a disk first. Staging this
# checkpoint is what the pre-merge step cannot afford: at ~433 GiB it exhausted
# a tpu7x-8 agent's disk 123 files in, and the rig never started. runai is
# already a dependency for exactly this (pyproject.toml: "Stream gs://
# checkpoints via --load-format runai_streamer").
#
# A local path keeps the recipes' prefetch strategy. The two are alternatives,
# not additions: safetensors_load_strategy is read only by vLLM's default
# loader, and RunaiModelStreamerLoader never consults it, so passing prefetch
# alongside runai would be silently inert.
LOAD_FORMAT_ARGS=(--safetensors-load-strategy=prefetch)
if [[ "${MODEL}" == gs://* ]]; then
  LOAD_FORMAT_ARGS=(--load-format runai_streamer)
fi
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
# claims 27901..27908. Deliberately clear of the DSv4 rig's 27801 block, so a
# run started while that one's sockets are still in TIME_WAIT does not bind
# into its range.
RAIDEN_CONTROLLER_PORT="${RAIDEN_CONTROLLER_PORT:-27901}"
# 9216 is the GLM-5.2 FP8 recipe's length. The NVFP4 recipes serve at 1024,
# which is one whole prefix-cache block: the smoke would drop every rung of
# its ladder and exit with nothing measured.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/glm52_fp4_offload_ci}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
# ~465 GB of weights, then a cold-cache compile; leave headroom.
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"
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

KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES_TO_USE}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"raiden_job_name":"glm52-fp4-offload-probe"}}'

# The serve line is the GLM-5.2 recipes' own, not the DSv4 rig's. From
# cloud-devkit-recipes, where both recipes for this model agree flag for flag:
# glm52-nvfp4-disagg-raiden-dp8.yml (the NVFP4 checkpoint, DP8 + EP) and
# vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml (the FP8
# sibling, which contributes --gpu-memory-utilization and max-model-len 9216).
#
#   no --quantization       hf_quant_config.json self-describes as NVFP4 and
#                           vLLM resolves it to modelopt_fp4. Neither recipe
#                           passes one; DSv4's deepseek_v4_fp8 is its own case.
#   no --trust-remote-code  glm_moe_dsa is an in-tree architecture, and no GLM
#                           recipe asks for remote code.
#   --kv-cache-dtype=fp8    both recipes; the checkpoint's own
#                           kv_cache_quant_algo. (The DSv4 rig says fp8_e4m3.)
#   parser flags            carried over as the recipes set them. They only
#                           shape how the chat response is split; the smoke
#                           gates on token_ids, which is upstream of that.
#   --enable-ep-weight-filter
#                           the FP8 v7x-16 recipe passes it and this rig had
#                           been missing it. It matters more here than there:
#                           model_loader_patches.py hooks
#                           RunaiModelStreamerLoader so that with the filter on
#                           each rank streams only its local experts, and
#                           without it all 8 ranks pulled the whole ~465 GB
#                           from GCS and discarded what EP had not assigned
#                           them.
#   weight loading         both recipes pass
#                           --safetensors-load-strategy=prefetch, which on 47
#                           shards is the difference between a load that
#                           streams and one that stalls per shard. Kept for a
#                           local MODEL only; a gs:// MODEL streams through
#                           runai instead. See LOAD_FORMAT_ARGS above.
#
# What the rig overrides, and why it has to:
#   --enable-prefix-caching  the recipes serve with --no-enable-prefix-caching.
#                            Inverted here on purpose: an offload store with no
#                            prefix cache in front of it has nothing to offload,
#                            and all three arms of the smoke would be cold.
#   --enable-prompt-tokens-details
#                            usage.prompt_tokens_details is what every
#                            quantity gate in the smoke is measured against.
#   --max-num-seqs=1 / --no-async-scheduling
#                            identical single-request batch geometry on every
#                            pass, so a cold/warm difference is the cache and
#                            nothing else. The recipes are tuned for
#                            throughput (64 seqs, async) and would not be.
#   --kv-transfer-config     the offload connector, in place of the recipes'
#                            disagg TPURaidenConnector.
#   --api-server-count=1     vLLM defaults this to the DP size, and 8 API
#                            server processes each build their own VllmConfig
#                            concurrently. On a gs:// MODEL that is a data
#                            race: ObjectStorageModel names its mirror dir
#                            from a hash of the URL alone, so all 8 share
#                            /root/.cache/vllm/assets/model_streamer/<hash>,
#                            and runai's pull_files downloads straight to the
#                            final path with no lock and no atomic rename.
#                            One process truncates hf_quant_config.json while
#                            another is in get_quant_config's json.load of it
#                            (weight_utils.py), which surfaces as a VllmConfig
#                            pydantic error, "Expecting value: line 1 column 1
#                            (char 0)", and a dead API server. One frontend
#                            costs this rig nothing: --max-num-seqs=1, one
#                            request at a time, all pinned to dp rank 0.

serve() {
  exec "${VLLM_BIN}" serve "${MODEL}" \
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
    "${LOAD_FORMAT_ARGS[@]}" \
    --gpu-memory-utilization="${GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --api-server-count=1 \
    --port "${PORT}" \
    --kv-transfer-config "$KVT"
}

mkdir -p "${LOG_DIR}" || exit 1
SERVER_LOG="${LOG_DIR}/server.log"

# Print to stdout: Buildkite's main log view surfaces stdout, and the failure
# reason must be readable there without digging through artifacts.
fail() {
  echo "GLM52_FP4_OFFLOAD_RIG_FAIL: $*"
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
  # The workers are retitled, so they survive the parent and keep /dev/vfio
  # open -- the next launch then dies with the device busy. Match on the comm
  # name: a -f pattern would also match this script's own command line and
  # kill the caller mid-cleanup. This rig owns all eight chips, so there is no
  # other engine on the host for it to catch.
  pkill -TERM '^VLLM' 2>/dev/null || true
  sleep 3
  pkill -KILL '^VLLM' 2>/dev/null || true
}
trap cleanup EXIT

# Both GLM-5.2 NVFP4 recipes pin transformers 5.14.1 before serving -- the
# checkpoint's config (model_type glm_moe_dsa, transformers_version 5.11.0)
# needs a recent one. Unlike a recipe pod this runs inside the CI image under
# test, so the pin is a floor rather than the recipe's unconditional
# reinstall: nothing happens when the image already ships 5.14.1 or newer, and
# TRANSFORMERS_PIN= skips the check entirely.
TRANSFORMERS_PIN="${TRANSFORMERS_PIN-5.14.1}"
if [[ -n "${TRANSFORMERS_PIN}" ]]; then
  if ! "${PYTHON_BIN}" - "${TRANSFORMERS_PIN}" <<'PYEOF'
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
    "${PYTHON_BIN}" -m pip install -q -U "transformers==${TRANSFORMERS_PIN}" \
      || "${PYTHON_BIN}" -m pip install -q --force-reinstall \
           "transformers==${TRANSFORMERS_PIN}" \
      || fail "could not install transformers==${TRANSFORMERS_PIN}"
    "${PYTHON_BIN}" -c "import transformers" \
      || fail "the transformers install is broken after pinning \
${TRANSFORMERS_PIN}"
  fi
fi

echo "===== launching GLM-5.2 NVFP4 (DP${DP_SIZE}/TP${TP_SIZE}), requests \
pinned to dp rank ${DP_RANK}, on port ${PORT} ====="
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
# gate. Two tokens stays far below one offload block, so this warm up publishes
# nothing the smoke could later mistake for its own cold prompt.
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

# Every gate in the smoke is measured in whole prefix-cache blocks, and its
# default (1024) is DSv4's page size, which the platform picks on a DSv4-only
# branch of update_block_size_for_backend. GLM-5.2 takes the generic branch and
# may land elsewhere, so read the size the engine actually chose out of its own
# startup line rather than assuming the sibling model's. A run whose granularity
# is wrong does not fail loudly -- it reports hit sizes against the wrong
# multiple -- so falling back to 1024 is only for the case where the line is
# absent, and it says so.
HIT_BLOCK_SIZE="${HIT_BLOCK_SIZE:-}"
if [[ -z "${HIT_BLOCK_SIZE}" ]]; then
  HIT_BLOCK_SIZE="$(grep -oE "Using KV cache block size: [0-9]+" "${SERVER_LOG}" \
    | tail -1 | grep -oE "[0-9]+$")"
  if [[ -n "${HIT_BLOCK_SIZE}" ]]; then
    echo "hit block size from the engine: ${HIT_BLOCK_SIZE}"
  else
    HIT_BLOCK_SIZE=1024
    echo "the engine never logged its KV cache block size; assuming \
${HIT_BLOCK_SIZE}. If the smoke reports hits against a different multiple, \
pass HIT_BLOCK_SIZE=<n>"
  fi
fi

SMOKE_ARGS=(--base "${BASE}" --model "${SERVED_MODEL_NAME}"
            --dp-rank "${DP_RANK}" --hit-block-size "${HIT_BLOCK_SIZE}")
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
# own activity never reaches the window.
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
echo "GLM52_FP4_OFFLOAD_RIG_OK"

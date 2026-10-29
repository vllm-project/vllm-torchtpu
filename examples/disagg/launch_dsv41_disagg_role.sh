#!/usr/bin/env bash
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
# DeepSeek-V4.1-Flash P/D disaggregation over Raiden Stage 3: ONE role
# (prefill or decode) on one tpu7x 2x2x1 host, DP=8 TP=1 EP=8 (P/D G4).
#
# Phase 1 of dsv41/PD_DESIGN.md: both peers run the same tree, so their KV
# layouts are identical by construction (G3: same vllm patches, same
# allocator, same aux partitions).
#   prefill: VLLM_DSV41_CED=1 (CED_PREFILL=1), PURE_20L_PREFILL=0, kv_producer
#   decode : VLLM_DSV41_CED=0, kv_consumer, no DSpark
#
# The model env block is the DSv4.1 decode block from
# dsv41/scripts/decode/run_dp8_scale_driver.py (handoff paste 6755399435850903)
# minus DSpark and the DecodeBenchConnector; the Raiden block is the one
# launch_dsv4_disagg_multihost.sh serves V4.0 under. Prefix caching and async
# scheduling stay off: DSv4's multi-group Stage-3 plan has no prefix-aware load.
#
# Required env:
#   ROLE              prefill | decode
#   ADVERTISE_HOST    this host's routable IP (the peer connects to it)
#   RUN_ID            shared by both roles (namespaces, engine ids)
# Optional env (defaults in brackets):
#   MODEL_PATH [/dev/shm/DeepSeek-V4.1-Flash]  PORT [8400 | 9400]
#   CTRL_PORT [27000 | 28000]  KV_TRANSFER_PORT [9100 | 9200]
#   SB_PORT_BASE [29500 | 29510]  DP_MASTER_PORT [29530 | 29520]
#   KV_PORT [14579]  SIDE_CHANNEL_PORT [9600]
#   CED_PREFILL [1]  MAX_NUM_BATCHED_TOKENS [2048 | 384]
#   COMPILE_SIZES [[2048] | [16, 64, 384]]  MAX_NUM_SEQS [64]
#   GPU_MEMORY_UTILIZATION [0.85]  MAX_MODEL_LEN [131072]
#   TPU_KV_SHM_POOL_GB [16]  CACHE_ROOT [/dev/shm/pd_cache_${ROLE}]
#   API_SERVER_COUNT [8]  DRY_RUN [0] (print the command and exit)

set -euo pipefail

ROLE="${ROLE:?ROLE must be prefill or decode}"
ADVERTISE_HOST="${ADVERTISE_HOST:?ADVERTISE_HOST must be the routable IP of this host}"
RUN_ID="${RUN_ID:?RUN_ID must be shared by both roles}"
MODEL_PATH="${MODEL_PATH:-/dev/shm/DeepSeek-V4.1-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"

case "${ROLE}" in
  prefill)
    PORT="${PORT:-8400}"; CTRL_PORT="${CTRL_PORT:-27000}"
    KV_TRANSFER_PORT="${KV_TRANSFER_PORT:-9100}"
    SB_PORT_BASE="${SB_PORT_BASE:-29500}"; DP_MASTER_PORT="${DP_MASTER_PORT:-29530}"
    KV_ROLE=kv_producer
    CED="${CED_PREFILL:-1}"
    ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
    DSPARK_TOKENS="${DSPARK_TOKENS:-5}"
    ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
    ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
    PREFIX_CACHE_RETENTION_INTERVAL="${PREFIX_CACHE_RETENTION_INTERVAL:-1024}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    if [[ "${MAX_NUM_BATCHED_TOKENS}" == "2048" ]]; then
      COMPILE_SIZES="${COMPILE_SIZES:-[512, 2048]}"
    else
      COMPILE_SIZES="${COMPILE_SIZES:-[512, 2048, ${MAX_NUM_BATCHED_TOKENS}]}"
    fi
    ;;
  decode)
    PORT="${PORT:-9400}"; CTRL_PORT="${CTRL_PORT:-28000}"
    KV_TRANSFER_PORT="${KV_TRANSFER_PORT:-9200}"
    SB_PORT_BASE="${SB_PORT_BASE:-29510}"; DP_MASTER_PORT="${DP_MASTER_PORT:-29520}"
    KV_ROLE=kv_consumer
    CED=0
    ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
    DSPARK_TOKENS="${DSPARK_TOKENS:-5}"
    ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-${ENABLE_DSPARK}}"
    ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
    PREFIX_CACHE_RETENTION_INTERVAL="${PREFIX_CACHE_RETENTION_INTERVAL:-0}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-384}"
    if [[ "${ENABLE_DSPARK}" == "1" ]]; then
      COMPILE_SIZES="${COMPILE_SIZES:-[16, 40, 48, 64, 96, 192, ${MAX_NUM_BATCHED_TOKENS}]}"
    else
      COMPILE_SIZES="${COMPILE_SIZES:-[16, 64, ${MAX_NUM_BATCHED_TOKENS}]}"
    fi
    ;;
  *)
    echo "ERROR: ROLE must be prefill or decode, got ${ROLE}" >&2
    exit 2
    ;;
esac
# tpu_runner._get_padded_token_len asserts that a step fits the largest bucket.
if ! grep -q "\b${MAX_NUM_BATCHED_TOKENS}\b" <<<"${COMPILE_SIZES}"; then
  echo "ERROR: COMPILE_SIZES ${COMPILE_SIZES} must include" \
    "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" >&2
  exit 2
fi

KV_PORT="${KV_PORT:-14579}"
SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT:-9600}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TPU_KV_SHM_POOL_GB="${TPU_KV_SHM_POOL_GB:-16}"
CACHE_ROOT="${CACHE_ROOT:-/dev/shm/pd_cache_${ROLE}}"
API_SERVER_COUNT="${API_SERVER_COUNT:-8}"

ulimit -c 0
ulimit -n 65536 || true
rm -rf /dev/shm/psm_* /dev/shm/sem.* /tmp/libengram_gather_omp_*.so
mkdir -p "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/xla" "${CACHE_ROOT}/inductor" \
  "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/plan_dump"
mkdir -p /root/.cache
ln -sfn "${CACHE_ROOT}/vllm" /root/.cache/vllm
ln -sfn "${CACHE_ROOT}/xla" /tmp/xla_cache
grep -q 'metadata.google.internal' /etc/hosts \
  || echo '127.0.0.1 metadata.google.internal' >>/etc/hosts
pgrep -f mock_metadata.py >/dev/null \
  || (LD_PRELOAD="" python3 /workspace/mock_metadata.py \
        >/workspace/mock_metadata.log 2>&1 &)
sleep 1

# --- TPU topology (one 2x2x1 host, 8 TensorCores, one worker per core) -----
unset TPU_PROCESS_PORT TPU_PROCESS_ADDRESSES TPU_WORKER_ID TPU_VISIBLE_CHIPS \
  TPU_RUNTIME_METRICS_PORTS TPU_SKIP_MDS_QUERY VBAR_CONTROL_SERVICE_URL \
  TPU_WORKER_HOSTNAMES TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT
sb=""
for i in 0 1 2 3 4 5 6 7; do sb+="${sb:+,}localhost:$((SB_PORT_BASE + i))"; done
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((SB_PORT_BASE - 1))
export TORCH_TPU_SLICEBUILDER_ADDRESSES="${sb}"
export TORCH_TPU_DP_MASTER_PORT="${DP_MASTER_PORT}"
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1 TOPOLOGY=2x2x1 TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_VISIBLE_DEVICES=0,1,2,3
[ -S /var/run/tpu-plugin/vbar.sock ] \
  && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"
export TPU_RUNTIME_METRICS_PORTS="8431,8432,8433,8434,8435,8436,8437,8438"
export GCE_METADATA_HOST=127.0.0.1:80 GCE_METADATA_IP=127.0.0.1
export GOOGLE_CLOUD_PROJECT=tpu-prod-env-one-vm GCLOUD_PROJECT=tpu-prod-env-one-vm
export LD_PRELOAD=/workspace/metadata_hook.so

# --- caches / runtime ------------------------------------------------------
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export VLLM_XLA_CACHE_PATH="${CACHE_ROOT}/xla"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_ROOT}/xla"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export TMPDIR="${CACHE_ROOT}/tmp"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export OMP_NUM_THREADS=6 GOMP_SPINCOUNT=1000
export USE_PHASED_PROFILER=false VLLM_LOG_POST_WARMUP_HBM=1
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export PYTHONPATH="/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 SPEC_WARMUP=0 VLLM_SPEC_WARMUP=0
# Native aborts (Raiden is C++) still print the Python stack, and nothing is lost
# in a block-buffered stdout when a worker dies.
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

# --- DeepSeek-V4.1 model env (decode block, no DSpark / DecodeBench) -------
export TPU_ROPE_CACHE_TRUNCATE=1 TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_COLLECTION_CHUNK_SIZE=8192 VLLM_TPU_FAST_TOKEN_SUBSTITUTION=0
export ENABLE_CONNECTOR=0 ENABLE_DECODE_BENCH_CONNECTOR=0
export USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v2
export MOE_FUSED_EP_ENABLE_W4A8=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1
export MOE_REQUANTIZE_BLOCK_SIZE=512 TPU_MOE_DENSE_EXPERT_THRESHOLD=0
export VLLM_TPU_MOE_FP8_DISPATCH=1 VLLM_DSV41_MOE_EP_V2_ALL39=0
export VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS=16384 TPU_MOE_SKIP_PADDED_TOKENS=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS=0 VLLM_DSV41_MOE_FP4_GMM1_FP8=0
export VLLM_DSV41_INDEXER_CAND_READ_16K=0 VLLM_DSV41_DECODER_TP_ATTN_VERIFY=0
export VLLM_DSV41_DECODER_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_CSA_UNION=0 VLLM_DSV41_CSA_UNION_SC=0
export VLLM_DSV41_CSA_UNION_VERIFY_BQ=6 VLLM_DSV41_CSA_UNION_BQ=6
export VLLM_DSV41_CSA_UNION_M=1536 VLLM_DSV41_CSA_UNION_MAX_TOKENS=0
export VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT=-1
export VLLM_DSV41_FAST_GROUPED_ATTN=1 VLLM_DSV41_REPL_SHARED_EXPERTS=1
export VLLM_DSV41_ENGRAM_H2D_PREFIX=0 VLLM_DSV41_TP_SHARD_ENGRAM=0
export VLLM_DSV41_CED="${CED}" VLLM_DSV41_HYBRID_CED_DECODE=0
export VLLM_DSV41_PURE_20L_PREFILL=0
export VLLM_DSV41_TP_ATTN=0 VLLM_DSV41_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_TP_ATTN_LOCAL_TOPK=1 VLLM_DSV41_TP_INDEXER=1
export VLLM_DSV41_TP_INDEXER_MIN_TOKENS=0 VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS=0
export VLLM_DSV41_L2_CAND_K_MIN_TOKENS=0 VLLM_DSV41_ULYSSES_MIN_TOKENS=0
export VLLM_DSV41_HOIST_CSA_GATHER=0 VLLM_DSV41_HOIST_CSA_MAX_TOKENS=0
export VLLM_DSV41_FUSED_SEAM_QUANT=1 VLLM_DSV41_FUSED_Q_ROPE=1
export VLLM_DSV41_DSPARK_SHARDED_CTX_KV=1 TPU_DEEPSEEK_V41_ENGRAM_DEBUG_VERIFY=0
export VLLM_SPLIT_TOKENIZE_MIN_CHARS=65536
# The decode LIBTPU_INIT_ARGS: SparseCore offloads off (the prefill flag
# --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true hangs DP=8).
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true --xla_tpu_enable_sparse_core_compute_offload=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_offloading_gather_to_sparsecore=false --xla_tpu_offload_all_supported_gathers_to_sparsecore=false --xla_tpu_enable_offloading_scatter_to_sparsecore=false --xla_tpu_enable_offloading_sort_to_sparsecore=false --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_offloading_reshape_to_sparsecore=false --xla_tpu_enable_offloading_reduce_to_sparsecore=false --xla_tpu_num_sparse_cores_for_gather_offloading=0 --xla_tpu_num_sparse_cores_for_scatter_offloading=0 --xla_tpu_num_sparse_cores_for_sort_offloading=0 --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false --xla_tpu_enable_sparse_core_async_all_to_all=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"

# --- Raiden Stage 3 (launch_dsv4_disagg_multihost.sh) -----------------------
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 TPU_RAIDEN_DSV4_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden TPU_RAIDEN_TRANSFER_PARALLELISM=1
export TPU_ENABLE_D2H_TRANSFER=1 TPU_HMA_D2H_COPY_IMPL=device_put
export TPU_HMA_MIN_DONE_RECVING_BATCH=1 TPU_HMA_MAX_DONE_RECVING_BATCH=1
export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1 TPU_HMA_MAX_INFLIGHT_PULLS=8
export TPU_MAX_HOST_KV_BUFFER_SIZE=64 TPU_P2P_WAIT_PULL_TIMEOUT=300
export TPU_KV_SHM_POOL_GB TPU_SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT}"
export TPU_RAIDEN_PLAN_DUMP_DIR="${CACHE_ROOT}/plan_dump"
unset TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
export TPU_KV_TRANSFER_NAMESPACE="${ROLE}_${RUN_ID}"
export TPU_RAIDEN_JOB_NAME="${ROLE}"
export TPU_RAIDEN_ENGINE_ID="${ROLE}-engine-${RUN_ID}"
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="${ADVERTISE_HOST}"
export TPU_RAIDEN_RESHARD_PORT_BASE="${CTRL_PORT}"
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=$((CTRL_PORT + 100))
export TPU_KV_TRANSFER_PORT="${KV_TRANSFER_PORT}"

kv_cfg='{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"'"${KV_ROLE}"'","kv_port":'"${KV_PORT}"'}'

if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  async_flag="--async-scheduling"
else
  async_flag="--no-async-scheduling"
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  prefix_cache_flags=(--enable-prefix-caching --prefix-cache-retention-interval "${PREFIX_CACHE_RETENTION_INTERVAL}")
else
  prefix_cache_flags=(--no-enable-prefix-caching)
fi

ENABLE_TOOL_AND_REASONING_PARSERS="${ENABLE_TOOL_AND_REASONING_PARSERS:-1}"
parser_flags=()
if [[ "${ENABLE_TOOL_AND_REASONING_PARSERS}" == "1" ]]; then
  parser_flags=(
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v41
    --reasoning-parser deepseek_v41
  )
fi

args=(
  serve "${MODEL_PATH}"
  --host 0.0.0.0 --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --api-server-count "${API_SERVER_COUNT}"
  --renderer-num-workers 4
  --prefill-schedule-interval 1
  --data-parallel-size 8 --data-parallel-size-local 8
  --tensor-parallel-size 1
  --enable-expert-parallel
  --kv-cache-dtype fp8
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --no-disable-hybrid-kv-cache-manager
  --enable-prompt-tokens-details
  "${prefix_cache_flags[@]}"
  "${parser_flags[@]}"
  "${async_flag}"
  --kv-transfer-config "${kv_cfg}"
  --compilation-config "{\"compile_sizes\": ${COMPILE_SIZES}}"
)

if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  export VLLM_SPEC_DECODE_TOKENS="${DSPARK_TOKENS}"
  args+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${DSPARK_TOKENS}}")
fi

echo "[PD-LAUNCH] role=${ROLE} run_id=${RUN_ID} advertise=${ADVERTISE_HOST}" \
  "port=${PORT} ctrl=${CTRL_PORT} kvt=${KV_TRANSFER_PORT} ced=${CED}" \
  "dspark=${ENABLE_DSPARK} async=${ASYNC_SCHEDULING}" \
  "mnbt=${MAX_NUM_BATCHED_TOKENS} compile_sizes=${COMPILE_SIZES}"
printf '[PD-LAUNCH] cmd: vllm'; printf ' %q' "${args[@]}"; echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec vllm "${args[@]}"

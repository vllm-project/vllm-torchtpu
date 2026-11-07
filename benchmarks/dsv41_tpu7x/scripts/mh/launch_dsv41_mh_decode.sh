#!/bin/bash
# In-pod launcher: ONE vLLM decode engine spanning the hosts of a tpu7x 2x2x2 (or 2x2x4) slice
# via vllm-torchtpu's "mp" multi-host DP path (no Ray). Every host runs this; role from HOST_INDEX.
#
# Required env (written to /tmp/mh_env.sh by run_mh_decode.py):
#   HOST_INDEX (0 = head), NUM_HOSTS, HEAD_IP, DP_LOCAL (8), POD_IP (this host's routable IP)
# Optional: DSPARK_TOKENS (5), MAX_NUM_SEQS (64), MNBT (384), COMPILE_SIZES, PORT (9400),
#           DP_RPC_PORT (29500), TORCH_TPU_DP_MASTER_PORT (29700), DSV41_EXTRA_SERVE_ARGS
set -uo pipefail
source /tmp/mh_env.sh
: "${HOST_INDEX:?}" "${NUM_HOSTS:?}" "${HEAD_IP:?}" "${DP_LOCAL:=8}" "${POD_IP:?}"
PORT="${PORT:-9400}"; DP_RPC_PORT="${DP_RPC_PORT:-29500}"
DP_SIZE=$(( DP_LOCAL * NUM_HOSTS ))
DSPARK_TOKENS="${DSPARK_TOKENS:-5}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"; MNBT="${MNBT:-384}"
COMPILE_SIZES="${COMPILE_SIZES:-[16, 40, 48, 64, 96, 192, 384]}"
CACHE_ROOT=/dev/shm/pd_cache_mh; mkdir -p "$CACHE_ROOT"/{vllm,xla,inductor,tmp} /dev/shm/profiles
LOG=/dev/shm/mh_decode.log

ulimit -c 0; ulimit -n 65536 || true
rm -rf /dev/shm/psm_* /dev/shm/sem.* 2>/dev/null
grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts
pgrep -f mock_metadata.py >/dev/null || (LD_PRELOAD="" python3 /workspace/mock_metadata.py > /workspace/mock_metadata.log 2>&1 &)
sleep 1

# ---- multi-host (mp backend): let tpu_mp_multihost.py build the slice; blank GKE's injected slice env
export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT='' MEGASCALE_SLICE_ID=''
export TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT='' TPU_WORKER_HOSTNAMES=''
unset TPU_WORKER_ID TPU_VISIBLE_CHIPS TPU_RUNTIME_METRICS_PORTS VBAR_CONTROL_SERVICE_URL TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT
unset TPU_HOST_BOUNDS HOST_BOUNDS TPU_TOPOLOGY TOPOLOGY TPU_CHIPS_PER_HOST_BOUNDS CHIPS_PER_HOST_BOUNDS TPU_ACCELERATOR_TYPE TPU_VISIBLE_DEVICES
unset TORCH_TPU_SLICEBUILDER_ADDRESSES MASTER_ADDR MASTER_PORT
export TPU_MULTIHOST_BACKEND=mp TPU_SKIP_MDS_QUERY=1
export VLLM_HOST_IP="$POD_IP"
export TORCH_TPU_DP_MASTER_PORT="${TORCH_TPU_DP_MASTER_PORT:-29700}"
[ -S /var/run/tpu-plugin/vbar.sock ] && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"
export GCE_METADATA_HOST=127.0.0.1:80 GCE_METADATA_IP=127.0.0.1
export GOOGLE_CLOUD_PROJECT=tpu-prod-env-one-vm GCLOUD_PROJECT=tpu-prod-env-one-vm
export LD_PRELOAD=/workspace/metadata_hook.so

# ---- same DSv4.1 decode runtime env as the single-host decode server
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm" VLLM_XLA_CACHE_PATH="${CACHE_ROOT}/xla"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_ROOT}/xla" TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor" TMPDIR="${CACHE_ROOT}/tmp"
export VLLM_ENGINE_READY_TIMEOUT_S=7200 OMP_NUM_THREADS=6 GOMP_SPINCOUNT=1000
export USE_PHASED_PROFILER=false VLLM_LOG_POST_WARMUP_HBM=1
export VLLM_PLUGINS=torchtpu,torchtpu_layers VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 TORCHINDUCTOR_AUTOGRAD_CACHE=0 SPEC_WARMUP=0 VLLM_SPEC_WARMUP=0
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export TPU_ROPE_CACHE_TRUNCATE=1 TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_COLLECTION_CHUNK_SIZE=8192 VLLM_TPU_FAST_TOKEN_SUBSTITUTION=0
export ENABLE_CONNECTOR=0 ENABLE_DECODE_BENCH_CONNECTOR=0
export USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v2 MOE_FUSED_EP_ENABLE_W4A8=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1
export MOE_REQUANTIZE_BLOCK_SIZE=512 TPU_MOE_DENSE_EXPERT_THRESHOLD=0
export VLLM_TPU_MOE_FP8_DISPATCH=1 VLLM_DSV41_MOE_EP_V2_ALL39=0
export VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS=16384 TPU_MOE_SKIP_PADDED_TOKENS=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS=0 VLLM_DSV41_MOE_FP4_GMM1_FP8=0
export VLLM_DSV41_INDEXER_CAND_READ_16K=0 VLLM_DSV41_DECODER_TP_ATTN_VERIFY=0 VLLM_DSV41_DECODER_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_CSA_UNION=0 VLLM_DSV41_CSA_UNION_SC=0 VLLM_DSV41_CSA_UNION_VERIFY_BQ=6 VLLM_DSV41_CSA_UNION_BQ=6
export VLLM_DSV41_CSA_UNION_M=1536 VLLM_DSV41_CSA_UNION_MAX_TOKENS=0 VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT=-1
export VLLM_DSV41_FAST_GROUPED_ATTN=1 VLLM_DSV41_REPL_SHARED_EXPERTS=1
export VLLM_DSV41_ENGRAM_H2D_PREFIX=0 VLLM_DSV41_TP_SHARD_ENGRAM=0
export VLLM_DSV41_CED=0 VLLM_DSV41_HYBRID_CED_DECODE=0 VLLM_DSV41_PURE_20L_PREFILL=0
export VLLM_DSV41_TP_ATTN=0 VLLM_DSV41_TP_ATTN_MIN_TOKENS=999999 VLLM_DSV41_TP_ATTN_LOCAL_TOPK=1 VLLM_DSV41_TP_INDEXER=1
export VLLM_DSV41_TP_INDEXER_MIN_TOKENS=0 VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS=0 VLLM_DSV41_L2_CAND_K_MIN_TOKENS=0 VLLM_DSV41_ULYSSES_MIN_TOKENS=0
export VLLM_DSV41_HOIST_CSA_GATHER=0 VLLM_DSV41_HOIST_CSA_MAX_TOKENS=0
export VLLM_DSV41_FUSED_SEAM_QUANT=1 VLLM_DSV41_FUSED_Q_ROPE=1 TPU_DEEPSEEK_V41_ENGRAM_DEBUG_VERIFY=0
export VLLM_DSV41_DSPARK_SHARDED_CTX_KV="${VLLM_DSV41_DSPARK_SHARDED_CTX_KV:-1}"
export VLLM_SPLIT_TOKENIZE_MIN_CHARS=65536
export VLLM_SPEC_DECODE_TOKENS="${DSPARK_TOKENS}"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true --xla_tpu_offload_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=524288 --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true"

args=(
  /dev/shm/DeepSeek-V4.1-Flash
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash
  --trust-remote-code
  --renderer-num-workers 4
  --prefill-schedule-interval 1
  --data-parallel-size "${DP_SIZE}" --data-parallel-size-local "${DP_LOCAL}"
  --data-parallel-address "${HEAD_IP}" --data-parallel-rpc-port "${DP_RPC_PORT}"
  --tensor-parallel-size 1
  --enable-expert-parallel
  --kv-cache-dtype fp8
  --max-model-len 131072
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MNBT}"
  --gpu-memory-utilization 0.85
  --no-disable-hybrid-kv-cache-manager
  --enable-prompt-tokens-details
  --enable-prefix-caching --prefix-cache-retention-interval 1024
  --async-scheduling
  --compilation-config "{\"compile_sizes\": ${COMPILE_SIZES}}"
)
if [[ "${DSPARK_TOKENS}" != "0" ]]; then
  args+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${DSPARK_TOKENS}}")
fi
if [[ -n "${DSV41_EXTRA_SERVE_ARGS:-}" ]]; then eval "args+=(${DSV41_EXTRA_SERVE_ARGS})"; fi

echo "[MH-LAUNCH] host=${HOST_INDEX}/${NUM_HOSTS} head=${HEAD_IP} pod_ip=${POD_IP} dp=${DP_SIZE} local=${DP_LOCAL} mnbt=${MNBT} seqs=${MAX_NUM_SEQS} k=${DSPARK_TOKENS} compile=${COMPILE_SIZES}" | tee -a "$LOG"
if [[ "${HOST_INDEX}" == "0" ]]; then
  printf '[MH-LAUNCH] cmd: vllm serve'; printf ' %q' "${args[@]}" --api-server-count 8 --host 0.0.0.0 --port "${PORT}"; echo
  exec vllm serve "${args[@]}" --api-server-count 8 --host 0.0.0.0 --port "${PORT}" >> "$LOG" 2>&1
else
  start_rank=$(( HOST_INDEX * DP_LOCAL ))
  printf '[MH-LAUNCH] cmd: vllm serve'; printf ' %q' "${args[@]}" --headless --data-parallel-start-rank "${start_rank}"; echo
  exec vllm serve "${args[@]}" --headless --data-parallel-start-rank "${start_rank}" >> "$LOG" 2>&1
fi

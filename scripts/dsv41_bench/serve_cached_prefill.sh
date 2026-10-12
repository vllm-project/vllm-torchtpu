#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Activate the pinned experiment environment before invoking this launcher.
set -Eeuo pipefail
model=${MODEL_PATH:-/models/DeepSeek-V4.1-Flash}
tp=${TP:-1}
dp=${DP:-8}
chunk=${CHUNK:-2048}
max_seqs=${MAX_SEQS:-1}
export TPU_RECLAIM_COMPILE_MEMORY=1
export OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm VLLM_TARGET_DEVICE=tpu
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=0
export TPU_DEEPSEEK_V41_IMPL=v2 TPU_DEEPSEEK_V41_CED=1 TPU_DEEPSEEK_V41_SP=1 TPU_DEEPSEEK_V41_LOCAL_Q=1 TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1
export USE_MOE_SPARSE_CORE=0 USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_ENABLE_W4A8=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1
export RUNAI_STREAMER_MEMORY_LIMIT=34359738368 TPU_SHARDED_LOAD_SYNC_EVERY=512
export VLLM_ENGINE_READY_TIMEOUT_S=7200 TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export LIBTPU_INIT_ARGS='--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0'
export VLLM_SERVER_DEV_MODE=1
# Cache roots, when supplied, must be keyed by source/runtime/topology/buckets.
kv_args=(--gpu-memory-utilization 0.85)
if [[ "${KV_BYTES:-auto}" != auto ]]; then
  kv_args+=(--kv-cache-memory-bytes "${KV_BYTES:-auto}")
fi
exec vllm serve "$model" \
  --load-format runai_streamer --served-model-name deepseek-v41 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --tensor-parallel-size "$tp" --data-parallel-size "$dp" \
  --enable-expert-parallel --enable-ep-weight-filter --language-model-only \
  --async-scheduling --enable-prefix-caching --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --engram-config '{"dp_shared_memory":false}' \
  --block-size 16 --max-model-len 82944 \
  --max-num-batched-tokens "$chunk" --max-num-seqs "$max_seqs" \
  "${kv_args[@]}" \
  --compilation-config "{\"compile_sizes\":[128,256,$chunk]}" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"$model\"}" \
  "$@"

#!/bin/bash
# shellcheck disable=SC2034  # Sourced by run_benchmarks.sh
# DP8 C384 quality gate using the steady DSpark decode serving configuration.
# The GSM8K floor is inherited from TP8;
# it is a quality requirement, not a newly measured DP8 baseline.
MODEL="deepseek-ai/DeepSeek-V4.1-Flash"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
ENABLE_PREFIX_CACHING=true
MAX_MODEL_LEN=98816
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=48
ASYNC_SCHEDULING=true
KV_CACHE_DTYPE=fp8
GPU_MEMORY_UTILIZATION=0.85
NUM_PROMPTS=384
ISL_OSL_CONFIGS="128:16"
CONCURRENCY_OPTIONS=384
RANDOM_RANGE_RATIO=0.0
BENCHMARK_TEMPERATURE=0
LM_EVAL_TASKS=gsm8k
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=384,max_retries=1,timeout=3600"
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}'
SERVER_READY_WAIT_MIN=120
EVAL_TOLERANCE=0.03
export OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0
export RUNAI_STREAMER_MEMORY_LIMIT=34359738368 TPU_SHARDED_LOAD_SYNC_EVERY=512
export MOE_FUSED_EP_PREPROCESS_BACKEND=topology
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_tpu_dvfs_p_state=7 --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--language-model-only --enable-ep-weight-filter --block-size=16 --attention-config={\"indexer_kv_dtype\":\"fp8\"} --engram-config={\"cpu_offload\":true} --additional-config={\"deepseek_v41\":{\"execution\":\"ced\",\"moe\":\"fused_ep_w4a8\"}} --compilation-config={\"compile_sizes\":[288,2048]} --speculative-config={\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"${MODEL_URI:-$MODEL}\"}"

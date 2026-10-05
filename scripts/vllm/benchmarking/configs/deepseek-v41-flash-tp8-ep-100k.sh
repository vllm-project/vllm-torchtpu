#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k long-context prefill (CSA2 / FP8 KV cache work).
# Sized for 1 x (100000 + 1024) tokens, concurrency 1; 2048-token prefill
# chunks, async scheduling. Primary long-context config.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=1
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
NUM_PROMPTS=2
ISL_OSL_CONFIGS="100000:1024"
CONCURRENCY_OPTIONS="1"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
LM_EVAL_TASKS="gsm8k"
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=1,max_retries=1,timeout=3600"
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}'
EVAL_TOLERANCE="0.03"
# MXFP4 expert processing alone takes ~40 min on v7x-8.
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
# V4.1 impl v2 (models/vllm/deepseek_v41/v2); use with KV_CACHE_DTYPE=fp8.
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
# 256 MiB held ~44k tokens in the 4096-len smoke run; 1280 MiB leaves margin for
# 103,424 tokens.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--no-enable-prefix-caching --enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=1342177280 --compilation-config={\"compile_sizes\":[2,2048]}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi

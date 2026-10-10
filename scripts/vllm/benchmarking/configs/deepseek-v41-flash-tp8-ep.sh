#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# vLLM v0.31.1rc0 / TorchTPU compiled text baseline. The shared eval flow uses
# GSM8K's first 100 test examples; preserve lm_eval's task and answer filters.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
ASYNC_SCHEDULING=false
# run_benchmarks.sh defaults to fp8, which the TPU V4.1 attention rejects: its
# SWA cache is BF16 and the compressed KV is packed FP8 on its own.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
ISL_OSL_CONFIGS="${ISL_OSL_CONFIGS:-128:16}"
CONCURRENCY_OPTIONS="${CONCURRENCY_OPTIONS:-2}"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
LM_EVAL_TASKS="gsm8k"
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=8,max_retries=1,timeout=3600"
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}'
SERVER_READY_WAIT_MIN="${SERVER_READY_WAIT_MIN:-60}"
EVAL_TOLERANCE="${EVAL_TOLERANCE:-0.03}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
DEEPSEEK_V41_COMPILE_SIZES="${DEEPSEEK_V41_COMPILE_SIZES:-[8,128,2048,$MAX_NUM_BATCHED_TOKENS]}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--no-async-scheduling --enable-ep-weight-filter --attention-config={\"indexer_kv_dtype\":\"fp8\"} --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --compilation-config={\"compile_sizes\":$DEEPSEEK_V41_COMPILE_SIZES}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi

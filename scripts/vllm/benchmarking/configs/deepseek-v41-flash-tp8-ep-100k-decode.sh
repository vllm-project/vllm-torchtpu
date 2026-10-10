#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k long-context decode, for run_decode_benchmark.sh.
# Sized for 16 x (100000 + 2048) tokens: ISL_OSL_CONFIGS and
# CONCURRENCY_OPTIONS are the decode benchmark's prompt/decode lengths and
# concurrency sweep. With 2048-token prefill chunks the first of 16 requests
# decodes ~735 tokens while the others prefill, so 2048 decode tokens leave
# ~1300 tokens where all 16 decode together.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
# fp8 selects v2's FP8 main KV record and the SparseMLA kernel. TPU V4.1 does
# not use the fp8_ds_mla layout, so auto would mean BF16 KV and the slower
# generic paged attention.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ISL_OSL_CONFIGS="100000:2048"
CONCURRENCY_OPTIONS="${CONCURRENCY_OPTIONS:-4 8 16}"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
# Required by run_benchmarks.sh for every config; no eval runs here.
EVAL_TOLERANCE="0.03"
# MXFP4 expert processing alone takes ~40 min on v7x-8.
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
# V4.1 impl v2 (models/vllm/deepseek_v41/v2). v1 cannot start this config: its
# startup warmup runs MAX_NUM_SEQS requests at MAX_MODEL_LEN through every
# token bucket and needs ~32 GiB of scratch beyond the weights at C16.
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
# 16 x the 1280 MiB single-request budget of deepseek-v41-flash-tp8-ep-100k.
# On CDK, 16 x 100k requests peaked at 25% of it.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-21474836480}"
COMPILE_SIZES="${COMPILE_SIZES:-[4,8,16,2048]}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=${KV_CACHE_MEMORY_BYTES} --compilation-config={\"compile_sizes\":${COMPILE_SIZES}}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi

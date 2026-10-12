#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k prefill-only throughput, with the prefill settings
# of PR #1617/#1673 (scripts/dsv41_bench/integrated_prefill_suite.sh) at C1.
# Each request has a 100k-token prompt and outputs one token, sent one at a
# time, so the result's total_token_throughput is the prefill tok/s (input
# tokens / duration) and TTFT is the time to prefill 100k tokens. A warmup
# run goes first, so no measured request pays for a first-use compile.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
# Off so a measured prompt never reuses KV from a warmup or earlier run with
# the same prompt, which would skip the prefill this benchmark measures.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=1
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
# fp8 selects v2's FP8 main KV record and the SparseMLA kernel. TPU V4.1 does
# not use the fp8_ds_mla layout, so auto would mean BF16 KV and the slower
# generic paged attention.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NUM_PROMPTS=4
ISL_OSL_CONFIGS="100000:1"
CONCURRENCY_OPTIONS="1"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
BENCHMARK_WARMUP_RUNS=1
# Required by run_benchmarks.sh for every config; no eval runs here.
EVAL_TOLERANCE="0.03"
# MXFP4 expert processing alone takes ~40 min on v7x-8.
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
# V4.1 impl v2 (models/vllm/deepseek_v41/v2).
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
# Token-sharded mHC, norms, Engram and MoE (#1617), plus the token-local q
# projection under it (#1673). SP needs every compile size >= 2 * TP.
export TPU_DEEPSEEK_V41_SP="${TPU_DEEPSEEK_V41_SP:-1}"
export TPU_DEEPSEEK_V41_LOCAL_Q="${TPU_DEEPSEEK_V41_LOCAL_Q:-1}"
# The 1280 MiB single-request budget of deepseek-v41-flash-tp8-ep-100k; the
# suite's 14 GiB with 16 seqs was slower at C1. The 128 size fits CED's
# 128-token decoder replay of the last chunk, which 2048 would pad.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=1342177280 --compilation-config={\"compile_sizes\":[16,128,2048,8192]}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi

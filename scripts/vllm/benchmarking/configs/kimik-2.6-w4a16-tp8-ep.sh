#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly benchmark config for Kimi-K2.6 (TP=8, EP, MoE W4A16)

MODEL="${MODEL:-gs://tpu-inference-hf-llm-model-checkpoints/Kimi-K2.6-original/}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K2.6}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="compressed-tensors"
ATTENTION_BACKEND="CUSTOM"

# Optimal environment variables for the Pallas MLA v2 kernel and hardware path
export MLA_TRANSPOSE_KV_CACHE=1
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE="vllm"

# Optimal allocation fractions for TPU/JAX memory sharing
export PJRT_ALLOCATOR_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Redirect intermediate compiler caches to RAM disk (/dev/shm) for speed
export VLLM_XLA_CACHE_PATH=/dev/shm/vllm_xla_cache
export TRITON_CACHE_DIR=/dev/shm/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor_cache

# Enable the SparseCore path for MoE sharding collectives
export USE_MOE_SPARSE_CORE=1

# Disable runtime weight requantization
export DISABLE_WEIGHT_REQUANTIZATION=1

# Custom scheduler settings for libtpu
export LIBTPU_INIT_ARGS="--xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_async_collective_fusion=false"

# Restrict vLLM to only load our local plugins, bypassing conflicting pre-installed image plugins
export VLLM_PLUGINS="torchtpu,torchtpu_layers"

# Streaming settings for GCS runai_streamer
export RUNAI_STREAMER_CONCURRENCY=8
export RUNAI_STREAMER_MEMORY_LIMIT=17179869184  # 16 GiB
# Sync TPU copies every 100 parameters to prevent host OOM on massive models
export VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL=100

# Limit memory-hungry multimodal preprocessors since this is a text-only run
EXTRA_SERVE_ARGS="--limit-mm-per-prompt={\"image\":0,\"vision_chunk\":0,\"video\":0} --trust-remote-code"
if [[ "$MODEL" == gs://* || "$MODEL" == s3://* || "$MODEL" == az://* ]]; then
  EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--load-format runai_streamer"
fi

MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=64
BLOCK_SIZE=256
KV_CACHE_DTYPE="fp8"
GPU_MEMORY_UTILIZATION=0.9

# Performance Benchmarking Parameters
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
BENCHMARK_TEMPERATURE="0"
PERF_TOLERANCE="0.05"
EVAL_TOLERANCE="0.03"

#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf + accuracy for DeepSeek-V4-Pro on TPU v7x-16, TP=1 x DP=16 with
# EP. DP=16 is forced by size: 805 GiB of weights is 72.8 GiB/rank.
# Streams from GCS by default; v7x-16 agents have no attached disk.
MODEL="deepseek-ai/DeepSeek-V4-Pro-0813"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/deepseek-v4-pro-0813}"

TENSOR_PARALLELISM=1
DATA_PARALLELISM=16
# This host's share of the DP ranks.
DATA_PARALLELISM_LOCAL=8
ENABLE_EP=true
QUANTIZATION="deepseek_v4_fp8"
KV_CACHE_DTYPE="fp8_e4m3"

MAX_MODEL_LEN=9216
# 512 is the fallback if a SparseCore fault halts the first batch.
MAX_NUM_BATCHED_TOKENS=1024
# Per DP engine: vllm 0.27 never divides max_num_seqs by data_parallel_size.
MAX_NUM_SEQS=256
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ISL_OSL_CONFIGS="8192:1024"
# Matches glm-5.2-fp8-dp16-ep cell for cell, so the two are comparable.
CONCURRENCY_OPTIONS="1024"
NUM_PROMPTS=2048
RANDOM_RANGE_RATIO="0.0"

export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE="vllm"
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
export MOE_REQUANTIZE_BLOCK_SIZE=512
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_HASH_TABLE_ROW_MAJOR=1
# Fused MoE v2. W4A8 is required too: on fp4 weights the first var alone is a
# no-op.
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE=2
export TPU_MOE_SKIP_PADDED_TOKENS=1
export RAGGED_GATHER_REDUCE_VERSION=v3
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_NO_USAGE_STATS=1
# fork after torch_tpu/JAX threads start can deadlock workers; see tests/conftest.py.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Bring-up is ~74 min: weight load plus a cold compile.
export TORCH_DIST_TIMEOUT=1200
export VLLM_RPC_TIMEOUT=1200000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
export VLLM_ENGINE_READY_TIMEOUT_S=14400

# REQUIRED on the Ray path, silent if omitted: vllm only copies prefixed vars
# into DP engine actors, so the model-defining vars above are dropped and the
# actors build a different model without failing.
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="NEW_MODEL_DESIGN,MODEL_IMPL_TYPE,MOE_REQUANTIZE_WEIGHT_DTYPE,MOE_REQUANTIZE_BLOCK_SIZE,TPU_ROPE_CACHE_ROW_MAJOR,TPU_MOE_HASH_TABLE_ROW_MAJOR,TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE,TORCH_DIST_TIMEOUT,USE_MOE_FUSED_EP_KERNEL,MOE_FUSED_EP_ENABLE_W4A8,MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE,TPU_MOE_SKIP_PADDED_TOKENS,RAGGED_GATHER_REDUCE_VERSION,LIBTPU_INIT_ARGS"

# mmlu_llama prefills an assistant turn; renders without a trailing eos_token.
export TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE=1
SERVER_READY_WAIT_MIN=240

# memory_limit: unbounded streamer buffers got weight load OOM-killed at
# 897.8 GiB of 945.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--model-loader-extra-config={\"memory_limit\":4294967296,\"concurrency\":8} --enable-ep-weight-filter --safetensors-load-strategy=prefetch --trust-remote-code --generation-config=vllm --override-generation-config={\"do_sample\":false,\"temperature\":0.0,\"model\":\"deepseek-ai/DeepSeek-V4-Pro-0813\"}"

export CAPTURE_PROFILE="${CAPTURE_PROFILE:-1}"
export USE_PHASED_PROFILER="$CAPTURE_PROFILE"

PROFILE_GCS_BASE="gs://tpu-commons-ci/xprof/deepseek-v4-pro/torchtpu"

BENCHMARK_TEMPERATURE=0
# todo(patemotter) decrease tolerance to 0.02 once we have more nightly results
EVAL_TOLERANCE="0.03"

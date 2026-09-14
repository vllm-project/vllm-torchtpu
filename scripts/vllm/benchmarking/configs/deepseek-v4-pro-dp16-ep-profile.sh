#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# xprof capture for DeepSeek-V4-Pro, its own CI job with its own server like
# deepseek-v4-flash-dp8-ep-profile: capture costs 2.6x throughput, so it cannot
# share a server with the run whose numbers are reported. Everything but the
# workload mirrors deepseek-v4-pro-dp16-ep and must stay in sync with it.
MODEL="deepseek-ai/DeepSeek-V4-Pro-0813"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/deepseek-v4-pro-0813}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=16
DATA_PARALLELISM_LOCAL=8
ENABLE_EP=true

QUANTIZATION="deepseek_v4_fp8"
KV_CACHE_DTYPE="fp8_e4m3"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=256
GPU_MEMORY_UTILIZATION=0.90
# Mirrors the perf cell so the trace shows the operating point CI reports.
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1024"
NUM_PROMPTS=2048
RANDOM_RANGE_RATIO="0.0"

export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE="vllm"
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
export MOE_REQUANTIZE_BLOCK_SIZE=512
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_HASH_TABLE_ROW_MAJOR=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_NO_USAGE_STATS=1
# fork after torch_tpu/JAX threads start can deadlock workers; see tests/conftest.py.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export TORCH_DIST_TIMEOUT=1200
export VLLM_RPC_TIMEOUT=1200000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200

# Required on the Ray path; see deepseek-v4-pro-dp16-ep.sh.
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="NEW_MODEL_DESIGN,MODEL_IMPL_TYPE,MOE_REQUANTIZE_WEIGHT_DTYPE,MOE_REQUANTIZE_BLOCK_SIZE,TPU_ROPE_CACHE_ROW_MAJOR,TPU_MOE_HASH_TABLE_ROW_MAJOR,TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE,TORCH_DIST_TIMEOUT"

export CAPTURE_PROFILE=1
export USE_PHASED_PROFILER=1

PROFILE_GCS_BUCKET="tpu-commons-ci"
PROFILE_GCS_PATH="xprof/deepseek-v4-pro/torchtpu"
PROFILE_GCS_BASE="gs://$PROFILE_GCS_BUCKET/$PROFILE_GCS_PATH"

export VLLM_ENGINE_READY_TIMEOUT_S=14400
export TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE=1
SERVER_READY_WAIT_MIN=240

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--api-server-count=1 --model-loader-extra-config={\"memory_limit\":4294967296,\"concurrency\":8} --enable-ep-weight-filter --safetensors-load-strategy=prefetch --trust-remote-code --generation-config=vllm --override-generation-config={\"do_sample\":false,\"temperature\":0.0,\"model\":\"deepseek-ai/DeepSeek-V4-Pro-0813\"}"

BENCHMARK_TEMPERATURE=0
# todo(patemotter) decrease tolerance to 0.02 once we have more nightly results
EVAL_TOLERANCE="0.03"

#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly xprof capture for DeepSeek-V4-Flash. Serve geometry matches
# deepseek-v4-flash-dp8-ep; the workload and the profiler differ.
MODEL="deepseek-ai/DeepSeek-V4-Flash"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
QUANTIZATION="deepseek_v4_fp8"
KV_CACHE_DTYPE="fp8_e4m3"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=256
GPU_MEMORY_UTILIZATION=0.9

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="2048"
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

export CAPTURE_PROFILE=1
export USE_PHASED_PROFILER=1

PROFILE_GCS_BUCKET="tpu-commons-ci"
PROFILE_GCS_PATH="xprof/deepseek-v4-flash/torchtpu"
PROFILE_GCS_BASE="gs://$PROFILE_GCS_BUCKET/$PROFILE_GCS_PATH"

export VLLM_ENGINE_READY_TIMEOUT_S=7200
SERVER_READY_WAIT_MIN=120

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--api-server-count=1 --trust-remote-code --generation-config=vllm --override-generation-config={\"do_sample\":false,\"temperature\":0.0,\"model\":\"deepseek-ai/DeepSeek-V4-Flash\"}"

BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"

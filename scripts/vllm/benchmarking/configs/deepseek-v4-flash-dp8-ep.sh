#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly throughput benchmark for DeepSeek-V4-Flash on TPU v7x-8, TP=1 x DP=8
# with EP.
MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
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

export VLLM_ENGINE_READY_TIMEOUT_S=7200
# mmlu_llama prefills an assistant turn, which needs this to render
# without a trailing eos_token.
export TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE=1
SERVER_READY_WAIT_MIN=120

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --generation-config=vllm --override-generation-config={\"do_sample\":false,\"temperature\":0.0,\"model\":\"deepseek-ai/DeepSeek-V4-Flash-0731\"}"

export CAPTURE_PROFILE="${CAPTURE_PROFILE:-1}"
export USE_PHASED_PROFILER="$CAPTURE_PROFILE"

PROFILE_GCS_BASE="gs://tpu-commons-ci/xprof/deepseek-v4-flash/torchtpu"

BENCHMARK_TEMPERATURE=0
# todo(patemotter) decrease tolerance to 0.02 once we have more nightly results
EVAL_TOLERANCE="0.03"

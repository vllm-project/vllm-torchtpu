#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly xprof capture for DeepSeek-V4-Flash. Serve geometry matches
# deepseek-v4-flash-dp8-ep; the workload and the profiler differ.
MODEL="deepseek-ai/DeepSeek-V4-Flash"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
# The checkpoint self-describes: quant_method=fp8, scale_fmt=ue8m0.
QUANTIZATION=""
KV_CACHE_DTYPE="fp8"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.9

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="2048"
NUM_PROMPTS=2048
RANDOM_RANGE_RATIO="0.0"

export CAPTURE_PROFILE=1
export USE_PHASED_PROFILER=1

export VLLM_ENGINE_READY_TIMEOUT_S=7200
SERVER_READY_WAIT_MIN=120

# With DP, vllm starts one API frontend per rank and they race resolving the
# model into the loader cache; a losing frontend dies without binding.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--api-server-count=1"

BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"

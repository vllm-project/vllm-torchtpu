#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Presubmit guard sweep for Qwen3-0.6B with TP=1 and CUSTOM attention.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="8"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"

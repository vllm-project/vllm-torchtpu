#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9
BENCHMARK_TEMPERATURE=0
export USE_MOE_SPARSE_CORE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="--block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EVAL_TOLERANCE="0.02"

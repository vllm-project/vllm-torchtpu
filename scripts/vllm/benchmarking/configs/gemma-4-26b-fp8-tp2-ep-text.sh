#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, TP=2 on v7x-8).
MODEL="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="32"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
EVAL_TOLERANCE="0.02"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0} --hf-overrides {\"architectures\":[\"Gemma4ForCausalLM\"]}"

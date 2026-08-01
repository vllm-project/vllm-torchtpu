#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, TP=2 on v7x-8).
MODEL="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="512 1024 2048"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
EVAL_TOLERANCE="0.02"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --max-num-seqs=512 --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0} --hf-overrides {\"architectures\":[\"Gemma4ForCausalLM\"]}"

#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, TP=2 on v7x-8), multimodal (image input).
MODEL="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="512"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
EVAL_TOLERANCE="0.02"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --max-num-seqs=512 --mm-processor-kwargs {\"size\":{\"longest_edge\":262144,\"shortest_edge\":3136}} --disable-chunked-mm-input --mm-encoder-tp-mode data --compilation-config {\"cudagraph_mm_encoder\":true}"

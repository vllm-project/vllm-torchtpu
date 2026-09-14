#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: mgoin/Kimi-K3-pruned75 (MoE + MoonViT), TP=8 on v7x-8,
# multimodal (image input). Used with evalscope mmmu_pro for the vision
# encoder path; the ISL/OSL sweep below is a smoke test only.
MODEL="mgoin/Kimi-K3-pruned75"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="128:1"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=1
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
ATTENTION_BACKEND="CUSTOM"
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=24576
MAX_NUM_SEQS=64
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --compilation-config {\"cudagraph_mm_encoder\":true}"

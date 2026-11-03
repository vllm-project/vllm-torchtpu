#!/bin/bash
# Muse Spark Open 816B-A42B, NVFP4, text only, TP8 on one v7x-8 host, as #2025's accuracy-validated
# TP8 recipe; GSM8K (5-shot, completions) only, against its 95.00% strict.
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
MODEL="meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="8"
NUM_PROMPTS=64
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=512
MAX_NUM_SEQS=8
SERVER_READY_WAIT_MIN=180
GPU_MEMORY_UTILIZATION=0.80
KV_CACHE_DTYPE="auto"
ATTENTION_BACKEND="CUSTOM"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--load-format safetensors --safetensors-load-strategy eager --tokenizer-mode hf --dtype bfloat16 --additional-config {\"muse_packed_experts\":true} --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0}"
LM_EVAL_TASKS="gsm8k"
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.02"

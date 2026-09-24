#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Configuration for Gemma-4-31B-it MTP Speculative Decoding Benchmark on TPU v7x-8.
# Matches: https://docs.google.com/spreadsheets/d/1tJDNUZ7DS2eSlyTpoEKgPHa-kY1djbTDLqPWWtDOrR0 (gid=1078954007)

MODEL="google/gemma-4-31B-it"
SERVED_MODEL_NAME="gemma-4-31b-it"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
KV_CACHE_DTYPE="fp8"
GPU_MEMORY_UTILIZATION="0.92"
ATTENTION_BACKEND="CUSTOM"
ASYNC_SCHEDULING="true"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="mtp"
SPEC_DRAFT_MODEL="google/gemma-4-31B-it-assistant"
SPEC_DRAFT_TP=8
SPEC_K_LIST="0 1 2 3 4 5"
SPEC_BS_LIST="1 4 8"
SPEC_NUM_PROMPTS=100
SPEC_WARMUP_PROMPTS=20
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

EXTRA_SERVE_ARGS="--language-model-only --no-enable-prefix-caching --limit-mm-per-prompt '{\"image\":0,\"video\":0,\"audio\":0}'"

export USE_BATCHED_RPA_LONGCTX=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_USE_V1=1
export SKIP_JAX_PRECOMPILE=1
export SPEC_WARMUP=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Configuration for Llama-3.3-70B-Instruct Eagle3 Speculative Decoding Benchmark on TPU v7x-8.
# Matches: https://docs.google.com/spreadsheets/d/1tJDNUZ7DS2eSlyTpoEKgPHa-kY1djbTDLqPWWtDOrR0 (gid=120868918)

MODEL="meta-llama/Llama-3.3-70B-Instruct"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ASYNC_SCHEDULING="false"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="eagle3"
SPEC_DRAFT_MODEL="nvidia/Llama-3.3-70B-Instruct-Eagle3"
SPEC_DRAFT_TP=1
SPEC_K_LIST="0 1 2 3 4"
SPEC_BS_LIST="1"
SPEC_NUM_PROMPTS=100
SPEC_WARMUP_PROMPTS=20
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

EXTRA_SERVE_ARGS="--no-enable-prefix-caching"

export MODEL_IMPL_TYPE=vllm
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

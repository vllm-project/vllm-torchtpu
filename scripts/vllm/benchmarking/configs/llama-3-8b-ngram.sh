#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Configuration for Llama-3-8B ngram Speculative Decoding Benchmark on TPU v7x-8.
# Matches: https://docs.google.com/spreadsheets/d/1tJDNUZ7DS2eSlyTpoEKgPHa-kY1djbTDLqPWWtDOrR0 (gid=1940133939)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ASYNC_SCHEDULING="false"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="ngram"
SPEC_DRAFT_MODEL=""
SPEC_K_LIST="0 1 2 4 8"
SPEC_BS_LIST="1 2"
SPEC_NUM_PROMPTS=100
SPEC_WARMUP_PROMPTS=20
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

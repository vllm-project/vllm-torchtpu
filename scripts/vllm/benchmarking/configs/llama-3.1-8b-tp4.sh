#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# fp8 KV-cache CONTROL config (b/510425663) — the WORKING case, TP=4 on a v6e-4.
#
# Why this model: 8 KV heads over TP=4 leaves 2 KV heads per chip, which is
# enough to fill an fp8 packing group, so fp8 already halves the KV cache here
# today. This config exists to prove the layout fix does not break the models
# that were never broken — the fix changes the cache layout for every model,
# not just the ones that pad.
#
# Expect: fp8 reports ~2x the "GPU KV cache size: N tokens" of bf16, BEFORE and
# AFTER the fix, with TTFT/TPOT unchanged. Any movement here is a regression.
#
# Pair with deepseek-r1-distill-qwen-7b-tp4.sh, which is the actual repro.
#
# Run both arms from this one file — do not edit it in between:
#   cd <repo>
#   KV_CACHE_DTYPE=auto RESULTS_DIR=/tmp/kvbench/before_llama_bf16 \
#     ./scripts/vllm/benchmarking/run_benchmarks.sh --config llama-3.1-8b-tp4
#   KV_CACHE_DTYPE=fp8  RESULTS_DIR=/tmp/kvbench/before_llama_fp8 \
#     ./scripts/vllm/benchmarking/run_benchmarks.sh --config llama-3.1-8b-tp4
#
# WEIGHTS: real weights are already in the local HF cache (~15GB), so this runs
# for real. That makes this config — unlike the dummy-weight repro config — the
# one to reuse for the accuracy check that must accompany a layout change.
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""

# Baseline arm is the KV dtype a user gets by default (auto -> model dtype ->
# bfloat16). Overridden per-arm from the environment; assigning it with :-
# rather than unconditionally is what makes the env override work at all.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Kept identical to the repro config so the two models' numbers are directly
# comparable. 1024:1024 reads TPOT, 8192:1024 reads TTFT.
ISL_OSL_CONFIGS="1024:1024 8192:1024"
CONCURRENCY_OPTIONS="32"

# 0.0 pins every request to exactly ISL/OSL; see the repro config for why this
# deviates from the repo's usual 0.8.
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-1}"

# Pinned rather than inherited so a future change to run_benchmarks.sh defaults
# cannot silently desync the "before" and "after" halves of this experiment.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Prefix caching stays off (run_benchmarks.sh default): reusing blocks across
# requests would decouple observed throughput from KV cache capacity, which is
# the exact quantity under test.
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"

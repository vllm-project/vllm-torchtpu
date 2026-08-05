#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# fp8 KV-cache repro config (b/510425663) — the BROKEN case, TP=4 on a v6e-4.
#
# Why this model: it has 4 KV heads, so TP=4 gives exactly 1 KV head per chip
# with no replication. That is the condition where get_kv_cache_shape() in
# src/vllm_torchtpu/layers/vllm/attention.py pads the fp8 packing dimension
# from 2 used slots up to 4, which exactly cancels the fp8 byte saving. Its
# head_dim=128 exercises the main RPA v3 kernel, and the model is dense with no
# sliding window, so nothing else moves between the two arms.
#
# Expect: bf16 and fp8 report the SAME "GPU KV cache size: N tokens" today.
# After the layout fix, fp8 should report roughly 2x bf16.
#
# Pair with llama-3.1-8b-tp4.sh (2 KV heads/chip), which already gets the full
# 2x from fp8 and therefore acts as the regression guard for the fix.
#
# Run BOTH arms from this one file — do not edit it in between, or the two runs
# stop being comparable:
#   cd <repo>
#   KV_CACHE_DTYPE=auto RESULTS_DIR=/tmp/kvbench/before_deepseek_bf16 \
#     ./scripts/vllm/benchmarking/run_benchmarks.sh --config deepseek-r1-distill-qwen-7b-tp4
#   KV_CACHE_DTYPE=fp8  RESULTS_DIR=/tmp/kvbench/before_deepseek_fp8 \
#     ./scripts/vllm/benchmarking/run_benchmarks.sh --config deepseek-r1-distill-qwen-7b-tp4
#
# WEIGHTS: real weights (~15GB) are in the local HF cache, so this runs for
# real and can therefore also carry the accuracy check that has to accompany a
# KV cache layout change. Set USE_DUMMY_WEIGHTS=1 to swap in random weights for
# a faster iteration loop — valid for capacity/TTFT/TPOT, which depend on tensor
# shapes and dtypes rather than weight values, but never valid for accuracy.
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""

if [ "${USE_DUMMY_WEIGHTS:-0}" = "1" ]; then
    _kvbench_load_args="--load-format dummy"
else
    _kvbench_load_args=""
fi
EXTRA_SERVE_ARGS="$_kvbench_load_args ${EXTRA_SERVE_ARGS:-}"

# Baseline arm is the KV dtype a user gets by default (auto -> model dtype ->
# bfloat16). Overridden per-arm from the environment; assigning it with :-
# rather than unconditionally is what makes the env override work at all.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# 1024:1024 is decode-dominated (reads TPOT); 8192:1024 is prefill-dominated
# (reads TTFT). Both are needed because the layout fix is expected to be
# neutral on decode but is the likeliest place to regress prefill — the JAX
# port of this fix measured prefill 580us -> 680us.
ISL_OSL_CONFIGS="1024:1024 8192:1024"
CONCURRENCY_OPTIONS="32"

# 0.0 pins every request to exactly ISL/OSL. The repo's other configs use 0.8,
# but sampled lengths add run-to-run variance, and this is a paired A/B where
# the effect we are chasing (TTFT/TPOT deltas) can be small.
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
#
# NOT "CUSTOM" here, unlike the repo's other configs. This model has 28 query
# heads over 4 KV heads = 7 query heads per KV head, and 7 is odd. The batched
# RPA kernel behind CUSTOM does `pltpu.bitcast(out, ...)` on its output
# accumulator (kernels/experimental/batched_rpa/kernel.py:94), which requires
# that dimension to be exactly packable — 2 values per 32-bit word for bf16.
# 7 is not divisible by 2, so every worker dies at startup with:
#   ValueError: Not implemented: the 2nd minor dim can not be perfectly packed
# The default FLASH_ATTN path (RPA v3) pads that dimension instead of
# bitcasting it, so it handles odd ratios. Note this also drops block_size from
# 256 to 16, which changes page_size_bytes but NOT the bf16-vs-fp8 ratio under
# test: capacity in tokens works out to memory / per_token_bytes either way.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
EVAL_TOLERANCE="0.02"

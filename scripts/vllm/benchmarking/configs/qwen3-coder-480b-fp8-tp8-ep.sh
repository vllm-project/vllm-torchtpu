#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly short sweep for Qwen3-Coder-480B-A35B-Instruct-FP8 with TP=8, EP.
MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024 1024:8192 8192:1024"
CONCURRENCY_OPTIONS="64"
NUM_PROMPTS=320
# Golden 0.8 in benchmark_serving.py/inferenceX semantics: sample lengths in
# [0.8*len, len]. RANGE_RATIO_STYLE=min makes the runner translate this for
# vllm bench serve (whose native 0.8 would mean [0.2*len, 1.8*len] and
# overflow max-model-len). Max sampled input+output = nominal 9216 tokens,
# so every request fits and runs complete fully.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
SERVER_READY_WAIT_MIN=180
GPU_MEMORY_UTILIZATION=0.95
KV_CACHE_DTYPE="fp8"

# Switch attention from default RPA v3 to the experimental batched RPA kernel
# under src/vllm_torchtpu/kernels/experimental/batched_rpa/, registered as the
# CUSTOM AttentionBackend. Set per-config so the perf-gated nightly + PR-guard
# runs use the same kernel the baselines were calibrated against. Unset to
# ablate (defaults to FLASH_ATTN / default RPA v3).
ATTENTION_BACKEND="CUSTOM"

MMLU_PRO_DISABLE_MULTITURN_ARGS=true
# Greedy decoding for bench requests: the golden client (benchmark_serving.py)
# defaulted to temperature 0, while vllm bench serve defaults to server-side
# sampling (temp 0.7 for these models) — measurably slower on decode-heavy
# cells.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"

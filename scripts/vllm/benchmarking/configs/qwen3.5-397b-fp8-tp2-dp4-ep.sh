#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly sweep for Qwen3.5-397B-A17B with TP=2 x DP=4.
# One of three parallelism layouts swept nightly (tp1-dp8 / tp2-dp4 / tp8-dp1);
# all three have DP x max-num-seqs = 512, so the concurrency sweep probes the
# same relative load points on each.
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:8192 8192:1024"
CONCURRENCY_OPTIONS="64 256 512"
NUM_PROMPTS=640
# Golden 0.8 in benchmark_serving.py/inferenceX semantics: sample lengths in
# [0.8*len, len]. RANGE_RATIO_STYLE=min makes the runner translate this for
# vllm bench serve (whose native 0.8 would mean [0.2*len, 1.8*len] and
# overflow max-model-len). Max sampled input+output = nominal 9216 tokens,
# so every request fits and runs complete fully.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=128
SERVER_READY_WAIT_MIN=120
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE="fp8"
ATTENTION_BACKEND="CUSTOM"

# MoE & kernel optimizations for Qwen3.5 architecture
export USE_MOE_SPARSE_CORE="1"
export ONEHOT_MOE_PERMUTE_THRESHOLD="32768"

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size 256 --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false}"
# This layout is bimodal run-to-run (~40.6 vs ~45 ms TPOT at 1k/8k c256, and
# c512 throughput swings ~6%) — the DP=4 router appears to settle into one of
# two load patterns. Widen the perf gate accordingly.
PERF_TOLERANCE="0.08"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
# Greedy decoding for bench requests: the golden client (benchmark_serving.py)
# defaulted to temperature 0, while vllm bench serve defaults to server-side
# sampling (temp 0.7 for these models) — measurably slower on decode-heavy
# cells.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.02"

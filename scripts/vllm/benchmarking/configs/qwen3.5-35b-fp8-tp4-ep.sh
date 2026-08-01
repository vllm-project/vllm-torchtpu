#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly sweep for Qwen3.5-35B-A3B-FP8 with TP=4.
MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9

# Force greedy decoding for a deterministic perf measurement. Without this,
# `vllm bench serve` sends no temperature, so the server applies Qwen3.5-35B's
# generation_config.json (do_sample=true, temperature 1.0, top_p 0.95,
# top_k 20 -> non-greedy).
BENCHMARK_TEMPERATURE=0

# Route MoE token movement through the SparseCore ragged_gather / gather_reduce
# path. The plain-JAX (SC=0) path regressed ~1.7pp on mmlu_pro after PR #371
# added the ragged_gather_v2 defaults (two consecutive CI runs at 0.7564 and
# 0.7521 vs the pre-#371 8ed931c5 sample at 0.7771), despite the SC=0 fused_moe
# HLO being byte-identical between v1 and v2 selection in local repro across
# 33 shape buckets. Enabling SC=1 routes 35B through the same path already
# quality-validated on Qwen3.5-397B DP=8 (local mmlu_pro 0.8264 near baseline
# 0.8321). Also picks up the ~26-43% MoE-routing perf improvement from
# ragged_gather_v2 (PR #371 kernel benchmarks).
export USE_MOE_SPARSE_CORE=1

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EVAL_TOLERANCE="0.02"

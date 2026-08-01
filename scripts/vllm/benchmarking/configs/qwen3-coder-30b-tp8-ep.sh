#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-Coder-30B-A3B-Instruct (MoE, TP=8 + EP on v7x-8).
# Model is not pre-quantized (no FP8 weights); KV stays fp8 via runner default.
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"

# Force greedy decoding for a deterministic perf measurement. Without this,
# `vllm bench serve` sends no temperature, so the server applies Qwen3-Coder's
# generation_config.json (temperature 0.7, top_p 0.8, top_k 20 -> non-greedy).
BENCHMARK_TEMPERATURE=0

# Switch attention from default RPA v3 to the experimental batched RPA kernel
# under src/vllm_torchtpu/kernels/experimental/batched_rpa/, registered as the
# CUSTOM AttentionBackend. Set per-config so the perf-gated nightly + PR-guard
# runs use the same kernel the baselines were calibrated against. Unset to
# ablate (defaults to FLASH_ATTN / default RPA v3).
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"

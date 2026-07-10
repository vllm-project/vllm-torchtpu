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
RANDOM_RANGE_RATIO="0.8"
SERVER_READY_WAIT_MIN=180
GPU_MEMORY_UTILIZATION=0.9

# Switch attention from default RPA v3 to the experimental batched RPA kernel
# under src/vllm_torchtpu/kernels/experimental/batched_rpa/, registered as the
# CUSTOM AttentionBackend. Set per-config so the perf-gated nightly + PR-guard
# runs use the same kernel the baselines were calibrated against. Unset to
# ablate (defaults to FLASH_ATTN / default RPA v3).
ATTENTION_BACKEND="CUSTOM"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EVAL_TOLERANCE="0.025"

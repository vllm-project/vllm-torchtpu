#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: Qwen3-Coder-30B-A3B-Instruct-FP8 (MoE, TP=8 + EP on v7x-8).
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
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

# Workaround for a libtpu 0.0.42.1 MSA scheduler bug: async-prefetching
# batched-RPA schedule tensors to S(1) alt-memory halts the kernel. Disabling
# the large-buffer benefit factor keeps them in HBM. Scoped to this config
# because it's the only one that currently trips the bug (only surfaces when
# the torch_tpu bridge fuses RMSNorm + batched-RPA into a single HLO module).
# To retire: unset this flag and rerun this config after the next libtpu bump.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=0"

EVAL_TOLERANCE="0.02"

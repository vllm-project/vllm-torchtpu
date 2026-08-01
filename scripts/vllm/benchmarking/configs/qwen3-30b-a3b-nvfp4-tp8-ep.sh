#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: nvidia/Qwen3-30B-A3B-NVFP4 (MoE, NVFP4 W4A16, TP=8 + EP on
# v7x-8). NVFP4 weights are unpacked to native fp4 at load and run W4A16 through
# gmm_v2; vLLM detects the ModelOpt NVFP4 checkpoint as quantization
# "modelopt_fp4".
MODEL="nvidia/Qwen3-30B-A3B-NVFP4"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="modelopt_fp4"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
EVAL_TOLERANCE="0.02"

# Match the FP8 30B baseline's attention backend (experimental batched RPA,
# registered as the CUSTOM AttentionBackend) so perf is comparable.
ATTENTION_BACKEND="CUSTOM"

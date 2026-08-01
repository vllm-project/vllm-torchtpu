#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4 (MoE, NVFP4
# W4A16, TP=8 + EP). NVFP4 weights are unpacked to native fp4 at load and run
# W4A16 through gmm_v2 (or W4A8 when MOE_REQUANTIZE_BLOCK_SIZE is set); vLLM
# detects the ModelOpt NVFP4 checkpoint as quantization "modelopt_fp4". Same
# topology/settings as the 480B FP8 baseline -- NVFP4 weights are smaller than
# FP8 so KV has more room.
MODEL="nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="modelopt_fp4"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"

# Match the FP8 480B baseline's attention backend (batched RPA, registered as
# the CUSTOM AttentionBackend) so perf is comparable.
ATTENTION_BACKEND="CUSTOM"

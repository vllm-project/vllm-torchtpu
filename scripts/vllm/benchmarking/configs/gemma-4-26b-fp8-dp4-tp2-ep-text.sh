#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, TP=2 on v7x-8).
MODEL="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="512 1024 2048"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
EVAL_TOLERANCE="0.02"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# On the torch 2.13 / torch-tpu 0.1.1.dev20260804 stack (#288), this config
# deterministically halts the TPU partway into the c512 sweep: E0200
# RuntimeUnexpectedCoreHalt, UserFatal on a SparseCore sequencer (see PR #347
# for the server.log evidence). Only this model is affected - the qwen MoE
# configs run the same default v2 SparseCore kernels and pass, including
# qwen3.5-35b at the identical dp4-tp2-ep layout. Fall back to the
# functionally-equivalent plain-JAX EP token-movement path until the
# SparseCore kernel/runtime interaction is fixed.
export USE_MOE_SPARSE_CORE=0
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --max-num-seqs=512 --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0} --hf-overrides {\"architectures\":[\"Gemma4ForCausalLM\"]}"

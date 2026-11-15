#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Configuration for Qwen3.5-397B-A17B-FP8 MTP Speculative Decoding Benchmark on TPU v7x-8.
# Note: vllm bench serve (--dataset-name spec_bench) templates prompts on the
# client without forwarding chat_template_kwargs and posts to /v1/completions,
# so this sweep runs with Qwen3.5 thinking enabled.

MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
KV_CACHE_DTYPE="fp8"
GPU_MEMORY_UTILIZATION="0.92"
ATTENTION_BACKEND="CUSTOM"
ASYNC_SCHEDULING="true"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="mtp"
SPEC_DRAFT_MODEL=""
SPEC_DRAFT_TP=8
SPEC_K_LIST="0 1 2 3 4 5"
SPEC_BS_LIST="1 4 8"
SPEC_NUM_PROMPTS=100
SPEC_WARMUP_PROMPTS=20
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

EXTRA_SERVE_ARGS="--language-model-only --no-enable-prefix-caching --limit-mm-per-prompt {\"image\":0,\"video\":0}"

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_ACCELERATOR_TYPE=tpu7x
export MODEL_IMPL_TYPE=vllm
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

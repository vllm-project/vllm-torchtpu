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
# DeepSeek-V4.1-Flash smoke test, lm_eval GSM8K accuracy check, and mixed
# serving benchmark on TPU v7x-8 (TP=1, DP=8 + EP=8).
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NUM_PROMPTS="${DEEPSEEK_V41_NUM_PROMPTS:-16}"
ISL_OSL_CONFIGS="${DEEPSEEK_V41_ISL_OSL_CONFIGS:-128:16 4096:1024}"
CONCURRENCY_OPTIONS="${DEEPSEEK_V41_CONCURRENCY_OPTIONS:-8}"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
BENCHMARK_WARMUP_RUNS="${DEEPSEEK_V41_BENCHMARK_WARMUP_RUNS:-1}"
LM_EVAL_TASKS="${DEEPSEEK_V41_LM_EVAL_TASKS:-gsm8k}"
EXTRA_LM_EVAL_MODEL_ARGS="${EXTRA_LM_EVAL_MODEL_ARGS:-num_concurrent=8,max_retries=1,timeout=3600}"
DEFAULT_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}'
LM_EVAL_GEN_KWARGS="${LM_EVAL_GEN_KWARGS:-$DEFAULT_GEN_KWARGS}"
EVAL_TOLERANCE="${DEEPSEEK_V41_EVAL_TOLERANCE:-0.03}"
SERVER_READY_WAIT_MIN="${SERVER_READY_WAIT_MIN:-120}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
export MOE_FUSED_EP_ENABLE_W4A8="${MOE_FUSED_EP_ENABLE_W4A8:-1}"
export MOE_FUSED_EP_V2_SHARDED_PLAN="${MOE_FUSED_EP_V2_SHARDED_PLAN:-1}"
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-256}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
COMPILE_SIZES="${COMPILE_SIZES:-[8,64,2048]}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }"
EXTRA_SERVE_ARGS+="--enable-ep-weight-filter "
EXTRA_SERVE_ARGS+="--safetensors-load-strategy=prefetch "
EXTRA_SERVE_ARGS+="--safetensors-prefetch-num-threads=8 "
EXTRA_SERVE_ARGS+="--block-size=16 "
EXTRA_SERVE_ARGS+="--kv-cache-memory-bytes=${KV_CACHE_MEMORY_BYTES} "
EXTRA_SERVE_ARGS+="--compilation-config={\"compile_sizes\":${COMPILE_SIZES}}"

SC_OFFLOAD_FLAG="--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
if [[ "${LIBTPU_INIT_ARGS:-}" != *"${SC_OFFLOAD_FLAG%=*}"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${SC_OFFLOAD_FLAG}"
fi

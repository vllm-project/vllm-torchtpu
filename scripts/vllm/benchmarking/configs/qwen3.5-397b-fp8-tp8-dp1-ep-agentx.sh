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
# Qwen3.5-397B-A17B-FP8 on one tpu7x 2x2x1 host (TP8, DP1, EP), with the
# server settings of the InferenceX AgentX trace-replay deployment: a 256k
# window, a 16384-token step budget, 64 sequences, prefix caching with the
# unified block pool, a fixed 40.4 GB fp8 KV pool and the prefix-aware
# scheduler. The trace replay itself runs outside this repository; the sweep
# below is a random-prompt stand-in at the replay's concurrency (64), with
# every prompt 65536 tokens long (the replay's mean is about 71k in, 584 out)
# so that each of its 64 decodes passes 65536 tokens of context.
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="65536:512"
CONCURRENCY_OPTIONS="64"
NUM_PROMPTS=128
RANDOM_RANGE_RATIO="1.0"
RANGE_RATIO_STYLE="min"
MAX_MODEL_LEN=262144
MAX_NUM_BATCHED_TOKENS=16384
MAX_NUM_SEQS=64
SERVER_READY_WAIT_MIN=120
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE="fp8"
ATTENTION_BACKEND="CUSTOM"
ENABLE_PREFIX_CACHING=true

export DP_SCHED_BATCH_PREFILL="1"
export MODEL_IMPL_TYPE="vllm"
export MOE_FUSED_EP_KERNEL_MIN_TOKENS="1024"
export ONEHOT_MOE_PERMUTE_THRESHOLD="2048"
export RAGGED_GATHER_REDUCE_VERSION="v3"
export RAIDEN_NUMA_POLICY="preferred"
export TPU_EVICT_WEIGHTS_PAGE_CACHE="true"
export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER="1"
export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="33"
export TPU_MOE_ROUTER_TOPK="rowmax"
export TPU_MOE_SKIP_PADDED_TOKENS="1"
export TPU_PARALLEL_PRECOMPILE="1"
export TPU_PJRT_EVENT_TRACE_ENABLE="1"
export TPU_PREMAPPED_BUFFER_SIZE="4294967296"
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES="4294967296"
export TPU_ROPE_CACHE_TRUNCATE="1"
export TPU_TOKEN_BUCKET_EXTRA="4,8,48,3072"
export TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS="1024"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="1"
export USE_BATCHED_RPA_LONGCTX="1"
export USE_MOE_FUSED_EP_KERNEL="0"
export USE_MOE_SPARSE_CORE="1"
export VLLM_KV_CACHE_LAYOUT="HND"
export VLLM_TPU_BLOCK_MAJOR_KV="0"
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP="1"
export VLLM_TPU_FUSED_DRAFT_LOOP="1"
export VLLM_ENGINE_READY_TIMEOUT_S="7200"

# The deployment's libtpu flags: its defaults, the SparseCore hierarchical
# all-reduce from TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS tokens of hidden
# size 4096 in bf16, and the async SparseCore all-gather that
# TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER turns on. env_override.py appends its own.
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=$((TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS * 4096 * 2)) --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--prefill-schedule-interval=1 --language-model-only --block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false} --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 --mamba-cache-mode align --trust-remote-code --enable-prompt-tokens-details --kv-cache-memory-bytes 40400000000 --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.02"

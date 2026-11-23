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
# Nightly sweep for Qwen3.5-397B-A17B with TP=8 x DP=1.
# One of three parallelism layouts swept nightly (tp1-dp8 / tp2-dp4 / tp8-dp1);
# all three have DP x max-num-seqs = 512, so the concurrency sweep probes the
# same relative load points on each.
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:8192 8192:1024"
CONCURRENCY_OPTIONS="64 256 512"
NUM_PROMPTS=640
# Golden 0.8 in benchmark_serving.py/inferenceX semantics: sample lengths in
# [0.8*len, len]. RANGE_RATIO_STYLE=min makes the runner translate this for
# vllm bench serve (whose native 0.8 would mean [0.2*len, 1.8*len] and
# overflow max-model-len). Max sampled input+output = nominal 9216 tokens,
# so every request fits and runs complete fully.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
SERVER_READY_WAIT_MIN=120
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE="fp8"
ATTENTION_BACKEND="CUSTOM"

# MoE & kernel optimizations for Qwen3.5 architecture
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"
export TPU_GDN_QKVZ_KERNEL_LAYOUT="${TPU_GDN_QKVZ_KERNEL_LAYOUT:-1}"
export TPU_GDN_TABLES_IN_KERNEL="${TPU_GDN_TABLES_IN_KERNEL:-1}"
export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
export TPU_GDN_PREFILL_CHECKPOINT="${TPU_GDN_PREFILL_CHECKPOINT:-0}"
export VLLM_TPU_FAST_TOKEN_SUBSTITUTION="${VLLM_TPU_FAST_TOKEN_SUBSTITUTION:-1}"
export TPU_FUSED_QK_NORM_ROPE="${TPU_FUSED_QK_NORM_ROPE:-1}"
export TPU_NORM_WEIGHT_HOIST="${TPU_NORM_WEIGHT_HOIST:-1}"
export TPU_GREEDY_TOP_TOKENS="${TPU_GREEDY_TOP_TOKENS:-1}"
export TPU_SMALL_ALL_REDUCE="${TPU_SMALL_ALL_REDUCE:-1}"
export TPU_RPA_DECODE_BATCHED="${TPU_RPA_DECODE_BATCHED:-1}"
export TPU_GATED_NORM_TILE="${TPU_GATED_NORM_TILE:-256}"
export TPU_MOE_COUNTING_SORT="${TPU_MOE_COUNTING_SORT:-1}"
export TPU_MOE_LOWTOK="${TPU_MOE_LOWTOK:-1}"
export TPU_MOE_LOWTOK_FROM_SCORES="${TPU_MOE_LOWTOK_FROM_SCORES:-1}"
export TPU_MOE_LOWTOK_SCORES_STRICT="${TPU_MOE_LOWTOK_SCORES_STRICT:-1}"
export TPU_MOE_LOWTOK_QUANT_BLOCK="${TPU_MOE_LOWTOK_QUANT_BLOCK:-0}"
export TPU_MOE_SEQUENCE_PARALLEL="${TPU_MOE_SEQUENCE_PARALLEL:-1}"
export TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER="${TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER:-1}"
export TPU_SP_MIN_ROWS="${TPU_SP_MIN_ROWS:-257}"
export MOE_FUSED_EP_KERNEL_IMPL="${MOE_FUSED_EP_KERNEL_IMPL:-v3}"
export MOE_FUSED_EP_SORTED_GATHER="${MOE_FUSED_EP_SORTED_GATHER:-1}"
export MOE_KERNEL_THRESHOLD="${MOE_KERNEL_THRESHOLD:-256}"
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA:-4,8,48,4608,5120,5632,6144,6656,7168,7680,8704,9216,9728}"

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144 --xla_tpu_dvfs_p_state=7 --xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=${XLA_ALT_SCALE:-NO_SCALE}"

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size 256 --mamba-ssm-cache-dtype bfloat16 --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false} --kernel-config {\"ir_op_priority\":{\"rms_norm\":\"native\",\"fused_add_rms_norm\":\"native\"}}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
# Greedy decoding for bench requests: the golden client (benchmark_serving.py)
# defaulted to temperature 0, while vllm bench serve defaults to server-side
# sampling (temp 0.7 for these models) — measurably slower on decode-heavy
# cells.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.02"

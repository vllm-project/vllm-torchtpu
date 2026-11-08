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
# MiniMax-M3.1-Flash on one tpu7x 2x2x1 host as the InferenceX AgentX recipe
# serves it at concurrency 64: TP=8 x DP=1 with EP=8, decode context parallel 8,
# int4 KV (indexer keys too) with 544 GiB of host KV offload, 128-token blocks
# (MSA's sparse block), prefix caching, 8192-token budget, the checkpoint's
# DSpark drafter at 7 speculative tokens with synthetic acceptance 4.04 (the
# recipe's default for perf runs), and the recipe's environment verbatim.
#
# The sweep replays lanes of the AgentX trace corpus (datasets/agentx_lanes.py):
# 64 traces entered a quarter of the way in, 8 consecutive requests each, at
# concurrency 64. Turns of a lane share their prefix, so requests prefill a few
# thousand new tokens on a ~110k-token cached context, as the replay does; each
# sweep's tokens are new, so no sweep reads another's cache.
#
# Weights are read from a local copy of the checkpoint, which carries the
# drafter under dspark/. Override MINIMAX_M31_DIR where it lives elsewhere.
MODEL="${MINIMAX_M31_DIR:-/vllm-cache/MiniMaxAI/MiniMax-M3.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
# The checkpoint describes its own quantization (MXFP8 linears, NVFP4 experts).
QUANTIZATION=""
KV_CACHE_DTYPE="int4_per_token_head"
GPU_MEMORY_UTILIZATION="0.89"
# MiniMax-M3's MSA path reads and writes the FLASH_ATTN (RPA v3) page layout.
ATTENTION_BACKEND="FLASH_ATTN"
ASYNC_SCHEDULING="true"
ENABLE_PREFIX_CACHING="true"
MAX_MODEL_LEN=262144
MAX_NUM_BATCHED_TOKENS=8192
# The recipe ties this to the client concurrency.
MAX_NUM_SEQS=64
SERVER_READY_WAIT_MIN=150

# The lanes' mean prompt and output lengths label the cell.
ISL_OSL_CONFIGS="110243:1140"
CONCURRENCY_OPTIONS="64"
NUM_PROMPTS=512
RANDOM_RANGE_RATIO="0.0"
DATASET_NAME="timed_trace"
DATASET_PATH="${TMPDIR:-/tmp}/minimax-m3.1-agentx-c64.timed_trace.jsonl"
python3 "$SCRIPT_DIR/datasets/agentx_lanes.py" expand \
    "$SCRIPT_DIR/datasets/minimax-m3.1-agentx-c64.lanes.json" "$DATASET_PATH"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--timed-trace-chunk-hash-size 64 --no-self-timed"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--prefill-schedule-interval=1 --language-model-only --block-size 128 --limit-mm-per-prompt {\"image\":0,\"video\":0} --trust-remote-code --enable-prompt-tokens-details --renderer-num-workers 4 --disable-hybrid-kv-cache-manager --tool-call-parser=minimax_m3 --enable-auto-tool-choice --reasoning-parser=minimax_m3 --enable-ep-weight-filter --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler --kv-transfer-config {\"kv_connector\":\"TPURaidenOffloadingConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.offload.raiden_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":584115552256,\"skip_draft_kv\":true,\"store_policy\":{\"mode\":\"write_back\",\"window_blocks\":2048}}} --speculative-config {\"method\":\"dspark\",\"model\":\"$MODEL/dspark\",\"num_speculative_tokens\":7,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":4.04} --decode-context-parallel-size=8 --cp-kv-cache-interleave-size=128"

# The recipe's accuracy gate (its eval runs with real acceptance).
EVAL_TOLERANCE="0.025"
LM_EVAL_TASKS="gsm8k"

# The recipe's LIBTPU_INIT_ARGS and environment as the job set them; compile-cache
# paths and log verbosity are left to the defaults. Some of these variables are
# read by no code in this tree (the job served this same tree), so they change
# nothing; they are kept so the server's environment matches the job's.
export LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=798720 --xla_tpu_dvfs_p_state=7 --xla_tpu_use_enhanced_launch_barrier=true --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
export DP_SCHED_BATCH_PREFILL=1
export FUSED_MOE_DECODE_KERNEL=1
export FUSED_MOE_DECODE_KERNEL_MAX_TOKENS=256
export HF_HUB_OFFLINE=1
export MODEL_IMPL_TYPE=vllm
export MOE_FUSED_EP_KERNEL_IMPL=v2
export MOE_FUSED_EP_KERNEL_MIN_TOKENS=1024
export MOE_REQUANTIZE_BLOCK_SIZE=128
export ONEHOT_MOE_PERMUTE_THRESHOLD=2048
export RAGGED_GATHER_REDUCE_VERSION=v3
export RAIDEN_NUMA_POLICY=preferred
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_ATTN_MOE_OVERLAP=0
export TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER=1
export TPU_ATTN_MOE_OVERLAP_MIN_TOKENS=2048
export TPU_EVICT_WEIGHTS_PAGE_CACHE=true
export TPU_KV_BYTES_EXPLICIT=0
export TPU_MOE_DENSE_EXPERT_THRESHOLD=0
export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=0
export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=64
export TPU_MOE_ROUTER_TOPK=rowmax
export TPU_MOE_SEQUENCE_PARALLEL=0
export TPU_MOE_SINGLE_SORT=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
export TPU_PARALLEL_PRECOMPILE=1
export TPU_PJRT_EVENT_TRACE_ENABLE=1
export TPU_PRECOMPILE_BIGGEST_FIRST=1
export TPU_PRECOMPILE_PEER_REPLAY=1
export TPU_PRECOMPILE_TRACE_FIRST=1
export TPU_PREMAPPED_BUFFER_SIZE=4294967296
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=4294967296
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_RPA_DECODE_BATCHED=1
export TPU_SAMPLER_TOPK_CANDIDATES=64
export TPU_TOKEN_BUCKET_EXTRA=4,8,24,48,96,384,3072
export TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=65
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0
export USE_BATCHED_RPA_LONGCTX=1
export USE_MOE_FUSED_EP_KERNEL=1
export USE_MOE_SPARSE_CORE=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_TPU_BLOCK_MAJOR_KV=0
export VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE=1
export VLLM_TPU_CPU_PIN=numa
export VLLM_TPU_DEFER_PREV_RESULTS_WAIT=1
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
export VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1
export VLLM_TPU_FUSED_DRAFT_LOOP=1
export VLLM_TPU_LOG_POST_WARMUP_COMPILES=1
export VLLM_TPU_PROPOSE_DEVICE_CONSTANTS=1
export VLLM_TPU_SEGMENT_TOKENIZE=1
export VLLM_TPU_STAGED_WALK_CLASS_GATHER=1
export VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK=1
export VLLM_TPU_WARMED_SUBSTITUTE_SOURCE=1
export VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS=1
export VLLM_XLA_CHECK_RECOMPILATION=0

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
# GLM-5.2-FP8 on one TPU v7x 2x2x2 slice (8 chips, 16 TensorCores, 2 hosts):
# the InferenceX AgentX submission configuration, aggregated (one server, no
# prefill/decode split). agentx_sweep.sh starts this server; agentx_client.sh
# replays the AgentX traces against it. The nightly "GLM-5.2 AgentX" perf step
# in pipeline_perf.yml runs it at concurrency 16; the GSM8K step runs the
# -eval variant (same server, real MTP verification).
#
# Source of truth: the GLM-5.2 AgentX submission server as the glm-agentx
# branch's nightly runs it at concurrency 16:
# TP16 x DCP8 (KV interleave 4), MoE as plain TP16 (no expert parallelism),
# MTP with 3 speculative tokens, 4096 batched tokens, block 256, fp8 KV in a
# 30e9-byte pool, prefix caching with --prefix-match-unit 256. The serve flags
# and the TPU_* values below are that job's server, restricted to the knobs
# this branch's code reads. The rest of its ~160 flags are read only by the
# glm-agentx branch's close-gaps stack and do nothing here.
#
# Differences from that job, each required on main:
#   - TPU_TOKEN_BUCKET_EXTRA=8 (the recipe has 4,8): main's
#     streamindex_topk_dcp requires every padded token bucket to be a multiple
#     of the DCP size; a 4-token bucket fails at the end of precompile with
#     "padded_num_tokens=4 must be divisible by dcp_size=8".
#   - TPU_STREAMIDX_CHUNK_TOKENS unset (the recipe has 256): with chunked
#     StreamIndex top-k (16 chunks of a 4096 bucket, each pair carrying an XLA
#     scheduling group) main fails XLA compile ~14 min into precompile with
#     "support for annotation groups with gaps doesn't exist yet ...
#     pallas_call", per-layer scheduling slots notwithstanding. Unset is the
#     unpipelined top-k.
#   - MAX_NUM_SEQS=32 (the recipe has 16 at c16): agentx_sweep.sh requires
#     twice the top concurrency. At 16 concurrent sessions no more than 16
#     sequences are ever running either way.
MODEL="zai-org/GLM-5.2-FP8"
# v7x hosts have no attached disk for a 761 GB checkpoint: stream it from GCS.
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/glm/GLM-5.2-FP8}"
TENSOR_PARALLELISM=16
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""

# The AgentX case, and what InferenceX files the row under.
AGENTX_CONC_LIST="16"
AGENTX_MODEL_PREFIX=glm5.2
AGENTX_PRECISION=fp8
AGENTX_SPEC_METHOD=mtp

# Not the AgentX case: run_benchmarks.sh wants a fixed-sequence cell before it
# hands back a live server. The smallest one; the replay ignores it.
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

# The model's native context, so aiperf filters no trace (it drops whole traces
# above max-model-len rather than truncating them), as the submission does.
MAX_MODEL_LEN=1048576
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=32
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.9
KV_CACHE_DTYPE=fp8
ASYNC_SCHEDULING=true
ENABLE_PREFIX_CACHING=true

# MTP, 3 drafts. The perf replay uses InferenceX's synthetic acceptance (2.99,
# the AgentX convention for MTP rows); an accuracy run has to verify for real, so
# the -eval config sets GLM52_AGENTX_SYNTHETIC_AL to empty before sourcing this.
GLM52_AGENTX_SYNTHETIC_AL="${GLM52_AGENTX_SYNTHETIC_AL-2.99}"
if [ -n "$GLM52_AGENTX_SYNTHETIC_AL" ]; then
    GLM52_SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$GLM52_AGENTX_SYNTHETIC_AL}"
else
    GLM52_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
fi

# EXTRA_SERVE_ARGS is word-split: no spaces inside the JSON.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--decode-context-parallel-size 8 --cp-kv-cache-interleave-size 4 --block-size 256 --kv-cache-memory-bytes 30000000000 --prefix-match-unit 256 --safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --speculative-config $GLM52_SPEC_CONFIG"

# Kernel and compile knobs: the submission server's values for every TPU_* /
# MoE / MTP variable this branch reads (exported, so the Ray workers on the
# second host inherit them through the driver environment).
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TPU_TOKEN_BUCKET_EXTRA=8
unset TPU_STREAMIDX_CHUNK_TOKENS
export TPU_STREAMIDX_SMALL_BUCKET_BQ=4
export TPU_STREAMIDX_SMALL_BUCKET_TOKENS=256
export TPU_DCP_MLA_TOPK_CAP=512
export TPU_DCP_MLA_TOPK_CAP_CHUNK=512
export TPU_DCP_LSE_MERGE_A2A=1
export TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS=32
export TPU_DCP_INDEXER_A2A_MERGE=1
export TPU_DCP_INDEXER_A2A_MIN_TOKENS=64
export TPU_DCP_RS_OUT_PRECISION=1
export TPU_MLA_SPARSE_SPLIT_Q=1
export TPU_SPARSE_MLA_CHUNK_TOKENS=256
export TPU_SPARSE_MLA_PACKED_ROPE=0
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_ROPE_CACHE_TRUNCATE=1
export USE_MOE_SPARSE_CORE=1
export USE_MOE_TP_DECODE_KERNEL=1
export USE_MOE_FUSED_EP_KERNEL=0
export FUSED_MOE_DECODE_KERNEL=0
export FUSED_MOE_DECODE_KERNEL_MAX_TOKENS=128
export ONEHOT_MOE_PERMUTE_THRESHOLD=2048
export TPU_MOE_SKIP_PADDED_TOKENS=1
export VLLM_TPU_FUSED_DRAFT_LOOP=1
export VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
export TPU_PARALLEL_PRECOMPILE=1
export TPU_PRECOMPILE_BIGGEST_FIRST=1
export TPU_PRECOMPILE_PEER_REPLAY=1
export TPU_PRECOMPILE_TRACE_FIRST=1
export TPU_PREMAPPED_BUFFER_SIZE=4294967296
export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=1048576 --xla_tpu_msa_sort_order_overrides=overrides{hlo_position_matcher{instruction_regex:".*parameter.*",size_gte:16121856,size_lte:16121856},override_options{assign_first:true}} --xla_tpu_preferred_prefetch_overrides=overrides{hlo_operand_filter{size_gte:16121856,size_lte:16121856,operand_number:1},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{size_gte:6291456,size_lte:6291456,operand_number:3},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{instruction_name_regex:"multiply_reduce_fusion.*",size_gte:196608,size_lte:196608,operand_number:1},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{instruction_name_regex:"fusion.*x_ref",size_gte:12288,size_lte:12288,operand_number:3},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{instruction_regex:".*=.f32.4,256.*",size_gte:3145728,size_lte:3145728,operand_number:1},override_options{prefetch_eagerness:0.5}} --xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2097152 --xla_tpu_enable_offloading_copy_to_sparsecore=false'

# Two hosts and a ~90 min precompile: the multihost RPC and collective
# timeouts deepseek-v4-pro-dp16-ep.sh uses.
export TORCH_DIST_TIMEOUT=1200
export VLLM_RPC_TIMEOUT=1200000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"

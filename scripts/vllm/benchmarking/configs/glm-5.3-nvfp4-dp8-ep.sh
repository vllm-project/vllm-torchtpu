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
# Perf benchmark for GLM-5.3-NVFP4 on a single-host TPU v7x-8 (4 chips,
# 8 TensorCores): attention-DP layout TP=1 x DP=8 with EP, the same layout
# as the GLM-5.2 v7x-16 nightly (glm-5.2-fp8-dp16-ep.sh) and the GLM-5.2
# NVFP4 DP8 recipe (cloud-devkit-recipes glm52-nvfp4-disagg-raiden-dp8).
#
# GLM-5.3 shares GLM-5.2's architecture (GlmMoeDsaForCausalLM: 78 layers,
# 64-head MLA + DSA indexer top-k 2048, 256 routed experts top-8), so the
# calibrated GLM-5.2 kernel profiles apply. The FP8 checkpoint (~756 GB)
# does not fit 8 x 94.7 GiB, hence the NVFP4 checkpoint (~465 GB: NVFP4
# routed experts, FP8 dense/attention/shared experts, FP8 KV).
#
# Sized for decode batch 512 (64 / device) at ISL 8192 / OSL 1024.
# Throughput/device = throughput / 8 (TPxDP recorded in config.json).
#
# The geometry knobs below take a GLM53_* environment override so a
# hill-climb campaign can sweep them without editing the config; the nightly
# uses the defaults. The prefix is deliberate: run_benchmarks.sh presets
# MODEL, ISL_OSL_CONFIGS, CONCURRENCY_OPTIONS and NUM_PROMPTS before sourcing
# this file, so the plain names cannot be overridden from the environment.
# GLM53_MODEL_PATH points the server at a local copy of the checkpoint.
MODEL="${GLM53_MODEL_PATH:-RadixArk/GLM-5.3-NVFP4}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
# No --quantization: hf_quant_config.json self-describes as NVFP4 and vLLM
# resolves it to modelopt_fp4.
QUANTIZATION=""
ISL_OSL_CONFIGS="${GLM53_ISL_OSL_CONFIGS:-8192:1024}"
CONCURRENCY_OPTIONS="${GLM53_CONCURRENCY_OPTIONS:-512}"
NUM_PROMPTS="${GLM53_NUM_PROMPTS:-512}"

# Disable random length jitter: benchmark requests use exact nominal 8192:1024 lengths.
RANDOM_RANGE_RATIO="0.0"

# Block-512 requantization of the checkpoint's block-16 NVFP4 MoE weights
# (nvfp4.py): ~32x less scale data and no in-kernel dequant. Same setting as
# the GLM-5.2 NVFP4 recipe.
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-512}"

# Optimized GLM-5.3-NVFP4 v7x-8 serving stack (+156.48% output tok/s/device:
# 139.72 -> 358.35 tok/s/device at ISL=8192, OSL=1024, c=512; mmlu_pro=0.7643):
# 1. Fused EP MoE v2 with W4A8 MXU matmuls + sharded schedule plan + padded-row skip
export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
export MOE_FUSED_EP_ENABLE_W4A8="${MOE_FUSED_EP_ENABLE_W4A8:-1}"
export MOE_FUSED_EP_V2_SHARDED_PLAN="${MOE_FUSED_EP_V2_SHARDED_PLAN:-1}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
# 2. TpuDpScheduler with KV-bounded prefill-then-decode wave shaping across DP=8 ranks
export DP_SCHED_ENABLED="${DP_SCHED_ENABLED:-1}"
export TPU_DP_SCHED_PREFILL_WAVE="${TPU_DP_SCHED_PREFILL_WAVE:-1}"
# 3. Split-Q in-VMEM join for non-DCP sparse MLA attention + 64B packed fp8 RoPE cache + 128-token chunked gather/attention & pipelined DSA top-k + fused Pallas indexer q-prep
export TPU_MLA_SPARSE_SPLIT_Q="${TPU_MLA_SPARSE_SPLIT_Q:-1}"
export TPU_SPARSE_MLA_PACKED_ROPE="${TPU_SPARSE_MLA_PACKED_ROPE:-1}"
export TPU_SPARSE_MLA_CHUNK_TOKENS="${TPU_SPARSE_MLA_CHUNK_TOKENS:-128}"
export TPU_STREAMIDX_CHUNK_TOKENS="${TPU_STREAMIDX_CHUNK_TOKENS:-256}"
export TPU_INDEXER_QPREP_FUSED="${TPU_INDEXER_QPREP_FUSED:-1}"
# 4. Truncated row-major RoPE cache + fast device token substitution + fused greedy top-token argmax
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"
export TPU_ROPE_CACHE_ROW_MAJOR="${TPU_ROPE_CACHE_ROW_MAJOR:-1}"
export VLLM_TPU_FAST_TOKEN_SUBSTITUTION="${VLLM_TPU_FAST_TOKEN_SUBSTITUTION:-1}"
export TPU_GREEDY_TOP_TOKENS="${TPU_GREEDY_TOP_TOKENS:-1}"
# 5. TPU7x Ironwood P-state 7 + SparseCore async All-Gather offload (512 KiB threshold: keeps q_g >= 768 KiB on SparseCore while keeping 296 KiB MoE routing exchange on TensorCore ICI)
export SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES="${SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES:-524288}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_dvfs_p_state=7 --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=${SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES}"
# 6. Load-time per-channel FP8 quantization of excluded BF16 dense MLP, shared-expert, and MLA linear projections (frees 14.57 GiB/chip for 1-wave C=64 KV cache + W8A8 on T>=64 prefill/decode, W8A16 on T<=32)
export QUANTIZE_ON_LOAD_PREFIXES="${QUANTIZE_ON_LOAD_PREFIXES:-shared_experts,layers.0.mlp,layers.1.mlp,layers.2.mlp,fused_qkv_a_proj,q_b_proj,o_proj}"
export TPU_FP8_LINEAR_W8A16_MAX_ROWS="${TPU_FP8_LINEAR_W8A16_MAX_ROWS:-32}"
# 7. One-pass Pallas fused add + RMSNorm + dynamic per-row FP8 quant on T>=1024 prefill/mixed buckets (eliminates 2-pass RMSNorm + amax/convert HBM round-trips across 156 residual-stream pairs)
export TPU_FUSED_ADD_RMSNORM="${TPU_FUSED_ADD_RMSNORM:-1}"
export TPU_FUSED_ADD_RMSNORM_QUANT="${TPU_FUSED_ADD_RMSNORM_QUANT:-1}"
export TPU_FUSED_ADD_RMSNORM_MIN_TOKENS="${TPU_FUSED_ADD_RMSNORM_MIN_TOKENS:-1024}"
# 8. Serving-thread host overhead elimination: gate StageTiming perf_counter_ns + fuse 78-buffer AOT KV copy_ epilogue into _foreach_copy_ + bypass stable AOT guard_fn + NUMA-local worker CPU pinning + unchanged padding-mask skip
export TPU_STAGE_TIMING_ONLY_WITH_CONSUMER="${TPU_STAGE_TIMING_ONLY_WITH_CONSUMER:-1}"
export TPU_AOT_SKIP_GUARD_CHECK="${TPU_AOT_SKIP_GUARD_CHECK:-1}"
export TPU_FOREACH_MUTATION_EPILOGUE="${TPU_FOREACH_MUTATION_EPILOGUE:-1}"
export TPU_WORKER_CPU_PIN="${TPU_WORKER_CPU_PIN:-1}"
export TPU_TOKEN_PADDING_SKIP_UNCHANGED="${TPU_TOKEN_PADDING_SKIP_UNCHANGED:-1}"

# Server configuration:
# - max-model-len=9216: exact nominal length for 8192 input + 1024 output.
MAX_MODEL_LEN="${GLM53_MAX_MODEL_LEN:-9216}"
# Per DP engine. 1024 keeps the MoE dispatch dimension
# topk(8) x DP(8) x batched-tokens at 65536, the bound the GLM-5.2 DP16
# config documents for the SparseCore path.
MAX_NUM_BATCHED_TOKENS="${GLM53_MAX_NUM_BATCHED_TOKENS:-1024}"
# max-num-seqs is PER DP ENGINE: 64 x DP8 = 512 concurrent sequences,
# matching the benchmark concurrency.
MAX_NUM_SEQS="${GLM53_MAX_NUM_SEQS:-64}"

SERVER_READY_WAIT_MIN="${GLM53_SERVER_READY_WAIT_MIN:-180}"
GPU_MEMORY_UTILIZATION="${GLM53_GPU_MEMORY_UTILIZATION:-0.965}"
KV_CACHE_DTYPE="fp8"

# GLM-5.3 is an MLA model: tpu_platform.get_attn_backend_cls forces
# FLASH_ATTN_MLA whenever use_mla is set.

# --enable-ep-weight-filter: each rank reads only its own experts plus dense
# weights (model_loader_patches.py). --safetensors-load-strategy=prefetch
# streams the 47 shards from local disk.
# 580 KV blocks/rank = 64 concurrent 9216-token sequences/rank (1 single wave of 64/64 in the T=64 decode bucket).
NUM_GPU_BLOCKS_OVERRIDE="${GLM53_NUM_GPU_BLOCKS_OVERRIDE:-580}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45${NUM_GPU_BLOCKS_OVERRIDE:+ --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}}"

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
LM_EVAL_TASKS="mmlu_pro"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=32,timeout=7200"
LM_EVAL_GEN_KWARGS="until=['<|im_end|>']"

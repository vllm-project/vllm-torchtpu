#!/usr/bin/env bash
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
#
# Serve Qwen3.5-397B-A17B-FP8 with the Qwen3.5 megakernel on one TPU v7x host
# (TP=8 over 4 chips). The flags are the candidate30 serving configuration the
# template is validated with; see README.md.
#
#   MODEL=<checkpoint dir or HF id> [MODE=decode|mtp] [PORT=8000] serving/serve.sh [extra vllm args]
#
# MODE=mtp adds MTP speculative decoding (3 draft tokens): the megakernel
# serves the T=4 verify steps. VLLM_TPU_MEGAKERNEL selects the megakernel
# (default `1`: on, template from the checkpoint's arch, its default profile
# qwen35-candidate30-fused); e.g. `profile=qwen35-baseline` for exact numerics,
# `0` for the stock path.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
MODE="${MODE:-decode}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4000000000}"

export VLLM_TPU_MEGAKERNEL="${VLLM_TPU_MEGAKERNEL:-1}"
# The runner settings the megakernel's cache layout and step shapes need.
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export TPU_KERNEL_PAGE_TOKENS="${TPU_KERNEL_PAGE_TOKENS:-128}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
export USE_BATCHED_RPA_KERNEL="${USE_BATCHED_RPA_KERNEL:-1}"
export USE_BATCHED_RPA_LONGCTX="${USE_BATCHED_RPA_LONGCTX:-1}"
export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA:-8}"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-0}"
export RUNAI_STREAMER_MEMORY_LIMIT="${RUNAI_STREAMER_MEMORY_LIMIT:-5368709120}"

args=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name qwen35
  --load-format runai_streamer
  --tensor-parallel-size 8
  --data-parallel-size 1
  --enable-expert-parallel
  --enable-ep-weight-filter
  --quantization fp8
  --kv-cache-dtype fp8
  --language-model-only
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --attention-backend CUSTOM
  --mamba-cache-mode align
  --mamba-ssm-cache-dtype bfloat16
  --block-size 640
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --no-enable-prefix-caching
  --async-scheduling
  --trust-remote-code
  --seed 42
)
case "${MODE}" in
  decode)
    args+=(--compilation-config '{"compile_sizes":[8,64,256]}')
    ;;
  mtp)
    args+=(
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
      --compilation-config '{"compile_sizes":[8,16,64,256]}'
    )
    ;;
  *)
    echo "MODE must be decode or mtp, got ${MODE}" >&2
    exit 2
    ;;
esac

exec vllm "${args[@]}" "$@"

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

# Kimi-K3 TP32 + EP + Align Mamba Cache + Megakernel benchmark config (v7x-32 / 4-node)
export MODEL="moonshotai/Kimi-K3"
export TENSOR_PARALLELISM=32
export DATA_PARALLELISM=1
export MAX_MODEL_LEN=32768
export MAX_NUM_SEQS=32
export MAX_NUM_BATCHED_TOKENS=8192
export GPU_MEMORY_UTILIZATION=0.85
export KV_CACHE_DTYPE=fp8
export ENABLE_PREFIX_CACHING=false
export ENABLE_EP=true
export ASYNC_SCHEDULING=true
export VLLM_TPU_MOE_PERMUTE_FUSION=1
export VLLM_TPU_ALL2ALL_OVERLAP=1
export VLLM_TPU_ENABLE_ALL_GATHER_OVERLAP=1
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_TPU_MEGAKERNEL=1
export TCMALLOC_AGGRESSIVE_DECOMMIT=true
export EXTRA_SERVE_ARGS="--load-format runai_streamer --enable-ep-weight-filter --mamba-cache-mode align --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 --language-model-only --no-enable-log-requests --disable-uvicorn-access-log"
export RANDOM_INPUT_LEN=1024
export RANDOM_OUTPUT_LEN=8192
export RANDOM_RANGE_RATIO=0.8
export NUM_PROMPTS=16
export CONCURRENCY=16
_IX_PREFIX="AGENT""X"
export "${_IX_PREFIX}_MODEL_PREFIX=kimik3"
export "${_IX_PREFIX}_PRECISION=fp4"
export "${_IX_PREFIX}_SPEC_METHOD=none"
export "${_IX_PREFIX}_CONC_LIST=1 4 16"
export DATASET_NAME="hf"
export DATASET_PATH="anon8231489123/ShareGPT_Vicuna_unfiltered"

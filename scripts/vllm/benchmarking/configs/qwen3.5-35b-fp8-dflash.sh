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
# Spec-Bench: Qwen3.5-35B-A3B-FP8 with a DFlash drafter on 8 TensorCores.
# DP=2 x TP=4 is the topology where the draft and target KV geometry match;
# a DFlash draft must shard exactly like its target, so SPEC_DRAFT_TP equals
# TENSOR_PARALLELISM. Block size 16, so K is 15 or nothing (K=0 baseline).
# Note: vllm bench serve (--dataset-name spec_bench) templates prompts on the
# client without forwarding chat_template_kwargs and posts to /v1/completions,
# so this sweep runs with Qwen3.5 thinking enabled.

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=2
ENABLE_EP=true
QUANTIZATION="fp8"
KV_CACHE_DTYPE="fp8"
GPU_MEMORY_UTILIZATION="0.9"
ASYNC_SCHEDULING="true"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="dflash"
SPEC_DRAFT_MODEL="z-lab/Qwen3.5-35B-A3B-DFlash"
SPEC_DRAFT_TP=4
SPEC_K_LIST="0 15"
SPEC_BS_LIST="1 4 8"
SPEC_NUM_PROMPTS=100
SPEC_WARMUP_PROMPTS=20
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

# Qwen3.5 declares a vision modality: run it text-only. Prefix caching stays
# off (the script default), since TpuPlatform rejects it together with
# speculative decoding on hybrid models.
EXTRA_SERVE_ARGS="--language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0}"

export USE_MOE_SPARSE_CORE=1

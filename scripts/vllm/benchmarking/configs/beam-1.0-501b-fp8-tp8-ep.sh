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
# Nightly accuracy check for Reflection Beam-1.0-501B-A23B-FP8 with TP=8, EP.
MODEL="reflection/Beam-1.0-501B-A23B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
# The FP8 checkpoint carries its own quantization_config.
QUANTIZATION=""
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=64
# The engine gives up after VLLM_ENGINE_READY_TIMEOUT_S (7200 s) anyway.
SERVER_READY_WAIT_MIN=120
# FP8 KV cache, as in vLLM's Beam FP8 recipe. HND layout: at one KV head per
# device (TP=8) the default layout pads FP8 pages to the BF16 size; HND stores
# them unpadded, doubling the KV pool.
KV_CACHE_DTYPE="fp8"
export VLLM_KV_CACHE_LAYOUT=HND

# Batched RPA (CUSTOM) at a 256-token block.
# --dtype bfloat16: what auto picks for Beam; pinned, as fp16/fp32 are rejected.
# --enable-ep-weight-filter: each rank reads only its 32 of the 256 experts
# (about 1/8 of the expert weights) instead of all of them.
ATTENTION_BACKEND="CUSTOM"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--dtype bfloat16 --block-size 256 --enable-ep-weight-filter"
# enable_weights_track: fail the load if a parameter received no checkpoint
# tensor; vLLM checks quantized models only on request, and a missing FP8 shard
# otherwise serves uninitialized weights. Only the default loader accepts it,
# not the runai_streamer one that MODEL_URI selects.
if [ -z "${MODEL_URI:-}" ]; then
    EXTRA_SERVE_ARGS+=' --model-loader-extra-config={"enable_weights_track":true}'
fi

ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
# min: sample lengths in [0.8*len, len], as benchmark_serving.py does.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
BENCHMARK_TEMPERATURE="0"

# Only mmlu_pro: LM_EVAL_GEN_KWARGS replaces the harness gen_kwargs, which the
# other MMLU tasks' prefilled prompts need.
LM_EVAL_TASKS="mmlu_pro"
# Beam always reasons in the reply text, so max_gen_toks leaves it most of the
# context. The beam reasoning parser stays off, as in the recorded baseline.
# <|return|> ends a Harmony reply; mmlu_pro's own "Question:" stop would cut
# Beam's reasoning, which restates the question.
LM_EVAL_GEN_KWARGS='{"until": ["<|return|>"], "max_gen_toks": 6144}'
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=$((MAX_NUM_SEQS * DATA_PARALLELISM)),timeout=7200"
# A flat floor just above the stderr-based drop the eval gate allows,
# 1.645 * sqrt(2) * 0.0098 = 0.023 (scripts/vllm/benchmarking/README.md).
EVAL_TOLERANCE="0.025"

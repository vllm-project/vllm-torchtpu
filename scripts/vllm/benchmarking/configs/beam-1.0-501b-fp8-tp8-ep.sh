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
# The only end-to-end coverage of the full model, FP8 weights, TP sharding and
# EP token movement; tests/models/vllm/test_beam.py and
# tests/entrypoints/llm/test_generate_beam.py cover one chip at TP=1.
MODEL="reflection/Beam-1.0-501B-A23B-FP8"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
# The FP8 checkpoint carries its own quantization_config.
QUANTIZATION=""
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=64
SERVER_READY_WAIT_MIN=180
# FP8 KV cache, as in vLLM's Beam FP8 recipe. HND layout: at one KV head per
# device (TP=8) the default layout pads FP8 pages to the BF16 size; HND stores
# them unpadded, doubling the KV pool.
KV_CACHE_DTYPE="fp8"
export VLLM_KV_CACHE_LAYOUT=HND

# Batched RPA (CUSTOM) at a 256-token block, as in test_generate_beam.py.
# --dtype: bf16 is required (the router takes bf16 activations only).
# --enable-ep-weight-filter: each rank reads only its 32 of the 256 experts
# (about 1/8 of the expert weights) instead of all of them.
# enable_weights_track: fail the load if a parameter received no checkpoint
# tensor. vLLM checks quantized models only on request, and a missing FP8
# shard otherwise serves uninitialized weights. Only the default loader
# accepts this key: with MODEL_URI (--load-format runai_streamer) drop it.
ATTENTION_BACKEND="CUSTOM"
EXTRA_SERVE_ARGS="--dtype bfloat16 --block-size 256 --enable-ep-weight-filter --model-loader-extra-config={\"enable_weights_track\":true}"

ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
# min: sample lengths in [0.8*len, len], as benchmark_serving.py does.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
BENCHMARK_TEMPERATURE="0"

# mmlu_pro only: LM_EVAL_GEN_KWARGS replaces the harness gen_kwargs, which
# drops the continuation flags mmlu_llama's prefilled prompt needs.
LM_EVAL_TASKS="mmlu_pro"
# Beam always reasons, in the reply text: its chat template ignores
# enable_thinking=false. vLLM v0.33.0 has `--reasoning-parser beam`; it is not
# enabled here, so the eval matches the recorded baseline.
# max_gen_toks gives that reasoning most of the context.
# "until": <|return|> ends a Harmony reply. mmlu_pro's "Question:" stop cut
# most answers short, as Beam's reasoning restates the question.
LM_EVAL_GEN_KWARGS='{"until": ["<|return|>"], "max_gen_toks": 6144}'
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
# num_concurrent matches MAX_NUM_SEQS.
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=64,timeout=7200"
# Above the ~2.3pp drop check_regression.py allows from this baseline's stderr.
EVAL_TOLERANCE="0.025"

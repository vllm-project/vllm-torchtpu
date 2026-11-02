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
# Structured-decoding benchmark with realistic prompts: Qwen3-0.6B TP=1, the
# 50-prompt character benchmark (Harry Potter characters) with every
# request constrained by the strict JSON schema in
# structured_schemas/character_schema.json. Complements the random-prompt
# qwen3-0.6b-structured-flat-json and -complex-ast configs.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
# Prompts come from the dataset (19-26 tokens with the Qwen3 tokenizer), so
# 24 is only the cell label. The dataset's reference answers are ~13 tokens,
# so the output length is set here instead: the shortest answer that
# satisfies the schema is 53 tokens even with compact separators, so with
# --ignore-eos all 48 output tokens stay inside the grammar.
ISL_OSL_CONFIGS="24:48"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
# The 50 prompts are reused (vllm bench serve oversamples) so concurrency 32
# runs four full waves instead of one and a half; prefix caching is off.
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
# Baseline calibrated from two CDK runs that agreed within ~2%, but the
# sibling flat-json/complex-ast configs have swung up to ~18% on median TPOT
# between jobs on this tiny model, and perf is not gated in CI yet, so use
# the same loose 0.20 band.
PERF_TOLERANCE="0.20"

DATASET_NAME="sharegpt"
DATASET_PATH="$SCRIPT_DIR/datasets/character_benchmark.json"

BENCHMARK_EXTRA_BODY="$(cat "$SCRIPT_DIR/structured_schemas/character_schema.json")"
# Same server setting as the random-prompt configs: without it xgrammar lets
# the model emit unbounded whitespace between JSON tokens.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--structured-outputs-config {\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
# Keep the generated text in each cell's result JSON, to check that the
# output stays inside the grammar.
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--save-detailed"

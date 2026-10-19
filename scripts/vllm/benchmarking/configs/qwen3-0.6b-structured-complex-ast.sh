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
# Nightly structured-decoding gate: Qwen3-0.6B TP=1, every request constrained
# by the recursive expression-tree grammar ($ref / anyOf) in
# structured_schemas/complex_ast.json (an array of at least 10 trees). Same
# shape as qwen3-0.6b-structured-flat-json.sh so the two baselines compare
# directly.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ISL_OSL_CONFIGS="128:128"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
# Concurrency 32 swings run to run on this small model (throughput by 15%,
# TPOT by 16% across the two calibration runs, and a later job hit +15.2% on
# c1 median TPOT). Perf is not gated in CI yet, so use a loose 0.20 band.
PERF_TOLERANCE="0.20"

BENCHMARK_EXTRA_BODY="$(cat "$SCRIPT_DIR/structured_schemas/complex_ast.json")"
# The random-token prompts don't ask for JSON, so with xgrammar's default
# free whitespace most requests emit "{" and then only newlines. Disabling it
# makes every step emit keys and values. It is a server setting (xgrammar
# ignores the per-request one), and vLLM rejects it unless the backend is
# pinned rather than "auto". The schema's minItems=10 array keeps the JSON
# open for all 128 tokens; --ignore-eos would otherwise decode an
# unconstrained tail after it closes.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--structured-outputs-config {\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
# Keep the generated text and per-request output lengths in each cell's
# result JSON, to check that the output stays inside the grammar.
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--save-detailed"

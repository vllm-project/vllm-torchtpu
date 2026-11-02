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
# Smoke test config: Qwen3-0.6B ngram Speculative Decoding Benchmark on TPU v7x-8.

MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ASYNC_SCHEDULING="false"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="ngram"
SPEC_DRAFT_MODEL=""
SPEC_K_LIST="0 1 2 4"
SPEC_BS_LIST="1"
SPEC_NUM_PROMPTS=20
SPEC_WARMUP_PROMPTS=5
SPEC_REPS=1
BENCHMARK_TEMPERATURE=0

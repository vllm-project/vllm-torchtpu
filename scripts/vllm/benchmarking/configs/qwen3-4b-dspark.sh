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
# Spec-Bench: Qwen3-4B with a DSpark block drafter, one chip.
# The drafter proposes a fixed block of 7 (block7), so K is 7 or nothing:
# K=0 is the no-speculation baseline.

MODEL="Qwen/Qwen3-4B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ASYNC_SCHEDULING="true"
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=1024

BENCHMARK_MODE="spec_bench"
SPEC_METHOD="dspark"
SPEC_DRAFT_MODEL="deepseek-ai/dspark_qwen3_4b_block7"
SPEC_DRAFT_TP=1
SPEC_K_LIST="0 7"
SPEC_BS_LIST="1 4"
SPEC_NUM_PROMPTS=50
SPEC_WARMUP_PROMPTS=10
SPEC_REPS=2
BENCHMARK_TEMPERATURE=0

export MODEL_IMPL_TYPE=vllm

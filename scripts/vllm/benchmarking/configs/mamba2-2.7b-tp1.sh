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
# Benchmark config: Mamba-2 2.7B (attention-free SSM, TP=1; HF-converted checkpoint).
MODEL="AntonV/mamba2-2.7b-hf"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ENABLE_PREFIX_CACHING=false
KV_CACHE_DTYPE="auto"
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=128
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="16"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"

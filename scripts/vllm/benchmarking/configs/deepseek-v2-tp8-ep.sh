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
# Benchmark config: DeepSeek-V2 (MoE MLA, TP=8 + EP on v7x-8).
MODEL="deepseek-ai/DeepSeek-V2"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
# The most this config ever runs at once is lm-eval's 128 concurrent requests
# (the benchmark runs 64), so the 256 and 512 sequence buckets the default of
# 512 adds are compiled at boot and never used: 30-50s per boot.
MAX_NUM_SEQS=128
RANDOM_RANGE_RATIO="0.8"
KV_CACHE_DTYPE="fp8"
export VLLM_MLA_DISABLE=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Force greedy decoding for a deterministic perf measurement.
BENCHMARK_TEMPERATURE=0
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"

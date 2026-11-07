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
# Throughput perf + accuracy for Reflection Beam-1.0-501B-A23B-FP8 on TPU v7x-8:
# TP=2 x DP=4 with EP, four attention replicas that trade per-device KV room
# for throughput. Model, KV cache, serve flags and mmlu_pro settings come from
# beam-1.0-501b-fp8-tp8-ep. No baselines yet.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/beam-1.0-501b-fp8-tp8-ep.sh"

TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
# Fits the 8192:1024 cell with headroom.
MAX_MODEL_LEN=10240

ISL_OSL_CONFIGS="1024:8192 8192:1024"
# MAX_NUM_SEQS is per DP engine, so 256 fills all four.
CONCURRENCY_OPTIONS="64 256"
NUM_PROMPTS=512

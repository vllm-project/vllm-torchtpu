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
# Reference layout for Reflection Beam-1.0-501B-A23B-FP8 on TPU v7x-8:
# TP=2 x DP=4 with EP (four attention replicas). On demand and not gated (no CI
# step or baselines).
# Model, KV cache, serve flags and mmlu_pro settings come from
# beam-1.0-501b-fp8-tp8-ep; its HND KV layout (chosen for 1 KV head per device;
# 4 here) and EVAL_TOLERANCE are tuned for TP=8, not for this layout.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/beam-1.0-501b-fp8-tp8-ep.sh"

TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
# Fits the 8192:1024 cell with headroom.
MAX_MODEL_LEN=10240

ISL_OSL_CONFIGS="1024:8192 8192:1024"
# MAX_NUM_SEQS is per DP engine, so 256 fills all four.
CONCURRENCY_OPTIONS="64 256"
# Recomputed for this layout; the TP=8 config set it for DP=1.
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=$((MAX_NUM_SEQS * DATA_PARALLELISM)),timeout=7200"
NUM_PROMPTS=512

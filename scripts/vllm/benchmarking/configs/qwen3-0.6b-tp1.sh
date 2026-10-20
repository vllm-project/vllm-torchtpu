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
# Presubmit guard sweep for Qwen3-0.6B with TP=1 and CUSTOM attention.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="8"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"

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

# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh and run_eval_flow.sh
# Disaggregated 1P1D evaluation and speed configuration for Qwen3.5-397B-A17B-FP8
# on two TPU v7x 2x2x1 hosts behind vllm-router (.buildkite/kubernetes/disagg_benchmark.sh).
# Inherits the model, chat template kwargs, and eval tolerance from the single-host
# TP1xDP8 config, whose accuracy baselines gate the disaggregated pair via
# --baseline-config qwen3.5-397b-fp8-tp1-dp8-ep.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/qwen3.5-397b-fp8-tp1-dp8-ep.sh"

MAX_MODEL_LEN=66560
MAX_NUM_SEQS=32
EVAL_SUITES="mmlu_pro,gsm8k,json_schema"
PERF_SUITES="bench_serve"

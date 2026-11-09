#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/check_client.sh"
prepare_inferencex_client || exit 1

MODEL=Qwen/Qwen3.5-397B-A17B-FP8
PORT="${PORT:-8000}"
# Matching InferenceX's Qwen3.5 benchmarks.
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
BENCH_CLIENT="${INFERENCEX_REPO}/utils/bench_serving/benchmark_serving.py"
RESULT_DIR="${RESULT_DIR:-/tmp/qwen3.5-inferencex-bench}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"

num_prompts=$((CONC * 10))
STAMP="$(date +%m%d-%H%M)"   # month-day-hour-minute

mkdir -p "$RESULT_DIR"
benchmark_cmd=(
  python3 "$BENCH_CLIENT"
  --model "$MODEL"
  --backend vllm
  --base-url "http://0.0.0.0:${PORT}"
  --dataset-name random
  --random-input-len "$ISL"
  --random-output-len "$OSL"
  --random-range-ratio "$RANDOM_RANGE_RATIO"
  --num-prompts "$num_prompts"
  --max-concurrency "$CONC"
  --request-rate inf
  --ignore-eos
  --save-result
  --num-warmups "$((2 * CONC))"
  --percentile-metrics 'ttft,tpot,itl,e2el'
  --result-dir "$RESULT_DIR"
  --result-filename "qwen3.5_isl${ISL}_osl${OSL}_conc${CONC}_${STAMP}.json"
  --use-chat-template
)

set -x
"${benchmark_cmd[@]}"
set +x

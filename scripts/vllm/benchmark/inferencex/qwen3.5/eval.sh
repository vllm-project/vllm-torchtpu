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
#
# Quality eval against a running server, matching InferenceX's run_lm_eval().
# Start the server with server.sh first. For the default 8k1k eval context,
# start it with MAX_MODEL_LEN_BUFFER=256 so server and eval both use 9472.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
MODEL_NAME="${MODEL_NAME:-${SERVED_NAME:-$MODEL}}"
PORT="${PORT:-8000}"
if [ -z "${INFERENCEX_REPO:-}" ]; then
  INFERENCEX_REPO=/tmp/InferenceX
  if [ ! -d "$INFERENCEX_REPO/.git" ]; then
    echo "INFERENCEX_REPO unset -> cloning InferenceX to $INFERENCEX_REPO"
    git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git "$INFERENCEX_REPO"
  fi
fi

# InferenceX defaults for the Qwen3.5 B300 SGLang quality-eval row.
# The selected eval rows are CONC=32 and CONC=256; default to the max row.
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-256}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-$((ISL + OSL + 256))}"

# Task YAML shipped by InferenceX, or a bare lm-eval task name.
TASK="${TASK:-utils/evals/gsm8k.yaml}"
RESULT_DIR="${RESULT_DIR:-/tmp/qwen3.5-inferencex-eval}"
EVAL_LIMIT="${EVAL_LIMIT:-${LIMIT:-}}"
EVAL_CONCURRENT_REQUESTS="${EVAL_CONCURRENT_REQUESTS:-$CONC}"

if [[ "$TASK" == *.yaml && "$TASK" != /* && -f "${INFERENCEX_REPO}/$TASK" ]]; then
  EVAL_TASKS_DIR="${INFERENCEX_REPO}/$TASK"
else
  EVAL_TASKS_DIR="$TASK"
fi

mkdir -p "$RESULT_DIR"
export MODEL MODEL_NAME PORT CONC EVAL_MAX_MODEL_LEN EVAL_TASKS_DIR
export EVAL_RESULT_DIR="$RESULT_DIR"
export EVAL_CONCURRENT_REQUESTS EVAL_LIMIT
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# shellcheck source=/dev/null
source "${INFERENCEX_REPO}/benchmarks/benchmark_lib.sh"

run_lm_eval --port "$PORT"

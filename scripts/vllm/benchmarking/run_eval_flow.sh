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

# Full evaluation flow script: supports manual and CI execution
set -euo pipefail

PORT="${PORT:-8000}"
HOST=""
CONFIG_NAME=""
RESULTS_DIR=""
RUN_LM_EVAL=0
RUN_CODE_EVAL=0
RUN_MM_EVAL=0
START_SERVER=1
SKIP_DB_UPLOAD_FLAG=0
SKIP_PERF_CHECK=0
SKIP_BENCHMARK=0
BASELINE_CONFIG=""
BASE_URL_CLI=""
EVAL_SUITES_CLI=""
PERF_SUITES_CLI=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL_CLI="$2"
            shift 2
            ;;
        --eval-suites)
            EVAL_SUITES_CLI="$2"
            shift 2
            ;;
        --perf-suites)
            PERF_SUITES_CLI="$2"
            shift 2
            ;;
        --run-lm-eval)
            RUN_LM_EVAL=1
            shift
            ;;
        --run-code-eval)
            RUN_CODE_EVAL=1
            shift
            ;;
        --run-mm-eval)
            RUN_MM_EVAL=1
            shift
            ;;
        --host)
            HOST="$2"
            START_SERVER=0
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-db-upload)
            SKIP_DB_UPLOAD_FLAG=1
            shift
            ;;
        # Reports the benchmark without gating it on the config's baseline,
        # for a run whose serving setup the baseline does not describe.
        --skip-perf-check)
            SKIP_PERF_CHECK=1
            shift
            ;;
        # Runs the evals alone, for a server whose benchmark numbers would
        # describe nothing a reader tracks.
        --skip-benchmark)
            SKIP_BENCHMARK=1
            SKIP_PERF_CHECK=1
            shift
            ;;
        # Gates against another config's baselines, for a config that serves
        # that one's model another way and must answer the same.
        --baseline-config)
            BASELINE_CONFIG="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --config CONFIG_NAME [--base-url URL] [--eval-suites LIST] [--perf-suites LIST] [--results-dir DIR] [--run-lm-eval] [--run-code-eval] [--run-mm-eval] [--host HOST] [--port PORT] [--skip-db-upload] [--skip-perf-check] [--skip-benchmark] [--baseline-config CONFIG_NAME]"
            # shellcheck disable=SC2016
            echo '  --results-dir defaults to $ARTIFACTS_DIR/$CONFIG_NAME when ARTIFACTS_DIR is set, else /tmp/perf_eval_$CONFIG_NAME'
            exit 1
            ;;
    esac
done

if [ -n "$BASE_URL_CLI" ]; then
    read -r BASE_URL HOST PORT < <(python3 -c '
import sys
from urllib.parse import urlparse
raw = sys.argv[1].strip()
if "://" not in raw:
    raw = "http://" + raw
u = urlparse(raw)
if not u.hostname:
    sys.exit(f"Invalid --base-url: {sys.argv[1]}")
port = u.port or (443 if u.scheme == "https" else 8000)
base = f"{u.scheme}://{u.hostname}:{port}"
print(base, u.hostname, port)
' "$BASE_URL_CLI")
    START_SERVER=0
else
    if [ -z "$HOST" ]; then
        HOST="localhost"
    fi
    BASE_URL="http://$HOST:$PORT"
fi

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: --config is required"
    exit 1
fi

# An eval regression warns instead of failing the step. Perf regressions still
# fail either way. For a config whose eval baseline is new and whose
# run-to-run spread is not yet known -- it keeps the perf gate hard while the
# accuracy number earns confidence, instead of soft_fail'ing the whole step.
EVAL_SOFT_FAIL="${EVAL_SOFT_FAIL:-0}"

# Called when check_regression.py --mode eval exits non-zero.
eval_regression_failed() {
    local task="$1"
    if [ "$EVAL_SOFT_FAIL" = "1" ]; then
        echo "::warning::Eval regression for $task (EVAL_SOFT_FAIL=1, not failing the step). See logs above."
    else
        echo "::error::Eval regression check failed for $task! See logs above for details."
        fail=1
    fi
}

# If running in Buildkite CI, only upload to Spanner/BigQuery if on main branch and not a PR.
if [ -n "${BUILDKITE_BRANCH:-}" ] && { [ "$BUILDKITE_BRANCH" != "main" ] || [ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]; }; then
    echo "Running on PR or non-main branch in Buildkite. Skipping database upload."
    SKIP_DB_UPLOAD_FLAG=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Register cleanup trap
if [ "$START_SERVER" = "1" ]; then
    trap 'echo "Executing cleanup..."; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM
fi

if [ -z "$RESULTS_DIR" ]; then
    # Under ARTIFACTS_DIR when the caller set one, so results are picked up as
    # artifacts without every step restating the path it already declared.
    if [ -n "${ARTIFACTS_DIR:-}" ]; then
        RESULTS_DIR="$ARTIFACTS_DIR/$CONFIG_NAME"
    else
        RESULTS_DIR="/tmp/perf_eval_$CONFIG_NAME"
    fi
fi
mkdir -p "$RESULTS_DIR"

BASELINE_CONFIG="${BASELINE_CONFIG:-$CONFIG_NAME}"
PERF_BASELINE="scripts/vllm/benchmarking/baselines/perf/$BASELINE_CONFIG.baseline.json"

PERF_LOG="$RESULTS_DIR/regression_check.md"

run_lm_eval() {
    local task="$1"
    local baseline="scripts/vllm/benchmarking/baselines/eval/${BASELINE_CONFIG}.${task}.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"

    # --model_args is one comma-separated string, not a repeatable flag, so
    # config extras merge in here. Last-wins per key: an extra can override.
    local model_args="model=$MODEL,base_url=http://$HOST:$PORT/v1/chat/completions,num_concurrent=128"
    if [ -n "$EXTRA_LM_EVAL_MODEL_ARGS" ]; then
        model_args="$model_args,$EXTRA_LM_EVAL_MODEL_ARGS"
    fi

    local lm_eval_args=(
        --tasks "$task"
        --seed "0,1234,None,1234"
        --output_path "$RESULTS_DIR"
    )

    case "$task" in
        humaneval_plus_tpu|mbpp_plus_tpu)
            # HF evaluate's code_eval metric has its own unsafe-code gate on
            # top of lm-eval's --confirm_run_unsafe_code. No --limit: the
            # full EvalPlus datasets (164 + 378 problems), like the evalplus
            # CLI ran them.
            export HF_ALLOW_CODE_EVAL=1
            lm_eval_args+=(
                --include_path "$SCRIPT_DIR/lm_eval_tasks"
                --model local-chat-completions
                --model_args "$model_args"
                --apply_chat_template
                --gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}'
                --confirm_run_unsafe_code
            )
            ;;
        *)
            lm_eval_args+=(
                --limit 100
                --model local-chat-completions
                --model_args "$model_args"
                --apply_chat_template
            )
            local gen_kwargs
            if [ "$task" = "mmlu_pro" ] && [ "$MMLU_PRO_DISABLE_MULTITURN_ARGS" = "true" ]; then
                gen_kwargs='{"chat_template_kwargs": {"enable_thinking": false}}'
            else
                lm_eval_args+=(--fewshot_as_multiturn true)
                gen_kwargs='{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}'
            fi
            # Replaced, not extended: --gen_kwargs is a merging argparse
            # action, so a second flag would union with the default.
            if [ -n "$LM_EVAL_GEN_KWARGS" ]; then
                gen_kwargs="$LM_EVAL_GEN_KWARGS"
            fi
            lm_eval_args+=(--gen_kwargs "$gen_kwargs")
            ;;
    esac

    # lm-eval builds every request (contexts, few-shot examples) before sending
    # the first, ~40s per task with the server idle, and each config rebuilt the
    # same ones. They follow from the arguments below, the lm-eval version and
    # our task definitions, not from the model: local-chat-completions leaves
    # the chat template to the server and fingerprints requests with an empty
    # tokenizer name. So they are cached per combination in the shared cache,
    # $HF_HOME/hub (a gcsfuse bucket on kube), for every config and later run. lm-eval's own cache key leaves
    # out --limit, --gen_kwargs and the task YAML contents, so the directory
    # name carries all of it.
    local request_spec=() skip_next=0 arg
    for arg in "${lm_eval_args[@]}"; do
        if [ "$skip_next" = 1 ]; then
            skip_next=0
            continue
        fi
        case "$arg" in
            --output_path|--model_args) skip_next=1; continue ;;
        esac
        request_spec+=("$arg")
    done
    request_spec+=("extra_model_args=$EXTRA_LM_EVAL_MODEL_ARGS")
    # A tokenizer setting makes requests depend on the model after all.
    if [[ "$EXTRA_LM_EVAL_MODEL_ARGS" == *tokeniz* ]]; then
        request_spec+=("model=$MODEL")
    fi
    # Best effort: if any input can't be read, the task runs uncached.
    local request_key="" request_cache=""
    request_key=$({
        python3 -c 'import importlib.metadata as m; print(m.version("lm_eval"))'
        printf '%s\n' "${request_spec[@]}"
        (cd "$SCRIPT_DIR/lm_eval_tasks" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
    } | sha256sum | cut -c1-16) || request_key=""
    if [ -n "$request_key" ]; then
        request_cache="${LM_EVAL_REQUEST_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/hub/lm_eval_requests}/$request_key"
        if mkdir -p "$request_cache"; then
            lm_eval_args+=(--cache_requests true)
        else
            request_cache=""
        fi
    fi

    echo "Running lm_eval for task: $task (request cache: ${request_cache:-off})..."
    echo "[cmd] lm_eval ${lm_eval_args[*]}"
    # lm-eval's progress bar redraws with carriage returns and ends its line
    # only when the run does. Container runtimes log by line, so an eval with
    # nothing else writing to the console shows no output at all until it
    # finishes - Kimi-K2.6's took 22 silent minutes on kube and read as a hang.
    # Turning each redraw into its own line makes every TQDM_MININTERVAL tick
    # visible; stdbuf stops tr holding them in a block buffer. pipefail keeps
    # lm_eval's exit status.
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 LM_HARNESS_CACHE_PATH="$request_cache" lm_eval "${lm_eval_args[@]}" 2>&1 | stdbuf -oL tr '\r' '\n'

    if [ -f "$baseline" ]; then
        echo "=== Checking Eval Regression for $task ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode eval \
          --tolerance "$EVAL_TOLERANCE" \
          --results-dir "$RESULTS_DIR" \
          --baseline "$baseline" 2>&1 | tee "$eval_log" || eval_regression_failed "$task"
    else
        echo "WARNING: Baseline not found for $task at $baseline. Skipping regression check."
    fi
}

# Multimodal eval via evalscope (https://evalscope.readthedocs.io/), hitting
# the live OpenAI-compatible endpoint directly -- lm-eval has no multimodal
# support, so mmmu_pro can't go through run_lm_eval. --limit is applied
# *per subset* by evalscope (30 subjects in mmmu_pro), not overall, so
# MM_EVAL_LIMIT=10 means <=300 total samples, not 10.
run_mm_eval() {
    local task="$1"
    local baseline="scripts/vllm/benchmarking/baselines/eval/${BASELINE_CONFIG}.${task}.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"
    local mm_workdir="${RESULTS_DIR}/${task}_evalscope"
    local mm_limit="${MM_EVAL_LIMIT:-4}"
    local mm_batch_size="${MM_EVAL_BATCH_SIZE:-128}"

    echo "Running evalscope for task: $task (limit=$mm_limit/subset, eval-batch-size=$mm_batch_size)..."
    rm -rf "$mm_workdir"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 evalscope eval \
        --model "$MODEL" \
        --eval-type openai_api \
        --api-url "http://$HOST:$PORT/v1/chat/completions" \
        --api-key EMPTY \
        --datasets "$task" \
        --limit "$mm_limit" \
        --eval-batch-size "$mm_batch_size" \
        --no-timestamp \
        --work-dir "$mm_workdir"

    local report_json
    report_json=$(find "$mm_workdir" -path "*/reports/*/${task}.json" | head -1)
    if [ -z "$report_json" ]; then
        echo "::error::evalscope did not produce a ${task}.json report under $mm_workdir"
        fail=1
        return
    fi

    # Translate evalscope's report into check_regression.py's lm-eval-style
    # results_*.json schema ({"results": {task: {"acc,none": score}}}) so the
    # existing eval regression/calibrate machinery works unmodified.
    python3 -c "
import json
report = json.load(open('$report_json'))
out = {'results': {'$task': {'acc,none': report['score']}}}
json.dump(out, open('$RESULTS_DIR/results_${task}.json', 'w'), indent=2)
print('$task: score=' + str(report['score']) + ' (num=' + str(report['num']) + ')')
"

    if [ -f "$baseline" ]; then
        echo "=== Checking Eval Regression for $task ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode eval \
          --tolerance "$EVAL_TOLERANCE" \
          --results-dir "$RESULTS_DIR" \
          --baseline "$baseline" 2>&1 | tee "$eval_log" || eval_regression_failed "$task"
    else
        echo "WARNING: Baseline not found for $task at $baseline. Skipping regression check."
    fi
}

run_json_schema_eval() {
    local task="json_schema"
    local baseline="scripts/vllm/benchmarking/baselines/eval/${BASELINE_CONFIG}.${task}.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"
    local results_json="${RESULTS_DIR}/results_${task}.json"
    local task_cfg="${JSON_SCHEMA_TASK_CONFIG:-$SCRIPT_DIR/task_configs/json_schema_default.yaml}"

    echo "Running JSON Schema evaluation ($task_cfg)..."
    local model_arg=()
    if [ -n "${MODEL:-}" ]; then
        model_arg=(--model "$MODEL")
    fi
    if ! python3 "$SCRIPT_DIR/eval_json_schema.py" \
        --base-url "$BASE_URL" \
        "${model_arg[@]}" \
        --task-config "$task_cfg" \
        --output-json "$results_json"; then
        echo "::error::JSON Schema evaluation client failed."
        fail=1
        return
    fi

    if [ -f "$baseline" ]; then
        echo "=== Checking Eval Regression for $task ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode eval \
          --tolerance "$EVAL_TOLERANCE" \
          --results-dir "$RESULTS_DIR" \
          --baseline "$baseline" 2>&1 | tee "$eval_log" || eval_regression_failed "$task"
    else
        echo "WARNING: Baseline not found for $task at $baseline. Skipping regression check."
    fi
}

run_agentx_perf() {
    local conc_list="${AGENTX_CONC_LIST:-}"
    local model_prefix="${AGENTX_MODEL_PREFIX:-}"
    local precision="${AGENTX_PRECISION:-}"
    local chips="${AGENTX_CHIPS:-}"
    if [ -z "$conc_list" ] || [ -z "$model_prefix" ] || [ -z "$precision" ]; then
        echo "::error::agentx suite requires AGENTX_CONC_LIST, AGENTX_MODEL_PREFIX, and AGENTX_PRECISION in $CONFIG_NAME.sh or env."
        fail=1
        return
    fi
    case "$chips" in
        ''|*[!0-9]*|0)
            echo "::error::agentx suite requires AGENTX_CHIPS=<physical_chips> (physical chips, not TP x DP ranks)."
            fail=1
            return
            ;;
    esac
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/../benchmark/inferencex/sweep_lib.sh"
    local ep=1
    if [ "${CONFIG_ENABLE_EP:-false}" = "true" ]; then
        ep=$(( CONFIG_TP * CONFIG_DP ))
    fi
    local disagg="${AGENTX_DISAGG:-0}"
    if [ "${CONFIG_IS_1P1D:-0}" = "1" ]; then
        disagg=1
    fi
    ix_export_topology "$CONFIG_TP" "$CONFIG_DP" "$ep" "$disagg"
    local agentx_args=(
        --server-url "$BASE_URL"
        --conc-list "${conc_list// /,}"
        --chips "$chips"
        --model "$MODEL"
        --label "agentx_${CONFIG_NAME}"
        --model-prefix "$model_prefix"
        --precision "$precision"
        --gsm8k-limit "${AGENTX_GSM8K_LIMIT:-0}"
        --results-dir "$RESULTS_DIR/agentx"
    )
    if [ -n "${AGENTX_REFERENCE_ROW:-}" ]; then
        agentx_args+=(--reference-row "$AGENTX_REFERENCE_ROW")
    fi
    if [ -n "${AGENTX_SPEC_METHOD:-}" ]; then
        agentx_args+=(--spec-method "$AGENTX_SPEC_METHOD")
    fi
    if [ -n "${AGENTX_DURATION:-}" ]; then
        agentx_args+=(--duration "$AGENTX_DURATION")
    fi
    if [ -n "${AGENTX_FAST:-}" ]; then
        agentx_args+=(--fast "$AGENTX_FAST")
    fi
    echo "Running InferenceX AgentX suite against $BASE_URL..."
    bash "$SCRIPT_DIR/../benchmark/inferencex/agentx_client.sh" "${agentx_args[@]}" || {
        echo "::error::AgentX benchmark suite failed."
        fail=1
    }
}

CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Disaggregated 1P1D configs branch on K3_1P1D_ROLE; default to decode when
# running client-only against an external endpoint so sourcing succeeds.
if [ "$START_SERVER" = "0" ] && [ -z "${K3_1P1D_ROLE:-}" ]; then
    export K3_1P1D_ROLE="decode"
fi

# Read config-declared suite defaults in an isolated subshell before launching.
eval "$(
    SCRIPT_DIR="$SCRIPT_DIR" K3_1P1D_ROLE="${K3_1P1D_ROLE:-}" bash -c '
        LM_EVAL_TASKS="mmlu_llama mmlu_pro"
        ENABLE_EP="false"
        EVAL_SUITES="${EVAL_SUITES:-}"
        PERF_SUITES="${PERF_SUITES:-}"
        JSON_SCHEMA_TASK_CONFIG="${JSON_SCHEMA_TASK_CONFIG:-}"
        AGENTX_CONC_LIST="${AGENTX_CONC_LIST:-}"
        AGENTX_MODEL_PREFIX="${AGENTX_MODEL_PREFIX:-}"
        AGENTX_PRECISION="${AGENTX_PRECISION:-}"
        AGENTX_CHIPS="${AGENTX_CHIPS:-}"
        AGENTX_REFERENCE_ROW="${AGENTX_REFERENCE_ROW:-}"
        AGENTX_SPEC_METHOD="${AGENTX_SPEC_METHOD:-}"
        AGENTX_DURATION="${AGENTX_DURATION:-}"
        AGENTX_FAST="${AGENTX_FAST:-}"
        AGENTX_GSM8K_LIMIT="${AGENTX_GSM8K_LIMIT:-}"
        AGENTX_DISAGG="${AGENTX_DISAGG:-}"
        source "$1"
        CONFIG_ENABLE_EP="$ENABLE_EP"
        CONFIG_IS_1P1D=0
        if [ -n "${TPU_KV_TRANSFER_NAMESPACE:-}" ]; then
            CONFIG_IS_1P1D=1
        fi
        for var in LM_EVAL_TASKS CONFIG_ENABLE_EP CONFIG_IS_1P1D EVAL_SUITES PERF_SUITES JSON_SCHEMA_TASK_CONFIG \
            AGENTX_CONC_LIST AGENTX_MODEL_PREFIX AGENTX_PRECISION AGENTX_CHIPS \
            AGENTX_REFERENCE_ROW AGENTX_SPEC_METHOD AGENTX_DURATION AGENTX_FAST \
            AGENTX_GSM8K_LIMIT AGENTX_DISAGG; do
            printf "%s=%q\n" "$var" "${!var}"
        done
    ' _ "$CONFIG_FILE"
)"

# Resolve selected performance suites.
RUN_BENCH_SERVE=0
RUN_AGENTX=0
effective_perf_suites="$PERF_SUITES_CLI"
if [ -z "$effective_perf_suites" ] && [ -n "${PERF_SUITES:-}" ]; then
    effective_perf_suites="$PERF_SUITES"
fi
if [ -n "$effective_perf_suites" ]; then
    IFS=',' read -r -a _perf_items <<< "$effective_perf_suites"
    for raw_item in "${_perf_items[@]}"; do
        item="${raw_item// /}"
        case "$item" in
            ""|none) ;;
            all) RUN_BENCH_SERVE=1; RUN_AGENTX=1 ;;
            bench_serve) RUN_BENCH_SERVE=1 ;;
            agentx) RUN_AGENTX=1 ;;
            *)
                echo "ERROR: Unknown --perf-suites item '$item' (expected bench_serve, agentx, all, or none)"
                exit 1
                ;;
        esac
    done
else
    if [ "$SKIP_BENCHMARK" = "0" ]; then
        RUN_BENCH_SERVE=1
    fi
fi

if [ "$RUN_BENCH_SERVE" = "0" ]; then
    SKIP_PERF_CHECK=1
fi

# Fail before any server boot: tokens per dollar divides by physical chips,
# and TP x DP counts ranks (two per chip on v7x), so there is no safe default.
if [ "$RUN_AGENTX" = "1" ]; then
    case "${AGENTX_CHIPS:-}" in
        ''|*[!0-9]*|0)
            echo "::error::agentx suite requires AGENTX_CHIPS=<physical_chips> (physical chips, not TP x DP ranks)."
            exit 1
            ;;
    esac
fi

# Resolve selected accuracy suites.
SELECTED_EVAL_SUITES=()
effective_eval_suites="$EVAL_SUITES_CLI"
if [ -z "$effective_eval_suites" ] && [ -n "${EVAL_SUITES:-}" ] && [ "$RUN_LM_EVAL" = "0" ] && [ "$RUN_CODE_EVAL" = "0" ] && [ "$RUN_MM_EVAL" = "0" ]; then
    effective_eval_suites="$EVAL_SUITES"
fi
if [ -n "$effective_eval_suites" ]; then
    IFS=',' read -r -a _eval_items <<< "$effective_eval_suites"
    for raw_item in "${_eval_items[@]}"; do
        item="${raw_item// /}"
        case "$item" in
            ""|none) ;;
            all)
                SELECTED_EVAL_SUITES+=(
                    mmlu_pro gsm8k gpqa_diamond aime2026
                    humaneval_plus_tpu mbpp_plus_tpu mmmu_pro json_schema
                )
                ;;
            mmlu_llama|mmlu_pro|gsm8k|gpqa_diamond|aime2026|humaneval_plus_tpu|mbpp_plus_tpu|mmmu_pro|json_schema|bfcl|ruler)
                SELECTED_EVAL_SUITES+=("$item")
                ;;
            *)
                echo "ERROR: Unknown --eval-suites item '$item'"
                exit 1
                ;;
        esac
    done
else
    if [ "$RUN_LM_EVAL" = "1" ]; then
        # shellcheck disable=SC2206  # space-separated task list, split on purpose
        SELECTED_EVAL_SUITES+=($LM_EVAL_TASKS)
    fi
    if [ "$RUN_CODE_EVAL" = "1" ]; then
        SELECTED_EVAL_SUITES+=(humaneval_plus_tpu mbpp_plus_tpu)
    fi
    if [ "$RUN_MM_EVAL" = "1" ]; then
        SELECTED_EVAL_SUITES+=(mmmu_pro)
    fi
fi

NEEDS_LIVE_SERVER=0
if [ "$RUN_BENCH_SERVE" = "1" ] || [ "$RUN_AGENTX" = "1" ] || [ "${#SELECTED_EVAL_SUITES[@]}" -gt 0 ]; then
    NEEDS_LIVE_SERVER=1
fi

# ========================================================
# 1. Initial Cleanup
# ========================================================
if [ "$START_SERVER" = "1" ] && [ "$NEEDS_LIVE_SERVER" = "1" ]; then
    echo "=== Initial Cleanup ==="
    bash "$SCRIPT_DIR/cleanup_server.sh"
fi

# ========================================================
# 2. Run Benchmarks (starts server and keeps it alive)
# ========================================================
echo "Running benchmarks for $CONFIG_NAME..."
BENCH_ARGS=(
  --config "$CONFIG_NAME"
  --results-dir "$RESULTS_DIR"
)
if [ "$START_SERVER" = "1" ]; then
  BENCH_ARGS+=(--keep-alive)
else
  BENCH_ARGS+=(--host "$HOST")
fi
BENCH_ARGS+=(--port "$PORT")
if [ "$RUN_BENCH_SERVE" = "0" ]; then
  if [ "$START_SERVER" = "1" ] && [ "$NEEDS_LIVE_SERVER" = "1" ]; then
    BENCH_ARGS+=(--skip-bench-cells)
  else
    # A dry run still writes the config.json the evals below read.
    BENCH_ARGS+=(--dry-run)
  fi
fi

./scripts/vllm/benchmarking/run_benchmarks.sh "${BENCH_ARGS[@]}"

# Read model name from config
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
CONFIG_TP=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("tensor_parallelism", 1))' "$RESULTS_DIR/config.json")
CONFIG_DP=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("data_parallelism", 1))' "$RESULTS_DIR/config.json")
MMLU_PRO_DISABLE_MULTITURN_ARGS=$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("mmlu_pro_disable_multiturn_args", False)).lower())' "$RESULTS_DIR/config.json")
LM_EVAL_TASKS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("lm_eval_tasks") or "mmlu_llama mmlu_pro")' "$RESULTS_DIR/config.json")
EXTRA_LM_EVAL_MODEL_ARGS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("extra_lm_eval_model_args", ""))' "$RESULTS_DIR/config.json")
LM_EVAL_GEN_KWARGS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("lm_eval_gen_kwargs", ""))' "$RESULTS_DIR/config.json")
EVAL_TOLERANCE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["eval_tolerance"])' "$RESULTS_DIR/config.json")
PERF_TOLERANCE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("perf_tolerance", 0.05))' "$RESULTS_DIR/config.json")

fail=0

# ========================================================
# 3. Check Perf Regression & Run AgentX
# ========================================================
if [ "$SKIP_PERF_CHECK" = "0" ] && [ -f "$PERF_BASELINE" ]; then
    echo "=== Checking Perf Regression ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode perf \
      --tolerance "$PERF_TOLERANCE" \
      --results-dir "$RESULTS_DIR" \
      --baseline "$PERF_BASELINE" 2>&1 | tee "$PERF_LOG" || { echo "::error::Perf regression check failed! See logs above for details."; fail=1; }
fi

if [ "$RUN_AGENTX" = "1" ]; then
    run_agentx_perf
fi

# ========================================================
# 4. Run Selected Accuracy Suites
# ========================================================
for suite in "${SELECTED_EVAL_SUITES[@]}"; do
    case "$suite" in
        mmlu_llama|mmlu_pro|gsm8k|gpqa_diamond|aime2026|humaneval_plus_tpu|mbpp_plus_tpu)
            run_lm_eval "$suite"
            ;;
        mmmu_pro)
            if ! command -v evalscope &>/dev/null; then
                echo "::error::mmmu_pro requires the 'evalscope' CLI (pip install evalscope)."
                fail=1
            else
                run_mm_eval "mmmu_pro"
            fi
            ;;
        json_schema)
            run_json_schema_eval
            ;;
        bfcl|ruler)
            echo "::error::Accuracy suite '$suite' runner is not wired in yet."
            fail=1
            ;;
    esac
done

# ========================================================
# 7. Upload results to BigQuery (and Spanner, if asked)
# ========================================================
echo "=== Uploading results ==="
UPLOAD_ARGS=("--results-dir" "$RESULTS_DIR")
if [ "${SKIP_DB_UPLOAD:-0}" = "1" ] || [ "${SKIP_DB_UPLOAD:-}" = "true" ] || [ "$SKIP_DB_UPLOAD_FLAG" = "1" ]; then
    UPLOAD_ARGS+=("--skip-db-upload")
fi
# Spanner is opt-in: its rows duplicate the BigQuery ones, keyed by the same
# RecordId. SKIP_SPANNER_UPLOAD=0 writes it as well; the guard above still
# skips both.
if [ "${SKIP_SPANNER_UPLOAD:-1}" != "0" ] && [ "${SKIP_SPANNER_UPLOAD:-}" != "false" ]; then
    UPLOAD_ARGS+=("--skip-spanner")
fi


python3 scripts/vllm/benchmarking/upload_results.py "${UPLOAD_ARGS[@]}" || echo "Warning: results upload failed"

# Opt-in per step, and under the same guard as the database upload, so a PR's
# numbers never reach the dashboard. SHAPE's topology is the slice's chips.
if [ "${UPLOAD_TO_VLLM_CI:-0}" = "1" ] && [ "$RUN_BENCH_SERVE" = "1" ] && [[ " ${UPLOAD_ARGS[*]} " != *" --skip-db-upload "* ]]; then
    echo "=== Publishing to the ci.vllm.ai Performance tab ==="
    topology="${SHAPE:-}"
    topology="${topology#*/}"
    if [[ ! "$topology" =~ ^[0-9]+(x[0-9]+)*$ ]] || [ -z "${TPU_ACCELERATOR_TYPE:-}" ] || [ -z "${WORKLOAD_IMAGE:-}" ]; then
        echo "Warning: UPLOAD_TO_VLLM_CI needs SHAPE=<machine-type>/<topology>, TPU_ACCELERATOR_TYPE and WORKLOAD_IMAGE; not publishing"
    else
        python3 scripts/vllm/benchmarking/publish_perf_dashboard.py "$RESULTS_DIR" \
            --device "$TPU_ACCELERATOR_TYPE" \
            --chips "$(( ${topology//x/*} ))" \
            --image "$WORKLOAD_IMAGE" ||
            echo "Warning: ci.vllm.ai publish failed"
    fi
fi

exit "$fail"

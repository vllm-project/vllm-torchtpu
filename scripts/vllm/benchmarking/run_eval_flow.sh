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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8000}"
HOST=""
HOST_PORT_CLI=0
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
BASE_URL_CLI_SET=0
EVAL_SUITES_CLI=""
EVAL_SUITES_CLI_SET=0
PERF_SUITES_CLI=""
PERF_SUITES_CLI_SET=0
SERVED_MODEL_NAME_CLI=""
SERVED_MODEL_NAME_CLI_SET=0

# A value never starts with "--", so "--config --dry-run" is a missing value.
require_arg() {
    if [ "$#" -lt 2 ] || [[ "$2" == --* ]]; then
        echo "ERROR: $1 requires a value"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            require_arg "$@"
            CONFIG_NAME="$2"
            shift 2
            ;;
        --results-dir)
            require_arg "$@"
            RESULTS_DIR="$2"
            shift 2
            ;;
        --base-url)
            require_arg "$@"
            BASE_URL_CLI="$2"
            BASE_URL_CLI_SET=1
            shift 2
            ;;
        --eval-suites)
            require_arg "$@"
            EVAL_SUITES_CLI="$2"
            EVAL_SUITES_CLI_SET=1
            shift 2
            ;;
        --perf-suites)
            require_arg "$@"
            PERF_SUITES_CLI="$2"
            PERF_SUITES_CLI_SET=1
            shift 2
            ;;
        # The name the endpoint serves the model under, when a router or
        # gateway does not serve it under the config's MODEL.
        --served-model-name)
            require_arg "$@"
            SERVED_MODEL_NAME_CLI="$2"
            SERVED_MODEL_NAME_CLI_SET=1
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
            require_arg "$@"
            HOST="$2"
            HOST_PORT_CLI=1
            START_SERVER=0
            shift 2
            ;;
        --port)
            require_arg "$@"
            PORT="$2"
            HOST_PORT_CLI=1
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
            shift
            ;;
        # Gates against another config's baselines, for a config that serves
        # that one's model another way and must answer the same.
        --baseline-config)
            require_arg "$@"
            BASELINE_CONFIG="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --config CONFIG_NAME [--base-url URL] [--served-model-name NAME] [--eval-suites LIST] [--perf-suites LIST] [--results-dir DIR] [--run-lm-eval] [--run-code-eval] [--run-mm-eval] [--host HOST] [--port PORT] [--skip-db-upload] [--skip-perf-check] [--skip-benchmark] [--baseline-config CONFIG_NAME]"
            # shellcheck disable=SC2016
            echo '  --results-dir defaults to $ARTIFACTS_DIR/$CONFIG_NAME when ARTIFACTS_DIR is set, else /tmp/perf_eval_$CONFIG_NAME'
            exit 1
            ;;
    esac
done

# BASE_URL is the one API root every client below is given. --base-url keeps
# its scheme and any path prefix, since a gateway may serve the API over https
# or under a path; only a trailing slash or /v1 on the path is dropped, because
# each client appends its own /v1/... route.
if [ "$BASE_URL_CLI_SET" = "1" ]; then
    if [ -z "$BASE_URL_CLI" ]; then
        echo "ERROR: --base-url cannot be empty"
        exit 1
    fi
    if [ "$HOST_PORT_CLI" = "1" ]; then
        echo "ERROR: --base-url cannot be combined with --host or --port"
        exit 1
    fi
    BASE_URL=$(python3 "$SCRIPT_DIR/endpoint_url.py" normalize "$BASE_URL_CLI")
    START_SERVER=0
else
    if [ -z "$HOST" ]; then
        HOST="localhost"
    fi
    BASE_URL="http://$HOST:$PORT"
fi

if [ "$SERVED_MODEL_NAME_CLI_SET" = "1" ] && [ -z "$SERVED_MODEL_NAME_CLI" ]; then
    echo "ERROR: --served-model-name cannot be empty"
    exit 1
fi

if [ "$EVAL_SUITES_CLI_SET" = "1" ] && [ -z "${EVAL_SUITES_CLI// /}" ]; then
    echo "ERROR: --eval-suites cannot be empty (use 'none' to run no accuracy suites)"
    exit 1
fi

if [ "$PERF_SUITES_CLI_SET" = "1" ] && [ -z "${PERF_SUITES_CLI// /}" ]; then
    echo "ERROR: --perf-suites cannot be empty (use 'none' to run no performance suites)"
    exit 1
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

PERF_LOG="$RESULTS_DIR/regression_check.md"

run_lm_eval() {
    local task="$1"
    local baseline="scripts/vllm/benchmarking/baselines/eval/${BASELINE_CONFIG}.${task}.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"

    # --model_args is one comma-separated string, not a repeatable flag, so
    # config extras merge in here. Last-wins per key: an extra can override.
    local model_args="model=$MODEL,base_url=$BASE_URL/v1/chat/completions,num_concurrent=128"
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
            case "$task" in
                gsm8k)
                    gen_kwargs='{"chat_template_kwargs": {"enable_thinking": false}}'
                    ;;
                mmlu_pro)
                    if [ "$MMLU_PRO_DISABLE_MULTITURN_ARGS" = "true" ]; then
                        gen_kwargs='{"chat_template_kwargs": {"enable_thinking": false}}'
                    else
                        lm_eval_args+=(--fewshot_as_multiturn true)
                        gen_kwargs='{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}'
                    fi
                    ;;
                *)
                    lm_eval_args+=(--fewshot_as_multiturn true)
                    gen_kwargs='{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}'
                    ;;
            esac
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
        --api-url "$BASE_URL/v1/chat/completions" \
        --api-key "${OPENAI_API_KEY:-EMPTY}" \
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

# Structured outputs via eval_json_schema.py. The pass rule is fixed, not
# calibrated: every response must be valid on every config, so there is one
# shared baseline at 1.0 and no tolerance.
run_json_schema_eval() {
    local task="json_schema"
    local pass_rule="scripts/vllm/benchmarking/baselines/eval/json_schema.pass_rule.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"

    echo "Running eval_json_schema.py for task: $task (task config: $JSON_SCHEMA_TASK_CONFIG)..."
    if ! python3 "$SCRIPT_DIR/eval_json_schema.py" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --task-config "$JSON_SCHEMA_TASK_CONFIG" \
        --output-json "$RESULTS_DIR/results_${task}.json"; then
        echo "::error::eval_json_schema.py could not evaluate $task. See logs above for details."
        fail=1
        return
    fi

    echo "=== Checking Eval Regression for $task ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode eval \
      --tolerance 0 \
      --results-dir "$RESULTS_DIR" \
      --baseline "$pass_rule" 2>&1 | tee "$eval_log" || eval_regression_failed "$task"
}

# Single source of truth for named --eval-suites entries:
#   "<suite>:<runner>:<scope>"
# Scope says when `all` includes the suite: "text" always, "config_lm_eval"
# only when the config's LM_EVAL_TASKS lists it (a config narrows that list to
# rule a task out for its model, and --run-lm-eval follows it too), and
# "multimodal" only for *multimodal* configs, matching run_eval_configs.sh.
EVAL_SUITE_TABLE=(
    "mmlu_llama:lm_eval:config_lm_eval"
    "mmlu_pro:lm_eval:config_lm_eval"
    "gsm8k:lm_eval:text"
    "humaneval_plus_tpu:lm_eval:text"
    "mbpp_plus_tpu:lm_eval:text"
    "mmmu_pro:mm_eval:multimodal"
    "json_schema:json_schema:text"
)

eval_suite_runner() {
    local target="$1" entry suite runner _
    for entry in "${EVAL_SUITE_TABLE[@]}"; do
        IFS=':' read -r suite runner _ <<< "$entry"
        if [ "$suite" = "$target" ]; then
            echo "$runner"
            return 0
        fi
    done
    return 1
}

append_all_eval_suites() {
    local entry suite _ scope
    for entry in "${EVAL_SUITE_TABLE[@]}"; do
        IFS=':' read -r suite _ scope <<< "$entry"
        case "$scope" in
            config_lm_eval)
                # Padded with spaces so a name only matches a whole word.
                [[ " ${LM_EVAL_TASKS//[$'\t\n']/ } " == *" $suite "* ]] || continue
                ;;
            multimodal)
                [[ "$CONFIG_NAME" == *multimodal* ]] || continue
                ;;
        esac
        SELECTED_EVAL_SUITES+=("$suite")
    done
}

CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read config-declared suite defaults in an isolated subshell before launching.
# Suite variables start empty, and JSON_SCHEMA_TASK_CONFIG at its default, so
# only the config file sets them, and config stdout is discarded so echoed
# notes are never eval'd. MODEL and BENCHMARK_MODE start as run_benchmarks.sh
# starts them, and come back under CONFIG_ names so this shell's own MODEL and
# environment stay as they were.
_config_vars=$(
    SCRIPT_DIR="$SCRIPT_DIR" bash -c '
        set -euo pipefail
        LM_EVAL_TASKS="mmlu_llama mmlu_pro"
        EVAL_SUITES=""
        PERF_SUITES=""
        JSON_SCHEMA_TASK_CONFIG="$SCRIPT_DIR/task_configs/json_schema_default.yaml"
        MODEL=""
        BENCHMARK_MODE="${BENCHMARK_MODE:-serving}"
        source "$1" >/dev/null
        for var in LM_EVAL_TASKS EVAL_SUITES PERF_SUITES JSON_SCHEMA_TASK_CONFIG; do
            printf "%s=%q\n" "$var" "${!var}"
        done
        printf "CONFIG_MODEL=%q\n" "$MODEL"
        printf "CONFIG_BENCHMARK_MODE=%q\n" "$BENCHMARK_MODE"
    ' _ "$CONFIG_FILE"
) || {
    echo "ERROR: Failed to read config defaults from $CONFIG_FILE" >&2
    exit 1
}
eval "$_config_vars"

# Resolve selected performance suites.
# Precedence (same as eval suites below):
#   --perf-suites CLI > legacy --skip-benchmark > config PERF_SUITES > bench_serve
RUN_BENCH_SERVE=0
if [ "$PERF_SUITES_CLI_SET" = "1" ]; then
    effective_perf_suites="$PERF_SUITES_CLI"
elif [ -n "$PERF_SUITES" ]; then
    effective_perf_suites="$PERF_SUITES"
else
    effective_perf_suites="bench_serve"
fi
IFS=',' read -r -a _perf_items <<< "$effective_perf_suites"
for raw_item in "${_perf_items[@]}"; do
    item="${raw_item// /}"
    if [ -z "$item" ]; then
        echo "ERROR: Empty item in --perf-suites: '$effective_perf_suites'"
        exit 1
    fi
    case "$item" in
        none) ;;
        all|bench_serve) RUN_BENCH_SERVE=1 ;;
        *)
            echo "ERROR: Unknown --perf-suites item '$item' (expected bench_serve, all, or none)"
            exit 1
            ;;
    esac
done
if [ "$PERF_SUITES_CLI_SET" = "0" ] && [ "$SKIP_BENCHMARK" = "1" ]; then
    RUN_BENCH_SERVE=0
fi

if [ "$RUN_BENCH_SERVE" = "0" ]; then
    SKIP_PERF_CHECK=1
fi

# Resolve selected accuracy suites.
# Precedence:
#   --eval-suites CLI > legacy --run-*-eval flags > config EVAL_SUITES > none
SELECTED_EVAL_SUITES=()
effective_eval_suites=""
if [ "$EVAL_SUITES_CLI_SET" = "1" ]; then
    effective_eval_suites="$EVAL_SUITES_CLI"
elif [ "$RUN_LM_EVAL" = "1" ] || [ "$RUN_CODE_EVAL" = "1" ] || [ "$RUN_MM_EVAL" = "1" ]; then
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
else
    effective_eval_suites="$EVAL_SUITES"
fi
if [ -n "$effective_eval_suites" ]; then
    IFS=',' read -r -a _eval_items <<< "$effective_eval_suites"
    for raw_item in "${_eval_items[@]}"; do
        item="${raw_item// /}"
        if [ -z "$item" ]; then
            echo "ERROR: Empty item in --eval-suites: '$effective_eval_suites'"
            exit 1
        fi
        case "$item" in
            none) ;;
            all) append_all_eval_suites ;;
            *)
                if eval_suite_runner "$item" >/dev/null; then
                    SELECTED_EVAL_SUITES+=("$item")
                else
                    echo "ERROR: Unknown --eval-suites item '$item'"
                    exit 1
                fi
                ;;
        esac
    done
fi

# Checked here, before any server boots: the json_schema suite would only find
# its task config missing after the boot and every suite that runs before it.
if [[ " ${SELECTED_EVAL_SUITES[*]} " == *" json_schema "* ]] && [ ! -f "$JSON_SCHEMA_TASK_CONFIG" ]; then
    echo "ERROR: JSON_SCHEMA_TASK_CONFIG of $CONFIG_NAME is not a file: $JSON_SCHEMA_TASK_CONFIG"
    exit 1
fi

NEEDS_LIVE_SERVER=0
if [ "$RUN_BENCH_SERVE" = "1" ] || [ "${#SELECTED_EVAL_SUITES[@]}" -gt 0 ]; then
    NEEDS_LIVE_SERVER=1
fi

# Compute once whether this run boots and owns a local server.
# --skip-benchmark keeps its meaning (unless --perf-suites explicitly re-enables
# bench_serve): no server is started or killed, and evals hit an already-running
# server on localhost:$PORT.
BOOT_SERVER=0
if [ "$START_SERVER" = "1" ] && [ "$NEEDS_LIVE_SERVER" = "1" ]; then
    if [ "$RUN_BENCH_SERVE" = "1" ] || [ "$SKIP_BENCHMARK" = "0" ]; then
        BOOT_SERVER=1
    fi
fi

# A server this script starts serves the config's MODEL, so another name would
# send every request to a model that is not there.
if [ "$SERVED_MODEL_NAME_CLI_SET" = "1" ] && [ "$BOOT_SERVER" = "1" ]; then
    echo "ERROR: --served-model-name needs a running server (--base-url, --host and --port, or --skip-benchmark for the one already on localhost:$PORT)"
    exit 1
fi

# A spec_bench config has no server outside its own (BS, K) sweep, which starts
# and stops one per cell, so there is none to keep for the evals. Checked here,
# before the cleanup below kills whatever is serving.
if [ "$BOOT_SERVER" = "1" ] && [ "$RUN_BENCH_SERVE" = "0" ] && [ "$CONFIG_BENCHMARK_MODE" = "spec_bench" ]; then
    echo "ERROR: $CONFIG_NAME is a spec_bench config and cannot run eval suites on a server this script starts. Point the evals at a running server with --host and --port or --base-url, or pass --skip-benchmark to use the one already on localhost:$PORT."
    exit 1
fi

# Perf baselines are per chip: baselines/perf/<chip>/<config>.baseline.json,
# <chip> read from TPU_ACCELERATOR_TYPE (or GCE metadata). Without a baseline
# the run reports without the perf gate and says so in a ::warning::, so an
# uncalibrated run is not mistaken for a gated one; without a readable chip it
# stops.
# --skip-perf-check and a run without bench_serve have no perf gate and need
# no chip, so the chip is read here: once the suites are resolved, and before
# the cleanup below kills whatever is serving.
PERF_BASELINE=""
if [ "$SKIP_PERF_CHECK" = "0" ]; then
    # The marker keeps whatever else the import prints out of the chip's name.
    TPU_CHIP=$(python3 -c 'from vllm_torchtpu import tpu_info; print("TPU_CHIP=" + tpu_info.parse_chip_version(tpu_info.get_tpu_type() or "").value)' | sed -n 's/^TPU_CHIP=//p' | tail -n 1) || TPU_CHIP=""
    if [ -z "$TPU_CHIP" ]; then
        echo "ERROR: could not read the TPU chip to pick the perf baselines; set TPU_ACCELERATOR_TYPE, or pass --skip-perf-check to report without the perf gate" >&2
        exit 1
    fi
    TPU_CHIP="v${TPU_CHIP#v}"
    PERF_BASELINE="scripts/vllm/benchmarking/baselines/perf/$TPU_CHIP/$BASELINE_CONFIG.baseline.json"
    if [ -f "$PERF_BASELINE" ]; then
        echo "Perf baseline for $TPU_CHIP: $PERF_BASELINE"
    else
        # TODO(#<issue>): fail here instead of warning, once every perf-lane
        # config has a baseline or passes --skip-perf-check.
        echo "::warning::No perf baseline for $TPU_CHIP at $PERF_BASELINE; reporting without the perf gate. Calibrate and check in a baseline to gate this config, or pass --skip-perf-check to report on purpose."
    fi
fi

# The name clients put in their requests. Empty means the config's MODEL, which
# is what a server this script starts serves. A server this run does not start
# is asked for its /v1/models list first, once and with no wait (the caller
# starts this script when the endpoint is ready), and the run stops here
# (set -e) unless the name it would send is on that list: every request for
# another name fails, and vllm bench serve still exits 0 with zeros. The name
# is --served-model-name when given, else MODEL.
SERVED_MODEL="$SERVED_MODEL_NAME_CLI"
if [ "$BOOT_SERVER" = "0" ] && [ "$NEEDS_LIVE_SERVER" = "1" ]; then
    RESOLVE_ARGS=(--config-model "$CONFIG_MODEL")
    if [ -n "$SERVED_MODEL_NAME_CLI" ]; then
        RESOLVE_ARGS+=(--served-model-name "$SERVED_MODEL_NAME_CLI")
    fi
    SERVED_MODEL=$(python3 "$SCRIPT_DIR/endpoint_url.py" resolve-model "$BASE_URL" "${RESOLVE_ARGS[@]}")
    if [ "$SERVED_MODEL" = "$CONFIG_MODEL" ]; then
        SERVED_MODEL=""
    fi
fi

# ========================================================
# 1. Initial Cleanup (only when this run boots a server)
# ========================================================
if [ "$BOOT_SERVER" = "1" ]; then
    trap 'echo "Executing cleanup..."; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM
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
if [ "$BOOT_SERVER" = "1" ]; then
  BENCH_ARGS+=(--keep-alive --port "$PORT")
  if [ "$RUN_BENCH_SERVE" = "0" ]; then
    BENCH_ARGS+=(--skip-bench-cells)
  fi
elif [ "$RUN_BENCH_SERVE" = "0" ]; then
  # A dry run still writes the config.json the evals below read. It sends no
  # request, so it is not given the endpoint or the served model name:
  # spec_bench configs reject an external endpoint even in a dry run.
  BENCH_ARGS+=(--dry-run)
else
  if [ "$BASE_URL_CLI_SET" = "1" ]; then
    BENCH_ARGS+=(--base-url "$BASE_URL")
  else
    BENCH_ARGS+=(--host "$HOST" --port "$PORT")
  fi
  if [ -n "$SERVED_MODEL" ]; then
    BENCH_ARGS+=(--served-model-name "$SERVED_MODEL")
  fi
fi

./scripts/vllm/benchmarking/run_benchmarks.sh "${BENCH_ARGS[@]}"

# Read model name from config
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
if [ "$BOOT_SERVER" = "0" ] && [ "$NEEDS_LIVE_SERVER" = "1" ] && [ -z "$SERVED_MODEL" ] && [ "$MODEL" != "$CONFIG_MODEL" ]; then
    echo "ERROR: $CONFIG_NAME sets MODEL to '$CONFIG_MODEL' when run_eval_flow.sh reads it and to '$MODEL' when run_benchmarks.sh does. $BASE_URL/v1/models was checked for the first name and the clients send the second." >&2
    exit 1
fi
if [ -n "$SERVED_MODEL" ]; then
    if [ "$SERVED_MODEL" != "$MODEL" ]; then
        echo "Sending requests for served model $SERVED_MODEL (config model: $MODEL)"
    fi
    MODEL="$SERVED_MODEL"
elif [ "$BOOT_SERVER" = "0" ] && [ "$NEEDS_LIVE_SERVER" = "1" ]; then
    echo "Sending requests for the config's model $MODEL, which $BASE_URL/v1/models lists"
fi
MMLU_PRO_DISABLE_MULTITURN_ARGS=$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("mmlu_pro_disable_multiturn_args", False)).lower())' "$RESULTS_DIR/config.json")
EXTRA_LM_EVAL_MODEL_ARGS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("extra_lm_eval_model_args", ""))' "$RESULTS_DIR/config.json")
LM_EVAL_GEN_KWARGS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("lm_eval_gen_kwargs", ""))' "$RESULTS_DIR/config.json")
EVAL_TOLERANCE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["eval_tolerance"])' "$RESULTS_DIR/config.json")
PERF_TOLERANCE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("perf_tolerance", 0.05))' "$RESULTS_DIR/config.json")

fail=0

# ========================================================
# 3. Check Perf Regression
# ========================================================
if [ "$SKIP_PERF_CHECK" = "0" ] && [ -f "$PERF_BASELINE" ]; then
    echo "=== Checking Perf Regression ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode perf \
      --tolerance "$PERF_TOLERANCE" \
      --results-dir "$RESULTS_DIR" \
      --baseline "$PERF_BASELINE" 2>&1 | tee "$PERF_LOG" || { echo "::error::Perf regression check failed! See logs above for details."; fail=1; }
fi

# ========================================================
# 4. Run Selected Accuracy Suites
# ========================================================
for suite in "${SELECTED_EVAL_SUITES[@]}"; do
    # --eval-suites names are checked above against EVAL_SUITE_TABLE, so any
    # name not in the table came from a config's LM_EVAL_TASKS under --run-lm-eval.
    suite_runner=$(eval_suite_runner "$suite" || echo "lm_eval")
    case "$suite_runner" in
        lm_eval)
            run_lm_eval "$suite"
            ;;
        mm_eval)
            if ! command -v evalscope &>/dev/null; then
                echo "::error::$suite requires the 'evalscope' CLI (pip install evalscope)."
                fail=1
            else
                run_mm_eval "$suite"
            fi
            ;;
        json_schema)
            run_json_schema_eval
            ;;
    esac
done

# ========================================================
# 5. Upload results to BigQuery (and Spanner, if asked)
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

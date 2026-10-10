#!/bin/bash
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
            echo "Usage: $0 --config CONFIG_NAME [--results-dir DIR] [--run-lm-eval] [--run-code-eval] [--run-mm-eval] [--host HOST] [--port PORT] [--skip-db-upload] [--skip-perf-check] [--skip-benchmark] [--baseline-config CONFIG_NAME]"
            # shellcheck disable=SC2016
            echo '  --results-dir defaults to $ARTIFACTS_DIR/$CONFIG_NAME when ARTIFACTS_DIR is set, else /tmp/perf_eval_$CONFIG_NAME'
            exit 1
            ;;
    esac
done

if [ -z "$HOST" ]; then
    HOST="localhost"
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
    # $HF_HOME/hub (a gcsfuse bucket on kube, the host's model cache on bare
    # metal), for every config and later run. lm-eval's own cache key leaves
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
    LM_HARNESS_CACHE_PATH="$request_cache" lm_eval "${lm_eval_args[@]}" 2>&1 | stdbuf -oL tr '\r' '\n'

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
    evalscope eval \
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

# ========================================================
# 1. Initial Cleanup
# ========================================================
if [ "$START_SERVER" = "1" ]; then
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
# A dry run still writes the config.json the evals below read.
if [ "$SKIP_BENCHMARK" = "1" ]; then
  BENCH_ARGS+=(--dry-run)
fi

./scripts/vllm/benchmarking/run_benchmarks.sh "${BENCH_ARGS[@]}"

# Read model name from config
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
MMLU_PRO_DISABLE_MULTITURN_ARGS=$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("mmlu_pro_disable_multiturn_args", False)).lower())' "$RESULTS_DIR/config.json")
LM_EVAL_TASKS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("lm_eval_tasks") or "mmlu_llama mmlu_pro")' "$RESULTS_DIR/config.json")
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
# 4. Run lm_eval
# ========================================================
if [ "$RUN_LM_EVAL" = "1" ]; then
    # shellcheck disable=SC2086  # space-separated task list, split on purpose
    for task in $LM_EVAL_TASKS; do
        run_lm_eval "$task"
    done
fi

# ========================================================
# 5. Run code-generation evals (EvalPlus datasets via lm-eval)
# ========================================================
if [ "$RUN_CODE_EVAL" = "1" ]; then
    run_lm_eval "humaneval_plus_tpu"
    run_lm_eval "mbpp_plus_tpu"
fi

# ========================================================
# 6. Run multimodal evals (evalscope, live API endpoint)
# ========================================================
if [ "$RUN_MM_EVAL" = "1" ]; then
    if ! command -v evalscope &>/dev/null; then
        echo "::error::--run-mm-eval requires the 'evalscope' CLI (pip install evalscope)."
        fail=1
    else
        run_mm_eval "mmmu_pro"
    fi
fi

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

exit "$fail"

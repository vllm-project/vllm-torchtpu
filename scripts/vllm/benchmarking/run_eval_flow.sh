#!/bin/bash
# Full evaluation flow script: supports manual and CI execution
set -euo pipefail

PORT="${PORT:-8000}"
HOST=""
CONFIG_NAME=""
RESULTS_DIR=""
RUN_LM_EVAL=0
RUN_EVALPLUS=0
START_SERVER=1

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
        --run-evalplus)
            RUN_EVALPLUS=1
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --config CONFIG_NAME [--results-dir DIR] [--run-lm-eval] [--run-evalplus] [--host HOST] [--port PORT]"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Register cleanup trap
if [ "$START_SERVER" = "1" ]; then
    trap 'echo "Executing cleanup..."; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM
fi

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="/tmp/perf_eval_$CONFIG_NAME"
fi
mkdir -p "$RESULTS_DIR"

PERF_BASELINE="scripts/vllm/benchmarking/baselines/perf/$CONFIG_NAME.baseline.json"
EVALPLUS_BASELINE="scripts/vllm/benchmarking/baselines/evalplus/$CONFIG_NAME.baseline.json"

PERF_LOG="$RESULTS_DIR/regression_check.md"
EVALPLUS_LOG="$RESULTS_DIR/evalplus_check.md"

run_evalplus() {
    local model="$1"
    local results_dir="$2"
    local root="$results_dir/evalplus"
    local rc=0
    rm -rf "$root"
    mkdir -p "$root"

    # Defaults if not set in env
    local datasets="${EVALPLUS_DATASETS:-humaneval mbpp}"
    local parallel="${EVALPLUS_PARALLEL:-8}"

    for dataset in $datasets; do
        echo "EvalPlus dataset: $dataset"
        evalplus.evaluate "$dataset" \
            --model "$model" \
            --backend openai \
            --base_url "http://$HOST:$PORT/v1" \
            --root "$root" \
            --greedy \
            --n_samples 1 \
            --bs 1 \
            --parallel "$parallel" \
            2>&1 | tee "$root/$dataset.log"
        dataset_rc=${PIPESTATUS[0]}
        if [ "$dataset_rc" -ne 0 ]; then
            echo "ERROR: EvalPlus failed for $dataset (exit $dataset_rc)"
            rc="$dataset_rc"
        fi
    done
    return "$rc"
}

run_lm_eval() {
    local task="$1"
    local baseline="scripts/vllm/benchmarking/baselines/eval/${CONFIG_NAME}.${task}.baseline.json"
    local eval_log="${RESULTS_DIR}/eval_check_${task}.md"

    local lm_eval_args=(
        --model local-chat-completions
        --model_args "model=$MODEL,base_url=http://$HOST:$PORT/v1/chat/completions,num_concurrent=128"
        --tasks "$task"
        --apply_chat_template
        --limit 100
        --seed "0,1234,None,1234"
        --output_path "$RESULTS_DIR"
    )

    if [ "$task" = "mmlu_pro" ] && [ "$MMLU_PRO_DISABLE_MULTITURN_ARGS" = "true" ]; then
        lm_eval_args+=(--gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}')
    else
        lm_eval_args+=(--fewshot_as_multiturn true)
        lm_eval_args+=(--gen_kwargs '{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}')
    fi

    echo "Running lm_eval for task: $task..."
    echo "[cmd] lm_eval ${lm_eval_args[*]}"
    lm_eval "${lm_eval_args[@]}"

    if [ -f "$baseline" ]; then
        echo "=== Checking Eval Regression for $task ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode eval \
          --tolerance "$EVAL_TOLERANCE" \
          --results-dir "$RESULTS_DIR" \
          --baseline "$baseline" 2>&1 | tee "$eval_log" || { echo "::error::Eval regression check failed for $task! See logs above for details."; fail=1; }
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

./scripts/vllm/benchmarking/run_benchmarks.sh "${BENCH_ARGS[@]}"

# Read model name from config
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
MMLU_PRO_DISABLE_MULTITURN_ARGS=$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("mmlu_pro_disable_multiturn_args", False)).lower())' "$RESULTS_DIR/config.json")
EVAL_TOLERANCE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["eval_tolerance"])' "$RESULTS_DIR/config.json")

fail=0
evalplus_rc=0

# ========================================================
# 3. Check Perf Regression
# ========================================================
if [ -f "$PERF_BASELINE" ]; then
    echo "=== Checking Perf Regression ==="
    python3 scripts/vllm/benchmarking/check_regression.py \
      --mode perf \
      --results-dir "$RESULTS_DIR" \
      --baseline "$PERF_BASELINE" 2>&1 | tee "$PERF_LOG" || { echo "::error::Perf regression check failed! See logs above for details."; fail=1; }
fi

# ========================================================
# 4. Run lm_eval
# ========================================================
if [ "$RUN_LM_EVAL" = "1" ]; then
    run_lm_eval "mmlu_llama"
    run_lm_eval "mmlu_pro"
fi

# ========================================================
# 5. Run EvalPlus
# ========================================================
if [ "$RUN_EVALPLUS" = "1" ]; then
    echo "=== Running EvalPlus ==="
    run_evalplus "$MODEL" "$RESULTS_DIR" || evalplus_rc=$?

    if [ "$evalplus_rc" -ne 0 ]; then
        fail=1
    fi

    if [ -f "$EVALPLUS_BASELINE" ]; then
        echo "=== Checking EvalPlus Regression ==="
        python3 scripts/vllm/benchmarking/check_regression.py \
          --mode evalplus \
          --results-dir "$RESULTS_DIR" \
          --baseline "$EVALPLUS_BASELINE" 2>&1 | tee "$EVALPLUS_LOG" || { echo "::error::EvalPlus regression check failed! See logs above for details."; fail=1; }
    fi
fi

exit "$fail"

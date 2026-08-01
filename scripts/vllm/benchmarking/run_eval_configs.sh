#!/bin/bash
# Simple loop to execute assigned benchmarking config files across parallel jobs using modulo.
set -euo pipefail

job_index="${BUILDKITE_PARALLEL_JOB:-0}"
job_count="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs provided."
    exit 0
fi

fail=0
for (( i = 0; i < ${#configs[@]}; i++ )); do
    # Select configs where (index % job_count) == job_index
    if (( i % job_count != job_index )); then
        continue
    fi

    config_name="${configs[i]}"
    # Strip any directory path or .sh extension if passed
    config_name="$(basename "$config_name" .sh)"

    echo "=========================================================="
    echo "=== Running Perf/Eval ($(( i + 1 ))/${#configs[@]}): $config_name (Parallel Job: $job_index / $job_count) ==="
    echo "=========================================================="

    eval_args=(
        --config "$config_name"
        --run-lm-eval
        --results-dir "/perf_eval_results/$config_name"
    )
    if [ "${RUN_EVALPLUS:-0}" = "1" ]; then
        eval_args+=(--run-evalplus)
    fi

    if ! bash ./scripts/vllm/benchmarking/run_eval_flow.sh "${eval_args[@]}"; then
        echo "::error::Perf/Eval failed for config: $config_name"
        fail=1
    fi
done

exit "$fail"

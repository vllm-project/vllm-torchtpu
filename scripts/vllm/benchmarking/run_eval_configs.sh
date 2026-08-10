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

    # torch_tpu's tier-2 compilation cache accumulates in /dev/shm across
    # configs and can exhaust the container's --shm-size, killing the server
    # mid-run. Configs load different models, so there is no reuse to keep.
    rm -rf /dev/shm/torch_tpu_cache

    # Same story for vLLM's on-disk compile cache: torch 2.13 additionally
    # saves an AOT-compiled copy under torch_compile_cache/torch_aot_compile
    # (~1.3 GiB per rank per config), and the container overlay shares the
    # host disk. Left unpruned, later configs fail with ENOSPC.
    rm -rf /root/.cache/vllm/torch_compile_cache

    eval_args=(
        --config "$config_name"
        --run-lm-eval
        --results-dir "/perf_eval_results/$config_name"
    )
    if [ "${RUN_CODE_EVAL:-0}" = "1" ]; then
        eval_args+=(--run-code-eval)
    fi
    if [[ "$config_name" == *multimodal* ]]; then
        eval_args+=(--run-mm-eval)
    fi

    if ! bash ./scripts/vllm/benchmarking/run_eval_flow.sh "${eval_args[@]}"; then
        echo "::error::Perf/Eval failed for config: $config_name"
        fail=1
    fi
done

exit "$fail"

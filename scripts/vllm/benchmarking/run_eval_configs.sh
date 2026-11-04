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

# Simple loop to execute assigned benchmarking config files across parallel jobs using modulo.
set -euo pipefail

job_index="${BUILDKITE_PARALLEL_JOB:-0}"
job_count="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs provided."
    exit 0
fi

# The server log only says that a TPU runtime failed; libtpu's own logs say
# why (a port it could not bind, a slice member it lost). Saves the ones written
# since the marker file $1 into $2, next to the failed config's results.
tpu_log_dir="${TPU_LOG_DIR:-/tmp/tpu_logs}"
save_tpu_logs() {
    local since="$1" dest="$2" files=()
    [ -d "$tpu_log_dir" ] || return 0
    mapfile -t files < <(cd "$tpu_log_dir" && find . -type f -newer "$since")
    [ "${#files[@]}" -gt 0 ] || return 0
    mkdir -p "$dest"
    # Processes still exiting can grow a file mid-read; tar then exits 1 but
    # still writes the archive.
    tar -czf "$dest/tpu_logs.tar.gz" -C "$tpu_log_dir" "${files[@]}" || true
    echo "Saved ${#files[@]} libtpu log files to $dest/tpu_logs.tar.gz"
}

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

    results_dir="${ARTIFACTS_DIR:-/perf_eval_results}/$config_name"
    eval_args=(
        --config "$config_name"
        --results-dir "$results_dir"
    )
    # Perf-only steps set RUN_LM_EVAL=0.
    if [ "${RUN_LM_EVAL:-1}" = "1" ]; then
        eval_args+=(--run-lm-eval)
    fi
    if [ "${RUN_CODE_EVAL:-0}" = "1" ]; then
        eval_args+=(--run-code-eval)
    fi
    if [[ "$config_name" == *multimodal* ]]; then
        eval_args+=(--run-mm-eval)
    fi

    started="$(mktemp)"
    if ! bash ./scripts/vllm/benchmarking/run_eval_flow.sh "${eval_args[@]}"; then
        echo "::error::Perf/Eval failed for config: $config_name"
        fail=1
        save_tpu_logs "$started" "$results_dir"
    fi
    rm -f "$started"
done

exit "$fail"

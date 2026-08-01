#!/bin/bash
set -e

export BENCHMARK_WARMUP_RUNS=1

# Call the new run_benchmarks.sh script
bash scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-fp8-tp8-ep --results-dir /tmp/benchmark_results

# Create sentinel file
touch /tmp/benchmark_done

echo "Pausing for CI runner to fetch the file..."
sleep infinity

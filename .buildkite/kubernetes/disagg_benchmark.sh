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

# The client of the 1P1D disaggregated benchmark: brings up the proxy in front
# of the two engines, sweeps concurrency and publishes each cell.
#
# Run as the `benchmark` role of manifests/workloads/qwen3.5-397b-1p1d.yaml,
# which passes WORKLOAD_NAME. The engines are reached by JobSet's per-pod DNS,
# built from that name.

set -e
# The image ships libtpu, which makes vLLM's platform
# detection pick TPU on this CPU-only pod and fail.
pip uninstall -y libtpu libtpu-nightly >/dev/null 2>&1 || true

# JobSet's per-pod DNS; ${WORKLOAD_NAME} is unique to this
# run.
prefill_host=${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}
decode_host=${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}

echo "Starting proxy server..."
python3 /root/torchtpu-vllm/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 \
  --port 10000 \
  --prefiller-host "$prefill_host" \
  --prefiller-port 8000 \
  --decoder-host "$decode_host" \
  --decoder-port 8000 &
proxy_pid=$!

# The proxy is backgrounded, so its death cannot fail the
# script by itself and surfaces hours later as a benchmark
# whose requests all failed.
require_proxy() {
  kill -0 "$proxy_pid" 2>/dev/null && return 0
  echo "ERROR: the proxy (pid $proxy_pid) exited; nothing can reach the engines."
  exit 1
}

echo "Waiting for prefill ($prefill_host) and decode ($decode_host) servers..."
# Bounded wait: a server whose engine dies during startup
# can leave its API process (and /health) hanging forever.
# -f so an error status is not read as healthy, and -m so a
# hung connection cannot stretch an iteration past the ten
# seconds this count assumes.
health_tries=0
until curl -fsS -m 10 "http://$prefill_host:8000/health" && curl -fsS -m 10 "http://$decode_host:8000/health"; do
  require_proxy
  health_tries=$((health_tries+1))
  if [ "$health_tries" -gt 1080 ]; then
    echo "ERROR: servers not healthy after 3h; failing the benchmark."
    exit 1
  fi
  echo "Waiting for prefill and decode health checks..."
  sleep 10
done

echo "Both prefill and decode servers are HEALTHY!"

mkdir -p /tmp/benchmark_results
model="Qwen/Qwen3.5-397B-A17B-FP8"

# One record id for the whole sweep, suffixed per cell, so
# a night's rows can be selected together.
record_base="kube-vllm-torchtpu-run-$(date +%Y%m%d-%H%M%S)"
publish_failed=0

publish_result() {
  local path=$1
  [ -f "$path" ] || return 0
  local name
  name=$(basename "$path")

  # The artifact first, and unconditionally: it is the only
  # copy that survives a BigQuery outage. Uploaded once - Buildkite adds a
  # second artifact rather than replacing one of the same name, so a retry
  # pass shows up as a duplicate rather than a correction.
  (cd "$(dirname "$path")" && buildkite-agent artifact upload "$name") \
    || { echo "WARNING: could not upload $name"; publish_failed=1; }

  # A cell whose requests failed describes a broken serving
  # path, not a measurement: stop rather than spend the
  # remaining hours measuring the same breakage. The result
  # is single-line JSON with one top-level "failed", so a
  # grep is enough.
  if grep -q -E '"failed": *[1-9]' "$path"; then
    echo "ERROR: $name recorded failed requests: $(grep -oE '"(completed|failed)": *[0-9]+' "$path" | tr '\n' ' ')"
    exit 1
  fi

  if [ "${BQ_UPLOAD:-0}" != "1" ]; then
    echo "BigQuery upload is off for this lane; $name is an artifact only."
    return 0
  fi
  # Not fatal on its own - the artifact is already up - but
  # the step goes red at the end so a silent gap in the
  # table cannot pass.
  python3 /root/torchtpu-vllm/.buildkite/scripts/parse_gke_results.py \
    "$path" "${record_base}-${name%.json}" \
    || { echo "WARNING: BigQuery insert failed for $name"; publish_failed=1; }
}

# Client flags mirror tpu_benchmark_daily's bench_all.sh and
# bench_prefill_ttft.sh; changing one here makes this lane's
# numbers incomparable with those.
run_cell() {
  local filename=$1 concurrency=$2 input_len=$3 output_len=$4 num_prompts=$5 pmetrics=$6
  local path="/tmp/benchmark_results/$filename"
  echo "Starting benchmark cell: $filename (c=$concurrency, i=$input_len, o=$output_len, n=$num_prompts)"
  vllm bench serve \
    --backend=openai \
    --endpoint=/v1/completions \
    --model="$model" \
    --dataset-name=random \
    --random-input-len="$input_len" \
    --random-output-len="$output_len" \
    --random-range-ratio=0 \
    --num-prompts="$num_prompts" \
    --request-rate=inf \
    --max-concurrency="$concurrency" \
    --ignore-eos \
    --temperature=0 \
    --seed=42 \
    --percentile-metrics="$pmetrics" \
    --metric-percentiles=50,90,99 \
    --host=localhost \
    --port=10000 \
    --label="${filename%.json}" \
    --append-result \
    --result-file="$path"

  # Per cell rather than at the end, so a sweep that runs
  # out its deadline still yields the cells that finished.
  publish_result "$path"
  sleep 10
}

# Warms the prefill + transfer + decode path before any
# measured cell.
run_cell "pd_disagg_decode_smoke_c8_i8192_o32.json" 8 8192 32 8 "ttft,tpot,itl,e2el"

# Prefill throughput sweep (bench_all.sh).
for c in 8 16 32 64 128 256; do
  run_cell "pd_disagg_prefill_c${c}_i8192_o1.json" "$c" 8192 1 512 "ttft,e2el"
done

# Single-request TTFT sweep (bench_prefill_ttft.sh). Longer
# inputs than these exceed the servers' 66560 max-model-len.
for ilen in 8192 16384 32768 65536; do
  run_cell "pd_disagg_ttft_c1_i${ilen}_o1.json" 1 "$ilen" 1 16 "ttft,e2el"
done

for round in 1 2 3; do
  run_cell "pd_disagg_decode_round${round}_c8_i8192_o1024.json" 8 8192 1024 8 "ttft,tpot,itl,e2el"
done
for round in 1 2 3; do
  run_cell "pd_disagg_decode_daily_round${round}_c256_i65536_o1024.json" 256 65536 1024 256 "ttft,tpot,itl,e2el"
done

echo "Benchmark complete! Results saved in /tmp/benchmark_results."

# A sweep whose numbers are missing from the table looks
# identical to a sweep that never ran.
if [ "$publish_failed" -ne 0 ]; then
  echo "ERROR: the benchmark completed but at least one result was not published."
  exit 1
fi

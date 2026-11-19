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

# The client of the Qwen3.5-397B 1P1D disaggregated benchmark: waits for both
# engines, brings up vllm-router (or the toy proxy) in front of the pair, runs
# the speed cells, and runs the unified accuracy/speed flow via run_eval_flow.sh.
#
# Run as the chipless `benchmark` role of manifests/workloads/qwen3.5-397b-1p1d.yaml,
# which passes WORKLOAD_NAME. The engines are reached by JobSet's per-pod DNS,
# built from that name.

set -euo pipefail

# The image ships libtpu, which makes vLLM's platform detection pick TPU on
# this CPU-only pod and fail.
pip uninstall -y libtpu libtpu-nightly >/dev/null 2>&1 || true

REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-perf_eval_results}"
mkdir -p "$ARTIFACTS_DIR"
ARTIFACTS_DIR="$(cd "$ARTIFACTS_DIR" && pwd)"
export ARTIFACTS_DIR

prefill_host="${PREFILL_HOST:-${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}}"
decode_host="${DECODE_HOST:-${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}}"
engine_port="${ENGINE_PORT:-8000}"
proxy_port="${PROXY_PORT:-10000}"
proxy_kind="${DISAGG_PROXY:-router}"
case "$proxy_kind" in
  router|toy) ;;
  *)
    echo "ERROR: DISAGG_PROXY must be 'router' or 'toy', got '$proxy_kind'" >&2
    exit 2
    ;;
esac

proxy_pid=""
# shellcheck disable=SC2317  # invoked by the EXIT trap
cleanup() {
  if [ -n "$proxy_pid" ]; then
    kill "$proxy_pid" 2>/dev/null || true
  fi
  if [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] && command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*" \
      || echo "WARNING: could not upload $ARTIFACTS_DIR artifacts"
  fi
}
trap cleanup EXIT
# Exit on a signal so cleanup runs once, from the EXIT trap, instead of the
# script resuming after it.
trap 'exit 130' INT
trap 'exit 143' TERM

require_proxy() {
  kill -0 "$proxy_pid" 2>/dev/null && return 0
  echo "ERROR: the proxy (pid $proxy_pid) exited; nothing can reach the engines."
  exit 1
}

echo "Waiting for prefill ($prefill_host:$engine_port) and decode ($decode_host:$engine_port) servers..."
health_tries=0
until curl -fsS -m 10 "http://${prefill_host}:${engine_port}/health" \
  && curl -fsS -m 10 "http://${decode_host}:${engine_port}/health"; do
  health_tries=$((health_tries + 1))
  if [ "$health_tries" -gt 1080 ]; then
    echo "ERROR: servers not healthy after 3h; failing the benchmark."
    exit 1
  fi
  echo "Waiting for prefill and decode health checks..."
  sleep 10
done
echo "Both prefill and decode servers are HEALTHY!"

proxy_log="$ARTIFACTS_DIR/proxy.log"
echo "Starting $proxy_kind proxy server on port $proxy_port..."
if [ "$proxy_kind" = "router" ]; then
  vllm-router --host 0.0.0.0 --port "$proxy_port" \
    --vllm-pd-disaggregation \
    --prefill "http://${prefill_host}:${engine_port}" \
    --decode "http://${decode_host}:${engine_port}" \
    --policy consistent_hash \
    --request-timeout-secs 5000 \
    --worker-startup-timeout-secs 300 \
    --health-check-interval-secs 30 \
    --disable-retries \
    --prometheus-host 0.0.0.0 --prometheus-port "$((proxy_port + 1))" \
    >"$proxy_log" 2>&1 &
  proxy_pid=$!
  proxy_health="health"
else
  python3 "$REPO_DIR/examples/disagg/toy_proxy_server.py" \
    --host 0.0.0.0 \
    --port "$proxy_port" \
    --prefiller-host "$prefill_host" \
    --prefiller-port "$engine_port" \
    --decoder-host "$decode_host" \
    --decoder-port "$engine_port" \
    >"$proxy_log" 2>&1 &
  proxy_pid=$!
  proxy_health="healthcheck"
fi

proxy_tries=0
until curl -fsS -m 5 "http://127.0.0.1:${proxy_port}/${proxy_health}" >/dev/null 2>&1; do
  require_proxy
  proxy_tries=$((proxy_tries + 1))
  if [ "$proxy_tries" -gt 60 ]; then
    echo "ERROR: proxy did not become healthy on port $proxy_port"
    tail -50 "$proxy_log" || true
    exit 1
  fi
  sleep 2
done

bench_results_dir="${BENCHMARK_RESULTS_DIR:-/tmp/benchmark_results}"
mkdir -p "$bench_results_dir"
model="Qwen/Qwen3.5-397B-A17B-FP8"

record_base="kube-vllm-torchtpu-run-$(date +%Y%m%d-%H%M%S)"
publish_failed=0

publish_result() {
  local path=$1
  [ -f "$path" ] || return 0
  local name
  name=$(basename "$path")

  (cd "$(dirname "$path")" && buildkite-agent artifact upload "$name") \
    || { echo "WARNING: could not upload $name"; publish_failed=1; }

  if grep -q -E '"failed": *[1-9]' "$path"; then
    echo "ERROR: $name recorded failed requests: $(grep -oE '"(completed|failed)": *[0-9]+' "$path" | tr '\n' ' ')"
    exit 1
  fi

  if [ "${BQ_UPLOAD:-0}" != "1" ] || { [ -n "${BUILDKITE_BRANCH:-}" ] && { [ "$BUILDKITE_BRANCH" != "main" ] || [ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]; }; }; then
    echo "BigQuery upload is off for this lane; $name is an artifact only."
    return 0
  fi
  python3 "$REPO_DIR/.buildkite/scripts/parse_gke_results.py" \
    "$path" "${record_base}-${name%.json}" \
    || { echo "WARNING: BigQuery insert failed for $name"; publish_failed=1; }
}

run_cell() {
  local filename=$1 concurrency=$2 input_len=$3 output_len=$4 num_prompts=$5 pmetrics=$6
  local path="$bench_results_dir/$filename"
  require_proxy
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
    --host=127.0.0.1 \
    --port="$proxy_port" \
    --label="${filename%.json}" \
    --append-result \
    --result-file="$path"

  publish_result "$path"
  sleep 10
}

disagg_config="${DISAGG_CONFIG:-qwen3.5-397b-fp8-1p1d}"
disagg_baseline_config="${DISAGG_BASELINE_CONFIG:-qwen3.5-397b-fp8-tp1-dp8-ep}"
disagg_eval_suites="${DISAGG_EVAL_SUITES:-mmlu_pro,gsm8k,json_schema}"
disagg_perf_suites="${DISAGG_PERF_SUITES:-none}"

# Default runs the existing pd_disagg_* cells once and passes --perf-suites none
# to run_eval_flow.sh; setting DISAGG_PERF_SUITES (e.g. bench_serve) delegates
# both speed and accuracy to run_eval_flow.sh without duplicating TPU time.
if [ -z "$disagg_perf_suites" ] || [ "$disagg_perf_suites" = "none" ]; then
  run_cell "pd_disagg_decode_smoke_c8_i8192_o32.json" 8 8192 32 8 "ttft,tpot,itl,e2el"

  for c in 8 16 32 64 128 256; do
    run_cell "pd_disagg_prefill_c${c}_i8192_o1.json" "$c" 8192 1 512 "ttft,e2el"
  done

  for ilen in 8192 16384 32768 65536; do
    run_cell "pd_disagg_ttft_c1_i${ilen}_o1.json" 1 "$ilen" 1 16 "ttft,e2el"
  done

  for round in 1 2 3; do
    run_cell "pd_disagg_decode_round${round}_c8_i8192_o1024.json" 8 8192 1024 8 "ttft,tpot,itl,e2el"
  done
  for round in 1 2 3; do
    run_cell "pd_disagg_decode_daily_round${round}_c256_i65536_o1024.json" 256 65536 1024 256 "ttft,tpot,itl,e2el"
  done
  echo "Benchmark cells complete! Results saved in $bench_results_dir."
fi

eval_rc=0
if { [ -n "$disagg_eval_suites" ] && [ "$disagg_eval_suites" != "none" ]; } \
  || { [ -n "$disagg_perf_suites" ] && [ "$disagg_perf_suites" != "none" ]; }; then
  require_proxy
  echo "Running unified evaluation flow against router at http://127.0.0.1:${proxy_port}..."
  eval_args=(
    --config "$disagg_config"
    --baseline-config "$disagg_baseline_config"
    --base-url "http://127.0.0.1:${proxy_port}"
    --eval-suites "${disagg_eval_suites:-none}"
    --perf-suites "${disagg_perf_suites:-none}"
    --skip-db-upload
  )
  if [ -n "$disagg_perf_suites" ] && [ "$disagg_perf_suites" != "none" ]; then
    eval_args+=(--skip-perf-check)
  fi
  (
    # set -e is off left of "||", so chain the cd explicitly.
    cd "$REPO_DIR" &&
    bash "./scripts/vllm/benchmarking/run_eval_flow.sh" "${eval_args[@]}"
  ) || eval_rc=$?
fi

if [ "$publish_failed" -ne 0 ]; then
  echo "ERROR: the benchmark completed but at least one result was not published."
  exit 1
fi

exit "$eval_rc"

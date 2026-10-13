#!/bin/bash
# Serve a benchmark config and measure its decode throughput with
# bench_decode_sliding_window.py.
#
# Usage:
#   ./scripts/vllm/benchmarking/run_decode_benchmark.sh --config CONFIG_NAME
#       [--results-dir DIR] [--host HOST] [--port PORT] [--rounds N]
#       [--isl-osl "PREFILL:DECODE ..."] [--concurrencies "C ..."] [--no-warmup]
#
# The server is started with `run_benchmarks.sh --config CONFIG_NAME
# --serve-only`, so it gets exactly the config's launch (MODEL_URI,
# EXTRA_SERVE_ARGS and the other env overrides apply as usual), and stopped at
# exit. With --host, an already running server is benchmarked instead.
#
# The workload comes from the config: each PREFILL:DECODE pair in
# ISL_OSL_CONFIGS is run at each concurrency in CONCURRENCY_OPTIONS, and the
# config's MAX_NUM_BATCHED_TOKENS lets the benchmark reject a decode length too
# short for all requests to decode at once. --isl-osl and --concurrencies
# override the config's lists.
#
# Before the measured runs, a short warmup at each concurrency goes through
# the same token buckets, so no measured round pays for a first-use compile.
#
# Results land in <results-dir>/decode/isl<P>_osl<D>_c<C>/ (summary.md,
# summary.json, rounds.csv, ...), with a table of all runs in
# <results-dir>/decode/sweep.md. Exits non-zero if any run has a failed
# request or a round without a valid measurement window.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8000}"
HOST=""
CONFIG_NAME=""
RESULTS_DIR=""
ROUNDS=1
ISL_OSL_OVERRIDE=""
CONCURRENCIES_OVERRIDE=""
WARMUP=1

usage() {
    echo "Usage: $0 --config CONFIG_NAME [--results-dir DIR] [--host HOST] [--port PORT] [--rounds N] [--isl-osl \"P:D ...\"] [--concurrencies \"C ...\"] [--no-warmup]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --isl-osl) ISL_OSL_OVERRIDE="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES_OVERRIDE="$2"; shift 2 ;;
        --no-warmup) WARMUP=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$CONFIG_NAME" ]; then
    usage
    exit 1
fi
CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../../../benchmark_runs/decode_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR/decode"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
DECODE_DIR="$RESULTS_DIR/decode"

# Read the workload and server shape from the config in a subshell, so its
# exports do not leak into this script; run_benchmarks.sh sources it for the
# server itself.
eval "$(
    bash -c '
        source "$1" > /dev/null
        : "${DATA_PARALLELISM:=1}" "${SERVER_READY_WAIT_MIN:=90}"
        : "${DECODE_PRIME_PREFIX_CACHE:=false}" "${DECODE_MIN_FULL_OVERLAP_SECONDS:=0}"
        declare -p MODEL TENSOR_PARALLELISM DATA_PARALLELISM \
            MAX_NUM_BATCHED_TOKENS ISL_OSL_CONFIGS CONCURRENCY_OPTIONS \
            SERVER_READY_WAIT_MIN DECODE_PRIME_PREFIX_CACHE \
            DECODE_MIN_FULL_OVERLAP_SECONDS
    ' _ "$CONFIG_FILE"
)"
# Optional config knobs for the measured runs: DECODE_PRIME_PREFIX_CACHE=true
# primes each round's prompts into the prefix cache first (the config must
# enable prefix caching), and DECODE_MIN_FULL_OVERLAP_SECONDS rejects a round
# whose all-requests-decoding interval is shorter.
measure_args=(--min-full-overlap-seconds "$DECODE_MIN_FULL_OVERLAP_SECONDS")
if [ "$DECODE_PRIME_PREFIX_CACHE" = "true" ]; then
    measure_args+=(--prime-prefix-cache)
fi
ISL_OSL_CONFIGS="${ISL_OSL_OVERRIDE:-$ISL_OSL_CONFIGS}"
CONCURRENCY_OPTIONS="${CONCURRENCIES_OVERRIDE:-$CONCURRENCY_OPTIONS}"

SERVER_URL="http://${HOST:-localhost}:$PORT"
SERVE_PID=""

# Invoked by the EXIT trap below; shellcheck 0.10 reports that as SC2317 and
# later versions as SC2329.
# shellcheck disable=SC2317,SC2329
stop_server() {
    if [ -n "$SERVE_PID" ]; then
        echo "Stopping the server..."
        # run_benchmarks.sh stops vllm serve from its own EXIT trap.
        kill -TERM "$SERVE_PID" 2>/dev/null || true
        wait "$SERVE_PID" 2>/dev/null || true
        bash "$SCRIPT_DIR/cleanup_server.sh" > /dev/null 2>&1 || true
        SERVE_PID=""
    fi
}

if [ -z "$HOST" ]; then
    trap stop_server EXIT INT TERM
    bash "$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" \
        --results-dir "$RESULTS_DIR" --port "$PORT" --serve-only \
        > "$RESULTS_DIR/serve.log" 2>&1 &
    SERVE_PID=$!
    echo "Starting $CONFIG_NAME (pid=$SERVE_PID); log: $RESULTS_DIR/serve.log, $RESULTS_DIR/server.log"
    max_wait=$(( SERVER_READY_WAIT_MIN * 60 ))
    waited=0
    until curl -s -o /dev/null --connect-timeout 1 "$SERVER_URL/health"; do
        if ! kill -0 "$SERVE_PID" 2>/dev/null; then
            echo "ERROR: the server exited during startup."
            tail -50 "$RESULTS_DIR/serve.log"
            [ -f "$RESULTS_DIR/server.log" ] && tail -200 "$RESULTS_DIR/server.log"
            exit 1
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: the server did not become ready within $SERVER_READY_WAIT_MIN minutes."
            exit 1
        fi
        sleep 10
        waited=$(( waited + 10 ))
    done
    echo "Server ready after ${waited}s"
fi

# Like run_benchmarks.sh: with MODEL_URI the server pulls the tokenizer into
# ObjectStorageModel(MODEL_URI).dir, so tokenize from there rather than the
# HF hub. TOKENIZER overrides; MODEL is the fallback.
TOKENIZER="${TOKENIZER:-}"
if [ -z "$TOKENIZER" ] && [ -n "${MODEL_URI:-}" ]; then
    pulled_dir=$(python3 -c 'import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)' "$MODEL_URI" 2>/dev/null | tail -1 || true)
    if [ -n "$pulled_dir" ] && [ -f "$pulled_dir/tokenizer_config.json" ]; then
        TOKENIZER="$pulled_dir"
    fi
fi
TOKENIZER="${TOKENIZER:-$MODEL}"

bench() {
    python3 "$SCRIPT_DIR/bench_decode_sliding_window.py" \
        --base-url "$SERVER_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
        --tensor-parallel-size "$TENSOR_PARALLELISM" \
        --data-parallel-size "$DATA_PARALLELISM" "$@"
}

echo "Decode benchmark: config=$CONFIG_NAME model=$MODEL isl_osl=[$ISL_OSL_CONFIGS] concurrency=[$CONCURRENCY_OPTIONS] rounds=$ROUNDS tokenizer=$TOKENIZER"

if [ "$WARMUP" = "1" ]; then
    # One full prefill step plus a tail chunk, then a short decode. Too short
    # for a valid measurement, so the exit status is ignored.
    mkdir -p "$DECODE_DIR/warmup"
    for concurrency in $CONCURRENCY_OPTIONS; do
        echo "Warmup at concurrency $concurrency"
        bench --output-dir "$DECODE_DIR/warmup/c$concurrency" \
            --concurrency "$concurrency" \
            --prefill-tokens $(( MAX_NUM_BATCHED_TOKENS + 1024 )) \
            --decode-tokens 64 --rounds 1 --window-seconds 1 --step-seconds 0.5 \
            > "$DECODE_DIR/warmup/c$concurrency.log" 2>&1 || true
    done
fi

exit_code=0
run_names=()
for isl_osl in $ISL_OSL_CONFIGS; do
    IFS=':' read -r prefill_tokens decode_tokens <<< "$isl_osl"
    for concurrency in $CONCURRENCY_OPTIONS; do
        name="isl${prefill_tokens}_osl${decode_tokens}_c${concurrency}"
        run_names+=("$name")
        echo "=== $name ==="
        if ! bench --output-dir "$DECODE_DIR/$name" \
            --concurrency "$concurrency" \
            --prefill-tokens "$prefill_tokens" --decode-tokens "$decode_tokens" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --rounds "$ROUNDS" "${measure_args[@]}" 2>&1 | tee "$DECODE_DIR/$name.log"; then
            echo "ERROR: $name failed"
            exit_code=1
        fi
    done
done

python3 - "$DECODE_DIR" "${run_names[@]}" <<'PY' | tee "$DECODE_DIR/sweep.md"
import json
import sys
from pathlib import Path


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


decode_dir = Path(sys.argv[1])
print(
    "|run|valid rounds|peak window tok/s|peak window TPOT P50 (ms)|"
    "full-overlap tok/s|TTFT P50 (s)|running max|waiting max|KV usage max|"
    "preemptions|"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in sys.argv[2:]:
    path = decode_dir / name / "summary.json"
    if not path.exists():
        print(f"|{name}|no summary|||||||||")
        continue
    summary = json.loads(path.read_text())
    best = summary["best"] or {}
    rows = summary["results"]

    def peak(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return max(values) if values else None

    print(
        f"|{name}|{summary['valid_rounds']}/{summary['requested_rounds']}|"
        f"{fmt(best.get('output_throughput'))}|{fmt(best.get('mean_tpot_ms'))}|"
        f"{fmt(best.get('full_overlap_throughput'))}|"
        f"{fmt(summary['aggregate']['ttft_p50_s']['p50'])}|"
        f"{fmt(peak('server_running_requests_max'))}|"
        f"{fmt(peak('server_waiting_requests_max'))}|"
        f"{fmt(peak('server_kv_cache_usage_max'))}|"
        f"{fmt(peak('server_preemptions'))}|"
    )
PY

exit "$exit_code"

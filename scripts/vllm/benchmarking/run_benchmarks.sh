#!/bin/bash
# Self-contained benchmark runner for torchtpu-vllm.
# Starts a vLLM server, runs benchmark_serving.py for each ISL/OSL x concurrency
# combo, saves results locally.
#
# Usage:
#   ./scripts/vllm/benchmarking/run_benchmarks.sh [--config CONFIG_NAME] [--dry-run] [--keep-alive]
#
# Examples:
#   ./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-smoke
#   ./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep
#
# Available configs (in scripts/vllm/benchmarking/configs/):
#   qwen3-coder-480b-fp8-tp8-ep  - Nightly target (TP=8, EP, FP8)
#   qwen3-coder-30b-tp8-ep       - Guard: Qwen3-Coder-30B-A3B-Instruct (TP=8, EP)
#   qwen3-coder-30b-fp8-tp8-ep   - Guard: Qwen3-Coder-30B-A3B-Instruct-FP8 (TP=8, EP)
#
# Profiler capture (diagnostic only):
#   CAPTURE_PROFILE=1     turn on TPU/XProf trace capture for the bench window.
#                         Server gets --profiler-config.profiler torch +
#                         torch_profiler_dir + ignore_frontend (the last skips
#                         AsyncLLM's CPU profiler we don't read). Traces go to
#                         <results>/profile. Implied when EXTRA_SERVE_ARGS sets
#                         --profiler-config.torch_profiler_dir, which then also
#                         chooses the trace directory; CAPTURE_PROFILE=0 opts
#                         back out. Bench gets --profile so /start_profile
#                         and /stop_profile fire
#                         around the main run; warmup runs aren't profiled.
#                         --profile costs roughly +11% TTFT and −4% throughput,
#                         so don't compare profile-on numbers to a non-profile
#                         baseline. Prefer a single ISL/OSL case for clean
#                         attribution. Recorded as capture_profile + profile_dir
#                         in config.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# =============================================================================
# Parse arguments
# =============================================================================
CONFIG_NAME="qwen3-coder-480b-fp8-tp8-ep"
DRY_RUN=0
RESULTS_DIR_OVERRIDE="${RESULTS_DIR:-}"
KEEP_ALIVE=0
HOST=""
PORT="${PORT:-8000}"
START_SERVER=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            echo "*** DRY RUN MODE: commands will be printed but not executed ***"
            shift
            ;;
        --keep-alive)
            KEEP_ALIVE=1
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
            echo "Usage: $0 [--config CONFIG_NAME] [--results-dir DIR] [--dry-run] [--keep-alive] [--host HOST] [--port PORT]"
            exit 1
            ;;
    esac
done

if [ -z "$HOST" ]; then
    HOST="localhost"
fi

# =============================================================================
# Load config
# =============================================================================
CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    find "$SCRIPT_DIR/configs/" -name '*.sh' -print0 2>/dev/null | xargs -0 -I{} basename {} .sh
    exit 1
fi

# Defaults (config can override)
MODEL=""
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="512:512"
CONCURRENCY_OPTIONS="1"
RANDOM_RANGE_RATIO=""
NUM_PROMPTS=""
# How RANDOM_RANGE_RATIO is interpreted:
#   symmetric - vllm bench serve native: sample lengths in [(1-r)*len, (1+r)*len]
#   min       - benchmark_serving.py / inferenceX client style: sample in
#               [r*len, len]. Translated to vllm bench serve's symmetric form
#               below. Sampled lengths never exceed the nominal length, so
#               nominal ISL+OSL <= max-model-len guarantees full completion.
RANGE_RATIO_STYLE="symmetric"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-0}"
# Empty => do not pass --temperature, so the server default applies
# (for Qwen3-Coder the model's generation_config enables sampling =>
# non-greedy: temp 0.7 / top_p 0.8 / top_k 20).
# Set to 0 for deterministic greedy decoding.
BENCHMARK_TEMPERATURE="${BENCHMARK_TEMPERATURE:-}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=false
EVAL_TOLERANCE=""
# Ratio tolerance for the perf regression gate (per-metric floors in
# check_regression.py still apply). Raise per config for noisier layouts.
PERF_TOLERANCE="0.05"

# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [ -z "$MODEL" ]; then
    echo "ERROR: Config must set MODEL"
    exit 1
fi
if [ -z "$RANDOM_RANGE_RATIO" ]; then
    echo "ERROR: Config must set RANDOM_RANGE_RATIO"
    exit 1
fi
if [ -z "$EVAL_TOLERANCE" ]; then
    echo "ERROR: Config must set EVAL_TOLERANCE"
    exit 1
fi
if [ "$RANGE_RATIO_STYLE" != "symmetric" ] && [ "$RANGE_RATIO_STYLE" != "min" ]; then
    echo "ERROR: RANGE_RATIO_STYLE must be 'symmetric' or 'min'"
    exit 1
fi
if ! [[ "$BENCHMARK_WARMUP_RUNS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: BENCHMARK_WARMUP_RUNS must be a non-negative integer"
    exit 1
fi

PORT="${PORT:-8000}"

# Verify vllm bench serve is available
if ! python3 -m vllm.entrypoints.cli.main bench serve --help &>/dev/null; then
    echo "ERROR: 'vllm bench serve' not available. Is vLLM installed?"
    exit 1
fi
echo "Using benchmark tool: vllm bench serve"

# =============================================================================
# Setup results directory
# =============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
safe_model=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
else
    RESULTS_DIR="$REPO_DIR/benchmark_runs/${safe_model}_tp${TENSOR_PARALLELISM}_${TIMESTAMP}"
fi
mkdir -p "$RESULTS_DIR"
log_file="$RESULTS_DIR/benchmark.log"

# Resolve profile state once so config.json and start_vllm_server agree.
# Configuring profiler_config by hand is itself a request to profile, so treat
# a caller-supplied torch_profiler_dir as CAPTURE_PROFILE=1 when the latter was
# not set either way. Otherwise the server is armed but the bench client never
# sends --profile, so /start_profile never fires and the run writes no traces.
# An explicit CAPTURE_PROFILE (0 or 1) always wins.
requested_profile_dir=$(printf '%s' "${EXTRA_SERVE_ARGS:-}" |
    sed -n 's/.*--profiler-config\.torch_profiler_dir[= ]\+\([^ ]\+\).*/\1/p')
if [ -z "${CAPTURE_PROFILE:-}" ] && [ -n "$requested_profile_dir" ]; then
    CAPTURE_PROFILE=1
    echo "Profiling inferred from --profiler-config.torch_profiler_dir in" \
        "EXTRA_SERVE_ARGS. Set CAPTURE_PROFILE=0 to override."
fi

PROFILE_DIR=""
if [ "${CAPTURE_PROFILE:-0}" = "1" ]; then
    # A caller-supplied directory wins over the default: the block appended in
    # start_vllm_server lands after EXTRA_SERVE_ARGS and argparse takes the
    # last occurrence, so otherwise the chosen directory is silently ignored.
    PROFILE_DIR="${requested_profile_dir:-$RESULTS_DIR/profile}"
    mkdir -p "$PROFILE_DIR"
fi

start_vllm_server() {
    local max_model_len=$1
    local max_num_batched_tokens=$2
    local max_num_seqs=$3
    local gpu_mem_util="${GPU_MEMORY_UTILIZATION:-0.95}"

    # Build extra args
    local extra_args=""
    if [ "$ENABLE_EP" = "true" ]; then
        extra_args="--enable-expert-parallel"
    fi
    if [ -n "$QUANTIZATION" ]; then
        extra_args="$extra_args --quantization $QUANTIZATION"
    fi
    # Default to the batched-RPA Pallas kernel (PallasBatchedRPAAttentionBackend).
    # Without --attention-backend CUSTOM, vLLM falls back to FLASH_ATTN with
    # block_size=16 and benchmarks regress ~12% (see PR #191 / 9467829).
    local attn_backend="${ATTENTION_BACKEND:-CUSTOM}"
    extra_args="$extra_args --attention-backend $attn_backend"
    if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
        extra_args="$extra_args $EXTRA_SERVE_ARGS"
    fi

    export PYTHONUNBUFFERED=1

    if [ -n "$PROFILE_DIR" ]; then
        # The --profiler-config flags gate vllm's /start_profile endpoint
        # and satisfy its ProfilerConfig validator. ignore_frontend skips
        # the AsyncLLM CPU profiler (we only read the TPU xplane).
        extra_args="$extra_args --profiler-config.profiler torch --profiler-config.torch_profiler_dir $PROFILE_DIR --profiler-config.ignore_frontend true"
        echo "Profiler capture enabled. Traces will be written to: $PROFILE_DIR"
    fi

    local prefix_caching_flag="--no-enable-prefix-caching"
    if [ "${ENABLE_PREFIX_CACHING:-false}" = "true" ]; then
        prefix_caching_flag="--enable-prefix-caching"
    fi
    local kv_cache_dtype="${KV_CACHE_DTYPE:-fp8}"
    local server_cmd="vllm serve ${MODEL} --tensor-parallel-size=$TENSOR_PARALLELISM --data-parallel-size=$DATA_PARALLELISM --max-model-len=$max_model_len --max-num-batched-tokens=$max_num_batched_tokens --max-num-seqs=$max_num_seqs --port $PORT --async-scheduling $prefix_caching_flag --gpu-memory-utilization=$gpu_mem_util --kv-cache-dtype=$kv_cache_dtype $extra_args"

    echo ""
    echo "================================================"
    echo "Starting vLLM server"
    echo "  Model: $MODEL"
    echo "  TP=$TENSOR_PARALLELISM DP=$DATA_PARALLELISM EP=$ENABLE_EP"
    echo "  max_model_len=$max_model_len"
    echo "  max_num_batched_tokens=$max_num_batched_tokens"
    echo "  max_num_seqs=$max_num_seqs"
    echo "  gpu_mem_util=$gpu_mem_util"
    echo "================================================"
    echo "[cmd] $server_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] Skipping server launch"
        return
    fi

    if curl -s -o /dev/null --connect-timeout 1 "http://localhost:$PORT/health" 2>/dev/null; then
        echo "WARNING: Port $PORT is already occupied! Running cleanup_server.sh before launching new server..."
        bash "$SCRIPT_DIR/cleanup_server.sh"
    fi

    $server_cmd >> "$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server started (pid=$SERVER_PID)"
    echo "Server log: $RESULTS_DIR/server.log"

    # Wait for server to be ready (env-overridable; large hybrid models need a
    # long AOT-precompile window before /health is up).
    local max_wait=$(( ${SERVER_READY_WAIT_MIN:-90} * 60 ))
    local waited=0
    while ! curl -s -o /dev/null --connect-timeout 1 "http://localhost:$PORT/health" 2>/dev/null; do
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: Server did not become ready within $((max_wait / 60)) minutes."
            stop_vllm_server
            exit 1
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: Server process died during startup."
            echo "========== Last 200 lines of server.log =========="
            tail -200 "$RESULTS_DIR/server.log"
            echo "==================================================="
            exit 1
        fi
        if [ "$waited" -gt 0 ] && [ $((waited % 60)) -eq 0 ]; then
            echo "Waiting for server... (${waited}s elapsed)"
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "Server ready on port $PORT (${waited}s)"
}

stop_vllm_server() {
    if [ "$DRY_RUN" = "1" ]; then
        return
    fi
    if [ "$KEEP_ALIVE" = "1" ]; then
        echo ""
        echo "Keeping vLLM server alive in background (--keep-alive)."
        return
    fi
    echo ""
    echo "Stopping vLLM server..."
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    sleep 2
    rm -f /tmp/libtpu_lockfile*
    echo "Server stopped."
}

run_benchmark_once() {
    local input_len=$1
    local output_len=$2
    local concurrency=$3
    local result_file=$4
    local bench_log=$5
    local is_warmup=${6:-0}

    # min-style ratios sample in [r*len, len]; vllm bench serve only supports
    # the symmetric [(1-h)*mid, (1+h)*mid], so pass mid = len*(1+r)/2 and
    # h = (1-r)/(1+r), which spans exactly the same range.
    local bench_input_len=$input_len
    local bench_output_len=$output_len
    local bench_range_ratio=$RANDOM_RANGE_RATIO
    if [ "$RANGE_RATIO_STYLE" = "min" ]; then
        bench_input_len=$(awk -v l="$input_len" -v r="$RANDOM_RANGE_RATIO" 'BEGIN{printf "%.0f", l*(1+r)/2}')
        bench_output_len=$(awk -v l="$output_len" -v r="$RANDOM_RANGE_RATIO" 'BEGIN{printf "%.0f", l*(1+r)/2}')
        bench_range_ratio=$(awk -v r="$RANDOM_RANGE_RATIO" 'BEGIN{printf "%.6f", (1-r)/(1+r)}')
    fi

    local profile_arg=""
    if [ "${CAPTURE_PROFILE:-0}" = "1" ] && [ "$is_warmup" != "1" ]; then
        profile_arg="--profile"
    fi

    # Force greedy (or any fixed temperature) when BENCHMARK_TEMPERATURE is set.
    # Otherwise vllm bench serve sends no temperature and the server's
    # generation_config decides sampling.
    local temperature_arg=()
    if [ -n "${BENCHMARK_TEMPERATURE:-}" ]; then
        temperature_arg=(--temperature "$BENCHMARK_TEMPERATURE")
    fi

    set +e
    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len "$bench_input_len" \
        --random-output-len "$bench_output_len" \
        --random-range-ratio "$bench_range_ratio" \
        --num-prompts "${NUM_PROMPTS:-320}" \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --ignore-eos \
        --result-filename "$result_file" \
        $profile_arg \
        "${temperature_arg[@]}" \
        --seed 42 2>&1 | tee -a "$bench_log"
    local bench_exit=${PIPESTATUS[0]}
    set -e

    return "$bench_exit"
}

# =============================================================================
# Compute server parameters from config
# =============================================================================
# Covers the worst-case sampled length for ISL/OSL=8192 with --random-range-ratio=0.8
# (max ≈ 1.8*8192 + 1.8*1024 = 16590); pre-16384 runs silently dropped ~40% of these.
max_model_len=${MAX_MODEL_LEN:-16384}
max_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
max_num_seqs=${MAX_NUM_SEQS:-512}

# JSON-encode capture_profile + profile_dir for config.json.
capture_profile_json=${CAPTURE_PROFILE:-0}
if [ -n "$PROFILE_DIR" ]; then
    profile_dir_json="\"$PROFILE_DIR\""
else
    profile_dir_json=null
fi
if [ -n "$BENCHMARK_TEMPERATURE" ]; then
    benchmark_temperature_json=$BENCHMARK_TEMPERATURE
else
    benchmark_temperature_json=null
fi

# Save config metadata
cat > "$RESULTS_DIR/config.json" << EOF
{
    "model": "$MODEL",
    "tensor_parallelism": $TENSOR_PARALLELISM,
    "data_parallelism": $DATA_PARALLELISM,
    "enable_ep": $ENABLE_EP,
    "quantization": "${QUANTIZATION:-none}",
    "isl_osl_configs": "$ISL_OSL_CONFIGS",
    "concurrency_options": "$CONCURRENCY_OPTIONS",
    "num_prompts": ${NUM_PROMPTS:-320},
    "random_range_ratio": $RANDOM_RANGE_RATIO,
    "range_ratio_style": "$RANGE_RATIO_STYLE",
    "benchmark_warmup_runs": $BENCHMARK_WARMUP_RUNS,
    "benchmark_temperature": $benchmark_temperature_json,
    "max_model_len": $max_model_len,
    "max_num_batched_tokens": $max_batched_tokens,
    "max_num_seqs": $max_num_seqs,
    "capture_profile": $capture_profile_json,
    "profile_dir": $profile_dir_json,
    "timestamp": "$TIMESTAMP",
    "mmlu_pro_disable_multiturn_args": $MMLU_PRO_DISABLE_MULTITURN_ARGS,
    "perf_tolerance": $PERF_TOLERANCE,
    "eval_tolerance": $EVAL_TOLERANCE
}
EOF

# =============================================================================
# Run benchmarks
# =============================================================================
echo ""
echo "================================================"
echo "Benchmark Configuration"
echo "  Config: $CONFIG_NAME"
echo "  Model: $MODEL"
echo "  TP=$TENSOR_PARALLELISM DP=$DATA_PARALLELISM EP=$ENABLE_EP"
echo "  ISL/OSL: $ISL_OSL_CONFIGS"
echo "  Concurrency: $CONCURRENCY_OPTIONS"
echo "  Num prompts: ${NUM_PROMPTS:-320}"
echo "  Benchmark warmup runs: $BENCHMARK_WARMUP_RUNS"
echo "  Benchmark temperature: ${BENCHMARK_TEMPERATURE:-<server default (non-greedy)>}"
echo "  Results: $RESULTS_DIR"
echo "================================================"

{
    echo "========================================"
    echo "START TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "MODEL: $MODEL"
    echo "CONFIG: $CONFIG_NAME"
    echo "TP: $TENSOR_PARALLELISM  DP: $DATA_PARALLELISM  EP: $ENABLE_EP"
    echo "========================================"
} > "$log_file"

# Large models need a long engine-core handshake window for AOT precompile
# (vLLM's default is 600s). Matches the golden cmds' setting.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

if [ "$START_SERVER" = "1" ]; then
    trap stop_vllm_server EXIT INT TERM
    start_vllm_server "$max_model_len" "$max_batched_tokens" "$max_num_seqs"
fi

exit_code=0

for isl_osl_config in $ISL_OSL_CONFIGS; do
    IFS=':' read -r input_len output_len <<< "$isl_osl_config"
    if [ -z "$input_len" ] || [ -z "$output_len" ]; then
        echo "ERROR: Invalid ISL/OSL config: $isl_osl_config"
        exit 1
    fi
    total=$((input_len + output_len))
    if [ "$total" -gt "$max_model_len" ]; then
        echo "Skipping isl${input_len}_osl${output_len} (total=$total > max_model_len=$max_model_len)"
        continue
    fi

    echo ""
    echo "ISL/OSL: ${input_len}/${output_len}"
    echo "------------------------------------------------"

    for concurrency in $CONCURRENCY_OPTIONS; do
        base_name="isl${input_len}_osl${output_len}_c${concurrency}"
        result_file="${RESULTS_DIR}/${base_name}.json"
        bench_log="${RESULTS_DIR}/${base_name}.log"

        echo "  Concurrency=$concurrency"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY RUN] Skipping"
            continue
        fi

        for ((warmup_idx = 1; warmup_idx <= BENCHMARK_WARMUP_RUNS; warmup_idx++)); do
            warmup_file="${RESULTS_DIR}/${base_name}.warmup${warmup_idx}.json"
            warmup_log="${RESULTS_DIR}/${base_name}.warmup${warmup_idx}.log"
            echo "    Warmup run $warmup_idx/$BENCHMARK_WARMUP_RUNS"
            if run_benchmark_once "$input_len" "$output_len" "$concurrency" "$warmup_file" "$warmup_log" 1; then
                warmup_exit=0
            else
                warmup_exit=$?
            fi
            if [ "$warmup_exit" -ne 0 ]; then
                echo "    WARMUP FAILED (exit $warmup_exit)"
                exit_code=$warmup_exit
                if [ "$START_SERVER" = "1" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
                    echo "ERROR: Server died during benchmark warmup"
                    tail -50 "$RESULTS_DIR/server.log"
                    break 3
                fi
                break
            fi
            echo "    Warmup OK -> $warmup_file"
        done

        if [ "$exit_code" -ne 0 ]; then
            continue
        fi

        if run_benchmark_once "$input_len" "$output_len" "$concurrency" "$result_file" "$bench_log"; then
            bench_exit=0
        else
            bench_exit=$?
        fi
        if [ "$bench_exit" -eq 0 ]; then
            echo "    OK -> $result_file"
        else
            echo "    FAILED (exit $bench_exit)"
            exit_code=$bench_exit
            if [ "$START_SERVER" = "1" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "ERROR: Server died during benchmark"
                tail -50 "$RESULTS_DIR/server.log"
                break 2
            fi
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================
{
    echo ""
    echo "========================================"
    echo "END TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "EXIT CODE: $exit_code"
    echo "========================================"
} >> "$log_file"

echo ""
echo "================================================"
if [ "$exit_code" -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Benchmark failed (exit code: $exit_code)"
fi
echo "Results: $RESULTS_DIR"
echo "================================================"

exit "$exit_code"

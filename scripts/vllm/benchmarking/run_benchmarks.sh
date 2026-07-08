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
#                         AsyncLLM's CPU profiler we don't read). Bench gets
#                         --profile so /start_profile and /stop_profile fire
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
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-0}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=false
EVAL_TOLERANCE=""

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
PROFILE_DIR=""
if [ "${CAPTURE_PROFILE:-0}" = "1" ]; then
    PROFILE_DIR="$RESULTS_DIR/profile"
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
    export MODEL_IMPL_TYPE=vllm

    if [ -n "$PROFILE_DIR" ]; then
        # tllm's worker reads VLLM_TORCH_PROFILER_DIR; the --profiler-config
        # flags are what gate vllm's /start_profile endpoint and satisfy its
        # ProfilerConfig validator. ignore_frontend skips the AsyncLLM CPU
        # profiler (we only read the TPU xplane).
        export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
        extra_args="$extra_args --profiler-config.profiler torch --profiler-config.torch_profiler_dir $VLLM_TORCH_PROFILER_DIR --profiler-config.ignore_frontend true"
        echo "Profiler capture enabled. Traces will be written to: $VLLM_TORCH_PROFILER_DIR"
    fi

    local server_cmd="vllm serve --model=${MODEL} --tensor-parallel-size=$TENSOR_PARALLELISM --data-parallel-size=$DATA_PARALLELISM --max-model-len=$max_model_len --max-num-batched-tokens=$max_num_batched_tokens --max-num-seqs=$max_num_seqs --port $PORT --async-scheduling --no-enable-prefix-caching --gpu-memory-utilization=$gpu_mem_util --kv-cache-dtype=fp8 $extra_args"

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

    local profile_arg=""
    if [ "${CAPTURE_PROFILE:-0}" = "1" ] && [ "$is_warmup" != "1" ]; then
        profile_arg="--profile"
    fi

    set +e
    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts 320 \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --save-result \
        --ignore-eos \
        --result-filename "$result_file" \
        $profile_arg \
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
    "benchmark_warmup_runs": $BENCHMARK_WARMUP_RUNS,
    "max_model_len": $max_model_len,
    "max_num_batched_tokens": $max_batched_tokens,
    "max_num_seqs": $max_num_seqs,
    "capture_profile": $capture_profile_json,
    "profile_dir": $profile_dir_json,
    "timestamp": "$TIMESTAMP",
    "mmlu_pro_disable_multiturn_args": $MMLU_PRO_DISABLE_MULTITURN_ARGS,
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
echo "  Benchmark warmup runs: $BENCHMARK_WARMUP_RUNS"
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

# Use uniform random MoE routing for consistent benchmarking
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random

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

#!/usr/bin/env bash
# Kimi-K3 InferenceX AgentX sweep on the v7x-32 pod. It starts one server from
# --config, runs agentx_client.sh against it, and stops it again. The cases -
# which concurrencies to replay, and the MAX_NUM_SEQS the server needs to serve
# them - are declared in the config, so one file says what is swept.
#
#   agentx_sweep.sh [--config kimi-k3-tp32-ep-agentx]
#       [--max-model-len 131072] [--kv fp8|auto] [--apc 0|1]
#       [--spec none|dspark] [--moe-requant 0|512] [--extra-serve-args "..."]
#       [--duration 1800] [--fast 1|0] [--ix-ref main]
#       [--gsm8k-limit 100] [--gsm8k-conc 8[,16,...]]
#       [--results-dir DIR] [--gcs-root gs://bucket/prefix]
#
# There is no flag for the cases: a one-off run edits AGENTX_CONC_LIST in the
# config, which keeps the list and the MAX_NUM_SEQS that has to cover it in one
# file and under review. This script checks the two agree rather than deriving
# one from the other, so a server too small for the ladder is a clear failure
# here instead of a scheduler bottleneck in the numbers.
#
# One server serves the whole ladder; see agentx_client.sh for why that is valid
# and for the client side. The kube 1P1D lane does not use this script: its
# servers are already running on TPU pods and it calls agentx_client.sh directly.
#
# Exit status: 1 if the server failed to boot or died, or if the client failed.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-agentx"
RESULTS_DIR="/perf_eval_results/k3-agentx-sweep"
MAX_MODEL_LEN_ARG=131072
DURATION=1800
FAST=1
IX_REF="main"
APC=0
KV_DTYPE="fp8"
SPEC="none"            # none | dspark
SPEC_TOKENS=7
SPEC_AL=3.84
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"
MOE_REQUANT=0          # 0 = W4A16 (default gmm_v2 path); 512 = W4A8 (fp8 activations)
PORT=8000
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN_ARG="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --fast)          FAST="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        --apc)           APC="$2"; shift 2 ;;
        --kv)            KV_DTYPE="$2"; shift 2 ;;
        --spec)          SPEC="$2"; shift 2 ;;
        --spec-tokens)   SPEC_TOKENS="$2"; shift 2 ;;
        --spec-al)       SPEC_AL="$2"; shift 2 ;;
        --gsm8k-limit)   GSM8K_LIMIT="$2"; shift 2 ;;
        --gsm8k-conc)    GSM8K_CONC="$2"; shift 2 ;;
        --moe-requant)   MOE_REQUANT="$2"; shift 2 ;;
        --extra-serve-args) export K3_EXTRA_SERVE_ARGS="$2"; shift 2 ;;
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3_SWEEP_TAG="k3-agentx-sweep"
# shellcheck source=scripts/vllm/benchmark/inferencex/kimi-k3/sweep_lib.sh
source "$SCRIPT_DIR/sweep_lib.sh"
CONFIG_FILE="$K3_BENCHMARKING_DIR/configs/${CONFIG_NAME}.sh"
[ -f "$CONFIG_FILE" ] || { log "FATAL: no such config: $CONFIG_FILE"; exit 2; }
mkdir -p "$RESULTS_DIR"
k3_set_gcs_root "$GCS_ARG"

# The cases and the server size the config declares, both plain assignments.
# Read in a subshell so the config's other variables do not leak in here;
# run_benchmarks.sh sources it again when it boots the server, which is where
# they belong.
# MAX_NUM_SEQS first and the list last: read's final variable takes the rest of
# the line, and AGENTX_CONC_LIST is space separated.
read -r CONFIG_MAX_NUM_SEQS CONFIG_CONC_LIST <<< "$(
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null 2>&1
    echo "${MAX_NUM_SEQS:-0} ${AGENTX_CONC_LIST:-}"
)" || true
CONC_LIST=$(printf '%s' "${CONFIG_CONC_LIST:-}" | tr ' ' ',')
if [ -z "$CONC_LIST" ]; then
    log "FATAL: $CONFIG_NAME declares no AGENTX_CONC_LIST"
    exit 2
fi
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
MAX_CONC=1
for c in "${CONCS[@]}"; do
    case "$c" in ''|*[!0-9]*) log "FATAL: bad concurrency: '$c'"; exit 1 ;; esac
    [ "$c" -gt "$MAX_CONC" ] && MAX_CONC=$c
done

# The throughput regime InferenceX measures wants the scheduler never to be the
# limit, so the server admits twice the ladder's top concurrency. Checked, not
# derived: the number is in the config beside the list it has to cover, and a
# mismatch is a boot-time failure rather than a quiet ceiling in the results.
NEED_SEQS=$((2 * MAX_CONC)); [ "$NEED_SEQS" -lt 4 ] && NEED_SEQS=4
if [ "${CONFIG_MAX_NUM_SEQS:-0}" -lt "$NEED_SEQS" ]; then
    log "FATAL: $CONFIG_NAME sets MAX_NUM_SEQS=${CONFIG_MAX_NUM_SEQS:-unset}, but concurrencies" \
        "${CONCS[*]} need at least $NEED_SEQS (2 x $MAX_CONC). Raise it in the config."
    exit 2
fi

# ---- Server geometry --------------------------------------------------------
export MAX_MODEL_LEN="$MAX_MODEL_LEN_ARG"
# MAX_NUM_SEQS is the config's; see the check above.
export KV_CACHE_DTYPE="$KV_DTYPE"
export PORT
if [ "$APC" = "1" ]; then
    export ENABLE_PREFIX_CACHING=true
    export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
    export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-prompt-tokens-details"
else
    export ENABLE_PREFIX_CACHING=false
fi
if [ "$MOE_REQUANT" != "0" ]; then
    # Same knob the DeepSeek-V4 configs export: fp4 weights re-blocked at load,
    # fp8 e4m3 activations in the MoE GEMM (W4A8). TPU twin of the GPU b12x arm.
    export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
    export MOE_REQUANTIZE_BLOCK_SIZE="$MOE_REQUANT"
fi
if [ "$SPEC" = "dspark" ]; then
    # Synthetic acceptance, as InferenceX pins it (golden AL per draft length).
    export ASYNC_SCHEDULING=false
    export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--speculative-config {\"method\":\"dspark\",\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_TOKENS,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SPEC_AL}"
fi

log "config=$CONFIG_NAME concs=${CONCS[*]} max_num_seqs=$CONFIG_MAX_NUM_SEQS ctx=$MAX_MODEL_LEN kv=$KV_DTYPE apc=$APC spec=$SPEC/$SPEC_TOKENS/$SPEC_AL moe_requant=$MOE_REQUANT extra=${K3_EXTRA_SERVE_ARGS:-} duration=$DURATION fast=$FAST gsm8k_limit=$GSM8K_LIMIT ix_ref=$IX_REF"

# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    bash "$K3_BENCHMARKING_DIR/cleanup_server.sh"
    k3_dump_server_log_context
    log "disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
}
trap cleanup EXIT

k3_boot_server "$CONFIG_NAME" || exit 1

# The weights run_benchmarks.sh pulled, so the client can tokenize locally
# instead of resolving the model on the hub.
TOK_DIR=""
MODEL_URI=$(k3_config_json_get model_uri)
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
fi

# What the server was configured as, for the result rows: the client did not
# choose any of it and does not reconstruct it.
LABEL="agentx_k3_tpu7x-32_ctx${MAX_MODEL_LEN}_kv${KV_DTYPE}_apc${APC}_spec${SPEC}_rq${MOE_REQUANT}"
SPEC_METHOD=none; [ "$SPEC" = "dspark" ] && SPEC_METHOD=mtp

bash "$SCRIPT_DIR/agentx_client.sh" \
    --server-url "http://127.0.0.1:$PORT" \
    --conc-list "$CONC_LIST" \
    --results-dir "$RESULTS_DIR" \
    --duration "$DURATION" \
    --fast "$FAST" \
    --ix-ref "$IX_REF" \
    --gsm8k-limit "$GSM8K_LIMIT" \
    --gsm8k-conc "$GSM8K_CONC" \
    --model "$(k3_config_json_get model)" \
    --max-model-len "$MAX_MODEL_LEN" \
    --label "$LABEL" \
    --spec-method "$SPEC_METHOD" \
    ${TOK_DIR:+--tokenizer-dir "$TOK_DIR"} \
    ${GCS_ARG:+--gcs-root "$GCS_ARG"}

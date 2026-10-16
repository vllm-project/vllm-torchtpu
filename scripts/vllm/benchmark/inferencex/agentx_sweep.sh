#!/usr/bin/env bash
# InferenceX AgentX sweep: start one server from --config, run
# agentx_client.sh against it, stop it again.
#
#   agentx_sweep.sh --chips 16 [--config kimi-k3-tp32-ep-agentx] [--results-dir DIR]
#       [--duration 1800] [--fast 1|0] [--ix-ref SHA]
#       [--gsm8k-limit 100] [--gsm8k-conc 8[,16,...]]
#       [--gcs-root gs://bucket/prefix] [--port 8000]
#
# Everything about the SERVER is in the config and nowhere else: the context
# window, the KV dtype, prefix caching, MoE requantisation, spec decode, the
# serve flags, the parallelism, and the cases (AGENTX_CONC_LIST, with the
# MAX_NUM_SEQS that has to cover it). There is deliberately no flag for any of
# them - a knob that is both a flag and a config variable ends up with a
# different default at each end, and kimi-k3-tp32-ep.sh explains why context,
# block size, KV dtype and max-num-seqs have to be read as one setting rather
# than four. A different server is a different config file; the variants beside
# the default one are a few lines each.
#
# This script's own flags are the ones that do not touch the server: how long to
# replay, which InferenceX ref to clone, how big the accuracy probe is, where
# results go.
#
# The shape is the config's business too. Nothing here assumes a chip count, a
# parallelism or a pod: the chip count comes from what the harness recorded at
# boot, and the client takes it as an input because it is what InferenceX
# divides throughput by.
#
# One server serves the whole ladder; see agentx_client.sh for why that is valid
# and for the client side. A lane whose servers are already running - a kube P/D
# JobSet, where the engines are on TPU pods and the client is on a CPU pod -
# does not use this script; it calls agentx_client.sh directly.
#
# Exit status: 1 if the server failed to boot or died, or if the client failed.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-agentx"
RESULTS_DIR="/perf_eval_results/agentx-sweep"
DURATION=1800
FAST=1
IX_REF=""            # empty: the pin in sweep_lib.sh
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"
# Physical TPU chips serving this benchmark. No default: see the note at the
# point of use.
CHIPS=""
PORT=8000
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --fast)          FAST="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        --gsm8k-limit)   GSM8K_LIMIT="$2"; shift 2 ;;
        --gsm8k-conc)    GSM8K_CONC="$2"; shift 2 ;;
        --chips)         CHIPS="$2"; shift 2 ;;
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IX_SWEEP_TAG="agentx-sweep"
# shellcheck source=scripts/vllm/benchmark/inferencex/sweep_lib.sh
source "$SCRIPT_DIR/sweep_lib.sh"
CONFIG_FILE="$IX_BENCHMARKING_DIR/configs/${CONFIG_NAME}.sh"
[ -f "$CONFIG_FILE" ] || { log "FATAL: no such config: $CONFIG_FILE"; exit 2; }
mkdir -p "$RESULTS_DIR"
ix_set_gcs_root "$GCS_ARG"
export PORT

# What the config says the run covers. Read in a subshell so the config's other
# variables do not leak in here; run_benchmarks.sh sources it again when it boots
# the server, which is where they belong. Its stderr is kept, because a config
# can refuse to load (kimi-k3-tp32-ep-concsweep.sh does) and a swallowed refusal
# would be read here as an unset variable.
# The list is read last: read's final variable takes the rest of the line, and
# AGENTX_CONC_LIST is space separated.
AGENTX_REFERENCE_ROW=$(
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null 2>&1
    echo "${AGENTX_REFERENCE_ROW:-}"
)
_cfg_values=$(
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null || exit 3
    echo "${MAX_NUM_SEQS:-0} ${AGENTX_SPEC_METHOD:-none} ${AGENTX_MODEL_PREFIX:-} ${AGENTX_PRECISION:-} ${TENSOR_PARALLELISM:-1} ${DATA_PARALLELISM:-1} ${ENABLE_EP:-false} ${AGENTX_CONC_LIST:-}"
) || { log "FATAL: $CONFIG_NAME failed to load; the boot would fail the same way"; exit 2; }
read -r CONFIG_MAX_NUM_SEQS SPEC_METHOD MODEL_PREFIX PRECISION CONFIG_TP CONFIG_DP CONFIG_ENABLE_EP CONFIG_CONC_LIST <<< "$_cfg_values"
for _req in MODEL_PREFIX PRECISION; do
    if [ -z "${!_req}" ]; then
        log "FATAL: $CONFIG_NAME declares no AGENTX_${_req}; it is what InferenceX files the row under"
        exit 2
    fi
done

CONC_LIST=$(printf '%s' "${CONFIG_CONC_LIST:-}" | tr ' ' ',')
if [ -z "$CONC_LIST" ]; then
    log "FATAL: $CONFIG_NAME declares no AGENTX_CONC_LIST"
    exit 2
fi
ix_check_conc_list "$CONC_LIST" || exit 1
ix_check_conc_list "$GSM8K_CONC" || exit 1
# PHYSICAL CHIPS, not ranks: a 2x2x4 v7x slice is 16 chips (4 hosts x
# google.com/tpu 4, which is what GKE allocates and what Google bills per
# chip-hour) and vLLM runs 32 ranks on it, two per chip. So TP x DP is twice the
# count and the parallelism cannot give it; the caller states it, with no
# default, because the chips are what tokens per dollar is divided by and a wrong
# count silently scales the one number this benchmark exists to report.
case "${CHIPS:-}" in
    ''|*[!0-9]*|0)
        log "FATAL: --chips N is required: the number of physical TPU chips serving" \
            "this benchmark (16 for one v7x-32 / 2x2x4 slice). Times the rate in" \
            "sweep_summary.py it is the cost tokens per dollar divides by."
        exit 2 ;;
esac
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
MAX_CONC=1
for c in "${CONCS[@]}"; do [ "$c" -gt "$MAX_CONC" ] && MAX_CONC=$c; done

# The throughput regime InferenceX measures wants the scheduler never to be the
# limit, so the server admits at least twice the ladder's top concurrency.
# Checked, not derived: the number is in the config beside the list it has to
# cover, and a mismatch is a failure here rather than a quiet ceiling in the
# results.
NEED_SEQS=$((2 * MAX_CONC)); [ "$NEED_SEQS" -lt 4 ] && NEED_SEQS=4
if [ "${CONFIG_MAX_NUM_SEQS:-0}" -lt "$NEED_SEQS" ]; then
    log "FATAL: $CONFIG_NAME sets MAX_NUM_SEQS=${CONFIG_MAX_NUM_SEQS:-unset}, but concurrencies" \
        "${CONCS[*]} need at least $NEED_SEQS (2 x $MAX_CONC). Raise it in the config."
    exit 2
fi
# Far above it is not wrong - the compile_sizes list holds smaller buckets, so a
# decode step still pads to the next one up - but it is compile time spent on
# buckets the ladder never reaches.
if [ "$CONFIG_MAX_NUM_SEQS" -gt $((4 * MAX_CONC)) ]; then
    log "WARNING: MAX_NUM_SEQS=$CONFIG_MAX_NUM_SEQS is far above the ladder's top" \
        "concurrency ($MAX_CONC); the extra decode buckets only cost compile time"
fi
# A probe wider than the server admits still scores correctly, it just queues.
for gc in $(printf '%s' "$GSM8K_CONC" | tr ',' ' '); do
    if [ "$gc" -gt "$CONFIG_MAX_NUM_SEQS" ]; then
        log "WARNING: --gsm8k-conc $gc exceeds MAX_NUM_SEQS=$CONFIG_MAX_NUM_SEQS;" \
            "that probe will queue rather than run concurrently"
    fi
done

log "config=$CONFIG_NAME concs=${CONCS[*]} max_num_seqs=$CONFIG_MAX_NUM_SEQS" \
    "spec=$SPEC_METHOD duration=$DURATION fast=$FAST gsm8k_limit=$GSM8K_LIMIT ix_ref=$IX_REF"

# shellcheck disable=SC2317  # run by the trap
cleanup() {
    log "cleanup"
    bash "$IX_BENCHMARKING_DIR/cleanup_server.sh"
    ix_dump_server_log_context
    # A failed boot is the most common failure, and the client - which does the
    # uploading - never ran, so without this the logs that explain it exist only
    # in a container that is about to go away.
    save_now "$RESULTS_DIR"
    log "disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
}
# INT and TERM as well as EXIT: bash does not run an EXIT trap when it dies on an
# untrapped signal, and a cancelled build sends TERM - which would leave the
# server holding every chip on the pod.
trap cleanup EXIT INT TERM

ix_boot_server "$CONFIG_NAME" || exit 1

# The weights run_benchmarks.sh pulled, so the client can tokenize locally
# instead of resolving the model on the hub.
TOK_DIR=""
MODEL_URI=$(ix_config_json_get model_uri)
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
fi

# The chips tokens per dollar is costed over, which is PHYSICAL CHIPS and not
# ranks: a 2x2x4 v7x slice is 16 chips (4 hosts x google.com/tpu 4, which is what
# GKE allocates and what Google bills per chip-hour), and vLLM runs 32 ranks on
# it - two per chip. So TP x DP is twice the count, and the parallelism cannot
# give it. The deployment states it, and there is no default, because a wrong
# count silently scales the one number this benchmark exists to report.
RANKS=$(( $(ix_config_json_get tensor_parallelism) * $(ix_config_json_get data_parallelism) ))
log "serving on $CHIPS chips with $RANKS ranks (TP x DP, from config.json)"
if [ "$RANKS" -gt 0 ] && [ "$RANKS" -ne $(( CHIPS * 2 )) ]; then
    log "WARNING: $RANKS ranks on $CHIPS chips is not the two-per-chip v7x shape;" \
        "check K3_AGENTX_CHIPS is the chip count and not the rank count"
fi

# The shape InferenceX files the row under, from the config the server booted:
# one aggregate worker set, vLLM's EP spanning TP x DP when the config enables
# it (run_benchmarks.sh --enable-expert-parallel).
EP=1; [ "$CONFIG_ENABLE_EP" != true ] || EP=$(( CONFIG_TP * CONFIG_DP ))
ix_export_topology "$CONFIG_TP" "$CONFIG_DP" "$EP" 0

# The config name is the label: it is the one file that defines this server, so
# it identifies the row better than a string of knobs reassembled here.
LABEL="agentx_${CONFIG_NAME}"

bash "$SCRIPT_DIR/agentx_client.sh" \
    --server-url "http://127.0.0.1:$PORT" \
    --conc-list "$CONC_LIST" \
    --chips "$CHIPS" \
    --results-dir "$RESULTS_DIR" \
    --duration "$DURATION" \
    --fast "$FAST" \
    ${IX_REF:+--ix-ref "$IX_REF"} \
    --gsm8k-limit "$GSM8K_LIMIT" \
    --gsm8k-conc "$GSM8K_CONC" \
    --model "$(ix_config_json_get model)" \
    --label "$LABEL" \
    --spec-method "$SPEC_METHOD" \
    --model-prefix "$MODEL_PREFIX" \
    --precision "$PRECISION" \
    ${AGENTX_REFERENCE_ROW:+--reference-row "$AGENTX_REFERENCE_ROW"} \
    ${TOK_DIR:+--tokenizer-dir "$TOK_DIR"} \
    ${GCS_ARG:+--gcs-root "$GCS_ARG"}

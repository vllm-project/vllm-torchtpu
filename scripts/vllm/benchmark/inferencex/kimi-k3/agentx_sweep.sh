#!/usr/bin/env bash
# Kimi-K3 InferenceX AgentX sweep: start one server from --config, run
# agentx_client.sh against it, stop it again.
#
#   agentx_sweep.sh [--config kimi-k3-tp32-ep-agentx] [--results-dir DIR]
#       [--duration 1800] [--fast 1|0] [--ix-ref main]
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
RESULTS_DIR="/perf_eval_results/k3-agentx-sweep"
DURATION=1800
FAST=1
IX_REF="main"
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"
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
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
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
export PORT

# What the config says the run covers. Read in a subshell so the config's other
# variables do not leak in here; run_benchmarks.sh sources it again when it boots
# the server, which is where they belong. Its stderr is kept, because a config
# can refuse to load (kimi-k3-tp32-ep-concsweep.sh does) and a swallowed refusal
# would be read here as an unset variable.
# The list is read last: read's final variable takes the rest of the line, and
# AGENTX_CONC_LIST is space separated.
_cfg_values=$(
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null || exit 3
    echo "${MAX_NUM_SEQS:-0} ${AGENTX_SPEC_METHOD:-none} ${AGENTX_CONC_LIST:-}"
) || { log "FATAL: $CONFIG_NAME failed to load; the boot would fail the same way"; exit 2; }
read -r CONFIG_MAX_NUM_SEQS SPEC_METHOD CONFIG_CONC_LIST <<< "$_cfg_values"

CONC_LIST=$(printf '%s' "${CONFIG_CONC_LIST:-}" | tr ' ' ',')
if [ -z "$CONC_LIST" ]; then
    log "FATAL: $CONFIG_NAME declares no AGENTX_CONC_LIST"
    exit 2
fi
k3_check_conc_list "$CONC_LIST" || exit 1
k3_check_conc_list "$GSM8K_CONC" || exit 1
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
    bash "$K3_BENCHMARKING_DIR/cleanup_server.sh"
    k3_dump_server_log_context
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

k3_boot_server "$CONFIG_NAME" || exit 1

# The weights run_benchmarks.sh pulled, so the client can tokenize locally
# instead of resolving the model on the hub.
TOK_DIR=""
MODEL_URI=$(k3_config_json_get model_uri)
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
fi

# The chip count InferenceX divides throughput by. For one server it is the
# parallelism the harness actually booted, which it recorded in config.json, so
# it follows the config instead of being asserted here.
CHIPS=$(( $(k3_config_json_get tensor_parallelism) * $(k3_config_json_get data_parallelism) ))
if [ "$CHIPS" -le 0 ]; then
    log "FATAL: could not read the chip count from $RESULTS_DIR/config.json"
    exit 1
fi
log "serving on $CHIPS chips (tensor_parallelism x data_parallelism, from config.json)"

# The config name is the label: it is the one file that defines this server, so
# it identifies the row better than a string of knobs reassembled here.
LABEL="agentx_k3_${CONFIG_NAME}"

bash "$SCRIPT_DIR/agentx_client.sh" \
    --server-url "http://127.0.0.1:$PORT" \
    --conc-list "$CONC_LIST" \
    --chips "$CHIPS" \
    --results-dir "$RESULTS_DIR" \
    --duration "$DURATION" \
    --fast "$FAST" \
    --ix-ref "$IX_REF" \
    --gsm8k-limit "$GSM8K_LIMIT" \
    --gsm8k-conc "$GSM8K_CONC" \
    --model "$(k3_config_json_get model)" \
    --label "$LABEL" \
    --spec-method "$SPEC_METHOD" \
    ${TOK_DIR:+--tokenizer-dir "$TOK_DIR"} \
    ${GCS_ARG:+--gcs-root "$GCS_ARG"}

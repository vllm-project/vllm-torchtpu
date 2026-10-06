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

# The client pod of kimi-k3-agentx-1p1d.yaml: chipless, drives a K3
# prefill/decode pair over HTTP, and holds the proxy as well as the benchmark.
# The engines only serve (k3_1p1d.sh with K3_PD_EXTERNAL_CLIENT=1). A pod of its
# own because aiperf's replay is the measurement, and on an engine host it would
# be loading the machine it is timing.
#
# In order: resolve both heads from WORKLOAD_NAME, wait for both /health, start
# the proxy, read the cases out of the serve config, run agentx_client.sh against
# the proxy, upload the results.
#
# Env (the manifest sets all of these):
#   WORKLOAD_NAME   the JobSet's name; each head is <name>-<role>-0-0.<name>
#   K3_PD_CONFIG    the serve config both engines booted, read here for the cases
#   ARTIFACTS_DIR   where results go, uploaded as the step's artifacts
#   AGENTX_CHIPS    physical TPU chips serving the pair; no default
# Optional: AGENTX_DURATION, AGENTX_FAST, AGENTX_GSM8K_LIMIT, AGENTX_GSM8K_CONC,
#   AGENTX_IX_REF, AGENTX_GCS_ROOT, AGENTX_READY_WAIT_MIN.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
IX_DIR="$REPO_DIR/scripts/vllm/benchmark/inferencex"
CONFIG_DIR="$REPO_DIR/scripts/vllm/benchmarking/configs"
CONFIG="${K3_PD_CONFIG:-kimi-k3-tp8-dp4-ep-agentx}"
PORT=8000
PROXY_PORT=8100

: "${WORKLOAD_NAME:?the JobSet name is how the engines are addressed}"
PREFILL_HOST="${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}"
DECODE_HOST="${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}"

log() { echo "[k3-agentx-client-pod] $*"; }

cd "$REPO_DIR" || { log "FATAL: no repo at $REPO_DIR"; exit 1; }
ARTIFACTS_DIR="${ARTIFACTS_DIR:-perf_eval_results}"
RESULTS_DIR="$ARTIFACTS_DIR/k3-agentx-1p1d"
mkdir -p "$RESULTS_DIR"

CONFIG_FILE="$CONFIG_DIR/${CONFIG}.sh"
[ -f "$CONFIG_FILE" ] || { log "FATAL: no such config: $CONFIG_FILE"; exit 2; }

# The cases and the server size the engines booted with, read in a subshell with
# the role set because the config branches on it. The list is read last: read's
# final variable takes the rest of the line.
_cfg=$(
    export K3_1P1D_ROLE=decode
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null || exit 3
    echo "${MAX_NUM_SEQS:-0} ${TENSOR_PARALLELISM:-0} ${DATA_PARALLELISM:-0} ${AGENTX_MODEL_PREFIX:-} ${AGENTX_PRECISION:-} ${AGENTX_CONC_LIST:-}"
) || { log "FATAL: $CONFIG failed to load; the engines would have failed the same way"; exit 2; }
read -r CFG_MAX_NUM_SEQS CFG_TP CFG_DP CFG_MODEL_PREFIX CFG_PRECISION CFG_CONC_LIST <<< "$_cfg"
for _req in CFG_MODEL_PREFIX CFG_PRECISION; do
    [ -n "${!_req}" ] || { log "FATAL: $CONFIG declares no AGENTX_${_req#CFG_}"; exit 2; }
done

CONC_LIST=$(printf '%s' "${CFG_CONC_LIST:-}" | tr ' ' ',')
[ -n "$CONC_LIST" ] || { log "FATAL: $CONFIG declares no AGENTX_CONC_LIST"; exit 2; }
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
MAX_CONC=1
for c in "${CONCS[@]}"; do
    [[ "$c" =~ ^[1-9][0-9]*$ ]] || { log "FATAL: bad concurrency '$c' in '$CONC_LIST'"; exit 1; }
    [ "$c" -gt "$MAX_CONC" ] && MAX_CONC=$c
done
NEED_SEQS=$((2 * MAX_CONC)); [ "$NEED_SEQS" -lt 4 ] && NEED_SEQS=4
if [ "${CFG_MAX_NUM_SEQS:-0}" -lt "$NEED_SEQS" ]; then
    log "FATAL: $CONFIG sets MAX_NUM_SEQS=$CFG_MAX_NUM_SEQS, but concurrencies ${CONCS[*]}" \
        "need at least $NEED_SEQS (2 x $MAX_CONC). Raise it in the config."
    exit 2
fi

# The per-chip divisor: PHYSICAL chips, both engines, stated by the manifest
# because only it knows what was allocated. TP x DP is ranks, and on v7x that is
# twice the chips, so deriving it here would halve the reported figure.
: "${AGENTX_CHIPS:?the manifest must state the physical chip count (32 for two 2x2x4 slices)}"
case "$AGENTX_CHIPS" in ''|*[!0-9]*|0) log "FATAL: bad AGENTX_CHIPS '$AGENTX_CHIPS'"; exit 2 ;; esac
CHIPS="$AGENTX_CHIPS"
RANKS=$(( CFG_TP * CFG_DP * 2 ))
if [ "$RANKS" -ne $(( CHIPS * 2 )) ]; then
    log "WARNING: the pair runs $RANKS ranks on $CHIPS chips, which is not the" \
        "two-per-chip v7x shape; check AGENTX_CHIPS is chips and not ranks"
fi

log "config=$CONFIG concs=${CONCS[*]} max_num_seqs=$CFG_MAX_NUM_SEQS chips=$CHIPS ranks=$RANKS" \
    "prefill=$PREFILL_HOST decode=$DECODE_HOST"

proxy_pid=""
# shellcheck disable=SC2317  # run by the trap
cleanup() {
    [ -n "$proxy_pid" ] && kill "$proxy_pid" 2>/dev/null
    if [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] && command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*" \
            || log "ERROR: results were produced but could not be uploaded"
    fi
}
# TERM as well as EXIT: a cancelled build does not run an EXIT trap, and the
# results of the points that did finish are worth having.
trap cleanup EXIT INT TERM

# wait_health <url> <what>: the engines stream 1.5 TB and compile, so the bound
# is hours. 4 h by default; AGENTX_READY_WAIT_MIN overrides it.
wait_health() {
    local url="$1" what="$2" deadline=$(( SECONDS + ${AGENTX_READY_WAIT_MIN:-240} * 60 ))
    log "waiting for $what at $url"
    until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" || true)" = "200" ]; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            log "FATAL: $what never became ready"
            return 1
        fi
        sleep 30
    done
    log "$what is ready"
}
wait_health "http://${PREFILL_HOST}:${PORT}/health" "the prefill head" || exit 1
wait_health "http://${DECODE_HOST}:${PORT}/health" "the decode head" || exit 1

# The block geometry each engine settled on: the pair only transfers KV when the
# two agree.
for url in "http://${PREFILL_HOST}:${PORT}" "http://${DECODE_HOST}:${PORT}"; do
    log "cache config at $url:"
    curl -sf --max-time 30 "$url/metrics" | grep -m1 '^vllm:cache_config_info' || true
done

# Both engines, prefill first, as srt_agentic.sh orders them: the proxy fronting
# the pair has no /metrics of its own, so aiperf reads each engine's.
export AIPERF_SERVER_METRICS_URLS="http://${PREFILL_HOST}:${PORT}/metrics,http://${DECODE_HOST}:${PORT}/metrics"

log "starting the proxy on port $PROXY_PORT"
python3 examples/disagg/toy_proxy_server.py \
    --host 0.0.0.0 --port "$PROXY_PORT" \
    --prefiller-hosts "$PREFILL_HOST" --prefiller-ports "$PORT" \
    --decoder-hosts "$DECODE_HOST" --decoder-ports "$PORT" \
    >"$RESULTS_DIR/proxy.log" 2>&1 &
proxy_pid=$!
if ! timeout 120 bash -c "until curl -sf -o /dev/null http://127.0.0.1:${PROXY_PORT}/healthcheck; do sleep 2; done"; then
    log "FATAL: the proxy did not come up"
    tail -50 "$RESULTS_DIR/proxy.log" || true
    exit 1
fi

# --server-url is the proxy, so every request takes the prefill-then-decode path,
# and the model card comes from its /v1/models. --engine-url is where /metrics is
# read instead: the proxy has none, and the queue depth and preemption counters
# are one engine's.
log "running the AgentX client against the proxy"
bash "$IX_DIR/agentx_client.sh" \
    --server-url "http://127.0.0.1:$PROXY_PORT" \
    --engine-url "http://${DECODE_HOST}:${PORT}" \
    --conc-list "$CONC_LIST" \
    --chips "$CHIPS" \
    --results-dir "$RESULTS_DIR" \
    --label "agentx_${CONFIG}" \
    --spec-method none \
    --model-prefix "$CFG_MODEL_PREFIX" \
    --precision "$CFG_PRECISION" \
    --duration "${AGENTX_DURATION:-1800}" \
    --fast "${AGENTX_FAST:-1}" \
    --ix-ref "${AGENTX_IX_REF:-main}" \
    --gsm8k-limit "${AGENTX_GSM8K_LIMIT:-100}" \
    --gsm8k-conc "${AGENTX_GSM8K_CONC:-4}" \
    ${AGENTX_GCS_ROOT:+--gcs-root "$AGENTX_GCS_ROOT"}
rc=$?
log "the AgentX client finished with rc=$rc"
exit "$rc"

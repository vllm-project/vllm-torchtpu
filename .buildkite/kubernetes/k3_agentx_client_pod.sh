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
# Two jobs, the shape InferenceX gives a Kimi K3 agentic row
# (infx/evals/EVALS.md). K3_AGENTX_EVAL_ONLY=0: the replay ladder, then the
# config's AGENTX_POST_EVAL_SUITES (the Kimi tool-call schema suite) on the same
# pair, InferenceX's post-eval of every K3 point. K3_AGENTX_EVAL_ONLY=1: no
# replay, GSM8K at the top of the ladder, the eval-only job every agentic row
# gets. Both evals run against the proxy on this pod, the loopback address
# InferenceX's run_eval expects.
#
# The AGENTX_* knobs are passed only when set, so agentx_client.sh keeps one
# default per value. A second default here is how build 245 ran against
# InferenceX main while the client had pinned a ref. Only --gsm8k-conc carries a
# value, the top of the ladder, where the client's own default is wider than
# this pair admits.
#
# Env (the manifest sets all of these):
#   WORKLOAD_NAME   the JobSet's name; each head is <name>-<role>-0-0.<name>
#   K3_PD_CONFIG    the serve config both engines booted, read here for the cases
#   ARTIFACTS_DIR   where results go, uploaded as the step's artifacts
#   AGENTX_CHIPS    physical TPU chips serving the pair; no default
# Optional: AGENTX_DURATION, AGENTX_FAST, AGENTX_GSM8K_LIMIT, AGENTX_GSM8K_CONC,
#   AGENTX_IX_REF, AGENTX_GCS_ROOT, AGENTX_READY_WAIT_MIN, K3_AGENTX_EVAL_ONLY,
#   AGENTX_EVAL_SMOKE=1 (the evals as a link check, not a result: 20 GSM8K
#   questions in place of the split, the verifier's two-case suite in place
#   of the full one), AGENTX_SKIP_POST_EVAL=1 (throughput: replay only).
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
EVAL_ONLY="${K3_AGENTX_EVAL_ONLY:-0}"
case "$EVAL_ONLY" in 0|1) ;; *) log "FATAL: K3_AGENTX_EVAL_ONLY must be 0 or 1, not '$EVAL_ONLY'"; exit 2 ;; esac
if [ "$EVAL_ONLY" = 1 ]; then
    RESULTS_DIR="$ARTIFACTS_DIR/k3-agentx-evals-1p1d"
else
    RESULTS_DIR="$ARTIFACTS_DIR/k3-agentx-1p1d"
fi
mkdir -p "$RESULTS_DIR"

CONFIG_FILE="$CONFIG_DIR/${CONFIG}.sh"
[ -f "$CONFIG_FILE" ] || { log "FATAL: no such config: $CONFIG_FILE"; exit 2; }

# The cases and the server size the engines booted with, read in a subshell with
# the role set because the config branches on it. The suite list travels comma
# joined, or as "-" when there is none, so it stays one field. The ladder is
# read last: read's final variable takes the rest of the line.
_cfg=$(
    export K3_1P1D_ROLE=decode
    # shellcheck source=/dev/null
    source "$CONFIG_FILE" >/dev/null || exit 3
    _post=$(printf '%s' "${AGENTX_POST_EVAL_SUITES:-}" | tr -s ' ' ',')
    echo "${MAX_NUM_SEQS:-0} ${TENSOR_PARALLELISM:-0} ${DATA_PARALLELISM:-0} ${AGENTX_MODEL_PREFIX:-} ${AGENTX_PRECISION:-} ${_post:--} ${AGENTX_CONC_LIST:-}"
) || { log "FATAL: $CONFIG failed to load; the engines would have failed the same way"; exit 2; }
read -r CFG_MAX_NUM_SEQS CFG_TP CFG_DP CFG_MODEL_PREFIX CFG_PRECISION CFG_POST_EVAL_SUITES CFG_CONC_LIST <<< "$_cfg"
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

# What tokens per dollar is costed over: PHYSICAL chips, both engines, stated by
# the manifest because only it knows what was allocated. TP x DP is ranks, and on
# v7x that is twice the chips, so deriving it here would halve the figure.
: "${AGENTX_CHIPS:?the manifest must state the physical chip count (32 for two 2x2x4 slices)}"
case "$AGENTX_CHIPS" in ''|*[!0-9]*|0) log "FATAL: bad AGENTX_CHIPS '$AGENTX_CHIPS'"; exit 2 ;; esac
CHIPS="$AGENTX_CHIPS"
RANKS=$(( CFG_TP * CFG_DP * 2 ))
if [ "$RANKS" -ne $(( CHIPS * 2 )) ]; then
    log "WARNING: the pair runs $RANKS ranks on $CHIPS chips, which is not the" \
        "two-per-chip v7x shape; check AGENTX_CHIPS is chips and not ranks"
fi

POST_EVAL_SUITES="$CFG_POST_EVAL_SUITES"
[ "${AGENTX_SKIP_POST_EVAL:-0}" != 1 ] || POST_EVAL_SUITES="-"
# A smoke proves the path - lm-eval installs, the proxy streams tool calls,
# validate_scores reads the staging - in minutes, with results that are
# marked as not the gate: a GSM8K slice (InferenceX's EVAL_LIMIT) and the
# verifier's two-case suite.
SMOKE="${AGENTX_EVAL_SMOKE:-0}"
case "$SMOKE" in 0|1) ;; *) log "FATAL: AGENTX_EVAL_SMOKE must be 0 or 1, not '$SMOKE'"; exit 2 ;; esac
GSM8K_LIMIT_ARG=""
if [ "$SMOKE" = 1 ]; then
    GSM8K_LIMIT_ARG=20
    POST_EVAL_SUITES="${POST_EVAL_SUITES//kimi_tool_call_schema_full/kimi_tool_call_schema}"
fi

log "config=$CONFIG concs=${CONCS[*]} max_num_seqs=$CFG_MAX_NUM_SEQS chips=$CHIPS ranks=$RANKS" \
    "prefill=$PREFILL_HOST decode=$DECODE_HOST eval_only=$EVAL_ONLY smoke=$SMOKE" \
    "post_eval_suites=$POST_EVAL_SUITES"

proxy_pid=""
# shellcheck disable=SC2317  # run by the trap
cleanup() {
    [ -n "$proxy_pid" ] && kill "$proxy_pid" 2>/dev/null
    if [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] && command -v buildkite-agent >/dev/null 2>&1; then
        # aiperf writes server_metrics_export.json per point, and at this context
        # it was 357 MB in build 247 - one sample per metric per scrape for the
        # whole window. The CSV beside it carries the same series, so drop the
        # JSON rather than upload it every night.
        find "$ARTIFACTS_DIR" -name server_metrics_export.json -delete 2>/dev/null
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
if [ "$EVAL_ONLY" = 1 ]; then
    # At the top of the ladder: InferenceX evaluates a multi-node deployment at
    # its highest concurrency of 16 or more, else its highest.
    log "running the AgentX GSM8K eval against the proxy"
    bash "$IX_DIR/agentx_eval_client.sh" \
        --server-url "http://127.0.0.1:$PROXY_PORT" \
        --engine-url "http://${DECODE_HOST}:${PORT}" \
        --suites gsm8k \
        --eval-conc "$MAX_CONC" \
        --chips "$CHIPS" \
        --results-dir "$RESULTS_DIR" \
        --spec-method none \
        --model-prefix "$CFG_MODEL_PREFIX" \
        --precision "$CFG_PRECISION" \
        ${AGENTX_IX_REF:+--ix-ref "$AGENTX_IX_REF"} \
        ${GSM8K_LIMIT_ARG:+--gsm8k-limit "$GSM8K_LIMIT_ARG"} \
        ${AGENTX_GCS_ROOT:+--gcs-root "$AGENTX_GCS_ROOT"}
    rc=$?
    log "the AgentX eval client finished with rc=$rc"
    exit "$rc"
fi

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
    --gsm8k-conc "${AGENTX_GSM8K_CONC:-$MAX_CONC}" \
    ${AGENTX_DURATION:+--duration "$AGENTX_DURATION"} \
    ${AGENTX_FAST:+--fast "$AGENTX_FAST"} \
    ${AGENTX_IX_REF:+--ix-ref "$AGENTX_IX_REF"} \
    ${AGENTX_GSM8K_LIMIT:+--gsm8k-limit "$AGENTX_GSM8K_LIMIT"} \
    ${AGENTX_GCS_ROOT:+--gcs-root "$AGENTX_GCS_ROOT"}
rc=$?
log "the AgentX client finished with rc=$rc"
if [ "$POST_EVAL_SUITES" = "-" ]; then
    exit "$rc"
fi

# InferenceX's post-eval for a Kimi K3 point: the tool-call schema suite on the
# pair the ladder just ran on, after the replay so it never shares the window.
# A dead pair fails here on its own; a failed point does not skip the suite,
# since the schema verdict is still worth having.
log "running the post-replay evals ($POST_EVAL_SUITES) against the proxy"
bash "$IX_DIR/agentx_eval_client.sh" \
    --server-url "http://127.0.0.1:$PROXY_PORT" \
    --engine-url "http://${DECODE_HOST}:${PORT}" \
    --suites "$POST_EVAL_SUITES" \
    --eval-conc "$MAX_CONC" \
    --chips "$CHIPS" \
    --results-dir "$RESULTS_DIR/post-eval" \
    --spec-method none \
    --model-prefix "$CFG_MODEL_PREFIX" \
    --precision "$CFG_PRECISION" \
    ${AGENTX_IX_REF:+--ix-ref "$AGENTX_IX_REF"} \
    ${AGENTX_GCS_ROOT:+--gcs-root "$AGENTX_GCS_ROOT"}
erc=$?
log "the post-replay evals finished with rc=$erc"
[ "$rc" -ne 0 ] || rc=$erc
exit "$rc"

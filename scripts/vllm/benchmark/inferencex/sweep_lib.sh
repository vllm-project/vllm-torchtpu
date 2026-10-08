#!/usr/bin/env bash
# Shared helpers for the InferenceX benchmark scripts:
#   agentx_sweep.sh          boots a server, then calls agentx_client.sh (boot,
#                            config.json, cleanup, log dump, GCS)
#   agentx_client.sh         benchmarks a server it did not boot (server URL,
#                            liveness, drain, engine counters, GCS)
#   kimi-k3/eval_conc_sweep.sh  boots a server, then runs lm_eval (all of it)
# Source it after setting:
#   IX_SWEEP_TAG   log tag, e.g. "k3-eval-sweep"
#   RESULTS_DIR    where the harness and these scripts write results
#   PORT           the vLLM server port - only needed when IX_SERVER_URL is
#                  unset, since that is the fallback ix_server_url resolves to
# Everything here talks to the server named by ix_server_url: the one the
# caller booted with run_benchmarks.sh --keep-alive, or one on another host.

log() { echo "[${IX_SWEEP_TAG:-k3-sweep}] $*"; }

# ix_check_conc_list "1,2,4" -> fails with a clear message on a non-integer.
ix_check_conc_list() {
    local c
    IFS=',' read -r -a _concs <<< "$1"
    for c in "${_concs[@]}"; do
        [[ "$c" =~ ^[1-9][0-9]*$ ]] || { log "ERROR: bad concurrency '$c' in '$1' (positive integers, comma separated)"; return 1; }
    done
}

# ix_set_gcs_root gs://bucket/prefix -> GCS_ROOT=gs://bucket/prefix/<build number>,
# or GCS_ROOT="" (no upload) when the argument is empty.
ix_set_gcs_root() {
    GCS_ROOT=""
    [ -n "${1:-}" ] || return 0
    # Said once, here: without gsutil every save_now is a silent no-op, so
    # --gcs-root would look honoured for the whole run and upload nothing.
    if ! command -v gsutil >/dev/null 2>&1; then
        log "WARNING: --gcs-root $1 was given but gsutil is not on PATH; nothing will be uploaded"
        return 0
    fi
    GCS_ROOT="${1%/}/${BUILDKITE_BUILD_NUMBER:-local}"
}

# save_now <path>... : copy each path to $GCS_ROOT/<results dir name>/ right
# away. A cancelled build never runs the exit trap, so every point is saved as
# soon as it is written, not only at exit. No-op without GCS_ROOT or gsutil.
save_now() {
    [ -n "${GCS_ROOT:-}" ] && command -v gsutil >/dev/null 2>&1 || return 0
    local x
    for x in "$@"; do
        [ -e "$x" ] || continue
        gsutil -q -m cp -r "$x" "$GCS_ROOT/$(basename "$RESULTS_DIR")/" 2>&1 | tail -2 \
            || log "WARNING: upload of $x to $GCS_ROOT failed"
    done
}

# ix_server_url: the base URL of the vLLM server to send requests to. Empty or
# unset IX_SERVER_URL means agentx_sweep.sh or eval_conc_sweep.sh started that
# server in this same container, so it is at 127.0.0.1:$PORT. A caller whose
# server runs on another host - a kube P/D JobSet, where the engines are on TPU
# pods and the client is on a CPU pod - sets IX_SERVER_URL to its URL. Resolved
# per call, so neither PORT nor IX_SERVER_URL has to be set before this file is
# sourced.
ix_server_url() { echo "${IX_SERVER_URL:-http://127.0.0.1:$PORT}"; }

# --max-time on every poll: an engine that is wedged but still listening would
# otherwise block the caller forever, and wait_server_idle only checks its
# deadline between polls, so its timeout would never fire.
server_alive() { curl -sf --max-time 10 "$(ix_server_url)/v1/models" >/dev/null; }

# Number of requests the engine holds (running + waiting), from /metrics.
# The anchor stops vllm:num_requests_waiting_by_reason{...} from matching too.
# Returns non-zero when /metrics could not be read, so a caller can tell
# "unreachable" from "drained" instead of reading both as 0.
ix_engine_queue_depth() {
    local body
    body=$(curl -s --max-time 30 "$(ix_server_url)/metrics") || return 1
    [ -n "$body" ] || return 1
    printf '%s' "$body" \
        | awk '/^vllm:num_requests_(running|waiting)( |\{)/{s+=$NF} END{print s+0}'
}

# Cumulative engine counters from /metrics that explain a bad point:
# preemptions (a preempted request is re-prefilled from its KV state) and
# finished requests. Printed as "preemptions=N finished=M".
ix_engine_counters() {
    curl -s --max-time 30 "$(ix_server_url)/metrics" | awk '
        /^vllm:num_preemptions_total( |\{)/ {p+=$NF}
        /^vllm:request_success_total( |\{)/ {f+=$NF}
        END {printf "preemptions=%d finished=%d\n", p, f}'
}

# wait_server_idle [timeout_s]: return when running+waiting has been 0 on 3
# polls 10 s apart, so the next point starts on a drained engine.
# shellcheck disable=SC2120  # the timeout argument is optional
wait_server_idle() {
    local timeout="${1:-1800}" t0 idle=0 n
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$timeout" ]; do
        if ! n=$(ix_engine_queue_depth); then
            log "WARNING: /metrics unreachable while draining"
            idle=0; sleep 10; continue
        fi
        if [ "${n%.*}" = "0" ]; then
            idle=$((idle + 1)); [ "$idle" -ge 3 ] && return 0
        else
            idle=0; log "drain: running+waiting=$n"
        fi
        sleep 10
    done
    log "WARNING: engine did not drain within ${timeout}s (running+waiting=$n)"
    return 1
}

# ix_check_disk [min_free_gb] : fail before the 30-min boot when the results
# mount or the container root has less than min_free_gb free. The boot itself
# writes tens of GB (compile caches, Ray logs), so the floor is 50 GB: a pod
# that passed a 10 GB check still died with "No space left on device" mid-boot.
ix_check_disk() {
    local min_gb="${1:-50}" d free ok=0
    for d in "$RESULTS_DIR" /tmp /; do
        [ -d "$d" ] || continue
        free=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ -n "$free" ] || continue
        log "disk: ${free} GB free on $d"
        if [ "$free" -lt "$min_gb" ]; then
            log "FATAL: only ${free} GB free on $d (need ${min_gb} GB); the pod's disk needs cleaning (docker image prune, old results) before this sweep can run"
            df -h "$d" 2>/dev/null || true
            ok=1
        fi
    done
    return $ok
}

# ix_boot_server <config> : cleanup, then run_benchmarks.sh --keep-alive
# (boots the server, runs the config's bench as a smoke test, leaves the server
# up). Returns 1 with a FATAL line when the boot or the smoke bench fails.
ix_boot_server() {
    local config="$1" t0 rc
    ix_check_disk 50 || return 1
    bash "$IX_BENCHMARKING_DIR/cleanup_server.sh"
    t0=$(date +%s)
    "$IX_BENCHMARKING_DIR/run_benchmarks.sh" --config "$config" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "FATAL: run_benchmarks.sh --config $config failed (rc=$rc): server boot or smoke bench failed; see server.log context below"
        return 1
    fi
    if ! server_alive; then
        log "FATAL: server not answering on port $PORT after run_benchmarks.sh returned 0"
        return 1
    fi
    log "server up after $(( $(date +%s) - t0 )) s (incl. the config's smoke bench)"
}

# ix_config_json_get <key> : a value from the harness's $RESULTS_DIR/config.json.
ix_config_json_get() {
    python3 -c 'import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
        "$RESULTS_DIR/config.json" "$1"
}

# ix_dump_server_log_context : error context + tail of server.log, for the
# exit trap, so a failed boot can be diagnosed from the job log alone.
ix_dump_server_log_context() {
    [ -f "$RESULTS_DIR/server.log" ] || return 0
    log "===== server.log: error context ====="
    tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' \
        | grep -v -E 'ActorHandle|ray_executor_v2.py|Failed to kill|client_mode_hook|auto_init_hook|_raylet|ray/_private' \
        | grep -n -E -B3 -A25 'Traceback|Error|Exception|NotImplemented|Killed|OOM|halt' | tail -150 | cut -c1-400
    log "===== server.log: last 40 non-blank lines ====="
    tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' | tail -40 | cut -c1-300
}

# run_benchmarks.sh, cleanup_server.sh and configs/ are shared with every other
# model's benchmarks and stay in scripts/vllm/benchmarking/; these scripts live
# one directory deeper, under benchmark/inferencex/.
IX_BENCHMARKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../benchmarking" && pwd)"

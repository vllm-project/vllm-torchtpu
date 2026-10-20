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

# InferenceX's quality evals for an AgentX row, against a server that is already
# serving. A row is not accepted on its own (infx/evals/EVALS.md): GSM8K runs as
# a separate eval-only job on the same deployment at its highest eval
# concurrency, against the thresholds.yaml floor, and a Kimi K3 point runs the
# Kimi-Vendor-Verifier tool-call schema suite after its replay. This runs either
# list of suites; the caller says which. Not specific to a model.
#
#   agentx_eval_client.sh --suites gsm8k[,kimi_tool_call_schema_full]
#       --eval-conc N --chips N --model-prefix NAME --precision LABEL
#       [--server-url http://127.0.0.1:PORT] [--engine-url URL]
#       [--runner-type tpu7x-32] [--spec-method none] [--results-dir DIR]
#       [--ix-ref SHA] [--gsm8k-limit N] [--gcs-root gs://...] [--port 8000]
#
# How a suite runs is InferenceX's: the checkout's benchmark_lib.sh is sourced
# and its run_eval called, as its own srt_eval.sh does. That installs lm-eval
# at their commit with the reasoning patch, fetches the pinned verifier
# (sha256-checked), applies their budgets, writes meta_env.json and stages the
# results; the verdict is their validate_scores.py with their thresholds.yaml,
# keyed by --model-prefix. The one pin here is the InferenceX ref.
#
# run_eval addresses 127.0.0.1:$PORT, so --server-url must be a loopback URL:
# the client pod runs the P/D proxy itself, the same shape as InferenceX's
# recipes, whose post-eval runs where the frontend is. A server elsewhere is
# refused rather than silently evaluated at the wrong address.
#
# --eval-conc is the deployment's eval concurrency: InferenceX takes a
# multi-node deployment's highest throughput concurrency of 16 or more, else
# its highest (infx/matrix/generate.py); the caller derives it from its ladder.
# --gsm8k-limit N is InferenceX's EVAL_LIMIT smoke slice, not a result the row
# can carry; the gate is the whole split. The vendor suites ignore both.
#
# Exit status: 1 if the server was unreachable, a suite could not run, or any
# suite failed validation; 0 otherwise. summary.md has each suite's verdict.
set -uo pipefail

PORT=8000               # only used when --server-url is absent
IX_SERVER_URL=""
RESULTS_DIR="/perf_eval_results/agentx-evals"
SUITES=""
EVAL_CONC=""
CHIPS=""
RUNNER_TYPE_ARG=""
MODEL_PREFIX_ARG=""
PRECISION_ARG=""
SPEC_METHOD="none"
IX_REF=""               # empty = IX_DEFAULT_REF, the pin in sweep_lib.sh
# Empty is the whole test split, which is what InferenceX gates on.
GSM8K_LIMIT=""
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)    IX_SERVER_URL="$2"; shift 2 ;;
        --engine-url)    IX_ENGINE_URL="$2"; shift 2 ;;
        --suites)        SUITES="$2"; shift 2 ;;
        --eval-conc)     EVAL_CONC="$2"; shift 2 ;;
        --chips)         CHIPS="$2"; shift 2 ;;
        --runner-type)   RUNNER_TYPE_ARG="$2"; shift 2 ;;
        --model-prefix)  MODEL_PREFIX_ARG="$2"; shift 2 ;;
        --precision)     PRECISION_ARG="$2"; shift 2 ;;
        --spec-method)   SPEC_METHOD="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        --gsm8k-limit)   GSM8K_LIMIT="$2"; shift 2 ;;
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$SUITES" ] || { echo "agentx_eval_client.sh: --suites is required" >&2; exit 2; }
IFS=',' read -r -a SUITE_LIST <<< "$SUITES"
for s in "${SUITE_LIST[@]}"; do
    case "$s" in
        gsm8k|kimi_tool_call_schema|kimi_tool_call_schema_full) ;;
        *) echo "agentx_eval_client.sh: unknown suite '$s' (gsm8k, kimi_tool_call_schema, kimi_tool_call_schema_full)" >&2; exit 2 ;;
    esac
done
[[ "$EVAL_CONC" =~ ^[1-9][0-9]*$ ]] || {
    echo "agentx_eval_client.sh: --eval-conc N is required (the deployment's eval concurrency; see the header)" >&2; exit 2
}
case "${CHIPS:-}" in
    ''|*[!0-9]*|0) echo "agentx_eval_client.sh: --chips N is required (the serving deployment's chip count, as agentx_client.sh takes it)" >&2; exit 2 ;;
esac
# The prefix is what thresholds.yaml is keyed by, so a wrong one silently
# applies another model's floor; neither label has a sensible default.
[ -n "$MODEL_PREFIX_ARG" ] || { echo "agentx_eval_client.sh: --model-prefix NAME is required (InferenceX keys its thresholds by it)" >&2; exit 2; }
[ -n "$PRECISION_ARG" ] || { echo "agentx_eval_client.sh: --precision LABEL is required (InferenceX files the row under it)" >&2; exit 2; }
[ -z "$GSM8K_LIMIT" ] || [[ "$GSM8K_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "agentx_eval_client.sh: --gsm8k-limit must be a positive integer, or empty for the whole split" >&2; exit 2
}
RUNNER_TYPE="${RUNNER_TYPE_ARG:-tpu7x-$CHIPS}"

IX_SERVER_URL="${IX_SERVER_URL%/}"
export IX_SERVER_URL
# Where /metrics is read from to drain between suites; a P/D proxy has none.
IX_ENGINE_URL="${IX_ENGINE_URL:-}"
IX_ENGINE_URL="${IX_ENGINE_URL%/}"
export IX_ENGINE_URL
# run_eval only addresses 127.0.0.1:$PORT (benchmark_lib.sh run_lm_eval and
# _run_kimi_tool_call_schema_eval), so the server has to be on this host, and
# PORT has to be its port rather than the default a bare --port would leave.
if [ -n "$IX_SERVER_URL" ]; then
    case "$IX_SERVER_URL" in
        http://127.0.0.1:*|http://localhost:*|http://0.0.0.0:*)
            PORT="${IX_SERVER_URL##*:}" ;;
        *)
            echo "agentx_eval_client.sh: --server-url must be a loopback URL with a port" \
                "(InferenceX's run_eval talks to 127.0.0.1:\$PORT; run this beside" \
                "the server or its proxy), got $IX_SERVER_URL" >&2
            exit 2 ;;
    esac
    [[ "$PORT" =~ ^[0-9]+$ ]] || { echo "agentx_eval_client.sh: no port in --server-url $IX_SERVER_URL" >&2; exit 2; }
fi
export PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IX_SWEEP_TAG="agentx-evals"
# shellcheck source=scripts/vllm/benchmark/inferencex/sweep_lib.sh
source "$SCRIPT_DIR/sweep_lib.sh"
IX_REF="${IX_REF:-$IX_DEFAULT_REF}"

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
ix_set_gcs_root "$GCS_ARG"

# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    log "results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
}
trap cleanup EXIT INT TERM

log "server=$(ix_server_url) suites=${SUITE_LIST[*]} eval_conc=$EVAL_CONC gsm8k_limit=${GSM8K_LIMIT:-all} ix_ref=$IX_REF"

if ! server_alive; then
    log "FATAL: no server answering /v1/models at $(ix_server_url)"
    exit 1
fi
# The served name and the context window, from the server, as agentx_client.sh
# reads them: every request carries the name, and the context is run_lm_eval's
# EVAL_MAX_MODEL_LEN, which sizes GSM8K's output budget.
_models_json=$(curl -sf --max-time 30 "$(ix_server_url)/v1/models") \
    || { log "FATAL: cannot read /v1/models from $(ix_server_url)"; exit 1; }
read -r MODEL CTX <<< "$(printf '%s' "$_models_json" | python3 -c '
import json, sys
card = json.load(sys.stdin)["data"][0]
print(card["id"], card.get("max_model_len") or 0)
')" || { log "FATAL: cannot parse /v1/models from $(ix_server_url)"; exit 1; }
if [ "${CTX:-0}" -le 0 ] 2>/dev/null; then
    log "FATAL: the server did not report max_model_len"
    exit 1
fi
log "served model: $MODEL, context $CTX"

# ---- Scratch and the InferenceX checkout ------------------------------------
SCRATCH=/tmp/.agentx_scratch
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
# Sets IX_ROOT and IX_EVALS; the same checkout agentx_client.sh would reuse.
ix_checkout "$SCRATCH/InferenceX" "$IX_REF" || exit $?
log "InferenceX $(git -C "$SCRATCH/InferenceX" rev-parse --short HEAD) (tree $IX_ROOT)"
for f in benchmarks/benchmark_lib.sh benchmarks/runtime_settings.sh \
         infx/evals/thresholds.yaml infx/evals/validate_scores.py; do
    [ -f "$IX_ROOT/$f" ] || { log "FATAL: no $f in the InferenceX checkout at $IX_ROOT"; exit 1; }
done

# ---- InferenceX's eval library -----------------------------------------------
# Their runtime defaults first (OPENAI_API_KEY, the endpoint-ready timeouts
# run_eval waits with), then the library, as agentx_client.sh does for the
# replay. The library does not touch the server when sourced.
# shellcheck disable=SC1091
source "$IX_ROOT/benchmarks/runtime_settings.sh"
export INFMAX_CONTAINER_WORKSPACE="$IX_ROOT"
# shellcheck disable=SC1091
source "$IX_ROOT/benchmarks/benchmark_lib.sh"

# What run_eval, _write_lm_eval_meta_json and validate_scores read, with the
# names multi_node/srt_eval.sh and llm-d/server.sh hand them. The deployment's
# shape (PREFILL_*, DECODE_*) comes from the caller, the same export the replay
# ran under, so InferenceX matches this eval to its row; meta_env.json's tp and
# ep are bridged from it (bridge_disagg_eval_metadata).
ix_require_topology || exit 2
export MODEL MODEL_NAME="$MODEL"
export EVAL_ONLY=true IS_AGENTIC=1 IS_MULTINODE
export EVAL_CONCURRENT_REQUESTS="$EVAL_CONC" CONC="$EVAL_CONC"
export EVAL_MAX_MODEL_LEN="$CTX"
export MODEL_PREFIX="$MODEL_PREFIX_ARG" PRECISION="$PRECISION_ARG" FRAMEWORK=vllm
export RUNNER_TYPE SPEC_DECODING="$SPEC_METHOD"
export IMAGE="${BUILDKITE_COMMIT:-local}"
# EVAL_LIMIT is run_lm_eval's smoke slice; the vendor suites ignore it.
[ -z "$GSM8K_LIMIT" ] || export EVAL_LIMIT="$GSM8K_LIMIT"

# ---- Run each suite, then judge it --------------------------------------------
# One directory per suite, which is the cwd run_eval stages into: with EVAL_ONLY
# and an agentic scenario it calls append_lm_eval_summary itself, which writes
# meta_env.json beside the results*.json and removes its scratch
# EVAL_RESULT_DIR. validate_scores.py globs that same directory.
FAIL=0
: > "$RESULTS_DIR/summary.md"
for idx in "${!SUITE_LIST[@]}"; do
    suite="${SUITE_LIST[$idx]}"
    dir="$RESULTS_DIR/$suite"
    rm -rf "$dir"; mkdir -p "$dir"
    log "===== $suite ($((idx+1))/${#SUITE_LIST[@]}) ====="
    if ! server_alive; then log "FATAL: server down before $suite"; FAIL=1; break; fi
    case "$suite" in
        gsm8k)                  export EVAL_FRAMEWORK=lm-eval; unset EVAL_SUITE ;;
        kimi_tool_call_schema*) export EVAL_FRAMEWORK=kimi-vendor EVAL_SUITE="$suite" ;;
    esac
    export EVAL_RESULT_DIR="$dir/raw"
    T0=$(date +%s)
    ( cd "$dir" && run_eval --port "$PORT" ) > "$dir/eval.log" 2>&1
    rc=$?
    tail -25 "$dir/eval.log" | cut -c1-300
    log "$suite: run_eval rc=$rc wall $(( $(date +%s) - T0 )) s"
    # Their gate, as their workflow runs it after the upload
    # (.github/workflows/benchmark-multinode-tmpl.yml): the thresholds and the
    # metric come from the checkout, the prefix from meta_env.json.
    ( cd "$dir" && PYTHONPATH="$IX_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -P \
        -m infx.evals.validate_scores --expected-concs "$EVAL_CONC" ) > "$dir/validate.log" 2>&1
    vrc=$?
    cat "$dir/validate.log"
    if [ "$rc" -ne 0 ] || [ "$vrc" -ne 0 ]; then
        FAIL=1
        verdict="FAIL (eval rc=$rc, validate rc=$vrc)"
    else
        verdict="PASS"
    fi
    [ -z "$GSM8K_LIMIT" ] || [ "$suite" != gsm8k ] \
        || verdict="$verdict - limit $GSM8K_LIMIT, a smoke, not the gate"
    { echo "### $suite: $verdict"; echo
      grep -E '^(PASS|FAIL|WARN)' "$dir/validate.log" | sed 's/^/    /'; echo; } >> "$RESULTS_DIR/summary.md"
    save_now "$dir" "$RESULTS_DIR/summary.md"
    if [ "$idx" -lt $(( ${#SUITE_LIST[@]} - 1 )) ]; then
        wait_server_idle || log "WARNING: the engine did not drain after $suite"
    fi
done

log "===== summary ($MODEL_PREFIX_ARG $PRECISION_ARG on $RUNNER_TYPE, eval conc $EVAL_CONC) ====="
cat "$RESULTS_DIR/summary.md"
server_alive || { log "ERROR: server is dead at the end of the evals"; FAIL=1; }
[ "$FAIL" = "0" ] || log "evals finished with failures; exiting 1"
exit $FAIL

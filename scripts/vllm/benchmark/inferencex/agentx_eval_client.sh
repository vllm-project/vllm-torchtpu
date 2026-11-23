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
# Kimi-Vendor-Verifier tool-call schema suite after its replay. This runs any
# list of those suites (plus mmlu_pro) against a local or remote endpoint.
#
#   agentx_eval_client.sh --suites gsm8k[,mmlu_pro,kimi_tool_call_schema_full]
#       [--eval-conc N] [--chips N] [--model-prefix NAME] [--precision LABEL]
#       [--server-url URL] [--engine-url URL]
#       [--runner-type tpu7x-32] [--spec-method none] [--results-dir DIR]
#       [--ix-ref SHA] [--gsm8k-limit N] [--mmlu-limit N]
#       [--gcs-root gs://...] [--port 8000]
#
# How gsm8k and kimi_tool_call_schema* run is InferenceX's: the checkout's
# benchmark_lib.sh is sourced and its run_eval called, as its own srt_eval.sh
# does. When --server-url points to a remote host, https, or a path prefix, a
# local loopback forwarder bridges 127.0.0.1:$PORT (which run_eval addresses)
# to that endpoint (timeout controlled by AGENTX_FORWARD_TIMEOUT_SEC, default
# 300s). Every suite also writes results_<suite>.json in --results-dir so
# check_regression.py --mode eval gates them uniformly.
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
MMLU_LIMIT="100"
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
        --mmlu-limit)    MMLU_LIMIT="$2"; shift 2 ;;
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$SUITES" ] || { echo "agentx_eval_client.sh: --suites is required" >&2; exit 2; }
IFS=',' read -r -a _raw_suites <<< "$SUITES"
SUITE_LIST=()
for _s in "${_raw_suites[@]}"; do
    s="${_s// /}"
    [ -n "$s" ] || continue
    case "$s" in
        mmlu_pro|gsm8k|kimi_tool_call_schema|kimi_tool_call_schema_full)
            SUITE_LIST+=("$s")
            ;;
        *)
            echo "agentx_eval_client.sh: unknown suite '$s' (mmlu_pro, gsm8k, kimi_tool_call_schema, kimi_tool_call_schema_full)" >&2
            exit 2
            ;;
    esac
done
[ "${#SUITE_LIST[@]}" -gt 0 ] || { echo "agentx_eval_client.sh: --suites is required" >&2; exit 2; }

# Standalone defaults when invoked outside a full AgentX job wrapper.
EVAL_CONC="${EVAL_CONC:-16}"
[[ "$EVAL_CONC" =~ ^[1-9][0-9]*$ ]] || {
    echo "agentx_eval_client.sh: --eval-conc must be a positive integer, got '$EVAL_CONC'" >&2; exit 2
}
CHIPS="${CHIPS:-1}"
case "$CHIPS" in
    ''|*[!0-9]*|0) echo "agentx_eval_client.sh: --chips must be a positive integer, got '$CHIPS'" >&2; exit 2 ;;
esac
[ -z "$GSM8K_LIMIT" ] || [[ "$GSM8K_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "agentx_eval_client.sh: --gsm8k-limit must be a positive integer, or empty for the whole split" >&2; exit 2
}
[[ "$MMLU_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "agentx_eval_client.sh: --mmlu-limit must be a positive integer, got '$MMLU_LIMIT'" >&2; exit 2
}
RUNNER_TYPE="${RUNNER_TYPE_ARG:-tpu7x-$CHIPS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT_URL_PY="$SCRIPT_DIR/../../benchmarking/endpoint_url.py"

# Normalize --server-url via the shared endpoint_url.py helper.
NEEDS_LOCAL_FORWARDER=0
if [ -n "$IX_SERVER_URL" ]; then
    _raw_server_url="$IX_SERVER_URL"
    IX_SERVER_URL=$(python3 "$ENDPOINT_URL_PY" normalize "$_raw_server_url") || {
        _shown="$_raw_server_url"
        [[ "$_shown" == *"@"* ]] && _shown="<redacted: contains '@'>"
        echo "agentx_eval_client.sh: invalid --server-url '$_shown'" >&2
        exit 2
    }
    _loopback_port=$(python3 -c '
import sys
from urllib.parse import urlparse
url = urlparse(sys.argv[1])
if (
    url.scheme == "http"
    and url.hostname in ("127.0.0.1", "localhost", "0.0.0.0")
    and not url.path.rstrip("/")
    and url.port is not None
):
    print(url.port)
else:
    print("forward")
' "$IX_SERVER_URL")
    if [ "$_loopback_port" = "forward" ]; then
        NEEDS_LOCAL_FORWARDER=1
    else
        PORT="$_loopback_port"
    fi
fi
export IX_SERVER_URL
# Where /metrics is read from to drain between suites; a P/D proxy has none.
IX_ENGINE_URL="${IX_ENGINE_URL:-}"
IX_ENGINE_URL="${IX_ENGINE_URL%/}"
export IX_ENGINE_URL
export PORT

export IX_SWEEP_TAG="agentx-evals"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sweep_lib.sh"
IX_REF="${IX_REF:-$IX_DEFAULT_REF}"

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
ix_set_gcs_root "$GCS_ARG"

_forward_pid=""
_forward_port_file=""
# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    if [ -n "$_forward_pid" ]; then
        kill "$_forward_pid" 2>/dev/null || true
    fi
    if [ -n "$_forward_port_file" ]; then
        rm -f "$_forward_port_file" 2>/dev/null || true
    fi
    log "cleanup"
    log "results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
}
trap cleanup EXIT INT TERM

if ! server_alive; then
    log "FATAL: no server answering /v1/models at $(ix_server_url)"
    exit 1
fi

# InferenceX's run_eval addresses 127.0.0.1:$PORT directly, so bridge a remote
# or path-prefixed --server-url onto a local ephemeral port.
if [ "$NEEDS_LOCAL_FORWARDER" = 1 ]; then
    _forward_port_file=$(mktemp)
    python3 "$SCRIPT_DIR/loopback_forwarder.py" \
        "$IX_SERVER_URL" "$_forward_port_file" \
        --timeout "${AGENTX_FORWARD_TIMEOUT_SEC:-300}" &
    _forward_pid=$!
    for _ in $(seq 1 50); do
        if [ -s "$_forward_port_file" ]; then
            PORT=$(cat "$_forward_port_file")
            export PORT
            break
        fi
        sleep 0.1
    done
    # PORT keeps its 8000 default until the port file appears; never probe that.
    if [ ! -s "$_forward_port_file" ] || ! curl -sf --max-time 10 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
        log "FATAL: local loopback forwarder to $IX_SERVER_URL did not come up"
        exit 1
    fi
fi

log "server=$(ix_server_url) (local_port=$PORT) suites=${SUITE_LIST[*]} eval_conc=$EVAL_CONC gsm8k_limit=${GSM8K_LIMIT:-all} ix_ref=$IX_REF"
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

if [ -z "$MODEL_PREFIX_ARG" ]; then
    MODEL_PREFIX_ARG=$(python3 -c '
import re, sys
m = sys.argv[1].lower()
if "kimi" in m and "k3" in m:
    print("kimik3")
else:
    stem = m.rsplit("/", 1)[-1]
    print(re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-") or "model")
' "$MODEL")
fi
PRECISION_ARG="${PRECISION_ARG:-fp8}"

# ---- Scratch and the InferenceX checkout ------------------------------------
SCRATCH="${AGENTX_SCRATCH_DIR:-/tmp/.agentx_scratch}"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
# Sets IX_ROOT and IX_EVALS; the same checkout agentx_client.sh would reuse.
ix_checkout "$SCRATCH/InferenceX" "$IX_REF" || exit $?
if [ -d "$SCRATCH/InferenceX/.git" ]; then
    log "InferenceX $(git -C "$SCRATCH/InferenceX" rev-parse --short HEAD) (tree $IX_ROOT)"
else
    log "InferenceX (tree $IX_ROOT)"
fi
for f in benchmarks/benchmark_lib.sh benchmarks/runtime_settings.sh \
         infx/evals/thresholds.yaml infx/evals/validate_scores.py; do
    [ -f "$IX_ROOT/$f" ] || { log "FATAL: no $f in the InferenceX checkout at $IX_ROOT"; exit 1; }
done

# ---- InferenceX's eval library -----------------------------------------------
# shellcheck disable=SC1091
source "$IX_ROOT/benchmarks/runtime_settings.sh"
export INFMAX_CONTAINER_WORKSPACE="$IX_ROOT"
# shellcheck disable=SC1091
source "$IX_ROOT/benchmarks/benchmark_lib.sh"

# Standalone callers need not pre-export topology; default to single-worker.
if [ -z "${IS_MULTINODE:-}" ]; then
    ix_export_topology 1 1 1 0
fi
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

normalize_suite_results() {
    local suite="$1" suite_dir="$2" out_json="$3"
    python3 -c '
import json, pathlib, sys
suite, suite_dir, out_path = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
lm_jsons = sorted(
    suite_dir.rglob("results*.json"),
    key=lambda p: p.stat().st_mtime,
)
if suite in ("mmlu_pro", "gsm8k") and lm_jsons:
    payload = json.loads(lm_jsons[-1].read_text())
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    sys.exit(0)

# Kimi tool-call schema suites emit a vendor JSON summary; translate it into
# check_regression.py results_<suite>.json schema.
candidates = sorted(
    (
        p for p in suite_dir.rglob("*.json")
        if p.name != "meta_env.json" and not p.name.startswith("samples_")
    ),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
score = None
raw_meta = {}
for path in candidates:
    try:
        doc = json.loads(path.read_text())
    except Exception:
        continue
    if not isinstance(doc, dict):
        continue
    if "results" in doc and isinstance(doc["results"], dict) and suite in doc["results"]:
        out_path.write_text(json.dumps(doc, indent=2) + "\n")
        sys.exit(0)
    for key in ("schema_accuracy", "tool_call_f1", "accuracy", "acc", "pass_rate", "score"):
        val = doc.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            score = float(val) / 100.0 if float(val) > 1.0 else float(val)
            raw_meta = doc
            break
    if score is None:
        ok = doc.get("successful_ToolCalls", doc.get("valid", doc.get("passed")))
        total = doc.get("total_ToolCalls", doc.get("total"))
        if (
            isinstance(ok, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            score = float(ok) / float(total)
            raw_meta = doc
    if score is not None:
        break

if score is None:
    sys.exit(f"Could not extract normalized score for {suite} under {suite_dir}")

normalized = {
    "results": {
        suite: {
            "acc,none": score,
            "acc_stderr,none": 0.0,
        }
    },
    "meta": {
        "source_suite": suite,
        "raw": raw_meta,
    },
}
out_path.write_text(json.dumps(normalized, indent=2) + "\n")
' "$suite" "$suite_dir" "$out_json"
}

# ---- Run each suite, then judge it --------------------------------------------
FAIL=0
: > "$RESULTS_DIR/summary.md"
for idx in "${!SUITE_LIST[@]}"; do
    suite="${SUITE_LIST[$idx]}"
    dir="$RESULTS_DIR/$suite"
    rm -rf "$dir"; mkdir -p "$dir"
    log "===== $suite ($((idx+1))/${#SUITE_LIST[@]}) ====="
    if ! server_alive; then log "FATAL: server down before $suite"; FAIL=1; break; fi

    if [ "$suite" = "mmlu_pro" ]; then
        T0=$(date +%s)
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENAI_API_KEY=EMPTY lm_eval \
            --tasks mmlu_pro \
            --limit "$MMLU_LIMIT" \
            --seed "0,1234,None,1234" \
            --model local-chat-completions \
            --apply_chat_template \
            --model_args "model=$MODEL,base_url=$(ix_server_url)/v1/chat/completions,num_concurrent=$EVAL_CONC,max_retries=3,timeout=1800" \
            --gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false, "thinking": false}}' \
            --output_path "$dir" > "$dir/eval.log" 2>&1
        rc=$?
        vrc=0
        tail -25 "$dir/eval.log" | cut -c1-300
        log "$suite: lm_eval rc=$rc wall $(( $(date +%s) - T0 )) s"
        if [ "$rc" -eq 0 ]; then
            echo "PASS: mmlu_pro completed (rc=0)" > "$dir/validate.log"
        else
            echo "FAIL: mmlu_pro lm_eval failed (rc=$rc)" > "$dir/validate.log"
        fi
    else
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
        ( cd "$dir" && PYTHONPATH="$IX_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -P \
            -m infx.evals.validate_scores --expected-concs "$EVAL_CONC" ) > "$dir/validate.log" 2>&1
        vrc=$?
        cat "$dir/validate.log"
    fi

    if [ "$rc" -eq 0 ]; then
        if ! normalize_suite_results "$suite" "$dir" "$RESULTS_DIR/results_${suite}.json"; then
            log "ERROR: failed to write normalized results_${suite}.json for $suite"
            rc=1
        fi
    fi

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
    if [ -f "$RESULTS_DIR/results_${suite}.json" ]; then
        save_now "$dir" "$RESULTS_DIR/results_${suite}.json" "$RESULTS_DIR/summary.md"
    else
        save_now "$dir" "$RESULTS_DIR/summary.md"
    fi
    if [ "$idx" -lt $(( ${#SUITE_LIST[@]} - 1 )) ]; then
        wait_server_idle || log "WARNING: the engine did not drain after $suite"
    fi
done

log "===== summary ($MODEL_PREFIX_ARG $PRECISION_ARG on $RUNNER_TYPE, eval conc $EVAL_CONC) ====="
cat "$RESULTS_DIR/summary.md"
server_alive || { log "ERROR: server is dead at the end of the evals"; FAIL=1; }
[ "$FAIL" = "0" ] || log "evals finished with failures; exiting 1"
exit $FAIL

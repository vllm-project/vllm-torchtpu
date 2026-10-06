#!/usr/bin/env bash
# The InferenceX AgentX client, run against a vLLM server that is already
# serving. Not specific to a model: what it is told are the server's URL, the
# chip count to divide by, and the labels InferenceX files the row under. It clones InferenceX, installs aiperf and the trace corpus,
# runs a GSM8K accuracy probe, then replays the AgentX trace corpus once per
# concurrency in --conc-list, ascending, waiting for the server to drain between
# points. The summary table is the output.
#
#   agentx_client.sh --conc-list 1,2,4,8|none --chips N --model-prefix NAME
#       --precision LABEL [--reference-row "..."] [--server-url URL]
#       [--engine-url URL] [--runner-type tpu7x-32] [--results-dir DIR]
#       [--duration 1800] [--fast 1|0] [--ix-ref main] [--tokenizer-dir DIR]
#       [--model NAME] [--max-model-len N] [--gsm8k-limit 100]
#       [--gsm8k-conc 8[,16,...]]
#       [--gcs-root gs://...] [--port 8000]
#
# It starts no server, and it is not told how one is deployed. agentx_sweep.sh
# starts one from a config and then calls this; a lane whose servers are already
# running calls this directly - a kube P/D JobSet does so from its CPU pod, with
# --server-url pointing at the proxy in front of however many prefill and decode
# engines it has. --chips is the one thing the deployment has to state, because
# per-chip throughput is a division and nothing in a single /v1/models response
# can tell this script what to divide by.
#
# One live server serves the whole ladder: aiperf's per-invocation cache-bust
# marker keeps the points' KV keyspaces disjoint, the same shape as InferenceX's
# benchmarks/multi_node/agentic_srt.sh.
#
# --conc-list is required and is not defaulted here. The committed list lives in
# the serve config (AGENTX_CONC_LIST), which is also what sizes the server's
# MAX_NUM_SEQS, and agentx_sweep.sh passes it through; a client that guessed its
# own would be measuring a server that was not built for it.
#
# Exit status: 1 if the server was unreachable or died, a GSM8K probe failed, or
# any replay point failed; 0 otherwise. aiperf's own per-point validation
# verdict is in the summary table.
set -uo pipefail

PORT=8000               # only used when --server-url is absent
# The server to send requests to. Empty means a server on this host at $PORT,
# which is what agentx_sweep.sh starts. --server-url, or AIPERF_SERVER_URL in
# the environment, names one somewhere else; AIPERF_SERVER_URL is the variable
# InferenceX's own srt_agentic.sh reads in the same situation.
IX_SERVER_URL="${AIPERF_SERVER_URL:-}"
RESULTS_DIR="/perf_eval_results/agentx-client"
CONC_LIST=""
DURATION=1800
FAST=1
# Pinned, not main. InferenceX refactors its benchmark harness under us: in one
# week it moved its tree into inferencex-e2e/, added ten env vars that
# build_replay_cmd validates, and then replaced benchmark_lib.sh - which this
# script sources - with the infx.bench package (#3610). A nightly that tracks
# main races those. 8ba71d0 is the last ref this lane ran clean on (build 65:
# both ladder points, ok=19 err=0, isl p50 337k). Bump it deliberately, with a
# run to show the new ref works; --ix-ref overrides for a one-off.
IX_REF="8ba71d0"
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"         # comma list allowed: one GSM8K run per value
TOK_DIR_ARG=""         # a local tokenizer directory, when the caller pulled the weights
MODEL_ARG=""           # served model name; read from the server when empty
MAX_MODEL_LEN_ARG=""   # context window; read from the server when empty
# The number of TPU chips serving this benchmark. InferenceX divides total
# throughput by it to get the per-chip number that is the scenario's headline
# metric, so it is the serving deployment's whole chip count: TP x DP for one
# server, and prefill chips + decode chips for a P/D set. It cannot be derived
# from one engine's TP, so the caller states it and there is no default.
CHIPS=""
RUNNER_TYPE_ARG=""     # InferenceX runner id; "tpu7x-$CHIPS" when empty
# What InferenceX files the result row under. Both describe the model being
# served, which this script does not choose, so the caller states them: the
# aggregate lane reads them out of the serve config, and so does the kube one.
MODEL_PREFIX_ARG=""
PRECISION_ARG=""
# An optional line printed under the summary, for reading our numbers against a
# published row. The caller supplies it because which row is comparable is a
# property of the model and the scenario, not of this client.
REFERENCE_ROW=""
# The result filename prefix, and the spec-decoding method to record with the
# row. Both describe the server, which this script did not configure, so the
# caller states them: agentx_sweep.sh composes them from the serve knobs it
# passed, and a kube lane from its own config. Defaulted so an ad-hoc run
# still writes somewhere sensible.
LABEL="agentx"
SPEC_METHOD="none"
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)    IX_SERVER_URL="$2"; shift 2 ;;
        --engine-url)    IX_ENGINE_URL="$2"; shift 2 ;;
        --conc-list)     CONC_LIST="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --fast)          FAST="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        --gsm8k-limit)   GSM8K_LIMIT="$2"; shift 2 ;;
        --gsm8k-conc)    GSM8K_CONC="$2"; shift 2 ;;
        --tokenizer-dir) TOK_DIR_ARG="$2"; shift 2 ;;
        --model)         MODEL_ARG="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN_ARG="$2"; shift 2 ;;
        --label)         LABEL="$2"; shift 2 ;;
        --spec-method)   SPEC_METHOD="$2"; shift 2 ;;
        --gcs-root)      GCS_ARG="$2"; shift 2 ;;
        --chips)         CHIPS="$2"; shift 2 ;;
        --runner-type)   RUNNER_TYPE_ARG="$2"; shift 2 ;;
        --model-prefix)  MODEL_PREFIX_ARG="$2"; shift 2 ;;
        --precision)     PRECISION_ARG="$2"; shift 2 ;;
        --reference-row) REFERENCE_ROW="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CONC_LIST" ]; then
    echo "agentx_client.sh: --conc-list is required" >&2
    exit 2
fi
case "${CHIPS:-}" in
    ''|*[!0-9]*|0) echo "agentx_client.sh: --chips N is required (the serving deployment's chip count; it is what per-chip throughput is divided by)" >&2; exit 2 ;;
esac
RUNNER_TYPE="${RUNNER_TYPE_ARG:-tpu7x-$CHIPS}"
# Both are labels InferenceX files the row under, and neither has a sensible
# default here, so they are required rather than guessed.
if [ -z "$MODEL_PREFIX_ARG" ]; then
    echo "agentx_client.sh: --model-prefix NAME is required (InferenceX files the row under it)" >&2
    exit 2
fi
if [ -z "$PRECISION_ARG" ]; then
    echo "agentx_client.sh: --precision LABEL is required (InferenceX files the row under it)" >&2
    exit 2
fi

if [ "$CONC_LIST" = "none" ]; then CONCS=(); else IFS=',' read -r -a CONCS <<< "$CONC_LIST"; fi
IFS=',' read -r -a GSM8K_CONCS <<< "$GSM8K_CONC"

IX_SERVER_URL="${IX_SERVER_URL%/}"
export IX_SERVER_URL
# Where /metrics is read from, when that is not the server requests go to: a P/D
# pair is driven through a proxy that has none, and the queue depth and the
# preemption counters are one engine's anyway.
IX_ENGINE_URL="${IX_ENGINE_URL:-}"
IX_ENGINE_URL="${IX_ENGINE_URL%/}"
export IX_ENGINE_URL
if [ -n "$IX_SERVER_URL" ]; then
    # benchmark_lib.sh works out its own URL when this is unset, which is what a
    # server on this host has always relied on, so it is set only when the
    # server is elsewhere.
    export AIPERF_SERVER_URL="$IX_SERVER_URL"
fi
export PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IX_SWEEP_TAG="agentx-client"
# shellcheck source=scripts/vllm/benchmark/inferencex/sweep_lib.sh
source "$SCRIPT_DIR/sweep_lib.sh"

# One validator for both lists, shared with eval_conc_sweep.sh, so a 0 or an
# 007 cannot reach aiperf's --concurrency or lm_eval's num_concurrent.
[ "$CONC_LIST" = "none" ] || ix_check_conc_list "$CONC_LIST" || exit 1
ix_check_conc_list "$GSM8K_CONC" || exit 1
SUMMARY_PY="$SCRIPT_DIR/sweep_summary.py"
mkdir -p "$RESULTS_DIR"
# Absolute from here on. aiperf resolves --output-artifact-dir against its own
# cwd while InferenceX's aggregation runs under cd "$INFMAX_CONTAINER_WORKSPACE"
# (the InferenceX tree), so a relative results dir has the replay writing one
# path and the aggregation reading another - which is how build 65 replayed
# successfully and then reported profile_export.jsonl not found. Upstream does
# not hit it because its results live inside that workspace; ours do not.
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
ix_set_gcs_root "$GCS_ARG"

# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    # Once, here: run_benchmarks.sh appends to server.log for the whole run, so
    # uploading it per point would re-send a growing file every time.
    save_now "$RESULTS_DIR/server.log"
    log "results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
    log "disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
}
trap cleanup EXIT INT TERM

log "server=$(ix_server_url) concs=${CONCS[*]} gsm8k_limit=$GSM8K_LIMIT gsm8k_conc=$GSM8K_CONC duration=$DURATION fast=$FAST ix_ref=$IX_REF"
log "host: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, python $(python3 --version 2>&1), uv $(uv --version 2>&1 || echo missing)"
log "disk (container view):"; df -h / /tmp /root/.cache /perf_eval_results /dev/shm 2>/dev/null

if ! server_alive; then
    log "FATAL: no server answering /v1/models at $(ix_server_url)"
    exit 1
fi
# The served name and the context window come from the server. vLLM puts
# max_model_len in the ModelCard (entrypoints/openai/models/serving.py), and the
# GSM8K probe needs it for lm_eval's max_length. A caller may pass --model or
# --max-model-len as what it believes it started; those are checked against the
# server rather than preferred, because a model id that disagrees 404s every
# request after a 30-minute boot, and a clamped context silently changes which
# traces are eligible.
_models_json=$(curl -sf --max-time 30 "$(ix_server_url)/v1/models") \
    || { log "FATAL: cannot read /v1/models from $(ix_server_url)"; exit 1; }
read -r _served_model _served_ctx <<< "$(printf '%s' "$_models_json" | python3 -c '
import json, sys
card = json.load(sys.stdin)["data"][0]
print(card["id"], card.get("max_model_len") or 0)
')" || { log "FATAL: cannot parse /v1/models from $(ix_server_url)"; exit 1; }
if [ -n "$MODEL_ARG" ] && [ "$MODEL_ARG" != "$_served_model" ]; then
    log "FATAL: --model '$MODEL_ARG' is not what the server serves ('$_served_model')"
    exit 1
fi
MODEL="$_served_model"
if [ -n "$MAX_MODEL_LEN_ARG" ] && [ "${_served_ctx:-0}" -gt 0 ] \
        && [ "$MAX_MODEL_LEN_ARG" != "$_served_ctx" ]; then
    log "WARNING: --max-model-len $MAX_MODEL_LEN_ARG but the server reports" \
        "$_served_ctx; using the server's"
fi
MAX_MODEL_LEN="${_served_ctx:-$MAX_MODEL_LEN_ARG}"
# Exported, not just set: benchmark_lib.sh builds the replay command from the
# environment, and this is what caps which traces are eligible. The 'replay
# cmd: line logged below shows the cap it ended up with.
export MAX_MODEL_LEN
if [ "${MAX_MODEL_LEN:-0}" -le 0 ] 2>/dev/null; then
    log "FATAL: the server did not report max_model_len; pass --max-model-len"
    exit 1
fi
log "served model: $MODEL, context $MAX_MODEL_LEN"

# ---- Client scratch (venv on root disk: /dev/shm is noexec) -----------------
SCRATCH=/tmp/.agentx_scratch
DATA_SCRATCH="$SCRATCH"
shm_avail=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${shm_avail:-}" ] && [ "$shm_avail" -gt $((20 * 1024 * 1024)) ]; then
    DATA_SCRATCH=/dev/shm/.agentx_scratch
fi
mkdir -p "$SCRATCH" "$DATA_SCRATCH"
export TMPDIR="$DATA_SCRATCH/tmp"; mkdir -p "$TMPDIR"
export AIPERF_UV_CACHE_DIR="$DATA_SCRATCH/uv-cache"
IX_DIR="$SCRATCH/InferenceX"
IX_URL=https://github.com/SemiAnalysisAI/InferenceX
# ix_layout_root <dir>: the directory inside the checkout that holds
# benchmarks/, utils/ and the evals, or nothing when the checkout is unusable.
# InferenceX moved that tree down into inferencex-e2e/ (the root now also holds
# collectivex/, operatorx/ and power_model/), and --ix-ref can still pin a
# commit from before the move, so this is probed rather than hardcoded.
ix_layout_root() {
    local d
    for d in "$1/inferencex-e2e" "$1"; do
        [ -d "$d/utils/aiperf/src" ] && { echo "$d"; return 0; }
    done
    return 1
}

if ! ix_layout_root "$IX_DIR" >/dev/null; then
    rm -rf "$IX_DIR"
    if [[ "$IX_REF" =~ ^[0-9a-f]{7,40}$ ]]; then
        # A commit cannot be cloned with --branch: fetch just that commit.
        if ! { git init -q "$IX_DIR" \
                && git -C "$IX_DIR" fetch -q --depth 1 "$IX_URL" "$IX_REF" \
                && git -C "$IX_DIR" checkout -q FETCH_HEAD \
                && git -C "$IX_DIR" submodule update -q --init --depth 1 --recursive; }; then
            log "FATAL: cannot fetch InferenceX commit $IX_REF from $IX_URL"; exit 1
        fi
    else
        git clone -q --depth 1 --branch "$IX_REF" --recurse-submodules --shallow-submodules "$IX_URL" "$IX_DIR" \
            || { log "FATAL: cannot clone InferenceX branch/tag $IX_REF from $IX_URL"; exit 1; }
    fi
fi
IX_ROOT=$(ix_layout_root "$IX_DIR") || {
    log "FATAL: $IX_DIR has no utils/aiperf/src under its root or inferencex-e2e/"; exit 1
}
# The eval task files and their lm-eval patch moved from utils/evals to
# infx/evals in the same reorganisation.
IX_EVALS=""
for _d in "$IX_ROOT/infx/evals" "$IX_ROOT/utils/evals"; do
    [ -d "$_d" ] && { IX_EVALS="$_d"; break; }
done
log "InferenceX $(git -C "$IX_DIR" rev-parse --short HEAD) (tree $IX_ROOT), aiperf $(git -C "$IX_ROOT/utils/aiperf" rev-parse --short HEAD)"

TOK_DIR=""
if [ -n "$TOK_DIR_ARG" ]; then
    if [ -f "$TOK_DIR_ARG/config.json" ]; then
        TOK_DIR="$TOK_DIR_ARG"
        log "client tokenizer: $TOK_DIR"
    else
        log "WARNING: --tokenizer-dir '$TOK_DIR_ARG' has no config.json; client will resolve $MODEL on the hub"
    fi
fi

sanity_probe() {
    log "--- probe: /v1/chat/completions ---"
    curl -s --max-time 300 "$(ix_server_url)/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]; m = c.get('message', {})
print('[k3-agentx-client] chat content:', repr(m.get('content')), '| reasoning:', repr((m.get('reasoning_content') or '')[:120]), '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
print('[k3-agentx-client] error:', r.get('error')) if 'error' in r else None
" || log "chat probe FAILED"
    log "--- probe: chat with tools + tool result ---"
    curl -s --max-time 300 "$(ix_server_url)/v1/chat/completions" -H 'Content-Type: application/json' -d @- << JSON | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'error' in r:
    print('[k3-agentx-client] tool probe ERROR:', json.dumps(r['error'])[:600])
else:
    c = r.get('choices', [{}])[0]; m = c.get('message', {})
    print('[k3-agentx-client] tool probe content:', repr((m.get('content') or '')[:200]), '| tool_calls:', json.dumps(m.get('tool_calls'))[:300], '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
"
{"model":"$MODEL","max_tokens":128,"temperature":0,
 "tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
 "messages":[{"role":"system","content":"You are a coding agent."},
             {"role":"user","content":"Read README.md and tell me its first line."},
             {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},
             {"role":"tool","tool_call_id":"call_1","content":"# Hello World\nThis is a test repo."}]}
JSON
}
sanity_probe

# ---- (1) GSM8K on the fresh server (before any ignore_eos load) -------------
FAIL=0
declare -a GSM8K_ROWS=()
for gc in "${GSM8K_CONCS[@]}"; do
    [ "$GSM8K_LIMIT" -gt 0 ] || break
    if [ -z "$IX_EVALS" ]; then
        log "ERROR: no evals tree in $IX_ROOT (infx/evals or utils/evals); skipping GSM8K"
        FAIL=1
        break
    fi
    log "===== GSM8K limit=$GSM8K_LIMIT num_concurrent=$gc ====="
    EVAL_DIR="$RESULTS_DIR/gsm8k_c$gc"; mkdir -p "$EVAL_DIR"
    T0=$(date +%s)
    # A server with a reasoning parser puts the answer in reasoning_content, so
    # the answer arrives in reasoning_content, which stock lm_eval never reads.
    # InferenceX's gsm8k.yaml + its lm_eval sitecustomize patch score it; the
    # patch targets lm-eval 0.4.9.x and overrides TemplateAPI.apply_chat_template,
    # so re-check it whenever the image's lm-eval pin moves.
    if [ ! -f "$IX_EVALS/patches/lm_eval_sitecustomize.py" ]; then
        log "FATAL: no lm_eval patch at $IX_EVALS/patches/lm_eval_sitecustomize.py;" \
            "unpatched lm_eval reads only 'content', so a model that answers in" \
            "reasoning_content scores 0.0 and looks like a model failure, not a" \
            "missing patch"
        exit 1
    fi
    PATCH_DIR=$(mktemp -d); cp "$IX_EVALS/patches/lm_eval_sitecustomize.py" "$PATCH_DIR/sitecustomize.py"
    PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    lm_eval --tasks "$IX_EVALS/gsm8k.yaml" --limit "$GSM8K_LIMIT" --seed "0,1234,None,1234" \
        --model local-chat-completions --apply_chat_template --log_samples \
        --model_args "model=$MODEL,base_url=$(ix_server_url)/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=$gc,timeout=1800,tokenized_requests=False,max_length=$MAX_MODEL_LEN" \
        --gen_kwargs "max_tokens=4096,temperature=0,top_p=1" \
        --output_path "$EVAL_DIR" > "$EVAL_DIR/lm_eval.log" 2>&1
    rc=$?
    tail -25 "$EVAL_DIR/lm_eval.log" | cut -c1-300
    log "gsm8k conc $gc: rc=$rc wall $(( $(date +%s) - T0 )) s"
    if [ "$rc" -ne 0 ]; then
        FAIL=1
        log "ERROR: GSM8K probe at conc $gc failed (rc=$rc); see $EVAL_DIR/lm_eval.log"
    fi
    rm -rf "$PATCH_DIR"
    GSM8K_ROWS+=("$(python3 "$SUMMARY_PY" gsm8k-row --output-dir "$EVAL_DIR" --conc "$gc")")
    save_now "$EVAL_DIR" "$RESULTS_DIR/config.json"
    wait_server_idle || log "WARNING: the engine did not drain after the conc $gc probe"
done

print_gsm8k_table() {
    [ "${#GSM8K_ROWS[@]}" -gt 0 ] || return 0
    log "===== GSM8K (limit $GSM8K_LIMIT, $LABEL) ====="
    { echo "| eval concurrency | strict-match | flexible-extract |"; echo "|---|---|---|"
      printf '%s\n' "${GSM8K_ROWS[@]}"; } | tee -a "$RESULTS_DIR/summary.md"
}

if [ "${#CONCS[@]}" -eq 0 ]; then
    : > "$RESULTS_DIR/summary.md"
    print_gsm8k_table
    save_now "$RESULTS_DIR/summary.md"
    log "no replay concurrencies requested (--conc-list none); done."
    exit $FAIL
fi

# ---- (2) AgentX replay, one concurrency at a time, ascending ----------------
# InferenceX's own defaults. build_replay_cmd validates ten variables and
# defaults none of them, and nothing inside InferenceX sources this file - the
# workflow driving it is meant to. Taking theirs keeps us at parity on the
# thresholds the submission is judged by. Our departures come after it.
if [ ! -f "$IX_ROOT/benchmarks/runtime_settings.sh" ]; then
    log "FATAL: no benchmarks/runtime_settings.sh in the InferenceX checkout at $IX_ROOT"
    exit 1
fi
# shellcheck source=/dev/null
source "$IX_ROOT/benchmarks/runtime_settings.sh"

# A proxy in front of a P/D pair does not re-export engine metrics, and neither
# does a plain server's /metrics reach aiperf through one. benchmark_lib.sh turns
# this list into one --server-metrics flag with a URL per engine, keeping
# endpoint_url per series, and srt_agentic.sh builds it the same way for anything
# but a dynamo frontend. One server here; a caller fronting several engines sets
# it to all of them.
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_METRICS_URLS:-$(ix_engine_url)/metrics}"

export INFMAX_CONTAINER_WORKSPACE="$IX_ROOT"
export AIPERF_RUNTIME_DIR="$SCRATCH/inferencex-agentic"
# The trace corpus goes into the HF cache; the pod mounts a persistent one at
# /root/.cache/huggingface, so use it when writable and fall back to scratch.
if [ -w /root/.cache/huggingface ] 2>/dev/null; then export HF_HOME=/root/.cache/huggingface; else export HF_HOME="$DATA_SCRATCH/hf_home"; fi
# 3.12 over their 3.11: the image's own interpreter, so uv fetches no second.
export AIPERF_PYTHON_VERSION="${AGENTX_PYTHON_VERSION:-3.12}"
export KV_OFFLOADING=none
unset KV_OFFLOAD_BACKEND
export DURATION MODEL
export MODEL_PREFIX="$MODEL_PREFIX_ARG"
# TP and EP_SIZE here are not our parallelism: they are what InferenceX divides
# total throughput by, so both carry the chip count (the same thing the CDK
# AgentX recipe calls TPU_CHIPS).
export TP="$CHIPS" EP_SIZE="$CHIPS" PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
# PRECISION is a label in the aggregate, not something the client measures: it
# is how InferenceX files the row, so it has to match what the published rows
# for this model use.
export FRAMEWORK=vllm
export PRECISION="$PRECISION_ARG"
export RUNNER_TYPE
SPEC_DECODING="$SPEC_METHOD"
export SPEC_DECODING
export IMAGE="${BUILDKITE_COMMIT:-local}"
export AGENTIC_OUTPUT_DIR="$RESULTS_DIR"
# No power measurement here; REQUIRE_POWER follows, and both are validated
# beside IS_MULTINODE (benchmark_lib.sh:3402).
export ENABLE_AGENTX_POWER=0
export REQUIRE_POWER="${REQUIRE_POWER:-false}"
# Recipe-owned, so runtime_settings.sh omits it. True: each role spans four
# hosts. The string, not 0/1.
export IS_MULTINODE="${IS_MULTINODE:-true}"
export AIPERF_EXPERIMENTAL_FAST="$FAST"
# Over their 1800: one ~70k-token turn can take >30 min on a prefill-bound
# engine. Unconditional, since runtime_settings.sh has just set it, and before
# POINT_CAP, which is arithmetic over it ($(( )) over an unset name is fatal
# under set -u).
export AGENTIC_WARMUP_GRACE_PERIOD="${AGENTX_WARMUP_GRACE:-3600}"
export AIPERF_DRAIN_TIMEOUT_SECONDS="${AIPERF_DRAIN_TIMEOUT_SECONDS:-3600}"

# Per-point hard cap, and the ladder's whole budget. The cap has to be large
# enough for warmup plus the window and small enough that it fires before the
# build's own timeout kills the step - otherwise the point is killed with no
# marker and the summary cannot say why a row is missing.
POINT_CAP=$(( AGENTIC_WARMUP_GRACE_PERIOD + DURATION + 900 ))
log "per-point cap ${POINT_CAP}s (warmup grace $AGENTIC_WARMUP_GRACE_PERIOD + window $DURATION + 900);" \
    "${#CONCS[@]} points, so the ladder needs up to $(( POINT_CAP * ${#CONCS[@]} / 60 )) min" \
    "on top of the boot - the step's timeout must exceed that"

# shellcheck source=/dev/null  # lives in the InferenceX checkout
source "$IX_ROOT/benchmarks/benchmark_lib.sh"
resolve_trace_source            # aiperf venv + corpus download (HF, public)
UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" -q blobfile tiktoken \
    || log "WARNING: blobfile install failed"

BASE_RESULT_FILENAME="$LABEL"
declare -a DONE_CONCS=()
for idx in "${!CONCS[@]}"; do
    c="${CONCS[$idx]}"
    export CONC="$c"
    export RESULT_FILENAME="${BASE_RESULT_FILENAME}_c${c}"
    export RESULT_DIR="$RESULTS_DIR/conc_${c}"
    mkdir -p "$RESULT_DIR"
    # Outside RESULT_DIR, which is uploaded as the step's artifacts.
    POINT_DONE="$SCRATCH/point_done_c${c}"
    rm -f "$POINT_DONE"
    log "===== AgentX conc $c ($((idx+1))/${#CONCS[@]}) ====="
    if ! server_alive; then log "FATAL: server down before conc $c"; FAIL=1; break; fi
    build_replay_cmd "$RESULT_DIR"
    if [ -n "$TOK_DIR" ]; then
        REPLAY_CMD="${REPLAY_CMD/--tokenizer $MODEL/--tokenizer $TOK_DIR}"
    fi
    log "replay cmd: $REPLAY_CMD"
    T0=$(date +%s)
    pre0=$(ix_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
    # aiperf can hang in its own shutdown after a warmup-drain timeout, so each
    # point runs under POINT_CAP. On expiry every aiperf process is killed and
    # the point is marked HARD_TIMEOUT.
    REPLAY_CAP="$POINT_CAP"
    ( run_agentic_replay_and_write_outputs "$RESULT_DIR" ) &
    replay_pid=$!
    ( sleep "$REPLAY_CAP"
      # Killing the sleep is not enough: the subshell would carry on and both
      # announce a timeout and mark the row with one.
      [ -e "$POINT_DONE" ] && exit 0
      echo "[$IX_SWEEP_TAG] conc $c: HARD TIMEOUT after ${REPLAY_CAP}s, killing the client"
      touch "$RESULT_DIR/HARD_TIMEOUT"
      pkill -TERM -f "aiperf" 2>/dev/null; sleep 20
      pkill -KILL -f "aiperf" 2>/dev/null; kill -KILL "$replay_pid" 2>/dev/null ) &
    watchdog_pid=$!
    wait "$replay_pid"; rc=$?
    touch "$POINT_DONE"
    pkill -P "$watchdog_pid" 2>/dev/null   # the watchdog's sleep
    kill "$watchdog_pid" 2>/dev/null; wait "$watchdog_pid" 2>/dev/null
    pkill -KILL -f "aiperf" 2>/dev/null   # never leave a client behind between points
    pre1=$(ix_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
    log "conc $c: replay+aggregate rc=$rc wall=$(( $(date +%s) - T0 )) s engine-preemptions=$(( ${pre1:-0} - ${pre0:-0} )) (a preempted request resumes from its saved state; a model with recurrent per-sequence state makes a non-zero count worth checking against output quality)"
    if [ ! -f "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json" ]; then
        FAIL=1
        log "ERROR: conc $c produced no aggregate ($RESULT_FILENAME.json); $([ -f "$RESULT_DIR/HARD_TIMEOUT" ] && echo 'hard timeout' || echo 'client or aggregation failed, see the replay log above')"
    fi
    save_now "$RESULT_DIR" "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json"
    DONE_CONCS+=("$c")
    # A point that starts on an undrained engine is measuring something else, so
    # record it rather than letting the next row look clean.
    if [ "$idx" -lt $(( ${#CONCS[@]} - 1 )) ] && ! wait_server_idle; then
        FAIL=1
        log "ERROR: the engine did not drain after conc $c; the next point starts dirty"
    fi
done

# ---- Summary ----------------------------------------------------------------
log "===== summary ($LABEL fast=$FAST) ====="
: > "$RESULTS_DIR/summary.md"
print_gsm8k_table
if ! python3 "$SUMMARY_PY" agentx-table --results-dir "$RESULTS_DIR" \
        --base "$BASE_RESULT_FILENAME" --concs "${DONE_CONCS[@]}" \
        | tee -a "$RESULTS_DIR/summary.md"; then
    FAIL=1
    log "ERROR: the AgentX table could not be rendered; the per-point JSON is in $RESULTS_DIR"
fi
save_now "$RESULTS_DIR/summary.md"
[ -z "$REFERENCE_ROW" ] || log "reference: $REFERENCE_ROW"
server_alive || { log "ERROR: server is dead at the end of the sweep"; FAIL=1; }
[ "$FAIL" = "0" ] || log "sweep finished with failures; exiting 1"
exit $FAIL

#!/usr/bin/env bash
# The InferenceX AgentX client for Kimi-K3, run against a vLLM server that is
# already serving. It clones InferenceX, installs aiperf and the trace corpus,
# runs a GSM8K accuracy probe, then replays the AgentX trace corpus once per
# concurrency in --conc-list, ascending, waiting for the server to drain between
# points. The summary table is the output.
#
#   agentx_client.sh --conc-list 1,2,4,8 [--server-url URL] [--results-dir DIR]
#       [--duration 1800] [--fast 1|0] [--ix-ref main] [--tokenizer-dir DIR]
#       [--model NAME] [--max-model-len N] [--gsm8k-limit 100]
#       [--gsm8k-conc 8[,16,...]]
#       [--gcs-root gs://...] [--port 8000]
#
# It starts no server. agentx_sweep.sh starts one from a config and then calls
# this; the kube 1P1D lane already has two servers on TPU pods and calls this
# directly from its CPU pod, with --server-url pointing at the proxy in front of
# the pair.
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

PORT=8000
# The server to send requests to. Empty means a server on this host at $PORT,
# which is what agentx_sweep.sh starts. --server-url, or AIPERF_SERVER_URL in
# the environment, names one somewhere else; AIPERF_SERVER_URL is the variable
# InferenceX's own srt_agentic.sh reads in the same situation.
K3_SERVER_URL="${AIPERF_SERVER_URL:-}"
RESULTS_DIR="/perf_eval_results/k3-agentx-client"
CONC_LIST=""
DURATION=1800
FAST=1
IX_REF="main"          # InferenceX git ref; pass --ix-ref <commit> to pin a run (e.g. 9acf24c = 2026-09-07, aiperf 754356e)
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"         # comma list allowed: one GSM8K run per value
TOK_DIR_ARG=""         # a local tokenizer directory, when the caller pulled the weights
MODEL_ARG=""           # served model name; read from the server when empty
MAX_MODEL_LEN_ARG=""   # context window; read from the server when empty
# The result filename prefix, and the spec-decoding method to record with the
# row. Both describe the server, which this script did not configure, so the
# caller states them: agentx_sweep.sh composes them from the serve knobs it
# passed, and the kube lane from its own config. Defaulted so an ad-hoc run
# still writes somewhere sensible.
LABEL="agentx_k3"
SPEC_METHOD="none"
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)    K3_SERVER_URL="$2"; shift 2 ;;
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
        --port)          PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CONC_LIST" ]; then
    echo "agentx_client.sh: --conc-list is required" >&2
    exit 2
fi
if [ "$CONC_LIST" = "none" ]; then CONCS=(); else IFS=',' read -r -a CONCS <<< "$CONC_LIST"; fi
IFS=',' read -r -a GSM8K_CONCS <<< "$GSM8K_CONC"
for c in "${CONCS[@]}" "${GSM8K_CONCS[@]}"; do
    case "$c" in ''|*[!0-9]*) echo "bad concurrency: '$c'" >&2; exit 1 ;; esac
done

K3_SERVER_URL="${K3_SERVER_URL%/}"
export K3_SERVER_URL
if [ -n "$K3_SERVER_URL" ]; then
    # benchmark_lib.sh works out its own URL when this is unset, which is what a
    # server on this host has always relied on, so it is set only when the
    # server is elsewhere.
    export AIPERF_SERVER_URL="$K3_SERVER_URL"
fi
export PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3_SWEEP_TAG="k3-agentx-client"
# shellcheck source=scripts/vllm/benchmark/inferencex/kimi-k3/sweep_lib.sh
source "$SCRIPT_DIR/sweep_lib.sh"
SUMMARY_PY="$SCRIPT_DIR/sweep_summary.py"
mkdir -p "$RESULTS_DIR"
k3_set_gcs_root "$GCS_ARG"

# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    log "results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
    log "disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
}
trap cleanup EXIT

log "server=$(k3_server_url) concs=${CONCS[*]} gsm8k_limit=$GSM8K_LIMIT gsm8k_conc=$GSM8K_CONC duration=$DURATION fast=$FAST ix_ref=$IX_REF"
log "host: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, python $(python3 --version 2>&1), uv $(uv --version 2>&1 || echo missing)"
log "disk (container view):"; df -h / /tmp /root/.cache /perf_eval_results /dev/shm 2>/dev/null

if ! server_alive; then
    log "FATAL: no server answering /v1/models at $(k3_server_url)"
    exit 1
fi
# The served name and the context window both come from the server, so neither
# can disagree with what is actually serving. vLLM puts max_model_len in the
# ModelCard (entrypoints/openai/models/serving.py), and the GSM8K probe needs it
# for lm_eval's max_length.
_models_json=$(curl -sf --max-time 30 "$(k3_server_url)/v1/models") \
    || { log "FATAL: cannot read /v1/models from $(k3_server_url)"; exit 1; }
read -r _served_model _served_ctx <<< "$(printf '%s' "$_models_json" | python3 -c '
import json, sys
card = json.load(sys.stdin)["data"][0]
print(card["id"], card.get("max_model_len") or 0)
')" || { log "FATAL: cannot parse /v1/models from $(k3_server_url)"; exit 1; }
MODEL="${MODEL_ARG:-$_served_model}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_ARG:-$_served_ctx}"
if [ "${MAX_MODEL_LEN:-0}" -le 0 ] 2>/dev/null; then
    log "FATAL: the server did not report max_model_len; pass --max-model-len"
    exit 1
fi
log "served model: $MODEL, context $MAX_MODEL_LEN"

# ---- Client scratch (venv on root disk: /dev/shm is noexec) -----------------
SCRATCH=/tmp/.k3_agentx_scratch
DATA_SCRATCH="$SCRATCH"
shm_avail=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${shm_avail:-}" ] && [ "$shm_avail" -gt $((20 * 1024 * 1024)) ]; then
    DATA_SCRATCH=/dev/shm/.k3_agentx_scratch
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
    curl -s "$(k3_server_url)/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]; m = c.get('message', {})
print('[k3-agentx-sweep] chat content:', repr(m.get('content')), '| reasoning:', repr((m.get('reasoning_content') or '')[:120]), '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
print('[k3-agentx-sweep] error:', r.get('error')) if 'error' in r else None
" || log "chat probe FAILED"
    log "--- probe: chat with tools + tool result ---"
    curl -s "$(k3_server_url)/v1/chat/completions" -H 'Content-Type: application/json' -d @- << JSON | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'error' in r:
    print('[k3-agentx-sweep] tool probe ERROR:', json.dumps(r['error'])[:600])
else:
    c = r.get('choices', [{}])[0]; m = c.get('message', {})
    print('[k3-agentx-sweep] tool probe content:', repr((m.get('content') or '')[:200]), '| tool_calls:', json.dumps(m.get('tool_calls'))[:300], '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
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
    # The server runs --reasoning-parser kimi_k3 (as the B300 recipe does), so
    # the answer arrives in reasoning_content, which stock lm_eval never reads.
    # InferenceX's gsm8k.yaml + its lm_eval sitecustomize patch score it; the
    # patch targets lm-eval 0.4.9.x and overrides TemplateAPI.apply_chat_template,
    # so re-check it whenever the image's lm-eval pin moves.
    PATCH_DIR=$(mktemp -d); cp "$IX_EVALS/patches/lm_eval_sitecustomize.py" "$PATCH_DIR/sitecustomize.py"
    PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    lm_eval --tasks "$IX_EVALS/gsm8k.yaml" --limit "$GSM8K_LIMIT" --seed "0,1234,None,1234" \
        --model local-chat-completions --apply_chat_template --log_samples \
        --model_args "model=$MODEL,base_url=$(k3_server_url)/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=$gc,timeout=1800,tokenized_requests=False,max_length=$MAX_MODEL_LEN" \
        --gen_kwargs "max_tokens=4096,temperature=0,top_p=1" \
        --output_path "$EVAL_DIR" > "$EVAL_DIR/lm_eval.log" 2>&1
    rc=$?
    tail -25 "$EVAL_DIR/lm_eval.log" | cut -c1-300
    log "gsm8k conc $gc: rc=$rc wall $(( $(date +%s) - T0 )) s"
    if [ "$rc" -ne 0 ]; then
        FAIL=1
        log "ERROR: GSM8K probe at conc $gc failed (rc=$rc); see $EVAL_DIR/lm_eval.log"
    fi
    GSM8K_ROWS+=("$(python3 "$SUMMARY_PY" gsm8k-row --output-dir "$EVAL_DIR" --conc "$gc")")
    save_now "$EVAL_DIR" "$RESULTS_DIR/server.log" "$RESULTS_DIR/config.json"
    wait_server_idle
done

if [ "${#CONCS[@]}" -eq 0 ]; then
    log "===== GSM8K summary (limit $GSM8K_LIMIT, $LABEL) ====="
    echo "| eval concurrency | strict-match | flexible-extract |"; echo "|---|---|---|"
    printf '%s\n' "${GSM8K_ROWS[@]}"
    log "no replay concurrencies requested (--conc-list none); done."
    exit $FAIL
fi

# ---- (2) AgentX replay, one concurrency at a time, ascending ----------------
export INFMAX_CONTAINER_WORKSPACE="$IX_ROOT"
export AIPERF_RUNTIME_DIR="$SCRATCH/inferencex-agentic"
# The trace corpus goes into the HF cache; the pod mounts a persistent one at
# /root/.cache/huggingface, so use it when writable and fall back to scratch.
if [ -w /root/.cache/huggingface ] 2>/dev/null; then export HF_HOME=/root/.cache/huggingface; else export HF_HOME="$DATA_SCRATCH/hf_home"; fi
export AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.12}"
export KV_OFFLOADING=none
unset KV_OFFLOAD_BACKEND
export DURATION MODEL
export MODEL_PREFIX=kimik3
export TP=32 EP_SIZE=32 PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
# PRECISION is a label in the aggregate: the checkpoint's experts are MXFP4
# (compressed-tensors weight_packed) with FP8 attention, which InferenceX
# files under "fp4", the same label as the published B300 vLLM run.
export FRAMEWORK=vllm RUNNER_TYPE=tpu7x-32 PRECISION=fp4
SPEC_DECODING="$SPEC_METHOD"
export SPEC_DECODING
export IMAGE="${BUILDKITE_COMMIT:-local}"
export AGENTIC_OUTPUT_DIR="$RESULTS_DIR"
export ENABLE_AGENTX_POWER=0
export AIPERF_EXPERIMENTAL_FAST="$FAST"
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
# One ~70k-token turn can take >30 min on a prefill-bound engine, so the
# warmup drain needs more than the client's default 1800 s grace.
export AGENTIC_WARMUP_GRACE_PERIOD="${AGENTIC_WARMUP_GRACE_PERIOD:-7200}"
export AIPERF_DRAIN_TIMEOUT_SECONDS="${AIPERF_DRAIN_TIMEOUT_SECONDS:-3600}"

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
    log "===== AgentX conc $c ($((idx+1))/${#CONCS[@]}) ====="
    if ! server_alive; then log "FATAL: server down before conc $c"; FAIL=1; break; fi
    build_replay_cmd "$RESULT_DIR"
    if [ -n "$TOK_DIR" ]; then
        REPLAY_CMD="${REPLAY_CMD/--tokenizer $MODEL/--tokenizer $TOK_DIR}"
    fi
    log "replay cmd: $REPLAY_CMD"
    T0=$(date +%s)
    pre0=$(k3_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
    # aiperf can hang in its own shutdown after a warmup-drain timeout, so each
    # point runs under a hard cap (warmup grace + window + 30 min). On expiry
    # every aiperf process is killed and the point is marked HARD_TIMEOUT.
    REPLAY_CAP=$(( AGENTIC_WARMUP_GRACE_PERIOD + DURATION + 1800 ))
    ( run_agentic_replay_and_write_outputs "$RESULT_DIR" ) &
    replay_pid=$!
    ( sleep "$REPLAY_CAP"
      echo "[$K3_SWEEP_TAG] conc $c: HARD TIMEOUT after ${REPLAY_CAP}s, killing the client"
      touch "$RESULT_DIR/HARD_TIMEOUT"
      pkill -TERM -f "aiperf" 2>/dev/null; sleep 20
      pkill -KILL -f "aiperf" 2>/dev/null; kill -KILL "$replay_pid" 2>/dev/null ) &
    watchdog_pid=$!
    wait "$replay_pid"; rc=$?
    pkill -P "$watchdog_pid" 2>/dev/null   # the watchdog's sleep
    kill "$watchdog_pid" 2>/dev/null; wait "$watchdog_pid" 2>/dev/null
    pkill -KILL -f "aiperf" 2>/dev/null   # never leave a client behind between points
    pre1=$(k3_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
    log "conc $c: replay+aggregate rc=$rc wall=$(( $(date +%s) - T0 )) s engine-preemptions=$(( ${pre1:-0} - ${pre0:-0} )) (a preempted request resumes from its saved state; K3 keeps a KDA recurrent state per sequence, so a non-zero count is worth checking against output quality)"
    if [ ! -f "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json" ]; then
        FAIL=1
        log "ERROR: conc $c produced no aggregate ($RESULT_FILENAME.json); $([ -f "$RESULT_DIR/HARD_TIMEOUT" ] && echo 'hard timeout' || echo 'client or aggregation failed, see the replay log above')"
    fi
    save_now "$RESULT_DIR" "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json" "$RESULTS_DIR/server.log"
    DONE_CONCS+=("$c")
    if [ "$idx" -lt $(( ${#CONCS[@]} - 1 )) ]; then wait_server_idle; fi
done

# ---- Summary ----------------------------------------------------------------
log "===== summary ($LABEL fast=$FAST) ====="
python3 "$SUMMARY_PY" agentx-table --results-dir "$RESULTS_DIR" --base "$BASE_RESULT_FILENAME" --concs "${DONE_CONCS[@]}" | tee "$RESULTS_DIR/summary.md"
save_now "$RESULTS_DIR/summary.md"
log "reference snapshot (InferenceX dashboard, 8x B300 vLLM, taken 2026-09-07; out tok/s per GPU @ P90 interactivity): c1 15.1@203, c2 18.6@133, c4 23.4@121, c8 40.6@64, c16 58.4@32, c24 66.3@22, c48 75.7@12.5, c70 87.6@7.9"
server_alive || { log "ERROR: server is dead at the end of the sweep"; FAIL=1; }
[ "$FAIL" = "0" ] || log "sweep finished with failures; exiting 1"
exit $FAIL

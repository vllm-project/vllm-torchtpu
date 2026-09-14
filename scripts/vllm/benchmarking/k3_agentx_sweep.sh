#!/usr/bin/env bash
# Kimi-K3 InferenceX AgentX sweep on the v7x-32 pod: boot the server ONCE,
# then run (1) a GSM8K accuracy probe on the fresh server, (2) the AgentX
# trace replay (aiperf, Claude Code session traces) at each concurrency in
# --conc-list, ascending, waiting for the engine to drain between points. Same
# shape as InferenceX's benchmarks/multi_node/agentic_srt.sh: aiperf's
# per-invocation cache-bust marker keeps the points' KV keyspaces disjoint, so
# one live server is valid for the whole ladder.
#
# Mirrors the InferenceX B300 vLLM recipe's client contract; the server
# geometry is configs/kimi-k3-tp32-ep-agentx.sh. The aiperf client drops
# traces whose peak input+output exceeds --max-model-len, so a run below the
# model's native context replays a subset of the corpus (the client logs how
# many traces survived) and is a bring-up number, not a submission.
#
#   k3_agentx_sweep.sh [--conc-list 1,2,4,8] [--max-model-len 131072]
#       [--kv fp8|auto] [--apc 0|1] [--spec none|dspark] [--fast 1|0]
#       [--duration 1800] [--gsm8k-limit 100] [--gsm8k-conc 8[,16,...]]
#       [--moe-requant 0|512] [--extra-serve-args "..."] [--ix-ref main]
#       [--results-dir DIR] [--gcs-root gs://bucket/prefix]
#
# --conc-list none runs only the GSM8K probe (at each --gsm8k-conc value).
# --fast 1 = 20 min profiling window, one warmup request per lane;
# --fast 0 = the canonical 30 min window (--duration) and full warmup.
# Results land in --results-dir (uploaded as Buildkite artifacts); --gcs-root
# additionally copies each point to <gcs-root>/<build number>/ as soon as it
# is written. --ix-ref takes a branch, tag or commit SHA of InferenceX; the
# default is a pinned SHA because the client library is sourced into this
# process (a moving ref would change the benchmark under us).
#
# Exit status: 1 if the server failed to boot or died, the GSM8K probe
# failed to run, or any replay point produced no aggregate JSON (hard timeout
# or client crash). aiperf's own validation verdict per point is in the
# aggregate, not in the exit status.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-agentx"
RESULTS_DIR="/perf_eval_results/k3-agentx-sweep"
CONC_LIST="1,2,4,8"
MAX_MODEL_LEN_ARG=131072
DURATION=1800
FAST=1
IX_REF="main"          # InferenceX git ref; pass --ix-ref <commit> to pin a run (e.g. 9acf24c = 2026-09-07, aiperf 754356e)
APC=0
KV_DTYPE="fp8"
SPEC="none"            # none | dspark  (dspark needs PR #702 in the branch)
SPEC_TOKENS=7
SPEC_AL=3.84
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC="8"        # comma list allowed, e.g. 8,16,32: one GSM8K run per value, fresh-server order
MOE_REQUANT=0         # 0 = W4A16 (default gmm_v2 path); 512 = W4A8 (fp8 activations)
PORT=8000
GCS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --conc-list)     CONC_LIST="$2"; shift 2 ;;
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
        *) log "unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$CONC_LIST" = "none" ]; then CONCS=(); else IFS=',' read -r -a CONCS <<< "$CONC_LIST"; fi
IFS=',' read -r -a GSM8K_CONCS <<< "$GSM8K_CONC"
MAX_CONC=1
for c in "${CONCS[@]}" "${GSM8K_CONCS[@]}"; do
    [[ "$c" =~ ^[1-9][0-9]*$ ]] || { log "bad concurrency '$c'"; exit 1; }
    [ "$c" -gt "$MAX_CONC" ] && MAX_CONC=$c
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3_SWEEP_TAG="k3-agentx-sweep"
# shellcheck source=scripts/vllm/benchmarking/k3_sweep_lib.sh
source "$SCRIPT_DIR/k3_sweep_lib.sh"
SUMMARY_PY="$SCRIPT_DIR/k3_sweep_summary.py"
mkdir -p "$RESULTS_DIR"
k3_set_gcs_root "$GCS_ARG"

# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    bash "$SCRIPT_DIR/cleanup_server.sh"
    k3_dump_server_log_context
    log "results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
    log "disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
    rm -rf /dev/shm/.k3_agentx_scratch
    if [ -n "$GCS_ROOT" ] && command -v gsutil >/dev/null 2>&1; then
        log "uploading $RESULTS_DIR to $GCS_ROOT"
        gsutil -q -m cp -r "$RESULTS_DIR" "$GCS_ROOT/" 2>&1 | tail -3 || log "WARNING: GCS upload to $GCS_ROOT failed"
    fi
}
trap cleanup EXIT INT TERM

# ---- Server geometry --------------------------------------------------------
export MAX_MODEL_LEN="$MAX_MODEL_LEN_ARG"
export MAX_NUM_SEQS=$((2 * MAX_CONC)); [ "$MAX_NUM_SEQS" -lt 4 ] && export MAX_NUM_SEQS=4
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

log "config=$CONFIG_NAME concs=${CONCS[*]} max_num_seqs=$MAX_NUM_SEQS ctx=$MAX_MODEL_LEN kv=$KV_DTYPE apc=$APC spec=$SPEC/$SPEC_TOKENS/$SPEC_AL moe_requant=$MOE_REQUANT extra=${K3_EXTRA_SERVE_ARGS:-} duration=$DURATION fast=$FAST gsm8k_limit=$GSM8K_LIMIT ix_ref=$IX_REF"
log "host: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, python $(python3 --version 2>&1), uv $(uv --version 2>&1 || echo missing)"
log "disk (container view):"; df -h / /tmp /root/.cache /perf_eval_results /dev/shm 2>/dev/null

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
if [ ! -d "$IX_DIR/utils/aiperf/src" ]; then
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
log "InferenceX $(git -C "$IX_DIR" rev-parse --short HEAD), aiperf $(git -C "$IX_DIR/utils/aiperf" rev-parse --short HEAD)"

# ---- Boot the server once ---------------------------------------------------
k3_boot_server "$CONFIG_NAME" || exit 1
MODEL=$(k3_config_json_get model)
MODEL_URI=$(k3_config_json_get model_uri)
TOK_DIR=""
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
    if [ -n "$TOK_DIR" ] && [ -f "$TOK_DIR/config.json" ]; then
        log "client tokenizer: $TOK_DIR"
    else
        log "WARNING: pulled dir '$TOK_DIR' unusable; client will resolve $MODEL on the hub"
        TOK_DIR=""
    fi
fi

sanity_probe() {
    log "--- probe: /v1/chat/completions ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]; m = c.get('message', {})
print('[k3-agentx-sweep] chat content:', repr(m.get('content')), '| reasoning:', repr((m.get('reasoning_content') or '')[:120]), '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
print('[k3-agentx-sweep] error:', r.get('error')) if 'error' in r else None
" || log "chat probe FAILED"
    log "--- probe: chat with tools + tool result ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- << JSON | python3 -c "
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
    log "===== GSM8K limit=$GSM8K_LIMIT num_concurrent=$gc ====="
    EVAL_DIR="$RESULTS_DIR/gsm8k_c$gc"; mkdir -p "$EVAL_DIR"
    T0=$(date +%s)
    # The server runs --reasoning-parser kimi_k3 (as the B300 recipe does), so
    # the answer arrives in reasoning_content, which stock lm_eval never reads.
    # InferenceX's gsm8k.yaml + its lm_eval sitecustomize patch score it; the
    # patch targets lm-eval 0.4.9.x and overrides TemplateAPI.apply_chat_template,
    # so re-check it whenever the image's lm-eval pin moves.
    PATCH_DIR=$(mktemp -d); cp "$IX_DIR/utils/evals/patches/lm_eval_sitecustomize.py" "$PATCH_DIR/sitecustomize.py"
    PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    lm_eval --tasks "$IX_DIR/utils/evals/gsm8k.yaml" --limit "$GSM8K_LIMIT" --seed "0,1234,None,1234" \
        --model local-chat-completions --apply_chat_template --log_samples \
        --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=$gc,timeout=1800,tokenized_requests=False,max_length=$MAX_MODEL_LEN" \
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
    log "===== GSM8K summary (limit $GSM8K_LIMIT, ctx $MAX_MODEL_LEN, kv $KV_DTYPE, moe_requant $MOE_REQUANT) ====="
    echo "| eval concurrency | strict-match | flexible-extract |"; echo "|---|---|---|"
    printf '%s\n' "${GSM8K_ROWS[@]}"
    log "no replay concurrencies requested (--conc-list none); done."
    exit $FAIL
fi

# ---- (2) AgentX replay, one concurrency at a time, ascending ----------------
export INFMAX_CONTAINER_WORKSPACE="$IX_DIR"
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
SPEC_DECODING=none; [ "$SPEC" = "dspark" ] && SPEC_DECODING=mtp
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
source "$IX_DIR/benchmarks/benchmark_lib.sh"
resolve_trace_source            # aiperf venv + corpus download (HF, public)
UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" -q blobfile tiktoken \
    || log "WARNING: blobfile install failed"

BASE_RESULT_FILENAME="agentx_k3_tpu7x-32_ctx${MAX_MODEL_LEN}_kv${KV_DTYPE}_apc${APC}_spec${SPEC}_rq${MOE_REQUANT}"
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
log "===== summary (ctx=$MAX_MODEL_LEN kv=$KV_DTYPE apc=$APC spec=$SPEC fast=$FAST) ====="
python3 "$SUMMARY_PY" agentx-table --results-dir "$RESULTS_DIR" --base "$BASE_RESULT_FILENAME" --concs "${DONE_CONCS[@]}" | tee "$RESULTS_DIR/summary.md"
save_now "$RESULTS_DIR/summary.md"
log "reference snapshot (InferenceX dashboard, 8x B300 vLLM, taken 2026-09-07; out tok/s per GPU @ P90 interactivity): c1 15.1@203, c2 18.6@133, c4 23.4@121, c8 40.6@64, c16 58.4@32, c24 66.3@22, c48 75.7@12.5, c70 87.6@7.9"
server_alive || { log "ERROR: server is dead at the end of the sweep"; FAIL=1; }
[ "$FAIL" = "0" ] || log "sweep finished with failures; exiting 1"
exit $FAIL

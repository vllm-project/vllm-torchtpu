#!/bin/bash
# Kimi-K3 accuracy vs client concurrency (issue #841).
# One server boot (config kimi-k3-tp32-ep-concsweep), then lm_eval at each
# concurrency in ascending order, for each task, in the nightly's eval form
# (local-chat-completions, plain chat, thinking off). Engine drained between
# points. Every point is uploaded as soon as it is scored.
#
#   k3_eval_conc_sweep.sh --results-dir DIR [--config NAME]
#                         [--conc-list 1,2,4,...] [--tasks mmlu_pro,gsm8k]
#                         [--gsm8k-limit 128] [--mmlu-limit 10]
#                         [--ctx 16384] [--max-num-seqs 256] [--port 8000]
#                         [--gcs-root gs://bucket/prefix]
#
# Results (per-point lm_eval output, samples, summary.md) land in --results-dir,
# which the pipeline uploads as Buildkite artifacts; --gcs-root additionally
# copies each point to <gcs-root>/<build number>/ as soon as it is scored.
#
# Question counts grow with concurrency so that the requested concurrency is
# really reached (k3_sweep_summary.py question-limit): gsm8k limit =
# max(--gsm8k-limit, 2*conc); mmlu_pro limit (per subject, 14 subjects) =
# max(--mmlu-limit, ceil(2*conc/14)).
#
# Exit status: 1 if the server failed to boot, died, or any lm_eval run
# returned non-zero; the summary table shows rc per point either way.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-concsweep"
RESULTS_DIR=""
CONC_LIST="1,2,4,8,16,32,64,128,256"
TASKS="mmlu_pro,gsm8k"
GSM8K_LIMIT=128
MMLU_LIMIT=10
PORT="${PORT:-8000}"
GCS_ARG=""
# K3's chat template names its thinking switch "thinking" (it ignores
# enable_thinking); off keeps every answer inside the task's token budget.
TEMPLATE_KWARGS='{"thinking": false}'
export K3_SWEEP_CTX="${K3_SWEEP_CTX:-16384}"
export K3_SWEEP_MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --config)       CONFIG_NAME="$2"; shift 2 ;;
        --conc-list)    CONC_LIST="$2"; shift 2 ;;
        --tasks)        TASKS="$2"; shift 2 ;;
        --gsm8k-limit)  GSM8K_LIMIT="$2"; shift 2 ;;
        --mmlu-limit)   MMLU_LIMIT="$2"; shift 2 ;;
        --ctx)          export K3_SWEEP_CTX="$2"; shift 2 ;;
        --max-num-seqs) export K3_SWEEP_MAX_NUM_SEQS="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --gcs-root)     GCS_ARG="$2"; shift 2 ;;
        *) echo "[k3-eval-sweep] unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$RESULTS_DIR" ] || { echo "[k3-eval-sweep] ERROR: --results-dir is required"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3_SWEEP_TAG="k3-eval-sweep"
# shellcheck source=scripts/vllm/benchmarking/k3_sweep_lib.sh
source "$SCRIPT_DIR/k3_sweep_lib.sh"
SUMMARY_PY="$SCRIPT_DIR/k3_sweep_summary.py"

k3_check_conc_list "$CONC_LIST" || exit 1
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
    python3 "$SUMMARY_PY" question-limit --task "$task" --conc 1 --gsm8k-limit "$GSM8K_LIMIT" --mmlu-limit "$MMLU_LIMIT" >/dev/null \
        || { log "ERROR: unsupported task '$task' (supported: mmlu_pro, gsm8k)"; exit 1; }
done
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.md"
k3_set_gcs_root "$GCS_ARG"

SUMMARY_PRINTED=0
# shellcheck disable=SC2317  # run by the EXIT trap
cleanup() {
    log "cleanup"
    bash "$SCRIPT_DIR/cleanup_server.sh"
    k3_dump_server_log_context
    if [ -f "$SUMMARY" ] && [ "$SUMMARY_PRINTED" = "0" ]; then log "===== summary (partial) ====="; cat "$SUMMARY"; fi
    if [ -n "$GCS_ROOT" ] && command -v gsutil >/dev/null 2>&1; then
        log "uploading $RESULTS_DIR to $GCS_ROOT"
        gsutil -q -m cp -r "$RESULTS_DIR" "$GCS_ROOT/" 2>&1 | tail -3 || log "WARNING: GCS upload to $GCS_ROOT failed"
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- boot
log "===== boot: config=$CONFIG_NAME concs=${CONCS[*]} tasks=${TASK_LIST[*]} gsm8k_limit>=$GSM8K_LIMIT mmlu_limit>=$MMLU_LIMIT ctx=$K3_SWEEP_CTX max_num_seqs=$K3_SWEEP_MAX_NUM_SEQS ====="
k3_boot_server "$CONFIG_NAME" || exit 1
MODEL=$(k3_config_json_get model)
save_now "$RESULTS_DIR/config.json"

{
    echo "# K3 accuracy vs eval concurrency (build ${BUILDKITE_BUILD_NUMBER:-local}, config $CONFIG_NAME, ctx $K3_SWEEP_CTX, max-num-seqs $K3_SWEEP_MAX_NUM_SEQS, thinking off, plain chat)"
    echo
    echo "| conc | task | questions | metric | score | stderr | wall s | lm_eval rc / log errors | engine preemptions during point |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

# ---------------------------------------------------------------- sweep
FAIL=0
for c in "${CONCS[@]}"; do
    for task in "${TASK_LIST[@]}"; do
        read -r limit n_q <<< "$(python3 "$SUMMARY_PY" question-limit --task "$task" --conc "$c" --gsm8k-limit "$GSM8K_LIMIT" --mmlu-limit "$MMLU_LIMIT")"
        if [ "$task" = "gsm8k" ]; then
            # gsm8k's default max_gen_toks (256) cuts K3's worked answers short.
            gen_kwargs="{\"chat_template_kwargs\": $TEMPLATE_KWARGS, \"max_gen_toks\": 1024}"
        else
            gen_kwargs="{\"chat_template_kwargs\": $TEMPLATE_KWARGS}"
        fi
        out="$RESULTS_DIR/${task}_c${c}"; mkdir -p "$out"
        log "===== conc=$c task=$task limit=$limit ($n_q questions) ====="
        if ! server_alive; then log "FATAL: server died before conc=$c $task"; FAIL=1; break 2; fi
        T0=$(date +%s)
        pre0=$(k3_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
        lm_eval --tasks "$task" --seed "0,1234,None,1234" --output_path "$out" --log_samples --limit "$limit" \
            --model local-chat-completions \
            --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=$c,max_retries=3,timeout=3600" \
            --apply_chat_template --gen_kwargs "$gen_kwargs" > "$out/lm_eval.log" 2>&1
        rc=$?
        wall=$(( $(date +%s) - T0 ))
        errs=$(grep -c -iE 'error|exception|timed out' "$out/lm_eval.log" || true)
        read -r metric score stderr <<< "$(python3 "$SUMMARY_PY" lm-eval-score --output-dir "$out" --task "$task")"
        pre1=$(k3_engine_counters | sed -E 's/preemptions=([0-9]+).*/\1/')
        preempt=$(( ${pre1:-0} - ${pre0:-0} ))
        log "conc=$c task=$task rc=$rc $metric=$score ±$stderr wall=${wall}s log-errors=$errs engine-preemptions=$preempt"
        if [ "$rc" -ne 0 ]; then
            FAIL=1
            log "ERROR: lm_eval failed for conc=$c task=$task (rc=$rc); last 40 lines of $out/lm_eval.log:"
            tail -40 "$out/lm_eval.log" | cut -c1-300
        fi
        echo "| $c | $task | $n_q | $metric | $score | $stderr | $wall | rc=$rc / $errs | $preempt |" >> "$SUMMARY"
        save_now "$out" "$SUMMARY"
        wait_server_idle || true
    done
done

log "===== summary ====="
cat "$SUMMARY"; SUMMARY_PRINTED=1
server_alive || { log "ERROR: server is dead at the end of the sweep"; FAIL=1; }
[ "$FAIL" = "0" ] || log "sweep finished with failures (see rc column); exiting 1"
exit $FAIL

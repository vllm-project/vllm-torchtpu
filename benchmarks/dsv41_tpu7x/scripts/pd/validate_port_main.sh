#!/usr/bin/env bash
# Validate the main-ported pair (vllm-torchtpu cpk/dsv41-mh-decode-main + vllm cpk/dsv41-vllm-decode-main) on TPU7x:
# wait for 2 free 2x2x1 hosts -> provision 1P1D with --runtime main into the best decode geometry -> PD gates -> N sweeps.
# Usage: validate_port_main.sh <out_dir> [repeats]
# Env: VTT_DIR / VLLM_DIR (checkouts of the ported vllm-torchtpu branch and the v0.33.0-based vLLM branch),
#      EXTRA_ENV="K=V K=V" (extra env for both servers), LIBTPU (override the pinned libtpu), KUBE_CTX, MAX_ATTEMPTS.
set -u
OUT=${1:?out dir}; REP=${2:-3}
CTX=${KUBE_CTX:-gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps}
PORT=${PORT_DIR:-/usr/local/google/home/chaitanyapk/.gemini/jetski/brain/386b2d65-9703-489e-bab9-9f2f69e34bbc/scratch/port}
SUFFIX=pm; P=chaitanyapk-dsv41-pd-dc-$SUFFIX
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"
# EXTRA_ENV: space-separated KEY=VAL list passed to both servers (one --env each).
EXTRA_ENV_ARGS=(); for kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARGS+=(--env "$kv"); done
k() { kubectl --context "$CTX" "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
for attempt in $(seq 1 "${MAX_ATTEMPTS:-12}"); do
  while true; do
    if [ "$(k get pod $P -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ] && \
       [ "$(k get pod chaitanyapk-dsv41-pd-pf-$SUFFIX -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; then
      log "both $SUFFIX pods already Running - resuming on them"; break
    fi
    n_ok=$(k get nodes -l cloud.google.com/gke-tpu-topology=2x2x1 -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .spec.taints[*]}{.key}{","}{end}{"\n"}{end}' 2>/dev/null | awk '$2=="google.com/tpu," {c++} END{print c+0}')
    n_busy=$(k get pods -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u | grep -c gke-tpu)
    free=$(( n_ok - n_busy )); log "healthy 2x2x1 nodes=$n_ok busy=$n_busy free=$free"
    [ "$free" -ge 2 ] && break; sleep 300
  done
  for lk in "$HERE"/../../locks/*-$SUFFIX.lock; do [ -f "$lk" ] && rm -f "$lk"; done
  log "===== attempt $attempt: provisioning 1P1D from the main port (runtime=main, libtpu=${LIBTPU:-pinned 0.0.50}, extra env=${EXTRA_ENV:-none}) ====="
  python3 -u "$HERE/run_pd.py" --ctx "$CTX" --suffix $SUFFIX --tag dsv41_port_main_validate \
    --vtt "${VTT_DIR:-$PORT/vtt-main}" --vllm "${VLLM_DIR:-$PORT/vllm-main}" --runtime main ${LIBTPU:+--libtpu "$LIBTPU"} \
    --prefill-ced 1 --decode-mnbt 768 --enable-dspark 1 --gpu-mem 0.85 \
    --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 "${EXTRA_ENV_ARGS[@]}" \
    --decode-env MAX_NUM_SEQS=128 --decode-env "COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384, 512, 768]" \
    --decode-env VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768 --decode-env ENABLE_PREFIX_CACHING=1 \
    --decode-env PREFIX_CACHE_RETENTION_INTERVAL=1024 \
    --keep-pods --allow-foreign --resume --pending-timeout 3600 2>&1 | tee "$OUT/provision_$attempt.log" \
    | grep -E "HEALTHY|GATE|STAGE3_CHECK|hard_gates|phase=|Error|died|Traceback|PIP\]"
  if grep -q "hard_gates_passed=True" "$OUT/provision_$attempt.log" && grep -q "STAGE3_CHECK_OK 1" "$OUT/provision_$attempt.log"; then
    log "GATES PASSED"; break
  fi
  dc_phase=$(k get pod "$P" -o jsonpath='{.status.phase}' 2>/dev/null); pf_phase=$(k get pod chaitanyapk-dsv41-pd-pf-$SUFFIX -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$dc_phase" != "Running" ] || [ "$pf_phase" != "Running" ]; then
    if grep -q "server failed during boot" "$OUT/provision_$attempt.log"; then
      log "server boot failure recorded in provision_$attempt.log (pods now dc=$dc_phase pf=$pf_phase); stopping for debugging"; exit 2
    fi
    log "pods not Running (dc=${dc_phase:-gone} pf=${pf_phase:-gone}): preempted - retrying"; k delete pod "$P" chaitanyapk-dsv41-pd-pf-$SUFFIX --ignore-not-found >/dev/null 2>&1; continue
  fi
  log "gates not passed; collecting decode log tail for debugging"
  k exec "$P" -- bash -c 'grep -n "Traceback\|Error\|error:" /dev/shm/pd_decode.log | tail -20; tail -60 /dev/shm/pd_decode.log' > "$OUT/decode_fail_$attempt.log" 2>&1
  k exec chaitanyapk-dsv41-pd-pf-$SUFFIX -- bash -c 'grep -n "Traceback\|Error\|error:" /dev/shm/pd_prefill*.log | tail -20' > "$OUT/prefill_fail_$attempt.log" 2>&1
  exit 2
done
k cp "$HERE/x1_step_anatomy.py" "$P:/tmp/x1_step_anatomy.py"
for i in $(seq 1 "$REP"); do
  tag=port_main_k5_mnbt768_rep$i; log "sweep $tag"
  k exec "$P" -- bash -c "rm -f /tmp/$tag.done; nohup setsid bash -c 'python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 4000 --out-len 1024 --concs 256,512 --dp 8 --k 5 --out /tmp/$tag.json > /tmp/$tag.log 2>&1; echo \$? > /tmp/$tag.done' >/dev/null 2>&1 < /dev/null & disown"
  for j in $(seq 1 60); do sleep 30; rc=$(k exec "$P" -- cat /tmp/$tag.done 2>/dev/null); [ -n "$rc" ] && break; done
  k cp "$P:/tmp/$tag.json" "$OUT/$tag.json" >/dev/null 2>&1; k cp "$P:/tmp/$tag.log" "$OUT/$tag.log" >/dev/null 2>&1
  grep "\[X1\]" "$OUT/$tag.log" | cut -c1-200
done
log "done"

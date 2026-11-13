#!/usr/bin/env bash
# Unattended E1 orchestrator for a churning spot pool:
#   loop { wait for >=2 free healthy 2x2x1 nodes -> run_pd --resume (reuses surviving role) ->
#          require HEALTHY + gates -> run_e1a_levels (skips completed levels) } until all levels done.
set -u
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
OUT=../../runs/pd/e1_results; LEVELS=${LEVELS:-128,192,256,384,512}; TR=${TR:-40}; TAG=${TAG:-e1a_tr40}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
all_done() { for n in ${LEVELS//,/ }; do f="$OUT/${TAG}_n${n}.json"; [ -s "$f" ] || return 1; python3 -c "import json,sys;s=json.load(open(sys.argv[1]))[-1];sys.exit(0 if s[\"failed_requests\"]<=0.05*max(1,s[\"total_requests\"]) else 1)" "$f" || return 1; done; return 0; }
free_nodes() {
  n_ok=$(kubectl --context "$CTX" get nodes -l cloud.google.com/gke-tpu-topology=2x2x1 \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .spec.taints[*]}{.key}{","}{end}{"\n"}{end}' 2>/dev/null \
    | awk '$2=="google.com/tpu," {c++} END{print c+0}')
  n_busy=$(kubectl --context "$CTX" get pods -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{" "}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | grep -v "dsv41-pd-..-e1 " | awk 'NF==2{print $2}' | sort -u | wc -l)
  ours=$(kubectl --context "$CTX" get pods -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -c "dsv41-pd-..-e1$")
  echo $(( n_ok - n_busy - ours )) "$ours"
}
attempt=0
until all_done; do
  attempt=$((attempt+1))
  # wait until (free nodes + our surviving pods) >= 2
  while true; do
    read -r free ours < <(free_nodes)
    need=$(( 2 - ours )); [ "$need" -lt 0 ] && need=0
    log "capacity: free=$free ours_running=$ours need=$need"
    [ "$free" -ge "$need" ] && break
    sleep 300
  done
  # release a stale lock if the previous launcher is dead
  for lk in ../../locks/bodaborg-tpu7x-spot-sps__chaitanyapk-dsv41-pd-*-e1.lock; do
    [ -f "$lk" ] || continue
    pid=$(python3 -c "import json,sys;print(json.load(open('$lk'))['pid'])" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then log "removing stale lock $lk (pid $pid dead)"; rm -f "$lk"; fi
  done
  log "===== provisioning attempt $attempt ====="
  python3 -u run_pd.py --ctx "$CTX" --suffix e1 --tag dsv41_e1_dside_saturation \
    --prefill-ced 1 --decode-mnbt 384 --gpu-mem 0.85 \
    --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 \
    --keep-pods --allow-foreign --resume --pending-timeout 3600 \
    2>&1 | tee "../../runs/logs/e1_prov_${attempt}.log" | grep -v "^Error from server\|^error: cannot exec\|^  \[" 
  if grep -q "\[prefill\] HEALTHY\|\[prefill\] resume: phase=Running server_alive=True" "../../runs/logs/e1_prov_${attempt}.log" \
     && grep -q "STAGE3_CHECK_OK 1\|hard_gates_passed=True" "../../runs/logs/e1_prov_${attempt}.log"; then
    log "1P1D ready; running E1 levels"
    bash ./run_e1a_levels.sh "$OUT" "$LEVELS" "$TR" "$TAG" 2>&1 | tee -a ../../runs/logs/e1a_levels.log
  else
    log "provisioning attempt $attempt did not reach healthy+gates; retry in 120s"; sleep 120
  fi
done
log "ALL E1 LEVELS DONE -> $OUT"

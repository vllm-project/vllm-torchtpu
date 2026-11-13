#!/usr/bin/env bash
# Driver: overnight decode-step ablation loop on the live decode pod (4 chips, DP8/EP8).
# For each config: (optionally) relaunch the server with env overrides, wait for health, run
# x1_step_anatomy at the given contexts/concurrencies, pull JSON+log to runs/pd/x1_results/.
set -u
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
P=chaitanyapk-dsv41-pd-dc-e1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../../runs/pd/x1_results; mkdir -p "$OUT"
k() { kubectl --context "$CTX" "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
pod_ok() { [ "$(k get pod "$P" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; }
healthy() { k exec "$P" -- curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:9400/health 2>/dev/null | grep -q 200; }

wait_healthy() {  # up to 45 min
  for i in $(seq 1 90); do
    pod_ok || { log "pod gone"; return 2; }
    healthy && return 0
    sleep 30
  done
  log "server not healthy after 45 min"; return 1
}

run_sweep() {  # tag ctx concs k
  local tag=$1 ctx=$2 concs=$3 kk=$4
  if [ -s "$OUT/x1_${tag}.json" ]; then log "skip $tag (done)"; return 0; fi
  log "sweep $tag ctx=$ctx concs=$concs k=$kk"
  k exec "$P" -- bash -c "nohup setsid python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx $ctx --out-len 512 --concs $concs --k $kk --out /tmp/x1_${tag}.json > /tmp/x1_${tag}.log 2>&1 < /dev/null & disown" || return 2
  for i in $(seq 1 90); do
    sleep 30
    pod_ok || { log "pod gone during $tag"; return 2; }
    if ! k exec "$P" -- pgrep -f "x1_step_anatomy.py.*--out /tmp/x1_${tag}.json" >/dev/null 2>&1; then break; fi
  done
  k cp "$P:/tmp/x1_${tag}.json" "$OUT/x1_${tag}.json" >/dev/null 2>&1
  k cp "$P:/tmp/x1_${tag}.log" "$OUT/x1_${tag}.log" >/dev/null 2>&1
  grep "\[X1\]" "$OUT/x1_${tag}.log" | cut -c1-220
}

relaunch() {  # env overrides...
  log "relaunch with: $*"
  k cp "$HERE/x1_relaunch.sh" "$P:/tmp/x1_relaunch.sh" >/dev/null 2>&1 || return 2
  k exec "$P" -- bash /tmp/x1_relaunch.sh "$@" | tail -4
  wait_healthy
}

ensure_pod() {
  pod_ok && return 0
  log "decode pod missing — attempting re-provision via run_pd --resume (decode role only)"
  # run_pd recreates missing roles; prefill role may also be recreated if a node is free.
  python3 -u "$HERE/run_pd.py" --ctx "$CTX" --suffix e1 --tag dsv41_x1_decode --prefill-ced 1 --decode-mnbt 384 --gpu-mem 0.85 \
    --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 --env ENABLE_PREFIX_CACHING=1 --env PREFIX_CACHE_RETENTION_INTERVAL=1024 \
    --keep-pods --allow-foreign --resume --pending-timeout 7200 2>&1 | grep -v "^Error from server\|^error: cannot exec\|^  \[" | tail -5
  # stage x1 tooling again
  k cp "$HERE/pd_gates.py" "$P:/tmp/pd_gates.py"; k cp "$HERE/x1_step_anatomy.py" "$P:/tmp/x1_step_anatomy.py"
  pod_ok
}

k cp "$HERE/x1_step_anatomy.py" "$P:/tmp/x1_step_anatomy.py" >/dev/null 2>&1

# ---- Config matrix (each block = one server config) ----
# 0. current server (K=5, prefix cache on): context sensitivity
ensure_pod && wait_healthy && {
  run_sweep k5_ctx2k    2000   64,128,256,512 5
  run_sweep k5_ctx100k  100000 64,128,256,512 5
}
# 1. DSpark off: pure target-model decode step (floor/slope without draft+verify)
ensure_pod && relaunch ENABLE_DSPARK=0 ASYNC_SCHEDULING=1 MAX_NUM_SEQS=64 "COMPILE_SIZES=[8, 16, 32, 64, 384]" && {
  run_sweep k0_ctx20k   20000  64,128,256,512 0
}
# 2. DSpark K=3 (4 tok/seq): more seqs per step budget, lower tau
ensure_pod && relaunch ENABLE_DSPARK=1 DSPARK_TOKENS=3 MAX_NUM_SEQS=96 "COMPILE_SIZES=[32, 64, 128, 256, 384]" && {
  run_sweep k3_ctx20k   20000  64,128,256,512,768 3
}
# 3. DSpark K=7 (8 tok/seq): higher tau potential, fewer seqs
ensure_pod && relaunch ENABLE_DSPARK=1 DSPARK_TOKENS=7 MAX_NUM_SEQS=48 "COMPILE_SIZES=[64, 128, 256, 384]" && {
  run_sweep k7_ctx20k   20000  64,128,256,384 7
}
# 4. back to K=5 baseline with async scheduling OFF (host/scheduler overhead share of the floor)
ensure_pod && relaunch ENABLE_DSPARK=1 DSPARK_TOKENS=5 ASYNC_SCHEDULING=0 MAX_NUM_SEQS=64 "COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384]" && {
  run_sweep k5_noasync_ctx20k 20000 64,128,256,512 5
}
# 5. restore K=5 async baseline and re-measure (drift check)
ensure_pod && relaunch ENABLE_DSPARK=1 DSPARK_TOKENS=5 ASYNC_SCHEDULING=1 MAX_NUM_SEQS=64 "COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384]" && {
  run_sweep k5_ctx20k_repeat 20000 64,128,256,512 5
}
log "X1 ablation loop finished"

#!/usr/bin/env bash
# E1-a driver: run the D-side saturation sweep one n-level per harness invocation so each level's
# results are pulled locally before the next starts (spot pool churns ~hourly). Re-runnable: skips
# levels whose JSON is already present locally. The harness runs DETACHED inside the pod (nohup setsid)
# and is polled, so a dropped kubectl exec cannot kill it.
#
# Usage: run_e1a_levels.sh <local_out_dir> [levels] [trunks_above_n16] [tag]
set -u
OUT=${1:?local out dir}; LEVELS=${2:-128,192,256,384,512}; TR=${3:-40}; TAG=${4:-e1a_tr40}; HIT=${HIT:-0.95}; DUR=${DUR:-180}
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
PF=chaitanyapk-dsv41-pd-pf-e1; DC=chaitanyapk-dsv41-pd-dc-e1
HERE=$(cd "$(dirname "$0")" && pwd)
DATASET=/usr/local/google/home/chaitanyapk/.gemini/jetski/brain/d089016f-e01c-4fb7-a19a-1cb94b419b06/scratch/requests_mocked.jsonl.xz
mkdir -p "$OUT"
k() { kubectl --context "$CTX" "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

pods_healthy() {
  k get pod "$PF" >/dev/null 2>&1 && k get pod "$DC" >/dev/null 2>&1 || return 1
  DC_IP=$(k get pod "$DC" -o jsonpath='{.status.podIP}' 2>/dev/null); [ -n "$DC_IP" ] || return 1
  ok=$(k exec "$PF" -- bash -c "for p in 8400 8401 8402 8403; do curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:\$p/health; echo; done; curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/healthcheck; curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://$DC_IP:9400/health" 2>/dev/null | grep -c 200)
  [ "${ok:-0}" -ge 6 ]
}

stage() {
  k cp "$HERE/bench_empirical_isl_dsv41.py" "$PF:/tmp/bench_empirical_isl_dsv41.py"
  k cp "$HERE/collect_pd_bench_logs.sh" "$PF:/tmp/collect_pd_bench_logs.sh"
  k exec "$PF" -- bash -c 'test -s /tmp/requests_mocked.jsonl.xz' 2>/dev/null || k cp "$DATASET" "$PF:/tmp/requests_mocked.jsonl.xz"
  k exec "$PF" -- bash -c 'mkdir -p /data && ln -sf /tmp/requests_mocked.jsonl.xz /data/requests_mocked.jsonl.xz && md5sum /tmp/requests_mocked.jsonl.xz'
}

pull() {
  local n=$1
  for f in "${TAG}_n${n}.json" "${TAG}_n${n}_n${n}_requests.jsonl" "${TAG}_n${n}.log"; do
    k cp "$PF:/tmp/$f" "$OUT/$f" >/dev/null 2>&1 && log "pulled $f"
  done
  k exec "$PF" -- bash -c 'tail -n 30000 /dev/shm/pd_proxy.log 2>/dev/null' > "$OUT/pd_proxy_tail_n${n}.log" 2>/dev/null
}

run_level() {
  local n=$1 warm=$2
  k exec "$PF" -- bash -c "cat > /tmp/run_${TAG}_n${n}.sh <<EOF
#!/usr/bin/env bash
cd /tmp
rm -f /tmp/${TAG}_n${n}.json /tmp/${TAG}_n${n}.done
python3 -u /tmp/bench_empirical_isl_dsv41.py \\
  --base-url http://127.0.0.1:8000 --prefill-url http://127.0.0.1:8400 --decode-url http://${DC_IP}:9400 \\
  --dataset /data/requests_mocked.jsonl.xz --model deepseek-ai/DeepSeek-V4.1-Flash \\
  --sessions ${n} --duration-per-n ${DUR} --cache-hit-rate ${HIT} --max-tokens 0 \\
  --dp-size 8 --trunks-above-n16 ${TR} --warmup-pass-s ${warm} --num-chips 8 --timeout 900 \\
  --output-json /tmp/${TAG}_n${n}.json > /tmp/${TAG}_n${n}.log 2>&1
echo \\\$? > /tmp/${TAG}_n${n}.done
EOF
chmod +x /tmp/run_${TAG}_n${n}.sh; nohup setsid bash /tmp/run_${TAG}_n${n}.sh >/dev/null 2>&1 < /dev/null & disown; sleep 2; pgrep -f 'sessions ${n} ' >/dev/null && echo LAUNCHED || echo LAUNCH_FAILED"
  # poll: done-file, or pod gone
  local t0=$SECONDS
  while true; do
    sleep 60
    if [ "$(k get pod "$PF" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Running" ] || [ "$(k get pod "$DC" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Running" ]; then log "a pod died during n=$n"; return 2; fi
    rc=$(k exec "$PF" -- bash -c "cat /tmp/${TAG}_n${n}.done 2>/dev/null" 2>/dev/null)
    if [ -n "$rc" ]; then log "n=$n harness finished rc=$rc after $((SECONDS - t0))s"; return 0; fi
    pull "$n" >/dev/null 2>&1   # keep partial log locally
    [ $(( (SECONDS - t0) % 300 )) -lt 60 ] && log "n=$n running ($((SECONDS - t0))s): $(k exec "$PF" -- bash -c "grep -c 'RESULTS FOR' /tmp/${TAG}_n${n}.log 2>/dev/null; tail -c 300 /tmp/${TAG}_n${n}.log 2>/dev/null | tr '\n' ' '" 2>/dev/null | cut -c1-240)"
    [ $((SECONDS - t0)) -gt 3600 ] && { log "n=$n exceeded 60 min; abandoning"; return 3; }
  done
}

IFS=',' read -ra NS <<< "$LEVELS"
first=1
for n in "${NS[@]}"; do
  if [ -s "$OUT/${TAG}_n${n}.json" ]; then
    # accept only if the run was not cut by a preemption (failed <= 5% of requests)
    if python3 - "$OUT/${TAG}_n${n}.json" <<'PY'
import json,sys; s=json.load(open(sys.argv[1]))[-1]
sys.exit(0 if s["failed_requests"] <= 0.05*max(1,s["total_requests"]) else 1)
PY
    then log "n=$n already done locally, skipping"; continue
    else log "n=$n local result has >5% failures (preempted run); quarantining and re-running"; mkdir -p "$OUT/preempted"; mv "$OUT/${TAG}_n${n}"* "$OUT/preempted/" 2>/dev/null; fi
  fi
  down=0
  while ! pods_healthy; do
    down=$((down+1)); log "waiting for healthy 1P1D (pods/servers down) ... [$down]"
    if [ "$(k get pod "$PF" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Running" ] || [ "$(k get pod "$DC" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Running" ]; then
      [ "$down" -ge 2 ] && { log "a pod is gone; returning to orchestrator for re-provisioning"; exit 2; }
    fi
    sleep 120
  done
  log "all endpoints healthy (decode $DC_IP)"; stage
  warm=0; [ "$first" = 1 ] && warm=8; first=0
  log "===== E1 level n=$n (trunks_above_n16=$TR) ====="
  run_level "$n" "$warm"; st=$?
  pull "$n"
  if [ ! -s "$OUT/${TAG}_n${n}.json" ]; then
    log "n=$n produced no JSON (status $st); will retry once pods are healthy"
    sleep 60
    exec "$HERE/run_e1a_levels.sh" "$OUT" "$LEVELS" "$TR" "$TAG"
  fi
  grep "RESULTS FOR\|Steady-State\|Client Decode\|Latency  \|Per-Chip\|ERRORS" "$OUT/${TAG}_n${n}.log" | cut -c1-240
done
log "E1 sweep complete: $OUT"

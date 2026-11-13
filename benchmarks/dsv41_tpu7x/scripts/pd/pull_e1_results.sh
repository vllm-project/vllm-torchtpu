#!/usr/bin/env bash
# Pull E1 benchmark artifacts from the prefill pod to the local run dir every INTERVAL seconds,
# so results survive spot eviction / DiskPressure pod loss. Usage: pull_e1_results.sh <local_dir> [pod] [interval]
set -u
OUT=${1:?local dir}; POD=${2:-chaitanyapk-dsv41-pd-pf-e1}; INTERVAL=${3:-180}
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
mkdir -p "$OUT"
while true; do
  ts=$(date +%H:%M:%S)
  if kubectl --context "$CTX" get pod "$POD" >/dev/null 2>&1; then
    for f in $(kubectl --context "$CTX" exec "$POD" -- bash -c 'ls /tmp/e1*.log /tmp/e1*.json /tmp/e1*_requests.jsonl 2>/dev/null' 2>/dev/null); do
      kubectl --context "$CTX" cp "$POD:$f" "$OUT/$(basename "$f")" >/dev/null 2>&1 && echo "[$ts] pulled $(basename "$f")"
    done
    # snapshot the proxy log tail too (TTFT stage split + transport failures)
    kubectl --context "$CTX" exec "$POD" -- bash -c 'tail -n 20000 /dev/shm/pd_proxy.log 2>/dev/null' > "$OUT/pd_proxy_tail.log" 2>/dev/null
  else
    echo "[$ts] pod $POD not found"
  fi
  sleep "$INTERVAL"
done

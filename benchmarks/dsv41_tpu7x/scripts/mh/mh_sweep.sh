#!/usr/bin/env bash
# Run an X1 sweep detached inside the mh head pod and poll until done (robust to kubectl exec drops).
# Usage: mh_sweep.sh <tag> <k> <concs> <ctx> [out_len]
set -u
tag=$1; k=$2; concs=$3; ctx=$4; out_len=${5:-256}
CTX=${KUBE_CTX:-gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps}
SUFFIX=${MH_SUFFIX:-mh1}
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$HERE/../../runs/pd/x1_results
k() { kubectl --context "$CTX" "$@"; }
# Head pod of the Indexed Job (completion index 0); override with P0=<pod> if needed.
P0=${P0:-$(k get pods -l dsv41-job=chaitanyapk-dsv41-mh-$SUFFIX -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -- "-$SUFFIX-0-" | head -1)}
[ -n "$P0" ] || { echo "no head pod found for job chaitanyapk-dsv41-mh-$SUFFIX (set P0=...)"; exit 1; }
k cp "$HERE/../pd/x1_step_anatomy.py" "$P0:/tmp/x1_step_anatomy.py" >/dev/null 2>&1
k exec "$P0" -- bash -c "rm -f /tmp/x1_$tag.done; nohup setsid bash -c 'python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx $ctx --out-len $out_len --concs $concs --dp 16 --k $k --out /tmp/x1_$tag.json > /tmp/x1_$tag.log 2>&1; echo \$? > /tmp/x1_$tag.done' >/dev/null 2>&1 < /dev/null & disown; echo launched"
for i in $(seq 1 120); do
  sleep 30
  rc=$(k exec "$P0" -- bash -c "cat /tmp/x1_$tag.done 2>/dev/null" 2>/dev/null)
  [ -n "$rc" ] && break
done
k exec "$P0" -- cat /tmp/x1_$tag.log 2>/dev/null | grep "\[X1\]" | cut -c1-240
k cp "$P0:/tmp/x1_$tag.json" "$OUT/x1_$tag.json" >/dev/null 2>&1; k cp "$P0:/tmp/x1_$tag.log" "$OUT/x1_$tag.log" >/dev/null 2>&1

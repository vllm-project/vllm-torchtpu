#!/usr/bin/env bash
# Retry acquiring a 2-host tpu7x 2x2x2 spot slice (GCE_STOCKOUT retry loop), then launch the
# multi-host DP16/EP16 decode server. Tries alternately: resize existing pool, or recreate it.
set -u
PROJECT=cloud-tpu-multipod-dev; REGION=us-central1; ZONE=us-central1-c
CLUSTER=bodaborg-tpu7x-spot-sps; POOL=cpk-spot-2x2x2; WP=cpk-v7x-2x2x2-wp
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
HERE=$(cd "$(dirname "$0")" && pwd)
log() { echo "[$(date +%H:%M:%S)] $*"; }
ready_nodes() {
  kubectl --context "$CTX" get nodes -l cloud.google.com/gke-nodepool=$POOL -o jsonpath='{range .items[*]}{.metadata.name} {range .status.conditions[?(@.type=="Ready")]}{.status}{end}{"\n"}{end}' 2>/dev/null | awk '$2=="True"' | wc -l
}
for attempt in $(seq 1 60); do
  n=$(ready_nodes)
  if [ "${n:-0}" -ge 2 ]; then
    log "2x2x2 slice ready ($n nodes) — launching multi-host decode"
    cd "$HERE" && python3 -u run_mh_decode.py --ctx "$CTX" --topology 2x2x2 --hosts 2 --suffix mh1 2>&1 | tee -a ../../runs/logs/mh_decode_mh1.log
    rc=$?
    [ "$rc" -eq 0 ] && bash "$HERE/mh_ici_check.sh" 2>&1 | tee -a ../../runs/logs/mh_ici_check.log
    exit $rc
  fi
  st=$(timeout 60 gcloud container node-pools describe $POOL --cluster $CLUSTER --region $REGION --project $PROJECT --format='value(status)' 2>/dev/null)
  log "attempt $attempt: pool status=${st:-missing} ready_nodes=${n:-0}"
  if [ -z "$st" ]; then
    timeout 900 gcloud container node-pools create $POOL --cluster $CLUSTER --region $REGION --project $PROJECT --node-locations $ZONE \
      --machine-type tpu7x-standard-4t --tpu-topology 2x2x2 --placement-policy $WP --num-nodes 2 --spot --disk-size 200 --enable-private-nodes \
      --additional-node-network network=bodaborg-tpu7x-spot-sps-privatenetwork-2-us-central1-ai1a,subnetwork=bodaborg-tpu7x-spot-sps-privatesubnet-2-us-central1-ai1a 2>&1 | tail -1
  elif [ "$st" = "ERROR" ] || [ "$st" = "RUNNING" ]; then
    # kick the MIG: resize to 0 then 2 (atomic resize retried by GCE)
    timeout 600 gcloud container clusters resize $CLUSTER --node-pool $POOL --region $REGION --project $PROJECT --num-nodes 0 --quiet 2>&1 | tail -1
    timeout 900 gcloud container clusters resize $CLUSTER --node-pool $POOL --region $REGION --project $PROJECT --num-nodes 2 --quiet 2>&1 | tail -1
  fi
  sleep 600
done
log "gave up acquiring 2x2x2 slice"

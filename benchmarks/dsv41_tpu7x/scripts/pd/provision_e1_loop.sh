#!/usr/bin/env bash
# Retry loop for E1 1P1D provisioning on a flaky spot pool. Stops on first success
# (run_pd exits 0 with both roles HEALTHY) or after MAX_ATTEMPTS.
set -u
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[$(date +%H:%M:%S)] ===== provisioning attempt $i/$MAX_ATTEMPTS ====="
  # Wait until at least 2 schedulable 2x2x1 nodes exist (taint only google.com/tpu) before spending a run dir.
  while true; do
    n_ok=$(kubectl --context "$CTX" get nodes -l cloud.google.com/gke-tpu-topology=2x2x1 \
      -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .spec.taints[*]}{.key}{","}{end}{"\n"}{end}' 2>/dev/null \
      | awk '$2=="google.com/tpu," {c++} END{print c+0}')
    n_busy=$(kubectl --context "$CTX" get pods -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u | wc -l)
    free=$(( n_ok - n_busy ))
    echo "[$(date +%H:%M:%S)] healthy 2x2x1 nodes=$n_ok busy=$n_busy free=$free"
    [ "$free" -ge 2 ] && break
    sleep 300
  done
  python3 -u run_pd.py --ctx "$CTX" --suffix e1 --tag dsv41_e1_dside_saturation \
    --prefill-ced 1 --decode-mnbt 384 --gpu-mem 0.85 \
    --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 \
    --keep-pods --allow-foreign --resume 2>&1 | tee "../../runs/logs/run_pd_e1_loop_${i}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && grep -q "\[decode\] HEALTHY" "../../runs/logs/run_pd_e1_loop_${i}.log" \
     && grep -q "\[prefill\] HEALTHY" "../../runs/logs/run_pd_e1_loop_${i}.log"; then
    echo "[$(date +%H:%M:%S)] PROVISION_OK attempt $i"; exit 0
  fi
  echo "[$(date +%H:%M:%S)] attempt $i failed (rc=$rc); retrying after 120s"
  sleep 120
done
echo "PROVISION_FAILED after $MAX_ATTEMPTS attempts"; exit 1

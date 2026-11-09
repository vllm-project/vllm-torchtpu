#!/usr/bin/env bash
# Validate the best single-host decode point (DP8/EP8, K=5, MNBT 768, 128 seqs/rank):
# provision 1P1D straight into that geometry, pass the PD gates, then run the x1 sweep twice.
# Usage: validate_best_decode.sh <out_dir> [repeats]
set -u
OUT=${1:?out dir}; REP=${2:-2}
CTX=${KUBE_CTX:-gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps}
P=chaitanyapk-dsv41-pd-dc-e1
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"
k() { kubectl --context "$CTX" "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
for lk in "$HERE"/../../locks/*e1.lock; do [ -f "$lk" ] && rm -f "$lk"; done
log "provisioning 1P1D (decode: K=5 MNBT 768 seqs 128 CSA 768 prefix-cache on)"
python3 -u "$HERE/run_pd.py" --ctx "$CTX" --suffix e1 --tag dsv41_validate_best_k5_mnbt768 \
  --prefill-ced 1 --decode-mnbt 768 --enable-dspark 1 --gpu-mem 0.85 \
  --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 \
  --decode-env MAX_NUM_SEQS=128 --decode-env "COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384, 512, 768]" \
  --decode-env VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768 --decode-env ENABLE_PREFIX_CACHING=1 \
  --decode-env PREFIX_CACHE_RETENTION_INTERVAL=1024 \
  --keep-pods --allow-foreign --resume --pending-timeout 3600 2>&1 | tee "$OUT/provision.log" \
  | grep -E "HEALTHY|GATE|STAGE3_CHECK|hard_gates|phase=|Error|died"
grep -q "hard_gates_passed=True" "$OUT/provision.log" || { log "gates did not pass"; exit 1; }
k cp "$HERE/x1_step_anatomy.py" "$P:/tmp/x1_step_anatomy.py"
for i in $(seq 1 "$REP"); do
  tag=validate_k5_mnbt768_rep$i
  log "sweep $tag"
  k exec "$P" -- bash -c "rm -f /tmp/$tag.done; nohup setsid bash -c 'python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 4000 --out-len 1024 --concs 256,512 --dp 8 --k 5 --out /tmp/$tag.json > /tmp/$tag.log 2>&1; echo \$? > /tmp/$tag.done' >/dev/null 2>&1 < /dev/null & disown"
  for j in $(seq 1 60); do sleep 30; rc=$(k exec "$P" -- cat /tmp/$tag.done 2>/dev/null); [ -n "$rc" ] && break; done
  k cp "$P:/tmp/$tag.json" "$OUT/$tag.json" >/dev/null 2>&1; k cp "$P:/tmp/$tag.log" "$OUT/$tag.log" >/dev/null 2>&1
  grep "\[X1\]" "$OUT/$tag.log" | cut -c1-200
done
log "done"

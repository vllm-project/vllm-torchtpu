#!/usr/bin/env bash
# csa2 代码的 2P1D 验证：prep → 部署（c512-csa2.env）→ 冒烟 → PD logprob 检查 → AgentX c512 3600s（在 dsv41-bench pod 里跑）
set -uo pipefail
cd "$(dirname "$0")"; source c512-csa2.env; source common.sh
bash prep_pod.sh || { echo "PREP FAILED"; exit 1; }
SKIP_PREP=1 bash deploy.sh || { echo "DEPLOY FAILED"; exit 1; }
P0_IP=$(pod_ip "$(role_pod p0)"); D_IP=$(pod_ip "$(role_pod d)")
echo "== smoke $(date +%T)"
curl -s -m 120 http://$P0_IP:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":12,"temperature":0}' | head -c 300; echo
echo "== logprob check $(date +%T)"
LP=$LOG_DIR/gke-logprob-$(date +%m%d-%H%M).log
/zzlfs/temu/.venv-epic/bin/python -I "$REPO/scripts/dsv41_pd/pd_logprob_check.py" --proxy http://$P0_IP:8000 --direct http://$D_IP:8200 \
  --tokens 3000 20000 100000 --offset 1000000 > "$LP" 2>&1; tail -8 "$LP" | cut -c1-200
echo "== agentx $(date +%T)"
TAG=${TAG:-gke2p1d-csa2-whale16k} bash bench_agentx.sh

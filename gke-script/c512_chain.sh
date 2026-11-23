#!/usr/bin/env bash
# 部署 c512 配置 → 冒烟测试 → AgentX c512
set -uo pipefail
cd /zzlfs/temu/vllm-torchtpu-1p1d/gke-script
source c512.env
SKIP_PREP=1 bash deploy.sh || { echo "DEPLOY FAILED"; exit 1; }
source common.sh
P0_IP=$(pod_ip "$(role_pod p0)")
echo "== smoke $(date +%T)"
for i in 1 2 3 4; do
  curl -s -m 120 http://$P0_IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v41\",\"messages\":[{\"role\":\"user\",\"content\":\"Session $i: what is the capital of France? Answer in one word.\"}],\"max_tokens\":16,\"temperature\":0}" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print("smoke", repr(r["choices"][0]["message"]["content"]))' || { echo "SMOKE FAILED"; exit 1; }
done
grep -E "PERF PROXY req_received" $LOG_DIR/gke-proxy-$(cat $LOG_DIR/gke-last-T-proxy).log | tail -4 | grep -oE "rank=[0-9]+ p_inst=[0-9]+"
echo "== agentx $(date +%T)"
TAG=gke2p1d bash run_agentx.sh

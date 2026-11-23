#!/usr/bin/env bash
# 服务端不动，只用 c512 配置重启 proxy，冒烟测试后跑 AgentX c512。
set -uo pipefail
cd "$(dirname "$0")"; source c512.env; source common.sh
bash wait_ready.sh p0 p1 d || exit 1
bash start_proxy.sh || exit 1
P0_IP=$(pod_ip "$(role_pod p0)")
for i in 1 2 3 4; do
  curl -s -m 120 http://$P0_IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v41\",\"messages\":[{\"role\":\"user\",\"content\":\"Session $i: capital of France?\"}],\"max_tokens\":8,\"temperature\":0}" \
    | python3 -c 'import json,sys; print("smoke", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' || { echo "SMOKE FAILED"; exit 1; }
done
grep "req_received" $LOG_DIR/gke-proxy-$(cat $LOG_DIR/gke-last-T-proxy).log | grep -oE "rank=[0-9]+ p_inst=[0-9]+"
TAG=${TAG:-gke2p1d-interleave} bash run_agentx.sh

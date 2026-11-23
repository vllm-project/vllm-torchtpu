#!/usr/bin/env bash
# （重新）在 P0 pod 里启动 2P1D proxy（:8000），PROXY_* 环境变量原样传进去。
#   bash start_proxy.sh                      # 例：PROXY_SESSION_AFFINITY_DP=8 bash start_proxy.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
P0=$(role_pod p0); P0_IP=$(pod_ip "$P0"); P1_IP=$(pod_ip "$(role_pod p1)"); D_IP=$(pod_ip "$(role_pod d)")
T=${T:-$(date +%m%d-%H%M)}
kx "$P0" "bash $GKE_DIR/pod_stop.sh proxy"
plog=$LOG_DIR/gke-proxy-$T.log
pass=$(env | grep -E '^PROXY_' | sed 's/^/export /; s/=/="/; s/$/";/' | tr '\n' ' ')
# 用 ; 不用 &&（见 deploy.sh）
kx "$P0" "cd /tmp; umask 0000; $pass setsid nohup bash $GKE_DIR/pod_proxy.sh $P0_IP $P1_IP $D_IP > $plog 2>&1 < /dev/null &"
echo "$T" > "$LOG_DIR/gke-last-T-proxy"
for _ in $(seq 60); do
  [[ "$(kx "$P0" "curl -s -m 3 -o /dev/null -w %{http_code} localhost:8000/healthcheck" 2>/dev/null)" == 200 ]] && { echo "proxy READY http://$P0_IP:8000  log=$plog"; exit 0; }
  sleep 2
done
echo "proxy not healthy, see $plog"; exit 1

#!/usr/bin/env bash
# 在 dsv41-bench pod 里后台跑 run_agentx.sh（开发机上调用），等它结束并打印汇总。
#   source c512-csa2.env && TAG=gke2p1d-csa2 bash bench_agentx.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
BENCH_POD=${BENCH_POD:-dsv41-bench}
P0_IP=$(pod_ip "$(role_pod p0)"); P1_IP=$(pod_ip "$(role_pod p1)"); D_IP=$(pod_ip "$(role_pod d)")
T=$(date +%m%d-%H%M); BLOG=$LOG_DIR/gke-bench-$T.log
pass="P0_IP=$P0_IP P1_IP=$P1_IP D_IP=$D_IP CONC=${CONC:-512} DURATION=${DURATION:-3600} TAG=${TAG:-gke2p1d}"
pass+=" P_ENV=$(printf %q "${P_ENV:-}") D_ENV=$(printf %q "${D_ENV:-}") $(env | grep -E '^PROXY_' | tr '\n' ' ')"
kubectl -n "$NS" exec "$BENCH_POD" -- bash -c "cd /tmp; umask 0000; env $pass setsid nohup bash $GKE_DIR/run_agentx.sh > $BLOG 2>&1 < /dev/null &"
echo "bench started in $BENCH_POD, log=$BLOG"
until grep -q "aiperf rc=" "$BLOG" 2>/dev/null; do
  sleep 60
  kubectl -n "$NS" exec "$BENCH_POD" -- bash -c "pgrep -f '[r]un_agentx.sh' >/dev/null" 2>/dev/null || { grep -q "aiperf rc=" "$BLOG" || { echo "bench exited without result"; tail -5 "$BLOG"; exit 1; }; }
done
cat "$BLOG"

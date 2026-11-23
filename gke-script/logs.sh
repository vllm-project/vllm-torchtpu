#!/usr/bin/env bash
# 看某个角色最新一次部署的日志（共享盘上）。
#   bash logs.sh p0|p1|d|proxy [-f]
source "$(dirname "$0")/common.sh"
r=${1:?usage: logs.sh p0|p1|d|proxy [-f]}
if [[ $r == proxy ]]; then T=$(cat "$LOG_DIR/gke-last-T-proxy"); else T=$(cat "$LOG_DIR/gke-last-T-$r"); fi
log=$LOG_DIR/gke-$r-$T.log
echo "== $log"
if [[ "${2:-}" == -f ]]; then tail -f "$log"; else tail -n "${N:-50}" "$log"; fi

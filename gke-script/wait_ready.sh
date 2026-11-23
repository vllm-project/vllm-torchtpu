#!/usr/bin/env bash
# 等各角色 /health 返回 200，或日志出现致命错误 / 进程退出。打印 READY / ERROR / TIMEOUT。
#   bash wait_ready.sh [p0 p1 d]      # 日志按 $LOG_DIR/gke-last-T-<role> 找
set -uo pipefail
source "$(dirname "$0")/common.sh"
ROLES=$(roles_from_args "$@")
FATAL="EngineDeadError|Engine core initialization failed|PjRtClient is not initialized|ValueError:|ManifestError|NameError|requires Linux with glibc|Traceback \(most recent call last\)"
declare -A POD LOG
for r in $ROLES; do POD[$r]=$(role_pod "$r"); LOG[$r]=$LOG_DIR/gke-$r-$(cat "$LOG_DIR/gke-last-T-$r").log; done
for i in $(seq ${WAIT_ITERS:-400}); do
  ok=1; line=""
  for r in $ROLES; do
    code=$(kx "${POD[$r]}" "curl -s -m 5 -o /dev/null -w %{http_code} localhost:$(role_port "$r")/health" 2>/dev/null)
    line+="$r=$code "
    [[ "$code" == 200 ]] || ok=0
    if grep -qE "$FATAL" "${LOG[$r]}" 2>/dev/null; then
      echo "ERROR $(date +%T) $r log=${LOG[$r]}"
      grep -hE "Error:|Error\b|initialization failed" "${LOG[$r]}" | grep -v 404 | head -8 | cut -c1-300
      exit 1
    fi
  done
  if ((ok)); then echo "READY $(date +%T) $line"; exit 0; fi
  (( i % 6 == 1 )) && echo "$(date +%T) waiting: $line"
  sleep 10
done
echo "TIMEOUT"; exit 2

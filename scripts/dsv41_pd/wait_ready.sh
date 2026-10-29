#!/usr/bin/env bash
# 等 P（:8100）和 D（:8200）都健康，或任一边日志出现致命错误 / 进程退出。打印 READY 或 ERROR（附错误行）。
#   bash wait_ready.sh [T]     # T 默认取 /tmp/epic_T（restart_epic.sh 写入）
T=${1:-$(cat /tmp/epic_T)}
P=/zzlfs/temu/logs/prefill-epic-$T.log
D=/zzlfs/temu/logs/decode-epic-$T.log
FATAL="EngineDeadError|Engine core initialization failed|PjRtClient is not initialized|ValueError:|ManifestError|NameError|requires Linux with glibc|Traceback \(most recent call last\)"
for i in $(seq 400); do
  p=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://10.128.0.175:8100/health)
  d=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://10.128.0.177:8200/health)
  if [[ "$p$d" == 200200 ]]; then echo "READY $(date +%T) P=$P D=$D"; exit 0; fi
  if grep -qE "$FATAL" "$P" "$D" 2>/dev/null; then
    echo "ERROR $(date +%T)"
    grep -hE "Error:|Error\b|initialization failed" "$P" "$D" | grep -v "404" | head -8 | cut -c1-300
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT"; exit 2

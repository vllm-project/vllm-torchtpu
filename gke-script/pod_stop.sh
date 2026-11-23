#!/usr/bin/env bash
# 在 pod 主容器里停掉 vLLM（含残留 worker），等 /dev/vfio 释放，清掉孤儿 /dev/shm/psm_*。
#   bash pod_stop.sh          # 只停 vLLM
#   bash pod_stop.sh proxy    # 只停 proxy
# 匹配模式只在本文件里，不会误杀调用者。基于 scripts/dsv41_pd/stop_vllm_local.sh（容器里是 root，无 sudo）。
if [[ "${1:-}" == proxy ]]; then
  pids=$(ps -eo pid,args | awk '/toy_proxy_server[.]py/ {print $1}')
  [ -n "$pids" ] && kill $pids 2>/dev/null; sleep 1
  pids=$(ps -eo pid,args | awk '/toy_proxy_server[.]py/ {print $1}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  echo "$(hostname): proxy stopped"; exit 0
fi
sel() { ps -eo pid,args | awk '/[.]venv(-[a-z]+)?\/bin\/vllm serve|VLLM::|resource_tracker import main|start_1p1d_epic[.]sh|pod_start[.]sh/ {print $1}'; }
pids=$(sel); [ -n "$pids" ] && kill $pids 2>/dev/null
for i in $(seq 40); do n=$(lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l); [ "$n" = 0 ] && break; sleep 3; done
pids=$(sel); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 2
echo "$(hostname): vfio holders=$(lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l)"
if [ -z "$(sel)" ]; then
  n=$(find /dev/shm -maxdepth 1 -name 'psm_*' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && find /dev/shm -maxdepth 1 -name 'psm_*' -delete && echo "$(hostname): removed $n stale /dev/shm/psm_*"
  # Engram 共享表文件在进程退出时 unlink；被 kill -9 的残留也清掉
  find /dev/shm -maxdepth 1 -name 'vllm_engram_*' -delete 2>/dev/null
  if [ -d /dev/shm/torch_tpu_cache ]; then
    gb=$(du -s --block-size=1G /dev/shm/torch_tpu_cache 2>/dev/null | cut -f1)
    [ "${gb:-0}" -gt 30 ] && rm -rf /dev/shm/torch_tpu_cache/* && echo "$(hostname): cleared /dev/shm/torch_tpu_cache (${gb}G)"
  fi
fi
# 本地盘超过 75% 时清 TorchInductor 缓存和 1 天没动过的 vLLM 编译缓存目录
use=$(df --output=pcent / | tail -1 | tr -dc 0-9)
if [ "${use:-0}" -gt 75 ]; then
  rm -rf /tmp/epic-tmp/torchinductor_* /tmp/torchinductor_*
  [ -d /tmp/vllm-cache/torch_compile_cache ] && find /tmp/vllm-cache/torch_compile_cache -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} +
  echo "$(hostname): root disk ${use}% -> pruned caches, now $(df --output=pcent / | tail -1)"
fi

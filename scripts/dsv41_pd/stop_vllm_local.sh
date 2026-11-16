#!/usr/bin/env bash
# 停掉本机的 vLLM 服务和残留 worker，等 /dev/vfio 释放。匹配模式只在本文件里，不会误杀调用者。
sel() { ps -eo pid,args | awk '/[.]venv(-[a-z]+)?\/bin\/vllm serve|VLLM::|resource_tracker import main/ {print $1}'; }
pids=$(sel); [ -n "$pids" ] && kill $pids 2>/dev/null
for i in $(seq 40); do n=$(sudo lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l); [ "$n" = 0 ] && break; sleep 3; done
pids=$(sel); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 2
echo "$(hostname): vfio holders=$(sudo lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l)"
# TorchInductor 的磁盘缓存每次重新编译都会长（2026-10-28 一天长到 159G，把 484G 根分区写满，P 编译时 ENOSPC 退出）。
use=$(df --output=pcent / | tail -1 | tr -dc 0-9)
if [ "${use:-0}" -gt 75 ]; then rm -rf /tmp/torchinductor_"$(whoami)"; echo "$(hostname): root disk ${use}% -> cleared torchinductor cache, now $(df --output=pcent / | tail -1)"; fi
# vLLM 的编译缓存（~/.cache/vllm/torch_compile_cache，按配置哈希分目录）也会一直长（2026-10-29 D 上 393G）：
# 根分区仍超过 75% 时，删掉 1 天以上没动过的哈希目录（正在用的配置会重新生成）。
use=$(df --output=pcent / | tail -1 | tr -dc 0-9)
if [ "${use:-0}" -gt 75 ] && [ -d ~/.cache/vllm/torch_compile_cache ]; then
  find ~/.cache/vllm/torch_compile_cache -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} +
  echo "$(hostname): root disk ${use}% -> pruned vllm compile cache, now $(df --output=pcent / | tail -1)"
fi
# 进程被杀（OOM、kill -9）时 Python SharedMemory 的 /dev/shm/psm_* 不会被回收，占着主机内存
# （2026-10-29 D 上积了 151 个、约 89G，和编译峰值叠加导致 worker 被 OOM kill）。此时本机已没有 vLLM 进程，可以全删。
if [ -z "$(sel)" ]; then
  n=$(find /dev/shm -maxdepth 1 -name 'psm_*' -user "$(whoami)" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && find /dev/shm -maxdepth 1 -name 'psm_*' -user "$(whoami)" -delete && echo "$(hostname): removed $n stale /dev/shm/psm_* files"
fi
# torch-tpu 的 eager 算子缓存写死在 /dev/shm/torch_tpu_cache（tmpfs，直接占主机内存），跨重启只增不减
# （2026-10-30 D 上 93G / 2.5 万个文件），和编译峰值叠加会让 worker 被 OOM kill。超过 30G 且本机已没有 vLLM 进程时清空。
if [ -z "$(sel)" ] && [ -d /dev/shm/torch_tpu_cache ]; then
  gb=$(du -s --block-size=1G /dev/shm/torch_tpu_cache 2>/dev/null | cut -f1)
  if [ "${gb:-0}" -gt 30 ]; then rm -rf /dev/shm/torch_tpu_cache/*; echo "$(hostname): cleared /dev/shm/torch_tpu_cache (${gb}G)"; fi
fi

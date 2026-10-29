#!/usr/bin/env bash
# 停掉本机的 vLLM 服务和残留 worker，等 /dev/vfio 释放。匹配模式只在本文件里，不会误杀调用者。
sel() { ps -eo pid,args | awk '/[.]venv\/bin\/vllm serve|VLLM::|resource_tracker import main/ {print $1}'; }
pids=$(sel); [ -n "$pids" ] && kill $pids 2>/dev/null
for i in $(seq 40); do n=$(sudo lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l); [ "$n" = 0 ] && break; sleep 3; done
pids=$(sel); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 2
echo "$(hostname): vfio holders=$(sudo lsof /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l)"
# TorchInductor 的磁盘缓存每次重新编译都会长（2026-10-28 一天长到 159G，把 484G 根分区写满，P 编译时 ENOSPC 退出）。
use=$(df --output=pcent / | tail -1 | tr -dc 0-9)
if [ "${use:-0}" -gt 75 ]; then rm -rf /tmp/torchinductor_"$(whoami)"; echo "$(hostname): root disk ${use}% -> cleared torchinductor cache, now $(df --output=pcent / | tail -1)"; fi

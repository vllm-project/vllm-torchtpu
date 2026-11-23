#!/usr/bin/env bash
# 每个角色：pod、IP、/dev/shm、NFS、vLLM 进程数、/health；P0 上的 proxy；最新日志尾部。
#   bash status.sh [p0 p1 d]     TAIL=5 控制日志行数
set -uo pipefail
source "$(dirname "$0")/common.sh"
for r in $(roles_from_args "$@"); do
  pod=$(role_pod "$r"); port=$(role_port "$r")
  [[ -n "$pod" ]] || { echo "== $r: pod not Running"; continue; }
  echo "== $r pod=$pod ip=$(pod_ip "$pod"):$port eph=$(eph_name "$pod")"
  kx "$pod" "echo \"  shm=\$(df -h --output=used,size /dev/shm | tail -1 | xargs) nfs=\$(mountpoint -q /zzlfs && echo ok || echo MISSING) disk=\$(df -h --output=pcent / | tail -1 | xargs) mem_avail=\$(free -g | awk '/Mem/{print \$7}')G vllm_procs=\$(ps -eo args | grep -cE '[v]llm serve|[V]LLM::') health=\$(curl -s -m 3 -o /dev/null -w %{http_code} localhost:$port/health)\""
  [[ $r == p0 ]] && kx "$pod" "echo \"  proxy=\$(curl -s -m 3 -o /dev/null -w %{http_code} localhost:8000/healthcheck)\""
  T=$(cat "$LOG_DIR/gke-last-T-$r" 2>/dev/null) && log=$LOG_DIR/gke-$r-$T.log && echo "  log=$log" && tail -n "${TAIL:-3}" "$log" 2>/dev/null | cut -c1-200 | sed 's/^/    /'
done

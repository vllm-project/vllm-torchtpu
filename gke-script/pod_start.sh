#!/usr/bin/env bash
# 在 pod 主容器里启动一个角色（由 deploy.sh 通过 kubectl exec 调用）：
#   bash pod_start.sh p0|p1|d
# 套在 scripts/dsv41_pd/start_1p1d_epic.sh 外面，只补 GKE 差异：
#   - 每个 P 用不同的 Raiden engine id / job name / namespace（D 按 kv_transfer_params 里的 src_engine_id 找 P）；
#   - 容器是 ubuntu:22.04（glibc 2.35）→ 强制用 overlay-el9 里的 tpu_sync；
#   - TMPDIR、VLLM_CACHE_ROOT 放本地盘（节点启动盘），不占 /dev/shm（=主机内存）；
#   - GKE 注入的 TPU_TOPOLOGY=2x2x2 / TPU_HOST_BOUNDS=1,1,2 等由 start_1p1d_epic.sh 覆盖为单机 2x2x1。
set -euo pipefail
ROLE_KEY="${1:?usage: pod_start.sh <p0|p1|d>}"
REPO=${REPO:-/zzlfs/temu/vllm-torchtpu-1p1d}
umask 0000   # 以 root 运行，日志、缓存放共享盘时开发机上的用户也能读写删

case "$ROLE_KEY" in
  p0|p1)
    idx=${ROLE_KEY#p}; ROLE=prefill
    export TPU_RAIDEN_ENGINE_ID="gke-prefill-engine-${idx}" TPU_RAIDEN_JOB_NAME="prefill-${idx}" TPU_KV_TRANSFER_NAMESPACE="gke_prefill_${idx}" ;;
  d)
    ROLE=decode
    export TPU_RAIDEN_ENGINE_ID="gke-decode-engine" TPU_RAIDEN_JOB_NAME="decode" TPU_KV_TRANSFER_NAMESPACE="gke_decode" ;;
  *) echo "unknown role $ROLE_KEY"; exit 1 ;;
esac

export CODE_DIR=${CODE_DIR:-$REPO}
export VENV=${VENV:-/zzlfs/temu/.venv-epic}
export USE_EL9_OVERLAY=1
export POD_IP=${POD_IP:-$(ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1)}
export TMPDIR=${TMPDIR:-/tmp/epic-tmp} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm-cache}
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"
echo "gke role=$ROLE_KEY pod=$(hostname) ip=$POD_IP engine_id=$TPU_RAIDEN_ENGINE_ID shm=$(df -h --output=size /dev/shm | tail -1)"
exec bash "$REPO/scripts/dsv41_pd/start_1p1d_epic.sh" "$ROLE"

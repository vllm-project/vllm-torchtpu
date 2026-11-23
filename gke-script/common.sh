# shellcheck shell=bash
# 2P1D on GKE（复用现有 DWS 占位 pod，不删不改 Job）的公共定义。在开发机上 source。
#
# 角色 → pod：
#   p0  zzl-tpuv7-8-job  completion-index 0   （2x2x2 切片的 host 0，单机模式跑 P，:8100；proxy :8000 也在这里）
#   p1  zzl-tpuv7-8-job  completion-index 1   （2x2x2 切片的 host 1，单机模式跑 P，:8100）
#   d   zzl-tpuv7-4q-job                      （2x2x1，D，:8200）
# pod 名带随机后缀，每次按 label 现查。

NS=${NS:-default}
P_JOB=${P_JOB:-zzl-tpuv7-8-job}
D_JOB=${D_JOB:-zzl-tpuv7-4q-job}
MAIN_C=${MAIN_C:-zzl-container}          # 占位 pod 里的主容器
REPO=${REPO:-/zzlfs/temu/vllm-torchtpu-1p1d}
GKE_DIR=${GKE_DIR:-$REPO/gke-script}
LOG_DIR=${LOG_DIR:-/zzlfs/temu/logs}
ALL_ROLES=(p0 p1 d)

_pod_by_label() {
  kubectl -n "$NS" get pod -l "$1" --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

role_pod() {
  case "$1" in
    p0) _pod_by_label "job-name=$P_JOB,batch.kubernetes.io/job-completion-index=0" ;;
    p1) _pod_by_label "job-name=$P_JOB,batch.kubernetes.io/job-completion-index=1" ;;
    d)  _pod_by_label "job-name=$D_JOB" ;;
    *)  echo "unknown role: $1 (p0|p1|d)" >&2; return 1 ;;
  esac
}

pod_ip() { kubectl -n "$NS" get pod "$1" -o jsonpath='{.status.podIP}'; }

role_port() { [[ "$1" == d ]] && echo 8200 || echo 8100; }

# 在主容器里跑命令
kx() { local pod=$1; shift; kubectl -n "$NS" exec "$pod" -c "$MAIN_C" -- bash -c "$*"; }

# 正在运行的特权 ephemeral 容器名（prep_pod.sh 加的，名字以 shmfix 开头）
eph_name() {
  kubectl -n "$NS" get pod "$1" -o json | python3 -c '
import json,sys
p=json.load(sys.stdin)
for s in p.get("status",{}).get("ephemeralContainerStatuses",[]) or []:
    if s["name"].startswith("shmfix") and "running" in (s.get("state") or {}):
        print(s["name"]); break'
}

# 在主容器的 mount namespace 里以特权执行（ephemeral 容器 --target 主容器，PID 1 就是主容器的 sleep）
kx_priv() { local pod=$1 eph; shift; eph=$(eph_name "$pod"); [[ -n "$eph" ]] || { echo "$pod: no running shmfix ephemeral container (run prep_pod.sh)" >&2; return 1; }
  kubectl -n "$NS" exec "$pod" -c "$eph" -- nsenter -t 1 -m -- bash -c "$*"; }

roles_from_args() { if (($#)); then echo "$@"; else echo "${ALL_ROLES[@]}"; fi; }

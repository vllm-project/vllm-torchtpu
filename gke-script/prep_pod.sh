#!/usr/bin/env bash
# 初始化占位 pod（可重复执行，已做过的步骤跳过），不重建 pod：
#   1. 加一个特权 ephemeral 容器（kubectl debug --profile=sysadmin --target=主容器），只跑 sleep。
#      注意：ephemeral 容器加了就会一直留在 pod spec 里（不能删），不影响主容器和 TPU。
#   2. 通过它 nsenter 进主容器的 mount namespace：/dev/shm 从默认 64M 改成 $SHM_SIZE（Engram DP 共享表 ~94G 要放这里），
#      挂 NFS $NFS_SRC 到 /zzlfs（venv、代码、日志都直接用共享盘）。
#   3. 主容器里 apt 装 git、python3.12（deadsnakes，/usr/bin/python3.12，.venv-epic 的符号链接指向这里）和 RDMA 运行库（tpu_sync 要）。
#   4. tokenizer 拷到 /tmp/dsv41-tokenizer；建 /tmp/epic-tmp、/tmp/vllm-cache（本地盘，即节点启动盘）。
#   5. SEED_CACHE=<目录>（可选）：/tmp/vllm-cache 为空时从共享盘拷编译缓存，如 /zzlfs/temu/cache-archive/d-epic-vllm-cache。
# 用法：bash prep_pod.sh [p0 p1 d]      （默认三个，并行）
set -uo pipefail
source "$(dirname "$0")/common.sh"
SHM_SIZE=${SHM_SIZE:-600G}
NFS_SRC=${NFS_SRC:-10.128.0.30:/zzlfs}
TOKENIZER_SRC=${TOKENIZER_SRC:-/zzlfs/temu/tools/dsv41-tokenizer}
EPH_IMAGE=${EPH_IMAGE:-ubuntu:24.04}

prep_one() {
  local role=$1 pod ip eph
  pod=$(role_pod "$role"); [[ -n "$pod" ]] || { echo "[$role] pod not found / not Running"; return 1; }
  ip=$(pod_ip "$pod")
  echo "[$role] pod=$pod ip=$ip"

  # 1. ephemeral 容器
  eph=$(eph_name "$pod")
  if [[ -z "$eph" ]]; then
    eph="shmfix-$(date +%s)"
    kubectl -n "$NS" debug "$pod" --profile=sysadmin --target="$MAIN_C" --image="$EPH_IMAGE" -c "$eph" -- sleep infinity >/dev/null 2>&1
    for _ in $(seq 60); do [[ -n "$(eph_name "$pod")" ]] && break; sleep 2; done
    [[ -n "$(eph_name "$pod")" ]] || { echo "[$role] ephemeral container $eph did not start"; return 1; }
    echo "[$role] added ephemeral container $eph"
  fi

  # 2. /dev/shm、NFS
  kx_priv "$pod" "
    cur=\$(df --output=size -B1G /dev/shm | tail -1 | tr -dc 0-9); want=\$(echo $SHM_SIZE | tr -dc 0-9)
    if [ \"\${cur:-0}\" -lt \"\$want\" ]; then mount -o remount,size=$SHM_SIZE /dev/shm; fi
    if ! mountpoint -q /zzlfs; then
      mkdir -p /zzlfs
      mount -i -t nfs4 -o vers=4.1,rsize=524288,wsize=1048576,hard,proto=tcp,timeo=600,addr=${NFS_SRC%%:*},clientaddr=$ip $NFS_SRC /zzlfs
    fi" || { echo "[$role] shm/nfs setup failed"; return 1; }

  # 3–5. 主容器里的软件包、tokenizer、目录
  kx "$pod" "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    if [ ! -x /usr/bin/python3.12 ] || ! dpkg -s libibverbs1 >/dev/null 2>&1 || ! command -v git >/dev/null; then
      apt-get update -qq
      apt-get install -y -qq software-properties-common curl ca-certificates >/dev/null
      add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
      apt-get update -qq
      apt-get install -y -qq python3.12 python3.12-venv libpython3.12 libibverbs1 librdmacm1 ibverbs-providers libnuma1 \
        iproute2 procps lsof psmisc rsync git >/dev/null 2>&1
    fi
    # 共享盘上的仓库属于开发机用户，容器里是 root：不加 safe.directory 时 git 拒绝读（启动日志里的 commit 字段要用）
    git config --global --get-all safe.directory | grep -qx '*' || git config --global --add safe.directory '*'
    mkdir -p /tmp/dsv41-tokenizer /tmp/epic-tmp /tmp/vllm-cache
    cp -r --no-preserve=mode,ownership $TOKENIZER_SRC/. /tmp/dsv41-tokenizer/
    if [ -n '${SEED_CACHE:-}' ] && [ -z \"\$(ls -A /tmp/vllm-cache)\" ]; then
      echo 'seeding /tmp/vllm-cache from ${SEED_CACHE:-}'; cp -r --no-preserve=mode,ownership '${SEED_CACHE:-}'/. /tmp/vllm-cache/
    fi
  " || { echo "[$role] package/tokenizer setup failed"; return 1; }

  kx "$pod" "echo \"[$role] shm=\$(df -h --output=size /dev/shm | tail -1 | tr -d ' ') nfs=\$(mountpoint -q /zzlfs && echo ok || echo MISSING) py=\$(/usr/bin/python3.12 -V 2>&1) disk_free=\$(df -h --output=avail / | tail -1 | tr -d ' ') cache=\$(du -sh /tmp/vllm-cache | cut -f1)\""
}

pids=()
for r in $(roles_from_args "$@"); do prep_one "$r" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc

#!/usr/bin/env bash
# 2P1D 一键部署（在开发机上跑）：初始化 pod → 停旧进程 → 启动 P0/P1/D → 等健康 → 在 P0 启动 proxy。
#   bash deploy.sh                    # 全部
#   ROLES="d" bash deploy.sh          # 只重启 D（不动 proxy，除非 PROXY=1）
#   SKIP_PREP=1 bash deploy.sh        # 跳过 prep_pod.sh
#   NO_WAIT=1 bash deploy.sh          # 不等健康、不启 proxy
# 传给启动脚本的环境变量：
#   COMMON_ENV="K=V ..."  三个角色都用；P_ENV="K=V ..."  两个 P；D_ENV="K=V ..."  D
#   默认 D_ENV 是 REPRODUCE.zh.md 第 3 节报告用的配置（async + MAX_NUM_SEQS=32 + compile [16,48,96,192,2048]）。
#   可用变量见 start_1p1d_epic.sh（ASYNC_SCHED、MAX_NUM_SEQS、PREFILL_COMPILE_SIZES、DECODE_COMPILE_SIZES、
#   API_SERVERS、VLLM_DP_WAVE_START_DELAY_MS、KV_BYTES、MAX_MODEL_LEN、EXTRA_ARGS …）。
# 日志：$LOG_DIR/gke-<role>-<T>.log（共享盘，开发机上直接看）；T 记在 $LOG_DIR/gke-last-T。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"
ROLES=${ROLES:-"p0 p1 d"}
COMMON_ENV=${COMMON_ENV:-}
P_ENV=${P_ENV:-}
D_ENV=${D_ENV-"MAX_NUM_SEQS=32 DECODE_COMPILE_SIZES=[16,48,96,192,2048]"}
T=$(date +%m%d-%H%M)

if [[ "${SKIP_PREP:-0}" != 1 ]]; then
  bash "$HERE/prep_pod.sh" $ROLES || { echo "prep failed"; exit 1; }
fi

declare -A POD IP
for r in "${ALL_ROLES[@]}"; do POD[$r]=$(role_pod "$r"); IP[$r]=$(pod_ip "${POD[$r]}"); done

for r in $ROLES; do kx "${POD[$r]}" "bash $GKE_DIR/pod_stop.sh" & done; wait

for r in $ROLES; do
  env_r="$COMMON_ENV"; [[ $r == d ]] && env_r+=" $D_ENV" || env_r+=" $P_ENV"
  log=$LOG_DIR/gke-$r-$T.log
  # 用 ; 不用 &&：& 只作用在最后一条命令上；否则整个链进后台子 shell，占着 exec 的输出流，kubectl exec 不返回
  kx "${POD[$r]}" "cd /tmp; umask 0000; env POD_IP=${IP[$r]} $env_r setsid nohup bash $GKE_DIR/pod_start.sh $r > $log 2>&1 < /dev/null &"
  echo "$r -> ${POD[$r]} (${IP[$r]}:$(role_port "$r")) log=$log env=[$env_r]"
done
for r in $ROLES; do echo "$T" > "$LOG_DIR/gke-last-T-$r"; done
echo "$T" > "$LOG_DIR/gke-last-T"

[[ "${NO_WAIT:-0}" == 1 ]] && exit 0
bash "$HERE/wait_ready.sh" $ROLES || exit 1

if [[ "$ROLES" == *p0* || "${PROXY:-0}" == 1 ]]; then
  T=$T bash "$HERE/start_proxy.sh"
fi

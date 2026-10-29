#!/usr/bin/env bash
# 停掉两台机器上的 vLLM（含残留 worker），确认 TPU（/dev/vfio）已释放，再用 start_1p1d_epic.sh 启动 P 和 D。
#   bash restart_epic.sh            # P 和 D 都重启
#   ROLES=decode bash restart_epic.sh
# 额外环境变量（ASYNC_SCHED、PREFILL_COMPILE_SIZES 等）原样传给启动脚本。
# 停进程的匹配模式在 stop_vllm_local.sh 里，不出现在任何命令行上，所以不会误杀调用者自己的 shell。
set -uo pipefail
P_HOST=zzl-tpu7x-slice-mig-2z6z
D_HOST=zzl-tpu7x-slice-mig-clsk
ROLES=${ROLES:-"prefill decode"}
T=$(date +%m%d-%H%M)
PASS_ENV=""
for v in VENV TPU_DUMMY_STEP_TIMING NO_KV_TRANSFER TP_SIZE DP_SIZE API_SERVERS MAX_BATCHED_TOKENS VLLM_DP_WAVE_START_DELAY_MS ASYNC_SCHED PREFILL_COMPILE_SIZES DECODE_COMPILE_SIZES KV_BYTES MAX_MODEL_LEN MAX_NUM_SEQS EXTRA_ARGS CODE_DIR TPU_RECLAIM_COMPILE_MEMORY MOE_FUSED_EP_PREPROCESS_BACKEND VLLM_LOGGING_PREFIX TPU_STAGE3_PROFILE; do
  [[ -n "${!v+x}" ]] && PASS_ENV+="$v=$(printf %q "${!v}") "
done

stop_host() {
  ssh -o BatchMode=yes "$1" "bash /zzlfs/temu/scripts/stop_vllm_local.sh"
}

for role in $ROLES; do
  host=$P_HOST; [[ $role == decode ]] && host=$D_HOST
  stop_host "$host"
done
for role in $ROLES; do
  host=$P_HOST; [[ $role == decode ]] && host=$D_HOST
  log=/zzlfs/temu/logs/${role}-epic-$T.log
  ssh -n -f -o BatchMode=yes "$host" "cd /zzlfs/temu && $PASS_ENV setsid nohup bash /zzlfs/temu/scripts/start_1p1d_epic.sh $role > $log 2>&1 < /dev/null &" > /dev/null 2>&1
  echo "$role -> $host log=$log"
done
echo "$T" > /tmp/epic_T

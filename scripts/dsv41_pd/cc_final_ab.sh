#!/usr/bin/env bash
# 最终 A/B：AgentX c512 3600s；每组都先重启 P（冷前缀缓存），D 只清前缀缓存。
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd)
run() {  # $1=tag  $2=P_ENV  $3=P_EXTRA
  ROLES=prefill P_ENV="$2" P_EXTRA="$3" bash $W/cc_start_pd.sh
  T=$(cat /zzlfs/temu/logs/.cc_T); P=/zzlfs/temu/logs/prefill-cc-$T.log; sleep 30
  for i in $(seq 1 300); do
    [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 10.128.0.62:8100/health) == 200 ]] && { echo "$1: P ready T=$T"; break; }
    grep -q -E "^Traceback|EngineCore failed" $P && { echo "$1: P FATAL"; return 1; }; sleep 10
  done
  curl -s 10.128.0.62:8100/v1/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v41","prompt":"Hi","max_tokens":2}' > /dev/null
  CONC=512 DURATION=3600 TAG=$1 PROF_AT=0 bash $W/cc_agentx.sh 2>&1 | tail -1
}
run cc-final-cap4e8 "TPU_DP_ATTN_STEP_CAP=4e8 TPU_DP_ATTN_HEAVY_CTX=131072 TPU_DP_ATTN_MIN_CHUNK=256" "--scheduler-cls vllm_torchtpu.core.dp_attn_balance_scheduler.TpuDpAttnBalanceScheduler"
run cc-final-base "" ""

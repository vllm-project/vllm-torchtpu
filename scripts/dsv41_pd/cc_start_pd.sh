#!/usr/bin/env bash
# 本轮（attention 均衡）用：从 attnbal worktree 起 P（10.128.0.62）/ D（10.128.0.39），带 xprof 和每步工作量日志。
#   ROLES="prefill decode" bash cc_start_pd.sh ；额外环境变量 P_ENV / D_ENV 原样加在对应启动命令前。
set -uo pipefail
P=${P_HOST:-10.128.0.62}; D=${D_HOST:-10.128.0.39}; ROLES=${ROLES:-"prefill decode"}
W=$(cd "$(dirname "$0")/../.." && pwd); S=$W/scripts/dsv41_pd/start_1p1d_epic.sh
T=$(date +%m%d-%H%M); echo $T > /zzlfs/temu/logs/.cc_T
COMMON="TMPDIR=/var/tmp/epic-tmp VLLM_CACHE_ROOT=/var/tmp/epic-vllm-cache CODE_DIR=$W VLLM_LOGGING_PREFIX='%(created).3f ' TPU_DSPARK_SAMPLED_DRAFT=1 TPU_DP_ATTN_WORK_LOG=/var/tmp/dpwork-$T"
PROF="--profiler-config.profiler torch --profiler-config.torch_profiler_dir /var/tmp/xprof --profiler-config.ignore_frontend true"
for r in $ROLES; do h=$P; [[ $r == decode ]] && h=$D; ssh -o BatchMode=yes $h "bash /zzlfs/temu/scripts/stop_vllm_local.sh" | tail -1; done
for r in $ROLES; do
  if [[ $r == prefill ]]; then
    ssh -n -f $P "cd /zzlfs/temu && $COMMON ${P_ENV:-} EXTRA_ARGS='--enable-prompt-tokens-details $PROF ${P_EXTRA:-}' setsid nohup bash $S prefill > /zzlfs/temu/logs/prefill-cc-$T.log 2>&1 < /dev/null &" >/dev/null 2>&1
  else
    ssh -n -f $D "cd /zzlfs/temu && $COMMON MAX_NUM_SEQS=64 DECODE_COMPILE_SIZES='[16,48,96,144,192,288,384,2048]' ${D_ENV:-} EXTRA_ARGS='$PROF ${D_EXTRA:-}' setsid nohup bash $S decode > /zzlfs/temu/logs/decode-cc-$T.log 2>&1 < /dev/null &" >/dev/null 2>&1
  fi
  echo "$r started T=$T"
done

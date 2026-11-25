#!/usr/bin/env bash
# 重启 P（给定 P_ENV / P_EXTRA）后跑一轮 AgentX：TAG=... DURATION=3600 P_ENV=... P_EXTRA=... bash cc_run_one.sh
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd)
ROLES=prefill P_ENV="${P_ENV:-}" P_EXTRA="${P_EXTRA:-}" bash $W/cc_start_pd.sh
T=$(cat /zzlfs/temu/logs/.cc_T); P=/zzlfs/temu/logs/prefill-cc-$T.log; sleep 30
for i in $(seq 1 300); do
  [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 10.128.0.62:8100/health) == 200 ]] && { echo "$TAG: P ready T=$T"; break; }
  grep -q -E "^Traceback|EngineCore failed" $P && { echo "$TAG: P FATAL"; grep -m2 -A8 -E "^Traceback|EngineCore failed" $P | cut -c1-300; exit 1; }; sleep 10
done
grep -m1 "attention-balance scheduler:" $P | cut -c60-400
curl -s 10.128.0.62:8100/v1/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v41","prompt":"Hi","max_tokens":2}' > /dev/null
CONC=${CONC:-512} DURATION=${DURATION:-3600} TAG=$TAG PROF_AT=0 bash $W/cc_agentx.sh 2>&1 | tail -1

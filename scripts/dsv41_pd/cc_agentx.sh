#!/usr/bin/env bash
# 本轮 AgentX：起 toy proxy（P 机 :8000，前缀路由 + D 路由 + 命中诊断），清两端前缀缓存，跑 run_agentx.sh；
# PROF_AT 秒后对 P 抓 PROF_SECS 秒 xprof（0 = 不抓）。
#   CONC=512 DURATION=900 TAG=base PROF_AT=1500 bash cc_agentx.sh
set -uo pipefail
P=${P_HOST:-10.128.0.62}; D=${D_HOST:-10.128.0.39}
W=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-run}; T=$(date +%m%d-%H%M)
PLOG=/zzlfs/temu/logs/proxy-cc-$TAG-$T.log
ssh -o BatchMode=yes $P 'pkill -f "[t]oy_proxy_server.py" ; true'; sleep 2
ssh -n -f $P "cd /zzlfs/temu && PROXY_SESSION_AFFINITY_DP=8 PROXY_DECODE_ROUTING=1 PROXY_HIT_DIAG=1 ${PROXY_ENV:-} \
  VLLM_PLUGINS= TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1 PROXY_TOKENIZER=/tmp/dsv41-tokenizer \
  setsid nohup /zzlfs/temu/.venv-epic/bin/python $W/toy_proxy_server.py --host 0.0.0.0 --port 8000 \
  --prefiller-host $P --prefiller-port 8100 --decoder-host $D --decoder-port 8200 > $PLOG 2>&1 < /dev/null &" >/dev/null 2>&1
for i in $(seq 60); do curl -s -o /dev/null --max-time 2 http://$P:8000/ && break; grep -q "startup complete" $PLOG 2>/dev/null && break; sleep 2; done
curl -sf -X POST http://$P:8100/reset_prefix_cache && curl -sf -X POST http://$D:8200/reset_prefix_cache; echo " caches reset; proxy log $PLOG"
if [[ "${PROF_AT:-0}" != "0" ]]; then
  ( sleep $PROF_AT; ssh -o BatchMode=yes $P 'rm -rf /var/tmp/xprof/*'
    curl -s -X POST http://$P:8100/start_profile; sleep ${PROF_SECS:-3}; curl -s -X POST http://$P:8100/stop_profile
    echo "xprof captured at $(date +%T)" >> $PLOG ) &
fi
P_ADDR=$P D_ADDR=$D PROXY_URL=http://$P:8000 CONC=${CONC:-512} DURATION=${DURATION:-900} TAG=$TAG bash $W/run_agentx.sh

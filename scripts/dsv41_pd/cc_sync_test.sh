#!/usr/bin/env bash
# 同步上游 PR 后的验证：重启 P、D（巨型请求处理开启），冒烟 + PD logprob 检查 + AgentX c512 3600s。
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd)
ROLES="prefill decode" P_ENV="TPU_DP_ATTN_STEP_CAP=0 TPU_P_WHALE_TOKENS=16384 TPU_P_WHALE_STEP_TOKENS=1024" \
  P_EXTRA="--scheduler-cls vllm_torchtpu.core.dp_attn_balance_scheduler.TpuDpAttnBalanceScheduler" bash $W/cc_start_pd.sh
T=$(cat /zzlfs/temu/logs/.cc_T); P=/zzlfs/temu/logs/prefill-cc-$T.log; D=/zzlfs/temu/logs/decode-cc-$T.log; sleep 30
for i in $(seq 1 480); do
  pc=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 10.128.0.62:8100/health); dc=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 10.128.0.39:8200/health)
  [[ $pc == 200 && $dc == 200 ]] && { echo "READY after $((i*10))s T=$T"; break; }
  grep -q -E "^Traceback|EngineCore failed|died unexpectedly|RuntimeUnexpectedCoreHalt" $P $D && { echo FATAL; grep -h -m2 -A10 -E "^Traceback|EngineCore failed" $P $D | cut -c1-300; exit 1; }
  sleep 10
done
ssh 10.128.0.62 'pkill -f "[t]oy_proxy_server.py"'; sleep 1
ssh -n -f 10.128.0.62 "cd /zzlfs/temu && PROXY_SESSION_AFFINITY_DP=8 PROXY_DECODE_ROUTING=1 VLLM_PLUGINS= TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1 PROXY_TOKENIZER=/tmp/dsv41-tokenizer setsid nohup /zzlfs/temu/.venv-epic/bin/python $W/toy_proxy_server.py --host 0.0.0.0 --port 8000 --prefiller-host 10.128.0.62 --prefiller-port 8100 --decoder-host 10.128.0.39 --decoder-port 8200 > /zzlfs/temu/logs/proxy-cc-synccheck-$T.log 2>&1 < /dev/null &" >/dev/null 2>&1; sleep 15
echo "== smoke"; curl -s 10.128.0.62:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":12,"temperature":0}' | head -c 300; echo
echo "== logprob check"; /zzlfs/temu/.venv-epic/bin/python -I $W/pd_logprob_check.py --proxy http://10.128.0.62:8000 --direct http://10.128.0.39:8200 --tokens 3000 20000 100000 --offset 1000000 > /zzlfs/temu/logs/logprob-$T.log 2>&1; tail -8 /zzlfs/temu/logs/logprob-$T.log
TAG=${TAG:-cc-sync-csa2-whale16k} CONC=512 DURATION=3600 PROF_AT=0 bash $W/cc_agentx.sh 2>&1 | tail -1

#!/bin/bash
# In-pod: relaunch the PD decode server with env overrides. Usage: x1_relaunch.sh KEY=VAL [KEY=VAL ...]
# Values may contain spaces/brackets (e.g. "COMPILE_SIZES=[8, 16, 32, 64, 384]").
set -u
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  grep -v "^export ${k}=" /tmp/pd_env.sh > /tmp/pd_env.new; mv /tmp/pd_env.new /tmp/pd_env.sh
  printf "export %s='%s'\n" "$k" "$v" >> /tmp/pd_env.sh
done
echo "[relaunch] env tail:"; tail -n 8 /tmp/pd_env.sh
# stop the running server (never kill ourselves / our parent shell)
for p in $(pgrep -f "VLLM::|EngineCore|vllm serve|in_pod_serve.sh|ApiServer" ); do
  [ "$p" = "$$" ] && continue; [ "$p" = "$PPID" ] && continue
  kill -9 "$p" 2>/dev/null || true
done
sleep 5
for i in $(seq 1 30); do curl -s -m 2 -o /dev/null http://127.0.0.1:9400/health || break; sleep 2; done
# checkpoint shards are deleted after load; re-stage if missing
if ! ls /dev/shm/DeepSeek-V4.1-Flash/*.safetensors >/dev/null 2>&1; then
  echo "[relaunch] re-staging checkpoint ..."; bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup_relaunch.log 2>&1; tail -2 /dev/shm/pd_setup_relaunch.log
  grep -q PD_SETUP_OK /dev/shm/pd_setup_relaunch.log || { echo "[relaunch] SETUP_FAILED"; exit 2; }
fi
rm -rf /dev/shm/profiles; mkdir -p /dev/shm/profiles
nohup setsid bash /tmp/in_pod_serve.sh decode > /dev/shm/x1_relaunch.log 2>&1 < /dev/null &
disown
echo "[relaunch] server launching"

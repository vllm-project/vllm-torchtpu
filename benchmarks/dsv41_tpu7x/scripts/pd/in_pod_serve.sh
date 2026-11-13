#!/bin/bash
# In-pod P/D server supervisor: launch one role via the vtt launcher, and delete the
# checkpoint from tmpfs once all 8 DP ranks have loaded it (as the decode driver
# does) when PD_DELETE_CKPT=1. Env comes from /tmp/pd_env.sh (written by run_pd.py).
# Usage: in_pod_serve.sh prefill|decode
set -uo pipefail
ROLE="$1"
source /tmp/pd_env.sh
export ROLE
LOG=/dev/shm/pd_${ROLE}.log
: > "$LOG"
bash /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh >> "$LOG" 2>&1 &
SRV=$!
echo "$SRV" > /dev/shm/pd_${ROLE}.pid
deleted=0
while kill -0 "$SRV" 2>/dev/null; do
  sleep 10
  if [ "$deleted" = 0 ] && [ "${PD_DELETE_CKPT:-1}" = 1 ]; then
    c=$(grep -c "Memory statistics |" "$LOG" || true)
    if [ "${c:-0}" -ge 8 ]; then
      rm -f /dev/shm/DeepSeek-V4.1-Flash/*.safetensors /dev/shm/DeepSeek-V4.1-Flash/.done_*
      echo "[PD-SERVE] deleted checkpoint shards after 8x Memory statistics" >> "$LOG"
      deleted=1
    fi
  fi
done
wait "$SRV"
echo "[PD-SERVE] server exited rc=$?" >> "$LOG"

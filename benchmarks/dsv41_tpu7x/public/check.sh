#!/usr/bin/env bash
# On the PREFILL pod, once both roles are healthy: start the proxy, run the PD gates and the KV-transfer check.
# Usage: check.sh <decode pod IP>
set -euo pipefail
DECODE_IP=${1:?decode pod IP}
V=/workspace/vllm-torchtpu; B=$V/benchmarks/dsv41_tpu7x
for p in 8400 8401 8402 8403; do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health)" = 200 ] || { echo "prefill :$p not healthy yet"; exit 1; }
done
[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$DECODE_IP:9400/health)" = 200 ] || { echo "decode $DECODE_IP:9400 not healthy yet"; exit 1; }
for p in $(pgrep -f toy_proxy_server || true); do kill -9 "$p" 2>/dev/null || true; done
cd $V && nohup setsid python3 examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8000 \
  --prefiller-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 --prefiller-ports 8400 8401 8402 8403 --prefiller-dp-size 0 \
  --decoder-hosts "$DECODE_IP" --decoder-ports 9400 --decoder-dp-size 8 > /dev/shm/pd_proxy.log 2>&1 < /dev/null &
for _ in $(seq 1 30); do [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/healthcheck)" = 200 ] && break; sleep 2; done
python3 -u $B/scripts/pd/pd_gates.py --decode "http://$DECODE_IP:9400" 2>&1 | tee /dev/shm/pd_gates.log | grep -E "GATE-|PD-GATES"
python3 $V/scripts/vllm/integration/check_dsv4_stage3_log.py /dev/shm/pd_prefill.log --min-sends 5 | tee /dev/shm/pd_stage3_check.txt

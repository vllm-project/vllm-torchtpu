#!/usr/bin/env bash
# On the DECODE pod: decode-only throughput sweep at 32 and 64 sequences per rank (K=5, greedy, 4K context,
# 1,024 output tokens). Repeat 1 runs on a cold prefix cache; read repeats 2 and 3. Usage: sweep.sh [repeats]
set -euo pipefail
REP=${1:-3}; B=/workspace/vllm-torchtpu/benchmarks/dsv41_tpu7x
cp $B/scripts/pd/pd_gates.py /tmp/pd_gates.py
for i in $(seq 1 "$REP"); do
  python3 -u $B/scripts/pd/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 4000 --out-len 1024 \
    --concs 256,512 --dp 8 --k 5 --out /dev/shm/sweep_rep$i.json > /dev/shm/sweep_rep$i.log 2>&1
  echo "--- repeat $i"; grep "\[X1\]" /dev/shm/sweep_rep$i.log
done

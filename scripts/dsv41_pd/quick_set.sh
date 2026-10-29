#!/usr/bin/env bash
# 快速对比用的子集：1k+1k / 32k+16k / 32k+168k / 512k+32k 的 1P，加 1k+1k 的 4P。
#   TAG=async bash quick_set.sh   → /zzlfs/temu/results/quick-<TAG>-<time>.jsonl / .log
set -uo pipefail
K=1024
OUT=/zzlfs/temu/results/quick-${TAG:-run}-$(date +%m%d-%H%M).jsonl
LOG=${OUT%.jsonl}.log
B="/zzlfs/temu/.venv/bin/python -I /zzlfs/temu/scripts/bench_cached_prefill.py --pin distinct --out $OUT"
{
  $B --cached $K --new $K --conc 1 --repeats 5
  $B --cached $K --new $K --conc 4 --repeats 5
  $B --cached $((32*K)) --new $((16*K)) --conc 1 --repeats 3
  $B --cached $((32*K)) --new $((168*K)) --conc 1 --repeats 2
  $B --cached $((512*K)) --new $((32*K)) --conc 1 --repeats 2
} > $LOG 2>&1
grep "^== " $LOG | sed 's/ pin=distinct.*(median)://'
echo "log=$LOG"

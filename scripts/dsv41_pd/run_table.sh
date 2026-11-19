#!/usr/bin/env bash
# 按客户表格跑完 11 个「cached + prefill」场景，每个场景 1P（conc=1）和 4P（conc=4），结果追加到 $OUT。
#   OUT=/zzlfs/temu/results/table-$(date +%m%d-%H%M).jsonl PIN=distinct bash run_table.sh
#   ONLY="1024:1024 8192:4096" bash run_table.sh      # 只跑部分场景
# k = 1024 token；客户的 168k 按 172032。长场景（总长 >= 200k）轮数少一些。
set -uo pipefail
PY=/zzlfs/temu/.venv/bin/python
BENCH=/zzlfs/temu/scripts/bench_cached_prefill.py
OUT=${OUT:-/zzlfs/temu/results/table-$(date +%m%d-%H%M).jsonl}
PIN=${PIN:-distinct}
CONCS=${CONCS:-"1 4"}
mkdir -p "$(dirname "$OUT")"
K=1024
SCENARIOS=${ONLY:-"$((1*K)):$((1*K)) $((8*K)):$((4*K)) $((32*K)):$((16*K)) $((64*K)):$((32*K)) $((168*K)):$((32*K)) \
$((32*K)):$((168*K)) $((64*K)):$((64*K)) $((256*K)):$((32*K)) $((512*K)):$((32*K)) $((768*K)):$((32*K)) $((32*K)):$((768*K))"}
echo "out=$OUT pin=$PIN concs=$CONCS"
for s in $SCENARIOS; do
  cached=${s%%:*}; new=${s##*:}
  if (( cached + new >= 200 * K )); then repeats=${LONG_REPEATS:-3}; else repeats=${REPEATS:-5}; fi
  for conc in $CONCS; do
    echo "######## $(date '+%F %T') cached=$cached new=$new conc=$conc pin=$PIN repeats=$repeats"
    "$PY" -I "$BENCH" --cached "$cached" --new "$new" --conc "$conc" --pin "$PIN" \
      --repeats "$repeats" --out "$OUT" || echo "!! failed cached=$cached new=$new conc=$conc"
  done
done
echo "######## $(date '+%F %T') done"

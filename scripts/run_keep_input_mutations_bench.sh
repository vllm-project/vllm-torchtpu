#!/bin/bash
# run_keep_input_mutations_bench.sh <outdir>: every (variant, mode) in its own process.
out=${1:-/tmp/kim_out}; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$here/../src:${PYTHONPATH:-}
for v in jaxop index; do
  for m in 0 1 2; do
    echo "=== variant=$v mode=$m"
    timeout 1800 python "$here/benchmark_keep_input_mutations.py" --variant $v --mode $m 2>&1 | tail -40 | tee -a "$out/kim.log"
  done
done
echo "=== SUMMARY"; grep '^RESULT' "$out/kim.log"

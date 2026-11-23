#!/bin/bash
# q38 HC-HIGHROWS microbench body (BENCH=script on cdk-tpu-exos, one 4-chip v7x host, 2x2x1, 8 TensorCores). Run from the tree
# root with PYTHONPATH=src. Pattern copied from DECODE-L3's HCBIND body (q38_decl3/run_hcbind.sh; credit) with counted changes:
# the bench (tests/q38_hchigh/tpu_bench_hcwide.py), the reader (q38_hchigh/hcw_read.py), the rows / arms, the output dir, the
# tree checks. On cdk-tpu-exos the bench sibling's Q38X_SCRIPT_OK=hcw/OK writes run01_A/result.json ok=true iff this body exits
# 0 and the marker exists. Selftests first (launch barrier, bench engagement rules + mutants, reader + mutants; any failure runs
# nothing), then ONE bench process under its own KILL guard (3300 s), then the reader. Marker hcw/OK (and exit 0) iff the
# reader printed a pod VERDICT that is not VOID (the combined verdict is read off both pods' JSONs afterwards: hcw_read.py --combine). Outputs under $CDK_OUTPUT_DIR/q38f-bench/script/hcw/ (hcw.json / .log, hlo/,
# hcw_verdict.txt, OK).
set -u
# Two pods run in parallel (DSTEP-LEAD 2026-11-20): `run_hcw.sh lo` (rows 160-288, CMP included) and `run_hcw.sh hi` (rows 384-496).
POD="${1:-}"
case "$POD" in
  lo) ROWS=160,192,240,288; ARMS=S512,S256,W1,W2,WD,W1C,W2C,CMP ;;
  hi) ROWS=384,448,496; ARMS=S512,S256,W1,W2,WD,W1C,W2C ;;
  *) echo "HCW ABORT: pod must be lo or hi (got '$POD')"; exit 3 ;;
esac
O="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script/hcw"
mkdir -p "$O"
rm -f "$O/OK"
B=tests/q38_hchigh/tpu_bench_hcwide.py
R=q38_hchigh/hcw_read.py
echo "HCW start pod $POD $(date -u +%FT%TZ) host $(hostname) commit $(cat COMMIT 2>/dev/null) inherited LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
echo "HCW tree src $(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10) (want $(cat q38_hchigh/WANT_SRC_HCW))"
for v in $(env | grep -oE '^TPU_QWEN4_HC_[A-Z0-9_]+'); do echo "HCW env $v=${!v} (the bench calls the kernels directly; no HC env key is read)"; done
fail_out() { echo "HCW ABORT: $1"; exit 3; }
[ "$(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10)" = "$(cat q38_hchigh/WANT_SRC_HCW)" ] \
  || fail_out "tree src is not the proven q38-hchigh-wide src"
python3 -B $B --barrier-selftest > "$O/barrier_selftest.txt" 2>&1 || { cat "$O/barrier_selftest.txt"; fail_out "launch-barrier selftest failed"; }
tail -1 "$O/barrier_selftest.txt"
python3 -B $B --selftest --mutants > "$O/bench_selftest.txt" 2>&1 || { cat "$O/bench_selftest.txt"; fail_out "bench selftest failed"; }
grep -E "SELFTEST (PASS|FAIL)|MUTANTS" "$O/bench_selftest.txt"
python3 -I -B $R --selftest --mutants > "$O/reader_selftest.txt" 2>&1 || { cat "$O/reader_selftest.txt"; fail_out "reader selftest failed"; }
tail -2 "$O/reader_selftest.txt"
echo "HCW bench start $(date -u +%T)Z (pod $POD: arms $ARMS; base + base2 = S512 alone; rows $ROWS; guard 3300s)"
timeout -s KILL 3300 python3 -B $B --rows $ROWS --arms $ARMS --out "$O/hcw.json" \
  --hlo-dir "$O/hlo" > "$O/hcw.log" 2>&1
r=$?; echo "HCW bench rc=$r end $(date -u +%T)Z"
grep -a "^HCW \|^Q38_" "$O/hcw.log" | cut -c1-900
[ $r = 0 ] || tail -30 "$O/hcw.log"
python3 -I -B $R --pod $POD "$O" > "$O/hcw_verdict.txt" 2>&1; rr=$?
echo "HCW reader rc=$rr (0 VERDICT, 1 VERDICT VOID, 2 no JSON)"; cut -c1-900 "$O/hcw_verdict.txt"
echo "HCW end $(date -u +%FT%TZ)"
if grep -q "^VERDICT " "$O/hcw_verdict.txt" && ! grep -q "^VERDICT VOID" "$O/hcw_verdict.txt"; then
  touch "$O/OK" && echo "HCW marker $O/OK written (Q38X_SCRIPT_OK=hcw/OK)"
  exit 0
fi
echo "HCW no marker (VOID / no verdict line)"
exit 1

#!/bin/bash
# run_hckmb.sh (q38 HCK, DSTEP lane under DSTEP-LEAD-6, 2026-11-23): the TPU_QWEN4_HC_DECODE_W1_SCHED kernel microbench body (BENCH=script on cdk-tpu-exos, one
# 4-chip v7x host, 2x2x1, 8 TensorCores). Bench tests/q38_hck/tpu_bench_hckmb.py (B1 decode LIBTPU = 2 served flags + p7 + barrier off,
# pinned): timed rows 288,320 in drift-cancelling ABBA order (B1, X1, X2, B2; B = the served W1F kernel, X = the key's schedule) plus a
# rotating phase over B / X / XD / XO; device bitwise health at 160,288,320,384,496 with the live negative control XM; reader
# q38_hck/hckmb_read.py. Pattern of the split car's body (q38_hck/run_split body): selftests first (launch barrier + clock, bench
# engagement / census rules + mutants, reader + mutants), then ONE bench process under its own KILL guard (3300 s), then the reader. Marker
# hckmb/OK (and exit 0) iff the reader printed a VERDICT that is not VOID (KEEP / FLAT / LOSS / NOT-BITWISE are verdicts of record).
# Outputs under $CDK_OUTPUT_DIR/q38f-bench/script/hckmb/ (hckmb.json / hckmb.log, hlo/, hckmb_verdict.txt, OK). Plain python only.
set -u
O="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script/hckmb"
mkdir -p "$O"
rm -f "$O/OK"
B=tests/q38_hck/tpu_bench_hckmb.py
R=q38_hck/hckmb_read.py
ROWS=288,320; HROWS=160,288,320,384,496
echo "HCKMB start $(date -u +%FT%TZ) host $(hostname) commit $(cat COMMIT 2>/dev/null) inherited LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
SRCH=$(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10)
echo "HCKMB tree src $SRCH (want $(cat q38_hck/WANT_SRC_HCKMB))"
for v in $(env | grep -oE '^TPU_QWEN4_HC_[A-Z0-9_]+'); do echo "HCKMB env $v=${!v} (the bench calls the kernels directly; no HC env key is read)"; done
fail_out() { echo "HCKMB ABORT: $1"; exit 3; }
[ "$SRCH" = "$(cat q38_hck/WANT_SRC_HCKMB)" ] || fail_out "tree src is not the q38-hck W1_SCHED src"
python3 -B $B --barrier-selftest > "$O/barrier_selftest.txt" 2>&1 || { cat "$O/barrier_selftest.txt"; fail_out "launch-barrier / clock selftest failed"; }
tail -1 "$O/barrier_selftest.txt"
python3 -B $B --selftest --mutants > "$O/bench_selftest.txt" 2>&1 || { cat "$O/bench_selftest.txt"; fail_out "bench selftest failed"; }
grep -E "SELFTEST (PASS|FAIL)|MUTANTS" "$O/bench_selftest.txt"
python3 -I -B $R --selftest --mutants > "$O/reader_selftest.txt" 2>&1 || { cat "$O/reader_selftest.txt"; fail_out "reader selftest failed"; }
tail -2 "$O/reader_selftest.txt"
echo "HCKMB bench start $(date -u +%T)Z (timed rows $ROWS in ABBA order + rotation; health rows $HROWS with negative control XM; guard 3300s)"
timeout -s KILL 3300 python3 -B $B --rows $ROWS --health-rows $HROWS --out "$O/hckmb.json" --hlo-dir "$O/hlo" > "$O/hckmb.log" 2>&1
r=$?; echo "HCKMB bench rc=$r end $(date -u +%T)Z"
grep -a "^HCKMB \|^Q38_" "$O/hckmb.log" | cut -c1-600
[ $r = 0 ] || tail -30 "$O/hckmb.log"
python3 -I -B $R "$O" > "$O/hckmb_verdict.txt" 2>&1; rr=$?
echo "HCKMB reader rc=$rr (0 VERDICT, 1 VERDICT VOID, 2 no JSON)"; cut -c1-900 "$O/hckmb_verdict.txt"
echo "HCKMB end $(date -u +%FT%TZ)"
if grep -q "^VERDICT " "$O/hckmb_verdict.txt" && ! grep -q "^VERDICT VOID" "$O/hckmb_verdict.txt"; then
  touch "$O/OK" && echo "HCKMB marker $O/OK written (Q38X_SCRIPT_OK=hckmb/OK)"
  exit 0
fi
echo "HCKMB no marker (VOID / no verdict line)"
exit 1

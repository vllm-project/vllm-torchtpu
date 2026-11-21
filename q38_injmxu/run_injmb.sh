#!/bin/bash
# run_injmb.sh (q38 INJMXU-BUILD, DSTEP lane, 2026-11-21): the INJ-MXU microbench body (BENCH=script on cdk-tpu-exos, one 4-chip v7x
# host, 2x2x1, 8 TensorCores). Bench tests/q38_injmxu/tpu_bench_injmb.py (HC-HIGHROWS-2's tpu_bench_w3.py + counted edits; served
# 2-flag LIBTPU + barrier off pinned, no dvfs) with --rows 288,320,352,384,448 --arms W1,W1F,W1T (W1 = hc_decode_wide mode 1; W1F =
# mode 1 fold=True = INJ-MXU; W1T = the W1 twin; the live negative control W1FM runs in health only), reader q38_injmxu/injmb_read.py.
# Pattern of HC-DECOMP's q38_hcdecomp/run_w1dc.sh (credit): selftests first (launch barrier, bench engagement rules + mutants, reader +
# mutants), then ONE bench process under its own KILL guard (3300 s), then the reader. Marker injmb/OK (and exit 0) iff the reader
# printed a VERDICT that is not VOID (KEEP / NO-GO / BOUND-FAIL are verdicts of record). Outputs under
# $CDK_OUTPUT_DIR/q38f-bench/script/injmb/ (injmb.json / injmb.log, hlo/, injmb_verdict.txt, OK). No unit-test runner (plain python only).
set -u
O="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script/injmb"
mkdir -p "$O"
rm -f "$O/OK"
B=tests/q38_injmxu/tpu_bench_injmb.py
R=q38_injmxu/injmb_read.py
ROWS=288,320,352,384,448; ARMS=W1,W1F,W1T
echo "INJMB start $(date -u +%FT%TZ) host $(hostname) commit $(cat COMMIT 2>/dev/null) inherited LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
SRCH=$(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10)
echo "INJMB tree src $SRCH (want $(cat q38_injmxu/WANT_SRC_INJMB))"
for v in $(env | grep -oE '^TPU_QWEN4_HC_[A-Z0-9_]+'); do echo "INJMB env $v=${!v} (the bench calls the kernels directly; no HC env key is read)"; done
fail_out() { echo "INJMB ABORT: $1"; exit 3; }
[ "$SRCH" = "$(cat q38_injmxu/WANT_SRC_INJMB)" ] || fail_out "tree src is not the q38-injmxu-k2 src"
python3 -B $B --barrier-selftest > "$O/barrier_selftest.txt" 2>&1 || { cat "$O/barrier_selftest.txt"; fail_out "launch-barrier selftest failed"; }
tail -1 "$O/barrier_selftest.txt"
python3 -B $B --selftest --mutants > "$O/bench_selftest.txt" 2>&1 || { cat "$O/bench_selftest.txt"; fail_out "bench selftest failed"; }
grep -E "SELFTEST (PASS|FAIL)|MUTANTS" "$O/bench_selftest.txt"
python3 -I -B $R --selftest --mutants > "$O/reader_selftest.txt" 2>&1 || { cat "$O/reader_selftest.txt"; fail_out "reader selftest failed"; }
tail -2 "$O/reader_selftest.txt"
echo "INJMB bench start $(date -u +%T)Z (arms $ARMS; base + base2 = W1 alone; negative control W1FM in health; rows $ROWS; guard 3300s)"
timeout -s KILL 3300 python3 -B $B --rows $ROWS --arms $ARMS --out "$O/injmb.json" --hlo-dir "$O/hlo" > "$O/injmb.log" 2>&1
r=$?; echo "INJMB bench rc=$r end $(date -u +%T)Z"
grep -a "^HCW \|^Q38_" "$O/injmb.log" | cut -c1-600
[ $r = 0 ] || tail -30 "$O/injmb.log"
python3 -I -B $R "$O" > "$O/injmb_verdict.txt" 2>&1; rr=$?
echo "INJMB reader rc=$rr (0 VERDICT, 1 VERDICT VOID, 2 no JSON)"; cut -c1-900 "$O/injmb_verdict.txt"
echo "INJMB end $(date -u +%FT%TZ)"
if grep -q "^VERDICT " "$O/injmb_verdict.txt" && ! grep -q "^VERDICT VOID" "$O/injmb_verdict.txt"; then
  touch "$O/OK" && echo "INJMB marker $O/OK written (Q38X_SCRIPT_OK=injmb/OK)"
  exit 0
fi
echo "INJMB no marker (VOID / no verdict line)"
exit 1

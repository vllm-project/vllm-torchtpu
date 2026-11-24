#!/bin/bash
# q38 L3 D60 car body (BENCH=script, one 4-chip v7x host, 2x2x1, 8 TensorCores): the fused MoE decode kernel, dense /
# lever 5b cr32 / lever 5c cr32p, at 160 / 192 / 208 / 224 / 256 rows (+ skewed 224s / 256s). Run from the tree root with
# PYTHONPATH=src. Sources q38_l3_d60/q38_script_result.sh (byte copy of scratch/l3/q38_script_result.sh, sha256
# 4ad51e5e) so the job's result.json is honest: ok only when the reader exits 0 with a valid VERDICT line.
# Order: selftests (barrier rule, bench rules + mutants, reader rules + mutants; any failure runs nothing), then ONE bench
# process under its KILL guard, then the pre-registered reader. Outputs under $CDK_OUTPUT_DIR/q38f-bench/script/d60/
# (d60_tc.json / .log, d60_verdict.txt). Exit = the reader's rc (0 KEEP / NOT KEEP / FAIL-EXACT, 1 NO VERDICT,
# 2 VOID / INVALID). No engine is built (script bench).
set -u
O="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script/d60"
mkdir -p "$O"
. q38_l3_d60/q38_script_result.sh
q38_result_clear
B=tests/q38_l3_d60/tpu_bench_moe_compact_pipe.py
R=q38_l3_d60/d60_read.py
echo "L3D60 start $(date -u +%FT%TZ) host $(hostname) commit $(cat COMMIT 2>/dev/null) inherited LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
echo "L3D60 tree src $(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10)"
abort() { echo "L3D60 ABORT: $1"; q38_result_write 0 l3d60 "abort: $1"; exit 3; }
python3 -B $B --barrier-selftest > "$O/barrier_selftest.txt" 2>&1 || { cat "$O/barrier_selftest.txt"; abort "barrier selftest"; }
tail -1 "$O/barrier_selftest.txt"
python3 -B $B --selftest --mutants > "$O/bench_selftest.txt" 2>&1 || { cat "$O/bench_selftest.txt"; abort "bench selftest"; }
grep -a "D60_BENCH_SELFTEST PASS\|D60_BENCH_MUTANTS" "$O/bench_selftest.txt"
python3 -I -B $R --selftest --mutants > "$O/reader_selftest.txt" 2>&1 || { cat "$O/reader_selftest.txt"; abort "reader selftest"; }
grep -a "D60_READ_SELFTEST PASS\|D60_READ_MUTANTS" "$O/reader_selftest.txt"
echo "L3D60 bench start $(date -u +%T)Z (guard 2100s)"
timeout -s KILL 2100 python3 -B $B --cases 160,192,208,224,256,224s,256s --arms dense,cr32,cr32p --layers 48 \
  --iters 20 --out "$O/d60_tc.json" > "$O/d60_tc.log" 2>&1
r=$?; echo "L3D60 bench rc=$r end $(date -u +%T)Z"
grep -a "^D60 \|^Q38_" "$O/d60_tc.log" | cut -c1-700
[ $r = 0 ] || tail -40 "$O/d60_tc.log"
python3 -I -B $R "$O" > "$O/d60_verdict.txt" 2>&1; rr=$?
echo "L3D60 reader rc=$rr (0 KEEP / NOT KEEP / FAIL-EXACT, 1 NO VERDICT, 2 VOID / INVALID)"; cut -c1-1200 "$O/d60_verdict.txt"
q38_result_from_verdict "$rr" "$O/d60_verdict.txt" "l3d60"
echo "L3D60 end $(date -u +%FT%TZ)"
exit $rr

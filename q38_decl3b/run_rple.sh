#!/bin/bash
# q38 DECODE-L3b-RPLEW body (the R form's PLE op, layers W / U; BENCH=script on cdk-tpu-exos, one 4-chip v7x host, 2x2x1,
# 8 TensorCores). Run from the tree root with PYTHONPATH=src. Copied from DECODE-L3-INJMXU-2's q38_decl3/run_injmxu.sh (credit
# DECODE-L3, DECODE-L3-INJMXU(-2); pattern from HCDEC-XLAHUNT's K3DX body) with counted changes: the bench
# (tests/q38_decl3b/tpu_bench_rple.py), the reader (q38_decl3b/rple_read.py), rows 128 / 160 / 192 / 256 (+ the non-uniform 160
# row, part of the bench plan) and arms A / W / WU, the output dir, the pool.py pin (this branch's file). On cdk-tpu-exos the
# bench sibling's Q38X_SCRIPT_OK=rple/OK writes run01_A/result.json ok=true iff this body exits 0 and the marker exists (so this
# body writes no result.json). Selftests first (launch barrier, bench helpers + mutants, reader + mutants; any failure runs
# nothing), then ONE bench process under its own KILL guard (3300 s), then the reader. Marker rple/OK (and exit 0) iff the reader
# printed a VERDICT that is not VOID. Outputs under $CDK_OUTPUT_DIR/q38f-bench/script/rple/ (rple.json / .log, rple_verdict.txt, OK).
set -u
O="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script/rple"
mkdir -p "$O"
rm -f "$O/OK"
B=tests/q38_decl3b/tpu_bench_rple.py
PLP=src/vllm_torchtpu/kernels/qwen4_exp/pool.py
PLP_SHA=$(cat q38_decl3b/POOL_SHA256_RPLE)
echo "RPLE start $(date -u +%FT%TZ) host $(hostname) commit $(cat COMMIT 2>/dev/null) inherited LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
echo "RPLE tree src $(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10) (want $(cat q38_decl3b/WANT_SRC_RPLE))"
fail_out() { echo "RPLE ABORT: $1"; exit 3; }
[ "$(sha256sum < $PLP | cut -c1-64)" = "$PLP_SHA" ] || fail_out "pool.py is not the q38-decl3b-rple file (sha256 ${PLP_SHA:0:8})"
for k in VLLM_TPU_PLE_WRDEFER VLLM_TPU_PLE_UNIFORM_COND VLLM_TPU_PLE_SKIP_IDLE_ROWS; do
  [ -z "${!k:-}" ] || fail_out "$k is set in the pod env (${!k}); the bench sets the arm flags itself"
done
python3 -B $B --barrier-selftest > "$O/barrier_selftest.txt" 2>&1 || { cat "$O/barrier_selftest.txt"; fail_out "launch-barrier selftest failed"; }
tail -1 "$O/barrier_selftest.txt"
python3 -B $B --selftest --mutants > "$O/bench_selftest.txt" 2>&1 || { cat "$O/bench_selftest.txt"; fail_out "bench selftest failed"; }
grep -E "SELFTEST (PASS|FAIL)|MUTANTS" "$O/bench_selftest.txt"
python3 -I -B q38_decl3b/rple_read.py --selftest --mutants > "$O/reader_selftest.txt" 2>&1 \
  || { cat "$O/reader_selftest.txt"; fail_out "RPLE reader selftest failed"; }
tail -2 "$O/reader_selftest.txt"
echo "RPLE bench start $(date -u +%T)Z (arms A / W / WU, base + base2 = A alone, rows 128 / 160 / 192 / 256 + 160nu; guard 3300s)"
timeout -s KILL 3300 python3 -B $B --rows 128,160,192,256 --out "$O/rple.json" > "$O/rple.log" 2>&1
r=$?; echo "RPLE bench rc=$r end $(date -u +%T)Z"
grep -a "^RPLE \|^Q38_" "$O/rple.log" | cut -c1-900
[ $r = 0 ] || tail -30 "$O/rple.log"
python3 -I -B q38_decl3b/rple_read.py "$O" > "$O/rple_verdict.txt" 2>&1; rr=$?
echo "RPLE reader rc=$rr (0 VERDICT KEEP / NOT-KEEP, 1 VERDICT VOID, 2 no JSON)"; cut -c1-900 "$O/rple_verdict.txt"
echo "RPLE end $(date -u +%FT%TZ)"
if grep -q "^VERDICT " "$O/rple_verdict.txt" && ! grep -q "^VERDICT VOID" "$O/rple_verdict.txt"; then
  touch "$O/OK" && echo "RPLE marker $O/OK written (Q38X_SCRIPT_OK=rple/OK)"
  exit 0
fi
echo "RPLE no marker (VOID / no verdict line)"
exit 1

#!/bin/bash
# q38 H8-FIX CPU proof body (CDK parasim-cpu job; run from the unpacked tree root). Exit code = verdict.
# Legs:
#   P0 versions, src hash + md5 of the fixed file at start (must equal EXPECT_SRC below)
#   P1 the red/green test tests/arfix/test_tp_ar_decode_jaxop_signature.py: 5 passed, 0 skipped (the arm-path test needs vLLM)
#   P2 existing AR-FIX decode-AR wiring + math tests (flag-off regression; -rs lists any skip)
#   P3 H8 kernel tests (unchanged kernel)
#   P4 annotation scans: L1's q38_l1_annscan.py = exactly the known online_fp8.py:138 PEP 695 false positive;
#      H8-FIX's extended scan = the pinned expected set (no new hit; no tp8_ar_decode_fn hit)
#   P5 mutants on a COPY of the tree (clean control must pass; M1 = 5d368f166's file bytes (the TPU bug), M2 = annotations
#      dropped, M3 = module import hidden under TYPE_CHECKING: each must fail P1; M1 must show the NameError)
#   P7 traced-program identity: jaxpr of global_fn (fixed) == 5d368f166's, both wires, kernel + psum rows, + a sensitivity control
#   P8 torch_tpu's REAL jax_op on the fixed fn (no NameError, return maps to Tensor) and on 5d368f166's fn (reports whether
#      the TPU NameError reproduces through the real entry point); INFO-only when torch_tpu is absent
#   P6 src hash + md5 at end == start == EXPECT_SRC (the run tree was never edited)
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH="$PWD/src"
EXPECT_SRC=e0671f901a
F=src/vllm_torchtpu/kernels/collectives/tp8_ar_decode_fn.py
fail=0
srch() { find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10; }
S0=$(srch); M0=$(md5sum < $F | cut -c1-32)
echo "P0 $(date -u +%FT%TZ) python $(python3 -c 'import sys;print(sys.version.split()[0])') jax $(python3 -c 'import jax;print(jax.__version__)' 2>&1 | tail -1) torch $(python3 -c 'import torch;print(torch.__version__)' 2>&1 | tail -1) vllm $(python3 -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1) torch_tpu $(python3 -c 'import importlib.util as u;print(bool(u.find_spec("torch_tpu")))' 2>&1 | tail -1)"
echo "P0 src $S0 (expect $EXPECT_SRC) commit $(cat COMMIT 2>/dev/null) md5 tp8_ar_decode_fn.py $M0"
[ "$S0" = "$EXPECT_SRC" ] || { echo "P0 FAIL src $S0 != $EXPECT_SRC"; fail=1; }
leg() { local n=$1; shift; echo "== $n: $*"; "$@" > /tmp/leg_$n.log 2>&1; local rc=$?; tail -12 /tmp/leg_$n.log; echo "$n rc=$rc"; [ $rc = 0 ] || fail=1; }
PYT="python3 -m pytest -q -p no:cacheprovider --noconftest -rs"
leg P1 $PYT tests/arfix/test_tp_ar_decode_jaxop_signature.py
P1L=$(grep -E '[0-9]+ (passed|failed|skipped|errors?)' /tmp/leg_P1.log | tail -1); echo "P1 summary: $P1L"
if echo "$P1L" | grep -qE '^=*\s*5 passed' && ! echo "$P1L" | grep -qE '[0-9]+ (failed|skipped|errors?)'; then echo "P1 OK 5 passed, 0 skipped"; else echo "P1 FAIL (want exactly 5 passed, no skip / fail / error)"; fail=1; fi
leg P2 $PYT tests/arfix/test_tp_ar_decode_wiring.py tests/arfix/test_tp_ar_decode_math.py
leg P3 $PYT tests/arfix/test_tp8_hier_all_reduce.py
echo "== P4 annotation scans"
python3 -I tests/q38_h8fix/q38_l1_annscan.py src > /tmp/scan_l1.txt 2>&1; cat /tmp/scan_l1.txt
if [ "$(grep -c '^HIT' /tmp/scan_l1.txt)" = 1 ] && grep -q '^HIT vllm_torchtpu/layers/adapter/quantization/online_fp8.py:138 def attach_online_fp8' /tmp/scan_l1.txt; then echo "P4 L1 OK (only the known PEP 695 false positive)"; else echo "P4 L1 FAIL"; fail=1; fi
python3 -I tests/q38_h8fix/q38_h8fix_annscan2.py src 2>&1 | LC_ALL=C sort > /tmp/scan2.txt
LC_ALL=C sort tests/q38_h8fix/fixtures/annscan2_expected_sorted.txt > /tmp/scan2_exp.txt
tail -n 3 /tmp/scan2.txt | grep SUMMARY
if diff /tmp/scan2_exp.txt /tmp/scan2.txt > /tmp/scan2.diff && ! grep -q 'tp8_ar_decode_fn' /tmp/scan2.txt; then echo "P4 scan2 OK (== pinned set, no tp8_ar_decode_fn hit)"; else echo "P4 scan2 FAIL"; cat /tmp/scan2.diff; fail=1; fi
leg P5 python3 tests/q38_h8fix/q38_h8fix_mutrun.py
cat /tmp/leg_P5.log
leg P7 python3 tests/q38_h8fix/q38_h8fix_jaxpr_identity.py
cat /tmp/leg_P7.log
leg P8 python3 tests/q38_h8fix/q38_h8fix_real_jaxop.py
cat /tmp/leg_P8.log
S1=$(srch); M1=$(md5sum < $F | cut -c1-32)
echo "P6 src start $S0 end $S1; md5 start $M0 end $M1"
[ "$S1" = "$S0" ] && [ "$M1" = "$M0" ] && [ "$S1" = "$EXPECT_SRC" ] || { echo "P6 FAIL the run tree changed"; fail=1; }
echo "Q38_H8FIX_SUMMARY $( [ $fail = 0 ] && echo PASS || echo FAIL )"
exit $fail

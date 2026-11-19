#!/bin/bash
# q38 L4 EAGER-FIX CPU proof (CDK CPU pod; run from the unpacked tree root, base tree ee289379f in ./q38_base).
# P0 versions + src hash; P1 tests/runner/test_q38_eagerfix.py (0 skipped); P2 compile-key registration (0 skipped);
# P3 regression HEAD vs BASE (every failure named; none new on HEAD); P4 mutants; P5 src unchanged.
# Last line Q38_EAGERFIX_SUMMARY PASS|FAIL; exit code = verdict.
set -u
export JAX_PLATFORMS=cpu PYTHONPATH=src
unset VLLM_TPU_SPEC_TAIL_REJ_PAD VLLM_TPU_SAMPLING_COPY_FULL VLLM_TPU_EAGER_SHAPE_LOG VLLM_TPU_VPS_WARM_DRAWS
PYT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -E '^[0-9]+ (passed|failed)|^=+ .*(passed|failed|error)' "$1" | tail -1; }
count() { summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' || echo 0; }

H=$(srch .)
echo "P0 $(date -u +%FT%TZ) $(python3 -c 'import sys, torch, importlib.util as u, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"), "torch_tpu", u.find_spec("torch_tpu") is not None)')"
echo "P0 src $H (expect $(cat tests/q38_eagerfix/EXPECT_SRC)) commit $(cat tests/q38_eagerfix/EXPECT_COMMIT) base_src $(srch q38_base) (expect d81237a81e)"
[ "$H" = "$(cat tests/q38_eagerfix/EXPECT_SRC)" ] && [ "$(srch q38_base)" = d81237a81e ] || FAIL="$FAIL P0-src"

echo "== P1 EAGER-FIX tests"
$PYT tests/runner/test_q38_eagerfix.py > /tmp/ef_p1.log 2>&1; rc=$?
grep -E '^(PASSED|FAILED|ERROR|SKIPPED)|^E |Error|passed|failed' /tmp/ef_p1.log | grep -v DeprecationWarning | tail -40
echo "P1 rc=$rc summary: $(summ /tmp/ef_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/ef_p1.log skipped)" = 0 ] && [ "$(count /tmp/ef_p1.log passed)" -gt 0 ] || FAIL="$FAIL P1"

echo "== P2 compile-key registration"
$PYT tests/compilation/test_tpu_compile_key_registration.py > /tmp/ef_p2.log 2>&1; rc=$?
grep -E '^(FAILED|ERROR)' /tmp/ef_p2.log | head
echo "P2 rc=$rc summary: $(summ /tmp/ef_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/ef_p2.log skipped)" = 0 ] || FAIL="$FAIL P2"

echo "== P3 regression HEAD vs BASE (ee289379f)"
REG=$(tr "\n" " " < tests/q38_eagerfix/p3_tests.txt)
echo "P3 files: $(wc -l < tests/q38_eagerfix/p3_tests.txt | tr -d ' ') (every test file naming the runner / worker / VPS verify, tests/compilation, tests/decfb2, patch registry; no entrypoints / kernels)"
timeout 2700 $PYT $REG > /tmp/ef_p3h.log 2>&1; rh=$?
(cd q38_base && PYTHONPATH=src timeout 2700 $PYT $REG > /tmp/ef_p3b.log 2>&1); rb=$?
[ $rh = 124 ] || [ $rb = 124 ] && { echo "P3 TIMEOUT head=$rh base=$rb"; FAIL="$FAIL P3-timeout"; }
echo "P3 HEAD rc=$rh $(summ /tmp/ef_p3h.log)"
echo "P3 BASE rc=$rb $(summ /tmp/ef_p3b.log)"
grep -E '^FAILED|^ERROR' /tmp/ef_p3h.log | sed 's/ - .*//' | sort > /tmp/ef_p3h.f
grep -E '^FAILED|^ERROR' /tmp/ef_p3b.log | sed 's/ - .*//' | sort > /tmp/ef_p3b.f
NEW=$(comm -23 /tmp/ef_p3h.f /tmp/ef_p3b.f)
# failed + errors: both are named FAILED / ERROR lines (i2 compared failed alone to all named lines)
NH=$(( $(count /tmp/ef_p3h.log failed) + $(count /tmp/ef_p3h.log error) ))
NB=$(( $(count /tmp/ef_p3b.log failed) + $(count /tmp/ef_p3b.log error) ))
echo "P3 failed+errors HEAD $NH (named $(wc -l < /tmp/ef_p3h.f | tr -d ' ')) BASE $NB (named $(wc -l < /tmp/ef_p3b.f | tr -d ' ')) new-on-HEAD: ${NEW:-none}"
sed 's/^/P3 HEAD /' /tmp/ef_p3h.f; sed 's/^/P3 BASE /' /tmp/ef_p3b.f
[ -z "$NEW" ] && [ "$NH" = "$(wc -l < /tmp/ef_p3h.f | tr -d ' ')" ] && [ "$(count /tmp/ef_p3h.log passed)" -gt 0 ] || FAIL="$FAIL P3"

echo "== P4 mutants"
python3 tests/q38_eagerfix/q38_eagerfix_mutrun.py . 2>&1 | grep -E '^CONTROL|^MUT|^Q38_EAGERFIX_MUT'; rc=${PIPESTATUS[0]}
echo "P4 rc=$rc"
[ $rc = 0 ] || FAIL="$FAIL P4"

H2=$(srch .)
echo "P5 src after $H2 (before $H)"
[ "$H2" = "$H" ] || FAIL="$FAIL P5"
if [ -z "$FAIL" ]; then echo "Q38_EAGERFIX_SUMMARY PASS"; exit 0; fi
echo "Q38_EAGERFIX_SUMMARY FAIL:$FAIL"; exit 1

#!/bin/bash
# q38 L6 lever ptrims (H0-2 PREP-TRIMS) CPU harness (CDK CPU pod only; never the Mac or the VM). Modelled on
# L6-ENGINE's tests/q38_l6/run_cpu_engalloc.sh.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh extra/model_config.json):
#   bash head/tests/q38_l6/run_cpu_ptrims.sh bench|dev|full|mut
# P0  versions; head / base src hashes (EXPECT_PTRIMS_SRC / _BASE_SRC); the exact set of src files that differ.
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base);
#     its mutant (expectation pointed at base while importing head) must FAIL.
# P1  tests/q38_l6/test_l6_ptrims.py: exactly P1_WANT passed, 0 failed, 0 skipped.
# P2  compile-key registration rows of the two ptrims flags: exactly 5 passed (2 parametrized tests x 2 flags +
#     the all-together test).
# B   bench: served plan, us per _prepare_inputs call at N = 4 / 9 / 11 / 25 / 52 running decode requests, arms
#     off / sc (STEADY + CARRY) / sce (+ envcache) / pt (+ ptrims) / ab (ptrims in step windows, paired), plus the
#     per-candidate fragments (BENCH_DONE ok=True).
# (bench stops after B; dev stops after a mutant survey)
# P2b collect-only census (head / base) of the regression files.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error /
#     a new failing id on head = FAIL).
# P4  mutants (q38_l6_ptrims_mutrun.py ptrims: control + every mutant must FAIL its target).
# P5  head src unchanged by the run.
# Last line: Q38_L6_PTRIMS_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=25
P2_WANT=5
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_ptrims_mutrun as m; print(len(m.MUTANTS['ptrims']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
finish() { [ -z "$FAIL" ] && { echo "Q38_L6_PTRIMS_SUMMARY PASS mode=$MODE"; exit 0; }; echo "Q38_L6_PTRIMS_SUMMARY FAIL mode=$MODE:$FAIL"; exit 1; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_PTRIMS_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_PTRIMS_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_PTRIMS_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_PTRIMS_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_PTRIMS_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

if [ "$MODE" = bench ]; then
  echo "== B bench (served plan)"
  (cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_ptrims.py --config "$W/extra/model_config.json" --fragments) > /tmp/l6pt_bench.log 2>&1; rc=$?
  grep -aE '^BENCH' /tmp/l6pt_bench.log
  echo "B rc=$rc"
  [ $rc = 0 ] && grep -aq '^BENCH_DONE ok=True' /tmp/l6pt_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6pt_bench.log; }
  finish
fi

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_ptrims.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6pt_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6pt_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6pt_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6pt_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_ptrims.py"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT -s --durations=8 tests/q38_l6/test_l6_ptrims.py) > /tmp/l6pt_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6pt_p1.log | head -60
grep -aE 's (call|setup) ' /tmp/l6pt_p1.log | head -8
echo "P1 rc=$rc summary: $(summ /tmp/l6pt_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6pt_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6pt_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6pt_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6pt_p1.log; }

if [ "$MODE" = mut ]; then
  echo "== P4 mutants"
  (cd "$H" && python3 tests/q38_l6/q38_l6_ptrims_mutrun.py . ptrims) > /tmp/l6pt_p4.log 2>&1; rc=$?
  grep -aE '^CONTROL|^MUT|^Q38_L6_PTRIMS_MUT' /tmp/l6pt_p4.log
  echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
  [ $rc = 0 ] && grep -aq "^Q38_L6_PTRIMS_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6pt_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/l6pt_p4.log; }
  finish
fi

echo "== P2 compile-key registration rows of the ptrims flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_PREP_TRIMS or together") > /tmp/l6pt_p2.log 2>&1; rc=$?
tail -4 /tmp/l6pt_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6pt_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6pt_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6pt_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

echo "== B bench (served plan)"
(cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_ptrims.py --config "$W/extra/model_config.json" --fragments) > /tmp/l6pt_bench.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6pt_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE ok=True' /tmp/l6pt_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6pt_bench.log; }

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3000 python3 tests/q38_l6/q38_l6_ptrims_mutrun.py . ptrims survey) > /tmp/l6pt_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_PTRIMS_MUT' /tmp/l6pt_p4.log
  finish
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py
tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_engine_alloc.py
tests/runner/test_mtp_async_contracts.py tests/runner/test_tpu_runner_step_seq_lens.py
tests/runner/test_q38_fused_prep_functional.py tests/runner/test_tpu_runner.py
tests/runner/test_dcp_runner_integration.py tests/runner/test_pcp_runner_integration.py
tests/runner/test_qwen4_exp_ple_host_ids.py tests/runner/test_qwen4_exp_inputs.py
tests/layers/core/test_attention_metadata.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py"
echo "== P2b collect-only census (head / base)"
RUN=""
for f in $FILES; do
  nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  echo "P2b $f head ${nh:-0} base ${nb:-0}"
  if [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then
    case " $KEY " in *" $f "*) echo "P2b KEY file collects nothing: $f"; FAIL="$FAIL P2b";; *) echo "P2b EXCLUDED (0 on both) $f";; esac
  else RUN="$RUN $f"; fi
done

echo "== P3 per-file regression HEAD vs BASE"
[ "$(sha256sum < $W/q38_l3_regress_perfile.sh | cut -c1-8)" = 4fb6f596 ] || FAIL="$FAIL P3-helper"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6pt "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6_ptrims_mutrun.py . ptrims) > /tmp/l6pt_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_PTRIMS_MUT' /tmp/l6pt_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_PTRIMS_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6pt_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6pt_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"
finish

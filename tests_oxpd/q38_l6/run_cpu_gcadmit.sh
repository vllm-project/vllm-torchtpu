#!/bin/bash
# q38 L6-GCADMIT CPU proof of the GC-ADMIT-FREEZE lever gcadmit (core/l6_gcadmit.py + its execute_model hook)
# (CDK CPU pod only; never the Mac or the VM). Modelled on L6-SYNC's tests/q38_l6/run_cpu_sync.sh.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_gcadmit.sh dev|full|soak
# P0  versions; head / base src hashes (EXPECT_GCADMIT_SRC / EXPECT_GCADMIT_BASE_SRC); the exact set of src files that
#     differ (EXPECT_GCADMIT_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base); its
#     mutant (expectation pointed at base while importing head) must FAIL exactly 1 test.
# P1  tests/q38_l6/test_l6_gcadmit.py: exactly 26 passed, 0 failed, 0 skipped (the import-guard test runs; 16 gcadmit + 10 gccensus).
# P2  compile-key registration rows of the three new flags: exactly 13 passed (2 parametrized tests x 6 + together).
# P6  lever-off identity: tests/q38_l6/l6_carry_identity.py on head and on base, L6CARRY_IDENTITY_ALL equal
#     (only where EXPECT_GCADMIT_P6 says 'carry'; a tree without the carry harness says 'none' and P6 is skipped
#     by that statement, not silently).
# (dev: then a mutant survey, stop)
# P2b collect-only census (head / base) of the regression files.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error /
#     a new failing id on head = FAIL).
# P4  mutants (q38_l6_gcadmit_mutrun.py gcadmit: control + every mutant must FAIL its target).
# P5  head src unchanged by the run.
# soak mode (tree-independent; run once): S leak soak (tests/q38_l6/soak_gcadmit.py, pre-stated bounds inside, SOAK_DONE
#     rc=0) and B gen-2 bench (tests/q38_l6/bench_gcadmit_gen2.py, arms a / b / c, BENCH_GCADMIT_DONE rc=0); P0 / P0b
#     first.
# Last line: Q38_L6_GCADMIT_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z0-9_]+'); do unset "$v"; done
unset VLLM_TPU_HOST_GC VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_CPU_PIN 2>/dev/null
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=26
P2_WANT=13
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_gcadmit_mutrun as m; print(len(m.MUTANTS['gcadmit']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_GCADMIT_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_GCADMIT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_GCADMIT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_GCADMIT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_GCADMIT_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_gcadmit.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6g_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6g_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6g_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6g_p0b.log; }

if [ "$MODE" = soak ]; then
  echo "== S leak soak (pre-stated bounds in tests/q38_l6/soak_gcadmit.py)"
  (cd "$H" && env $CPUENV timeout -s KILL 5400 python3 tests/q38_l6/soak_gcadmit.py) > /tmp/l6g_soak.log 2>&1; rc=$?
  grep -aE '^SOAK' /tmp/l6g_soak.log
  echo "S rc=$rc"
  [ $rc = 0 ] && grep -aq '^SOAK_DONE rc=0' /tmp/l6g_soak.log || { FAIL="$FAIL S"; tail -60 /tmp/l6g_soak.log; }
  echo "== B gen-2 walk, arms a (base) / b (gcadmit) / c (untracked arrays), about 10^7 references"
  (cd "$H" && env $CPUENV timeout -s KILL 5400 python3 tests/q38_l6/bench_gcadmit_gen2.py) > /tmp/l6g_bench.log 2>&1; rc=$?
  grep -aE '^BENCH' /tmp/l6g_bench.log
  echo "B rc=$rc"
  [ $rc = 0 ] && grep -aq '^BENCH_GCADMIT_DONE rc=0' /tmp/l6g_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6g_bench.log; }
  HS2=$(srch "$H")
  echo "P5 head src after $HS2 (before $HS)"
  [ "$HS2" = "$HS" ] || FAIL="$FAIL P5"
  [ -z "$FAIL" ] && { echo "Q38_L6_GCADMIT_SUMMARY PASS mode=soak"; exit 0; }
  echo "Q38_L6_GCADMIT_SUMMARY FAIL mode=soak:$FAIL"; exit 1
fi

echo "== P1 test_l6_gcadmit.py"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT --durations=8 tests/q38_l6/test_l6_gcadmit.py tests/q38_l6/test_l6_gccensus.py) > /tmp/l6g_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6g_p1.log | head -60
grep -aE 's (call|setup) ' /tmp/l6g_p1.log | head -8
echo "P1 rc=$rc summary: $(summ /tmp/l6g_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6g_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6g_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6g_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6g_p1.log; }

echo "== P2 compile-key registration rows of the three new flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_GC_ADMIT or L6_GC_CENSUS or together") > /tmp/l6g_p2.log 2>&1; rc=$?
tail -4 /tmp/l6g_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6g_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6g_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6g_p2.log failed)" = 0 ] || { FAIL="$FAIL P2"; tail -40 /tmp/l6g_p2.log; }

P6=$(cat $H/tests/q38_l6/EXPECT_GCADMIT_P6)
echo "== P6 lever-off identity ($P6)"
if [ "$P6" = carry ]; then
  ih=$(cd "$H" && env $CPUENV timeout -s KILL 1500 python3 tests/q38_l6/l6_carry_identity.py 2>/tmp/l6g_p6h.err | tee /tmp/l6g_p6h.log | grep -a '^L6CARRY_IDENTITY_ALL')
  ib=$(cd "$B" && env $CPUENV timeout -s KILL 1500 python3 tests/q38_l6/l6_carry_identity.py 2>/tmp/l6g_p6b.err | tee /tmp/l6g_p6b.log | grep -a '^L6CARRY_IDENTITY_ALL')
  grep -a '^L6CARRY_IDENTITY_FILE' /tmp/l6g_p6h.log /tmp/l6g_p6b.log
  echo "P6 head $ih"; echo "P6 base $ib"
  [ -n "$ih" ] && [ "$ih" = "$ib" ] && grep -aq "L6CARRY_IDENTITY_FILE $(realpath $H)/src/" /tmp/l6g_p6h.log \
    && grep -aq "L6CARRY_IDENTITY_FILE $(realpath $B)/src/" /tmp/l6g_p6b.log || { FAIL="$FAIL P6"; tail -20 /tmp/l6g_p6h.err /tmp/l6g_p6b.err; }
elif [ "$P6" = none ]; then
  echo "P6 not run: EXPECT_GCADMIT_P6=none (this tree has no carry harness; the flag-off evidence is P1's execute_model tests)"
else
  echo "P6 bad EXPECT_GCADMIT_P6 '$P6'"; FAIL="$FAIL P6"
fi

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3000 python3 tests/q38_l6/q38_l6_gcadmit_mutrun.py . gcadmit survey) > /tmp/l6g_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_GCADMIT_MUT' /tmp/l6g_p4.log
  [ -z "$FAIL" ] && { echo "Q38_L6_GCADMIT_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_GCADMIT_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/compilation/test_additional_config_hash_patch.py
tests/test_envs.py tests/test_envs_builders.py tests/worker/test_tpu_worker.py tests/core/test_host_knobs.py
tests/runner/test_tpu_runner.py tests/runner/test_tpu_runner_kv_connector_lifecycle.py tests/runner/test_tpu_runner_pp.py
tests/runner/test_tpu_runner_preemption.py tests/runner/test_pp_last_stage_output.py tests/runner/test_host_phases.py
tests/runner/test_prompt_tail.py tests/test_prompt_ids_ndarray_pd.py tests/q38mm_dm/test_prompt_ids_ndarray.py
tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_jitter.py tests/q38_l6/test_l6_ab_envcache.py
tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_steady.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_defer.py tests/q38_l6/test_l6_fctx.py
tests/q38_l6/test_l6_eos_canary.py tests/core/test_gc_slow.py tests/core/test_engine_phases.py
tests/platforms/test_patch_registry.py tests/test_package_import.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/runner/test_tpu_runner.py tests/worker/test_tpu_worker.py
tests/q38_l6/test_l6_engine_alloc.py"
echo "== P2b collect-only census (head / base)"
RUN=""
for f in $FILES; do
  nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  echo "P2b $f head ${nh:-0} base ${nb:-0}"
  if [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then
    case " $(echo $KEY) " in *" $f "*) echo "P2b KEY file collects nothing: $f"; FAIL="$FAIL P2b";; *) echo "P2b EXCLUDED (0 on both) $f";; esac
  else RUN="$RUN $f"; fi
done

echo "== P3 per-file regression HEAD vs BASE"
[ "$(sha256sum < $W/q38_l3_regress_perfile.sh | cut -c1-8)" = 4fb6f596 ] || FAIL="$FAIL P3-helper"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6g "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6_gcadmit_mutrun.py . gcadmit) > /tmp/l6g_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_GCADMIT_MUT' /tmp/l6g_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_GCADMIT_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6g_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/l6g_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_GCADMIT_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_GCADMIT_SUMMARY FAIL mode=full:$FAIL"; exit 1

#!/bin/bash
# q38 L6 lever adcarry CPU proof (CDK CPU pod only; never the Mac or the VM). Modelled on L6-ENGINE's
# tests/q38_l6/run_cpu_engalloc.sh.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_adc.sh dev|full
# P0  versions; head / base src hashes (EXPECT_ADC_SRC / EXPECT_ADC_BASE_SRC); the exact set of src files that differ.
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base);
#     its mutant (expectation pointed at base while importing head) must FAIL.
# P1  tests/q38_l6/test_l6_adapt_carry.py: exactly P1_WANT passed, 0 failed, 0 skipped.
# P1b tests/q38_l6/test_l6_decode_carry.py + test_l6_steady.py on head (the CARRY / STEADY suites the lever hooks
#     into): exactly P1B_WANT passed, 0 failed, 0 skipped.
# P2  compile-key registration rows of the two adcarry flags: exactly 5 passed (2 parametrized tests x 2 flags +
#     the all-together test).
# P6  lever-off identity: l6_carry_identity.py and l6_adc_identity.py print the same *_IDENTITY_ALL digest on head
#     and on base (each importing its own tree).
# B   bench: host us per step at N = 4 / 9 / 11 / 25 / 52 requests, sc vs sca arms; identity must pass.
# (dev stops here, after a mutant survey)
# P2b collect-only census (head / base) of the regression files.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error /
#     a new failing id on head = FAIL).
# P4  mutants (q38_l6_adc_mutrun.py adc: control + every mutant must FAIL its target).
# P5  head src unchanged by the run.
# Last line: Q38_L6_ADC_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
unset VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_D1N_AB_PERIOD_MS Q38_L6ADC_EXPECT_ROOT Q38_L6CARRY_EXPECT_ROOT Q38_L6_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=$(cat $H/tests/q38_l6/EXPECT_ADC_P1)
P1B_WANT=$(cat $H/tests/q38_l6/EXPECT_ADC_P1B)
P2_WANT=5
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_adc_mutrun as m; print(len(m.MUTANTS['adc']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_ADC_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_ADC_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_ADC_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_ADC_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_ADC_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6ADC_EXPECT_ROOT="$(realpath $B)/src" $PT tests/q38_l6/test_l6_adapt_carry.py -k test_imports_the_staged_tree) > /tmp/l6adc_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6adc_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6adc_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6adc_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_adapt_carry.py"
(cd "$H" && env $CPUENV Q38_L6ADC_EXPECT_ROOT="$(realpath $H)/src" $PT -s --durations=10 tests/q38_l6/test_l6_adapt_carry.py) > /tmp/l6adc_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |^STREAM |^ENGAGEMENT |Error|assert' /tmp/l6adc_p1.log | head -80
grep -aE 's (call|setup) ' /tmp/l6adc_p1.log | head -10
echo "P1 rc=$rc summary: $(summ /tmp/l6adc_p1.log) (want $P1_WANT passed)"
[ $rc = 0 ] && [ "$(count /tmp/l6adc_p1.log passed)" = "$P1_WANT" ] && [ "$(count /tmp/l6adc_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6adc_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6adc_p1.log; }

echo "== P1b tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py (head)"
(cd "$H" && env $CPUENV Q38_L6CARRY_EXPECT_ROOT="$(realpath $H)/src" Q38_L6_EXPECT_TREE="$(realpath $H)" $PT tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py) > /tmp/l6adc_p1b.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) ' /tmp/l6adc_p1b.log | head -40
echo "P1b rc=$rc summary: $(summ /tmp/l6adc_p1b.log) (want $P1B_WANT passed)"
[ $rc = 0 ] && [ "$(count /tmp/l6adc_p1b.log passed)" = "$P1B_WANT" ] && [ "$(count /tmp/l6adc_p1b.log failed)" = 0 ] \
  && [ "$(count /tmp/l6adc_p1b.log skipped)" = 0 ] || { FAIL="$FAIL P1b"; tail -80 /tmp/l6adc_p1b.log; }

echo "== P2 compile-key registration rows of the adcarry flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_ADAPT_CARRY or together") > /tmp/l6adc_p2.log 2>&1; rc=$?
tail -4 /tmp/l6adc_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6adc_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6adc_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6adc_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

echo "== P6 lever-off identity (head vs base)"
for s in l6_carry_identity l6_adc_identity; do
  (cd "$H" && env $CPUENV timeout -s KILL 1800 python3 $H/tests/q38_l6/$s.py) > /tmp/l6adc_${s}_h.log 2>&1; rh=$?
  (cd "$B" && env $CPUENV timeout -s KILL 1800 python3 $H/tests/q38_l6/$s.py) > /tmp/l6adc_${s}_b.log 2>&1; rb=$?
  ah=$(grep -a '_IDENTITY_ALL ' /tmp/l6adc_${s}_h.log | awk '{print $2}'); ab=$(grep -a '_IDENTITY_ALL ' /tmp/l6adc_${s}_b.log | awk '{print $2}')
  fh=$(grep -a '_IDENTITY_FILE ' /tmp/l6adc_${s}_h.log | awk '{print $2}'); fb=$(grep -a '_IDENTITY_FILE ' /tmp/l6adc_${s}_b.log | awk '{print $2}')
  echo "P6 $s head rc=$rh all=$ah file=$fh"
  echo "P6 $s base rc=$rb all=$ab file=$fb"
  grep -a '_IDENTITY ' /tmp/l6adc_${s}_h.log | sed 's/^/P6 head /' | head -12
  grep -a '_IDENTITY ' /tmp/l6adc_${s}_b.log | sed 's/^/P6 base /' | head -12
  case "$fh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P6-$s-file";; esac
  case "$fb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P6-$s-file";; esac
  [ $rh = 0 ] && [ $rb = 0 ] && [ -n "$ah" ] && [ "$ah" = "$ab" ] && echo "P6 $s identity EQUAL" \
    || { echo "P6 $s identity DIFFERS"; FAIL="$FAIL P6-$s"; tail -30 /tmp/l6adc_${s}_h.log /tmp/l6adc_${s}_b.log; }
done

echo "== B bench (served plan, host us per step)"
(cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_adc.py) > /tmp/l6adc_bench.log 2>&1; rc=$?
grep -aE '^(BENCH)' /tmp/l6adc_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6adc_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6adc_bench.log; }

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 5400 python3 tests/q38_l6/q38_l6_adc_mutrun.py . adc survey) > /tmp/l6adc_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_ADC_MUT' /tmp/l6adc_p4.log
  grep -aE -A12 'SURVIVED' /tmp/l6adc_p4.log | head -150
  [ -z "$FAIL" ] && { echo "Q38_L6_ADC_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_ADC_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py
tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py
tests/q38_l6/test_l6_defer.py tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_eos_canary.py
tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_jitter.py tests/q38_l6/test_l6_lockstep.py
tests/q38_l6/test_l6_sab_compose.py tests/runner/test_postfwd_fused.py tests/runner/test_postfwd_runner_wiring.py
tests/runner/test_postfwd_static_args.py tests/runner/test_q38_fused_prep_functional.py
tests/runner/test_qwen4_exp_inputs.py tests/runner/test_qwen4_exp_ple_host_ids.py tests/runner/test_tpu_runner.py
tests/runner/test_q38_eagerfix.py tests/runner/test_spec_tail_port.py tests/runner/test_draft_carries_port.py
tests/test_q38_mmhost_engagement.py tests/megakernel_runtime/test_served_ab.py tests/models/vllm/test_qwen4_exp_mtp.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/q38_l6/test_l6_ab_envcache.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py"
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
regress_perfile l6adc "$H" "$B" 1800 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && timeout -s KILL 7200 python3 tests/q38_l6/q38_l6_adc_mutrun.py . adc) > /tmp/l6adc_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_ADC_MUT' /tmp/l6adc_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_ADC_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6adc_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/l6adc_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_ADC_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_ADC_SUMMARY FAIL mode=full:$FAIL"; exit 1

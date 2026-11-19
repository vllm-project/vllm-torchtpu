#!/bin/bash
# q38 L6-H2 overflow unit l6_h2_cheap CPU proof (CDK CPU pod only; never the Mac or the VM): levers fskip (FENCE-SKIP),
# toklists (TOKEN-LISTS-ONCE, port of IX a46d91d20's async-output part, Jordan Dotzel) and aotskip (CF-2 AOTSKIP-AB).
# Modelled on tests/q38_l6/run_cpu_engalloc.sh. Usage (from the unpacked bundle dir holding head/ base/
# q38_l3_regress_perfile.sh): bash head/tests/q38_l6/run_cpu_h2cheap.sh dev|full
# P0  versions; head / base src hashes (EXPECT_H2C_SRC / EXPECT_H2C_BASE_SRC); the exact set of src files that differ.
# P0b import guard: head / base each import their own src; its mutant (expectation pointed at base while importing
#     head) must FAIL, for each of the three new test files.
# P1  tests/q38_l6/test_l6_{fskip,toklists,aotskip}.py: exactly P1_WANT passed, 0 failed, 0 skipped.
# P2  compile-key registration rows of the five new flags: exactly 11 passed (2 parametrized tests x 5 flags + together).
# B   CPU microbench (tests/q38_l6/bench_l6_h2cheap.py): BENCH_H2C_DONE identity=PASS.
# P7  flags-off identity: tests/q38_l6/l6_carry_identity.py (every output that leaves _prepare_inputs + per-step upload
#     counts, all L6 levers unset) gives the same L6CARRY_IDENTITY_ALL digest on head and on base, each importing its src.
# (dev stops here, after a mutant survey)
# P2b collect-only census (head / base); P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary /
#     0 tests / collection error / a new failing id on head = FAIL); P4 mutants (q38_l6_h2cheap_mutrun.py all: control +
#     every mutant must FAIL its target); P5 head src unchanged by the run.
# Last line: Q38_L6_H2C_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
unset VLLM_TPU_AOT_SKIP_GUARD_CHECK
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=98
P2_WANT=11
NEW="tests/q38_l6/test_l6_fskip.py tests/q38_l6/test_l6_toklists.py tests/q38_l6/test_l6_aotskip.py"
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_h2cheap_mutrun as m; print(len(m.MUTANTS['all']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_H2C_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_H2C_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_H2C_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_H2C_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_H2C_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
for t in $NEW; do
  (cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT $t -k test_import_resolves_to_the_tree_under_test) > /tmp/l6h2c_p0b.log 2>&1; rc=$?
  echo "P0b guard mutant $t (expect base while importing head) rc=$rc: $(summ /tmp/l6h2c_p0b.log)"
  [ $rc != 0 ] && [ "$(count /tmp/l6h2c_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6h2c_p0b.log; }
done

echo "== P1 $NEW"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT -s --durations=10 $NEW) > /tmp/l6h2c_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |^FSKIP_|^AOTSKIP_|Error|assert' /tmp/l6h2c_p1.log | head -80
grep -aE 's (call|setup) ' /tmp/l6h2c_p1.log | head -10
echo "P1 rc=$rc summary: $(summ /tmp/l6h2c_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6h2c_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6h2c_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6h2c_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6h2c_p1.log; }

echo "== P2 compile-key registration rows of the five new flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_FENCE_SKIP or L6_TOKEN_LISTS or L6_AOT_SKIP or together") > /tmp/l6h2c_p2.log 2>&1; rc=$?
tail -4 /tmp/l6h2c_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6h2c_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6h2c_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6h2c_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

echo "== B CPU microbench"
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src:tests/q38_l6 timeout -s KILL 2400 python3 tests/q38_l6/bench_l6_h2cheap.py) > /tmp/l6h2c_bench.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6h2c_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_H2C_DONE identity=PASS' /tmp/l6h2c_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6h2c_bench.log; }

echo "== P7 flags-off identity: head vs base (b488e1cb0), l6_carry_identity.py with every L6 lever unset"
(cd "$H" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6h2c_p7h.log 2>&1; rch=$?
(cd "$B" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6h2c_p7b.log 2>&1; rcb=$?
AH=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6h2c_p7h.log | awk '{print $2}')
AB=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6h2c_p7b.log | awk '{print $2}')
FH=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6h2c_p7h.log | awk '{print $2}')
FB=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6h2c_p7b.log | awk '{print $2}')
echo "P7 rc head=$rch base=$rcb all head=$AH base=$AB file head=$FH base=$FB"
case "$FH" in "$(realpath "$H")/src/"*) okh=1;; *) okh=0;; esac
case "$FB" in "$(realpath "$B")/src/"*) okb=1;; *) okb=0;; esac
[ $rch = 0 ] && [ $rcb = 0 ] && [ -n "$AH" ] && [ "$AH" = "$AB" ] && [ $okh = 1 ] && [ $okb = 1 ] \
  || { FAIL="$FAIL P7"; tail -30 /tmp/l6h2c_p7h.log; tail -30 /tmp/l6h2c_p7b.log; }

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3600 python3 tests/q38_l6/q38_l6_h2cheap_mutrun.py . all survey) > /tmp/l6h2c_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_H2C_MUT' /tmp/l6h2c_p4.log
  [ -z "$FAIL" ] && { echo "Q38_L6_H2C_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_H2C_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/test_envs_builders.py
tests/core/test_host_knobs.py tests/core/test_q38_d1n_ab.py tests/runner/test_host_phases.py
tests/platforms/test_patch_registry.py tests/compilation/test_aot_guard_skip.py
tests/layers/core/test_attention_metadata.py tests/runner/test_fast_metadata_build_port.py tests/runner/test_block_table_stage_inc.py
tests/runner/test_qwen4_exp_inputs.py tests/runner/test_q38_fused_prep_functional.py
tests/runner/test_qwen4_exp_ple_host_ids.py tests/models/vllm/test_qwen4_exp.py tests/models/vllm/test_qwen4_exp_ple_host_compilation.py
tests/runner/test_tpu_runner.py tests/runner/test_tpu_runner_pp.py tests/runner/test_pcp_runner_integration.py
tests/runner/test_dcp_runner_integration.py tests/runner/test_mtp_async_contracts.py tests/runner/test_prompt_tail.py
tests/runner/test_tpu_runner_preemption.py tests/runner/test_tpu_runner_async_output.py
tests/runner/test_postfwd_runner_wiring.py tests/runner/test_postfwd_fused.py tests/runner/test_spec_tail_port.py
tests/runner/test_q38_eagerfix.py tests/runner/test_fast_token_substitution.py tests/runner/test_pcp_async_substitution.py
tests/runner/test_mamba_state_seed_copies.py tests/spec_decode/test_speculative_decoding_manager.py
tests/layers/core/test_pcp_sequence_layout.py
tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_jitter.py
tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_defer.py tests/q38_l6/test_l6_steady.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_eos_canary.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/platforms/test_patch_registry.py tests/compilation/test_aot_guard_skip.py tests/runner/test_tpu_runner_async_output.py tests/q38_l6/test_l6_decode_carry.py"
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
regress_perfile l6h2c "$H" "$B" 1800 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6_h2cheap_mutrun.py . all) > /tmp/l6h2c_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_H2C_MUT' /tmp/l6h2c_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_H2C_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6h2c_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6h2c_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_H2C_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_H2C_SUMMARY FAIL mode=full:$FAIL"; exit 1

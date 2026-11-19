#!/bin/bash
# q38 L6 P-METER + admission census + NODEFER census CPU proof (CDK CPU pod only, never the Mac or the VM).
# head = q38-l6-pmeter (log-only follow-up commits on the instrument tree: P-METER, admission census, NODEFER census),
# base = q38-l6-instr 8b09369cb.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_pmeter.sh smoke|full
# P0 versions; head / base src hashes (EXPECT_PM_SRC / EXPECT_PM_BASE_SRC); the exact set of src files that differ (EXPECT_PM_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and base/ for base); its mutant
#     (test_import_resolves_to_the_tree_under_test expecting base while importing head) must FAIL.
# P1 tests/q38_l6/test_l6_pmeter.py, tests/q38_l6/test_l6_admit_census.py, tests/q38_l6/test_l6_nodefer.py, tests/q38_l6instr/test_l6instr_submarks.py,
#    tests/runner/test_host_phases_gc.py, tests/runner/test_host_phases_subs.py, tests/q38_hostsub/test_hostsub_reader.py,
#    tests/runner/test_block_table_stage_inc.py: exactly 67 passed, 0 failed, 0 skipped (the import guards run, not skip).
# P1b flags-off identity: the instrument with every flag off reads no clock, writes no ring entry and logs nothing
#     (test_off_reads_no_clock_and_logs_nothing / test_other_lever_on_pmeter_off_writes_nothing, inside P1), and the
#     HOSTPHASES line is identical with the census on or off (test_host_phases_subs, inside P1); here: host_phases tests pass.
# P2 compile-key rows of the 7 P-METER + 2 NODEFER flags (registered + host-only) + the all-together test: exactly 19 passed.
# (smoke stops here, after the P2b census)
# P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error / a new failing
#    id on head = FAIL); a file collecting 0 tests on both trees is excluded and named; the KEY files must run.
# P4 mutants: q38_l6_pmeter_mutrun.py (control + 32), the reader's q38_hostsub_read_mutrun.py (baseline + 19),
#    L6-INSTR's q38_l6instr_mutrun.py (control + 22, still killed on this head); each must FAIL its target.
# P5 microbench at N = 4 / 9 / 11 / 25 / 52 (bench_l6_pmeter.py; PMBENCH lines, informational, must complete).
# P6 head src unchanged by the run.
# Last line: Q38_L6_PMETER_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L6_[A-Z_]+|HOST_PHASES[A-Z_]*)'); do unset "$v"; done
unset Q38_L6PM_EXPECT_TREE Q38_L6INSTR_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
E=$H/tests/q38_l6

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_PM_SRC)); base src $BS (expect $(cat $E/EXPECT_PM_BASE_SRC))"
[ "$HS" = "$(cat $E/EXPECT_PM_SRC)" ] && [ "$BS" = "$(cat $E/EXPECT_PM_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $E/EXPECT_PM_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6PM_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_pmeter.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6pm_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6pm_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6pm_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6pm_p0b.log; }

P1F="tests/q38_l6/test_l6_pmeter.py tests/q38_l6/test_l6_admit_census.py tests/q38_l6/test_l6_nodefer.py tests/q38_l6instr/test_l6instr_submarks.py
tests/runner/test_host_phases_gc.py tests/runner/test_host_phases_subs.py tests/q38_hostsub/test_hostsub_reader.py
tests/runner/test_block_table_stage_inc.py"
echo "== P1 $(echo $P1F)"
(cd "$H" && env $CPUENV Q38_L6PM_EXPECT_TREE="$(realpath $H)" Q38_L6INSTR_EXPECT_TREE="$(realpath $H)" $PT $P1F) > /tmp/l6pm_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6pm_p1.log | head -80
echo "P1 rc=$rc summary: $(summ /tmp/l6pm_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6pm_p1.log passed)" = 67 ] && [ "$(count /tmp/l6pm_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6pm_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -120 /tmp/l6pm_p1.log; }
(cd "$H" && env $CPUENV $PT tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py) > /tmp/l6pm_p1b.log 2>&1; rc=$?
echo "P1b rc=$rc summary: $(summ /tmp/l6pm_p1b.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6pm_p1b.log passed)" -gt 0 ] && [ "$(count /tmp/l6pm_p1b.log failed)" = 0 ] || { FAIL="$FAIL P1b"; tail -40 /tmp/l6pm_p1b.log; }

echo "== P2 compile-key rows of the P-METER / NODEFER flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_PMETER or L6_INJECT or L6_NODEFER or together") > /tmp/l6pm_p2.log 2>&1; rc=$?
tail -4 /tmp/l6pm_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6pm_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6pm_p2.log passed)" = 19 ] && [ "$(count /tmp/l6pm_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py tests/runner/test_host_phases_subs.py
tests/runner/test_host_phases_gc.py tests/compilation/test_tpu_compile_key_registration.py
tests/test_envs.py tests/test_envs_builders.py tests/core/test_host_knobs.py
tests/runner/test_tpu_runner.py tests/runner/test_mamba_state_seed_copies.py tests/runner/test_q38_gdn_seed_copy_packed.py
tests/runner/test_postfwd_fused.py tests/runner/test_postfwd_runner_wiring.py tests/runner/test_postfwd_static_args.py
tests/runner/test_q38_eagerfix.py tests/runner/test_qwen4_exp_inputs.py tests/runner/test_q38_fused_prep_functional.py
tests/runner/test_prompt_tail.py tests/runner/test_mtp_async_contracts.py
tests/runner/test_carry_rows_fresh_port.py tests/runner/test_draft_carries_port.py tests/runner/test_stackd_precompile_fixes.py
tests/runner/test_spec_tail_port.py tests/runner/test_fast_token_substitution.py
tests/runner/test_tpu_runner_async_output.py tests/runner/test_spec_verify_served_only.py tests/runner/test_pp_last_stage_output.py
tests/runner/test_fast_metadata_build_port.py tests/runner/test_block_table_stage_inc.py tests/layers/core/test_attention_metadata.py
tests/decfb2/test_vps_fb_wiring.py tests/q38mm_dstock/test_vps_wiring.py tests/decfb2/test_vps_harness_no_leak.py
tests/runner/test_tpu_runner_preemption.py tests/runner/test_tpu_runner_pp.py
tests/spec_decode/test_eagle3.py tests/spec_decode/test_dflash.py tests/spec_decode/test_draft_top_tokens_port.py
tests/spec_decode/test_speculative_decoding_manager.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_defer.py
tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_jitter.py tests/q38_l6/test_l6_sab_compose.py
tests/q38_l6instr/test_l6instr_submarks.py tests/q38_hostsub/test_hostsub_reader.py
tests/megakernel_runtime/test_qwen38_real_traffic.py tests/megakernel_runtime/test_served_ab.py
tests/models/vllm/test_qwen4_exp_mtp.py tests/distributed/test_qwen4_exp_pd_connector.py"
KEY="tests/runner/test_host_phases.py tests/runner/test_host_phases_subs.py tests/compilation/test_tpu_compile_key_registration.py
tests/runner/test_block_table_stage_inc.py tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py
tests/q38_l6/test_l6_lockstep.py tests/spec_decode/test_eagle3.py tests/runner/test_tpu_runner.py tests/q38_l6instr/test_l6instr_submarks.py"
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

if [ "$MODE" = smoke ]; then
  [ -z "$FAIL" ] && { echo "Q38_L6_PMETER_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_L6_PMETER_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE (q38-l6-instr 8b09369cb)"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6pm "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants (P-METER + admission census)"
(cd "$H" && python3 tests/q38_l6/q38_l6_pmeter_mutrun.py .) > /tmp/l6pm_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6PM_MUT' /tmp/l6pm_p4.log
echo "P4 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_L6PM_MUT PASS killed 32/32 of 32' /tmp/l6pm_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/l6pm_p4.log; }
echo "== P4b reader mutants"
python3 -I "$H/tests/q38_hostsub/q38_hostsub_read_mutrun.py" "$H/tests/q38_hostsub/q38_hostsub_read.py" > /tmp/l6pm_p4b.log 2>&1; rc=$?
grep -aE '^BASELINE|^MUT|^Q38_HOSTSUB_READ_MUT' /tmp/l6pm_p4b.log
[ $rc = 0 ] && grep -aq '^Q38_HOSTSUB_READ_MUT PASS killed 19/19' /tmp/l6pm_p4b.log || { FAIL="$FAIL P4b"; tail -30 /tmp/l6pm_p4b.log; }
echo "== P4c L6-INSTR mutants on this head"
(cd "$H" && python3 tests/q38_l6instr/q38_l6instr_mutrun.py .) > /tmp/l6pm_p4c.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6INSTR_MUT' /tmp/l6pm_p4c.log
[ $rc = 0 ] && grep -aq '^Q38_L6INSTR_MUT PASS killed 22/22 of 22' /tmp/l6pm_p4c.log || { FAIL="$FAIL P4c"; tail -40 /tmp/l6pm_p4c.log; }

echo "== P5 microbench (N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV python3 tests/q38_l6/bench_l6_pmeter.py) > /tmp/l6pm_p5.log 2>&1; rc=$?
grep -aE '^PMBENCH' /tmp/l6pm_p5.log
[ $rc = 0 ] && grep -aq '^PMBENCH_DONE' /tmp/l6pm_p5.log && [ "$(grep -ac '^PMBENCH N=' /tmp/l6pm_p5.log)" = 5 ] || { FAIL="$FAIL P5"; tail -30 /tmp/l6pm_p5.log; }

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"

[ -z "$FAIL" ] && { echo "Q38_L6_PMETER_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_PMETER_SUMMARY FAIL mode=full:$FAIL"; exit 1

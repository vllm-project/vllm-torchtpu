#!/bin/bash
# q38 L6-INSTR (H1 rung 1) CPU proof of the log-only instrument commit on q38-l6-instr (CDK CPU pod only, never the Mac or the VM).
# Base = q38-l6-next 5685da398, so the diff set is L4-C's HOSTSUBS cherry-picks plus L6-INSTR's instruments.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6instr/run_cpu.sh smoke|full
# P0 versions; head / base src hashes (EXPECT_SRC / EXPECT_BASE_SRC); the exact set of src files that differ (EXPECT_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base); its mutant
#     (test_import_resolves_to_the_tree_under_test expecting base while importing head) must FAIL.
# P1 tests/q38_l6instr/test_l6instr_submarks.py, tests/runner/test_host_phases_gc.py, tests/runner/test_host_phases_subs.py,
#    tests/q38_hostsub/test_hostsub_reader.py: exactly 35 passed, 0 failed, 0 skipped (the import guard runs, not skips);
#    P1b test_host_phases.py / test_host_phases_slow.py pass.
# P2 compile-key rows of VLLM_TPU_HOST_PHASES_SUBS and VLLM_TPU_HOST_PHASES_GC + the all-together test: exactly 5 passed.
# (smoke stops here, after the P2b census)
# P3 per-file regression HEAD vs BASE (L3's regress_perfile 66a42736: no summary / 0 tests / collection error / a new failing
#    id on head = FAIL); a file collecting 0 tests on both trees is excluded and named; the KEY files must run.
# P4 mutants: q38_l6instr_mutrun.py (control + 22), L4-C's q38_hostsub_mutrun.py (control + 11), the reader's
#    q38_hostsub_read_mutrun.py (baseline + 12); each must FAIL its target.
# P5 head src unchanged by the run.
# Last line: Q38_L6INSTR_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L6_[A-Z_]+|HOST_PHASES[A-Z_]*)'); do unset "$v"; done
unset Q38_L6INSTR_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6instr/EXPECT_SRC)); base src $BS (expect $(cat $H/tests/q38_l6instr/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6instr/EXPECT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6instr/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6instr/EXPECT_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6INSTR_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6instr/test_l6instr_submarks.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6i_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6i_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6i_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6i_p0b.log; }

P1F="tests/q38_l6instr/test_l6instr_submarks.py tests/runner/test_host_phases_gc.py tests/runner/test_host_phases_subs.py tests/q38_hostsub/test_hostsub_reader.py"
echo "== P1 $P1F"
(cd "$H" && env $CPUENV Q38_L6INSTR_EXPECT_TREE="$(realpath $H)" $PT $P1F) > /tmp/l6i_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6i_p1.log | head -60
echo "P1 rc=$rc summary: $(summ /tmp/l6i_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6i_p1.log passed)" = 35 ] && [ "$(count /tmp/l6i_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6i_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -80 /tmp/l6i_p1.log; }
(cd "$H" && env $CPUENV $PT tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py) > /tmp/l6i_p1b.log 2>&1; rc=$?
echo "P1b rc=$rc summary: $(summ /tmp/l6i_p1b.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6i_p1b.log passed)" -gt 0 ] && [ "$(count /tmp/l6i_p1b.log failed)" = 0 ] || { FAIL="$FAIL P1b"; tail -40 /tmp/l6i_p1b.log; }

echo "== P2 compile-key registration rows of VLLM_TPU_HOST_PHASES_SUBS / VLLM_TPU_HOST_PHASES_GC"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "HOST_PHASES_SUBS or HOST_PHASES_GC or together") > /tmp/l6i_p2.log 2>&1; rc=$?
tail -4 /tmp/l6i_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6i_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6i_p2.log passed)" = 5 ] && [ "$(count /tmp/l6i_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py tests/compilation/test_tpu_compile_key_registration.py
tests/test_envs.py tests/test_envs_builders.py tests/core/test_host_knobs.py
tests/runner/test_tpu_runner.py tests/runner/test_mamba_state_seed_copies.py tests/runner/test_q38_gdn_seed_copy_packed.py
tests/runner/test_postfwd_fused.py tests/runner/test_postfwd_runner_wiring.py tests/runner/test_postfwd_static_args.py
tests/runner/test_q38_eagerfix.py tests/runner/test_qwen4_exp_inputs.py tests/runner/test_q38_fused_prep_functional.py
tests/runner/test_qwen4_exp_ple_host_ids.py tests/runner/test_prompt_tail.py tests/runner/test_mtp_async_contracts.py
tests/runner/test_carry_rows_fresh_port.py tests/runner/test_draft_carries_port.py tests/runner/test_stackd_precompile_fixes.py
tests/runner/test_tpu_runner_pp_profile.py tests/runner/test_spec_tail_port.py tests/runner/test_fast_token_substitution.py
tests/runner/test_tpu_runner_async_output.py tests/runner/test_spec_verify_served_only.py tests/runner/test_pp_last_stage_output.py
tests/runner/test_fast_metadata_build_port.py
tests/spec_decode/test_eagle3.py tests/spec_decode/test_dflash.py tests/spec_decode/test_draft_top_tokens_port.py
tests/spec_decode/test_speculative_decoding_manager.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_defer.py
tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py
tests/q38mm_dstock/test_vps_wiring.py tests/decfb2/test_vps_fb_wiring.py tests/test_q38_mmhost_engagement.py
tests/megakernel_runtime/test_module_globals.py tests/megakernel_runtime/test_qwen38_guards.py
tests/megakernel_runtime/test_qwen38_real_traffic.py tests/megakernel_runtime/test_qwen38_whole_selfcheck.py
tests/megakernel_runtime/test_served_ab.py tests/megakernel_runtime/test_wholecheck.py
tests/models/vllm/test_qwen4_exp_mtp.py tests/distributed/test_qwen4_exp_pd_connector.py
tests/layers/adapter/test_indexer_topk_rows_update.py tests/q38_deco44_c32mk/test_imported_megakernel_identity.py
tests/q38_deco44_c32mk/test_mk_plus_join.py tests/q38_l1serve/test_d14_served_transport.py"
KEY="tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py tests/compilation/test_tpu_compile_key_registration.py
tests/runner/test_mamba_state_seed_copies.py tests/runner/test_q38_fused_prep_functional.py tests/q38_l6/test_l6_decode_carry.py
tests/spec_decode/test_eagle3.py tests/runner/test_tpu_runner.py"
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
  [ -z "$FAIL" ] && { echo "Q38_L6INSTR_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_L6INSTR_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE (q38-l6-next 5685da398)"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6instr "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants (L6-INSTR)"
(cd "$H" && python3 tests/q38_l6instr/q38_l6instr_mutrun.py .) > /tmp/l6i_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6INSTR_MUT' /tmp/l6i_p4.log
echo "P4 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_L6INSTR_MUT PASS killed 22/22 of 22' /tmp/l6i_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6i_p4.log; }
echo "== P4b mutants (L4-C HOSTSUBS)"
(cd "$H" && python3 tests/q38_hostsub/q38_hostsub_mutrun.py .) > /tmp/l6i_p4b.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_HOSTSUB_MUT' /tmp/l6i_p4b.log
echo "P4b rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_HOSTSUB_MUT PASS killed 11/11 of 11' /tmp/l6i_p4b.log || { FAIL="$FAIL P4b"; tail -40 /tmp/l6i_p4b.log; }
echo "== P4c reader mutants"
python3 -I "$H/tests/q38_hostsub/q38_hostsub_read_mutrun.py" "$H/tests/q38_hostsub/q38_hostsub_read.py" > /tmp/l6i_p4c.log 2>&1; rc=$?
grep -aE '^BASELINE|^MUT|^Q38_HOSTSUB_READ_MUT' /tmp/l6i_p4c.log
[ $rc = 0 ] && grep -aq '^Q38_HOSTSUB_READ_MUT PASS killed 12/12' /tmp/l6i_p4c.log || { FAIL="$FAIL P4c"; tail -30 /tmp/l6i_p4c.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6INSTR_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6INSTR_SUMMARY FAIL mode=full:$FAIL"; exit 1

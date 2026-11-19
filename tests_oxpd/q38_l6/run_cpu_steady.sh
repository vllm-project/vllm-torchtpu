#!/bin/bash
# q38 L6-STEADY CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_steady.sh smoke|full
# P0 versions; head / base src hashes (EXPECT_STEADY_SRC / EXPECT_STEADY_BASE_SRC); the exact set of src files that differ
#    (EXPECT_STEADY_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and base/ for base); its mutant
#    (the guard test expecting base while importing head) must FAIL.
# P1 tests/q38_l6/test_l6_steady.py + test_l6_ab_envcache.py: exactly 52 passed, 0 failed, 0 skipped
#    (the import-guard test runs, not skips).
# P2 compile-key registration rows of the two steady flags + the all-together test: exactly 5 passed.
# P2b collect-only census of the P3 files on head and base.                        (smoke stops here)
# P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error / a new
#    failing id on head = FAIL). A file that collects 0 tests on BOTH trees is excluded and named; KEY files must run.
# P4 mutants (q38_l6steady_mutrun.py: control + 20 mutants, each must FAIL its target, not ERROR).
# P5 head src unchanged by the run.
# Last line: Q38_L6STEADY_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
unset Q38_L6_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
E=$H/tests/q38_l6
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_STEADY_SRC)) commit $(cat $E/EXPECT_STEADY_COMMIT); base src $BS (expect $(cat $E/EXPECT_STEADY_BASE_SRC))"
[ "$HS" = "$(cat $E/EXPECT_STEADY_SRC)" ] && [ "$BS" = "$(cat $E/EXPECT_STEADY_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $E/EXPECT_STEADY_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_ab_envcache.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6s_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6s_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6s_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6s_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_ab_envcache.py) > /tmp/l6s_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6s_p1.log | head -60
echo "P1 rc=$rc summary: $(summ /tmp/l6s_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6s_p1.log passed)" = 52 ] && [ "$(count /tmp/l6s_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6s_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6s_p1.log; }

echo "== P2 compile-key registration rows of the steady flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "STEADY or together") > /tmp/l6s_p2.log 2>&1; rc=$?
tail -4 /tmp/l6s_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6s_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6s_p2.log passed)" = 5 ] && [ "$(count /tmp/l6s_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/layers/core/test_attention_metadata.py
tests/runner/test_fast_metadata_build_port.py tests/runner/test_block_table_stage_inc.py tests/runner/test_qwen4_exp_inputs.py
tests/runner/test_q38_fused_prep_functional.py tests/test_q38_mmhost_engagement.py tests/runner/test_tpu_runner.py
tests/runner/test_tpu_runner_pp.py tests/runner/test_pcp_runner_integration.py tests/runner/test_dcp_runner_integration.py
tests/runner/test_mtp_async_contracts.py tests/runner/test_prompt_tail.py tests/runner/test_tpu_runner_preemption.py
tests/runner/test_tpu_runner_async_output.py tests/runner/test_host_phases.py tests/platforms/test_patch_registry.py
tests/q38_l6/test_l6_ab_envcache.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/layers/core/test_attention_metadata.py
tests/runner/test_fast_metadata_build_port.py tests/runner/test_block_table_stage_inc.py tests/runner/test_qwen4_exp_inputs.py"
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
  [ -z "$FAIL" ] && { echo "Q38_L6STEADY_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_L6STEADY_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6steady "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6steady_mutrun.py .) > /tmp/l6s_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6STEADY_MUT' /tmp/l6s_p4.log
echo "P4 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_L6STEADY_MUT PASS killed 20/20 of 20' /tmp/l6s_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6s_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6STEADY_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6STEADY_SUMMARY FAIL mode=full:$FAIL"; exit 1

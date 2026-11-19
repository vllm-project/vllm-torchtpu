#!/bin/bash
# q38 L6 lever carry (VLLM_TPU_L6_DECODE_CARRY) CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh), with EXPECT_SRC / EXPECT_BASE_SRC set:
#   bash head/tests/q38_l6/run_cpu_carry.sh smoke|full
# P0 versions, src hashes (head == EXPECT_SRC, base == EXPECT_BASE_SRC) and the exact set of src files that differ;
# P1 tests/q38_l6/test_l6_decode_carry.py: exactly 44 passed, 0 failed, 0 skipped, with the import guard pointed at
#    head/src; then the guard alone pointed at base/src must FAIL;
# P2 compile-key registration rows of the two flags: exactly 4 passed;                (smoke stops here, after P2b)
# P2b collect-only census of the P3 files on head and base (printed);
# P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error / a new
#    failing id on head = FAIL). A file that collects 0 tests on BOTH trees is excluded and named; the KEY files must run;
# P4 mutants (q38_l6carry_mutrun.py: control + 26 src mutants + the import-guard mutant, each must FAIL its target);
# P5 head src unchanged by the run;
# P6 lever-off identity: l6_carry_identity.py prints the same L6CARRY_IDENTITY_ALL digest on head and on base, each
#    importing its own tree.
# Last line: Q38_L6CARRY_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for k in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$k"; done
unset VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_D1N_AB_PERIOD_MS Q38_L6CARRY_EXPECT_ROOT
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect ${EXPECT_SRC:-?}); base src $BS (expect ${EXPECT_BASE_SRC:-?})"
[ "$HS" = "${EXPECT_SRC:-}" ] && [ "$BS" = "${EXPECT_BASE_SRC:-}" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/compilation/tpu_compiler.py and head/src/vllm_torchtpu/compilation/tpu_compiler.py differ
Files base/src/vllm_torchtpu/core/l6_ab.py and head/src/vllm_torchtpu/core/l6_ab.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
Only in head/src/vllm_torchtpu/runner: decode_carry.py
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (5 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P1 test_l6_decode_carry.py (guard -> head/src)"
(cd "$H" && env $CPUENV Q38_L6CARRY_EXPECT_ROOT="$H/src" $PT tests/q38_l6/test_l6_decode_carry.py) > /tmp/l6c_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR) |Error|assert' /tmp/l6c_p1.log | head -60
echo "P1 rc=$rc summary: $(summ /tmp/l6c_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6c_p1.log passed)" = 44 ] && [ "$(count /tmp/l6c_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6c_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6c_p1.log; }
(cd "$H" && env $CPUENV Q38_L6CARRY_EXPECT_ROOT="$B/src" $PT tests/q38_l6/test_l6_decode_carry.py::test_imports_the_staged_tree) > /tmp/l6c_p1g.log 2>&1; rc=$?
echo "P1 guard->base rc=$rc summary: $(summ /tmp/l6c_p1g.log) (must fail)"
[ $rc != 0 ] && [ "$(count /tmp/l6c_p1g.log failed)" = 1 ] || FAIL="$FAIL P1-guard"

echo "== P2 compile-key registration rows of the carry flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k L6_DECODE_CARRY) > /tmp/l6c_p2.log 2>&1; rc=$?
tail -4 /tmp/l6c_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6c_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6c_p2.log passed)" = 4 ] && [ "$(count /tmp/l6c_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/q38_l6/test_l6_ab_envcache.py
tests/runner/test_postfwd_runner_wiring.py tests/runner/test_postfwd_fused.py tests/runner/test_postfwd_static_args.py
tests/runner/test_dcp_runner_integration.py tests/runner/test_pcp_runner_integration.py tests/runner/test_spec_tail_port.py
tests/runner/test_q38_eagerfix.py tests/runner/test_draft_carries_port.py tests/runner/test_fast_token_substitution.py
tests/runner/test_tpu_runner_preemption.py tests/runner/test_prompt_tail.py tests/runner/test_qwen4_exp_inputs.py
tests/runner/test_q38_fused_prep_functional.py tests/runner/test_pcp_async_substitution.py
tests/runner/test_tpu_runner_async_output.py tests/runner/test_mamba_state_seed_copies.py tests/test_envs.py
tests/test_envs_builders.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/q38_l6/test_l6_ab_envcache.py
tests/runner/test_postfwd_runner_wiring.py tests/runner/test_dcp_runner_integration.py tests/test_envs_builders.py"
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
  [ -z "$FAIL" ] && { echo "Q38_L6CARRY_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_L6CARRY_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6carry "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6carry_mutrun.py "$H" "$B") > /tmp/l6c_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6CARRY_MUT' /tmp/l6c_p4.log
echo "P4 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_L6CARRY_MUT PASS killed 27/27 of 27' /tmp/l6c_p4.log || { FAIL="$FAIL P4"; tail -80 /tmp/l6c_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

echo "== P6 lever-off identity head vs base"
(cd "$H" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6c_p6h.log 2>&1; rch=$?
(cd "$B" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6c_p6b.log 2>&1; rcb=$?
grep -a '^L6CARRY_IDENTITY' /tmp/l6c_p6h.log | sed 's/^/P6 head /'
grep -a '^L6CARRY_IDENTITY' /tmp/l6c_p6b.log | sed 's/^/P6 base /'
AH=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6c_p6h.log | awk '{print $2}')
AB=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6c_p6b.log | awk '{print $2}')
FH=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6c_p6h.log | awk '{print $2}')
FB=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6c_p6b.log | awk '{print $2}')
echo "P6 rc head=$rch base=$rcb all head=$AH base=$AB file head=$FH base=$FB"
case "$FH" in "$(realpath "$H")/src/"*) okh=1;; *) okh=0;; esac
case "$FB" in "$(realpath "$B")/src/"*) okb=1;; *) okb=0;; esac
[ $rch = 0 ] && [ $rcb = 0 ] && [ -n "$AH" ] && [ "$AH" = "$AB" ] && [ $okh = 1 ] && [ $okb = 1 ] \
  || { FAIL="$FAIL P6"; tail -40 /tmp/l6c_p6h.log; tail -40 /tmp/l6c_p6b.log; }

[ -z "$FAIL" ] && { echo "Q38_L6CARRY_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6CARRY_SUMMARY FAIL mode=full:$FAIL"; exit 1

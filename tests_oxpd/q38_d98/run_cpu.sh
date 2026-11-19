#!/bin/bash
# q38 L4 X-H14-D98R (D98 fast compiled call on the D14 serve stack) CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_d98/run_cpu.sh smoke|full
# P0 versions, src hashes (head == EXPECT_SRC, base == EXPECT_BASE_SRC) and the exact set of src files that differ;
# P1 test_fast_compiled_call.py: passed > 0, 0 failed, 0 skipped, D98_WRAPPED_AT_IMPORT >= 10 (fresh process);
# P2 compile-key registration rows of the 4 flags: exactly 8 passed.            (smoke stops here, after P2b)
# P2b collect-only census of the P3 files on head and base (printed);
# P3 per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596: no summary / 0 tests / collection error / a new
#    failing id on head = FAIL). A file that collects 0 tests on BOTH trees is excluded and named; the KEY files must run;
# P4 CPU dispatch microbench (M, CPU caveat); P5 option (b) probe; P6 mutants (q38_d98_mutrun.py: 16 / 16 killed);
# P7 head src unchanged by the run.
# Last line: Q38_D98R_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
unset VLLM_TPU_AOT_SKIP_GUARD_CHECK VLLM_TPU_FAST_COMPILED_CALL VLLM_TPU_FAST_CALL_GUARD_MEMO VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE VLLM_TPU_FAST_CALL_LOG_EVERY
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_d98/EXPECT_SRC)) commit $(cat $H/tests/q38_d98/EXPECT_COMMIT); base src $BS (expect $(cat $H/tests/q38_d98/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_d98/EXPECT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_d98/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/compilation/tpu_compiler.py and head/src/vllm_torchtpu/compilation/tpu_compiler.py differ
Files base/src/vllm_torchtpu/env_override.py and head/src/vllm_torchtpu/env_override.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
Only in head/src/vllm_torchtpu/compilation: fast_compiled_call.py
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (5 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P1 test_fast_compiled_call.py"
(cd "$H" && env $CPUENV $PT -s tests/compilation/test_fast_compiled_call.py) > /tmp/d98_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR) |Error|assert' /tmp/d98_p1.log | head -40
NW=$(grep -aoE 'D98_WRAPPED_AT_IMPORT [0-9]+' /tmp/d98_p1.log | grep -oE '[0-9]+$' | head -1)
echo "P1 rc=$rc summary: $(summ /tmp/d98_p1.log); wrapped at import ${NW:-none}"
[ $rc = 0 ] && [ "$(count /tmp/d98_p1.log passed)" -gt 0 ] && [ "$(count /tmp/d98_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/d98_p1.log skipped)" = 0 ] && [ "${NW:-0}" -ge 10 ] || { FAIL="$FAIL P1"; tail -60 /tmp/d98_p1.log; }
grep -aoE 'D98_WRAPPED [A-Za-z0-9_.]+' /tmp/d98_p1.log | sort -u | sed 's/^/P1 /'

echo "== P2 compile-key registration rows of the 4 FAST_C* flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k 'FAST_COMPILED_CALL or FAST_CALL_') > /tmp/d98_p2.log 2>&1; rc=$?
tail -4 /tmp/d98_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/d98_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/d98_p2.log passed)" = 8 ] && [ "$(count /tmp/d98_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/platforms/test_patch_registry.py tests/runner/test_q38_eagerfix.py
tests/compilation/test_tpu_compiler.py tests/core/test_host_knobs.py tests/test_envs_builders.py tests/runner/test_host_phases.py
tests/runner/test_host_phases_slow.py tests/runner/test_host_phases_subs.py tests/runner/test_fast_metadata_build_port.py
tests/runner/test_spec_tail_port.py tests/runner/test_pp_last_stage_output.py tests/megakernel_runtime/test_qwen38_real_traffic.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/platforms/test_patch_registry.py tests/runner/test_q38_eagerfix.py"
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

if [ "$MODE" = smoke ]; then
  [ -z "$FAIL" ] && { echo "Q38_D98R_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_D98R_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE (HOSTSUB head bee788a27)"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile d98r "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 CPU dispatch microbench (M, CPU pod, aot_eager stand-in)"
(cd "$H" && env $CPUENV python3 tests/q38_d98/d98_microbench.py 3000 7) > /tmp/d98_p4.log 2>&1
grep -aE '^D98MB' /tmp/d98_p4.log
grep -aq 'D98MB_SUMMARY PASS' /tmp/d98_p4.log || { tail -20 /tmp/d98_p4.log; FAIL="$FAIL P4"; }

echo "== P5 option (b) feasibility probe"
(cd "$H" && env $CPUENV python3 tests/q38_d98/d98_probe_b.py) 2>&1 | grep -aE '^D98B' || FAIL="$FAIL P5"

echo "== P6 mutants"
(cd "$H" && python3 tests/q38_d98/q38_d98_mutrun.py .) > /tmp/d98_p6.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_D98_MUT' /tmp/d98_p6.log
echo "P6 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_D98_MUT PASS killed 16/16' /tmp/d98_p6.log || { FAIL="$FAIL P6"; tail -30 /tmp/d98_p6.log; }

HS2=$(srch "$H")
echo "P7 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P7"

[ -z "$FAIL" ] && { echo "Q38_D98R_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_D98R_SUMMARY FAIL mode=full:$FAIL"; exit 1

#!/bin/bash
# q38 L6-MK CPU proof (CDK CPU pod only; never the Mac or the VM): levers mkctx, mkthin, fuserpc.
# Modelled on tests/q38_l6/run_cpu_engalloc.sh. Usage (from the unpacked bundle dir holding head/ base/
# q38_l3_regress_perfile.sh):   bash head/tests/q38_l6/run_cpu_mk.sh dev|full
# P0  versions; head / base src hashes (EXPECT_MK_SRC / EXPECT_MK_BASE_SRC); the exact set of src files that differ.
# P0b import guard: vllm_torchtpu resolves under head/ (and base/ for base); its mutant (expectation = base while
#     importing head) must FAIL.
# P1  test_l6_mk.py: exactly P1_WANT passed; P1f test_l6_fuse_rpc.py: exactly P1F_WANT passed; 0 failed, 0 skipped.
# P1b flags-off / flags-on identity against base: the same random streams (l6_mk_harness.py, 3 seeds each) on base,
#     head with the flags unset, and head with the flag on give identical digests (mkctx, mkthin).
# P2  compile-key registration rows of the L6-MK flags: exactly P2_WANT passed.
# B   microbench at N = 4 / 9 / 11 / 25 / 52 (bench_l6_mk.py); every arm's outputs identical (BENCH_DONE identity=PASS).
# (dev stops here; mut stops here after a mutant survey)
# P2b collect-only census (head / base) of the regression files; P3 per-file regression HEAD vs BASE (L3's
#     regress_perfile 4fb6f596); P4 mutants (q38_l6_mk_mutrun.py mk: control + every mutant must FAIL its target);
# P5  head src unchanged by the run.
# Last line: Q38_L6_MK_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=28
P1F_WANT=20
P2_WANT=19
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_mk_mutrun as m; print(len(m.MUTANTS['mk']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"), "torch_tpu", m.version("torch_tpu"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_MK_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_MK_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_MK_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_MK_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_MK_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_mk.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6mk_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6mk_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6mk_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6mk_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_mk.py (mkctx, mkthin)"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" timeout -s KILL 1800 $PT -s --durations=10 tests/q38_l6/test_l6_mk.py) > /tmp/l6mk_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |^ENGAGEMENT |^MKTHIN_SOURCES|Error|assert' /tmp/l6mk_p1.log | head -60
grep -aE 's (call|setup) ' /tmp/l6mk_p1.log | head -10
echo "P1 rc=$rc summary: $(summ /tmp/l6mk_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6mk_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6mk_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6mk_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6mk_p1.log; }

echo "== P1f tests/q38_l6/test_l6_fuse_rpc.py (fuserpc, e2kvonly; real vLLM RPC rig)"
(cd "$H" && env $CPUENV VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120 timeout -s KILL 2400 $PT -s --durations=10 -o faulthandler_timeout=240 tests/q38_l6/test_l6_fuse_rpc.py) > /tmp/l6mk_p1f.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |^ENGAGEMENT |^FUSERPC_SOURCES|^RIG_ERROR|Error|assert|Thread 0x|File "' /tmp/l6mk_p1f.log | head -120
grep -aE 's (call|setup) ' /tmp/l6mk_p1f.log | head -10
echo "P1f rc=$rc summary: $(summ /tmp/l6mk_p1f.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6mk_p1f.log passed)" = $P1F_WANT ] && [ "$(count /tmp/l6mk_p1f.log failed)" = 0 ] \
  && [ "$(count /tmp/l6mk_p1f.log skipped)" = 0 ] || { FAIL="$FAIL P1f"; tail -200 /tmp/l6mk_p1f.log; }

if [ "$MODE" = p1f ]; then  # diagnostic: P0 / P1 / P1f only
  [ -z "$FAIL" ] && { echo "Q38_L6_MK_SUMMARY PASS mode=p1f"; exit 0; }
  echo "Q38_L6_MK_SUMMARY FAIL mode=p1f:$FAIL"; exit 1
fi

echo "== P1b identity against base (same streams on base, head flags unset, head flag on)"
HARN=$H/tests/q38_l6/l6_mk_harness.py
for lever in mkctx mkthin; do
  case $lever in mkctx) FLAG=VLLM_TPU_L6_MK_CTX;; mkthin) FLAG=VLLM_TPU_L6_MK_THIN_TT;; esac
  for seed in 0 1 2; do
    db=$(cd "$B" && env $CPUENV timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    dh=$(cd "$H" && env $CPUENV timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    do=$(cd "$H" && env $CPUENV $FLAG=1 timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    echo "P1b $lever seed=$seed base: $db"
    echo "P1b $lever seed=$seed head-off: $dh"
    echo "P1b $lever seed=$seed head-on: $do"
    if [ -n "$db" ] && [ "$db" = "$dh" ] && [ "$db" = "$do" ]; then echo "P1b $lever seed=$seed IDENTICAL"; else echo "P1b $lever seed=$seed DIFFER"; FAIL="$FAIL P1b-$lever-$seed"; fi
  done
done

echo "== P2 compile-key registration rows of the L6-MK flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_MK or L6_FUSE or L6_OUT_KVONLY or together") > /tmp/l6mk_p2.log 2>&1; rc=$?
tail -4 /tmp/l6mk_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6mk_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6mk_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6mk_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

echo "== B microbench"
(cd "$H" && env $CPUENV timeout -s KILL 3600 python3 tests/q38_l6/bench_l6_mk.py) > /tmp/l6mk_bench.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6mk_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6mk_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6mk_bench.log; }

if [ "$MODE" = dev ] || [ "$MODE" = mut ]; then
  if [ "$MODE" = mut ]; then
    echo "== P4 mutant survey (mut)"
    (cd "$H" && timeout -s KILL 5000 python3 tests/q38_l6/q38_l6_mk_mutrun.py . mk survey) > /tmp/l6mk_p4.log 2>&1
    grep -aE '^CONTROL|^MUT|^Q38_L6_MK_MUT' /tmp/l6mk_p4.log
  fi
  [ -z "$FAIL" ] && { echo "Q38_L6_MK_SUMMARY PASS mode=$MODE"; exit 0; }
  echo "Q38_L6_MK_SUMMARY FAIL mode=$MODE:$FAIL"; exit 1
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/test_envs_builders.py
tests/platforms/test_patch_registry.py tests/compilation/test_tpu_compiler.py tests/compilation/test_aot_guard_skip.py
tests/compilation/test_fast_compiled_call.py tests/megakernel_runtime/test_served_ab.py
tests/megakernel_runtime/test_qwen38_whole_selfcheck.py tests/megakernel_runtime/test_wholecheck.py
tests/megakernel_runtime/test_qwen38_real_traffic.py tests/megakernel_runtime/test_l1d07b1_gdn_meta_shared.py
tests/megakernel_runtime/test_module_globals.py tests/q38_l1serve/test_d14_served_transport.py
tests/worker/test_tpu_worker.py tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_fctx.py
tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_engine_alloc.py
tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_eos_canary.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/megakernel_runtime/test_served_ab.py
tests/megakernel_runtime/test_wholecheck.py tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_engine_alloc.py"
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
regress_perfile l6mk "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && timeout -s KILL 7000 python3 tests/q38_l6/q38_l6_mk_mutrun.py . mk) > /tmp/l6mk_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_MK_MUT' /tmp/l6mk_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_MK_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6mk_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6mk_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_MK_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_MK_SUMMARY FAIL mode=full:$FAIL"; exit 1

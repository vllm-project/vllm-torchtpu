#!/bin/bash
# q38 L6-SYNC CPU proof of the SYNC-GC (syncgc) and MAINPIN-2 (mainpin2) levers, the numa_bind port and D1e's GC timer
# (CDK CPU pod only; never the Mac or the VM). Modelled on L6-ENGINE's tests/q38_l6/run_cpu_engalloc.sh.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_sync.sh dev|full
# P0  versions; head / base src hashes (EXPECT_SYNC_SRC / EXPECT_SYNC_BASE_SRC); the exact set of src files that differ
#     (EXPECT_SYNC_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base);
#     its mutant (expectation pointed at base while importing head) must FAIL exactly 1 test.
# P1  tests/q38_l6/test_l6_jitter.py (32) + tests/worker/test_numa_bind.py (26) + tests/core/test_host_knobs.py (12)
#     + the two worker-section tests of tests/compilation/test_additional_config_hash_patch.py: exactly 72 passed,
#     0 failed, 0 skipped (the import-guard test runs, not skips).
# P2  compile-key registration rows of the five new flags: exactly 11 passed (2 parametrized tests x 5 flags + the
#     all-together test).
# B   GC-pause microbench (tests/q38_l6/bench_sync_gc.py; synthetic decode-shaped heap, the real lever in the sync
#     arms): BENCH_SYNCGC_DONE rc=0 (0 automatic collections and every lockstep collection on a multiple of N).
# (dev stops here, after a mutant survey)
# P2b collect-only census (head / base) of the regression files.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error /
#     a new failing id on head = FAIL).
# P4  mutants (q38_l6_sync_mutrun.py sync: control + every mutant must FAIL its target).
# P5  head src unchanged by the run.
# Last line: Q38_L6_SYNC_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z0-9_]+'); do unset "$v"; done
unset VLLM_TPU_HOST_GC VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_CPU_PIN 2>/dev/null
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=72
P2_WANT=11
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_sync_mutrun as m; print(len(m.MUTANTS['sync']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_SYNC_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_SYNC_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_SYNC_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_SYNC_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_SYNC_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_jitter.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6s_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6s_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6s_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6s_p0b.log; }

echo "== P1 test_l6_jitter.py + test_numa_bind.py + test_host_knobs.py + the worker-section hash tests"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT --durations=8 tests/q38_l6/test_l6_jitter.py \
  tests/worker/test_numa_bind.py tests/core/test_host_knobs.py \
  tests/compilation/test_additional_config_hash_patch.py::test_ignored_keys_match_the_keys_the_runner_reads \
  tests/compilation/test_additional_config_hash_patch.py::test_worker_section_does_not_change_the_hash) > /tmp/l6s_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6s_p1.log | head -60
grep -aE 's (call|setup) ' /tmp/l6s_p1.log | head -8
echo "P1 rc=$rc summary: $(summ /tmp/l6s_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6s_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6s_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6s_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6s_p1.log; }

echo "== P2 compile-key registration rows of the five new flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_MAIN_PIN2 or L6_NUMA_BIND or L6_SYNC_GC or together") > /tmp/l6s_p2.log 2>&1; rc=$?
tail -4 /tmp/l6s_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6s_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6s_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6s_p2.log failed)" = 0 ] || { FAIL="$FAIL P2"; tail -40 /tmp/l6s_p2.log; }

echo "== B GC-pause microbench (synthetic decode-shaped heap; the real lever drives the sync arms)"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_sync_gc.py) > /tmp/l6s_bench.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6s_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_SYNCGC_DONE rc=0' /tmp/l6s_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6s_bench.log; }

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3000 python3 tests/q38_l6/q38_l6_sync_mutrun.py . sync survey) > /tmp/l6s_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_SYNC_MUT' /tmp/l6s_p4.log
  [ -z "$FAIL" ] && { echo "Q38_L6_SYNC_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_SYNC_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/q38_l6/test_l6_jitter.py tests/core/test_host_knobs.py tests/compilation/test_tpu_compile_key_registration.py
tests/compilation/test_additional_config_hash_patch.py tests/test_envs.py tests/test_envs_builders.py
tests/worker/test_tpu_worker.py tests/worker/test_cpu_pinning.py tests/worker/test_tpu_worker_rank_binding.py
tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py
tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_defer.py tests/q38_l6/test_l6_steady.py
tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_eos_canary.py
tests/core/test_q38_d1n_ab.py tests/core/test_engine_phases.py tests/platforms/test_patch_registry.py
tests/test_package_import.py tests/megakernel_runtime/test_served_ab.py tests/spec_decode/test_eagle3.py"
KEY="tests/q38_l6/test_l6_jitter.py tests/core/test_host_knobs.py tests/compilation/test_tpu_compile_key_registration.py
tests/compilation/test_additional_config_hash_patch.py tests/worker/test_tpu_worker.py"
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
regress_perfile l6s "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6_sync_mutrun.py . sync) > /tmp/l6s_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_SYNC_MUT' /tmp/l6s_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_SYNC_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6s_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/l6s_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_SYNC_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_SYNC_SUMMARY FAIL mode=full:$FAIL"; exit 1

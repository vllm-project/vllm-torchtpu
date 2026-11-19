#!/bin/bash
# q38 L6 lever engalloc CPU proof (CDK CPU pod only; never the Mac or the VM). Modelled on L6-HOST's
# tests/q38_l6/run_cpu.sh (q38-l6-trims b5ec67de2 / 4a0cdb6d5).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh extra/model_config.json):
#   bash head/tests/q38_l6/run_cpu_engalloc.sh dev|mut|full
# P0  versions; head / base src hashes (EXPECT_ENGALLOC_SRC / _BASE_SRC); the exact set of src files that differ.
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base);
#     its mutant (expectation pointed at base while importing head) must FAIL.
# P1  tests/q38_l6/test_l6_engine_alloc.py + test_l6_ab_envcache.py: exactly 37 passed, 0 failed, 0 skipped.
# P2  compile-key registration rows of the two engalloc flags: exactly 5 passed (2 parametrized tests x 2 flags +
#     the all-together test).
# B   bench: served plan, us per allocate_slots / cache_blocks call at N = 3 / 6 / 15 / 25 / 52 running requests,
#     base / twins / fast arms; twins vs fast identity and the CHECK=1 arm must pass (BENCH_DONE identity=PASS).
# (dev stops here, after a mutant survey)
# P2b collect-only census (head / base) of the regression files.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection error /
#     a new failing id on head = FAIL).
# P4  mutants (q38_l6_engalloc_mutrun.py engalloc: control + every mutant must FAIL its target).
# P5  head src unchanged by the run.
# Last line: Q38_L6_ENGALLOC_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=37
P2_WANT=5
MUT_WANT=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_engalloc_mutrun as m; print(len(m.MUTANTS['engalloc']))")
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_ENGALLOC_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_ENGALLOC_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_ENGALLOC_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_ENGALLOC_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_ENGALLOC_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }
(cd "$H" && env $CPUENV python3 -c "import vllm_torchtpu, vllm.v1.core.kv_cache_manager as k; print('P0 vllm kv_cache_manager', k.__file__)" 2>&1 | tail -1)

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6/test_l6_engine_alloc.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6ea_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6ea_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6ea_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6ea_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_ab_envcache.py"
(cd "$H" && env $CPUENV Q38_L6_EXPECT_TREE="$(realpath $H)" $PT -s --durations=8 tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_ab_envcache.py) > /tmp/l6ea_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |^STREAM |^ENGAGEMENT |Error|assert' /tmp/l6ea_p1.log | head -60
grep -aE 's (call|setup) ' /tmp/l6ea_p1.log | head -8
echo "P1 rc=$rc summary: $(summ /tmp/l6ea_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6ea_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6ea_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6ea_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -120 /tmp/l6ea_p1.log; }

if [ "$MODE" = mut ]; then  # short check of the boundary mutants alone (L6-HOST's request): control + 8 mutants
  echo "== P4b boundary mutants (set bound)"
  (cd "$H" && timeout -s KILL 2400 python3 tests/q38_l6/q38_l6_engalloc_mutrun.py . bound) > /tmp/l6ea_p4b.log 2>&1; rc=$?
  grep -aE '^CONTROL|^MUT|^Q38_L6_ENGALLOC_MUT' /tmp/l6ea_p4b.log
  NB=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_engalloc_mutrun as m; print(len(m.MUTANTS['bound']))")
  echo "P4b rc=$rc (want killed $NB/$NB)"
  [ $rc = 0 ] && grep -aq "^Q38_L6_ENGALLOC_MUT PASS killed $NB/$NB of $NB" /tmp/l6ea_p4b.log || { FAIL="$FAIL P4b"; tail -60 /tmp/l6ea_p4b.log; }
  [ -z "$FAIL" ] && { echo "Q38_L6_ENGALLOC_SUMMARY PASS mode=mut"; exit 0; }
  echo "Q38_L6_ENGALLOC_SUMMARY FAIL mode=mut:$FAIL"; exit 1
fi

echo "== P2 compile-key registration rows of the engalloc flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_ENGINE_ALLOC or together") > /tmp/l6ea_p2.log 2>&1; rc=$?
tail -4 /tmp/l6ea_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6ea_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6ea_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6ea_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

echo "== B bench (served plan)"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_engine_alloc.py --config "$W/extra/model_config.json") > /tmp/l6ea_bench.log 2>&1; rc=$?
grep -aE '^(BENCH|PROFILE)' /tmp/l6ea_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6ea_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6ea_bench.log; }

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3000 python3 tests/q38_l6/q38_l6_engalloc_mutrun.py . engalloc survey) > /tmp/l6ea_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6_ENGALLOC_MUT' /tmp/l6ea_p4.log
  [ -z "$FAIL" ] && { echo "Q38_L6_ENGALLOC_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_ENGALLOC_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/test_envs_builders.py
tests/core/test_kv_twin_groups.py tests/core/test_engine_phases.py tests/core/test_host_knobs.py
tests/core/test_q38_d1n_ab.py tests/core/test_prefix_cache_policy_vllm.py tests/q38_l6/test_l6_ab_envcache.py
tests/platforms/test_patch_registry.py tests/platforms/test_mamba_checkpoint_retention.py
tests/platforms/test_dcp_fine_prefix_hits_patch.py tests/test_eagle_block_drop.py
tests/distributed/test_qwen4_exp_decode_prefix_cache.py"
KEY="tests/compilation/test_tpu_compile_key_registration.py tests/test_envs.py tests/core/test_kv_twin_groups.py tests/q38_l6/test_l6_ab_envcache.py"
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
regress_perfile l6ea "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6_engalloc_mutrun.py . engalloc) > /tmp/l6ea_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6_ENGALLOC_MUT' /tmp/l6ea_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6_ENGALLOC_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6ea_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/l6ea_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6_ENGALLOC_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_ENGALLOC_SUMMARY FAIL mode=full:$FAIL"; exit 1

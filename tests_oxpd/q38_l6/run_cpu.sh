#!/bin/bash
# q38 L6-V6 combined CPU proof on q38-l6-v6 = q38-l6-v5 e80411f84 (q38-l6-v4-eos b488e1cb0 + INSTR + HP-REPORT + PTRIMS +
# GATE + eos2 + reader rows dl_ / df_ + SEED + SYNC + ADC; v5c7 j-137cabfd PASS) + ADMIT-INC (q38-l6-admit 7c8b301de
# 07dfbfc87: lever admitinc) + MK (q38-l6-mk 797679b3e..7060ee3f0: levers mkctx, mkthin, fuserpc, e2kvonly; mkdirect
# registered only) + CARRY-LEAN / PREP-SHORT (q38-l6-pshort 39310b8d3 4b7dd1220 52fe14d7e 262390425 4a04ca7bd 033b1feb5:
# levers clean, pshort; ADC's merge context ported into the short path) + P-METER / admission census / NODEFER census
# (q38-l6-pmeter ca4cc16f6..80ea2224a: levers pmeter, inject, injectall, nodefer; log-only; ADMIT-INC's count-change
# pack upload bracketed by the census too) + L6-H2 (q38-l6-h2cheap-v2 f68c9b83f..a106b9a51: levers fskip, toklists,
# aotskip; flag-off and unarmed; PREP-SHORT sets fskip's carried mark) + GCADMIT + gen-2 census (q38-l6-gcadmit
# 9199bd0a3..59ba01ce0: lever gcadmit; GC_CENSUS_* log-only) + the composition tests (test_l6_v6_compose.py)
# + P7b, the ab-active syncgc /
# MAINPIN-2 unset leg (l6_ab_unset_leg.sh = body v4, sha256 7667c92b: L6-HOST's v2 with P-METER's and AOTSKIP's step
# hooks pinned at their exact places in the execute_model block it replays, and mutants M6 / M6r / M7 / M7r).
# CDK CPU pod only, never the Mac or the VM.
# Base = base2 = v3c b551e9b84 (src 532ecea777); base3 = q38-l6-v4-eos b488e1cb0 (src b7a9294877) for the flags-off
# identities (P7 / P7a / P7b) and L6-MK's P1b identity.
# Usage (from the unpacked bundle dir holding head/ base/ base2/ base3/ q38_l3_regress_perfile.sh l6ab_read*.py
# extra/model_config.json):
#   bash head/tests/q38_l6/run_cpu.sh smoke|dev|part1|part2|full
# Modes. The proof of record is part1 + part2, two CPU jobs, each with exact counts and its own summary line
# (v5c7 took 4,795 s in one job; v6 adds three pieces):
#   part1 = P0 P0b B* + lanes T T2 T3 R1 R2a R2b + P6 P7 P7a P7b P5;
#   part2 = P0 + lanes M1a M1b M2 M3 M4 M5 M6 M7 M8 M9 (every P4 mutant suite and P4x) + P5;
#   full  = both, in one job; smoke = P0 P0b + lanes T T2 T3; dev = P0 P0b P1v P1q P1a P4v P4x (the composition).
# P0  versions; head / base src hashes (EXPECT_SRC / EXPECT_BASE_SRC); the exact set of src files that differ (EXPECT_DIFF).
# P0b import guards: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/ (and under base/ for base); each piece's
#     guard pointed at base while importing head must FAIL (exactly 1 failed): l6_ab envcache, carry, engalloc, INSTR,
#     HP-REPORT, PTRIMS, GATE, SEED, SYNC, ADC, ADMIT-INC, MK, PSHORT, P-METER, L6-H2 x 3, GCADMIT, V6 compose (19 mutants).
# B   L6-ENGINE bench (served plan; extra/model_config.json): BENCH_DONE identity=PASS.
# Bp  PTRIMS microbench, served plan, N = 4 / 9 / 11 / 25 / 52, arms off / sc / sce / pt / ab: BENCH_DONE ok=True.
# Bh  HP-REPORT CPU timing, old vs new report: line_mismatches=0.
# Bs  SEED microbench (seed loop + STEADY's per-step metadata clone vs setupfree), N = 4 / 9 / 11 / 25 / 52: BENCH_DONE ok=True.
# Bg  SYNC-GC GC-pause microbench (synthetic decode-shaped heap): BENCH_SYNCGC_DONE rc=0.
# Ba  ADC microbench (host us per step, N = 4 / 9 / 11 / 25 / 52, sc vs sca): BENCH_DONE identity=PASS.
# Bd  ADMIT-INC microbench (65 groups, N = 4 / 9 / 11 / 25 / 52): BENCH_DONE identity=PASS.
# Bm  L6-MK microbench (mkctx / mkthin / fuserpc / e2kvonly, N = 4 / 9 / 11 / 25 / 52): BENCH_DONE identity=PASS.
# (No CARRY-LEAN / PREP-SHORT bench: lead ruling 11-14, clean / pshort failed their cards' CPU timing bars and stay
#  compiled in, flag-off and unarmed; only their in-process correctness tests and mutants run here.)
# Bpm P-METER microbench (_modify_prev_results on a served-shaped stub, N = 4 / 9 / 11 / 25 / 52): PMBENCH_DONE and 5 rows.
# Bh2 L6-H2 microbench (toklists N = 4 / 9 / 11 / 25 / 52, fskip skip counts, aotskip per call): BENCH_H2C_DONE identity=PASS.
# (The B* steps run alone, before the lanes, so their timings are not shared with other work.)
# Lanes run side by side, each printed live with a [lane] prefix (each mutant runner works on its own temp copy; P5
# checks head src is untouched):
#   T:   P1  tests/q38_l6/test_l6_{ab_envcache,fctx,lockstep,jitter,sab_compose,defer,steady,decode_carry,engine_alloc,
#            eos_canary}.py: exactly 203 passed, 0 failed, 0 skipped.
#        P1i L6-INSTR: test_l6instr_submarks.py, test_host_phases_gc.py, test_host_phases_subs.py, test_hostsub_reader.py: 35.
#        P1j test_host_phases.py + test_host_phases_slow.py: exactly 11.
#        P1h HP-REPORT: tests/q38_l6hprep/test_hp_report.py: exactly 63.
#        P1p PTRIMS: tests/q38_l6/test_l6_ptrims.py: exactly 25.
#        P1g GATE: tests/q38_l6/test_l6_gate.py: exactly 22 (prints GATE_STREAM / GATE_STEADY engagement lines).
#        P1s SEED: tests/q38_l6/test_l6_setupfree.py: exactly 20.
#        P1y SYNC: tests/worker/test_numa_bind.py + tests/core/test_host_knobs.py + the two worker-section hash tests of
#            tests/compilation/test_additional_config_hash_patch.py: exactly 40 (26 + 12 + 2).
#        P1a ADC: tests/q38_l6/test_l6_adapt_carry.py: exactly 57.
#        P2  compile-key rows of the 71 L6 flags + VLLM_TPU_CPU_PROBE_S + VLLM_TPU_HOST_PHASES_SUBS / _GC: exactly 149
#            passed (2 parametrized tests x 74 flags + the all-together test).
#        P2c flag census (string check): every VLLM_TPU_L6_* getter in envs.py (exactly 71) is typed, compile-ignored and
#            has a registration row, and every l6_ab.LEVERS value (exactly 30) is one of them; its mutant (one flag
#            dropped from the compile-ignored set on a copy) must FAIL.
#   T2:  P1d ADMIT-INC: tests/q38_l6/test_l6_admit_inc.py: exactly 26.
#        P1m L6-MK: tests/q38_l6/test_l6_mk.py: exactly 28 (ENGAGEMENT lines).
#        P1f L6-MK: tests/q38_l6/test_l6_fuse_rpc.py (real vLLM RPC rig): exactly 20 (ENGAGEMENT lines).
#        P1mb L6-MK identity: l6_mk_harness.py mkctx / mkthin, seeds 0 / 1 / 2, on base3 (b488e1cb0), head with the
#            flags unset and head with the flag on: 6 x IDENTICAL.
#   T3:  P1q PSHORT: tests/q38_l6/test_l6_pshort.py: exactly 105 (text / mirror pins re-pinned on the composed tree).
#        P1v V6 compose: tests/q38_l6/test_l6_v6_compose.py: exactly 15 (CARRY-LEAN x U2 carry_rerecord 2, PREP-SHORT
#            x ADC 2, PREP-SHORT x L6-INSTR tail sub-marks 1, ADMIT-INC x the admission census 1, NODEFER's wrapper
#            byte-identical with the census off 4, FENCE-SKIP x PREP-SHORT 2, TOKEN-LISTS-ONCE off vs base3's
#            get_output 1 (Q38_L6_BASE3_SRC), the U2 copy pin 1, the import guard 1).
#        P1h2 L6-H2: tests/q38_l6/test_l6_{fskip,toklists,aotskip}.py: exactly 98 (40 + 41 + 17).
#        P1gc GCADMIT: tests/q38_l6/test_l6_gcadmit.py + test_l6_gccensus.py: exactly 26 (16 + 10).
#        P1pm P-METER: tests/q38_l6/test_l6_pmeter.py + test_l6_admit_census.py + test_l6_nodefer.py: exactly 25 (15 + 4 + 6).
#   R1 / R2a / R2b: P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596: no summary / 0 tests / collection
#        error / a new failing id on head = FAIL; 3600 s per file) over V6_REGRESS_FILES (96 = v5's 88 +
#        tests/compilation/test_tpu_compiler.py from MK's list + tests/decfb2/test_vps_harness_no_leak.py from P-METER's +
#        tests/layers/core/test_pcp_sequence_layout.py, tests/models/vllm/test_qwen4_exp.py and
#        tests/models/vllm/test_qwen4_exp_ple_host_compilation.py from L6-H2's + tests/q38mm_dm/test_prompt_ids_ndarray.py,
#        tests/runner/test_tpu_runner_kv_connector_lifecycle.py and tests/test_prompt_ids_ndarray_pd.py from GCADMIT's;
#        every other file of the new pieces' own regression lists is in v5's 88 or absent on v3c; R1 = test_wholestep.py,
#        R2a / R2b = the other 95 alternately). No file may be excluded. The L6 import guards point at each tree's own src.
#   M1a: P4 trims (control + 49), P4e end-token canary (control + 7), P4s SEED setupfree (control + 23).
#   M1b: P4b STEADY (control + 20), P4c CARRY (control + 26 + guard), P4d ENGINE (control + 28).
#   M2:  P4i INSTR (control + 23), P4j HOSTSUBS (control + 11), P4k HOSTSUBS reader (baseline 32 / 32 checks + 21),
#        P4h HP-REPORT (control + 6), P4p PTRIMS (control + 21), P4g GATE (control + 17 + guard), P4x every suite's
#        anchors exactly once on head (exactly 592; q38_l6_v6_anchors.py) and its mutant (one anchor duplicated on a
#        copy: FAIL, bad=3).
#   M3:  P4y SYNC (control + 47).
#   M4:  P4a ADC (control + 45).
#   M5:  P4ad ADMIT-INC (control + 30).
#   M6:  P4mk L6-MK (control + 44).
#   M7:  P4ps PSHORT (control + 41), P4v V6 compose (control + 12; Q38_L6_BASE3_SRC).
#   M9:  P4h2 L6-H2 fskip / toklists / aotskip (control + 45).
#   M8:  P4pm P-METER + admission census + NODEFER (control + 32), P4gc GCADMIT + gen-2 census (control + 36).
# P6  the L6AB reader v5 (bundle root l6ab_read.py): selftest + 12 / 12 mutants.
# P7  lever-off identity: the same L6CARRY_IDENTITY_ALL digest on head and on base2 (v3c b551e9b84), and on head and on
#     base3 (b488e1cb0), each importing its own src; and the same L6ADC_IDENTITY_ALL digest on head and base3.
# P7b the ab-active syncgc / MAINPIN-2 unset leg (tests/q38_l6/l6_ab_unset_leg.sh = body v4, sha256 7667c92b, from the
#     bundle root with head/ and base3/): Q38_L6_ABUNSET_SUMMARY PASS and rc 0. v4 pins P-METER's and L6-H2's AOTSKIP
#     step hooks at their exact places in execute_model's l6 block; its own mutants M6 / M6r (P-METER hook dropped /
#     moved) and M7 / M7r (AOTSKIP hook dropped / moved) must fail that pin.
# P5  head src unchanged by the run.
# Last line: Q38_L6_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L6_[A-Z0-9_]+|HOST_PHASES[A-Z0-9_]*)'); do unset "$v"; done
unset VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_D1N_AB_PERIOD_MS VLLM_TPU_SERVED_AB
case "$MODE" in smoke|dev|part1|part2|full) ;; *) echo "Q38_L6_SUMMARY FAIL mode=$MODE: unknown mode"; exit 2;; esac
unset Q38_L6PM_EXPECT_TREE Q38_L6_EXPECT_TREE Q38_L6CARRY_EXPECT_ROOT Q38_L6GATE_EXPECT_ROOT Q38_L6INSTR_EXPECT_TREE Q38_L6HPREP_EXPECT_TREE Q38_L6ADC_EXPECT_ROOT
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
LF=/tmp/l6_lanes; rm -rf $LF; mkdir -p $LF
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
# exact <log> <passed>: rc 0 and exactly <passed> passed, 0 failed, 0 skipped, 0 errors.
exact() { [ "$(count "$1" passed)" = "$2" ] && [ "$(count "$1" failed)" = 0 ] && [ "$(count "$1" skipped)" = 0 ] \
  && [ "$(count "$1" error)" = 0 ] && [ "$(count "$1" errors)" = 0 ]; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

p0b() {
echo "== P0b import guards"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
guardmut() {  # guardmut <tag> <ENV=VALUE> <pytest target>: must FAIL with exactly 1 failed
  local tag=$1 kv=$2 tgt=$3 rc
  (cd "$H" && env $CPUENV "$kv" $PT $tgt) > /tmp/l6_p0b_$tag.log 2>&1; rc=$?
  echo "P0b $tag guard mutant ($kv) rc=$rc: $(summ /tmp/l6_p0b_$tag.log) (must fail)"
  [ $rc != 0 ] && [ "$(count /tmp/l6_p0b_$tag.log failed)" = 1 ] || { FAIL="$FAIL P0b-$tag"; tail -20 /tmp/l6_p0b_$tag.log; }
}
guardmut envcache "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_ab_envcache.py -k test_import_resolves_to_the_tree_under_test"
guardmut carry "Q38_L6CARRY_EXPECT_ROOT=$B/src" "tests/q38_l6/test_l6_decode_carry.py::test_imports_the_staged_tree"
guardmut engalloc "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_engine_alloc.py -k test_import_resolves_to_the_tree_under_test"
guardmut instr "Q38_L6INSTR_EXPECT_TREE=$(realpath $B)" "tests/q38_l6instr/test_l6instr_submarks.py -k test_import_resolves_to_the_tree_under_test"
guardmut hprep "Q38_L6HPREP_EXPECT_TREE=$(realpath $B)" "tests/q38_l6hprep/test_hp_report.py -k test_import_resolves_to_the_tree_under_test"
guardmut ptrims "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_ptrims.py -k test_import_resolves_to_the_tree_under_test"
guardmut gate "Q38_L6GATE_EXPECT_ROOT=$B/src" "tests/q38_l6/test_l6_gate.py::test_imports_the_staged_tree"
guardmut seed "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_setupfree.py -k test_import_resolves_to_the_tree_under_test"
guardmut sync "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_jitter.py -k test_import_resolves_to_the_tree_under_test"
guardmut adc "Q38_L6ADC_EXPECT_ROOT=$(realpath $B)/src" "tests/q38_l6/test_l6_adapt_carry.py -k test_imports_the_staged_tree"
guardmut admitinc "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_admit_inc.py -k test_import_resolves_to_the_tree_under_test"
guardmut mk "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_mk.py -k test_import_resolves_to_the_tree_under_test"
guardmut pshort "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_pshort.py -k test_import_resolves_to_the_tree_under_test"
guardmut pmeter "Q38_L6PM_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_pmeter.py -k test_import_resolves_to_the_tree_under_test"
for h2 in fskip toklists aotskip; do
  guardmut h2$h2 "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_$h2.py -k test_import_resolves_to_the_tree_under_test"
done
guardmut gcadmit "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_gcadmit.py -k test_import_resolves_to_the_tree_under_test"
guardmut v6 "Q38_L6_EXPECT_TREE=$(realpath $B)" "tests/q38_l6/test_l6_v6_compose.py -k test_import_resolves_to_the_tree_under_test"
}
[ "$MODE" = part2 ] || p0b

ptest() {  # ptest <step> <want passed> <extra grep or -> <env...> -- <pytest args...>
  local step=$1 want=$2 pat=$3 rc; shift 3
  local envs=() to=() ; while [ "$1" != "--" ]; do case "$1" in @*) to=(timeout -s KILL "${1#@}");; *) envs+=("$1");; esac; shift; done; shift
  echo "== $step $*"
  (cd "$H" && env $CPUENV "${envs[@]}" "${to[@]}" $PT "$@") > /tmp/l6_$step.log 2>&1; rc=$?
  grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6_$step.log | head -40
  [ "$pat" != - ] && grep -aE "$pat" /tmp/l6_$step.log | head -60
  echo "$step rc=$rc summary: $(summ /tmp/l6_$step.log) (want exactly $want passed)"
  [ $rc = 0 ] && exact /tmp/l6_$step.log "$want" || { FAIL="$FAIL $step"; tail -80 /tmp/l6_$step.log; }
}
laneT() {  # P1 P1i P1j P1h P1p P1g P1s P1y P1a P2 P2c
  FAIL=""
ptest P1 203 - "Q38_L6_EXPECT_TREE=$(realpath $H)" "Q38_L6CARRY_EXPECT_ROOT=$H/src" -- \
  tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_lockstep.py tests/q38_l6/test_l6_jitter.py \
  tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_defer.py tests/q38_l6/test_l6_steady.py tests/q38_l6/test_l6_decode_carry.py \
  tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_eos_canary.py
ptest P1i 35 - "Q38_L6INSTR_EXPECT_TREE=$(realpath $H)" -- \
  tests/q38_l6instr/test_l6instr_submarks.py tests/runner/test_host_phases_gc.py tests/runner/test_host_phases_subs.py \
  tests/q38_hostsub/test_hostsub_reader.py
ptest P1j 11 - "Q38_L6_NOENV=1" -- tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py
ptest P1h 63 - "Q38_L6HPREP_EXPECT_TREE=$(realpath $H)" -- tests/q38_l6hprep/test_hp_report.py
ptest P1p 25 - "Q38_L6_EXPECT_TREE=$(realpath $H)" -- -s --durations=8 tests/q38_l6/test_l6_ptrims.py
ptest P1g 22 '^(GATE_STREAM|GATE_STEADY) ' "Q38_L6GATE_EXPECT_ROOT=$H/src" -- -s --durations=8 tests/q38_l6/test_l6_gate.py
ptest P1s 20 - "Q38_L6_EXPECT_TREE=$(realpath $H)" -- -s --durations=8 tests/q38_l6/test_l6_setupfree.py
ptest P1y 40 - "Q38_L6_EXPECT_TREE=$(realpath $H)" -- tests/worker/test_numa_bind.py tests/core/test_host_knobs.py \
  tests/compilation/test_additional_config_hash_patch.py::test_ignored_keys_match_the_keys_the_runner_reads \
  tests/compilation/test_additional_config_hash_patch.py::test_worker_section_does_not_change_the_hash
ptest P1a 57 - "Q38_L6ADC_EXPECT_ROOT=$(realpath $H)/src" -- -s --durations=10 tests/q38_l6/test_l6_adapt_carry.py

echo "== P2 compile-key registration rows of the L6 flags + CPU_PROBE + HOST_PHASES_SUBS / _GC"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "L6_ or together or CPU_PROBE or HOST_PHASES_SUBS or HOST_PHASES_GC") > /tmp/l6_p2.log 2>&1; rc=$?
tail -4 /tmp/l6_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6_p2.log) (want exactly 149 passed)"
[ $rc = 0 ] && exact /tmp/l6_p2.log 149 || FAIL="$FAIL P2"

echo "== P2c L6 flag census (string check) + its mutant"
cat > /tmp/l6_p2c.py <<'PY'
import re, sys
root = sys.argv[1]
env = open(root + "/src/vllm_torchtpu/envs.py").read()
tc = open(root + "/src/vllm_torchtpu/compilation/tpu_compiler.py").read()
reg = open(root + "/tests/compilation/test_tpu_compile_key_registration.py").read()
ab = open(root + "/src/vllm_torchtpu/core/l6_ab.py").read()
ign = re.search(r"_TPU_COMPILE_ENV_IGNORED[^=]*=[^\n]*\n(.*?)\n[)}\]]", tc, re.S).group(1)
flags = sorted(set(re.findall(r'^\s*"(VLLM_TPU_L6_[A-Z0-9_]+)"\s*:', env, re.M)))
bad = []
for f in flags:
    typed = len(re.findall(r"^\s*%s\s*:" % f, env, re.M))
    ci = ign.count('"%s"' % f)
    rows = reg.count('("%s"' % f)
    print(f"P2c {f} typed={typed} compile_ignored={ci} reg_rows={rows}")
    if (typed, ci, rows) != (1, 1, 1):
        bad.append(f)
levers = re.findall(r'^\s*"([a-z0-9]+)": "(VLLM_TPU_L6_[A-Z0-9_]+)",', re.search(r"^LEVERS: dict.*?^}", ab, re.S | re.M).group(0), re.M)
for name, f in levers:
    print(f"P2c lever {name} -> {f} registered={f in flags}")
    if f not in flags:
        bad.append(f)
print(f"P2c flags {len(flags)} levers {len(levers)} bad {len(bad)} {' '.join(bad)}")
sys.exit(0 if not bad and len(flags) == int(sys.argv[2]) and len(levers) == int(sys.argv[3]) else 1)
PY
python3 -I /tmp/l6_p2c.py "$H" 71 30 > /tmp/l6_p2c.log 2>&1; rc=$?
cat /tmp/l6_p2c.log
[ $rc = 0 ] && grep -aq '^P2c flags 71 levers 30 bad 0' /tmp/l6_p2c.log || FAIL="$FAIL P2c"
M2C=$(mktemp -d /tmp/l6_p2cmut.XXXX); mkdir -p $M2C/src/vllm_torchtpu/compilation $M2C/src/vllm_torchtpu/core $M2C/tests/compilation
cp $H/src/vllm_torchtpu/envs.py $M2C/src/vllm_torchtpu/; cp $H/src/vllm_torchtpu/core/l6_ab.py $M2C/src/vllm_torchtpu/core/
cp $H/tests/compilation/test_tpu_compile_key_registration.py $M2C/tests/compilation/
sed 's/^    "VLLM_TPU_L6_PREP_TRIMS_CHECK",$/    # dropped by the P2c mutant/' $H/src/vllm_torchtpu/compilation/tpu_compiler.py > $M2C/src/vllm_torchtpu/compilation/tpu_compiler.py
python3 -I /tmp/l6_p2c.py "$M2C" 71 30 > /tmp/l6_p2cm.log 2>&1; rc=$?
echo "P2c mutant (VLLM_TPU_L6_PREP_TRIMS_CHECK dropped from the compile-ignored set) rc=$rc: $(tail -1 /tmp/l6_p2cm.log) (must fail)"
[ $rc != 0 ] && grep -aq '^P2c flags 71 levers 30 bad 1 VLLM_TPU_L6_PREP_TRIMS_CHECK$' /tmp/l6_p2cm.log || FAIL="$FAIL P2c-mutant"
rm -rf "$M2C"
  echo "$FAIL" > $LF/T.fail
}

laneT2() {  # P1d P1m P1f P1mb
  FAIL=""
ptest P1d 26 '^ADMIT_BT_SRC_SHA1 ' "Q38_L6_EXPECT_TREE=$(realpath $H)" -- -s --durations=10 tests/q38_l6/test_l6_admit_inc.py
ptest P1m 28 '^(ENGAGEMENT|MKTHIN_SOURCES)' "Q38_L6_EXPECT_TREE=$(realpath $H)" @1800 -- -s --durations=10 tests/q38_l6/test_l6_mk.py
ptest P1f 20 '^(ENGAGEMENT|FUSERPC_SOURCES|RIG_ERROR)' "Q38_L6_EXPECT_TREE=$(realpath $H)" "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120" @2400 -- \
  -s --durations=10 -o faulthandler_timeout=240 tests/q38_l6/test_l6_fuse_rpc.py
echo "== P1mb L6-MK identity (same streams on base3 = b488e1cb0, head flags unset, head flag on)"
local HARN=$H/tests/q38_l6/l6_mk_harness.py lever FLAG seed db dh do
for lever in mkctx mkthin; do
  case $lever in mkctx) FLAG=VLLM_TPU_L6_MK_CTX;; mkthin) FLAG=VLLM_TPU_L6_MK_THIN_TT;; esac
  for seed in 0 1 2; do
    db=$(cd "$W/base3" && env $CPUENV timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    dh=$(cd "$H" && env $CPUENV timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    do=$(cd "$H" && env $CPUENV $FLAG=1 timeout -s KILL 600 python3 "$HARN" $lever $seed 2>&1 | grep -a '^L6MK_TRACE' | tail -1)
    echo "P1mb $lever seed=$seed base3: $db"
    echo "P1mb $lever seed=$seed head-off: $dh"
    echo "P1mb $lever seed=$seed head-on: $do"
    if [ -n "$db" ] && [ "$db" = "$dh" ] && [ "$db" = "$do" ]; then echo "P1mb $lever seed=$seed IDENTICAL"; else echo "P1mb $lever seed=$seed DIFFER"; FAIL="$FAIL P1mb-$lever-$seed"; fi
  done
done
  echo "$FAIL" > $LF/T2.fail
}

laneT3() {  # P1q P1v
  FAIL=""
ptest P1q 105 - "Q38_L6_EXPECT_TREE=$(realpath $H)" @3000 -- -s --durations=12 tests/q38_l6/test_l6_pshort.py
ptest P1v 15 '(ADC_SHORT|FSKIP_SHORT) ' "Q38_L6_EXPECT_TREE=$(realpath $H)" "Q38_L6_BASE3_SRC=$W/base3/src" @2400 -- -s --durations=10 tests/q38_l6/test_l6_v6_compose.py
ptest P1pm 25 - "Q38_L6PM_EXPECT_TREE=$(realpath $H)" @2400 -- tests/q38_l6/test_l6_pmeter.py tests/q38_l6/test_l6_admit_census.py \
  tests/q38_l6/test_l6_nodefer.py
ptest P1h2 98 'FSKIP_SCAN ' "Q38_L6_EXPECT_TREE=$(realpath $H)" @3000 -- -s --durations=10 tests/q38_l6/test_l6_fskip.py \
  tests/q38_l6/test_l6_toklists.py tests/q38_l6/test_l6_aotskip.py
ptest P1gc 26 - "Q38_L6_EXPECT_TREE=$(realpath $H)" @2400 -- --durations=8 tests/q38_l6/test_l6_gcadmit.py tests/q38_l6/test_l6_gccensus.py
  echo "$FAIL" > $LF/T3.fail
}

benches() {
echo "== B L6-ENGINE bench (served plan)"
[ -f "$W/extra/model_config.json" ] || FAIL="$FAIL B-noconfig"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_engine_alloc.py --config "$W/extra/model_config.json") > /tmp/l6_bench.log 2>&1; rc=$?
grep -aE '^(BENCH|PROFILE)' /tmp/l6_bench.log
echo "B rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6_bench.log || { FAIL="$FAIL B"; tail -60 /tmp/l6_bench.log; }

echo "== Bp PTRIMS microbench (served plan; N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_ptrims.py --config "$W/extra/model_config.json" --fragments) > /tmp/l6_benchp.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchp.log
echo "Bp rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE ok=True' /tmp/l6_benchp.log || { FAIL="$FAIL Bp"; tail -60 /tmp/l6_benchp.log; }

echo "== Bh HP-REPORT CPU timing, old (per-phase) vs new (grouped) report"
(cd "$H" && env $CPUENV python3 tests/q38_l6hprep/hp_report_bench.py 400) > /tmp/l6_benchh.log 2>&1; rc=$?
tail -8 /tmp/l6_benchh.log
[ $rc = 0 ] && grep -aq 'line_mismatches=0' /tmp/l6_benchh.log || FAIL="$FAIL Bh"

echo "== Bs SEED microbench (seed loop, STEADY's metadata clone vs setupfree; N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_seed.py) > /tmp/l6_benchs.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchs.log
echo "Bs rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE ok=True' /tmp/l6_benchs.log || { FAIL="$FAIL Bs"; tail -60 /tmp/l6_benchs.log; }

echo "== Bg SYNC-GC GC-pause microbench (synthetic decode-shaped heap; the real lever drives the sync arms)"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_sync_gc.py) > /tmp/l6_benchg.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchg.log
echo "Bg rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_SYNCGC_DONE rc=0' /tmp/l6_benchg.log || { FAIL="$FAIL Bg"; tail -60 /tmp/l6_benchg.log; }

echo "== Ba ADC microbench (served plan, host us per step; N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 3000 python3 tests/q38_l6/bench_l6_adc.py) > /tmp/l6_bencha.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_bencha.log
echo "Ba rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6_bencha.log || { FAIL="$FAIL Ba"; tail -60 /tmp/l6_bencha.log; }

echo "== Bd ADMIT-INC microbench (65 groups; N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_admit_inc.py) > /tmp/l6_benchd.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchd.log
echo "Bd rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6_benchd.log || { FAIL="$FAIL Bd"; tail -60 /tmp/l6_benchd.log; }

echo "== Bm L6-MK microbench (N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 3600 python3 tests/q38_l6/bench_l6_mk.py) > /tmp/l6_benchm.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchm.log
echo "Bm rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_DONE identity=PASS' /tmp/l6_benchm.log || { FAIL="$FAIL Bm"; tail -60 /tmp/l6_benchm.log; }

echo "== Bpm P-METER microbench (N = 4 / 9 / 11 / 25 / 52)"
(cd "$H" && env $CPUENV timeout -s KILL 2400 python3 tests/q38_l6/bench_l6_pmeter.py) > /tmp/l6_benchpm.log 2>&1; rc=$?
grep -aE '^PMBENCH' /tmp/l6_benchpm.log
echo "Bpm rc=$rc"
[ $rc = 0 ] && grep -aq '^PMBENCH_DONE' /tmp/l6_benchpm.log && [ "$(grep -ac '^PMBENCH N=' /tmp/l6_benchpm.log)" = 5 ] || { FAIL="$FAIL Bpm"; tail -60 /tmp/l6_benchpm.log; }

echo "== Bh2 L6-H2 microbench (toklists N = 4 / 9 / 11 / 25 / 52; fskip; aotskip)"
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src:tests/q38_l6 timeout -s KILL 2400 python3 tests/q38_l6/bench_l6_h2cheap.py) > /tmp/l6_benchh2.log 2>&1; rc=$?
grep -aE '^BENCH' /tmp/l6_benchh2.log
echo "Bh2 rc=$rc"
[ $rc = 0 ] && grep -aq '^BENCH_H2C_DONE identity=PASS' /tmp/l6_benchh2.log || { FAIL="$FAIL Bh2"; tail -60 /tmp/l6_benchh2.log; }
}

finish() {
  HS2=$(srch "$H")
  echo "P5 head src after $HS2 (before $HS)"
  [ "$HS2" = "$HS" ] || FAIL="$FAIL P5"
  [ -z "$FAIL" ] && { echo "Q38_L6_SUMMARY PASS mode=$MODE"; exit 0; }
  echo "Q38_L6_SUMMARY FAIL mode=$MODE:$FAIL"; exit 1
}

if [ "$MODE" = dev ]; then  # the composition only: P1v P1q P1a P4v P4x
  SAVE="$FAIL"; FAIL=""
  ptest P1v 15 '(ADC_SHORT|FSKIP_SHORT) ' "Q38_L6_EXPECT_TREE=$(realpath $H)" "Q38_L6_BASE3_SRC=$W/base3/src" @2400 -- -s --durations=10 tests/q38_l6/test_l6_v6_compose.py
  ptest P1q 105 - "Q38_L6_EXPECT_TREE=$(realpath $H)" @3000 -- -s --durations=12 tests/q38_l6/test_l6_pshort.py
  ptest P1a 57 - "Q38_L6ADC_EXPECT_ROOT=$(realpath $H)/src" -- -s --durations=10 tests/q38_l6/test_l6_adapt_carry.py
  FAIL="$SAVE$FAIL"
  mutstep_dev() { local step=$1 pat=$2 rc; shift 2; echo "== $step $*"; (cd "$H" && "$@") > /tmp/l6_$step.log 2>&1; rc=$?
    grep -aE '^CONTROL|^MUT|^Q38_|^ANCHORS' /tmp/l6_$step.log; echo "$step rc=$rc"
    [ $rc = 0 ] && grep -aqE "$pat" /tmp/l6_$step.log || { echo "$step FAILED"; tail -60 /tmp/l6_$step.log; FAIL="$FAIL $step"; }; }
  mutstep_dev P4v '^Q38_L6_V6_MUT PASS killed 12/12 of 12' env Q38_L6_BASE3_SRC=$W/base3/src python3 tests/q38_l6/q38_l6_v6_mutrun.py . v6
  mutstep_dev P4x '^Q38_L6_V6_ANCHORS PASS total=592 bad=0 ' python3 -I "$H/tests/q38_l6/q38_l6_v6_anchors.py" "$H"
  finish
fi

case "$MODE" in smoke|part1|full) [ "$MODE" = smoke ] || benches ;; esac

if [ "$MODE" = smoke ]; then
  echo "== lanes T T2 T3 start $(date -u +%FT%TZ)"
  runlane0() { local name=$1; shift; ( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; "$@"; echo "lane $name done $(date -u +%FT%TZ)" ) 2>&1 | sed -u "s/^/[$name] /" & }
  runlane0 T laneT; runlane0 T2 laneT2; runlane0 T3 laneT3; wait
  for l in T T2 T3; do
    [ -f $LF/$l.fail ] || { echo "lane $l wrote no verdict"; FAIL="$FAIL lane-$l"; continue; }
    echo "lane $l verdict: $( [ -z "$(cat $LF/$l.fail | tr -d ' ')" ] && echo PASS || echo "FAIL$(cat $LF/$l.fail)")"
    FAIL="$FAIL$(cat $LF/$l.fail)"
  done
  finish
fi

# ---------------- lanes ----------------

laneR() {  # laneR <name> <tag> <file>...: P3 per-file regression HEAD vs BASE (v3c b551e9b84) over the given files
  local name=$1 tag=$2 LFAIL=""; shift 2
  echo "== P3-$name per-file regression HEAD vs BASE (v3c b551e9b84): $# files"
  [ "$(sha256sum < $W/q38_l3_regress_perfile.sh | cut -c1-8)" = 4fb6f596 ] || LFAIL="$LFAIL P3-helper"
  . "$W/q38_l3_regress_perfile.sh"
  # Each tree's own src for the L6 import guards (paths relative to the tree the helper cd's into).
  CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src Q38_L6PM_EXPECT_TREE=. Q38_L6CARRY_EXPECT_ROOT=src Q38_L6_EXPECT_TREE=. Q38_L6GATE_EXPECT_ROOT=src Q38_L6INSTR_EXPECT_TREE=. Q38_L6HPREP_EXPECT_TREE=. Q38_L6ADC_EXPECT_ROOT=src"
  regress_perfile $tag "$H" "$B" 3600 "$@" || LFAIL="$LFAIL P3-$name"
  echo "$LFAIL" > $LF/$name.fail
}

mutstep() {  # mutstep <step> <pass regex> <cmd...>
  local step=$1 pat=$2 rc; shift 2
  echo "== $step $*"
  (cd "$H" && "$@") > /tmp/l6_$step.log 2>&1; rc=$?
  grep -aE '^CONTROL|^BASELINE|^MUT|^Q38_|^L6AB_READ' /tmp/l6_$step.log
  echo "$step rc=$rc"
  [ $rc = 0 ] && grep -aqE "$pat" /tmp/l6_$step.log || { echo "$step FAILED"; tail -60 /tmp/l6_$step.log; return 1; }
}

laneM1b() {
  local LFAIL=""
  mutstep P4b '^Q38_L6STEADY_MUT PASS killed 20/20 of 20' python3 tests/q38_l6/q38_l6steady_mutrun.py . || LFAIL="$LFAIL P4b"
  mutstep P4c '^Q38_L6CARRY_MUT PASS killed 27/27 of 27' python3 tests/q38_l6/q38_l6carry_mutrun.py "$H" "$B" || LFAIL="$LFAIL P4c"
  mutstep P4d '^Q38_L6_ENGALLOC_MUT PASS killed 28/28 of 28' python3 tests/q38_l6/q38_l6_engalloc_mutrun.py . engalloc || LFAIL="$LFAIL P4d"
  echo "$LFAIL" > $LF/M1b.fail
}

laneM1a() {
  local LFAIL=""
  mutstep P4 '^Q38_L6_MUT PASS killed 49/49 of 49' python3 tests/q38_l6/q38_l6_mutrun.py . trims || LFAIL="$LFAIL P4"
  mutstep P4e '^Q38_L6_MUT PASS killed 7/7 of 7' python3 tests/q38_l6/q38_l6_mutrun.py . eos || LFAIL="$LFAIL P4e"
  mutstep P4s '^Q38_L6_SEED_MUT PASS killed 23/23 of 23' python3 tests/q38_l6/q38_l6_seed_mutrun.py . seed || LFAIL="$LFAIL P4s"
  echo "$LFAIL" > $LF/M1a.fail
}

laneM2() {
  local LFAIL="" PTW
  mutstep P4i '^Q38_L6INSTR_MUT PASS killed 23/23 of 23' python3 tests/q38_l6instr/q38_l6instr_mutrun.py . || LFAIL="$LFAIL P4i"
  mutstep P4j '^Q38_HOSTSUB_MUT PASS killed 11/11 of 11' python3 tests/q38_hostsub/q38_hostsub_mutrun.py . || LFAIL="$LFAIL P4j"
  mutstep P4k '^Q38_HOSTSUB_READ_MUT PASS killed 21/21' python3 -I "$H/tests/q38_hostsub/q38_hostsub_read_mutrun.py" "$H/tests/q38_hostsub/q38_hostsub_read.py" || LFAIL="$LFAIL P4k"
  mutstep P4h '^Q38_L6HPREP_MUT PASS killed 6/6 of 6' python3 tests/q38_l6hprep/q38_l6hprep_mutrun.py . || LFAIL="$LFAIL P4h"
  PTW=$(python3 -c "import sys; sys.path.insert(0, '$H/tests/q38_l6'); import q38_l6_ptrims_mutrun as m; print(len(m.MUTANTS['ptrims']))")
  echo "P4p ptrims mutants $PTW (want 21)"
  [ "$PTW" = 21 ] || LFAIL="$LFAIL P4p-count"
  mutstep P4p '^Q38_L6_PTRIMS_MUT PASS killed 21/21 of 21' python3 tests/q38_l6/q38_l6_ptrims_mutrun.py . ptrims || LFAIL="$LFAIL P4p"
  mutstep P4g '^Q38_L6GATE_MUT PASS killed 18/18 of 18' python3 tests/q38_l6/q38_l6gate_mutrun.py "$H" "$B" || LFAIL="$LFAIL P4g"
  echo "== P4x every suite's anchors exactly once on head (string check) + its mutant"
  python3 -I "$H/tests/q38_l6/q38_l6_v6_anchors.py" "$H" > /tmp/l6_p4x.log 2>&1; rc=$?
  cat /tmp/l6_p4x.log
  [ $rc = 0 ] && grep -aq '^Q38_L6_V6_ANCHORS PASS total=592 bad=0 ' /tmp/l6_p4x.log || LFAIL="$LFAIL P4x"
  local MX; MX=$(mktemp -d /tmp/l6_p4xmut.XXXX)
  cp -R "$H/src" "$H/tests" "$MX/"
  printf '\n_V6_ANCHOR_DUP = """\n    key = (start_index, num_reqs, stride, nsched.tobytes())\n"""\n' >> "$MX/src/vllm_torchtpu/runner/l6_ptrims.py"
  python3 -I "$H/tests/q38_l6/q38_l6_v6_anchors.py" "$MX" > /tmp/l6_p4xm.log 2>&1; rc=$?
  grep -aE '^ANCHORS BAD|^Q38_L6_V6_ANCHORS' /tmp/l6_p4xm.log
  echo "P4x mutant (the PTRIMS cache-key anchor duplicated on a copy) rc=$rc (must fail, bad=3)"
  [ $rc != 0 ] && grep -aq '^Q38_L6_V6_ANCHORS FAIL total=592 bad=3 ' /tmp/l6_p4xm.log || LFAIL="$LFAIL P4x-mutant"
  rm -rf "$MX"
  echo "$LFAIL" > $LF/M2.fail
}

laneM3() {
  local LFAIL=""
  mutstep P4y '^Q38_L6_SYNC_MUT PASS killed 47/47 of 47' python3 tests/q38_l6/q38_l6_sync_mutrun.py . sync || LFAIL="$LFAIL P4y"
  echo "$LFAIL" > $LF/M3.fail
}

laneM4() {
  local LFAIL=""
  mutstep P4a '^Q38_L6_ADC_MUT PASS killed 45/45 of 45' python3 tests/q38_l6/q38_l6_adc_mutrun.py . adc || LFAIL="$LFAIL P4a"
  echo "$LFAIL" > $LF/M4.fail
}

laneM5() {
  local LFAIL=""
  mutstep P4ad '^Q38_L6_ADMITINC_MUT PASS killed 30/30 of 30' python3 tests/q38_l6/q38_l6_admitinc_mutrun.py . || LFAIL="$LFAIL P4ad"
  echo "$LFAIL" > $LF/M5.fail
}

laneM6() {
  local LFAIL=""
  mutstep P4mk '^Q38_L6_MK_MUT PASS killed 44/44 of 44' timeout -s KILL 9000 python3 tests/q38_l6/q38_l6_mk_mutrun.py . mk || LFAIL="$LFAIL P4mk"
  echo "$LFAIL" > $LF/M6.fail
}

laneM7() {
  local LFAIL=""
  mutstep P4ps '^Q38_L6_PSHORT_MUT PASS killed 41/41 of 41' python3 tests/q38_l6/q38_l6_pshort_mutrun.py . pshort || LFAIL="$LFAIL P4ps"
  mutstep P4v '^Q38_L6_V6_MUT PASS killed 12/12 of 12' env Q38_L6_BASE3_SRC=$W/base3/src python3 tests/q38_l6/q38_l6_v6_mutrun.py . v6 || LFAIL="$LFAIL P4v"
  echo "$LFAIL" > $LF/M7.fail
}

laneM8() {
  local LFAIL=""
  mutstep P4pm '^Q38_L6PM_MUT PASS killed 32/32 of 32' python3 tests/q38_l6/q38_l6_pmeter_mutrun.py . || LFAIL="$LFAIL P4pm"
  mutstep P4gc '^Q38_L6_GCADMIT_MUT PASS killed 36/36 of 36' python3 tests/q38_l6/q38_l6_gcadmit_mutrun.py . gcadmit || LFAIL="$LFAIL P4gc"
  echo "$LFAIL" > $LF/M8.fail
}

laneM9() {
  local LFAIL=""
  mutstep P4h2 '^Q38_L6_H2C_MUT PASS killed 45/45 of 45' python3 tests/q38_l6/q38_l6_h2cheap_mutrun.py . all || LFAIL="$LFAIL P4h2"
  echo "$LFAIL" > $LF/M9.fail
}

# Lanes side by side, each printed live with a [lane] prefix (so a timeout still leaves every finished step in the
# log) and each with one BLAS / OpenMP thread (the pod has 8 CPUs of a 32-core node). Regression: R1 = test_wholestep.py
# (45 min alone in v16 j-543abc02), R2a / R2b = the other 87 files of V5_REGRESS_FILES, alternately.
RFILES=$(grep -v -x 'tests/megakernel/test_wholestep.py' $H/tests/q38_l6/V6_REGRESS_FILES)
R2A=$(echo "$RFILES" | awk 'NR % 2 == 1')
R2B=$(echo "$RFILES" | awk 'NR % 2 == 0')
runlane() {  # runlane <name> <cmd...>
  local name=$1; shift
  ( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; "$@"; echo "lane $name done $(date -u +%FT%TZ)" ) 2>&1 | sed -u "s/^/[$name] /" &
}
case "$MODE" in
  part1) LANES="T T2 T3 R1 R2a R2b" ;;
  part2) LANES="M1a M1b M2 M3 M4 M5 M6 M7 M8 M9" ;;
  full) LANES="T T2 T3 R1 R2a R2b M1a M1b M2 M3 M4 M5 M6 M7 M8 M9" ;;
esac
echo "== lanes $LANES start $(date -u +%FT%TZ); regression files $(wc -l < $H/tests/q38_l6/V6_REGRESS_FILES | tr -d ' ') = 1 + $(echo "$R2A" | wc -l | tr -d ' ') + $(echo "$R2B" | wc -l | tr -d ' ')"
for l in $LANES; do
  case $l in
    T) runlane T laneT ;;
    T2) runlane T2 laneT2 ;;
    T3) runlane T3 laneT3 ;;
    R1) runlane R1 laneR R1 l6v6w tests/megakernel/test_wholestep.py ;;
    R2a) runlane R2a laneR R2a l6v6a $R2A ;;
    R2b) runlane R2b laneR R2b l6v6b $R2B ;;
    *) runlane $l lane$l ;;
  esac
done
wait
echo "== lanes done $(date -u +%FT%TZ)"
for l in $LANES; do
  [ -f $LF/$l.fail ] || { echo "lane $l wrote no verdict"; FAIL="$FAIL lane-$l"; continue; }
  echo "lane $l verdict: $( [ -z "$(cat $LF/$l.fail | tr -d ' ')" ] && echo PASS || echo "FAIL$(cat $LF/$l.fail)")"
  FAIL="$FAIL$(cat $LF/$l.fail)"
done

[ "$MODE" = part2 ] && finish

echo "== P6 L6AB reader (bundle root l6ab_read.py): selftest + mutants"
python3 -I "$W/l6ab_read.py" --selftest > /tmp/l6_p6a.log 2>&1; rc=$?
tail -3 /tmp/l6_p6a.log
python3 -I "$W/l6ab_read_mut.py" "$W/l6ab_read.py" > /tmp/l6_p6b.log 2>&1; rc2=$?
grep -aE '^MUT|^CONTROL|^L6AB_READ_MUT' /tmp/l6_p6b.log
[ $rc = 0 ] && grep -aq '^L6AB_READ SELFTEST PASS' /tmp/l6_p6a.log && [ $rc2 = 0 ] \
  && grep -aq '^L6AB_READ_MUT PASS killed 12/12' /tmp/l6_p6b.log || FAIL="$FAIL P6"

echo "== P7 lever-off identity: head vs base2 = v3c b551e9b84 and head vs base3 = b488e1cb0"
for tag in base2 base3; do
  BX=$W/$tag
  [ -d "$BX/src/vllm_torchtpu" ] || { FAIL="$FAIL P7-no$tag"; continue; }
  (cd "$H" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6_p7h.log 2>&1; rch=$?
  (cd "$BX" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6_p7b.log 2>&1; rcb=$?
  AH=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6_p7h.log | awk '{print $2}')
  AB=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6_p7b.log | awk '{print $2}')
  FH=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6_p7h.log | awk '{print $2}')
  FB=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6_p7b.log | awk '{print $2}')
  echo "P7 $tag rc head=$rch $tag=$rcb all head=$AH $tag=$AB file head=$FH $tag=$FB"
  case "$FH" in "$(realpath "$H")/src/"*) okh=1;; *) okh=0;; esac
  case "$FB" in "$(realpath "$BX")/src/"*) okb=1;; *) okb=0;; esac
  [ $rch = 0 ] && [ $rcb = 0 ] && [ -n "$AH" ] && [ "$AH" = "$AB" ] && [ $okh = 1 ] && [ $okb = 1 ] \
    || { FAIL="$FAIL P7-$tag"; tail -30 /tmp/l6_p7h.log; tail -30 /tmp/l6_p7b.log; }
done

echo "== P7a ADC lever-off identity (l6_adc_identity.py): head vs base3 = b488e1cb0"
(cd "$H" && env $CPUENV timeout -s KILL 1800 python3 "$H/tests/q38_l6/l6_adc_identity.py") > /tmp/l6_p7ah.log 2>&1; rch=$?
(cd "$W/base3" && env $CPUENV timeout -s KILL 1800 python3 "$H/tests/q38_l6/l6_adc_identity.py") > /tmp/l6_p7ab.log 2>&1; rcb=$?
AH=$(grep -a '^L6ADC_IDENTITY_ALL ' /tmp/l6_p7ah.log | awk '{print $2}')
AB=$(grep -a '^L6ADC_IDENTITY_ALL ' /tmp/l6_p7ab.log | awk '{print $2}')
FH=$(grep -a '^L6ADC_IDENTITY_FILE ' /tmp/l6_p7ah.log | awk '{print $2}')
FB=$(grep -a '^L6ADC_IDENTITY_FILE ' /tmp/l6_p7ab.log | awk '{print $2}')
echo "P7a rc head=$rch base3=$rcb all head=$AH base3=$AB file head=$FH base3=$FB"
case "$FH" in "$(realpath "$H")/src/"*) okh=1;; *) okh=0;; esac
case "$FB" in "$(realpath "$W/base3")/src/"*) okb=1;; *) okb=0;; esac
[ $rch = 0 ] && [ $rcb = 0 ] && [ -n "$AH" ] && [ "$AH" = "$AB" ] && [ $okh = 1 ] && [ $okb = 1 ] \
  || { FAIL="$FAIL P7a"; tail -30 /tmp/l6_p7ah.log; tail -30 /tmp/l6_p7ab.log; }

echo "== P7b ab-active syncgc / MAINPIN-2 unset leg (tests/q38_l6/l6_ab_unset_leg.sh, from the bundle root)"
P7BSHA=$(sha256sum < "$H/tests/q38_l6/l6_ab_unset_leg.sh" | cut -c1-64)
echo "P7b body sha256 $P7BSHA (want 7667c92b4291865219ad121179b15124e46ff9dc2ffcd8b4581b42083227ab78)"
[ "$P7BSHA" = 7667c92b4291865219ad121179b15124e46ff9dc2ffcd8b4581b42083227ab78 ] || FAIL="$FAIL P7b-sha"
(cd "$W" && timeout -s KILL 5400 bash "$H/tests/q38_l6/l6_ab_unset_leg.sh") > /tmp/l6_p7bleg.log 2>&1; rc=$?
grep -aE '^(P7b|L6ABUNSET_|Q38_L6_ABUNSET)' /tmp/l6_p7bleg.log | cut -c1-400
echo "P7b rc=$rc"
[ $rc = 0 ] && tail -1 /tmp/l6_p7bleg.log | grep -aq '^Q38_L6_ABUNSET_SUMMARY PASS' || { FAIL="$FAIL P7b"; tail -80 /tmp/l6_p7bleg.log; }

finish

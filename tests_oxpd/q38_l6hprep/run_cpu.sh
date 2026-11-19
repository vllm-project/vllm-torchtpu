#!/bin/bash
# q38 L6-INSTR T-1(ii) HP-REPORT CPU proof on q38-l6-hpreport (CDK CPU pod only, never the Mac or the VM).
# Base = q38-l6-next 5685da398 (src 8409fbb544); the only src change is runner/host_phases.py.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6hprep/run_cpu.sh full
# P0 versions; head / base src hashes (EXPECT_SRC / EXPECT_BASE_SRC); the exact diff set (one file).
# P0b import guard: vllm_torchtpu.__file__ resolves under head/ and base/; its mutant (expect base while importing head) FAILS.
# P1 tests/q38_l6hprep/test_hp_report.py: exactly 63 passed, 0 failed, 0 skipped (byte-identical HOSTPHASES line over 40 random
#    ragged windows, bitwise statistics over 20, the known-value line, the window order, the import guard);
#    P1b test_host_phases.py / test_host_phases_slow.py pass. No new flag, so no compile-key row.
# P3 per-file regression HEAD vs BASE (L3's regress_perfile 4fb6f596) over the host_phases readers / writers.
# P4 mutants (q38_l6hprep_mutrun.py: control + 6, incl. the percentile-axis mutant), each must FAIL its target.
# P6 CPU timing of the old (per-phase) vs new (grouped) report, interleaved: line_mismatches must be 0.
# P5 head src unchanged by the run.
# Last line: Q38_L6HPREP_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L6_[A-Z_]+|HOST_PHASES[A-Z_]*)'); do unset "$v"; done
unset Q38_L6HPREP_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6hprep/EXPECT_SRC)); base src $BS (expect $(cat $H/tests/q38_l6hprep/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6hprep/EXPECT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6hprep/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT="Files base/src/vllm_torchtpu/runner/host_phases.py and head/src/vllm_torchtpu/runner/host_phases.py differ"
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (1 entry)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6HPREP_EXPECT_TREE="$(realpath $B)" $PT tests/q38_l6hprep/test_hp_report.py -k test_import_resolves_to_the_tree_under_test) > /tmp/l6h_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base while importing head) rc=$rc: $(summ /tmp/l6h_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/l6h_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6h_p0b.log; }

echo "== P1 tests/q38_l6hprep/test_hp_report.py"
(cd "$H" && env $CPUENV Q38_L6HPREP_EXPECT_TREE="$(realpath $H)" $PT tests/q38_l6hprep/test_hp_report.py) > /tmp/l6h_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |Error|assert' /tmp/l6h_p1.log | head -40
echo "P1 rc=$rc summary: $(summ /tmp/l6h_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6h_p1.log passed)" = 63 ] && [ "$(count /tmp/l6h_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6h_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -60 /tmp/l6h_p1.log; }
(cd "$H" && env $CPUENV $PT tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py) > /tmp/l6h_p1b.log 2>&1; rc=$?
echo "P1b rc=$rc summary: $(summ /tmp/l6h_p1b.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6h_p1b.log passed)" -gt 0 ] && [ "$(count /tmp/l6h_p1b.log failed)" = 0 ] || { FAIL="$FAIL P1b"; tail -40 /tmp/l6h_p1b.log; }

echo "== P3 per-file regression HEAD vs BASE (q38-l6-next 5685da398)"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile l6hprep "$H" "$B" 1500 tests/runner/test_host_phases.py tests/runner/test_host_phases_slow.py \
  tests/runner/test_prompt_tail.py tests/q38_l6/test_l6_ab_envcache.py tests/compilation/test_tpu_compile_key_registration.py \
  tests/runner/test_tpu_runner.py || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6hprep/q38_l6hprep_mutrun.py .) > /tmp/l6h_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6HPREP_MUT' /tmp/l6h_p4.log
[ $rc = 0 ] && grep -aq '^Q38_L6HPREP_MUT PASS killed 6/6 of 6' /tmp/l6h_p4.log || { FAIL="$FAIL P4"; tail -30 /tmp/l6h_p4.log; }

echo "== P6 CPU timing, old (per-phase) vs new (grouped) report"
(cd "$H" && env $CPUENV python3 tests/q38_l6hprep/hp_report_bench.py 400) > /tmp/l6h_p6.log 2>&1; rc=$?
cat /tmp/l6h_p6.log | tail -8
[ $rc = 0 ] && grep -aq 'line_mismatches=0' /tmp/l6h_p6.log || FAIL="$FAIL P6"

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_L6HPREP_SUMMARY PASS mode=$MODE"; exit 0; }
echo "Q38_L6HPREP_SUMMARY FAIL mode=$MODE:$FAIL"; exit 1

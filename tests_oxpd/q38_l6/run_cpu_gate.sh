#!/bin/bash
# q38 L6-GATE CPU proof: the default-off batch-size gates of CARRY and STEADY (VLLM_TPU_L6_CARRY_MIN_REQS /
# VLLM_TPU_L6_STEADY_MIN_REQS) on q38-l6-gate, against base q38-l6-v4-eos b488e1cb0. CDK CPU pod only, never the Mac
# or the VM. Modelled on run_cpu_engalloc.sh / run_cpu_carry.sh.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l6/run_cpu_gate.sh dev|full
# P0  versions; head / base src hashes (EXPECT_GATE_SRC / EXPECT_GATE_BASE_SRC); the exact set of src files that differ
#     (EXPECT_GATE_DIFF).
# P0b import guard: with PYTHONPATH=src, vllm_torchtpu.__file__ resolves under head/src (and base/src for base); the
#     guard test pointed at base/src while importing head must FAIL (exactly 1 failed).
# P1  tests/q38_l6/test_l6_gate.py: exactly 22 passed, 0 failed, 0 skipped (guard -> head/src); prints the
#     GATE_STREAM / GATE_STEADY engagement lines.
# P2  compile-key registration rows of the two gate flags: exactly 5 passed (2 parametrized tests x 2 flags + the
#     all-together test).
# (dev stops here, after a mutant survey)
# P2b collect-only census (head / base) of the regression files; the KEY files must collect on both.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile, sha256 prefix 4fb6f596: no summary / 0 tests /
#     collection error / a new failing id on head = FAIL). The import guards of the L6 files point at each tree's own src.
# P4  mutants (q38_l6gate_mutrun.py: control + 17 src mutants + the import-guard mutant, each must FAIL its target).
# P5  head src unchanged by the run.
# P6  lever-off identity: l6_carry_identity.py prints the same L6CARRY_IDENTITY_ALL digest on head and base (gates and
#     levers unset), each importing its own src.
# P7  the existing L6 mutant suites' anchors (q38_l6_mutrun / steady / carry / engalloc) still match exactly once on
#     head (string check; their suites are not rerun here).
# Last line: Q38_L6_GATE_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_L6_[A-Z_]+'); do unset "$v"; done
unset VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_D1N_AB_PERIOD_MS VLLM_TPU_SERVED_AB Q38_L6GATE_EXPECT_ROOT Q38_L6CARRY_EXPECT_ROOT Q38_L6_EXPECT_TREE
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
P1_WANT=22
P2_WANT=5
MUT_WANT=18
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE nproc $(nproc) $(python3 -c 'import sys, torch, numpy, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_l6/EXPECT_GATE_SRC)); base src $BS (expect $(cat $H/tests/q38_l6/EXPECT_GATE_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_l6/EXPECT_GATE_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_l6/EXPECT_GATE_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $H/tests/q38_l6/EXPECT_GATE_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import guard"
gh=$(cd "$H" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
gb=$(cd "$B" && env $CPUENV python3 -c 'import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))' 2>&1 | tail -1)
echo "P0b head imports $gh; base imports $gb"
case "$gh" in "$(realpath $H)"/src/*) ;; *) FAIL="$FAIL P0b-head";; esac
case "$gb" in "$(realpath $B)"/src/*) ;; *) FAIL="$FAIL P0b-base";; esac
(cd "$H" && env $CPUENV Q38_L6GATE_EXPECT_ROOT="$B/src" $PT tests/q38_l6/test_l6_gate.py::test_imports_the_staged_tree) > /tmp/l6g_p0b.log 2>&1; rc=$?
echo "P0b guard mutant (expect base/src while importing head) rc=$rc: $(summ /tmp/l6g_p0b.log) (must fail)"
[ $rc != 0 ] && [ "$(count /tmp/l6g_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-mutant"; tail -20 /tmp/l6g_p0b.log; }

echo "== P1 tests/q38_l6/test_l6_gate.py (guard -> head/src)"
(cd "$H" && env $CPUENV Q38_L6GATE_EXPECT_ROOT="$H/src" $PT -s --durations=8 tests/q38_l6/test_l6_gate.py) > /tmp/l6g_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR|SKIPPED) |GATE_STREAM |GATE_STEADY |Error|assert' /tmp/l6g_p1.log | head -80
grep -aE 's (call|setup) ' /tmp/l6g_p1.log | head -8
echo "P1 rc=$rc summary: $(summ /tmp/l6g_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6g_p1.log passed)" = $P1_WANT ] && [ "$(count /tmp/l6g_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/l6g_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -150 /tmp/l6g_p1.log; }

echo "== P2 compile-key registration rows of the gate flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "MIN_REQS or together") > /tmp/l6g_p2.log 2>&1; rc=$?
tail -4 /tmp/l6g_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/l6g_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/l6g_p2.log passed)" = $P2_WANT ] && [ "$(count /tmp/l6g_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

if [ "$MODE" = dev ]; then
  echo "== P4 mutant survey (dev)"
  (cd "$H" && timeout -s KILL 3000 python3 tests/q38_l6/q38_l6gate_mutrun.py "$H" "$B") > /tmp/l6g_p4.log 2>&1
  grep -aE '^CONTROL|^MUT|^Q38_L6GATE_MUT' /tmp/l6g_p4.log
  [ -z "$FAIL" ] && { echo "Q38_L6_GATE_SUMMARY PASS mode=dev"; exit 0; }
  echo "Q38_L6_GATE_SUMMARY FAIL mode=dev:$FAIL"; exit 1
fi

FILES="tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/runner/test_tpu_runner.py
tests/compilation/test_tpu_compile_key_registration.py tests/q38_l6/test_l6_ab_envcache.py tests/q38_l6/test_l6_lockstep.py
tests/q38_l6/test_l6_sab_compose.py tests/q38_l6/test_l6_engine_alloc.py tests/q38_l6/test_l6_eos_canary.py
tests/q38_l6/test_l6_fctx.py tests/q38_l6/test_l6_jitter.py tests/q38_l6/test_l6_defer.py tests/test_envs.py
tests/test_envs_builders.py tests/runner/test_postfwd_runner_wiring.py tests/runner/test_tpu_runner_async_output.py
tests/runner/test_qwen4_exp_inputs.py tests/runner/test_tpu_runner_preemption.py"
KEY="tests/q38_l6/test_l6_decode_carry.py tests/q38_l6/test_l6_steady.py tests/runner/test_tpu_runner.py
tests/compilation/test_tpu_compile_key_registration.py tests/q38_l6/test_l6_ab_envcache.py"
echo "== P2b collect-only census (head / base)"
RUN=""
for f in $FILES; do
  nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  echo "P2b $f head ${nh:-0} base ${nb:-0}"
  if [ "${nh:-0}" = 0 ] || [ "${nb:-0}" = 0 ]; then
    case " $(echo $KEY) " in *" $f "*) echo "P2b KEY file collects nothing on a tree: $f"; FAIL="$FAIL P2b";; esac
  fi
  if [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then echo "P2b EXCLUDED (0 on both) $f"; else RUN="$RUN $f"; fi
done

echo "== P3 per-file regression HEAD vs BASE"
[ "$(sha256sum < $W/q38_l3_regress_perfile.sh | cut -c1-8)" = 4fb6f596 ] || FAIL="$FAIL P3-helper"
. "$W/q38_l3_regress_perfile.sh"
# Each tree's own src for the L6 import guards (paths relative to the tree the helper cd's into).
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src Q38_L6CARRY_EXPECT_ROOT=src Q38_L6_EXPECT_TREE=. Q38_L6GATE_EXPECT_ROOT=src"
regress_perfile l6gate "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_l6/q38_l6gate_mutrun.py "$H" "$B") > /tmp/l6g_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_L6GATE_MUT' /tmp/l6g_p4.log
echo "P4 rc=$rc (want killed $MUT_WANT/$MUT_WANT)"
[ $rc = 0 ] && grep -aq "^Q38_L6GATE_MUT PASS killed $MUT_WANT/$MUT_WANT of $MUT_WANT" /tmp/l6g_p4.log || { FAIL="$FAIL P4"; tail -80 /tmp/l6g_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

echo "== P6 lever-off identity head vs base"
(cd "$H" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6g_p6h.log 2>&1; rch=$?
(cd "$B" && env $CPUENV python3 "$H/tests/q38_l6/l6_carry_identity.py") > /tmp/l6g_p6b.log 2>&1; rcb=$?
grep -a '^L6CARRY_IDENTITY' /tmp/l6g_p6h.log | sed 's/^/P6 head /'
grep -a '^L6CARRY_IDENTITY' /tmp/l6g_p6b.log | sed 's/^/P6 base /'
AH=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6g_p6h.log | awk '{print $2}')
AB=$(grep -a '^L6CARRY_IDENTITY_ALL ' /tmp/l6g_p6b.log | awk '{print $2}')
FH=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6g_p6h.log | awk '{print $2}')
FB=$(grep -a '^L6CARRY_IDENTITY_FILE ' /tmp/l6g_p6b.log | awk '{print $2}')
echo "P6 rc head=$rch base=$rcb all head=$AH base=$AB"
case "$FH" in "$(realpath "$H")/src/"*) okh=1;; *) okh=0;; esac
case "$FB" in "$(realpath "$B")/src/"*) okb=1;; *) okb=0;; esac
[ $rch = 0 ] && [ $rcb = 0 ] && [ -n "$AH" ] && [ "$AH" = "$AB" ] && [ $okh = 1 ] && [ $okb = 1 ] \
  || { FAIL="$FAIL P6"; tail -40 /tmp/l6g_p6h.log; tail -40 /tmp/l6g_p6b.log; }

echo "== P7 existing L6 mutant anchors on head (string check)"
python3 -I - "$H" > /tmp/l6g_p7.log 2>&1 <<'PY'
import importlib.util, os, sys
root = sys.argv[1]
sys.argv = [sys.argv[0], root, root]
def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(root, "tests/q38_l6", name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
rows = []
for fn in ("q38_l6carry_mutrun.py", "q38_l6steady_mutrun.py", "q38_l6gate_mutrun.py"):
    rows += [(fn, mu[0], mu[1], mu[2]) for mu in load(fn).MUTANTS]
for fn in ("q38_l6_mutrun.py", "q38_l6_engalloc_mutrun.py"):
    for k, lst in load(fn).MUTANTS.items():
        rows += [(fn + ":" + k, mu[0], mu[1], mu[2]) for mu in lst]
bad = [(f, n, open(os.path.join(root, p)).read().count(o)) for f, n, p, o in rows
       if open(os.path.join(root, p)).read().count(o) != 1]
for b in bad:
    print("P7 BAD", *b)
print(f"P7 anchors {len(rows)} bad {len(bad)}")
sys.exit(1 if bad or not rows else 0)
PY
rc=$?
cat /tmp/l6g_p7.log | tail -5
[ $rc = 0 ] || FAIL="$FAIL P7"

[ -z "$FAIL" ] && { echo "Q38_L6_GATE_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_L6_GATE_SUMMARY FAIL mode=full:$FAIL"; exit 1

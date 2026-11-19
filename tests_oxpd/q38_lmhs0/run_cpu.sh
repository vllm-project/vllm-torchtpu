#!/bin/bash
# q38 L4 LMH-STREAM stage 0 CPU proof (CDK CPU pod only; never the Mac or the VM). After DRAFTEMB-BUILD's run_cpu.sh (credited).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_lmhs0/run_cpu.sh smoke | num | regress A | regress B | mut LO HI | full
# head = q38-l4-lmhs0 (VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT / _VERIFY), base = q38-l4-vfy-v5 2e166829a.
# PASS rules, fixed before the run:
# P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC; the src diff set is exactly WANT (3 files).
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the 4 modules the lever touches or reads resolve inside
#     realpath(<tree>/src) on head and base (10/10 INSIDE); with PYTHONPATH = an empty dir they do not (10/10 mutants killed);
#     head's tree with PYTHONPATH = base/src is not INSIDE head (5/5 killed).
# P1  tests/q38_lmhs0/test_lmh_hidden_major.py: exactly 29 passed, 0 failed / skipped / errors; LMHS0_FLAGOFF IDENTICAL x3;
#     LMHS0_CENSUS x2 with bytes=159006720 and target_sha_unchanged=True; LMHS0_SHARDED x2 with hm==stock N/N and
#     hm==full_argmax N/N; LMHS0_ENGAGE draft_calls=2 ... target_calls=0; LMHS0_WIRING combos=32 armed=6.
# P2  compile-key rows of the two flags (-k LMH_HIDDEN_MAJOR): exactly 4 passed, 0 failed.
# P2c the composed flag-off identity tests/l4vfy/test_vfy_v5_flagoff.py (now chaining this lever's resolver): exactly 3 passed.
# P2b collect-only census of every RP_FILES candidate on head and base (printed; a KEY file collecting 0 = FAIL).
# (smoke = P0 P0b P1 P2 P2c P2b)
# PN  (num) tests/q38_lmhs0/q38_lmhs0_numerics.py: last line 'LMHS0_NUM PASS' (top-1 stock==hm and A==B >= 99.99 % of
#     rows at 16 / 32 / 64 / 208 rows; hidden-major method imported from head/src).
# P3  per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over RP_FILES group A (KEY) or B (RUN);
#     a RUN file collecting 0 on both trees is EXCLUDED and named.
# P4  mutants (q38_lmhs0_mutrun.py): 'MUTRUN_ORIGIN ... INSIDE', control ok, every mutant in LO..HI KILLED:
#     'Q38_LMHS0_MUT PASS killed N/N of N range LO-HI of 28' (all 28 in full mode).
# P5  head src unchanged by the run.
# Last line: Q38_LMHS0_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_lmhs0
export JAX_PLATFORMS=cpu
unset VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY TPU_LM_HEAD_HIDDEN_MAJOR
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
NMUT=28
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }
label() { case "$MODE" in mut) echo "mut range ${2:-?}-${3:-?}";; regress) echo "regress ${2:-?}";; *) echo "$MODE";; esac; }
LAB=$(label "$@")
finish() { HS2=$(srch "$H"); echo "P5 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P5"
  [ -z "$FAIL" ] && { echo "Q38_LMHS0_SUMMARY PASS mode=$LAB"; exit 0; }; echo "Q38_LMHS0_SUMMARY FAIL mode=$LAB:$FAIL"; exit 1; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$LAB nproc $(nproc) $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
Files base/src/vllm_torchtpu/runner/utils.py and head/src/vllm_torchtpu/runner/utils.py differ
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (3 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

if [ "$MODE" != regress ] && [ "$MODE" != num ]; then
  echo "== P0b import origin (cd <tree>; PYTHONPATH=src) with its empty-PYTHONPATH and cross-tree mutants"
  MODS="vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.runner.utils vllm_torchtpu.runner.tpu_runner vllm_torchtpu.spec_decode.utils"
  where() { (cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH="$2" python3 -c "import importlib, os; m = importlib.import_module('$3'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1); }
  E=$(mktemp -d /tmp/lmhs0empty.XXXXXX); ins=0; mk=0; xk=0
  for t in head base; do
    want=$(cd "$W/$t" && realpath src)
    for mod in $MODS; do
      got=$(where "$W/$t" src $mod); case "$got" in "$want"/*) v=INSIDE; ins=$((ins + 1));; *) v=FALLBACK;; esac
      echo "ORIGIN $t $mod ${got:-none} $v"
      got=$(where "$W/$t" "$E" $mod); case "$got" in "$want"/*) echo "ORIGIN MUTANT SURVIVED $t $mod (empty PYTHONPATH still INSIDE)";; *) mk=$((mk + 1));; esac
      if [ $t = head ]; then got=$(where "$H" "$B/src" $mod); case "$got" in "$want"/*) echo "ORIGIN XTREE SURVIVED $mod";; *) xk=$((xk + 1));; esac; fi
    done
  done
  echo "P0b INSIDE $ins/10, empty-PYTHONPATH mutants killed $mk/10, cross-tree mutants killed $xk/5"
  [ $ins = 10 ] && [ $mk = 10 ] && [ $xk = 5 ] || FAIL="$FAIL P0b"
fi

if [ "$MODE" = num ]; then
  echo "== PN numerics (16 / 32 / 64 / 208 rows; Qwen3.8 TP8 shard 31056 x 2560 bf16)"
  # tee: each cell's line reaches the job log as it is printed, so a timeout still leaves the finished cells.
  (cd "$H" && env $CPUENV python3 -u tests/q38_lmhs0/q38_lmhs0_numerics.py "$H/src") 2>&1 | tee /tmp/lmhs0_pn.log \
    | grep -a --line-buffered -E '^LMHS0_NUM'; rc=${PIPESTATUS[0]}
  echo "PN rc=$rc"
  [ $rc = 0 ] && [ "$(grep -aE '^LMHS0_NUM (PASS|FAIL)' /tmp/lmhs0_pn.log | tail -1)" = "LMHS0_NUM PASS" ] \
    && grep -aq '^LMHS0_NUM_ORIGIN .* INSIDE ' /tmp/lmhs0_pn.log || { FAIL="$FAIL PN"; tail -40 /tmp/lmhs0_pn.log; }
  finish
fi

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI"
  (cd "$H" && python3 tests/q38_lmhs0/q38_lmhs0_mutrun.py . $LO $HI) > /tmp/lmhs0_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN_ORIGIN|^CONTROL|^MUT |^Q38_LMHS0_MUT' /tmp/lmhs0_p4.log
  echo "P4 rc=$rc"
  [ $rc = 0 ] && grep -aq "^Q38_LMHS0_MUT PASS killed $N/$N of $N range $LO-$HI of $NMUT " /tmp/lmhs0_p4.log \
    && grep -aq '^MUTRUN_ORIGIN runner.utils .* INSIDE$' /tmp/lmhs0_p4.log || { FAIL="$FAIL P4"; tail -80 /tmp/lmhs0_p4.log; }
  finish
fi

KEYF=$(awk '$1 == "KEY" {print $2}' $Q/RP_FILES); RUNF=$(awk '$1 == "RUN" {print $2}' $Q/RP_FILES)
ncol() { (cd "$1" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider -p no:logging "$2" 2>&1 \
  | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1); }

if [ "$MODE" = regress ]; then
  G=${2:?A|B}; [ "$G" = A ] && SET=$KEYF || SET=$RUNF
  RUN=""
  for f in $SET; do
    nh=$(ncol "$H" $f); nb=$(ncol "$B" $f); echo "P2b $f head ${nh:-0} base ${nb:-0}"
    if [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then
      [ "$G" = A ] && { echo "P2b KEY file collects nothing: $f"; FAIL="$FAIL P2b"; } || echo "P2b EXCLUDED (0 on both) $f"
    else RUN="$RUN $f"; fi
  done
  echo "== P3 per-file regression HEAD vs BASE (2e166829a), group $G"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile lmhs0$G "$H" "$B" 2400 $RUN || FAIL="$FAIL P3"
  finish
fi

echo "== P1 test_lmh_hidden_major.py"
(cd "$H" && env $CPUENV $PT -s tests/q38_lmhs0/test_lmh_hidden_major.py) > /tmp/lmhs0_p1.log 2>&1; rc=$?
grep -aE '^LMHS0_' /tmp/lmhs0_p1.log
grep -aE '^(FAILED|ERROR) ' /tmp/lmhs0_p1.log | head -40
echo "P1 rc=$rc summary: $(summ /tmp/lmhs0_p1.log)"
NF=$(grep -acE '^LMHS0_FLAGOFF [a-z_]+ IDENTICAL$' /tmp/lmhs0_p1.log)
NC=$(grep -aE '^LMHS0_CENSUS ' /tmp/lmhs0_p1.log | grep -a 'bytes=159006720' | grep -ac 'target_sha_unchanged=True')
NS=$(grep -aE '^LMHS0_SHARDED ' /tmp/lmhs0_p1.log | grep -aE 'hm==stock ([0-9]+)/\1 ' | grep -acE 'hm==full_argmax ([0-9]+)/\1$')
NE=$(grep -acE '^LMHS0_ENGAGE draft_calls=2 .* target_calls=0$' /tmp/lmhs0_p1.log)
NW=$(grep -acE '^LMHS0_WIRING combos=32 armed=6$' /tmp/lmhs0_p1.log)
echo "P1 flagoff IDENTICAL $NF/3, census $NC/2, sharded $NS/2, engage $NE/1, wiring $NW/1"
[ $rc = 0 ] && exact /tmp/lmhs0_p1.log 29 && [ $NF = 3 ] && [ $NC = 2 ] && [ $NS = 2 ] && [ $NE = 1 ] && [ $NW = 1 ] \
  || { FAIL="$FAIL P1"; tail -80 /tmp/lmhs0_p1.log; }

echo "== P2 compile-key registration rows of the two flags"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k LMH_HIDDEN_MAJOR) > /tmp/lmhs0_p2.log 2>&1; rc=$?
grep -aE 'PASSED|FAILED' /tmp/lmhs0_p2.log | head; echo "P2 rc=$rc summary: $(summ /tmp/lmhs0_p2.log)"
[ $rc = 0 ] && exact /tmp/lmhs0_p2.log 4 || FAIL="$FAIL P2"

echo "== P2c composed flag-off identity (tests/l4vfy/test_vfy_v5_flagoff.py)"
(cd "$H" && env $CPUENV $PT -s tests/l4vfy/test_vfy_v5_flagoff.py) > /tmp/lmhs0_p2c.log 2>&1; rc=$?
grep -aE '^Q38_L4VFY_V5_FLAGOFF|^(FAILED|ERROR) ' /tmp/lmhs0_p2c.log; echo "P2c rc=$rc summary: $(summ /tmp/lmhs0_p2c.log)"
[ $rc = 0 ] && exact /tmp/lmhs0_p2c.log 3 || { FAIL="$FAIL P2c"; tail -40 /tmp/lmhs0_p2c.log; }

echo "== P2b collect-only census (head / base)"
for f in $KEYF $RUNF; do
  nh=$(ncol "$H" $f); nb=$(ncol "$B" $f); echo "P2b $f head ${nh:-0} base ${nb:-0}"
  case " $(echo $KEYF) " in *" $f "*) [ "${nh:-0}" != 0 ] || { echo "P2b KEY file collects nothing on head: $f"; FAIL="$FAIL P2b"; };; esac
done

if [ "$MODE" = full ]; then
  echo "== P3 per-file regression HEAD vs BASE (2e166829a), groups A + B"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile lmhs0F "$H" "$B" 2400 $KEYF $RUNF || FAIL="$FAIL P3"
  echo "== P4 mutants (all $NMUT)"
  (cd "$H" && python3 tests/q38_lmhs0/q38_lmhs0_mutrun.py .) > /tmp/lmhs0_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN_ORIGIN|^CONTROL|^MUT |^Q38_LMHS0_MUT' /tmp/lmhs0_p4.log
  [ $rc = 0 ] && grep -aq "^Q38_LMHS0_MUT PASS killed $NMUT/$NMUT of $NMUT " /tmp/lmhs0_p4.log || FAIL="$FAIL P4"
fi
finish

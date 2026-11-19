#!/bin/bash
# q38 L4 X-D14-DRAFT-2 (c) + EMB-REP-TGT CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_draftemb/run_cpu.sh smoke | regress A | regress B | mut LO HI | full
# head = q38-l4-draftemb (VLLM_TPU_DRAFT_EMBED_REPL + VLLM_TPU_TARGET_EMBED_REPL), base = q38-l4-draftasm-vfyhook 9b8d39a74.
# PASS rules, fixed before the run:
# P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC; the src diff set is exactly WANT (3 files).
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the 4 modules the lever reads resolve inside
#     realpath(<tree>/src) on head and base (10/10 INSIDE); with PYTHONPATH = an empty dir they read FALLBACK
#     (10/10 mutants killed); head's harness with PYTHONPATH = base/src is not INSIDE head (5/5 killed).
# P1  tests/q38_draftemb/test_draft_embed_repl.py + test_target_embed_repl.py: exactly 27 passed (18 + 9), 0 failed /
#     skipped / errors; 4 DRAFTEMB_REPLAY lines each with agree=N/N and bits_mismatch all 0; 4 DRAFTEMB_CENSUS lines;
#     2 TGTEMB_BITS, 2 TGTEMB_CENSUS and 1 TGTEMB_SITES line.
# P1b q38_draftemb_fit.py: last line 'DRAFTEMB_FIT PASS'.
# P2  compile-key rows of VLLM_TPU_DRAFT_EMBED_REPL and VLLM_TPU_TARGET_EMBED_REPL: exactly 4 passed, 0 failed.
# P2b collect-only census of every RP_FILES candidate on head and base (printed; a KEY file collecting 0 = FAIL).
# (smoke = P0 P0b P1 P1b P2 P2b)
# P3  per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over RP_FILES group A (KEY) or B (RUN);
#     a RUN file collecting 0 on both trees is EXCLUDED and named.
# P4  mutants (q38_draftemb_mutrun.py): 'MUTRUN_ORIGIN ... INSIDE', control ok, every mutant in LO..HI KILLED:
#     'Q38_DRAFTEMB_MUT PASS killed N/N of N range LO-HI of 30' (all 30 in full mode).
# P5  head src unchanged by the run.
# Last line: Q38_DRAFTEMB_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_draftemb
export JAX_PLATFORMS=cpu
unset VLLM_TPU_DRAFT_EMBED_REPL VLLM_TPU_TARGET_EMBED_REPL VLLM_TPU_DRAFT_EMBED_H8 VLLM_TPU_QSA_DRAFT_SHARED_SELECTION VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }
label() { case "$MODE" in mut) echo "mut range ${2:-?}-${3:-?}";; regress) echo "regress ${2:-?}";; *) echo "$MODE";; esac; }
LAB=$(label "$@")
finish() { HS2=$(srch "$H"); echo "P5 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P5"
  [ -z "$FAIL" ] && { echo "Q38_DRAFTEMB_SUMMARY PASS mode=$LAB"; exit 0; }; echo "Q38_DRAFTEMB_SUMMARY FAIL mode=$LAB:$FAIL"; exit 1; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$LAB $(python3 -c 'import sys, torch, jax, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "jax", jax.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/__init__.py and head/src/vllm_torchtpu/__init__.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py and head/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py differ
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (3 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

if [ "$MODE" != regress ]; then
  echo "== P0b import origin (cd <tree>; PYTHONPATH=src) with its empty-PYTHONPATH and cross-tree mutants"
  MODS="vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.models.vllm.qwen4_exp.mtp vllm_torchtpu.spec_decode.utils vllm_torchtpu.spec_decode.draft_asm"
  where() { (cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH="$2" python3 -c "import importlib, os; m = importlib.import_module('$3'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1); }
  E=$(mktemp -d /tmp/dembempty.XXXXXX); ins=0; mk=0; xk=0
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

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI"
  (cd "$H" && python3 tests/q38_draftemb/q38_draftemb_mutrun.py . $LO $HI) > /tmp/demb_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN_ORIGIN|^CONTROL|^MUT |^Q38_DRAFTEMB_MUT' /tmp/demb_p4.log
  echo "P4 rc=$rc"
  [ $rc = 0 ] && grep -aq "^Q38_DRAFTEMB_MUT PASS killed $N/$N of $N range $LO-$HI of 30 " /tmp/demb_p4.log \
    && grep -aq '^MUTRUN_ORIGIN mtp .* INSIDE$' /tmp/demb_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/demb_p4.log; }
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
  echo "== P3 per-file regression HEAD vs BASE (9b8d39a74), group $G"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile demb$G "$H" "$B" 2400 $RUN || FAIL="$FAIL P3"
  finish
fi

echo "== P1 test_draft_embed_repl.py + test_target_embed_repl.py"
(cd "$H" && env $CPUENV $PT -s tests/q38_draftemb/test_draft_embed_repl.py tests/q38_draftemb/test_target_embed_repl.py) > /tmp/demb_p1.log 2>&1; rc=$?
grep -aE '^DRAFTEMB_(REPLAY|CENSUS)|^TGTEMB_' /tmp/demb_p1.log
grep -aE '^(FAILED|ERROR) ' /tmp/demb_p1.log | head -40
echo "P1 rc=$rc summary: $(summ /tmp/demb_p1.log)"
NR=$(grep -acE '^DRAFTEMB_REPLAY .* bits_mismatch embed=0 fused=0 sample=0 multi=0$' /tmp/demb_p1.log)
NA=$(grep -aE '^DRAFTEMB_REPLAY ' /tmp/demb_p1.log | grep -acE 'agree=([0-9]+)/\1 \(100\.000%\)')
NC=$(grep -acE '^DRAFTEMB_CENSUS ' /tmp/demb_p1.log)
NTB=$(grep -acE '^TGTEMB_BITS .* bitwise_vs_stock=8/8 ' /tmp/demb_p1.log); NTC=$(grep -acE '^TGTEMB_CENSUS ' /tmp/demb_p1.log)
NTS=$(grep -acE '^TGTEMB_SITES ' /tmp/demb_p1.log)
echo "P1 replay lines bitwise $NR/4, full agreement $NA/4, census lines $NC/4; target bits $NTB/2, census $NTC/2, sites $NTS/1"
[ $rc = 0 ] && exact /tmp/demb_p1.log 27 && [ $NR = 4 ] && [ $NA = 4 ] && [ $NC = 4 ] && [ $NTB = 2 ] && [ $NTC = 2 ] \
  && [ $NTS = 1 ] || { FAIL="$FAIL P1"; tail -80 /tmp/demb_p1.log; }

echo "== P1b HBM fit (real pooled_pool_blocks on measured inputs)"
(cd "$H" && env $CPUENV python3 tests/q38_draftemb/q38_draftemb_fit.py) > /tmp/demb_p1b.log 2>&1; rc=$?
grep -aE '^FIT|^DRAFTEMB_FIT' /tmp/demb_p1b.log
[ $rc = 0 ] && [ "$(tail -1 /tmp/demb_p1b.log)" = "DRAFTEMB_FIT PASS" ] || { FAIL="$FAIL P1b"; tail -30 /tmp/demb_p1b.log; }

echo "== P2 compile-key registration rows of VLLM_TPU_DRAFT_EMBED_REPL and VLLM_TPU_TARGET_EMBED_REPL"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "DRAFT_EMBED_REPL or TARGET_EMBED_REPL") > /tmp/demb_p2.log 2>&1; rc=$?
grep -aE 'PASSED|FAILED' /tmp/demb_p2.log | head; echo "P2 rc=$rc summary: $(summ /tmp/demb_p2.log)"
[ $rc = 0 ] && exact /tmp/demb_p2.log 4 || FAIL="$FAIL P2"

echo "== P2b collect-only census (head / base)"
for f in $KEYF $RUNF; do
  nh=$(ncol "$H" $f); nb=$(ncol "$B" $f); echo "P2b $f head ${nh:-0} base ${nb:-0}"
  case " $(echo $KEYF) " in *" $f "*) [ "${nh:-0}" != 0 ] || { echo "P2b KEY file collects nothing on head: $f"; FAIL="$FAIL P2b"; };; esac
done

if [ "$MODE" = full ]; then
  echo "== P3 per-file regression HEAD vs BASE (9b8d39a74), groups A + B"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile dembF "$H" "$B" 2400 $KEYF $RUNF || FAIL="$FAIL P3"
  echo "== P4 mutants (all 30)"
  (cd "$H" && python3 tests/q38_draftemb/q38_draftemb_mutrun.py .) > /tmp/demb_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN_ORIGIN|^CONTROL|^MUT |^Q38_DRAFTEMB_MUT' /tmp/demb_p4.log
  [ $rc = 0 ] && grep -aq '^Q38_DRAFTEMB_MUT PASS killed 30/30 of 30 ' /tmp/demb_p4.log || FAIL="$FAIL P4"
fi
finish

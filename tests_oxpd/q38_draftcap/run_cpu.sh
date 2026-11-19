#!/bin/bash
# q38 L4 DRAFT-CAP CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_draftcap/run_cpu.sh smoke          P0 P0b P1 P1c P2 P6
#   bash head/tests/q38_draftcap/run_cpu.sh regress A|B    P0 P0b P3 (half A = the first half of RP_FILES, B = the rest) P6
#   bash head/tests/q38_draftcap/run_cpu.sh mut LO HI      P0 P0b P4 over mutants LO..HI P6
# head = q38-l4-draftcap (EXPECT_COMMIT, src EXPECT_SRC); base = q38-l4-draft1b 29ea1e98d (src EXPECT_BASE_SRC).
# PASS rules, fixed before the run:
# P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC; the src diff set is exactly WANT_DIFF.
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the touched modules resolve inside realpath(<tree>/src)
#     on head (6) and base (5, no draft_cap): 11/11 INSIDE; the same imports with PYTHONPATH = an empty dir read
#     FALLBACK: 11/11 mutants killed.
# P1  new tests on head, each file exactly EXPECT_P1 (file:passed), 0 failed / skipped / errors; marker line counts
#     exactly EXPECT_LINES (prefix:count).
# P1c DRAFT1-B's DH-D and flag-off suites with the flag unset: on head each has the outcome it has on base (printed both;
#     a head failure base lacks FAILS).
# P2  compile-key rows of VLLM_TPU_DRAFT_ASM_CAP: exactly 2 passed, 0 failed.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over the RP_FILES half. A candidate whose base
#     screen counts 0 tests is EXCLUDED and named; a KEY file may not be excluded.
# P4  q38_draftcap_mutrun.py: 'MUTRUN_ORIGIN ... INSIDE', control ok, 'Q38_DRAFTCAP_MUT PASS killed N/N of N range LO-HI'.
# P6  head src unchanged by the run.
# Last line: Q38_DRAFTCAP_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_draftcap
export JAX_PLATFORMS=cpu
unset VLLM_TPU_DRAFT_ASM_FOLD VLLM_TPU_DRAFT_ASM_CHECK_STEPS VLLM_TPU_DRAFT_ASM_AB VLLM_TPU_DRAFT_ASM_CAP
unset VLLM_TPU_DRAFT_ASM_CARRY_LOOP VLLM_TPU_DRAFT_ASM_ONE VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB VLLM_TPU_DRAFT_ASM_ONE_AB
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort "$Q/WANT_DIFF")
WANT_N=$(echo "$WANT" | wc -l | tr -d ' ')
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($WANT_N entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import origin (cd <tree>; PYTHONPATH=src) and its empty-PYTHONPATH mutant"
EMPTY=$(mktemp -d /tmp/dcap_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
nin=0; nkill=0; nmod=0
for t in head base; do
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.spec_decode.eagle3 vllm_torchtpu.spec_decode.draft_asm \
             vllm_torchtpu.spec_decode.draft_one vllm_torchtpu.spec_decode.draft_cap; do
    [ $t = base ] && [ $mod = vllm_torchtpu.spec_decode.draft_cap ] && continue
    nmod=$((nmod + 1))
    r=$(origin "$W/$t" $mod src); echo "ORIGIN $t $mod ${r#* } ${r%% *}"; [ "${r%% *}" = INSIDE ] && nin=$((nin + 1))
    m=$(origin "$W/$t" $mod "$EMPTY"); echo "ORIGIN $t:MUTANT $mod ${m#* } ${m%% *}"; [ "${m%% *}" = FALLBACK ] && nkill=$((nkill + 1))
  done
done
echo "P0b inside $nin/$nmod; empty-PYTHONPATH mutants reading FALLBACK $nkill/$nmod"
[ $nmod = 11 ] && [ $nin = 11 ] && [ $nkill = 11 ] || FAIL="$FAIL P0b"

finish() {
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_DRAFTCAP_SUMMARY PASS mode=$1"; exit 0; }
  echo "Q38_DRAFTCAP_SUMMARY FAIL mode=$1:$FAIL"; exit 1
}

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI (slice)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_draftcap/q38_draftcap_mutrun.py . $LO $HI) > /tmp/dcap_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_DRAFTCAP_MUT|^(FAILED|ERROR|E) ' /tmp/dcap_p4.log | cut -c1-700
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  TOTAL=$(python3 -c "import ast,sys; t=ast.parse(open('$H/tests/q38_draftcap/q38_draftcap_mutrun.py').read()); print(next(len(n.value.elts) for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MUTANTS'))")
  [ $rc = 0 ] && grep -aq "^Q38_DRAFTCAP_MUT PASS killed $N/$N of $N range $LO-$HI of $TOTAL " /tmp/dcap_p4.log \
    && grep -aq '^MUTRUN_ORIGIN .* INSIDE$' /tmp/dcap_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/dcap_p4.log; }
  finish "mut range $LO-$HI"
fi

if [ "$MODE" = regress ]; then
  HALF=${2:?A or B}
  echo "== P3 base screen of the RP_FILES half $HALF (4 at a time)"
  KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
  ALL=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
  NALL=$(echo "$ALL" | wc -l | tr -d ' '); NA=$(( (NALL + 1) / 2 ))
  if [ "$HALF" = A ]; then CANDS=$(echo "$ALL" | head -n $NA); else CANDS=$(echo "$ALL" | tail -n +$((NA + 1))); fi
  screen1() { local f=$1 log=/tmp/dcap_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
    (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
    echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
  export -f screen1 summ; export B CPUENV PT
  echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/dcap_screen.txt
  sed 's/^/P3 /' /tmp/dcap_screen.txt
  RUN=""
  for f in $CANDS; do
    log=/tmp/dcap_scr_$(echo $f | tr '/.' '__').log
    n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
    if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
    else case " $(echo $KEY) " in *" $f "*) echo "P3 KEY file counts 0 on base: $f"; FAIL="$FAIL P3-key";; *) echo "P3 EXCLUDED (0 counted on base) $f";; esac; fi
  done
  echo "== P3 per-file regression HEAD vs BASE (29ea1e98d) over $(echo $RUN | wc -w) files (half $HALF)"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile draftcap "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"
  finish "regress $HALF"
fi

echo "== P1 new tests (head)"
for f in $(cat $Q/EXPECT_P1); do
  n=${f%%:*}; want=${f##*:}; t0=$(date +%s)
  (cd "$H" && env $CPUENV $PT -s tests/q38_draftcap/test_$n.py) > /tmp/dcap_p1_$n.log 2>&1; rc=$?
  echo "P1 $n rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ /tmp/dcap_p1_$n.log) (want $want passed)"
  grep -aE '^(FAILED|ERROR) ' /tmp/dcap_p1_$n.log | cut -c1-600
  [ $rc = 0 ] && exact /tmp/dcap_p1_$n.log $want || { FAIL="$FAIL P1-$n"; grep -aE '^E |Error|error' /tmp/dcap_p1_$n.log | head -80; tail -60 /tmp/dcap_p1_$n.log; }
done
cat /tmp/dcap_p1_*.log | grep -aE '^DRAFTCAP_[A-Z]+ ' | cut -c1-900
for f in $(cat $Q/EXPECT_LINES); do
  pre=${f%%:*}; want=${f##*:}
  got=$(cat /tmp/dcap_p1_*.log | grep -ac "^DRAFTCAP_$pre ")
  echo "P1 lines DRAFTCAP_$pre $got (want $want)"
  [ "$got" = "$want" ] || FAIL="$FAIL P1-lines-$pre"
done

echo "== P1c DRAFT1-B's DH-D and flag-off suites (head vs base, flag unset)"
for tid in tests/q38_draft1b/test_draft1b_dhd.py tests/q38_draft1b/test_draft1b_flagoff.py; do
  (cd "$H" && env $CPUENV $PT "$tid") > /tmp/dcap_p1c_h.log 2>&1; rh=$?
  (cd "$B" && env $CPUENV $PT "$tid") > /tmp/dcap_p1c_b.log 2>&1; rb=$?
  echo "P1c $tid head rc=$rh ($(summ /tmp/dcap_p1c_h.log)) base rc=$rb ($(summ /tmp/dcap_p1c_b.log))"
  [ $rh = 0 ] || [ $rb != 0 ] || { FAIL="$FAIL P1c"; tail -30 /tmp/dcap_p1c_h.log; }
done

echo "== P2 compile-key registration rows of VLLM_TPU_DRAFT_ASM_CAP"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "DRAFT_ASM_CAP") > /tmp/dcap_p2.log 2>&1; rc=$?
echo "P2 rc=$rc summary: $(summ /tmp/dcap_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/dcap_p2.log passed)" = 2 ] && [ "$(count /tmp/dcap_p2.log failed)" = 0 ] || { FAIL="$FAIL P2"; tail -30 /tmp/dcap_p2.log; }

finish smoke

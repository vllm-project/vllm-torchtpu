#!/bin/bash
# q38 L4 T1-K1FUSE CPU proof (CDK CPU pod only; never the Mac or the VM). K1FUSE-BUILD (helper of L4-E).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_k1fuse/run_cpu.sh smoke          P0 P0b P1 P1c P2 P6
#   bash head/tests/q38_k1fuse/run_cpu.sh regress A|B    P0 P0b P3 (RP_FILES half A / B) P6
#   bash head/tests/q38_k1fuse/run_cpu.sh mut LO HI      P0 P0b P4 over mutants LO..HI P6
# head = q38-l4-k1fuse (EXPECT_COMMIT, src EXPECT_SRC); base = the composed base 7867a18ef (src EXPECT_BASE_SRC) =
# q38-l4-vfy-v5 2e166829a + q38-l4-draft1b 29ea1e98d.
# PASS rules, fixed before the run:
# P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC; the src diff set is exactly WANT_DIFF.
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the touched modules resolve inside realpath(<tree>/src)
#     on head (8) and base (7, no k1fuse): 15/15 INSIDE; with PYTHONPATH = an empty dir they read FALLBACK: 15/15
#     (G01: the import guard's own mutant).
# P1  new tests on head, each file exactly EXPECT_P1 (file:passed), 0 failed / skipped / errors; marker line counts
#     exactly EXPECT_LINES (prefix:count); every K1FUSE_CENSUS line reads 'calls 2 -> 1'.
# P1c the composed levers' own suites (VFY-ONE, DRAFT1-B, DRAFT-ASM) with their own pytest flags, head vs base: each
#     file counts >= 1 test on both, and head has no failing test id that base lacks.
# P2  compile-key rows of the three K1FUSE variables: exactly 6 passed, 0 failed.
# P3  per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over the RP_FILES half. A candidate whose base
#     screen counts 0 tests is EXCLUDED and named; a KEY file may not be excluded.
# P4  q38_k1fuse_mutrun.py: 'MUTRUN_ORIGIN ... INSIDE', control ok, 'Q38_K1FUSE_MUT PASS killed N/N of N range LO-HI'.
# P6  head src unchanged by the run.
# Last line: Q38_K1FUSE_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_k1fuse
export JAX_PLATFORMS=cpu
unset VLLM_TPU_DRAFT_ASM_FOLD VLLM_TPU_DRAFT_ASM_CHECK_STEPS VLLM_TPU_DRAFT_ASM_AB VLLM_TPU_VPS_VFY_ONE
unset VLLM_TPU_DRAFT_ASM_CARRY_LOOP VLLM_TPU_DRAFT_ASM_ONE VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB VLLM_TPU_DRAFT_ASM_ONE_AB
unset VLLM_TPU_K1FUSE VLLM_TPU_K1FUSE_AB VLLM_TPU_K1FUSE_CHECK VLLM_TPU_VPS_VFY_ONE_CHECK VLLM_TPU_VPS_VFY_ONE_AB
unset VLLM_TPU_VPS_FUSED_READ_OFFSETS VLLM_TPU_L6_AB_STEPS VLLM_TPU_SERVED_AB
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
PTN="python3 -m pytest -q -p no:cacheprovider --noconftest -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }
fails() { grep -aE '^(FAILED|ERROR) ' "$1" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' | sort -u; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort "$Q/WANT_DIFF")
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import origin (cd <tree>; PYTHONPATH=src) and its empty-PYTHONPATH mutant (G01)"
EMPTY=$(mktemp -d /tmp/k1f_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
nin=0; nkill=0; nmod=0
for t in head base; do
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.compilation.tpu_compiler vllm_torchtpu.runner.tpu_runner \
             vllm_torchtpu.spec_decode.eagle3 vllm_torchtpu.spec_decode.draft_one vllm_torchtpu.runner.vfy_one \
             vllm_torchtpu.spec_decode.k1fuse; do
    [ $t = base ] && [ $mod = vllm_torchtpu.spec_decode.k1fuse ] && continue
    nmod=$((nmod + 1))
    r=$(origin "$W/$t" $mod src); echo "ORIGIN $t $mod ${r#* } ${r%% *}"; [ "${r%% *}" = INSIDE ] && nin=$((nin + 1))
    m=$(origin "$W/$t" $mod "$EMPTY"); echo "ORIGIN $t:G01 $mod ${m#* } ${m%% *}"; [ "${m%% *}" = FALLBACK ] && nkill=$((nkill + 1))
  done
done
echo "P0b inside $nin/$nmod; G01 empty-PYTHONPATH reads FALLBACK $nkill/$nmod"
[ $nmod = 15 ] && [ $nin = 15 ] && [ $nkill = 15 ] || FAIL="$FAIL P0b"

finish() {
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_K1FUSE_SUMMARY PASS mode=$1"; exit 0; }
  echo "Q38_K1FUSE_SUMMARY FAIL mode=$1:$FAIL"; exit 1
}

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI (slice)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_k1fuse/q38_k1fuse_mutrun.py . $LO $HI) > /tmp/k1f_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_K1FUSE_MUT|^(FAILED|ERROR|E) ' /tmp/k1f_p4.log | cut -c1-700
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  TOTAL=$(python3 -c "import ast; t=ast.parse(open('$H/tests/q38_k1fuse/q38_k1fuse_mutrun.py').read()); print(next(len(n.value.elts) for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MUTANTS'))")
  [ $rc = 0 ] && grep -aq "^Q38_K1FUSE_MUT PASS killed $N/$N of $N range $LO-$HI of $TOTAL " /tmp/k1f_p4.log \
    && grep -aq '^MUTRUN_ORIGIN .* INSIDE$' /tmp/k1f_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/k1f_p4.log; }
  finish "mut range $LO-$HI"
fi

if [ "$MODE" = regress ]; then
  HALF=${2:?A or B}
  echo "== P3 base screen of the RP_FILES half $HALF (4 at a time)"
  KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
  ALL=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
  NALL=$(echo "$ALL" | wc -l | tr -d ' '); NA=$(( (NALL + 1) / 2 ))
  if [ "$HALF" = A ]; then CANDS=$(echo "$ALL" | head -n $NA); else CANDS=$(echo "$ALL" | tail -n +$((NA + 1))); fi
  screen1() { local f=$1 log=/tmp/k1f_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
    (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
    echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
  export -f screen1 summ; export B CPUENV PT
  echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/k1f_screen.txt
  sed 's/^/P3 /' /tmp/k1f_screen.txt
  RUN=""
  for f in $CANDS; do
    log=/tmp/k1f_scr_$(echo $f | tr '/.' '__').log
    n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
    if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
    else case " $(echo $KEY) " in *" $f "*) echo "P3 KEY file counts 0 on base: $f"; FAIL="$FAIL P3-key";; *) echo "P3 EXCLUDED (0 counted on base) $f";; esac; fi
  done
  echo "== P3 per-file regression HEAD vs BASE (7867a18ef) over $(echo $RUN | wc -w) files (half $HALF)"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile k1fuse "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"
  finish "regress $HALF"
fi

echo "== P1 new tests (head)"
for f in $(cat $Q/EXPECT_P1); do
  n=${f%%:*}; want=${f##*:}; t0=$(date +%s)
  file=tests/q38_k1fuse/test_k1fuse.py; [ "$n" = k1fuse ] || file=tests/q38_k1fuse/test_k1fuse_$n.py
  (cd "$H" && env $CPUENV $PT -s $file) > /tmp/k1f_p1_$n.log 2>&1; rc=$?
  echo "P1 $n rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ /tmp/k1f_p1_$n.log) (want $want passed)"
  grep -aE '^(FAILED|ERROR) ' /tmp/k1f_p1_$n.log | cut -c1-600
  [ $rc = 0 ] && exact /tmp/k1f_p1_$n.log $want || { FAIL="$FAIL P1-$n"; grep -aE '^E |Error|error' /tmp/k1f_p1_$n.log | head -80; tail -80 /tmp/k1f_p1_$n.log; }
done
cat /tmp/k1f_p1_*.log | grep -aE '^(K1FUSE_[A-Z]+|Q38_K1FUSE_COUNTS) ' | cut -c1-900
for f in $(cat $Q/EXPECT_LINES); do
  pre=${f%%:*}; want=${f##*:}
  got=$(cat /tmp/k1f_p1_*.log | grep -ac "^K1FUSE_$pre ")
  echo "P1 lines K1FUSE_$pre $got (want $want)"
  [ "$got" = "$want" ] || FAIL="$FAIL P1-lines-$pre"
done
bad=$(cat /tmp/k1f_p1_*.log | grep -a '^K1FUSE_CENSUS ' | grep -avc 'calls 2 -> 1 ')
echo "P1 census lines not reading 'calls 2 -> 1': $bad"
[ "$bad" = 0 ] || FAIL="$FAIL P1-census"

echo "== P1c the composed levers' suites with their own flags (head vs base 7867a18ef)"
p1c() {  # p1c <flags: PT|PTN> <pythonpath> <file>
  local cmd=$PT; [ "$1" = PTN ] && cmd=$PTN
  (cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=$2 $cmd "$3") > /tmp/k1f_p1c_h.log 2>&1
  (cd "$B" && env JAX_PLATFORMS=cpu PYTHONPATH=$2 $cmd "$3") > /tmp/k1f_p1c_b.log 2>&1
  local nh nb new
  nh=$(summ /tmp/k1f_p1c_h.log | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
  nb=$(summ /tmp/k1f_p1c_b.log | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
  new=$(comm -23 <(fails /tmp/k1f_p1c_h.log) <(fails /tmp/k1f_p1c_b.log) | tr '\n' ' ')
  echo "P1c $3 head ($(summ /tmp/k1f_p1c_h.log)) base ($(summ /tmp/k1f_p1c_b.log)) new failures: ${new:-none}"
  if [ "${nh:-0}" -lt 1 ] || [ "${nb:-0}" -lt 1 ] || [ -n "$new" ]; then FAIL="$FAIL P1c"; tail -30 /tmp/k1f_p1c_h.log; fi
}
p1c PTN src tests/l4vfy/test_vfy_one.py
p1c PTN src tests/l4vfy/test_vfy_agree.py
p1c PT src tests/l4vfy/test_vfy_dha_compose.py
p1c PTN src tests/l4vfy/test_vfy_v5_flagoff.py
p1c PTN src tests/l4vfy/test_vfy_flagoff.py
for f in units stagea multirank dhd flagoff; do p1c PT src tests/q38_draft1b/test_draft1b_$f.py; done
for f in units propose multirank; do p1c PT src:tests/q38_draftasm tests/q38_draftasm/test_draftasm_$f.py; done

echo "== P2 compile-key registration rows of the K1FUSE variables"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "K1FUSE") > /tmp/k1f_p2.log 2>&1; rc=$?
echo "P2 rc=$rc summary: $(summ /tmp/k1f_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/k1f_p2.log passed)" = 6 ] && [ "$(count /tmp/k1f_p2.log failed)" = 0 ] || { FAIL="$FAIL P2"; tail -30 /tmp/k1f_p2.log; }

finish smoke

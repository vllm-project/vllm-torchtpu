#!/bin/bash
# q38 L4 VFY-ONE CPU proof (CDK CPU pod only; never the Mac or the VM). VFY-BUILD (helper of L4-E), 2026-11-14.
# After D83S-BUILD's tests/q38_d83s/run_cpu.sh (credited).
# Head = q38-l4-vfy: VLLM_TPU_VPS_VFY_ONE (T-3 VFY-ONE + DH-C at value 2, default off) on
#        base = q38-l4-vfy f24edcc94 (src dfb0b699ea) = q38-L1-v3c b551e9b84 + D83S's commits cherry-picked.
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_vfy/run_cpu.sh smoke | mut <LO> <HI> | regress <LO> <HI>   (regress: RP_FILES entries LO..HI)
# Every PASS condition, stated before the run:
#   P0  (all modes) versions; src hashes head == EXPECT_SRC and base == EXPECT_BASE_SRC; the exact src diff set (envs.py,
#       compilation/tpu_compiler.py, core/l6_ab.py, runner/tpu_runner.py, spec_decode/eagle3.py differ; only in head:
#       runner/vfy_one.py).
#   P0o (smoke) import origin: vllm_torchtpu, .envs, .compilation.tpu_compiler, .runner.vfy_one, .runner.tpu_runner
#       resolve INSIDE head/src (base: vllm_torchtpu, .envs, .runner.tpu_runner INSIDE base/src) under cwd <tree> +
#       PYTHONPATH=src; mutant: an empty PYTHONPATH must read FALLBACK on head and on base. run_vfy.py repeats the check
#       and adds the RESOLUTION mutant (raise appended to a staged copy of vfy_one.py).
#   P1  q38_l4vfy/run_vfy.py on head: smoke = clean (IMPORT + CLEAN of test_vfy_one.py / test_vfy_agree.py with exact
#       counts + D83's test_vps_ro_fold.py 86 passed); mut LO HI = the same + mutants LO..HI, every one KILLED. Exit 0
#       and the summary line VERDICT=PASS (+ mutants_killed=N/N invalid=0 in mut).
#   P2  (smoke) compile-key registration on head: the VFY rows exactly 7 passed (registered + parses x3, hashed x1,
#       host-only x2, values 1 and 2 share one hash x1); the whole file 0 failed; and a probe of head's
#       _TPU_COMPILE_ENV_IGNORED: VPS_VFY_ONE NOT in it (hashed), _CHECK and _AB in it; l6_ab.LEVERS["vfyone"] is _AB.
#   P2b (regress) collect-only census of RP_FILES entries LO..HI on head and base (a KEY file must collect on both).
#   P3  (regress) per-file regression HEAD vs BASE over the union with L3's regress_perfile v2 (4fb6f596): a missing
#       summary, 0 tests, a collection error or a new failing id on head = FAIL.
#   P6  (all modes) head src unchanged by the run. Last line: Q38_L4VFY_PROOF PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}; LO=${2:-0}; HI=${3:-0}
case "$MODE" in smoke) ;; mut|regress) [ "$LO" -ge 1 ] && [ "$HI" -ge "$LO" ] || { echo "Q38_L4VFY_PROOF FAIL mode=$MODE: bad range"; exit 2; } ;;
  *) echo "Q38_L4VFY_PROOF FAIL mode=$MODE: unknown mode"; exit 2 ;; esac
TAG=$MODE; [ "$MODE" != smoke ] && TAG="$MODE range $LO-$HI"
W=$(pwd); H=$W/head; B=$W/base; E=$H/tests/q38_vfy
export JAX_PLATFORMS=cpu
unset VLLM_TPU_VPS_VFY_ONE VLLM_TPU_VPS_VFY_ONE_CHECK VLLM_TPU_VPS_VFY_ONE_AB VLLM_TPU_VPS_FUSED_READ_OFFSETS \
      VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT VLLM_TPU_VPS_WARM_DRAWS \
      VLLM_TPU_AOT_SKIP_GUARD_CHECK VLLM_TPU_HOST_PHASES_SUBS VLLM_TPU_FAST_COMPILED_CALL VLLM_TPU_FAST_CALL_GUARD_MEMO \
      VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE VLLM_TPU_DRAFT_EMBED_H8 VLLM_TPU_VPS_TOPK_KEYS VLLM_TPU_L6_AB_STEPS \
      VLLM_TPU_L6_AB_PERIOD_MS VLLM_TPU_L6_AB_CLASSES VLLM_TPU_L6_AB_CONFIRM VLLM_TPU_L6_AB_DIR VLLM_TPU_SERVED_AB
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
L=/tmp/l4vfy_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() {
  [ "$(count "$1" passed)" = "$2" ] && [ "$(count "$1" failed)" = 0 ] && [ "$(count "$1" skipped)" = 0 ] \
    && [ "$(count "$1" error)" = 0 ] && [ "$(count "$1" errors)" = 0 ]
}

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$TAG $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1) nproc $(nproc)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_SRC)); base src $BS (expect $(cat $E/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $E/EXPECT_SRC)" ] && [ "$BS" = "$(cat $E/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'EOL'
Files base/src/vllm_torchtpu/compilation/tpu_compiler.py and head/src/vllm_torchtpu/compilation/tpu_compiler.py differ
Files base/src/vllm_torchtpu/core/l6_ab.py and head/src/vllm_torchtpu/core/l6_ab.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
Files base/src/vllm_torchtpu/spec_decode/eagle3.py and head/src/vllm_torchtpu/spec_decode/eagle3.py differ
Only in head/src/vllm_torchtpu/runner: vfy_one.py
EOL
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (6 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

if [ "$MODE" = smoke ]; then
  echo "== P0o import origin (cwd <tree>, PYTHONPATH=src; mutant: empty PYTHONPATH must read FALLBACK)"
  origin() {
    local lab=$1 t=$2 mod=$3 pp=${4:-src} want got v
    want=$(cd "$t" && realpath src)
    got=$(cd "$t" && env JAX_PLATFORMS=cpu PYTHONPATH="$pp" python3 -c "import importlib, os; m = importlib.import_module('$mod'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
    case "$got" in "$want"/*) v=INSIDE ;; *) v=FALLBACK ;; esac
    echo "ORIGIN $lab $mod ${got:-none} $v"
    [ "$v" = INSIDE ]
  }
  NO=0; NM=0
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.compilation.tpu_compiler vllm_torchtpu.runner.vfy_one vllm_torchtpu.runner.tpu_runner; do
    origin head "$H" $mod && NO=$((NO + 1)) || FAIL="$FAIL P0o-head:$mod"
  done
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.runner.tpu_runner; do
    origin base "$B" $mod && NO=$((NO + 1)) || FAIL="$FAIL P0o-base:$mod"
  done
  EMPTY=$(mktemp -d /tmp/l4vfy_empty.XXXXXX)
  for t in head base; do
    if origin "$t:MUTANT" "$W/$t" vllm_torchtpu "$EMPTY"; then echo "P0o MUTANT SURVIVED $t"; FAIL="$FAIL P0o-mutant:$t"
    else echo "P0o MUTANT KILLED $t (an empty PYTHONPATH reads FALLBACK)"; NM=$((NM + 1)); fi
  done
  rm -rf "$EMPTY"
  echo "P0o inside $NO/8 mutants killed $NM/2"
fi

if [ "$MODE" != regress ]; then
  if [ "$MODE" = smoke ]; then ARGS=clean; else ARGS="mutants $LO $HI"; fi
  echo "== P1 q38_l4vfy/run_vfy.py $ARGS (head)"
  (cd "$H" && python3 -u q38_l4vfy/run_vfy.py $ARGS) > $L/p1.log 2>&1; rc=$?
  grep -aE '^(SRC-|IMPORT|CLEAN|  Q38_L4|MUTANT|  md5|Q38_L4VFY_SUMMARY)' $L/p1.log
  echo "P1 rc=$rc"
  SUM=$(grep -a '^Q38_L4VFY_SUMMARY' $L/p1.log | tail -1)
  if [ $rc = 0 ] && echo "$SUM" | grep -q "src=$HS import=PASS clean=PASS" && echo "$SUM" | grep -q 'VERDICT=PASS'; then
    if [ "$MODE" = mut ] && ! echo "$SUM" | grep -qE "range=$LO-$HI mutants_killed=([0-9]+)/\1 invalid=0"; then FAIL="$FAIL P1-mutants"; fi
  else FAIL="$FAIL P1"; tail -150 $L/p1.log; fi
fi

if [ "$MODE" = smoke ]; then
  echo "== P2 compile-key registration (the VFY rows; the whole file; the hashed / ignored probe)"
  (cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src:q38_l4vfy $PT -p q38_l4vfy_origin \
     tests/compilation/test_tpu_compile_key_registration.py -k "VFY_ONE or vfy_one" -vv) > $L/p2.log 2>&1; rc=$?
  grep -aE 'PASSED|FAILED|ERROR|^Q38_L4VFY_ORIGIN' $L/p2.log | sed 's/^/P2 /'
  echo "P2 rc=$rc summary: $(summ $L/p2.log)"
  OL=$(grep -a '^Q38_L4VFY_ORIGIN' $L/p2.log | tail -1)
  HSRC=$(cd "$H" && realpath src)
  ENVSF=$(echo "$OL" | grep -oE 'envs=[^ ]+' | cut -d= -f2-); TCF=$(echo "$OL" | grep -oE 'tc_mod=[^ ]+' | cut -d= -f2-)
  ENVSF=$(cd "$H" && realpath "$ENVSF" 2>/dev/null); TCF=$(cd "$H" && realpath "$TCF" 2>/dev/null)
  case "$ENVSF" in "$HSRC"/*) case "$TCF" in "$HSRC"/*) echo "P2 origin envs + tc_mod INSIDE head/src" ;; *) echo "P2 origin tc_mod OUTSIDE"; FAIL="$FAIL P2-origin" ;; esac ;;
    *) echo "P2 origin envs OUTSIDE ($ENVSF)"; FAIL="$FAIL P2-origin" ;; esac
  [ $rc = 0 ] && exact $L/p2.log 7 || { FAIL="$FAIL P2"; tail -40 $L/p2.log; }
  (cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py) > $L/p2w.log 2>&1; rc=$?
  echo "P2w rc=$rc summary: $(summ $L/p2w.log)"
  [ $rc = 0 ] && [ "$(count $L/p2w.log passed)" -gt 7 ] && [ "$(count $L/p2w.log failed)" = 0 ] || { FAIL="$FAIL P2w"; tail -40 $L/p2w.log; }
  (cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src python3 - <<'EOP'
import os
import vllm_torchtpu.compilation.tpu_compiler as tc
from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab
here = os.path.realpath("src") + os.sep
for m in (tc, envs, l6_ab):
    assert os.path.realpath(m.__file__).startswith(here), m.__file__
ign = tc._TPU_COMPILE_ENV_IGNORED
rows = [("VLLM_TPU_VPS_VFY_ONE", "hashed"), ("VLLM_TPU_VPS_VFY_ONE_CHECK", "ignored"), ("VLLM_TPU_VPS_VFY_ONE_AB", "ignored")]
ok = True
for n, want in rows:
    got = "ignored" if n in ign else "hashed"
    print("P2 ROW %s registered=%s default=%r class=%s (want %s)" % (n, n in envs.environment_variables,
          envs.environment_variables[n](), got, want))
    ok &= got == want and n in envs.environment_variables
print("P2 ROW l6_ab lever vfyone -> %s" % l6_ab.LEVERS.get("vfyone"))
ok &= l6_ab.LEVERS.get("vfyone") == "VLLM_TPU_VPS_VFY_ONE_AB"
print("P2 ROWS", "OK" if ok else "BAD")
raise SystemExit(0 if ok else 1)
EOP
  ) > $L/p2r.log 2>&1; rc=$?
  tail -6 $L/p2r.log
  [ $rc = 0 ] && grep -aq '^P2 ROWS OK' $L/p2r.log || FAIL="$FAIL P2-rows"
fi

if [ "$MODE" = regress ]; then
  FILES=$(grep -vE '^\s*(#|$)' $E/RP_FILES | sed -n "${LO},${HI}p" | tr '\n' ' ')
  KEY=$(grep -vE '^\s*(#|$)' $E/RP_KEY | tr '\n' ' ')
  echo "== P2b collect-only census of the regression union (head / base)"
  RUN=""; NR=0; NX=0
  for f in $FILES; do
    nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    echo "P2b $f head ${nh:-0} base ${nb:-0}"
    case " $KEY " in *" $f "*) iskey=1 ;; *) iskey=0 ;; esac
    if [ $iskey = 1 ] && { [ "${nh:-0}" = 0 ] || [ "${nb:-0}" = 0 ]; }; then echo "P2b KEY file does not collect on both: $f"; FAIL="$FAIL P2b"
    elif [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then echo "P2b EXCLUDED (0 on both) $f"; NX=$((NX + 1))
    else RUN="$RUN $f"; NR=$((NR + 1)); fi
  done
  echo "P2b union $(echo $FILES | wc -w | tr -d ' ') files: run $NR, excluded $NX"
  echo "== P3 per-file regression HEAD vs BASE (f24edcc94) over the union"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile vfy "$H" "$B" 1500 $RUN > $L/p3.log 2>&1; rc=$?
  cat $L/p3.log
  if [ $rc != 0 ]; then
    FAIL="$FAIL P3"
    i=0
    for f in $RUN; do
      i=$((i + 1))
      grep -aqF "vfy RP $f: " $L/p3.log && grep -aF "vfy RP $f: " $L/p3.log | grep -aq -- '-> FAIL' || continue
      echo "P3 tip log of $f (/tmp/vfy_rp_${i}_tip.log): FAILED / E lines"
      grep -aE '^(FAILED|ERROR) |^E  ' /tmp/vfy_rp_${i}_tip.log | head -60
    done
  fi
fi

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"

[ -z "$FAIL" ] && { echo "Q38_L4VFY_PROOF PASS mode=$TAG"; exit 0; }
echo "Q38_L4VFY_PROOF FAIL mode=$TAG:$FAIL"; exit 1

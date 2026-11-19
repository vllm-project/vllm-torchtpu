#!/bin/bash
# q38 L4 AGREE proof on q38-l4-agree (CDK CPU pod only). AGREE-BUILD (helper of L4-E), 2026-11-14.
# Tree = q38-l4-agree = q38-l4-vfy-v5 2e166829a + runner/l4_agree.py (one engine-step clock, one capability gather) with
# DRAFT-ASM and VFY-ONE ported onto it. Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l4agree/run_agree.sh smoke | mut <LO> <HI> | regress A|B | bench
# (q38-l4-agree commit 2 adds KEYLITE, VLLM_TPU_VPS_VFY_ONE_KEYLITE: P0's diff set gains envs.py and tpu_compiler.py;
#  smoke gains P2-keylite and test_vfy_one under KEYLITE=1; mutants 31..42 are KEYLITE's; bench is its microbench.)
# PASS rules, fixed before the run:
#   P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC (2e166829a); diff -rq base/src head/src == exactly
#       compilation/tpu_compiler.py, envs.py, draft_asm.py, vfy_one.py, tpu_runner.py differ + Only in head:
#       runner/l4_agree.py.
#   P0o (smoke) vllm_torchtpu, .runner.l4_agree, .runner.vfy_one, .spec_decode.draft_asm, .spec_decode.eagle3,
#       .runner.tpu_runner resolve INSIDE head/src; with PYTHONPATH an empty dir, vllm_torchtpu resolves OUTSIDE (mutant killed).
#   P1  (smoke) the composed tree's own suites at their pinned counts (q38_vfy/run_compose.sh, v4 of record j-3c511bba):
#       test_vfy_dha_compose 17; test_vfy_one 78; test_vfy_agree 15; D83 test_vps_ro_fold 86; DRAFT-ASM units 56,
#       propose 51, multirank 8, flagoff 1 (its 3 AST tests vs 3d2f53ec9 deselected, as on v5); test_vfy_v5_flagoff 3.
#   P2  (smoke) the new tests: test_l4agree_units 17, test_l4agree_multirank 17, test_l4agree_flagoff 6
#       (incl. the import-origin test with Q38_L4AGREE_EXPECT_ROOT = head/src), test_vfy_keylite 23, and
#       test_vfy_one again with VLLM_TPU_VPS_VFY_ONE_KEYLITE=1: 78.
#   PB  (bench) tests/q38_l4agree/bench_vfy_keylite.py: 'Q38_KEYLITE_BENCH_DONE ok=True' (numbers are reported).
#   P3  (smoke) compile-key registration, whole file: 0 failed, > 7 passed.
#   P4  (smoke) composition mutants q38_l4vfy/run_compose_mut.py: 'Q38_L4VFY_COMPOSE_MUT PASS killed=9/9'.
#   PM  (mut) tests/q38_l4agree/q38_l4agree_mutrun.py LO HI: 'Q38_L4AGREE_MUT PASS killed=N/N'.
#   PR  (regress) per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596): A = DRAFT-ASM's RP_FILES (conftest
#       on), B = VFY-ONE's RP_FILES not in A (--noconftest), as each lever ran them; a file collecting 0 on both sides is
#       EXCLUDED and named, a KEY file may not be.
#   P6  head src unchanged at the end. Last line: Q38_L4AGREE_PROOF PASS|FAIL mode=<..> [failed legs].
set -u
MODE=${1:-smoke}; LO=${2:-0}; HI=${3:-0}
W=$(pwd); H=$W/head; B=$W/base; E=$H/tests/q38_l4agree
export JAX_PLATFORMS=cpu
unset VLLM_TPU_VPS_VFY_ONE VLLM_TPU_VPS_VFY_ONE_CHECK VLLM_TPU_VPS_VFY_ONE_AB VLLM_TPU_VPS_FUSED_READ_OFFSETS \
      VLLM_TPU_DRAFT_ASM_FOLD VLLM_TPU_DRAFT_ASM_CHECK_STEPS VLLM_TPU_DRAFT_ASM_AB VLLM_TPU_L6_AB_STEPS VLLM_TPU_SERVED_AB \
      VLLM_TPU_L6_AB_PERIOD_MS VLLM_TPU_L6_AB_CLASSES VLLM_TPU_L6_AB_CONFIRM VLLM_TPU_L6_AB_DIR
PTN="python3 -m pytest -q -p no:cacheprovider --noconftest -rfEs"   # logging kept: caplog tests
PTC="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
FAIL=""; L=/tmp/l4agree_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
not_in() { local x=$1 y; shift; for y in "$@"; do [ "$x" = "$y" ] && return 1; done; return 0; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }
HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $LO $HI $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1) nproc $(nproc)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_SRC)); base src $BS (expect $(cat $E/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $E/EXPECT_SRC)" ] && [ "$BS" = "$(cat $E/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'EOL'
Files base/src/vllm_torchtpu/compilation/tpu_compiler.py and head/src/vllm_torchtpu/compilation/tpu_compiler.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
Files base/src/vllm_torchtpu/runner/vfy_one.py and head/src/vllm_torchtpu/runner/vfy_one.py differ
Files base/src/vllm_torchtpu/spec_decode/draft_asm.py and head/src/vllm_torchtpu/spec_decode/draft_asm.py differ
Only in head/src/vllm_torchtpu/runner: l4_agree.py
EOL
)
[ "$DIFF" = "$WANT" ] || { echo "P0 src diff is not the stated set"; FAIL="$FAIL P0-diff"; }

run() {  # run <tag> <want> <pytest cmd...> (from head/)
  local tag=$1 want=$2; shift 2
  (cd "$H" && env JAX_PLATFORMS=cpu "$@") > $L/$tag.log 2>&1; local rc=$?
  echo "$tag rc=$rc want $want :: $(summ $L/$tag.log)"
  grep -aE '^Q38_[A-Z0-9_]+_COUNTS|^Q38_L4AGREE_(FLAGOFF|ORIGIN)|^Q38_L4VFY_V5_FLAGOFF' $L/$tag.log | head -8
  if [ $rc = 0 ] && exact $L/$tag.log $want; then :; else FAIL="$FAIL $tag"; grep -aE '^(FAILED|ERROR) |^E  ' $L/$tag.log | head -40; fi
}

if [ "$MODE" = smoke ]; then
  echo "== P0o import origin"
  NI=0
  for mod in vllm_torchtpu vllm_torchtpu.runner.l4_agree vllm_torchtpu.runner.vfy_one vllm_torchtpu.spec_decode.draft_asm vllm_torchtpu.spec_decode.eagle3 vllm_torchtpu.runner.tpu_runner; do
    got=$(cd "$H" && PYTHONPATH=src python3 -c "import importlib, os; print(os.path.realpath(importlib.import_module('$mod').__file__))" 2>/dev/null | tail -1)
    case "$got" in "$(cd $H && realpath src)"/*) echo "ORIGIN $mod INSIDE"; NI=$((NI + 1)) ;; *) echo "ORIGIN $mod OUTSIDE ${got:-none}"; FAIL="$FAIL P0o:$mod" ;; esac
  done
  EMPTY=$(mktemp -d); got=$(cd "$H" && PYTHONPATH=$EMPTY python3 -c "import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))" 2>/dev/null | tail -1); rm -rf $EMPTY
  case "$got" in "$(cd $H && realpath src)"/*) echo "P0o MUTANT SURVIVED"; FAIL="$FAIL P0o-mutant" ;; *) echo "P0o MUTANT KILLED (${got:-no vllm_torchtpu outside the tree})" ;; esac
  echo "P0o inside $NI/6"
  echo "== P1 the composed tree's suites"
  run P1-compose 17 env PYTHONPATH=src $PTC -s tests/l4vfy/test_vfy_dha_compose.py
  run P1-vfyone 78 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_one.py
  run P1-vfyagree 15 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_agree.py
  run P1-d83 86 env PYTHONPATH=src $PTN tests/l4tail/test_vps_ro_fold.py
  for fw in units:56 propose:51 multirank:8; do
    f=${fw%%:*}; n=${fw#*:}
    run P1-da-$f $n env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_$f.py
  done
  run P1-da-flagoff 1 env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_flagoff.py \
    --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_eagle3_flag_off_is_3d2f53ec9 \
    --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_tpu_runner_flag_off_is_3d2f53ec9 \
    --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_flag_off_check_bites
  run P1-v5flagoff 3 env PYTHONPATH=src $PTN -s tests/l4vfy/test_vfy_v5_flagoff.py
  echo "== P2 the new tests"
  run P2-units 17 env PYTHONPATH=src $PTC -s tests/q38_l4agree/test_l4agree_units.py
  run P2-multirank 17 env PYTHONPATH=src $PTC -s tests/q38_l4agree/test_l4agree_multirank.py
  run P2-flagoff 6 env PYTHONPATH=src Q38_L4AGREE_EXPECT_ROOT=$H/src $PTC -s tests/q38_l4agree/test_l4agree_flagoff.py
  run P2-keylite 23 env PYTHONPATH=src $PTC -s tests/q38_l4agree/test_vfy_keylite.py
  run P2-vfyone-keylite 78 env VLLM_TPU_VPS_VFY_ONE_KEYLITE=1 PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_one.py
  echo "== P3 compile-key registration"
  (cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src $PTN tests/compilation/test_tpu_compile_key_registration.py) > $L/p3.log 2>&1; rc=$?
  echo "P3 rc=$rc :: $(summ $L/p3.log)"
  [ $rc = 0 ] && [ "$(count $L/p3.log failed)" = 0 ] && [ "$(count $L/p3.log passed)" -gt 7 ] || { FAIL="$FAIL P3"; tail -30 $L/p3.log; }
  echo "== P4 composition mutants"
  (cd "$H" && python3 -u q38_l4vfy/run_compose_mut.py) > $L/p4.log 2>&1; rc=$?
  grep -aE '^(CLEAN|MUTANT|Q38_L4VFY_COMPOSE_MUT)' $L/p4.log
  [ $rc = 0 ] && grep -aq 'Q38_L4VFY_COMPOSE_MUT PASS killed=9/9' $L/p4.log || { FAIL="$FAIL P4"; tail -60 $L/p4.log; }
fi

if [ "$MODE" = bench ]; then
  echo "== PB KEYLITE microbench (alone in this pod)"
  (cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1 python3 -u tests/q38_l4agree/bench_vfy_keylite.py) > $L/pb.log 2>&1; rc=$?
  grep -aE '^Q38_KEYLITE_BENCH' $L/pb.log
  [ $rc = 0 ] && grep -aq 'Q38_KEYLITE_BENCH_DONE ok=True' $L/pb.log || { FAIL="$FAIL PB"; tail -40 $L/pb.log; }
fi

if [ "$MODE" = mut ]; then
  echo "== PM mutants $LO..$HI"
  (cd "$H" && python3 -u tests/q38_l4agree/q38_l4agree_mutrun.py "$LO" "$HI") > $L/pm.log 2>&1; rc=$?
  grep -aE '^(Q38_L4AGREE_MUT|CLEAN|MUTANT)' $L/pm.log
  N=$((HI - LO + 1))
  [ $rc = 0 ] && grep -aq "Q38_L4AGREE_MUT PASS killed=$N/$N" $L/pm.log || { FAIL="$FAIL PM"; tail -80 $L/pm.log; }
fi

if [ "$MODE" = regress ]; then
  CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
  DA_ALL=$(grep -aE '^(KEY|RUN) ' $H/tests/q38_draftasm/RP_FILES | awk '{print $2}')
  DA_KEY=$(grep -aE '^KEY ' $H/tests/q38_draftasm/RP_FILES | awk '{print $2}')
  VF_ALL=$(grep -vE '^\s*(#|$)' $H/tests/q38_vfy/RP_FILES | awk '!seen[$0]++')
  VF_KEY=$(grep -vE '^\s*(#|$)' $H/tests/q38_vfy/RP_KEY)
  if [ "$LO" = A ]; then FILES=$DA_ALL; KEY=$DA_KEY; PT="$PTC"; NOCONF=""
  else
    FILES=""
    for f in $VF_ALL; do not_in "$f" $DA_ALL && FILES="$FILES $f"; done
    KEY=$VF_KEY; PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"; NOCONF="--noconftest"
  fi
  echo "== PR-$LO collect-only census (head / base)"
  RUN=""; NR=0; NX=0
  for f in $FILES; do
    nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider $NOCONF -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider $NOCONF -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    echo "PR census $f head ${nh:-0} base ${nb:-0}"
    case " $(echo $KEY) " in *" $f "*) iskey=1 ;; *) iskey=0 ;; esac
    if [ $iskey = 1 ] && { [ "${nh:-0}" = 0 ] || [ "${nb:-0}" = 0 ]; }; then echo "PR KEY file does not collect on both: $f"; FAIL="$FAIL PR-key"
    elif [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then echo "PR EXCLUDED (0 on both) $f"; NX=$((NX + 1))
    else RUN="$RUN $f"; NR=$((NR + 1)); fi
  done
  echo "PR-$LO files $(echo $FILES | wc -w | tr -d ' '): run $NR, excluded $NX"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile agree$LO "$H" "$B" 1500 $RUN > $L/pr.log 2>&1; rc=$?
  cat $L/pr.log
  [ $rc = 0 ] || FAIL="$FAIL PR"
fi

HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
[ -z "$FAIL" ] && { echo "Q38_L4AGREE_PROOF PASS mode=$MODE $LO $HI"; exit 0; }
echo "Q38_L4AGREE_PROOF FAIL mode=$MODE $LO $HI:$FAIL"; exit 1

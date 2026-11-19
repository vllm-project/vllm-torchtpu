#!/bin/bash
# q38 L4 VFY-ONE x DRAFT-ASM composed proof on q38-l4-vfy-v5 (CDK CPU pod only). VFY-BUILD (helper of L4-E), 2026-11-14.
# Tree = q38-l4-vfy-v5: q38-l4-vfy (v3c b551e9b84 + D83S + VFY-ONE) merged with q38-l4-draftasm-vfyhook (q38-l4-stack-d1
# + DRAFT-ASM v4 462da96ed + the VFY-ONE seed hooks). Usage (from the unpacked bundle dir holding head/):
#   bash head/tests/q38_vfy/run_compose.sh
# PASS rules, fixed before the run:
#   P0  head src == EXPECT_SRC_V5.
#   P0o vllm_torchtpu, .runner.vfy_one, .spec_decode.draft_asm, .spec_decode.eagle3, .runner.tpu_runner resolve INSIDE
#       head/src (PYTHONPATH=src); an empty PYTHONPATH reads FALLBACK (mutant killed).
#   P1  tests/l4vfy/test_vfy_dha_compose.py: exactly 17 passed (C1-C8 + counts), 0 failed / skipped / errors.
#   P1b VFY-ONE's own files on the composed tree: test_vfy_one.py 78, test_vfy_agree.py 15, D83's test_vps_ro_fold.py 86.
#   P1c DRAFT-ASM's files on the composed tree: units 56, propose 51, multirank 8; flagoff 1 (its three AST tests,
#       the two identities and their bite check, compare against 3d2f53ec9 and do not apply to a tree that also carries
#       v3c + D83S + VFY-ONE: deselected, replaced by P1d).
#   P1d tests/l4vfy/test_vfy_v5_flagoff.py: exactly 3 passed (both flags off, eagle3.py and tpu_runner.py resolve to the
#       composed base q38-l4-vfy-v5base 8344dce32 AST for AST; the check bites on an unguarded hook of each lever).
#   P2  compile-key registration, whole file: 0 failed (VFY-ONE, D83 and DRAFT-ASM rows together).
#   P4  q38_l4vfy/run_compose_mut.py: 'Q38_L4VFY_COMPOSE_MUT PASS killed=9/9'.
#   P6  head src unchanged. Last line: Q38_L4VFY_COMPOSE PASS|FAIL [failed steps].
set -u
W=$(pwd); H=$W/head; E=$H/tests/q38_vfy
export JAX_PLATFORMS=cpu
unset VLLM_TPU_VPS_VFY_ONE VLLM_TPU_VPS_VFY_ONE_CHECK VLLM_TPU_VPS_VFY_ONE_AB VLLM_TPU_VPS_FUSED_READ_OFFSETS \
      VLLM_TPU_DRAFT_ASM_FOLD VLLM_TPU_DRAFT_ASM_CHECK_STEPS VLLM_TPU_DRAFT_ASM_AB VLLM_TPU_L6_AB_STEPS VLLM_TPU_SERVED_AB
PTN="python3 -m pytest -q -p no:cacheprovider --noconftest -rfEs"  # logging kept: caplog tests
PTC="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
FAIL=""; L=/tmp/l4vfyc_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }
HS=$(srch "$H")
echo "P0 $(date -u +%FT%TZ) $(python3 -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__)' 2>&1 | tail -1) head src $HS (expect $(cat $E/EXPECT_SRC_V5))"
[ "$HS" = "$(cat $E/EXPECT_SRC_V5)" ] || FAIL="$FAIL P0"
echo "== P0o import origin"
NI=0
for mod in vllm_torchtpu vllm_torchtpu.runner.vfy_one vllm_torchtpu.spec_decode.draft_asm vllm_torchtpu.spec_decode.eagle3 vllm_torchtpu.runner.tpu_runner; do
  got=$(cd "$H" && PYTHONPATH=src python3 -c "import importlib, os; print(os.path.realpath(importlib.import_module('$mod').__file__))" 2>/dev/null | tail -1)
  case "$got" in "$(cd $H && realpath src)"/*) echo "ORIGIN $mod INSIDE"; NI=$((NI + 1)) ;; *) echo "ORIGIN $mod OUTSIDE ${got:-none}"; FAIL="$FAIL P0o:$mod" ;; esac
done
EMPTY=$(mktemp -d); got=$(cd "$H" && PYTHONPATH=$EMPTY python3 -c "import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))" 2>/dev/null | tail -1); rm -rf $EMPTY
case "$got" in "$(cd $H && realpath src)"/*) echo "P0o MUTANT SURVIVED"; FAIL="$FAIL P0o-mutant" ;; *) echo "P0o MUTANT KILLED ($got)" ;; esac
echo "P0o inside $NI/5"
run() {  # run <tag> <want> <pytest cmd...>
  local tag=$1 want=$2; shift 2
  (cd "$H" && env JAX_PLATFORMS=cpu "$@") > $L/$tag.log 2>&1; local rc=$?
  echo "$tag rc=$rc want $want :: $(summ $L/$tag.log)"
  grep -aE '^Q38_L4VFY_COMPOSE_COUNTS|^Q38_L4VFY_COUNTS|^Q38_L4VFY_AGREE_COUNTS' $L/$tag.log | head -3
  if [ $rc = 0 ] && exact $L/$tag.log $want; then :; else FAIL="$FAIL $tag"; grep -aE '^(FAILED|ERROR) |^E  ' $L/$tag.log | head -40; fi
}
echo "== P1 composition"
run P1 17 env PYTHONPATH=src $PTC -s tests/l4vfy/test_vfy_dha_compose.py
echo "== P1b VFY-ONE files"
run P1b-one 78 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_one.py
run P1b-agree 15 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_agree.py
run P1b-d83 86 env PYTHONPATH=src $PTN tests/l4tail/test_vps_ro_fold.py
echo "== P1c DRAFT-ASM files"
for fw in units:56 propose:51 multirank:8; do
  f=${fw%%:*}; n=${fw#*:}
  run P1c-$f $n env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_$f.py
done
run P1c-flagoff 1 env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_flagoff.py \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_eagle3_flag_off_is_3d2f53ec9 \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_tpu_runner_flag_off_is_3d2f53ec9 \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_flag_off_check_bites
echo "== P1d composed flag-off AST identity"
run P1d 3 env PYTHONPATH=src $PTN -s tests/l4vfy/test_vfy_v5_flagoff.py
grep -a 'Q38_L4VFY_V5_FLAGOFF' $L/P1d.log
echo "== P2 compile-key registration"
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src $PTN tests/compilation/test_tpu_compile_key_registration.py) > $L/p2.log 2>&1; rc=$?
echo "P2 rc=$rc :: $(summ $L/p2.log)"
[ $rc = 0 ] && [ "$(count $L/p2.log failed)" = 0 ] && [ "$(count $L/p2.log passed)" -gt 7 ] || { FAIL="$FAIL P2"; tail -30 $L/p2.log; }
echo "== P4 composition mutants"
(cd "$H" && python3 -u q38_l4vfy/run_compose_mut.py) > $L/p4.log 2>&1; rc=$?
grep -aE '^(CLEAN|MUTANT|Q38_L4VFY_COMPOSE_MUT)' $L/p4.log
[ $rc = 0 ] && grep -aq 'Q38_L4VFY_COMPOSE_MUT PASS killed=9/9' $L/p4.log || { FAIL="$FAIL P4"; tail -60 $L/p4.log; }
HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
[ -z "$FAIL" ] && { echo "Q38_L4VFY_COMPOSE PASS"; exit 0; }
echo "Q38_L4VFY_COMPOSE FAIL:$FAIL"; exit 1

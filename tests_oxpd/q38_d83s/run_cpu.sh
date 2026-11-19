#!/bin/bash
# q38 L4 L1-D83-SERVED CPU proof (CDK CPU pod only; never the Mac or the VM). D83S-BUILD (helper of L4-D), 2026-11-13.
# Head = q38-l4-d83s: D83 (VLLM_TPU_VPS_FUSED_READ_OFFSETS, default off) rebased onto q38-l4-stack 11b0c44ef (f0ba1ccad)
#        + the warm-draws rebase fix b5b49a8d7 + the flag-first gate fix (src EXPECT_SRC) + these tests-only commits.
# Base = q38-l4-stack 11b0c44ef (src 6dd149f456): 965b3c6e7 + AOTSKIP + HOSTSUBS + D98R + DRAFT-2 + S1 (all default off).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_d83s/run_cpu.sh smoke|full
# Both modes (every PASS condition stated here, before the run):
#   P0  versions; src hashes head == EXPECT_SRC and base == EXPECT_BASE_SRC; the exact 3-entry src diff set (envs.py,
#       compilation/tpu_compiler.py, runner/tpu_runner.py).
#   P0o import origin: vllm_torchtpu, .envs, .compilation.tpu_compiler, .runner.tpu_runner resolve INSIDE head/src
#       (base: vllm_torchtpu, .envs, .runner.tpu_runner INSIDE base/src) under cwd <tree> + PYTHONPATH=src; mutant: an
#       empty PYTHONPATH must read FALLBACK on head and on base (else the check does not bite). run_d83.py repeats the
#       check under its own PYTHONPATH and adds the RESOLUTION mutant (raise appended to a staged copy of tpu_runner.py).
#   P1  q38_l4tail/run_d83.py on head: smoke = mode clean (IMPORT + CLEAN: exactly 86 passed, 0 failed / skipped /
#       errors, origin INSIDE, counts exact), full = mode mutants (the same + 34 mutants, every one KILLED: rc 1, >= 1
#       failed, 0 errors, origin INSIDE, not by a clean-failing test). Exit 0 and the summary line VERDICT=PASS
#       (+ mutants_killed=34/34 invalid=0 in full).
#   P2  compile-key registration on head: the three D83 flags' rows exactly 6 passed (registered + parses x3, hashed x1,
#       host-only x2); the whole file 0 failed; the origin plugin's envs / tc_mod INSIDE head/src; and a probe of
#       head's _TPU_COMPILE_ENV_IGNORED: FUSED_READ_OFFSETS NOT in it (hashed), _CHECK and _CHECK_FAULT in it.
# full only:
#   P2b collect-only census of the regression union on head and base (a KEY file must collect on both; a file that
#       collects 0 on both is excluded and named).
#   P3  per-file regression HEAD vs BASE (11b0c44ef) over the union with L3's regress_perfile v2 (4fb6f596): a missing
#       summary, 0 tests, a collection error or a new failing id on head = FAIL.
# Both: P6 head src unchanged by the run. Last line: Q38_L4D83S_PROOF PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
case "$MODE" in smoke|full) ;; *) echo "Q38_L4D83S_PROOF FAIL mode=$MODE: unknown mode"; exit 2 ;; esac
W=$(pwd); H=$W/head; B=$W/base; E=$H/tests/q38_d83s
export JAX_PLATFORMS=cpu
unset VLLM_TPU_VPS_FUSED_READ_OFFSETS VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT \
      VLLM_TPU_VPS_WARM_DRAWS VLLM_TPU_AOT_SKIP_GUARD_CHECK VLLM_TPU_HOST_PHASES_SUBS VLLM_TPU_FAST_COMPILED_CALL \
      VLLM_TPU_FAST_CALL_GUARD_MEMO VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE VLLM_TPU_DRAFT_EMBED_H8 VLLM_TPU_VPS_TOPK_KEYS
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
L=/tmp/l4d83s_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() {  # exact <log> <passed> -> rc 0 iff exactly <passed> passed, 0 failed, 0 skipped, 0 errors
  [ "$(count "$1" passed)" = "$2" ] && [ "$(count "$1" failed)" = 0 ] && [ "$(count "$1" skipped)" = 0 ] \
    && [ "$(count "$1" error)" = 0 ] && [ "$(count "$1" errors)" = 0 ]
}

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_SRC)) commit $(cat $E/EXPECT_COMMIT); base src $BS (expect $(cat $E/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $E/EXPECT_SRC)" ] && [ "$BS" = "$(cat $E/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'EOL'
Files base/src/vllm_torchtpu/compilation/tpu_compiler.py and head/src/vllm_torchtpu/compilation/tpu_compiler.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/runner/tpu_runner.py and head/src/vllm_torchtpu/runner/tpu_runner.py differ
EOL
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (3 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }
diff -rq "$B/tests" "$H/tests" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort | sed 's/^/P0 tests diff (info) /'

echo "== P0o import origin (cwd <tree>, PYTHONPATH=src; mutant: empty PYTHONPATH must read FALLBACK)"
origin() {  # origin <label> <tree> <module> [<pythonpath>] -> prints ORIGIN line; rc 0 iff INSIDE <tree>/src
  local lab=$1 t=$2 mod=$3 pp=${4:-src} want got v
  want=$(cd "$t" && realpath src)
  got=$(cd "$t" && env JAX_PLATFORMS=cpu PYTHONPATH="$pp" python3 -c "import importlib, os; m = importlib.import_module('$mod'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) v=INSIDE ;; *) v=FALLBACK ;; esac
  echo "ORIGIN $lab $mod ${got:-none} $v"
  [ "$v" = INSIDE ]
}
NO=0; NM=0
for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.compilation.tpu_compiler vllm_torchtpu.runner.tpu_runner; do
  origin head "$H" $mod && NO=$((NO + 1)) || FAIL="$FAIL P0o-head:$mod"
done
for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.runner.tpu_runner; do
  origin base "$B" $mod && NO=$((NO + 1)) || FAIL="$FAIL P0o-base:$mod"
done
EMPTY=$(mktemp -d /tmp/l4d83s_empty.XXXXXX)
for t in head base; do
  if origin "$t:MUTANT" "$W/$t" vllm_torchtpu "$EMPTY"; then echo "P0o MUTANT SURVIVED $t"; FAIL="$FAIL P0o-mutant:$t"
  else echo "P0o MUTANT KILLED $t (an empty PYTHONPATH reads FALLBACK)"; NM=$((NM + 1)); fi
done
rm -rf "$EMPTY"
echo "P0o inside $NO/7 mutants killed $NM/2"

if [ "$MODE" = smoke ]; then RM=clean; else RM=mutants; fi
echo "== P1 q38_l4tail/run_d83.py $RM (head)"
(cd "$H" && python3 -u q38_l4tail/run_d83.py $RM) > $L/p1.log 2>&1; rc=$?
grep -aE '^(SRC-|IMPORT|CLEAN|  Q38_L4|MUTANT|  md5|Q38_L4D83S_SUMMARY)' $L/p1.log
echo "P1 rc=$rc"
SUM=$(grep -a '^Q38_L4D83S_SUMMARY' $L/p1.log | tail -1)
if [ $rc = 0 ] && echo "$SUM" | grep -q "mode=$RM src=$HS import=PASS clean=PASS" && echo "$SUM" | grep -q 'VERDICT=PASS'; then
  if [ "$RM" = mutants ] && ! echo "$SUM" | grep -q 'mutants_killed=34/34 invalid=0 dropped=0'; then FAIL="$FAIL P1-mutants"; fi
else FAIL="$FAIL P1"; tail -80 $L/p1.log; fi

echo "== P2 compile-key registration (the three D83 flags' rows; the whole file; the hashed / ignored probe)"
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src:q38_l4tail $PT -p q38_l4tail_origin \
   tests/compilation/test_tpu_compile_key_registration.py -k VPS_FUSED_READ_OFFSETS -vv) > $L/p2.log 2>&1; rc=$?
grep -aE 'PASSED|FAILED|ERROR|^Q38_L4TAIL_ORIGIN' $L/p2.log | sed 's/^/P2 /'
echo "P2 rc=$rc summary: $(summ $L/p2.log)"
OL=$(grep -a '^Q38_L4TAIL_ORIGIN' $L/p2.log | tail -1)
HSRC=$(cd "$H" && realpath src)
ENVSF=$(echo "$OL" | grep -oE 'envs=[^ ]+' | cut -d= -f2-); TCF=$(echo "$OL" | grep -oE 'tc_mod=[^ ]+' | cut -d= -f2-)
ENVSF=$(cd "$H" && realpath "$ENVSF" 2>/dev/null); TCF=$(cd "$H" && realpath "$TCF" 2>/dev/null)
case "$ENVSF" in "$HSRC"/*) case "$TCF" in "$HSRC"/*) echo "P2 origin envs + tc_mod INSIDE head/src" ;; *) echo "P2 origin tc_mod OUTSIDE"; FAIL="$FAIL P2-origin" ;; esac ;;
  *) echo "P2 origin envs OUTSIDE ($ENVSF)"; FAIL="$FAIL P2-origin" ;; esac
[ $rc = 0 ] && exact $L/p2.log 6 || { FAIL="$FAIL P2"; tail -30 $L/p2.log; }
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py) > $L/p2w.log 2>&1; rc=$?
echo "P2w rc=$rc summary: $(summ $L/p2w.log)"
[ $rc = 0 ] && [ "$(count $L/p2w.log passed)" -gt 6 ] && [ "$(count $L/p2w.log failed)" = 0 ] || { FAIL="$FAIL P2w"; tail -30 $L/p2w.log; }
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src python3 - <<'EOP'
import os
import vllm_torchtpu.compilation.tpu_compiler as tc
from vllm_torchtpu import envs
here = os.path.realpath("src") + os.sep
assert os.path.realpath(tc.__file__).startswith(here) and os.path.realpath(envs.__file__).startswith(here)
ign = tc._TPU_COMPILE_ENV_IGNORED
rows = [("VLLM_TPU_VPS_FUSED_READ_OFFSETS", "hashed"), ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", "ignored"),
        ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", "ignored")]
ok = True
for n, want in rows:
    got = "ignored" if n in ign else "hashed"
    print("P2 ROW %s registered=%s default=%r class=%s (want %s)" % (n, n in envs.environment_variables,
          envs.environment_variables[n](), got, want))
    ok &= got == want and n in envs.environment_variables
print("P2 ROWS", "OK" if ok else "BAD")
raise SystemExit(0 if ok else 1)
EOP
) > $L/p2r.log 2>&1; rc=$?
cat $L/p2r.log | tail -6
[ $rc = 0 ] && grep -aq '^P2 ROWS OK' $L/p2r.log || FAIL="$FAIL P2-rows"

if [ "$MODE" = full ]; then
  FILES="tests/compilation/test_tpu_compile_key_registration.py tests/q38mm_dstock/test_vps_wiring.py
tests/decfb2/test_vps_fb_wiring.py tests/runner/test_spec_verify_served_only.py tests/runner/test_draft_carries_port.py
tests/compilation/test_fast_compiled_call.py tests/runner/test_host_phases_subs.py tests/q38_samp1/test_vps_topk_keys.py
tests/platforms/test_patch_registry.py tests/test_envs.py tests/test_envs_builders.py tests/compilation/test_tpu_compiler.py
tests/compilation/test_compile_patches.py tests/compilation/test_bucket_filter.py tests/compilation/test_compile_all_ranges_patch.py
tests/compilation/test_additional_config_hash_patch.py tests/compilation/test_kernel_reload.py tests/runner/test_host_phases.py
tests/runner/test_host_phases_slow.py tests/runner/test_fast_metadata_build_port.py tests/runner/test_spec_tail_port.py
tests/runner/test_postfwd_runner_wiring.py tests/runner/test_pp_last_stage_output.py tests/runner/test_prompt_tail.py
tests/megakernel_runtime/test_qwen38_guards.py tests/megakernel_runtime/test_qwen38_real_traffic.py
tests/megakernel_runtime/test_wholecheck.py tests/megakernel_runtime/test_module_globals.py tests/runner/test_q38_eagerfix.py
tests/core/test_host_knobs.py tests/models/vllm/test_qwen4_exp_mtp.py tests/models/vllm/test_qwen4_exp.py
tests/models/vllm/test_qwen4_exp_model.py tests/layers/adapter/quantization/test_fp8_embedding.py
tests/q38mm_dstock/test_vp_sampler.py tests/decfb2/test_vps_fallback.py tests/decfb2/test_vps_harness_no_leak.py"
  KEY="tests/compilation/test_tpu_compile_key_registration.py tests/q38mm_dstock/test_vps_wiring.py tests/decfb2/test_vps_fb_wiring.py
tests/runner/test_spec_verify_served_only.py tests/runner/test_draft_carries_port.py tests/compilation/test_fast_compiled_call.py
tests/runner/test_host_phases_subs.py tests/q38_samp1/test_vps_topk_keys.py tests/runner/test_q38_eagerfix.py
tests/q38mm_dstock/test_vp_sampler.py tests/decfb2/test_vps_fallback.py"
  echo "== P2b collect-only census of the regression union (head / base)"
  RUN=""; NR=0; NX=0
  for f in $FILES; do
    nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
    echo "P2b $f head ${nh:-0} base ${nb:-0}"
    case " $(echo $KEY) " in *" $f "*) iskey=1 ;; *) iskey=0 ;; esac
    if [ $iskey = 1 ] && { [ "${nh:-0}" = 0 ] || [ "${nb:-0}" = 0 ]; }; then echo "P2b KEY file does not collect on both: $f"; FAIL="$FAIL P2b"
    elif [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then echo "P2b EXCLUDED (0 on both) $f"; NX=$((NX + 1))
    else RUN="$RUN $f"; NR=$((NR + 1)); fi
  done
  echo "P2b union $(echo $FILES | wc -w | tr -d ' ') files: run $NR, excluded $NX"
  echo "== P3 per-file regression HEAD vs BASE (11b0c44ef) over the union"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile d83s "$H" "$B" 1500 $RUN > $L/p3.log 2>&1; rc=$?
  cat $L/p3.log
  if [ $rc != 0 ]; then
    FAIL="$FAIL P3"
    i=0
    for f in $RUN; do  # the helper's per-file logs are /tmp/d83s_rp_<i>_tip.log: show why each failing file failed on head
      i=$((i + 1))
      grep -aqF "d83s RP $f: " $L/p3.log && grep -aF "d83s RP $f: " $L/p3.log | grep -aq -- '-> FAIL' || continue
      echo "P3 tip log of $f (/tmp/d83s_rp_${i}_tip.log): FAILED / E lines"
      grep -aE '^(FAILED|ERROR) |^E  ' /tmp/d83s_rp_${i}_tip.log | head -60
    done
  fi
fi

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"

[ -z "$FAIL" ] && { echo "Q38_L4D83S_PROOF PASS mode=$MODE"; exit 0; }
echo "Q38_L4D83S_PROOF FAIL mode=$MODE:$FAIL"; exit 1

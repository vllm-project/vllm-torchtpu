#!/bin/bash
# q38 L4 X-D14-DRAFT-2 (a) CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_draft2/run_cpu.sh smoke|full
# P0 versions, src hashes (head == EXPECT_SRC, base == EXPECT_BASE_SRC) and the exact set of src files that differ;
# P1 test_draft_embed_h8.py: exactly 4 passed, 0 failed, 0 skipped;
# P2 compile-key registration rows of the flag: exactly 2 passed.            (smoke stops here, after P2b)
# P2b collect-only census of the P3 files on head and base (printed);
# P3 per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596: no summary / 0 tests / collection error / a new
#    failing id on head = FAIL). A file that collects 0 tests on BOTH trees is excluded and named; the KEY files must run;
# P4 mutants (q38_draft2_mutrun.py: control + 9 mutants, each must FAIL its target);
# P5 head src unchanged by the run.
# Last line: Q38_DRAFT2_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base
export JAX_PLATFORMS=cpu
unset VLLM_TPU_DRAFT_EMBED_H8 VLLM_TPU_AOT_SKIP_GUARD_CHECK
PT="python3 -m pytest -q -p no:cacheprovider --noconftest -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $H/tests/q38_draft2/EXPECT_SRC)) commit $(cat $H/tests/q38_draft2/EXPECT_COMMIT); base src $BS (expect $(cat $H/tests/q38_draft2/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $H/tests/q38_draft2/EXPECT_SRC)" ] && [ "$BS" = "$(cat $H/tests/q38_draft2/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/__init__.py and head/src/vllm_torchtpu/__init__.py differ
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py and head/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py differ
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (3 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P1 test_draft_embed_h8.py"
(cd "$H" && env $CPUENV $PT tests/q38_draft2/test_draft_embed_h8.py) > /tmp/d2_p1.log 2>&1; rc=$?
grep -aE '^(FAILED|ERROR) |Error|assert' /tmp/d2_p1.log | head -40
echo "P1 rc=$rc summary: $(summ /tmp/d2_p1.log)"
[ $rc = 0 ] && [ "$(count /tmp/d2_p1.log passed)" = 4 ] && [ "$(count /tmp/d2_p1.log failed)" = 0 ] \
  && [ "$(count /tmp/d2_p1.log skipped)" = 0 ] || { FAIL="$FAIL P1"; tail -60 /tmp/d2_p1.log; }

echo "== P2 compile-key registration rows of VLLM_TPU_DRAFT_EMBED_H8"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k DRAFT_EMBED_H8) > /tmp/d2_p2.log 2>&1; rc=$?
tail -4 /tmp/d2_p2.log
echo "P2 rc=$rc summary: $(summ /tmp/d2_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/d2_p2.log passed)" = 2 ] && [ "$(count /tmp/d2_p2.log failed)" = 0 ] || FAIL="$FAIL P2"

FILES="tests/models/vllm/test_qwen4_exp_mtp.py tests/models/vllm/test_qwen4_exp.py tests/models/vllm/test_qwen4_exp_model.py
tests/layers/adapter/quantization/test_fp8_embedding.py tests/compilation/test_tpu_compile_key_registration.py
tests/test_envs_builders.py tests/platforms/test_patch_registry.py"
KEY="tests/models/vllm/test_qwen4_exp_mtp.py tests/compilation/test_tpu_compile_key_registration.py"
echo "== P2b collect-only census (head / base)"
RUN=""
for f in $FILES; do
  nh=$(cd "$H" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  nb=$(cd "$B" && env $CPUENV python3 -m pytest -q --collect-only -p no:cacheprovider --noconftest -p no:logging "$f" 2>&1 | grep -aoE '^[0-9]+ tests? collected|^([0-9]+) tests?' | grep -oE '^[0-9]+' | tail -1)
  echo "P2b $f head ${nh:-0} base ${nb:-0}"
  if [ "${nh:-0}" = 0 ] && [ "${nb:-0}" = 0 ]; then
    case " $KEY " in *" $f "*) echo "P2b KEY file collects nothing: $f"; FAIL="$FAIL P2b";; *) echo "P2b EXCLUDED (0 on both) $f";; esac
  else RUN="$RUN $f"; fi
done

if [ "$MODE" = smoke ]; then
  [ -z "$FAIL" ] && { echo "Q38_DRAFT2_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_DRAFT2_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 per-file regression HEAD vs BASE (965b3c6e7)"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile draft2 "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
(cd "$H" && python3 tests/q38_draft2/q38_draft2_mutrun.py .) > /tmp/d2_p4.log 2>&1; rc=$?
grep -aE '^CONTROL|^MUT|^Q38_DRAFT2_MUT' /tmp/d2_p4.log
echo "P4 rc=$rc"
[ $rc = 0 ] && grep -aq '^Q38_DRAFT2_MUT PASS killed 9/9 of 9' /tmp/d2_p4.log || { FAIL="$FAIL P4"; tail -30 /tmp/d2_p4.log; }

HS2=$(srch "$H")
echo "P5 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P5"

[ -z "$FAIL" ] && { echo "Q38_DRAFT2_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_DRAFT2_SUMMARY FAIL mode=full:$FAIL"; exit 1

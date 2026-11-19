#!/bin/bash
# q38 STACK-D v4 composed check DRAFT-1 x D73 (CDK CPU pod only; never the Mac or the VM). D83S-BUILD for L4-E / L1, 2026-11-13.
# Head = q38-l4-d1xd73: q38-L1-v3c b551e9b84 + q38-l4-stack-d1 3d2f53ec9 (DRAFT-1) + q38-l2-d73gate a4333fb7d (D73 gate), both merged
# without conflicts, + tests/kernels/qwen4_exp/dcp2/test_dcp2_draft1_x_d73.py and this runner (tests only).
# Usage (from the unpacked bundle dir holding head/):  bash head/tests/q38_d1xd73/run_cpu.sh
# PASS rules, fixed before the run:
#   P0  head src == EXPECT_SRC.
#   P0o import origin (cd head; PYTHONPATH=src, as P1 runs): vllm_torchtpu, .envs, .kernels.qwen4_exp.pooled_qsa,
#       .kernels.qwen4_exp.fused_kv_write resolve INSIDE realpath(head/src) (4/4); FALLBACK mutant: with PYTHONPATH = an empty dir
#       vllm_torchtpu must resolve OUTSIDE head/src; RESOLUTION mutant: a scratch copy of src with `raise` appended to
#       kernels/qwen4_exp/pooled_qsa.py must make the pooled_qsa import fail with that raise.
#   P1  test_dcp2_draft1_x_d73.py on head: exactly 5 passed (DCP 2 x partial_kernels {reference, production}, DCP 1, engagement
#       x {dcp, replicated}), 0 failed / skipped / errors; 3 'D1XD73_BITWISE dcp=<1|2> steps 8' lines (2 x dcp=2, 1 x dcp=1); the 8 D1XD73_ENGAGE lines with draft1=off d73=on '3 0 3', every other combination '0 3 0'; every D1XD73_ORIGIN path
#       INSIDE head/src.
#   P4  in a scratch copy of head (src + tests + pyproject.toml): CONTROL unmutated, exactly 5 passed; MX1 (qsa_cache_writes' early
#       return moved after D73's fused decision: the draft takes D73's choice and skips write_main_cache with the env on): the
#       file must FAIL with exactly 5 failed, 0 passed, 0 errors (both the bitwise and the engagement items bite); the scratch
#       copy's import origin is INSIDE scratch/src for both runs; the scratch dir is deleted.
#   P6  head src unchanged by the run.
# Last line: Q38_D1XD73_SUMMARY PASS|FAIL [failed steps].
set -u
W=$(pwd); H=$W/head; E=$H/tests/q38_d1xd73
export JAX_PLATFORMS=cpu
unset VLLM_TPU_QSA_FUSED_KV_WRITE VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE VLLM_TPU_QSA_DRAFT_SHARED_SELECTION
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs -s"
TF=tests/kernels/qwen4_exp/dcp2/test_dcp2_draft1_x_d73.py
PQ=src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py
FAIL=""
L=/tmp/d1xd73_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
norm() { sed -E 's/^[.FEsx]+(D1XD73_)/\1/' "$1"; }  # -s output: a test's first print follows the progress dots
origin() {  # origin <tree> <module> <pythonpath> -> realpath of the module file ("none" if the import fails)
  (cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH="$3" python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
}

HS=$(srch "$H")
echo "P0 $(date -u +%FT%TZ) $(python3 -c 'import sys, torch, jax, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "jax", jax.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $E/EXPECT_SRC)) commit $(cat $E/EXPECT_COMMIT)"
[ "$HS" = "$(cat $E/EXPECT_SRC)" ] || FAIL="$FAIL P0-src"

echo "== P0o import origin"
HSRC=$(cd "$H" && realpath src); NI=0
for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.kernels.qwen4_exp.pooled_qsa vllm_torchtpu.kernels.qwen4_exp.fused_kv_write; do
  got=$(origin "$H" $mod src)
  case "$got" in "$HSRC"/*) v=INSIDE; NI=$((NI + 1)) ;; *) v=OUTSIDE; FAIL="$FAIL P0o:$mod" ;; esac
  echo "ORIGIN head $mod ${got:-none} $v"
done
EMPTY=$(mktemp -d /tmp/d1xd73_empty.XXXXXX)
got=$(origin "$H" vllm_torchtpu "$EMPTY")
case "$got" in "$HSRC"/*) echo "P0o MUTANT FALLBACK SURVIVED ($got)"; FAIL="$FAIL P0o-fallback" ;;
  *) echo "P0o MUTANT FALLBACK KILLED (empty PYTHONPATH -> ${got:-none})" ;; esac
rm -rf "$EMPTY"
RS=$(mktemp -d /tmp/d1xd73_resolve.XXXXXX); cp -a "$H/src" "$RS/src"
printf "\nraise RuntimeError('q38 D1XD73 resolution mutant')\n" >> "$RS/$PQ"
(cd "$RS" && env JAX_PLATFORMS=cpu PYTHONPATH=src python3 -c "import vllm_torchtpu.kernels.qwen4_exp.pooled_qsa") > $L/resolve.log 2>&1; rc=$?
if [ $rc != 0 ] && grep -aq 'q38 D1XD73 resolution mutant' $L/resolve.log; then echo "P0o MUTANT RESOLUTION KILLED (rc=$rc)"
else echo "P0o MUTANT RESOLUTION SURVIVED (rc=$rc)"; FAIL="$FAIL P0o-resolution"; tail -5 $L/resolve.log; fi
rm -rf "$RS"
echo "P0o inside $NI/4"

echo "== P1 $TF (head)"
t0=$(date +%s)
(cd "$H" && env JAX_PLATFORMS=cpu PYTHONPATH=src $PT $TF) > $L/p1raw.log 2>&1; rc=$?
norm $L/p1raw.log > $L/p1.log
grep -aE '^(D1XD73_|FAILED|ERROR)' $L/p1.log
echo "P1 rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ $L/p1.log)"
ok=1
[ $rc = 0 ] && [ "$(count $L/p1.log passed)" = 5 ] && [ "$(count $L/p1.log failed)" = 0 ] && [ "$(count $L/p1.log skipped)" = 0 ] \
  && [ "$(count $L/p1.log error)" = 0 ] && [ "$(count $L/p1.log errors)" = 0 ] || ok=0
[ "$(grep -acE '^D1XD73_BITWISE dcp=2 steps 8 ' $L/p1.log)" = 2 ] && [ "$(grep -acE '^D1XD73_BITWISE dcp=1 steps 8 ' $L/p1.log)" = 1 ] || ok=0
for v in dcp replicated; do
  for d1 in off on; do
    for e in off on; do
      if [ $d1 = off ] && [ $e = on ]; then want="fused_layer_write=3 write_main_cache=0 ring_read_concurrent=3"
      else want="fused_layer_write=0 write_main_cache=3 ring_read_concurrent=0"; fi
      grep -aq "^D1XD73_ENGAGE variant=$v draft1=$d1 d73=$e $want\$" $L/p1.log || { ok=0; echo "P1 missing: ENGAGE $v draft1=$d1 d73=$e $want"; }
    done
  done
done
NO=$(grep -ac '^D1XD73_ORIGIN' $L/p1.log)
BAD=$(grep -a '^D1XD73_ORIGIN' $L/p1.log | tr ' ' '\n' | grep '=' | cut -d= -f2- | grep -vc "^$HSRC/")
echo "P1 origin lines $NO, outside head/src $BAD"
[ "$NO" = 3 ] && [ "$BAD" = 0 ] || ok=0
[ $ok = 1 ] && echo "P1 PASS" || { echo "P1 FAIL"; FAIL="$FAIL P1"; tail -60 $L/p1.log; }

echo "== P4 scratch copy: CONTROL and MX1 (the draft takes D73's fused decision)"
S=$(mktemp -d /tmp/d1xd73_mut.XXXXXX); cp -a "$H/src" "$H/tests" "$S/"; cp "$H/pyproject.toml" "$S/"
SSRC=$(cd "$S" && realpath src)
so=$(origin "$S" vllm_torchtpu.kernels.qwen4_exp.pooled_qsa src)
case "$so" in "$SSRC"/*) echo "P4 origin scratch pooled_qsa $so INSIDE" ;; *) echo "P4 origin scratch $so OUTSIDE"; FAIL="$FAIL P4-origin" ;; esac
(cd "$S" && env JAX_PLATFORMS=cpu PYTHONPATH=src $PT $TF) > $L/p4craw.log 2>&1; rc=$?
norm $L/p4craw.log > $L/p4c.log
echo "P4 CONTROL rc=$rc summary: $(summ $L/p4c.log)"
[ $rc = 0 ] && [ "$(count $L/p4c.log passed)" = 5 ] && [ "$(count $L/p4c.log failed)" = 0 ] \
  && [ "$(grep -a '^D1XD73_ORIGIN' $L/p4c.log | tr ' ' '\n' | grep '=' | cut -d= -f2- | grep -vc "^$SSRC/")" = 0 ] \
  || { FAIL="$FAIL P4-control"; tail -30 $L/p4c.log; }
python3 - "$S/$PQ" <<'EOP' > $L/p4m_apply.log 2>&1
import sys
p = sys.argv[1]
s = open(p).read()
early = ("    if not index_state:\n"
         "        pool = write_main_cache(pool, rows_packed, page, logical, main_valid, **slot_kw)\n"
         "        return pool, req, logical, valid\n")
fused_end = ("    if not fused:\n"
             "        pool = write_main_cache(pool, rows_packed, page, logical, main_valid, **slot_kw)\n")
assert s.count(early) == 1 and s.count(fused_end) == 1, (s.count(early), s.count(fused_end))
assert s.index(early) < s.index(fused_end)
s = s.replace(early, "").replace(fused_end, fused_end + "    if not index_state:\n        return pool, req, logical, valid\n")
open(p, "w").write(s)
print("MX1 applied")
EOP
rc=$?; cat $L/p4m_apply.log | tail -2
if [ $rc != 0 ] || cmp -s "$S/$PQ" "$H/$PQ"; then echo "P4 MX1 ABORT (not applied)"; FAIL="$FAIL P4-MX1-apply"
else
  so=$(origin "$S" vllm_torchtpu.kernels.qwen4_exp.pooled_qsa src)
  case "$so" in "$SSRC"/*) echo "P4 origin MX1 pooled_qsa $so INSIDE" ;; *) echo "P4 origin MX1 $so OUTSIDE"; FAIL="$FAIL P4-origin" ;; esac
  (cd "$S" && env JAX_PLATFORMS=cpu PYTHONPATH=src $PT $TF) > $L/p4m.log 2>&1; rc=$?
  grep -aE '^(FAILED|ERROR)|^E .*(differ|written|AssertionError|assert)' $L/p4m.log | head -16
  echo "P4 MX1 rc=$rc summary: $(summ $L/p4m.log)"
  if [ $rc = 1 ] && [ "$(count $L/p4m.log failed)" = 5 ] && [ "$(count $L/p4m.log passed)" = 0 ] \
     && [ "$(count $L/p4m.log error)" = 0 ] && [ "$(count $L/p4m.log errors)" = 0 ] && ! grep -aqE 'errors? during collection' $L/p4m.log; then
    echo "P4 MX1 KILLED (5/5 failed)"
  else echo "P4 MX1 SURVIVED or INVALID"; FAIL="$FAIL P4-MX1"; tail -40 $L/p4m.log; fi
fi
rm -rf "$S"

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
[ -z "$FAIL" ] && { echo "Q38_D1XD73_SUMMARY PASS src=$HS bitwise=3/3 engage=8/8 mx1=KILLED fallback=KILLED resolution=KILLED"; exit 0; }
echo "Q38_D1XD73_SUMMARY FAIL src=$HS:$FAIL"; exit 1

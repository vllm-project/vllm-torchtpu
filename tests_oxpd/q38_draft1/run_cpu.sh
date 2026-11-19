#!/bin/bash
# q38 L4 X-D14-DRAFT-1 CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_draft1/run_cpu.sh smoke|full
#   bash head/tests/q38_draft1/run_cpu.sh mut LO HI      (a P4 slice: P0, P0b, P4 over mutants LO..HI, P6)
# PASS rules, fixed before the run:
# Composed tree (STACK-D v3 entry): head = q38-l4-stack-d1 (q38-l4-stack 11b0c44ef + X-D14-DRAFT-1 + its engagement marker),
# base = q38-l4-stack 11b0c44ef.
# P0 head src == EXPECT_SRC, base src == EXPECT_BASE_SRC (6dd149f456, 11b0c44ef); the src diff set is exactly WANT (6 files).
# P0b import origin (the lead's rule): run as the harness runs (cd <tree>; PYTHONPATH=src), vllm_torchtpu and the 6 changed
#    modules resolve inside realpath(<tree>/src) on head and base (14/14 INSIDE); the same imports with PYTHONPATH = an
#    empty dir read FALLBACK (14/14 mutants killed).
# P1 new tests on head. Kernel file: exactly 5 passed, 0 failed, 0 skipped, 0 errors; the 4 DRAFT1_CENSUS lines
#    (variant dcp / replicated x flag off ring=6 / flag on ring=0) and 3 DRAFT1_BITWISE lines (steps 8).
#    Torch file: exactly 26 passed (14 routing + 5 engagement-marker + 7 producer-guard cases), 0 failed, 0 skipped, 0 errors.
# P2 compile-key rows of VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE: exactly 2 passed, 0 failed.
# P5 flag-off identity: q38_draft1_jaxpr_identity.py prints 14 JAXPR lines and no ERROR on head, base and base again;
#    base == base again (determinism) and head == base, line for line.
# P5b the identity check bites: a head copy with qsa_cache_writes' index_state default flipped to False prints 14 lines,
#    no ERROR, and all 14 differ from base.
# (smoke = P0 P1 P2 P5 P5b)
# P3 per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over the RP_FILES candidates. A candidate whose
#    base screen counts 0 tests (all skipped / none collected) is EXCLUDED and named; a KEY file may not be excluded.
# P4 mutants: q38_draft1_mutrun.py, 'MUTRUN_ORIGIN ... INSIDE', control + 29 mutants: 'Q38_DRAFT1_MUT PASS killed 29/29 of 29'.
# P6 head src unchanged by the run.
# mut LO HI: P0 and P0b as above, then P4 over mutants LO..HI (1-based, inclusive) with its own control and origin probe:
#    'Q38_DRAFT1_MUT PASS killed N/N of N range LO-HI of 29', N = HI - LO + 1; then P6.
# Last line: Q38_DRAFT1_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-full}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_draft1
export JAX_PLATFORMS=cpu
unset VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE VLLM_TPU_QSA_DRAFT_SHARED_SELECTION
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, jax, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "jax", jax.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort <<'L'
Files base/src/vllm_torchtpu/envs.py and head/src/vllm_torchtpu/envs.py differ
Files base/src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py and head/src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py differ
Files base/src/vllm_torchtpu/kernels/qwen4_exp/qsa.py and head/src/vllm_torchtpu/kernels/qwen4_exp/qsa.py differ
Files base/src/vllm_torchtpu/layers/adapter/custom_ops/qsa_op.py and head/src/vllm_torchtpu/layers/adapter/custom_ops/qsa_op.py differ
Files base/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py and head/src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py differ
Files base/src/vllm_torchtpu/models/vllm/qwen4_exp/qsa.py and head/src/vllm_torchtpu/models/vllm/qwen4_exp/qsa.py differ
L
)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK (6 entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import origin (cd <tree>; PYTHONPATH=src) and its empty-PYTHONPATH mutant"
EMPTY=$(mktemp -d /tmp/d1_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
nin=0; nkill=0
for t in head base; do
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.kernels.qwen4_exp.pooled_qsa vllm_torchtpu.kernels.qwen4_exp.qsa \
             vllm_torchtpu.layers.adapter.custom_ops.qsa_op vllm_torchtpu.models.vllm.qwen4_exp.qsa vllm_torchtpu.models.vllm.qwen4_exp.mtp; do
    r=$(origin "$W/$t" $mod src); echo "ORIGIN $t $mod ${r#* } ${r%% *}"; [ "${r%% *}" = INSIDE ] && nin=$((nin + 1))
    m=$(origin "$W/$t" $mod "$EMPTY"); echo "ORIGIN $t:MUTANT $mod ${m#* } ${m%% *}"; [ "${m%% *}" = FALLBACK ] && nkill=$((nkill + 1))
  done
done
echo "P0b inside $nin/14; empty-PYTHONPATH mutants reading FALLBACK $nkill/14"
[ $nin = 14 ] && [ $nkill = 14 ] || FAIL="$FAIL P0b"

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI (slice)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_draft1/q38_draft1_mutrun.py . $LO $HI) > /tmp/d1_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_DRAFT1_MUT' /tmp/d1_p4.log | cut -c1-600
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aq "^Q38_DRAFT1_MUT PASS killed $N/$N of $N range $LO-$HI of 29 " /tmp/d1_p4.log \
    && grep -aq '^MUTRUN_ORIGIN .* INSIDE$' /tmp/d1_p4.log || { FAIL="$FAIL P4"; tail -30 /tmp/d1_p4.log; }
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_DRAFT1_SUMMARY PASS mode=mut range $LO-$HI"; exit 0; }
  echo "Q38_DRAFT1_SUMMARY FAIL mode=mut range $LO-$HI:$FAIL"; exit 1
fi

echo "== P1 new tests (head)"
KF=tests/kernels/qwen4_exp/dcp2/test_dcp2_draft_no_index_state.py
TF=tests/models/vllm/test_qwen4_exp_draft_no_index_state.py
t0=$(date +%s)
(cd "$H" && env $CPUENV $PT -s $KF) > /tmp/d1_p1k.log 2>&1; rck=$?
echo "P1 kernel rc=$rck $(( $(date +%s) - t0 ))s summary: $(summ /tmp/d1_p1k.log)"
grep -aE 'DRAFT1_|^(FAILED|ERROR) ' /tmp/d1_p1k.log | cut -c1-900
ncen=$(grep -acE 'DRAFT1_CENSUS variant=(dcp|replicated) flag=(off ring=6|on ring=0) ' /tmp/d1_p1k.log)
nbit=$(grep -acE 'DRAFT1_BITWISE .*steps 8 ' /tmp/d1_p1k.log)
echo "P1 census lines $ncen (want 4) bitwise lines $nbit (want 3)"
[ $rck = 0 ] && exact /tmp/d1_p1k.log 5 && [ "$ncen" = 4 ] && [ "$nbit" = 3 ] || { FAIL="$FAIL P1k"; tail -80 /tmp/d1_p1k.log; }
t0=$(date +%s)
(cd "$H" && env $CPUENV $PT $TF) > /tmp/d1_p1t.log 2>&1; rct=$?
echo "P1 torch rc=$rct $(( $(date +%s) - t0 ))s summary: $(summ /tmp/d1_p1t.log)"
[ $rct = 0 ] && exact /tmp/d1_p1t.log 26 || { FAIL="$FAIL P1t"; tail -80 /tmp/d1_p1t.log; }

echo "== P2 compile-key registration rows of VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE"
(cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k DRAFT_NO_INDEX_STATE) > /tmp/d1_p2.log 2>&1; rc=$?
echo "P2 rc=$rc summary: $(summ /tmp/d1_p2.log)"
[ $rc = 0 ] && [ "$(count /tmp/d1_p2.log passed)" = 2 ] && [ "$(count /tmp/d1_p2.log failed)" = 0 ] || { FAIL="$FAIL P2"; tail -30 /tmp/d1_p2.log; }

echo "== P5 flag-off jaxpr identity (head vs base 11b0c44ef)"
jx() { (cd "$1" && env JAX_PLATFORMS=cpu PYTHONPATH=src python3 "$Q/q38_draft1_jaxpr_identity.py" "$1") > "$2" 2>&1; }
jx "$H" /tmp/d1_jh.log; jx "$B" /tmp/d1_jb1.log; jx "$B" /tmp/d1_jb2.log
for f in jh jb1 jb2; do echo "P5 $f: $(grep -ac '^JAXPR [^ ]* [0-9a-f]\{16\} eqns=' /tmp/d1_$f.log) lines, $(grep -ac '^JAXPR .* ERROR' /tmp/d1_$f.log) errors, $(grep -a '^JAXPR_ALL' /tmp/d1_$f.log)"; done
paste -d' ' <(grep -a '^JAXPR ' /tmp/d1_jh.log) <(grep -a '^JAXPR ' /tmp/d1_jb1.log | awk '{print "| base", $3}') | sed 's/^/P5 /'
p5ok=1
for f in jh jb1 jb2; do
  [ "$(grep -ac '^JAXPR [^ ]* [0-9a-f]\{16\} eqns=' /tmp/d1_$f.log)" = 14 ] && [ "$(grep -ac '^JAXPR .* ERROR' /tmp/d1_$f.log)" = 0 ] || p5ok=0
done
cmp -s <(grep -a '^JAXPR' /tmp/d1_jb1.log) <(grep -a '^JAXPR' /tmp/d1_jb2.log) || { echo "P5 base not deterministic"; p5ok=0; }
cmp -s <(grep -a '^JAXPR' /tmp/d1_jh.log) <(grep -a '^JAXPR' /tmp/d1_jb1.log) || { echo "P5 head != base"; p5ok=0; }
[ $p5ok = 1 ] && echo "P5 identity OK" || { FAIL="$FAIL P5"; tail -40 /tmp/d1_jh.log; tail -20 /tmp/d1_jb1.log; }
echo "== P5b identity bites: head with qsa_cache_writes' index_state default flipped to False must differ from base on all 14"
M=/tmp/d1_jmut; rm -rf $M; mkdir -p $M; cp -r "$H/src" "$H/tests" $M/
python3 - "$M/src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    compressed_dcp: tuple[jax.Array, int] | None = None,\n    index_state: bool = True,\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, old.replace("True", "False")))
PY
jx "$M" /tmp/d1_jm.log
nm=$(grep -ac '^JAXPR [^ ]* [0-9a-f]\{16\} eqns=' /tmp/d1_jm.log); ne=$(grep -ac '^JAXPR .* ERROR' /tmp/d1_jm.log)
nd=$(diff <(grep -a '^JAXPR ' /tmp/d1_jm.log) <(grep -a '^JAXPR ' /tmp/d1_jb1.log) | grep -c '^<')
echo "P5b mutant: $nm lines, $ne errors, $nd of 14 differ from base"
[ "$nm" = 14 ] && [ "$ne" = 0 ] && [ "$nd" = 14 ] || { FAIL="$FAIL P5b"; tail -20 /tmp/d1_jm.log; }

if [ "$MODE" = smoke ]; then
  [ -z "$FAIL" ] && { echo "Q38_DRAFT1_SUMMARY PASS mode=smoke"; exit 0; }
  echo "Q38_DRAFT1_SUMMARY FAIL mode=smoke:$FAIL"; exit 1
fi

echo "== P3 base screen of the RP_FILES candidates (4 at a time)"
KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
CANDS=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
screen1() { local f=$1 log=/tmp/d1_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
  (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
  echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
export -f screen1 summ; export B CPUENV PT
echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/d1_screen.txt
cat /tmp/d1_screen.txt | sed 's/^/P3 /'
RUN=""
for f in $CANDS; do
  log=/tmp/d1_scr_$(echo $f | tr '/.' '__').log
  n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
  if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
  else case " $(echo $KEY) " in *" $f "*) echo "P3 KEY file counts 0 on base: $f"; FAIL="$FAIL P3-key";; *) echo "P3 EXCLUDED (0 counted on base) $f";; esac; fi
done
echo "== P3 per-file regression HEAD vs BASE (11b0c44ef) over $(echo $RUN | wc -w) files"
. "$W/q38_l3_regress_perfile.sh"
regress_perfile draft1 "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"

echo "== P4 mutants"
t0=$(date +%s)
(cd "$H" && python3 tests/q38_draft1/q38_draft1_mutrun.py .) > /tmp/d1_p4.log 2>&1; rc=$?
grep -aE '^MUTRUN_ORIGIN|^CONTROL|^MUT|^Q38_DRAFT1_MUT' /tmp/d1_p4.log | cut -c1-600
echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc = 0 ] && grep -aq '^Q38_DRAFT1_MUT PASS killed 29/29 of 29' /tmp/d1_p4.log && grep -aq '^MUTRUN_ORIGIN .* INSIDE$' /tmp/d1_p4.log \
  || { FAIL="$FAIL P4"; tail -30 /tmp/d1_p4.log; }

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"

[ -z "$FAIL" ] && { echo "Q38_DRAFT1_SUMMARY PASS mode=full"; exit 0; }
echo "Q38_DRAFT1_SUMMARY FAIL mode=full:$FAIL"; exit 1

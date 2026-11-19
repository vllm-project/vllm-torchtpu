#!/bin/bash
# q38 L4 SEEDFIX CPU proof (VLLM_TPU_SEEDED_SAMPLING; CDK CPU pod only).
# Usage (from the unpacked bundle dir holding head/ base/ probe_admission.py q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l4_seedfix/run_cpu.sh smoke | full | rp | mut LO HI | dist | bench
# head = EXPECT_COMMIT (src EXPECT_SRC); base = EXPECT_BASE_SRC (composition base q38-l4-samp2-vfy a05fdf823).
# PASS rules, fixed before the run:
# P0  head / base src hashes equal EXPECT_SRC / EXPECT_BASE_SRC; the src diff set equals EXPECT_DIFF exactly.
# P0b import origin (cd <tree>; PYTHONPATH=src): every ORIGIN_MODS module resolves inside realpath(<tree>/src) on head and
#     base; with PYTHONPATH = an empty dir every probe reads FALLBACK (a killed mutant); the in-test guard with
#     Q38_SEEDFIX_EXPECT_ROOT = base/src while importing head must FAIL (exactly 1 failed).
# PA  admission (GATE 3 part): TpuPlatform.validate_request on head with the flag unset / 1 / 0 vs base: head-off prints the
#     same PROBE_VALIDATE results as base (seeded -> REJECT ValueError: JAX does not support per-request seed.); head-unset
#     and head-on ACCEPT the seeded request; unseeded and greedy-seeded ACCEPT everywhere.
# P1  tests/q38_l4_seedfix/test_seedfix.py (minus the distribution test) on head: exactly EXPECT_P1 passed, 0 failed /
#     skipped / errors, and the EXPECT_CENSUS census lines.
# P2  compile-key registration rows of VLLM_TPU_SEEDED_SAMPLING (+ the all-together test): exactly EXPECT_P2 passed.
# PX  cross-tree identity (GATE 1 + GATE 3): q38_seedfix_xtree.py on base, head (flag unset) and head (flag 0), compared by
#     its --compare rules (head-off == base everywhere; head-on == base on unseeded rows, uniforms and the shared generator;
#     head-on seeded tokens differ).
# P3  (full, rp) per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over RP_FILES; a candidate that counts 0
#     tests on base is EXCLUDED and named; a KEY file may not be excluded.
# P4  (full) q38_seedfix_mutrun.py: control + every mutant killed; (mut LO HI) the slice LO..HI with its own control.
# PD  (full, dist) the seeded path's accept / reject distribution test (chi2 < 11.34, |z| < 3), 1 passed, 1 SEEDFIX_OUTCOME.
# PB  (full, bench) the host-cost microbench (0 / 1 / all seeded at R = 4 / 8 / 16), its BENCH lines.
# P6  head src unchanged by the run.
# Last line: Q38_SEEDFIX_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_l4_seedfix
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L4|L6|VPS|SEEDED)_[A-Z_]+'); do unset "$v"; done
unset Q38_SEEDFIX_EXPECT_ROOT
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 host $(hostname) nproc $(nproc) $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/  */ /g')"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)) commit $(cat $Q/EXPECT_COMMIT); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort < $Q/EXPECT_DIFF)
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import origin (cd <tree>; PYTHONPATH=src) and its empty-PYTHONPATH mutant"
EMPTY=$(mktemp -d /tmp/sf_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
nprobe=0; nin=0; nkill=0
for t in head base; do
  for w in $(cat $Q/ORIGIN_MODS); do
    mod=${w#head:}; [ "$w" != "$mod" ] && [ $t = base ] && continue
    nprobe=$((nprobe + 1))
    r=$(origin "$W/$t" $mod src); echo "ORIGIN $t $mod ${r#* } ${r%% *}"; [ "${r%% *}" = INSIDE ] && nin=$((nin + 1))
    m=$(origin "$W/$t" $mod "$EMPTY"); echo "ORIGIN $t:MUTANT $mod ${m#* } ${m%% *}"; [ "${m%% *}" = FALLBACK ] && nkill=$((nkill + 1))
  done
done
echo "P0b inside $nin/$nprobe; empty-PYTHONPATH mutants reading FALLBACK $nkill/$nprobe"
[ $nprobe -ge 5 ] && [ $nin = $nprobe ] && [ $nkill = $nprobe ] || FAIL="$FAIL P0b"
(cd "$H" && env $CPUENV Q38_SEEDFIX_EXPECT_ROOT="$(realpath $B/src)" $PT tests/q38_l4_seedfix/test_seedfix.py::test_imports_the_tree_under_test) > /tmp/sf_p0b.log 2>&1; rc=$?
echo "P0b in-test guard mutant (expect base/src while importing head) rc=$rc: $(summ /tmp/sf_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/sf_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-guard"; tail -20 /tmp/sf_p0b.log; }

finish() {
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_SEEDFIX_SUMMARY PASS mode=$*"; exit 0; }
  echo "Q38_SEEDFIX_SUMMARY FAIL mode=$*:$FAIL"; exit 1
}

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI (slice)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_l4_seedfix/q38_seedfix_mutrun.py . $LO $HI) > /tmp/sf_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_SEEDFIX_MUT' /tmp/sf_p4.log | cut -c1-600
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aq "^Q38_SEEDFIX_MUT PASS killed $N/$N of $N range $LO-$HI of " /tmp/sf_p4.log \
    && ! grep -aq '^MUTRUN_ORIGIN .* FALLBACK$' /tmp/sf_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/sf_p4.log; }
  finish "mut range $LO-$HI"
fi

if [ "$MODE" = smoke ] || [ "$MODE" = full ]; then
  echo "== PA admission: TpuPlatform.validate_request, base vs head (flag unset / 1 / 0)"
  pa() { (cd "$1" && env $CPUENV $2 python3 "$W/probe_admission.py" "$3" 2>&1 | grep -aE '^PROBE_VALIDATE\[' | sed -E "s/^PROBE_VALIDATE\[[^]]*\] //"); }
  PB_=$(pa "$B" "" base); PU=$(pa "$H" "" head); P1_=$(pa "$H" "VLLM_TPU_SEEDED_SAMPLING=1" head); P0_=$(pa "$H" "VLLM_TPU_SEEDED_SAMPLING=0" head)
  echo "$PB_" | sed 's/^/PA base /'; echo "$PU" | sed 's/^/PA head-unset /'; echo "$P1_" | sed 's/^/PA head-on /'; echo "$P0_" | sed 's/^/PA head-off /'
  pa_ok=1
  [ "$PB_" = "$P0_" ] && [ "$(echo "$PB_" | wc -l)" = 3 ] || pa_ok=0
  echo "$PB_" | grep -q "seed': 7} sampling_type=RANDOM_SEED -> REJECT ValueError: JAX does not support per-request seed.$" || pa_ok=0
  for x in "$PU" "$P1_"; do
    echo "$x" | grep -q "seed': 7} sampling_type=RANDOM_SEED -> ACCEPT$" || pa_ok=0
    [ "$(echo "$x" | grep -c ' -> ACCEPT$')" = 3 ] || pa_ok=0
  done
  (cd "$B" && env $CPUENV python3 "$W/probe_admission.py" gen 2>&1 | grep -aE '^PROBE_GEN' | sed 's/^/PA /')
  echo "PA head-off == base: $([ "$PB_" = "$P0_" ] && echo yes || echo no); seeded accepted with flag unset / 1: $pa_ok"
  [ $pa_ok = 1 ] || FAIL="$FAIL PA"

  echo "== P1 tests/q38_l4_seedfix/test_seedfix.py + test_seedfix_vrng.py (head; distribution test deselected)"
  t0=$(date +%s)
  (cd "$H" && env $CPUENV timeout -s KILL 5400 $PT -s tests/q38_l4_seedfix/test_seedfix.py tests/q38_l4_seedfix/test_seedfix_vrng.py -k "not distribution") > /tmp/sf_p1.log 2>&1; rc=$?
  grep -aE '^SEEDFIX_[A-Z_]+ |^(FAILED|ERROR) ' /tmp/sf_p1.log | cut -c1-900
  cenok=1
  while read -r tag want; do
    [ -n "$tag" ] || continue
    got=$(grep -acE "^$tag " /tmp/sf_p1.log); echo "P1 $tag lines $got (want $want)"; [ "$got" = "$want" ] || cenok=0
  done < $Q/EXPECT_CENSUS
  echo "P1 rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ /tmp/sf_p1.log) (want $(cat $Q/EXPECT_P1) passed)"
  [ $rc = 0 ] && exact /tmp/sf_p1.log "$(cat $Q/EXPECT_P1)" && [ $cenok = 1 ] || { FAIL="$FAIL P1"; grep -aE '^E  |^_____|Error|assert' /tmp/sf_p1.log | head -150; tail -60 /tmp/sf_p1.log; }

  echo "== P2 compile-key registration rows of VLLM_TPU_SEEDED_SAMPLING"
  (cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "SEEDED_SAMPLING or together") > /tmp/sf_p2.log 2>&1; rc=$?
  echo "P2 rc=$rc summary: $(summ /tmp/sf_p2.log) (want $(cat $Q/EXPECT_P2) passed)"
  [ $rc = 0 ] && exact /tmp/sf_p2.log "$(cat $Q/EXPECT_P2)" || { FAIL="$FAIL P2"; tail -30 /tmp/sf_p2.log; }

  echo "== PX cross-tree identity: base vs head (flag unset) vs head (flag 0)"
  XT=$Q/q38_seedfix_xtree.py
  (cd "$B" && env $CPUENV timeout -s KILL 2400 python3 $XT base) > /tmp/sf_xb.log 2>&1; rb=$?
  (cd "$H" && env $CPUENV timeout -s KILL 2400 python3 $XT head_on) > /tmp/sf_xon.log 2>&1; ron=$?
  (cd "$H" && env $CPUENV VLLM_TPU_SEEDED_SAMPLING=0 timeout -s KILL 2400 python3 $XT head_off) > /tmp/sf_xoff.log 2>&1; roff=$?
  grep -ah '^XTREE_ORIGIN\|^XTREE_DONE' /tmp/sf_xb.log /tmp/sf_xon.log /tmp/sf_xoff.log | sed 's/^/PX /'
  python3 $XT --compare /tmp/sf_xb.log /tmp/sf_xon.log /tmp/sf_xoff.log > /tmp/sf_xc.log 2>&1; rc=$?
  sed 's/^/PX /' /tmp/sf_xc.log | head -60
  echo "PX rc base=$rb on=$ron off=$roff compare=$rc"
  [ $rb = 0 ] && [ $ron = 0 ] && [ $roff = 0 ] && [ $rc = 0 ] && grep -q '^XTREE_SUMMARY PASS' /tmp/sf_xc.log \
    && grep -q 'inside=True seedfix=no' /tmp/sf_xb.log && grep -q 'inside=True seedfix=yes' /tmp/sf_xon.log \
    || { FAIL="$FAIL PX"; tail -30 /tmp/sf_xb.log /tmp/sf_xon.log /tmp/sf_xoff.log; }
fi

if [ "$MODE" = full ] || [ "$MODE" = rp ]; then
  echo "== P3 base screen of the RP_FILES candidates (4 at a time)"
  KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
  CANDS=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
  screen1() { local f=$1 log=/tmp/sf_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
    (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
    echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
  export -f screen1 summ; export B CPUENV PT
  echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/sf_screen.txt
  sed 's/^/P3 /' /tmp/sf_screen.txt
  RUN=""
  for f in $CANDS; do
    log=/tmp/sf_scr_$(echo $f | tr '/.' '__').log
    n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
    if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
    else case " $(echo $KEY) " in *" $f "*) echo "P3 KEY file counts 0 on base: $f"; FAIL="$FAIL P3-key";; *) echo "P3 EXCLUDED (0 counted on base) $f";; esac; fi
  done
  echo "== P3 per-file regression HEAD vs BASE over $(echo $RUN | wc -w) files"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile seedfix "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"
fi

if [ "$MODE" = full ]; then
  echo "== P4 mutants (all)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_l4_seedfix/q38_seedfix_mutrun.py .) > /tmp/sf_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_SEEDFIX_MUT' /tmp/sf_p4.log | cut -c1-600
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aqE '^Q38_SEEDFIX_MUT PASS killed ([0-9]+)/\1 of \1 ' /tmp/sf_p4.log \
    && ! grep -aq '^MUTRUN_ORIGIN .* FALLBACK$' /tmp/sf_p4.log || { FAIL="$FAIL P4"; tail -60 /tmp/sf_p4.log; }
fi

if [ "$MODE" = full ] || [ "$MODE" = dist ]; then
  echo "== PD seeded-path accept / reject distribution"
  t0=$(date +%s)
  (cd "$H" && env $CPUENV timeout -s KILL 7000 $PT -s tests/q38_l4_seedfix/test_seedfix.py -k distribution) > /tmp/sf_pd.log 2>&1; rc=$?
  grep -aE '^SEEDFIX_OUTCOME |^(FAILED|ERROR) |^E ' /tmp/sf_pd.log | cut -c1-900
  echo "PD rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ /tmp/sf_pd.log)"
  [ $rc = 0 ] && [ "$(count /tmp/sf_pd.log passed)" = 1 ] && [ "$(grep -ac '^SEEDFIX_OUTCOME ' /tmp/sf_pd.log)" = 1 ] || { FAIL="$FAIL PD"; tail -40 /tmp/sf_pd.log; }
fi

if [ "$MODE" = full ] || [ "$MODE" = bench ]; then
  echo "== PB host-cost microbench"
  (cd "$H" && env $CPUENV timeout -s KILL 5000 python3 tests/q38_l4_seedfix/q38_seedfix_bench.py) > /tmp/sf_pb.log 2>&1; rc=$?
  grep -aE '^BENCH' /tmp/sf_pb.log | cut -c1-400
  echo "PB rc=$rc"
  [ $rc = 0 ] && grep -aq '^BENCH_DONE' /tmp/sf_pb.log || { FAIL="$FAIL PB"; tail -40 /tmp/sf_pb.log; }
fi

finish "$MODE"

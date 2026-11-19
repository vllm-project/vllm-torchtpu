#!/bin/bash
# q38 L4 VPSHOST CPU proof (CDK CPU pod only; never the Mac or the VM).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l4_vpshost/run_cpu.sh smoke | full | rp | mut LO HI
# head = branch q38-l4-vpshost (EXPECT_COMMIT, src EXPECT_SRC); base = q38-L1-v3c b551e9b84 (src EXPECT_BASE_SRC).
# PASS rules, fixed before the run:
# P0  head / base src hashes equal EXPECT_SRC / EXPECT_BASE_SRC; the src diff set equals EXPECT_DIFF exactly.
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the changed modules resolve inside realpath(<tree>/src)
#     on head and base; the same imports with PYTHONPATH = an empty dir read FALLBACK (every probe a killed mutant);
#     the in-test guard with Q38_VPSHOST_EXPECT_ROOT = base/src while importing head must FAIL (exactly 1 failed).
# P1  tests/q38_l4_vpshost/test_l4_vpshost.py on head: exactly EXPECT_P1 passed, 0 failed, 0 skipped, 0 errors, and
#     EXPECT_CENSUS SAMPFB_CENSUS lines.
# P2  compile-key registration rows of the VPSHOST envs (+ the all-together test): exactly EXPECT_P2 passed, 0 failed.
# P5  compile identity: q38_vpshost_compiled_identity.py prints the same COMPILED lines on head and base (rc 0, no
#     LEVERREAD); P5b the census bites: a head copy with fallback_rows forced to 0 inside _vps_verify_fb_impl differs
#     from base in exactly 1 COMPILED line.
# B   bench_vpshost.py on head exits 0 with BENCH_DONE (descriptive numbers; no pass bar).
# (smoke = P0 P0b P1 P2 P5 P5b B)
# P3  (full, rp) per-file regression HEAD vs BASE (L3's regress_perfile v2 4fb6f596) over RP_FILES; a candidate that counts
#     0 tests on base is EXCLUDED and named; a KEY file may not be excluded.
# P4  (full) q38_vpshost_mutrun.py: control + every mutant killed ('Q38_VPSHOST_MUT PASS killed N/N of N');
#     (mut LO HI) the slice LO..HI with its own control and origin probe.
# P6  head src unchanged by the run.
# Last line: Q38_VPSHOST_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_l4_vpshost
export JAX_PLATFORMS=cpu
for v in $(env | grep -oE '^VLLM_TPU_(L4|L6)_[A-Z_]+'); do unset "$v"; done
unset Q38_VPSHOST_EXPECT_ROOT
PT="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1"
FAIL=""
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
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
EMPTY=$(mktemp -d /tmp/vh_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
# ORIGIN_MODS: one module per word; a "head:" prefix marks a module new on head (probed on head only).
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
(cd "$H" && env $CPUENV Q38_VPSHOST_EXPECT_ROOT="$(realpath $B/src)" $PT tests/q38_l4_vpshost/test_l4_vpshost.py::test_imports_the_tree_under_test) > /tmp/vh_p0b.log 2>&1; rc=$?
echo "P0b in-test guard mutant (expect base/src while importing head) rc=$rc: $(summ /tmp/vh_p0b.log)"
[ $rc != 0 ] && [ "$(count /tmp/vh_p0b.log failed)" = 1 ] || { FAIL="$FAIL P0b-guard"; tail -20 /tmp/vh_p0b.log; }

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P4 mutants $LO-$HI (slice)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_l4_vpshost/q38_vpshost_mutrun.py . $LO $HI) > /tmp/vh_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_VPSHOST_MUT' /tmp/vh_p4.log | cut -c1-600
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aq "^Q38_VPSHOST_MUT PASS killed $N/$N of $N range $LO-$HI of " /tmp/vh_p4.log \
    && ! grep -aq '^MUTRUN_ORIGIN .* FALLBACK$' /tmp/vh_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/vh_p4.log; }
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_VPSHOST_SUMMARY PASS mode=mut range $LO-$HI"; exit 0; }
  echo "Q38_VPSHOST_SUMMARY FAIL mode=mut range $LO-$HI:$FAIL"; exit 1
fi

if [ "$MODE" = smoke ] || [ "$MODE" = full ]; then
  echo "== P1 tests/q38_l4_vpshost/test_l4_vpshost.py (head)"
  t0=$(date +%s)
  (cd "$H" && env $CPUENV $PT -s tests/q38_l4_vpshost/test_l4_vpshost.py) > /tmp/vh_p1.log 2>&1; rc=$?
  grep -aE '(SAMPFB|STATSASYNC|SAMPSNAP)[A-Z]*_[A-Z]+ |^(FAILED|ERROR) ' /tmp/vh_p1.log | cut -c1-700
  cenok=1
  while read -r tag want; do
    [ -n "$tag" ] || continue
    got=$(grep -acE "$tag rows=" /tmp/vh_p1.log); echo "P1 $tag lines $got (want $want)"; [ "$got" = "$want" ] || cenok=0
  done < $Q/EXPECT_CENSUS
  echo "P1 rc=$rc $(( $(date +%s) - t0 ))s summary: $(summ /tmp/vh_p1.log) (want $(cat $Q/EXPECT_P1) passed)"
  [ $rc = 0 ] && exact /tmp/vh_p1.log "$(cat $Q/EXPECT_P1)" && [ $cenok = 1 ] || { FAIL="$FAIL P1"; tail -80 /tmp/vh_p1.log; }

  echo "== P2 compile-key registration rows of the VPSHOST envs"
  (cd "$H" && env $CPUENV $PT tests/compilation/test_tpu_compile_key_registration.py -k "VLLM_TPU_L4_ or together") > /tmp/vh_p2.log 2>&1; rc=$?
  echo "P2 rc=$rc summary: $(summ /tmp/vh_p2.log)"
  [ $rc = 0 ] && exact /tmp/vh_p2.log "$(cat $Q/EXPECT_P2)" || { FAIL="$FAIL P2"; tail -30 /tmp/vh_p2.log; }

  echo "== P5 compile identity (head vs base b551e9b84)"
  python3 -I $Q/q38_vpshost_compiled_identity.py "$H" > /tmp/vh_ch.log 2>&1; rch=$?
  python3 -I $Q/q38_vpshost_compiled_identity.py "$B" > /tmp/vh_cb.log 2>&1; rcb=$?
  echo "P5 head rc=$rch $(grep -a '^COMPILED_ALL' /tmp/vh_ch.log); base rc=$rcb $(grep -a '^COMPILED_ALL' /tmp/vh_cb.log); LEVERREAD head $(grep -ac '^LEVERREAD' /tmp/vh_ch.log)"
  grep -a '^COMPILED ' /tmp/vh_ch.log | sed 's/^/P5 /'
  [ $rch = 0 ] && [ "$(grep -ac '^COMPILED ' /tmp/vh_ch.log)" -ge 5 ] \
    && cmp -s <(grep -a '^COMPILED' /tmp/vh_ch.log) <(grep -a '^COMPILED' /tmp/vh_cb.log) \
    || { FAIL="$FAIL P5"; diff <(grep -a '^COMPILED' /tmp/vh_ch.log) <(grep -a '^COMPILED' /tmp/vh_cb.log) | head; cat /tmp/vh_ch.log | grep -a LEVERREAD; }
  M=/tmp/vh_cmut; rm -rf $M; mkdir -p $M; cp -r "$H/src" $M/
  python3 - "$M/src/vllm_torchtpu/runner/tpu_runner.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        fallback_rows=fallback_rows,\n        cand_draws=cand_draws,\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "        fallback_rows=0,\n        cand_draws=cand_draws,\n"))
PY
  python3 -I $Q/q38_vpshost_compiled_identity.py "$M" > /tmp/vh_cm.log 2>&1
  nd=$(diff <(grep -a '^COMPILED ' /tmp/vh_cm.log) <(grep -a '^COMPILED ' /tmp/vh_cb.log) | grep -c '^<')
  echo "P5b identity bites: mutated head differs from base in $nd COMPILED line(s) (want 1): $(diff <(grep -a '^COMPILED ' /tmp/vh_cm.log) <(grep -a '^COMPILED ' /tmp/vh_cb.log) | grep '^<' | head -2 | tr '\n' ' ')"
  [ "$nd" = 1 ] || FAIL="$FAIL P5b"

  echo "== B bench_vpshost.py (served image CPU; descriptive)"
  (cd "$H" && env $CPUENV timeout -s KILL 1800 python3 tests/q38_l4_vpshost/bench_vpshost.py) > /tmp/vh_b.log 2>&1; rc=$?
  grep -aE '^BENCH' /tmp/vh_b.log
  [ $rc = 0 ] && grep -aq '^BENCH_DONE' /tmp/vh_b.log || { FAIL="$FAIL B"; tail -40 /tmp/vh_b.log; }
fi

if [ "$MODE" = full ] || [ "$MODE" = rp ]; then
  echo "== P3 base screen of the RP_FILES candidates (4 at a time)"
  KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
  CANDS=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
  screen1() { local f=$1 log=/tmp/vh_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
    (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
    echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
  export -f screen1 summ; export B CPUENV PT
  echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/vh_screen.txt
  sed 's/^/P3 /' /tmp/vh_screen.txt
  RUN=""
  for f in $CANDS; do
    log=/tmp/vh_scr_$(echo $f | tr '/.' '__').log
    n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
    if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
    else case " $(echo $KEY) " in *" $f "*) echo "P3 KEY file counts 0 on base: $f"; FAIL="$FAIL P3-key";; *) echo "P3 EXCLUDED (0 counted on base) $f";; esac; fi
  done
  echo "== P3 per-file regression HEAD vs BASE (b551e9b84) over $(echo $RUN | wc -w) files"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile vpshost "$H" "$B" 1500 $RUN || FAIL="$FAIL P3"
fi

if [ "$MODE" = full ]; then
  echo "== P4 mutants (all)"
  t0=$(date +%s)
  (cd "$H" && python3 tests/q38_l4_vpshost/q38_vpshost_mutrun.py .) > /tmp/vh_p4.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT|^Q38_VPSHOST_MUT' /tmp/vh_p4.log | cut -c1-600
  echo "P4 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aqE '^Q38_VPSHOST_MUT PASS killed ([0-9]+)/\1 of \1 ' /tmp/vh_p4.log \
    && ! grep -aq '^MUTRUN_ORIGIN .* FALLBACK$' /tmp/vh_p4.log || { FAIL="$FAIL P4"; tail -40 /tmp/vh_p4.log; }
fi

HS2=$(srch "$H")
echo "P6 head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
[ -z "$FAIL" ] && { echo "Q38_VPSHOST_SUMMARY PASS mode=$MODE"; exit 0; }
echo "Q38_VPSHOST_SUMMARY FAIL mode=$MODE:$FAIL"; exit 1

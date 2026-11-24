#!/bin/bash
# q38_l3_regress_perfile.sh -- L3, 2026-11-13 (lead: L2's empty-vs-empty lesson). Source it inside a CPU-job proof body.
#   regress_perfile <tag> <tip root> <ctl root> <timeout s> <file> [<file> ...]
# Runs pytest SEPARATELY per test file on the tip tree and on the control tree, and compares per file. A file FAILS the leg if,
# on either side, its log has no pytest summary line ("N passed / failed / error ... in Ns"), or the summary counts 0 tests,
# or it reports "no tests ran" / collection errors, or the tip has a failing test id the control does not have. Tests that fail
# on BOTH sides are printed as UNCOMPARED (they carry no regression signal). Uses $PT (pytest command, word-split) and $CPUENV
# (env prefix, word-split) from the caller. Prints one line per file and a final '<tag> REGRESS_PERFILE rc=<0|1>'; returns rc.
regress_count() {  # total tests in the LAST pytest summary line of $1 (0 if none)
  local s; s=$(grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|xfailed|xpassed)(, )?)+.* in [0-9.]+s' "$1" | tail -1)
  [ -n "$s" ] || { echo 0; return; }
  echo "$s" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}'
}
regress_fails() { grep -aE '^(FAILED|ERROR) ' "$1" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' | sort -u; }
regress_perfile() {
  local tag=$1 tip=$2 ctl=$3 to=$4; shift 4
  local rc=0 f i=0 lt lc nt nc new both
  [ $# -ge 1 ] || { echo "$tag REGRESS_PERFILE no files -> FAIL"; return 1; }
  for f in "$@"; do
    i=$((i + 1)); lt=/tmp/${tag}_rp_${i}_tip.log; lc=/tmp/${tag}_rp_${i}_ctl.log
    [ -f "$tip/$f" ] && [ -f "$ctl/$f" ] || { echo "$tag RP $f: missing on tip or ctl -> FAIL"; rc=1; continue; }
    (cd "$tip" && timeout -s KILL "$to" env $CPUENV $PT "$f" > "$lt" 2>&1)
    (cd "$ctl" && timeout -s KILL "$to" env $CPUENV $PT "$f" > "$lc" 2>&1)
    nt=$(regress_count "$lt"); nc=$(regress_count "$lc")
    new=$(comm -23 <(regress_fails "$lt") <(regress_fails "$lc") | tr '\n' ' ')
    both=$(comm -12 <(regress_fails "$lt") <(regress_fails "$lc") | wc -l | tr -d ' ')
    if [ "$nt" -lt 1 ] || [ "$nc" -lt 1 ] || grep -qa 'no tests ran' "$lt" "$lc" || grep -qaE 'errors? during collection' "$lt" "$lc"; then
      echo "$tag RP $f: tip tests $nt ctl tests $nc (no summary / zero tests / collection error) -> FAIL"; rc=1
    elif [ -n "$new" ]; then
      echo "$tag RP $f: tip tests $nt ctl tests $nc; NEW failures on tip: $new-> FAIL"; rc=1
    else
      echo "$tag RP $f: tip tests $nt ctl tests $nc; new failures none; UNCOMPARED (fail on both) $both -> ok"
    fi
  done
  echo "$tag REGRESS_PERFILE rc=$rc"; return $rc
}

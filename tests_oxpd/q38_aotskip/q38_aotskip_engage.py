#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-H14-AOTSKIP engagement reader (stdlib only; run with python3 -I).

Reads one arm's log segment(s) and counts, per process rank, the lines
compilation/aot_guard_skip.py writes for every AOT artifact:
  AOT_GUARD_SKIP engaged: rank=R cls=C guard_check_enabled=False      (fresh compile, skip applied)
  AOT_GUARD_SKIP cache-loaded: rank=R cls=C guard_check_enabled=B     (cache load; counts only if B is False)
  AOT_GUARD_SKIP not engaged: rank=R cls=C (...)                      (always bad)

Engaged mode (default): ENGAGED iff exactly --ranks distinct ranks appear, every
rank has at least --min-per-rank good lines (engaged + cache-loaded with
guard_check_enabled=False), and there is no bad line (not engaged, or a
cache-loaded artifact whose guard check is still on). Otherwise NOT_ENGAGED.
Off mode (--expect-off, for arms without the flag): OFF iff no AOT_GUARD_SKIP
line and no install line appear; otherwise LEAK.

Prints one line:
  AOTSKIP_ENG mode=<on|off> ranks=<seen>/<want> min_good=<k> good=<n> bad=<n> classes=<C:n,...> verdict=<V> why=<...>
Exit 0 iff the verdict is ENGAGED (engaged mode) or OFF (off mode).
"""

import argparse
import re
import sys

LINE = re.compile(
    r"AOT_GUARD_SKIP (engaged|cache-loaded|not engaged): rank=(-?\d+) cls=(\S+)"
    r"(?: guard_check_enabled=(\S+))?"
)
INSTALL = "AOT-compiled models skip the per-call guard check"


def read(lines, ranks, min_per_rank, expect_off):
    good, bad, classes = {}, 0, {}
    seen = set()
    install = 0
    any_line = 0
    for ln in lines:
        if INSTALL in ln:
            install += 1
        m = LINE.search(ln)
        if not m:
            continue
        any_line += 1
        kind, rank, cls, state = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        seen.add(rank)
        classes[cls] = classes.get(cls, 0) + 1
        ok = (kind == "engaged" and state == "False") or (
            kind == "cache-loaded" and state == "False"
        )
        if ok:
            good[rank] = good.get(rank, 0) + 1
        else:
            bad += 1
    cls_s = ",".join(f"{k}:{v}" for k, v in sorted(classes.items())) or "-"
    if expect_off:
        verdict = "OFF" if any_line == 0 and install == 0 else "LEAK"
        why = "no AOT_GUARD_SKIP lines" if verdict == "OFF" else (
            f"{any_line} skip lines, {install} install lines in an arm without the flag")
        return (f"AOTSKIP_ENG mode=off ranks={len(seen)}/{ranks} min_good=0 good="
                f"{sum(good.values())} bad={bad} classes={cls_s} verdict={verdict} why={why}"), verdict == "OFF"
    min_good = min((good.get(r, 0) for r in seen), default=0)
    why = []
    if len(seen) != ranks:
        why.append(f"ranks seen {len(seen)} != {ranks}")
    if min_good < min_per_rank:
        why.append(f"min good per rank {min_good} < {min_per_rank}")
    if bad:
        why.append(f"{bad} bad lines")
    verdict = "ENGAGED" if not why else "NOT_ENGAGED"
    return (f"AOTSKIP_ENG mode=on ranks={len(seen)}/{ranks} min_good={min_good} good="
            f"{sum(good.values())} bad={bad} classes={cls_s} verdict={verdict} why="
            f"{'; '.join(why) or 'ok'}"), verdict == "ENGAGED"


def selftest():
    def L(kind, r, cls="M", st="False"):
        tail = f" guard_check_enabled={st}" if kind != "not engaged" else " (evaluate_guards=True, x)"
        return f"(Worker pid=1) INFO 11-13 [aot_guard_skip.py:70] AOT_GUARD_SKIP {kind}: rank={r} cls={cls}{tail}"
    full = [L("engaged", r, c) for r in range(8) for c in ("A", "B", "C")]
    cases = [
        ("all ranks 3 fresh", full, 8, 3, False, True),
        ("one rank short", full[:-1], 8, 3, False, False),
        ("a rank missing", [x for x in full if "rank=7 " not in x], 8, 3, False, False),
        ("cache-loaded off counts", [L("cache-loaded", r, c) for r in range(8) for c in "ABC"], 8, 3, False, True),
        ("cache-loaded still on", full + [L("cache-loaded", 0, "D", "True")], 8, 3, False, False),
        ("a not-engaged line", full + [L("not engaged", 2)], 8, 3, False, False),
        ("engaged but state True", [L("engaged", r, c, "True") for r in range(8) for c in "ABC"], 8, 3, False, False),
        ("no lines (flag ignored)", ["INFO something else"], 8, 3, False, False),
        ("off mode clean", ["INFO x", "INFO y"], 8, 0, True, True),
        ("off mode leak", ["INFO " + INSTALL], 8, 0, True, False),
        ("off mode leak line", [L("engaged", 0)], 8, 0, True, False),
    ]
    bad = 0
    for name, lines, ranks, mn, off, want in cases:
        line, ok = read(lines, ranks, mn, off)
        if ok != want:
            bad += 1
            print(f"SELFTEST FAIL {name}: got {ok}, want {want}: {line}")
    n = len(cases)
    print(f"AOTSKIP_ENG_SELFTEST {'PASS' if not bad else 'FAIL'} ({n - bad}/{n} cases)")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--min-per-rank", type=int, default=3)
    ap.add_argument("--expect-off", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.logs:
        print("AOTSKIP_ENG verdict=NO_INPUT")
        return 2
    lines = []
    for p in a.logs:
        with open(p, errors="replace") as f:
            lines.extend(f)
    line, ok = read(lines, a.ranks, a.min_per_rank, a.expect_off)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

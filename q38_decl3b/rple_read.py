# SPDX-License-Identifier: Apache-2.0
"""q38 DECODE-L3b-RPLEW reader (2026-11-19). Reads the bench JSON (tests/q38_decl3b/tpu_bench_rple.py) and applies the
PRE-STATED rules of the R-form PLE car (layers W and U of scratch/decl3b/rple_design_v1.md). Pattern from DECODE-L3-INJMXU-2's
q38_decl3/injmxu_read.py (credit DECODE-L3, DECODE-L3-INJMXU(-2)).

  Per-call us per case from the bench: base / base2 = arm A (RM4096's PLE flags) alone before and after; A / W / WU interleaved
  (W = A + VLLM_TPU_PLE_WRDEFER=4; WU = W + VLLM_TPU_PLE_UNIFORM_COND=1). Rows 128 / 160 / 192 / 256 (uniform k3 windows,
  count 40 / 40 / 48 / 64) and the non-uniform row 160nu (one request of 1 token; pred False, the cond's stock branch).
  VOID rows (any one -> VERDICT VOID):
    PLAN        rows, counts, arms, cases, chain 24 / 4 or the non-uniform row differ from the plan
    COMPILE     any (row, arm) not compiled, or any planned timing case missing (base, A, W, WU, base2 at every row + 160nu)
    ENGAGEMENT  any (row, arm) whose chain does not engage its arm (A: no ring, no cond; W: ring, no cond; WU: ring + cond)
    HEALTH      any row whose W / WU pool bytes or output bits differ from A's
    PRED        pred not True at a uniform row, or not False at 160nu
    SPREAD      at any row, |base - base2| / min(base, base2) > 5 %
    BARRIER     the bench's final libtpu string does not carry exactly one --xla_tpu_use_enhanced_launch_barrier=false
  KEEP W   iff W - A <= -50 us per call at 160, and W - A <= |base - base2| at 128, 192 and 256.
  KEEP WU  iff WU - A <= -100 us per call at 160 AND 192, WU - A <= |base - base2| at 128 and 256, and at 160nu
           WU - W <= |base - base2| (the cond's stock branch costs what W costs, within the row's spread).
  Verdicts: 'VERDICT KEEP W+WU' / 'VERDICT KEEP W' / 'VERDICT KEEP WU (W bar missed)' / 'VERDICT NOT-KEEP' /
  'VERDICT VOID (<rows>)'. Descriptive only: every delta, the split halves, and the deltas against the prediction (I).
Exit 0 for any KEEP / NOT-KEEP (a valid read), 1 for VERDICT VOID, 2 when there is no JSON.
"""
from __future__ import annotations

import copy
import json
import os
import sys

PLAN_ROWS = (128, 160, 192, 256)
PLAN_COUNTS = {"128": 40, "160": 40, "192": 48, "256": 64}
PLAN_ARMS = ("A", "W", "WU")
PLAN_CASES = ("base", "A", "W", "WU", "base2")
PLAN_CHAIN = (24, 4)
NU = "160nu"
SPREAD_MAX = 0.05
W_KEEP_US = -50.0
W_KEEP_ROWS = (160,)
W_GUARD_ROWS = (128, 192, 256)
WU_KEEP_US = -100.0
WU_KEEP_ROWS = (160, 192)
WU_GUARD_ROWS = (128, 256)
PRED_US = {"W": {128: -69, 160: -87, 192: -105, 256: -140}, "WU": {160: (-227, -367)}}  # I, rple_design_v1 sections 2-3
LB = "--xla_tpu_use_enhanced_launch_barrier"
KEYS = [str(r) for r in PLAN_ROWS] + [NU]


def barrier_ok(s) -> bool:
    toks = [t for t in (s or "").split() if t.split("=")[0] == LB]
    return toks == [LB + "=false"]


def us(d, key, case):
    v = (((d.get("cases") or {}).get(key) or {}).get(case) or {}).get("us_per_call")
    return float(v) if isinstance(v, (int, float)) else None


def spread_us(d, key):
    return abs(us(d, key, "base") - us(d, key, "base2"))


def check_plan(d) -> list:
    p = d.get("plan") or {}
    bad = []
    if tuple(p.get("rows") or ()) != PLAN_ROWS:
        bad.append(f"rows {p.get('rows')}")
    if (p.get("counts") or {}) != PLAN_COUNTS:
        bad.append(f"counts {p.get('counts')}")
    if tuple(p.get("arms") or ()) != PLAN_ARMS:
        bad.append(f"arms {p.get('arms')}")
    if tuple(p.get("cases") or ()) != PLAN_CASES:
        bad.append(f"cases {p.get('cases')}")
    if p.get("nonuniform_row") != 160:
        bad.append(f"nonuniform_row {p.get('nonuniform_row')}")
    if tuple((d.get("host") or {}).get("chain") or ()) != PLAN_CHAIN:
        bad.append(f"chain {(d.get('host') or {}).get('chain')}")
    return bad


def check_compile(d) -> list:
    bad = []
    for k in KEYS:
        for a in PLAN_ARMS:
            rec = ((d.get("engagement") or {}).get(k) or {}).get(a) or {}
            if not rec.get("compiled"):
                bad.append(f"compile {a}@{k}")
        for c in PLAN_CASES:
            if us(d, k, c) is None:
                bad.append(f"{c}@{k}")
    return bad


def check_engagement(d) -> list:
    return [f"{a}@{k}" for k in KEYS for a in PLAN_ARMS
            if (((d.get("engagement") or {}).get(k) or {}).get(a) or {}).get("compiled")
            and (((d.get("engagement") or {}).get(k) or {}).get(a) or {}).get("engaged") is not True]


def check_health(d) -> list:
    return [k for k in KEYS if not ((d.get("health") or {}).get(k) or {}).get("ok")]


def check_pred(d) -> list:
    p = d.get("pred") or {}
    bad = [k for k in KEYS[:-1] if p.get(k) is not True]
    if p.get(NU) is not False:
        bad.append(NU)
    return bad


def check_spread(d) -> list:
    bad = []
    for k in KEYS:
        b, b2 = us(d, k, "base"), us(d, k, "base2")
        if b is None or b2 is None:
            continue
        if b <= 0 or b2 <= 0 or abs(b - b2) / min(b, b2) > SPREAD_MAX:
            bad.append(f"{k} ({b:.1f} vs {b2:.1f})")
    return bad


def keep_w(d) -> bool:
    ok = all(us(d, str(r), "W") - us(d, str(r), "A") <= W_KEEP_US for r in W_KEEP_ROWS)
    return ok and all(us(d, str(r), "W") - us(d, str(r), "A") <= spread_us(d, str(r)) for r in W_GUARD_ROWS)


def keep_wu(d) -> bool:
    ok = all(us(d, str(r), "WU") - us(d, str(r), "A") <= WU_KEEP_US for r in WU_KEEP_ROWS)
    ok = ok and all(us(d, str(r), "WU") - us(d, str(r), "A") <= spread_us(d, str(r)) for r in WU_GUARD_ROWS)
    return ok and (us(d, NU, "WU") - us(d, NU, "W")) <= spread_us(d, NU)


def verdict(d) -> tuple:
    void = []
    for name, fn in (("PLAN", check_plan), ("COMPILE", check_compile), ("ENGAGEMENT", check_engagement),
                     ("HEALTH", check_health), ("PRED", check_pred), ("SPREAD", check_spread)):
        bad = fn(d)
        if bad:
            void.append(f"{name}: {', '.join(bad[:8])}")
    if not barrier_ok((d.get("host") or {}).get("libtpu_final")):
        void.append("BARRIER")
    if void:
        return "VERDICT VOID (" + "; ".join(void) + ")", 1
    w, wu = keep_w(d), keep_wu(d)
    if w and wu:
        return "VERDICT KEEP W+WU", 0
    if w:
        return "VERDICT KEEP W", 0
    if wu:
        return "VERDICT KEEP WU (W bar missed)", 0
    return "VERDICT NOT-KEEP", 0


def describe(d) -> list:
    out = []
    for k in KEYS:
        b, b2 = us(d, k, "base"), us(d, k, "base2")
        a, w, wu = us(d, k, "A"), us(d, k, "W"), us(d, k, "WU")
        if None in (b, b2, a, w, wu):
            continue
        r = int(k[:3])
        pw = PRED_US["W"].get(r) if k != NU else None
        out.append(f"ROW {k}: base {b:.1f} A {a:.1f} W {w:.1f} WU {wu:.1f} base2 {b2:.1f} us/call; W-A {w - a:+.1f} "
                   f"WU-A {wu - a:+.1f} WU-W {wu - w:+.1f} |base-base2| {abs(b - b2):.1f}"
                   + (f"; prediction (I) W-A {pw}" if pw is not None else "")
                   + (f", WU-A {PRED_US['WU'][r]}" if k == '160' else ""))
    return out


def _synthetic(keep_w_ok=True, keep_wu_ok=True):
    cases = {}
    for k in KEYS:
        a = {"128": 600.0, "160": 700.0, "192": 800.0, "256": 1000.0, NU: 700.0}[k]
        w = a - (90.0 if keep_w_ok else 30.0)
        wu = w - 150.0 if keep_wu_ok else w + 10.0
        if k == NU:
            wu = w + 2.0
        cases[k] = {"base": {"us_per_call": a + 3}, "A": {"us_per_call": a}, "W": {"us_per_call": w},
                    "WU": {"us_per_call": wu}, "base2": {"us_per_call": a - 3}}
    eng = {k: {"A": {"compiled": True, "engaged": True}, "W": {"compiled": True, "engaged": True},
               "WU": {"compiled": True, "engaged": True}} for k in KEYS}
    return {"plan": {"rows": list(PLAN_ROWS), "counts": dict(PLAN_COUNTS), "arms": list(PLAN_ARMS),
                     "cases": list(PLAN_CASES), "nonuniform_row": 160},
            "host": {"chain": list(PLAN_CHAIN), "libtpu_final": "--a=1 " + LB + "=false"},
            "cases": cases, "engagement": eng, "health": {k: {"ok": True} for k in KEYS},
            "pred": {**{k: True for k in KEYS[:-1]}, NU: False}}


def _scenarios():
    s = {"keep both": (_synthetic(), "VERDICT KEEP W+WU"),
         "keep W only": (_synthetic(keep_wu_ok=False), "VERDICT KEEP W"),
         "not keep": (_synthetic(False, False), "VERDICT NOT-KEEP")}
    d = _synthetic(keep_w_ok=False)
    s["keep WU only"] = (d, "VERDICT KEEP WU (W bar missed)")
    d = _synthetic()
    d["cases"][NU]["WU"]["us_per_call"] = d["cases"][NU]["W"]["us_per_call"] + 20.0  # cond overhead above the spread
    s["cond overhead"] = (d, "VERDICT KEEP W")
    d = _synthetic()
    d["cases"]["256"]["W"]["us_per_call"] = d["cases"]["256"]["A"]["us_per_call"] + 10.0  # W slower at a guard row
    s["W guard row"] = (d, "VERDICT KEEP WU (W bar missed)")
    d = _synthetic()
    d["cases"]["192"]["WU"]["us_per_call"] = d["cases"]["192"]["A"]["us_per_call"] - 60.0  # WU bar missed at 192
    s["WU bar at 192"] = (d, "VERDICT KEEP W")
    for name, mut in (("missing case", lambda x: x["cases"]["192"].pop("base2")),
                      ("not compiled", lambda x: x["engagement"]["128"]["WU"].update(compiled=False)),
                      ("not engaged", lambda x: x["engagement"][NU]["WU"].update(engaged=False)),
                      ("health", lambda x: x["health"]["160"].update(ok=False)),
                      ("pred uniform", lambda x: x["pred"].update({"192": False})),
                      ("pred nonuniform", lambda x: x["pred"].update({NU: True})),
                      ("spread", lambda x: x["cases"]["160"]["base2"].update(us_per_call=600.0)),
                      ("barrier", lambda x: x["host"].update(libtpu_final=LB + "=true")),
                      ("plan rows", lambda x: x["plan"].update(rows=[128, 160])),
                      ("plan counts", lambda x: x["plan"]["counts"].update({"160": 48})),
                      ("plan chain", lambda x: x["host"].update(chain=[8, 4]))):
        d = _synthetic()
        mut(d)
        s["void " + name] = (d, "VERDICT VOID")
    return s


def selftest(mutants: bool) -> int:
    bad = 0
    for name, (d, want) in _scenarios().items():
        got, _ = verdict(copy.deepcopy(d))
        ok = got.startswith(want) if want == "VERDICT VOID" else got == want
        bad += not ok
        print(f"Q38_RPLE_READ_SELFTEST {name}: {got[:120]} ({'ok' if ok else 'BAD, want ' + want})")
    if mutants:
        g = globals()
        muts = {
            "W bar -30 us": ("W_KEEP_US", -30.0), "WU bar -50 us": ("WU_KEEP_US", -50.0),
            "WU keep rows 160 only": ("WU_KEEP_ROWS", (160,)), "no W guard rows": ("W_GUARD_ROWS", ()),
            "spread 50 %": ("SPREAD_MAX", 0.5), "chain unchecked": ("PLAN_CHAIN", (8, 4)),
            "nu row ignored": ("keep_wu", lambda d: all(us(d, str(r), "WU") - us(d, str(r), "A") <= WU_KEEP_US
                                                        for r in WU_KEEP_ROWS)),
            "pred unchecked": ("check_pred", lambda d: []), "health unchecked": ("check_health", lambda d: []),
            "engagement unchecked": ("check_engagement", lambda d: []),
            "barrier unchecked": ("barrier_ok", lambda s: True),
        }
        killed = 0
        for name, (key, val) in muts.items():
            real = g[key]
            g[key] = val
            try:
                caught = any(not ((verdict(copy.deepcopy(d))[0].startswith(w) if w == "VERDICT VOID"
                                   else verdict(copy.deepcopy(d))[0] == w)) for d, w in _scenarios().values())
            except Exception:  # noqa: BLE001
                caught = True
            finally:
                g[key] = real
            killed += caught
            print(f"Q38_RPLE_READ_SELFTEST MUTANT {name}: {'killed' if caught else 'SURVIVED'}")
        print(f"Q38_RPLE_READ_SELFTEST MUTANTS {killed}/{len(muts)} killed")
        bad += killed != len(muts)
    print("Q38_RPLE_READ_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def main(argv) -> int:
    if argv and argv[0] == "--selftest":
        return selftest("--mutants" in argv)
    path = argv[0] if argv else "."
    if os.path.isdir(path):
        path = os.path.join(path, "rple.json")
    if not os.path.exists(path):
        print(f"VERDICT VOID (no JSON at {path})")
        return 2
    d = json.load(open(path))
    for ln in describe(d):
        print(ln)
    v, rc = verdict(d)
    print(v)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

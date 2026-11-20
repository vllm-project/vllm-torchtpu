# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS microbench reader (2026-11-20). Reads the bench JSON (tests/q38_hchigh/tpu_bench_hcwide.py) and applies the
PRE-STATED rules of reports/prereg/q38_hchigh_mbench_prereg (locked before any bench data). Structure copied from DECODE-L3's
HCBIND reader (q38_decl3/hcbind_read.py, credit).

  Per-call us per arm from the interleaved phase; base and base2 are S512 timed alone before and after.
  VOID (any one -> VERDICT VOID):
    SPREAD      at any planned row, |base - base2| / min(base, base2) > 5 %
    ENGAGEMENT  any planned (row, arm) whose compiled long program does not engage its kernel (engaged is not True)
    COMPILE     any planned (row, arm) missing or not compiled, or base / base2 missing at any row
    harness     PLAN (rows 160,192,240,288,384,448,496; arms S512,S256,W1,W2,WD,W1C,W2C,CMP; chain 24/4), HEALTH (every row ok),
                BARRIER (exactly one --xla_tpu_use_enhanced_launch_barrier=false on the final and at-client libtpu strings)
  Decisions (us per call, interleaved phase; the bars are the step KEEP bars 0.4 ms at 448 rows and 0.2 ms at 192 rows / 97 calls):
    W1    GO     iff S512 - W1 >= 4.1 at 448 AND >= 2.1 at 192 AND n1_bitwise at every row
          GO-NB  iff the speed holds but some row is not bitwise (back to DSTEP-LEAD as NUMERICS-CHANGING)
          NO-GO  otherwise
    W2    GO     iff S512 - W2 >= 4.1 at 448 AND >= 2.1 at 192 AND the locked N2 device bound holds at every row
          N2-MISS iff the bound fails at any row (does not ship; the lead takes it to the user), whatever the speed
          NO-GO  otherwise
    S256  GO     iff S512 - S256 >= 4.1 at 448 AND S256 - S512 <= |base - base2| at 288 AND at 384 (env-only lever)
          NO-GO  otherwise
  Descriptive only: WD / W1C / W2C (the wide kernel's copies alone / compute alone), CMP (<= 256 rows), S256 - S512 at
  160 / 192 / 240 (the same program: an in-run A/A), the split halves, the per-row deltas in ms per step (x 97).
Exit 0 for a VERDICT line that is not VOID, 1 for VERDICT VOID, 2 when there is no JSON.
"""
from __future__ import annotations

import json
import os
import sys

PLAN_ROWS = (160, 192, 240, 288, 384, 448, 496)
PLAN_ARMS = ("S512", "S256", "W1", "W2", "WD", "W1C", "W2C", "CMP")
PLAN_CHAIN = (24, 4)
CMP_MAX = 256
SPREAD_MAX = 0.05
BAR_448, BAR_192 = 4.1, 2.1
CALLS = 97
LB = "--xla_tpu_use_enhanced_launch_barrier"


def planned(arm, rows):
    return arm != "CMP" or rows <= CMP_MAX


def barrier_ok(s) -> bool:
    return [t for t in (s or "").split() if t.split("=")[0] == LB] == [LB + "=false"]


def us(case: dict, name: str):
    v = (case.get(name) or {}).get("us_per_call")
    return float(v) if isinstance(v, (int, float)) else None


def read(d: dict) -> tuple:
    """-> (verdict line, detail lines)."""
    void, lines = [], []
    p, host = d.get("plan") or {}, d.get("host") or {}
    if tuple(p.get("rows") or ()) != PLAN_ROWS:
        void.append(f"PLAN rows {p.get('rows')}")
    if tuple(p.get("arms") or ()) != PLAN_ARMS:
        void.append(f"PLAN arms {p.get('arms')}")
    if tuple(host.get("chain") or ()) != PLAN_CHAIN:
        void.append(f"PLAN chain {host.get('chain')}")
    if not (barrier_ok(host.get("libtpu_final")) and barrier_ok(host.get("libtpu_at_client"))):
        void.append("BARRIER")
    spread = {}
    for r in PLAN_ROWS:
        h = (d.get("health") or {}).get(str(r)) or {}
        if not h.get("ok"):
            void.append(f"HEALTH {r}")
        c = (d.get("cases") or {}).get(str(r)) or {}
        e = (d.get("engagement") or {}).get(str(r)) or {}
        for arm in PLAN_ARMS:
            if not planned(arm, r):
                continue
            if us(c, arm) is None or not (e.get(arm) or {}).get("compiled"):
                void.append(f"COMPILE {arm}@{r}")
            elif (e.get(arm) or {}).get("engaged") is not True:
                void.append(f"ENGAGEMENT {arm}@{r}")
        b, b2 = us(c, "base"), us(c, "base2")
        if b is None or b2 is None:
            void.append(f"COMPILE base@{r}")
        else:
            spread[r] = abs(b - b2)
            if spread[r] / min(b, b2) > SPREAD_MAX:
                void.append(f"SPREAD {r} {spread[r] / min(b, b2):.3f}")
    if void:
        return "VERDICT VOID (" + "; ".join(void[:12]) + ")", lines
    cases = d["cases"]
    delta = {arm: {r: us(cases[str(r)], "S512") - us(cases[str(r)], arm) for r in PLAN_ROWS if planned(arm, r)}
             for arm in PLAN_ARMS if arm != "S512"}
    for r in PLAN_ROWS:
        c = cases[str(r)]
        lines.append(f"ROW {r}: " + " ".join(f"{a} {us(c, a):.3f}" for a in ("base",) + PLAN_ARMS + ("base2",)
                                              if (a in ("base", "base2") or planned(a, r))) + " us/call")
        lines.append(f"ROW {r} S512 - X (us/call | ms/step x {CALLS}): " + " ".join(
            f"{a} {delta[a][r]:+.3f} | {delta[a][r] * CALLS / 1e3:+.3f}" for a in ("W1", "W2", "S256") + (("CMP",) if planned("CMP", r) else ())))
    h = d["health"]
    n1_bit = {r: all(h[str(r)].get("n1_bitwise") or [False]) for r in PLAN_ROWS}
    n2_ok = {r: bool((h[str(r)].get("n2") or {}).get("ok")) for r in PLAN_ROWS}
    lines.append("N1 bitwise per row: " + " ".join(f"{r}:{'Y' if n1_bit[r] else 'N'}" for r in PLAN_ROWS))
    lines.append("N2 device bound per row: " + " ".join(f"{r}:{'Y' if n2_ok[r] else 'N'}" for r in PLAN_ROWS))
    speed = {a: delta[a][448] >= BAR_448 and delta[a][192] >= BAR_192 for a in ("W1", "W2")}
    w1 = ("GO" if all(n1_bit.values()) else "GO-NB") if speed["W1"] else "NO-GO"
    w2 = "N2-MISS" if not all(n2_ok.values()) else ("GO" if speed["W2"] else "NO-GO")
    s256 = "GO" if (delta["S256"][448] >= BAR_448 and -delta["S256"][288] <= spread[288]
                    and -delta["S256"][384] <= spread[384]) else "NO-GO"
    return f"VERDICT W1 {w1} | W2 {w2} | S256 {s256}", lines


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest() | (mutants() if "--mutants" in argv else 0)
    path = argv[0]
    f = os.path.join(path, "hcw.json") if os.path.isdir(path) else path
    if not os.path.exists(f):
        print("NO JSON", f)
        return 2
    verdict, lines = read(json.load(open(f)))
    for ln in lines:
        print(ln)
    print(verdict)
    return 1 if verdict.startswith("VERDICT VOID") else 0


# ---------------------------------------------------------------- selftest
def _fixture(s512=40.0, w1=20.0, w2=18.0, s256=35.0, bit=True, n2=True, spread=0.2, eng=True):
    lb = "--a=1 " + LB + "=false"
    d = {"plan": {"rows": list(PLAN_ROWS), "arms": list(PLAN_ARMS)},
         "host": {"chain": [24, 4], "libtpu_final": lb, "libtpu_at_client": lb}, "cases": {}, "health": {}, "engagement": {}}
    for r in PLAN_ROWS:
        k = r / 448.0
        c = {"base": {"us_per_call": s512 * k}, "base2": {"us_per_call": s512 * k + spread}}
        vals = {"S512": s512 * k, "S256": (s256 if r > 256 else s512) * k, "W1": w1 * k, "W2": w2 * k, "WD": 10.0, "W1C": 12.0,
                "W2C": 11.0, "CMP": 30.0 * k}
        for a in PLAN_ARMS:
            if planned(a, r):
                c[a] = {"us_per_call": vals[a]}
        d["cases"][str(r)] = c
        d["health"][str(r)] = {"ok": True, "n1_bitwise": [bit, True, True], "n2": {"ok": n2}}
        d["engagement"][str(r)] = {a: {"compiled": True, "engaged": eng} for a in PLAN_ARMS if planned(a, r)}
    return d


def selftest() -> int:
    cases = [
        ("all GO", _fixture(), "VERDICT W1 GO | W2 GO | S256 GO"),
        ("W1 not bitwise -> GO-NB", _fixture(bit=False), "VERDICT W1 GO-NB | W2 GO | S256 GO"),
        ("N2 bound miss -> N2-MISS even when fast", _fixture(n2=False), "VERDICT W1 GO | W2 N2-MISS | S256 GO"),
        ("W1 / W2 too slow -> NO-GO", _fixture(w1=37.0, w2=37.0), "VERDICT W1 NO-GO | W2 NO-GO | S256 GO"),
        ("S256 not faster at 448 -> NO-GO", _fixture(s256=40.5), "VERDICT W1 GO | W2 GO | S256 NO-GO"),
        ("192 bar binds (448 passes) -> NO-GO", _fixture(w1=35.5, w2=35.5), "VERDICT W1 NO-GO | W2 NO-GO | S256 GO"),
        ("spread 6 % -> VOID", _fixture(spread=2.4), "VERDICT VOID"),
        ("not engaged -> VOID", _fixture(eng=False), "VERDICT VOID"),
    ]
    bad = 0
    for label, d, want in cases:
        got = read(d)[0]
        ok = got.startswith(want)
        print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} {label}: {got[:120]}")
        bad += not ok
    d = _fixture()
    d["cases"]["288"]["S256"]["us_per_call"] = d["cases"]["288"]["S512"]["us_per_call"] + 1.0
    ok = read(d)[0].startswith("VERDICT W1 GO | W2 GO | S256 NO-GO")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} S256 fast at 448 but slower than the spread at 288 -> NO-GO")
    bad += not ok
    d = _fixture()
    del d["cases"]["448"]["W2"]
    ok = read(d)[0].startswith("VERDICT VOID")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} a missing case -> VOID")
    bad += not ok
    d = _fixture()
    d["host"]["libtpu_final"] = "--a=1"
    ok = read(d)[0].startswith("VERDICT VOID")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} no launch-barrier token -> VOID")
    bad += not ok
    cfg = PLAN_ROWS == (160, 192, 240, 288, 384, 448, 496) and (BAR_448, BAR_192, SPREAD_MAX, CALLS) == (4.1, 2.1, 0.05, 97) \
        and planned("CMP", 256) and not planned("CMP", 288) and planned("W1", 496)
    print(f"HCWREAD_SELFTEST {'ok ' if cfg else 'BAD'} rules: bars {BAR_448} / {BAR_192} us/call, spread {SPREAD_MAX}")
    bad += not cfg
    print("HCWREAD_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants() -> int:
    g = globals()
    orig = {k: g[k] for k in ("BAR_448", "BAR_192", "SPREAD_MAX", "barrier_ok", "planned")}
    muts = {
        "448 bar halved": ("BAR_448", 2.05),
        "192 bar dropped": ("BAR_192", -1e9),
        "spread rule off": ("SPREAD_MAX", 1e9),
        "barrier check blind": ("barrier_ok", lambda s: True),
        "CMP planned at every row": ("planned", lambda arm, rows: True),
    }
    bad = 0
    import contextlib
    import io
    for name, (sym, val) in muts.items():
        g[sym] = val
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = selftest()
        finally:
            for k, v in orig.items():
                g[k] = v
        print(f"HCWREAD_MUTANT {'killed  ' if rc else 'SURVIVED'} {name}")
        bad += not rc
    print("HCWREAD_MUTANTS", "PASS" if not bad else f"FAIL ({bad} survived)", f"({len(muts)} mutants)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

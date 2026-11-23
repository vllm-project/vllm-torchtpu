# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS microbench reader (2026-11-20). Reads the bench JSONs (tests/q38_hchigh/tpu_bench_hcwide.py) of the two pods
(lo: rows 160 / 192 / 240 / 288; hi: rows 384 / 448 / 496; run in parallel, DSTEP-LEAD 2026-11-20) and applies the PRE-STATED rules
of reports/prereg/q38_hchigh_mbench_prereg (locked before any bench data). Structure copied from DECODE-L3's HCBIND reader
(q38_decl3/hcbind_read.py, credit).

  Per-call us per arm from each pod's interleaved phase; base and base2 are S512 timed alone before and after, per row.
  Pod VOID (any one -> that pod's VERDICT VOID; the combined verdict is VOID if either pod is):
    SPREAD      at any row of the pod, |base - base2| / min(base, base2) > 5 %
    ENGAGEMENT  any planned (row, arm) whose compiled long program does not engage its kernel (engaged is not True)
    COMPILE     any planned (row, arm) missing or not compiled, or base / base2 missing at any row
    harness     PLAN (the pod's rows; arms S512,S256,W1,W2,WD,W1C,W2C + CMP on lo; chain 24/4), HEALTH (every row ok),
                BARRIER (exactly one --xla_tpu_use_enhanced_launch_barrier=false on the final and at-client libtpu strings)
  Decisions on the merged pods (us per call; bars = the step KEEP bars 0.4 ms at 448 rows and 0.2 ms at 192 rows / 97 calls):
    W1    GO     iff S512 - W1 >= 4.1 at 448 AND >= 2.1 at 192 AND n1_bitwise at every row
          GO-NB  iff the speed holds but some row is not bitwise (back to DSTEP-LEAD as NUMERICS-CHANGING)
          NO-GO  otherwise
    S256  GO     iff S512 - S256 >= 4.1 at 448 AND S256 - S512 <= |base - base2| at 288 AND at 384 (env-only lever)
          NO-GO  otherwise
    W2    timed only; its numerics class is UNRESOLVED (CPU proof cpu1 P4 harness FAIL, with DSTEP-LEAD / the lead):
          N2-MISS       iff the locked N2 device bound fails at any row (does not ship; the lead takes it to the user)
          FAST-UNRESOLVED iff S512 - W2 >= 4.1 at 448 AND >= 2.1 at 192;  SLOW-UNRESOLVED otherwise
  Descriptive only: WD / W1C / W2C, CMP (<= 256 rows), S256 - S512 at 160 / 192 / 240 (the same program: an in-run A/A), the split
  halves, the per-row deltas in ms per step (x 97).
usage: hcw_read.py --pod lo|hi <outputs dir or hcw.json>      (in-pod: 'VERDICT POD <pod> VALID' or 'VERDICT VOID (...)')
       hcw_read.py --combine <lo dir|json> <hi dir|json>        ('VERDICT W1 .. | S256 .. | W2 ..' or 'VERDICT VOID (...)')
       hcw_read.py --selftest [--mutants]
Exit 0 for a verdict that is not VOID, 1 for VOID, 2 when a JSON is missing.
"""
from __future__ import annotations

import json
import os
import sys

POD_ROWS = {"lo": (160, 192, 240, 288), "hi": (384, 448, 496)}
PLAN_ROWS = POD_ROWS["lo"] + POD_ROWS["hi"]
PLAN_ARMS = ("S512", "S256", "W1", "W2", "WD", "W1C", "W2C", "CMP")
PLAN_CHAIN = (24, 4)
CMP_MAX = 256
SPREAD_MAX = 0.05
BAR_448, BAR_192 = 4.1, 2.1
CALLS = 97
LB = "--xla_tpu_use_enhanced_launch_barrier"


def planned(arm, rows):
    return arm != "CMP" or rows <= CMP_MAX


def pod_arms(pod):
    return tuple(a for a in PLAN_ARMS if any(planned(a, r) for r in POD_ROWS[pod]))


def barrier_ok(s) -> bool:
    return [t for t in (s or "").split() if t.split("=")[0] == LB] == [LB + "=false"]


def us(case: dict, name: str):
    v = (case.get(name) or {}).get("us_per_call")
    return float(v) if isinstance(v, (int, float)) else None


def check_pod(d: dict, pod: str) -> tuple:
    """-> (void reasons, spread per row)."""
    void, spread = [], {}
    p, host = d.get("plan") or {}, d.get("host") or {}
    if tuple(p.get("rows") or ()) != POD_ROWS[pod]:
        void.append(f"PLAN {pod} rows {p.get('rows')}")
    if tuple(p.get("arms") or ()) != pod_arms(pod):
        void.append(f"PLAN {pod} arms {p.get('arms')}")
    if tuple(host.get("chain") or ()) != PLAN_CHAIN:
        void.append(f"PLAN {pod} chain {host.get('chain')}")
    if not (barrier_ok(host.get("libtpu_final")) and barrier_ok(host.get("libtpu_at_client"))):
        void.append(f"BARRIER {pod}")
    for r in POD_ROWS[pod]:
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
    return void, spread


def row_lines(d: dict, rows) -> list:
    out = []
    for r in rows:
        c = d["cases"][str(r)]
        out.append(f"ROW {r}: " + " ".join(f"{a} {us(c, a):.3f}" for a in ("base",) + PLAN_ARMS + ("base2",)
                                            if (a in ("base", "base2") or planned(a, r))) + " us/call")
        out.append(f"ROW {r} S512 - X (us/call | ms/step x {CALLS}): " + " ".join(
            f"{a} {us(c, 'S512') - us(c, a):+.3f} | {(us(c, 'S512') - us(c, a)) * CALLS / 1e3:+.3f}"
            for a in ("W1", "W2", "S256") + (("CMP",) if planned("CMP", r) else ())))
        h = d["health"][str(r)]
        out.append(f"ROW {r} numerics: n1_bitwise {h.get('n1_bitwise')} n2 {json.dumps(h.get('n2'))[:300]} s256_bitwise "
                   f"{h.get('s256_bitwise')} cmp_tol {(h.get('cmp_tol') or {}).get('ok')}")
    return out


def read_pod(d: dict, pod: str) -> tuple:
    void, _ = check_pod(d, pod)
    if void:
        return "VERDICT VOID (" + "; ".join(void[:12]) + ")", []
    return f"VERDICT POD {pod} VALID (rows {','.join(map(str, POD_ROWS[pod]))})", row_lines(d, POD_ROWS[pod])


def read_combined(dlo: dict, dhi: dict) -> tuple:
    vlo, slo = check_pod(dlo, "lo")
    vhi, shi = check_pod(dhi, "hi")
    if vlo or vhi:
        return "VERDICT VOID (" + "; ".join((vlo + vhi)[:12]) + ")", []
    d = {k: {**(dlo.get(k) or {}), **(dhi.get(k) or {})} for k in ("cases", "health", "engagement")}
    spread = {**slo, **shi}
    cases, h = d["cases"], d["health"]
    delta = {a: {r: us(cases[str(r)], "S512") - us(cases[str(r)], a) for r in PLAN_ROWS if planned(a, r)}
             for a in PLAN_ARMS if a != "S512"}
    lines = row_lines(d, PLAN_ROWS)
    n1_bit = {r: all(h[str(r)].get("n1_bitwise") or [False]) for r in PLAN_ROWS}
    n2_ok = {r: bool((h[str(r)].get("n2") or {}).get("ok")) for r in PLAN_ROWS}
    lines.append("N1 bitwise per row: " + " ".join(f"{r}:{'Y' if n1_bit[r] else 'N'}" for r in PLAN_ROWS))
    lines.append("N2 device bound per row: " + " ".join(f"{r}:{'Y' if n2_ok[r] else 'N'}" for r in PLAN_ROWS))
    speed = {a: delta[a][448] >= BAR_448 and delta[a][192] >= BAR_192 for a in ("W1", "W2")}
    w1 = ("GO" if all(n1_bit.values()) else "GO-NB") if speed["W1"] else "NO-GO"
    s256 = "GO" if (delta["S256"][448] >= BAR_448 and -delta["S256"][288] <= spread[288]
                    and -delta["S256"][384] <= spread[384]) else "NO-GO"
    w2 = "N2-MISS" if not all(n2_ok.values()) else ("FAST-UNRESOLVED" if speed["W2"] else "SLOW-UNRESOLVED")
    return f"VERDICT W1 {w1} | S256 {s256} | W2 {w2}", lines


def _load(path):
    f = os.path.join(path, "hcw.json") if os.path.isdir(path) else path
    return json.load(open(f)) if os.path.exists(f) else None


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest() | (mutants() if "--mutants" in argv else 0)
    if argv[:1] == ["--pod"] and len(argv) == 3 and argv[1] in POD_ROWS:
        d = _load(argv[2])
        if d is None:
            print("NO JSON", argv[2])
            return 2
        verdict, lines = read_pod(d, argv[1])
    elif argv[:1] == ["--combine"] and len(argv) == 3:
        dlo, dhi = _load(argv[1]), _load(argv[2])
        if dlo is None or dhi is None:
            print("NO JSON", argv[1:])
            return 2
        verdict, lines = read_combined(dlo, dhi)
    else:
        print(__doc__)
        return 2
    for ln in lines:
        print(ln)
    print(verdict)
    return 1 if verdict.startswith("VERDICT VOID") else 0


# ---------------------------------------------------------------- selftest
def _fixture(pod, s512=40.0, w1=20.0, w2=18.0, s256=35.0, bit=True, n2=True, spread=0.2, eng=True):
    lb = "--a=1 " + LB + "=false"
    d = {"plan": {"rows": list(POD_ROWS[pod]), "arms": list(pod_arms(pod))},
         "host": {"chain": [24, 4], "libtpu_final": lb, "libtpu_at_client": lb}, "cases": {}, "health": {}, "engagement": {}}
    for r in POD_ROWS[pod]:
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


def _both(**kw):
    return _fixture("lo", **kw), _fixture("hi", **kw)


def selftest() -> int:
    cases = [
        ("all GO, W2 fast", _both(), "VERDICT W1 GO | S256 GO | W2 FAST-UNRESOLVED"),
        ("W1 not bitwise -> GO-NB", _both(bit=False), "VERDICT W1 GO-NB | S256 GO | W2 FAST-UNRESOLVED"),
        ("N2 bound miss -> N2-MISS even when fast", _both(n2=False), "VERDICT W1 GO | S256 GO | W2 N2-MISS"),
        ("W1 / W2 too slow -> NO-GO / SLOW", _both(w1=37.0, w2=37.0), "VERDICT W1 NO-GO | S256 GO | W2 SLOW-UNRESOLVED"),
        ("S256 not faster at 448 -> NO-GO", _both(s256=40.5), "VERDICT W1 GO | S256 NO-GO | W2 FAST-UNRESOLVED"),
        ("192 bar binds (448 passes) -> NO-GO", _both(w1=35.5, w2=35.5), "VERDICT W1 NO-GO | S256 GO | W2 SLOW-UNRESOLVED"),
        ("spread 6 % -> VOID", _both(spread=2.4), "VERDICT VOID"),
        ("not engaged -> VOID", _both(eng=False), "VERDICT VOID"),
    ]
    bad = 0
    for label, (lo, hi), want in cases:
        got = read_combined(lo, hi)[0]
        ok = got.startswith(want)
        print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} {label}: {got[:120]}")
        bad += not ok
    lo, hi = _both()
    lo["cases"]["288"]["S256"]["us_per_call"] = lo["cases"]["288"]["S512"]["us_per_call"] + 1.0
    ok = read_combined(lo, hi)[0].startswith("VERDICT W1 GO | S256 NO-GO")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} S256 fast at 448 but slower than the spread at 288 -> NO-GO")
    bad += not ok
    lo, hi = _both()
    del hi["cases"]["448"]["W2"]
    ok = read_combined(lo, hi)[0].startswith("VERDICT VOID") and read_pod(hi, "hi")[0].startswith("VERDICT VOID")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} a missing case -> VOID (combined and in-pod)")
    bad += not ok
    lo, hi = _both()
    lo["host"]["libtpu_final"] = "--a=1"
    ok = read_combined(lo, hi)[0].startswith("VERDICT VOID") and read_pod(lo, "lo")[0].startswith("VERDICT VOID")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} no launch-barrier token -> VOID")
    bad += not ok
    lo, hi = _both()
    ok = read_pod(lo, "lo")[0].startswith("VERDICT POD lo VALID") and read_pod(hi, "hi")[0].startswith("VERDICT POD hi VALID") \
        and read_pod(lo, "hi")[0].startswith("VERDICT VOID") and read_combined(hi, lo)[0].startswith("VERDICT VOID")
    print(f"HCWREAD_SELFTEST {'ok ' if ok else 'BAD'} in-pod VALID; a pod read against the other pod's plan -> VOID")
    bad += not ok
    cfg = PLAN_ROWS == (160, 192, 240, 288, 384, 448, 496) and (BAR_448, BAR_192, SPREAD_MAX, CALLS) == (4.1, 2.1, 0.05, 97) \
        and planned("CMP", 256) and not planned("CMP", 288) and planned("W1", 496) and "CMP" not in pod_arms("hi") \
        and pod_arms("lo") == PLAN_ARMS
    print(f"HCWREAD_SELFTEST {'ok ' if cfg else 'BAD'} rules: bars {BAR_448} / {BAR_192} us/call, spread {SPREAD_MAX}, pods {POD_ROWS}")
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

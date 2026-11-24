#!/usr/bin/env python3
"""q38 L3 D60 reader (stdlib only; run with python3 -I): the verdict of the D60 kernel script-bench car.

  python3 -I -B d60_read.py <dir with d60_tc.json>      -> verdict lines; last line VERDICT ...
  python3 -I -B d60_read.py --selftest [--mutants]

Input: tests/q38_l3_d60/tpu_bench_moe_compact_pipe.py's JSON, cases 160 / 192 / 208 / 224 / 256 (near-uniform
routing) and 224s / 256s (skewed), arms dense / cr32 / cr32p. All times are medians in us per layer. Rules, in order
(the prereg states the same):
 1. VOID (rc 2): JSON missing or unparseable, ok false, an interpret or non-TPU host, a case or arm missing, or fewer
    than MIN_SAMPLES samples for an arm.
 2. INVALID (rc 2): an arm not engaged (kernel name per arm and bucket), or the launch-barrier tokens at client
    creation are not exactly one --xla_tpu_use_enhanced_launch_barrier=false.
 3. FAIL-EXACT (rc 0): any arm's output differs from dense's (1 call or chain) at any case, a non-finite output, a
    positive control (dense, cr32p) that moved nothing, or under 90 % nonzero output elements.
 4. gd(b) = dense - cr32p, g5(b) = cr32 - cr32p; noise n(b) = the largest split-half figure of the three arms;
    bar(b) = max(1.0, 2 n(b)).
 5. NO VERDICT (rc 1): bar(224) > 5 or bar(256) > 5, or validity V1 fails: dense(256) outside [143, 239] us per
    layer (MDC1 j-0cd4e9dc's bench dense 191 +- 25 %, the same harness and geometry).
 6. MIN* = the smallest b in {160, 192, 224} with gd(b') >= bar(b') at b and at every larger bucket b' up to 224
    (208 included); none: MIN* = 256.
 7. KEEP iff [ g5(256) >= max(3.0, bar(256)) or (MIN* <= 224 and gd(224) >= max(5.0, bar(224))) ] and no harm:
    g5(256) >= -bar(256), g5(256s) >= -bar(256), and, if MIN* <= 224, gd(224s) >= -bar(224).
    KEEP names the flag value FUSED_MOE_DECODE_COMPACT_PIPE = MIN* (256 when only the 256 clause holds).
 8. Otherwise NOT KEEP (rc 0).
Descriptive: the served-weighted c128 step delta (ms) = 49 x sum_b w_b x g_b / 1000 with L5 3.1's T85 c128 shares
(160 13.2 %, 192 3.1 %, 224 4.1 %, 256 49.6 %; 208 is not a served bucket), g_b = gd(b) for MIN* <= b <= 224, g_256 =
g5(256) (FINAL runs cr32 there), else 0. Bench numbers, not an engine measurement.
"""
from __future__ import annotations

import json
import os
import sys

UNIFORM = (160, 192, 208, 224, 256)
SKEWED = ("224s", "256s")
ARMS = ("dense", "cr32", "cr32p")
MIN_SAMPLES = 15
LB_OFF = "--xla_tpu_use_enhanced_launch_barrier=false"
LB = "--xla_tpu_use_enhanced_launch_barrier"
W_C128 = {160: 0.132, 192: 0.031, 224: 0.041, 256: 0.496}
MOE_CALLS = 49
RULES = {"bar_floor": 1.0, "bar_mult": 2.0, "noisy": 5.0, "keep256": 3.0, "keep224": 5.0, "v1_lo": 143.0, "v1_hi": 239.0}


def bar(case, r=RULES):
    sh = [case[a].get("split_half") for a in ARMS]
    if any(v is None for v in sh):
        return float("inf")
    return max(r["bar_floor"], r["bar_mult"] * max(sh))


def med(case, arm):
    return float(case[arm]["us_per_layer_median"])


def void_reason(res):
    if not isinstance(res, dict) or not res.get("ok"):
        return "result not ok"
    host = res.get("host", {})
    if host.get("interpret") or "TPU" not in str(host.get("kind", "")).upper():
        return f"not a TPU run (interpret {host.get('interpret')}, kind {host.get('kind')})"
    cases = res.get("cases", {})
    for c in [str(b) for b in UNIFORM] + list(SKEWED):
        if c not in cases:
            return f"case {c} missing"
        for a in ARMS:
            if a not in cases[c]:
                return f"case {c} arm {a} missing"
            if len(cases[c][a].get("samples", [])) < MIN_SAMPLES:
                return f"case {c} arm {a}: {len(cases[c][a].get('samples', []))} samples < {MIN_SAMPLES}"
    return None


def invalid_reason(res):
    toks = [t for t in str(res["host"].get("libtpu_at_client", "")).split() if t.split("=")[0] == LB]
    if toks != [LB_OFF]:
        return f"launch-barrier tokens at client {toks}"
    for c, case in res["cases"].items():
        for a in ARMS:
            if case[a].get("engaged") is not True:
                return f"case {c} arm {a} not engaged (kernel {case[a].get('kernel_names')})"
    return None


def exact_reason(res):
    for c, case in res["cases"].items():
        for a in ARMS:
            v = case[a]
            if v.get("mismatch_vs_dense_1call") != 0 or v.get("mismatch_vs_dense_chain") != 0:
                return (f"case {c} arm {a}: mismatches vs dense 1-call {v.get('mismatch_vs_dense_1call')} chain "
                        f"{v.get('mismatch_vs_dense_chain')}")
            if v.get("finite") is not True:
                return f"case {c} arm {a}: non-finite output"
            if v.get("nonzero_out", 0) < 0.9 * v.get("elements", 1):
                return f"case {c} arm {a}: nonzero {v.get('nonzero_out')} of {v.get('elements')}"
        for a in ("dense", "cr32p"):
            if not case[a].get("posctl_mismatch"):
                return f"case {c} arm {a}: positive control moved nothing"
    return None


def min_star(gd, bars):
    cands = [b for b in (160, 192, 224)]
    for b in cands:
        span = [x for x in UNIFORM if b <= x <= 224]
        if all(gd[x] >= bars[x] for x in span):
            return b
    return 256


def decide(res, r=RULES):
    """-> (verdict, rc, lines)"""
    lines = []
    why = void_reason(res)
    if why:
        return "VOID", 2, [f"VOID: {why}"]
    why = invalid_reason(res)
    if why:
        return "INVALID", 2, [f"INVALID: {why}"]
    cs = res["cases"]
    for c in [str(b) for b in UNIFORM] + list(SKEWED):
        lines.append(f"case {c}: " + " ".join(f"{a} {med(cs[c], a):.2f}" for a in ARMS)
                     + f" | gd {med(cs[c], 'dense') - med(cs[c], 'cr32p'):+.2f} g5 {med(cs[c], 'cr32') - med(cs[c], 'cr32p'):+.2f}"
                     + f" bar {bar(cs[c], r):.2f} routing {cs[c].get('stats')}")
    why = exact_reason(res)
    if why:
        return "FAIL-EXACT", 0, lines + [f"FAIL-EXACT: {why}"]
    gd = {b: med(cs[str(b)], "dense") - med(cs[str(b)], "cr32p") for b in UNIFORM}
    g5 = {b: med(cs[str(b)], "cr32") - med(cs[str(b)], "cr32p") for b in UNIFORM}
    bars = {b: bar(cs[str(b)], r) for b in UNIFORM}
    if bars[224] > r["noisy"] or bars[256] > r["noisy"]:
        return "NO VERDICT", 1, lines + [f"NO VERDICT: noise bar 224 {bars[224]:.2f} / 256 {bars[256]:.2f} > {r['noisy']}"]
    d256 = med(cs["256"], "dense")
    if not r["v1_lo"] <= d256 <= r["v1_hi"]:
        return "NO VERDICT", 1, lines + [f"NO VERDICT: V1 dense(256) {d256:.2f} outside [{r['v1_lo']}, {r['v1_hi']}]"]
    ms = min_star(gd, bars)
    g5s = med(cs["256s"], "cr32") - med(cs["256s"], "cr32p")
    gd224s = med(cs["224s"], "dense") - med(cs["224s"], "cr32p")
    c256 = g5[256] >= max(r["keep256"], bars[256])
    c224 = ms <= 224 and gd[224] >= max(r["keep224"], bars[224])
    harm = []
    if g5[256] < -bars[256]:
        harm.append(f"g5(256) {g5[256]:+.2f} < -bar {bars[256]:.2f}")
    if g5s < -bars[256]:
        harm.append(f"g5(256s) {g5s:+.2f} < -bar {bars[256]:.2f}")
    if ms <= 224 and gd224s < -bars[224]:
        harm.append(f"gd(224s) {gd224s:+.2f} < -bar {bars[224]:.2f}")
    keep = (c256 or c224) and not harm
    flag = ms if c224 else 256
    g_served = {b: (gd[b] if ms <= b <= 224 else 0.0) for b in (160, 192, 224)}
    g_served[256] = g5[256]
    served = MOE_CALLS * sum(W_C128[b] * g_served[b] for b in W_C128) / 1000.0
    lines.append(f"MIN* {ms}; clause256 {c256} (g5 {g5[256]:+.2f} vs {max(r['keep256'], bars[256]):.2f}); clause224 "
                 f"{c224} (gd {gd[224]:+.2f} vs {max(r['keep224'], bars[224]):.2f}); harm {harm or 'none'}")
    lines.append(f"DESCRIPTIVE served-weighted c128 step delta {-served:+.3f} ms (bench; 49 MoE calls; T85 c128 shares)")
    if keep:
        return "KEEP", 0, lines + [f"KEEP FUSED_MOE_DECODE_COMPACT_PIPE={flag} FUSED_MOE_DECODE_COMPACT_PIPE_ROWS=32"]
    return "NOT KEEP", 0, lines


def read_dir(d):
    p = os.path.join(d, "d60_tc.json")
    try:
        res = json.load(open(p))
    except Exception as exc:  # noqa: BLE001 - unreadable is VOID
        return "VOID", 2, [f"VOID: cannot read {p}: {type(exc).__name__}: {exc}"]
    return decide(res)


# ---------------------------------------------------------------- selftest + mutants
def _fixture(dense=None, cr32=None, cr32p=None, sh=0.2, **kw):
    dense = dense or {160: 120.0, 192: 130.0, 208: 140.0, 224: 150.0, 256: 190.0, "224s": 145.0, "256s": 185.0}
    cr32 = cr32 or {160: 125.0, 192: 132.0, 208: 135.0, 224: 136.0, 256: 162.0, "224s": 140.0, "256s": 165.0}
    cr32p = cr32p or {160: 121.0, 192: 125.0, 208: 128.0, 224: 129.0, 256: 155.0, "224s": 133.0, "256s": 158.0}
    cases = {}
    for c in [str(b) for b in UNIFORM] + list(SKEWED):
        k = int(c) if c.isdigit() else c
        tok = int(c.rstrip("s"))
        case = {"stats": {}}
        for a, t in (("dense", dense), ("cr32", cr32), ("cr32p", cr32p)):
            name = f"fused_moe_decode_fold-e_64-m_{tok}-h_2560" + {"dense": "", "cr32": "-cr_32", "cr32p": "-cr_32p"}[a]
            case[a] = {"us_per_layer_median": t[k], "split_half": sh, "samples": [t[k]] * 20,
                       "mismatch_vs_dense_1call": 0, "mismatch_vs_dense_chain": 0, "finite": True,
                       "nonzero_out": 100, "elements": 100, "engaged": True, "kernel_names": [name],
                       "posctl_mismatch": 5 if a != "cr32" else None}
        cases[c] = case
    res = {"ok": True, "host": {"interpret": False, "kind": "TPU7x", "libtpu_at_client": "--a=1 " + LB_OFF},
           "cases": cases}
    for k, v in kw.items():
        v(res)
    return res


def _set(case, arm, key, val):
    def f(res):
        res["cases"][case][arm][key] = val
    return f


def selftest():
    bad = 0
    T = [
        ("keep both clauses", _fixture(), "KEEP", "COMPACT_PIPE=192"),
        ("keep 256 only", _fixture(dense={160: 120.0, 192: 125.0, 208: 128.0, 224: 129.5, 256: 190.0, "224s": 133.0,
                                          "256s": 185.0}), "KEEP", "COMPACT_PIPE=256"),
        ("keep 224 only (MIN 224)", _fixture(cr32p={160: 121.0, 192: 131.0, 208: 128.0, 224: 129.0, 256: 161.0,
                                                    "224s": 133.0, "256s": 164.0}), "KEEP", "COMPACT_PIPE=224"),
        ("not keep: small gains", _fixture(cr32p={160: 120.5, 192: 129.5, 208: 139.0, 224: 148.0, 256: 160.5,
                                                  "224s": 144.0, "256s": 164.0}), "NOT KEEP", None),
        ("not keep: harm at 256", _fixture(cr32p={160: 121.0, 192: 125.0, 208: 128.0, 224: 129.0, 256: 170.0,
                                                  "224s": 133.0, "256s": 158.0}), "NOT KEEP", None),
        ("not keep: skewed harm", _fixture(cr32p={160: 121.0, 192: 125.0, 208: 128.0, 224: 129.0, 256: 155.0,
                                                  "224s": 160.0, "256s": 158.0}), "NOT KEEP", None),
        ("fail-exact chain", _fixture(x=_set("208", "cr32p", "mismatch_vs_dense_chain", 3)), "FAIL-EXACT", None),
        ("fail-exact 1call cr32", _fixture(x=_set("256", "cr32", "mismatch_vs_dense_1call", 1)), "FAIL-EXACT", None),
        ("fail-exact posctl", _fixture(x=_set("160", "cr32p", "posctl_mismatch", 0)), "FAIL-EXACT", None),
        ("fail-exact nonfinite", _fixture(x=_set("224s", "dense", "finite", False)), "FAIL-EXACT", None),
        ("invalid engagement", _fixture(x=_set("224", "cr32p", "engaged", False)), "INVALID", None),
        ("invalid barrier", _fixture(x=lambda r: r["host"].update(libtpu_at_client=LB + "=true")), "INVALID", None),
        ("void interpret", _fixture(x=lambda r: r["host"].update(interpret=True)), "VOID", None),
        ("void missing case", _fixture(x=lambda r: r["cases"].pop("256s")), "VOID", None),
        ("void few samples", _fixture(x=_set("192", "cr32", "samples", [1.0] * 5)), "VOID", None),
        ("no verdict noisy", _fixture(sh=3.0), "NO VERDICT", None),
        ("no verdict V1", _fixture(x=_set("256", "dense", "us_per_layer_median", 120.0)), "NO VERDICT", None),
        ("min* skips a 192 dip", _fixture(cr32p={160: 110.0, 192: 129.8, 208: 128.0, 224: 129.0, 256: 155.0,
                                                 "224s": 133.0, "256s": 158.0}), "KEEP", "COMPACT_PIPE=224"),
    ]
    for name, fx, want, flag in T:
        v, rc, lines = decide(fx)
        ok = v == want and (flag is None or any(flag in ln for ln in lines))
        want_rc = {"KEEP": 0, "NOT KEEP": 0, "FAIL-EXACT": 0, "NO VERDICT": 1, "VOID": 2, "INVALID": 2}[want]
        ok &= rc == want_rc
        if not ok:
            print(f"D60_READ_SELFTEST MISMATCH {name}: got {v} rc {rc} want {want} {flag or ''}; {lines[-2:]}")
        bad += not ok
    v, rc, lines = decide(_fixture())
    served = [ln for ln in lines if ln.startswith("DESCRIPTIVE")]
    # gd 160..224 at MIN 192: 192 +5, 224 +21; g5(256) +7 -> 49 * (0.031*5 + 0.041*21 + 0.496*7) / 1000
    want_served = -49 * (0.031 * 5.0 + 0.041 * 21.0 + 0.496 * 7.0) / 1000
    if not served or f"{want_served:+.3f}" not in served[0]:
        print(f"D60_READ_SELFTEST MISMATCH served arithmetic: {served} want {want_served:+.3f}")
        bad += 1
    print("D60_READ_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants():
    g = globals()
    orig = {k: g[k] for k in ("bar", "min_star", "exact_reason", "invalid_reason", "void_reason", "decide")}
    muts = {
        "bar_no_floor": ("bar", lambda case, r=RULES: 0.0),
        "min_star_ignores_larger": ("min_star", lambda gd, bars: next((b for b in (160, 192, 224) if gd[b] >= bars[b]), 256)),
        "exact_ignores_chain": ("exact_reason", lambda res: None if all(
            res["cases"][c][a]["mismatch_vs_dense_1call"] == 0 for c in res["cases"] for a in ARMS) else "x"),
        "no_engagement_check": ("invalid_reason", lambda res: None),
        "no_sample_floor": ("void_reason", lambda res: None if res.get("ok") and not res["host"].get("interpret") else "x"),
    }
    killed = 0

    def run_selftest():
        try:
            return selftest()
        except Exception as exc:  # noqa: BLE001 - a selftest that crashes under a mutant did not pass
            print(f"D60_READ_SELFTEST raised {type(exc).__name__}: {exc}")
            return 1

    for name, (fn, repl) in muts.items():
        g[fn] = repl
        try:
            rc = run_selftest()
        finally:
            g.update(orig)
        killed += rc != 0
        print(f"D60_READ_MUTANT {name}: {'killed' if rc else 'SURVIVED'}")
    # rule-constant mutants: no V1 band; KEEP at 256 without the 3 us floor
    RULES["v1_lo"] = 0.0
    try:
        rc = run_selftest()
    finally:
        RULES["v1_lo"] = 143.0
    killed += rc != 0
    print(f"D60_READ_MUTANT no_v1_band: {'killed' if rc else 'SURVIVED'}")
    RULES["keep256"] = 0.0
    try:
        rc = run_selftest()
    finally:
        RULES["keep256"] = 3.0
    killed += rc != 0
    print(f"D60_READ_MUTANT keep256_floor_zero: {'killed' if rc else 'SURVIVED'}")
    # no-harm clause dropped
    def decide_no_harm(res, r=RULES):
        v, rc_, lines = orig["decide"](res, r)
        if v == "NOT KEEP" and any("harm [" in ln for ln in lines):
            return "KEEP", 0, lines + ["KEEP FUSED_MOE_DECODE_COMPACT_PIPE=192"]
        return v, rc_, lines
    g["decide"] = decide_no_harm
    try:
        rc = run_selftest()
    finally:
        g.update(orig)
    killed += rc != 0
    print(f"D60_READ_MUTANT no_harm_clause: {'killed' if rc else 'SURVIVED'}")
    n = len(muts) + 3
    print(f"D60_READ_MUTANTS {killed}/{n} killed")
    return 0 if killed == n else 1


def main(argv):
    if argv[:1] == ["--selftest"]:
        rc = selftest()
        return rc | (mutants() if "--mutants" in argv else 0)
    if len(argv) != 1:
        print("usage: d60_read.py <dir> | --selftest [--mutants]")
        return 2
    v, rc, lines = read_dir(argv[0])
    for ln in lines:
        print(ln)
    print(f"VERDICT {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# SPDX-License-Identifier: Apache-2.0
"""q38 HCK kernel microbench reader (tests/q38_hck/tpu_bench_hckmb.py JSON; TPU_QWEN4_HC_DECODE_W1_SCHED). Pure Python (no numpy) so the
in-pod read and a Mac re-read agree byte for byte; structure of hcksplit_read.py.
Order of judgement:
  1 VOID       CLOCK (string-exact libtpu final / at-client, token-exact single --xla_tpu_dvfs_p_state=7, served flags, no region trace),
               PLAN (rows 288 / 320, health rows 160 / 288 / 320 / 384 / 496, arms B X XD XO, ABBA B X X B, chain 24 / 4, iters 150),
               HEALTH (B finite + 8 cores equal at every health row), NEGCTL (XM's mixed differs from B's at every health row),
               ENGAGEMENT (B and X compiled + HLO-engaged + census at the timed rows; B and X census at every health row),
               COMPLETE (B1, X1, X2, B2 timed at both rows)
  2 BITWISE    X bitwise equal to B (mixed, injection gate, h) at every health row; else VERDICT HCKMB NOT-BITWISE (a verdict, not VOID)
  3 DECIDE     per timed row: DELTA = mean(B1, B2) - mean(X1, X2) us per call; N = max(|B1 - B2|, |X1 - X2|) (the run's own A/A);
               KEEP iff DELTA >= max(2.0, 2N) at 288 AND 320; LOSS iff DELTA <= -max(1.0, 2N) at either row (LOSS first); else FLAT.
               BIG-FISH flag (descriptive): DELTA >= 7.06 at both rows (0.5 STEP ms/step at 97 calls, factor 0.73); >= 5.15 at factor 1.0.
               STEP ms/step (I) = x * 97 * [0.73, 1.0] / 1000. Positions and drift (B2 - B1) printed; XD / XO from the rotation phase.
usage: hckmb_read.py <dir|json>  |  --selftest [--mutants]"""
import copy
import json
import os
import sys

ROWS = (288, 320)
HEALTH_ROWS = (160, 288, 320, 384, 496)
ARMS = ("B", "X", "XD", "XO")
ABBA = ("B", "X", "X", "B")
PLAN_CHAIN = (24, 4)
PLAN_ITERS = 150
KEEP_MIN = 2.0
LOSS_MIN = 1.0
CALLS = 97
CONV = (0.73, 1.0)
BIG_US = 7.06
MARG_US = 5.15
DVFS = "--xla_tpu_dvfs_p_state"
RT = "--xla_enable_custom_call_region_trace"
SERVED_FLAGS = ("--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
                "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false")
WANT_LIBTPU = " ".join(SERVED_FLAGS) + " " + DVFS + "=7 --xla_tpu_use_enhanced_launch_barrier=false"
ENV_OVERRIDE = ("--xla_tpu_use_dynamic_smem_negotiation=true "
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 "
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864")
WANT_AT_CLIENT = ENV_OVERRIDE + " " + WANT_LIBTPU


def _us(d, k):
    v = (d or {}).get(k)
    return v.get("us_per_call") if isinstance(v, dict) else None


def dvfs_tokens(s):
    return [t for t in (s or "").split() if t.split("=", 1)[0] == DVFS]


def clock_ok(host):
    fin, atc = host.get("libtpu_final", ""), host.get("libtpu_at_client", "")
    why = []
    if fin != WANT_LIBTPU:
        why.append("libtpu_final != WANT_LIBTPU")
    if atc != WANT_AT_CLIENT:
        why.append("libtpu_at_client != WANT_AT_CLIENT")
    for name, s in (("final", fin), ("at_client", atc)):
        if dvfs_tokens(s) != [DVFS + "=7"]:
            why.append(f"{name} dvfs tokens {dvfs_tokens(s)}")
        if not all(f in s.split() for f in SERVED_FLAGS):
            why.append(f"{name} lacks a served flag")
        if any(t.split("=", 1)[0] == RT for t in s.split()):
            why.append(f"{name} carries the region-trace flag")
    return not why, "; ".join(why)


def census_ok(res, rows, arm):
    return bool((((res.get("census") or {}).get(str(rows)) or {}).get("ok") or {}).get(arm))


def engaged(res, rows, arm):
    e = ((res.get("engagement") or {}).get(str(rows)) or {}).get(arm) or {}
    return bool(e.get("compiled") and e.get("engaged") and census_ok(res, rows, arm))


def bitwise(res, rows, arm):
    b = ((((res.get("health") or {}).get(str(rows)) or {}).get("bitwise_vs_B")) or {}).get(arm)
    return bool(isinstance(b, list) and len(b) == 3 and all(x is True for x in b))


def negctl(res, rows):
    b = ((((res.get("health") or {}).get(str(rows)) or {}).get("bitwise_vs_B")) or {}).get("XM")
    return bool(isinstance(b, list) and len(b) == 3 and b[0] is False)


def void_reasons(res):
    why = []
    host = res.get("host") or {}
    ok, w = clock_ok(host)
    if not ok:
        why.append("CLOCK (" + w + ")")
    plan = res.get("plan") or {}
    if (list(plan.get("rows") or []) != list(ROWS) or list(plan.get("health_rows") or []) != list(HEALTH_ROWS)
            or list(plan.get("arms") or []) != list(ARMS) or list(plan.get("abba") or []) != list(ABBA)
            or tuple(host.get("chain") or ()) != PLAN_CHAIN or host.get("iters") != PLAN_ITERS):
        why.append("PLAN")
    for r in HEALTH_ROWS:
        h = (res.get("health") or {}).get(str(r)) or {}
        if not (h.get("ok") and (h.get("finite") or {}).get("B") and (h.get("cores_equal") or {}).get("B")):
            why.append(f"HEALTH {r}")
        if not negctl(res, r):
            why.append(f"NEGCTL {r}")
        for arm in ("B", "X"):
            if not census_ok(res, r, arm):
                why.append(f"CENSUS {r} {arm}")
    for r in ROWS:
        for arm in ("B", "X"):
            if not engaged(res, r, arm):
                why.append(f"ENGAGEMENT {r} {arm}")
        pos = ((res.get("cases") or {}).get(str(r)) or {}).get("positions") or {}
        if any(_us(pos, k) is None for k in ("B1", "X1", "X2", "B2")):
            why.append(f"COMPLETE {r}")
    return why


def step_ms(x):
    return "%+.3f to %+.3f" % (x * CALLS * CONV[0] / 1000.0, x * CALLS * CONV[1] / 1000.0)


def read(res):
    why = void_reasons(res)
    if why:
        return ["VERDICT VOID (" + "; ".join(why) + ")"], 1
    nb = [(r, a, (((res["health"][str(r)].get("ndiff_vs_B")) or {}).get(a))) for r in HEALTH_ROWS for a in ("X",) if not bitwise(res, r, a)]
    lines, delta, noise = [], {}, {}
    for r in ROWS:
        pos = res["cases"][str(r)]["positions"]
        b1, x1, x2, b2 = (_us(pos, k) for k in ("B1", "X1", "X2", "B2"))
        d = (b1 + b2) / 2.0 - (x1 + x2) / 2.0
        n = max(abs(b1 - b2), abs(x1 - x2))
        delta[r], noise[r] = d, n
        lines.append(f"ROW {r} ABBA B1@1 {b1:.3f} X1@2 {x1:.3f} X2@3 {x2:.3f} B2@4 {b2:.3f} us/call | DELTA {d:+.3f} = {step_ms(d)} STEP ms/step "
                     f"(I) | N {n:.3f} | drift B2 - B1 {b2 - b1:+.3f}, X2 - X1 {x2 - x1:+.3f}")
        rot = res["cases"][str(r)].get("rotation") or {}
        rb = _us(rot, "B")
        if rb is not None:
            lines.append(f"ROW {r} rotation (descriptive): " + " ".join(
                f"{a} {_us(rot, a):.3f} (B - {a} {rb - _us(rot, a):+.3f})" for a in ARMS if _us(rot, a) is not None))
    for r in HEALTH_ROWS:
        bw = ((res["health"][str(r)].get("bitwise_vs_B")) or {})
        lines.append(f"HEALTH {r} bitwise vs B: " + " ".join(f"{a} {bw.get(a)}" for a in ("X", "XD", "XO", "XM")))
    if nb:
        return ["VERDICT HCKMB NOT-BITWISE (" + "; ".join(f"{r} {a} ndiff {d}" for r, a, d in nb) + ")"] + lines, 0
    if any(delta[r] <= -max(LOSS_MIN, 2 * noise[r]) for r in ROWS):
        v = "LOSS"
    elif all(delta[r] >= max(KEEP_MIN, 2 * noise[r]) for r in ROWS):
        v = "KEEP"
    else:
        v = "FLAT"
    big = "BIG-FISH" if all(delta[r] >= BIG_US for r in ROWS) else ("BIG-FISH@1.0" if all(delta[r] >= MARG_US for r in ROWS) else "below 0.5 ms")
    head = (f"VERDICT HCKMB {v} | DELTA288 {delta[288]:+.3f} DELTA320 {delta[320]:+.3f} us/call | N288 {noise[288]:.3f} N320 {noise[320]:.3f} | "
            f"bars KEEP >= max({KEEP_MIN}, 2N), LOSS <= -max({LOSS_MIN}, 2N) | {big}")
    return [head] + lines, 0


def load(path):
    p = os.path.join(path, "hckmb.json") if os.path.isdir(path) else path
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- selftest
def fixture(b=(26.0, 26.2), x=(21.0, 21.1), xm_differs=True):
    pc = lambda u: {"us_per_call": u}  # noqa: E731
    res = {"host": {"libtpu_final": WANT_LIBTPU, "libtpu_at_client": WANT_AT_CLIENT, "iters": 150, "chain": [24, 4]},
           "plan": {"rows": list(ROWS), "health_rows": list(HEALTH_ROWS), "arms": list(ARMS), "abba": list(ABBA)},
           "census": {}, "health": {}, "engagement": {}, "cases": {}}
    for r in HEALTH_ROWS:
        res["census"][str(r)] = {"ok": {a: True for a in ARMS}}
        res["health"][str(r)] = {"ok": True, "finite": {"B": True}, "cores_equal": {"B": True},
                                 "bitwise_vs_B": {"X": [True] * 3, "XD": [True] * 3, "XO": [True] * 3,
                                                  "XM": [not xm_differs, True, True]},
                                 "ndiff_vs_B": {"X": [0, 0, 0]}}
    for r in ROWS:
        d = 1.0 if r == 320 else 0.0
        res["engagement"][str(r)] = {a: {"compiled": True, "engaged": True} for a in ARMS}
        res["cases"][str(r)] = {"positions": {"B1": pc(b[0] + d), "X1": pc(x[0] + d), "X2": pc(x[1] + d), "B2": pc(b[1] + d)},
                                "rotation": {"B": pc(b[0] + d), "X": pc(x[0] + d), "XD": pc(b[0] - 1 + d), "XO": pc(b[0] - 3 + d)}}
    return res


def _cases():
    def mod(fn, **kw):
        r = fixture(**kw)
        fn(r)
        return r

    C = [("KEEP and BIG-FISH", fixture(b=(28.5, 28.6), x=(21.0, 21.1)), "VERDICT HCKMB KEEP"),
         ("KEEP below 0.5 ms", fixture(), "VERDICT HCKMB KEEP"),
         ("FLAT (delta 1.5)", fixture(b=(22.5, 22.6), x=(21.0, 21.1)), "VERDICT HCKMB FLAT"),
         ("FLAT: delta 2.5 under a noisy A/A (2N = 3.0)", fixture(b=(25.0, 26.5), x=(23.0, 23.0)), "VERDICT HCKMB FLAT"),
         ("LOSS", fixture(b=(21.0, 21.1), x=(23.0, 23.1)), "VERDICT HCKMB LOSS"),
         ("LOSS first even if one row would KEEP", mod(lambda r: r["cases"]["320"]["positions"].update(X1={"us_per_call": 30.0}, X2={"us_per_call": 30.0})),
          "VERDICT HCKMB LOSS"),
         ("NOT-BITWISE at 496", mod(lambda r: r["health"]["496"]["bitwise_vs_B"].update(X=[False, True, True])), "VERDICT HCKMB NOT-BITWISE"),
         ("VOID negative control dead", fixture(xm_differs=False), "VERDICT VOID (NEGCTL"),
         ("VOID clock dvfs=3", mod(lambda r: r["host"].update(libtpu_final=WANT_LIBTPU.replace("=7", "=3"))), "VERDICT VOID (CLOCK"),
         ("VOID clock plus-joined", mod(lambda r: r["host"].update(libtpu_at_client=WANT_AT_CLIENT.replace(DVFS + "=7", DVFS + "=7+X=1"))), "VERDICT VOID (CLOCK"),
         ("VOID health B", mod(lambda r: r["health"]["160"]["finite"].update(B=False)), "VERDICT VOID (HEALTH 160"),
         ("VOID engagement X", mod(lambda r: r["engagement"]["320"].update(X={"compiled": True, "engaged": False})), "VERDICT VOID (ENGAGEMENT 320 X"),
         ("VOID census X at 384", mod(lambda r: r["census"]["384"]["ok"].update(X=False)), "VERDICT VOID (CENSUS 384 X"),
         ("VOID plan order", mod(lambda r: r["plan"].update(abba=["B", "X", "B", "X"])), "VERDICT VOID (PLAN"),
         ("VOID complete", mod(lambda r: r["cases"]["288"]["positions"].pop("X2")), "VERDICT VOID (COMPLETE 288")]
    return C


def selftest():
    bad = 0
    for label, res, want in _cases():
        lines, rc = read(copy.deepcopy(res))
        ok = lines[0].startswith(want) and (rc == 1) == want.startswith("VERDICT VOID")
        print(f"HCKMBREAD_SELFTEST {'ok ' if ok else 'BAD'} {label}: {lines[0][:150]}")
        bad += not ok
    for s_, want in ((DVFS + "=7", True), (DVFS + "=7+X=1", False), (DVFS + "=70", False), (DVFS + "=7 " + DVFS + "=7", False), ("", False)):
        got = dvfs_tokens(s_) == [DVFS + "=7"]
        print(f"HCKMBREAD_SELFTEST {'ok ' if got == want else 'BAD'} dvfs token rule {s_!r}: {got}")
        bad += got != want
    cfg = (ROWS == (288, 320) and HEALTH_ROWS == (160, 288, 320, 384, 496) and ABBA == ("B", "X", "X", "B") and PLAN_ITERS == 150
           and KEEP_MIN == 2.0 and LOSS_MIN == 1.0 and CALLS == 97 and CONV == (0.73, 1.0) and BIG_US == 7.06 and MARG_US == 5.15)
    print(f"HCKMBREAD_SELFTEST {'ok ' if cfg else 'BAD'} pre-stated constants")
    bad += not cfg
    print("HCKMBREAD_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants():
    g = globals()
    names = ("void_reasons", "negctl", "bitwise", "KEEP_MIN", "dvfs_tokens", "engaged")
    orig = {k: g[k] for k in names}

    def regex_dvfs(s):
        import re
        m = re.search(DVFS + r"=(\S+)", s or "")
        return [DVFS + "=" + m.group(1)[:1]] if m else []

    muts = {
        "negative control ignored": ("negctl", lambda res, rows: True),
        "bitwise gate ignored": ("bitwise", lambda res, rows, arm: True),
        "KEEP bar at 1.0": ("KEEP_MIN", 1.0),
        "dvfs parsed by regex": ("dvfs_tokens", regex_dvfs),
        "census rule dropped": ("void_reasons", lambda res: [w for w in orig["void_reasons"](res) if not w.startswith("CENSUS")]),
        "engagement ignores the HLO rule": ("engaged", lambda res, rows, arm: orig["engaged"].__globals__["census_ok"](res, rows, arm)),
        "plan rule dropped": ("void_reasons", lambda res: [w for w in orig["void_reasons"](res) if w != "PLAN"]),
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
        caught = rc != 0
        print(f"HCKMBREAD_MUTANT {'killed  ' if caught else 'SURVIVED'} {name}")
        bad += not caught
    print("HCKMBREAD_MUTANTS", "PASS" if not bad else f"FAIL ({bad} survived)", f"({len(muts)} mutants)")
    return 0 if not bad else 1


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest() | (mutants() if "--mutants" in argv else 0)
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        res = load(argv[0])
    except (OSError, ValueError) as e:
        print(f"VERDICT VOID (no JSON: {type(e).__name__})")
        return 2
    lines, rc = read(res)
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

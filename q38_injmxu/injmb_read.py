# SPDX-License-Identifier: Apache-2.0
"""q38 INJMXU-BUILD INJ-MXU microbench reader (tests/q38_injmxu/tpu_bench_injmb.py JSON, --rows 288,320,352,384,448 --arms
W1,W1F,W1T; rules of reports/prereg/q38_injmxu_mb_prereg_LOCKED.md; device health = reports/prereg/q38_injmxu_bound_v2_LOCKED.md s5).
Structure and clock rule of HC-DECOMP's w1dc_read.py (credit); logit-set method of the CMP harness hcc_numerics.py (credit), in
pure Python (no numpy) so the in-pod read and a Mac re-read agree byte for byte.
  Order of judgement:
    1 VOID     CLOCK (libtpu_final == WANT_LIBTPU, libtpu_at_client == WANT_AT_CLIENT, both served flags, no dvfs_p_state),
               PLAN (rows / arms / chain 24/4), HEALTH (every row: finite, 8 cores equal, W1T bitwise W1, gates recorded),
               NEGCTL (the live negative control W1FM must FAIL the device leg at every row)
    2 BOUND    the bound v2 device leg at every row, E = W1F vs A = W1: G1 (tau_prob = 2^-14 S, |L| and tau inflated by 1 + 2^-6),
               G3, G4, O1 (mixed and h bitwise). Any failure -> VERDICT INJMXU BOUND-FAIL (the timing read is void, the bound fails)
    3 VOID     timing: COMPILE / ENGAGEMENT (W1 / W1T = 24 _w1 calls, W1F = 24 _w1f calls, 0 up relayout), base / base2 present,
               SPREAD |base - base2| / min > 5 %, A/A |W1 - W1T| > 1.0 us/call at 288, 320 or 352
    4 DECIDE   DELTA = W1 - W1F (us/call, interleaved phase): VERDICT INJMXU KEEP iff DELTA >= 4.0 at 320 AND >= 4.0 at 352 AND
               >= 3.5 at 288; else VERDICT INJMXU NO-GO. 384 / 448 descriptive; ms/step x 96. The p7 (dvfs) saving is unmeasured.
usage: injmb_read.py <dir|json>  |  --selftest [--mutants]"""
import json
import math
import os
import struct
import sys

ROWS = (288, 320, 352, 384, 448)
ARMS = ("W1", "W1F", "W1T")
PLAN_CHAIN = (24, 4)
SPREAD_MAX = 0.05
AA_MAX = 1.0
AA_ROWS = (288, 320, 352)
BARS = {288: 3.5, 320: 4.0, 352: 4.0}
CALLS = 96
WANT_LIBTPU = ("--xla_tpu_use_minor_sharding_for_major_trivial_input=true "
               "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false "
               "--xla_tpu_use_enhanced_launch_barrier=false")
ENV_OVERRIDE = ("--xla_tpu_use_dynamic_smem_negotiation=true "
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 "
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864")
WANT_AT_CLIENT = ENV_OVERRIDE + " " + WANT_LIBTPU
SERVED_FLAGS = ("--xla_tpu_use_minor_sharding_for_major_trivial_input=true", "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false")
BR = 4
TOLG = 4 * 2.0 ** -24
KWIN = 512
INFL = 1.0 + 2.0 ** -6
TAU_PROB = 2.0 ** -14
GSLACK = 2.0 ** -22


# ---------------------------------------------------------------- bf16 / logit sets (pure Python)
def _f32bits(x):
    return struct.unpack(">I", struct.pack(">f", x))[0]


def from_ord16(k):
    bits = k if k >= 0 else ((-k) | 0x8000)
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def ord16_round(x):
    """ord16 of the bf16 nearest x (round to nearest even on the f32 bits)."""
    u = _f32bits(x)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return -(r & 0x7FFF) if r & 0x8000 else r


def _sig(x):
    if x < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit_set(g):
    """(lo, hi, empty, saturated): the ord16 interval of bf16 logits b with |2 sigmoid(b / 4) - g| <= TOLG within +-KWIN of the
    nearest bf16 of 4 ln(g / (2 - g)) (hcc_numerics.logit_sets; the set is an interval, so it is grown outward from the centre)."""
    b0 = BR * math.log(g / (2.0 - g)) if 0.0 < g < 2.0 else 0.0
    k0 = ord16_round(b0)
    lo_w, hi_w = max(k0 - KWIN, -0x7F7F), min(k0 + KWIN, 0x7F7F)

    def ok(k):
        return abs(2.0 * _sig(from_ord16(k) / BR) - g) <= TOLG

    seed = next((k for k in (k0, k0 - 1, k0 + 1, k0 - 2, k0 + 2) if lo_w <= k <= hi_w and ok(k)), None)
    if seed is None:
        return None, None, True, False
    lo = hi = seed
    while lo - 1 >= lo_w and ok(lo - 1):
        lo -= 1
    while hi + 1 <= hi_w and ok(hi + 1):
        hi += 1
    return lo, hi, False, (lo == lo_w or hi == hi_w)


def ulp_bf16(x):
    if x >= 2.0 ** -126:
        m, e = math.frexp(x)
        return math.ldexp(1.0, e - 8)
    return 2.0 ** -133


def device_leg(gE, gA, L, S, o1=None):
    """Bound v2 s5 on one row's gates ([T][4] lists): G1 (tau_prob), G3, G4, and O1 from the bench's bitwise records."""
    n = n_diff = n_g1 = n_g3 = n_g4 = n_sat = 0
    worst = (0.0, None, None)
    for r in range(len(gA)):
        for j in range(len(gA[r])):
            e, a, l, s = gE[r][j], gA[r][j], L[r][j], S[r][j]
            n += 1
            eq = _f32bits(e) == _f32bits(a)
            lE, hE, emE, saE = logit_set(e)
            lA, hA, emA, saA = logit_set(a)
            if emE:
                n_g4 += 1
            if eq:
                continue
            n_diff += 1
            if emE or emA:
                n_g1 += 1
                n_g3 += 1
                continue
            tau = TAU_PROB * s * INFL
            allow = tau + ulp_bf16(abs(l) * INFL + tau)
            over = lE <= hA and lA <= hE
            dist = 0.0 if over else abs(from_ord16(lA if hE < lA else lE) - from_ord16(hE if hE < lA else hA))
            dg = abs(e - a)
            sat = saE or saA
            if sat:
                n_sat += 1
                g1 = g3 = dg <= tau / 8.0 + GSLACK
            else:
                g1 = dist <= allow
                single_same = lE == hE == lA == hA
                g3 = (not single_same) and dg <= dist / 8.0 + GSLACK
                if dist / allow > worst[0]:
                    worst = (dist / allow, r, abs(l) / s if s > 0 else None)
            n_g1 += not g1
            n_g3 += not g3
    rec = {"n": n, "diff": n_diff, "sat": n_sat, "G1_over": n_g1, "G3_over": n_g3, "G4_empty": n_g4, "worst_ratio": round(worst[0], 6),
           "worst_row": worst[1], "worst_L_over_S": None if worst[2] is None else round(worst[2], 6)}
    rec["G1"], rec["G3"], rec["G4"] = n_g1 == 0, n_g3 == 0, n_g4 == 0
    rec["O1"] = bool(o1 and o1.get("mixed_bitwise") and o1.get("h_bitwise"))
    rec["ok"] = rec["G1"] and rec["G3"] and rec["G4"] and rec["O1"]
    return rec


# ---------------------------------------------------------------- the read
def us(case, name):
    v = (case.get(name) or {}).get("us_per_call")
    return float(v) if isinstance(v, (int, float)) else None


def clock_ok(host):
    f, c = host.get("libtpu_final"), host.get("libtpu_at_client")
    return (f == WANT_LIBTPU and c == WANT_AT_CLIENT and all(x in str(c).split() for x in SERVED_FLAGS)
            and "dvfs_p_state" not in str(f) + str(c))


def structural(d):
    void = []
    p, host = d.get("plan") or {}, d.get("host") or {}
    if not clock_ok(host):
        void.append("VOID-CLOCK (libtpu_final %r)" % str(host.get("libtpu_final"))[:120])
    if tuple(p.get("rows") or ()) != ROWS or tuple(p.get("arms") or ()) != ARMS or tuple(host.get("chain") or ()) != PLAN_CHAIN:
        void.append("PLAN %s / %s / %s" % (p.get("rows"), p.get("arms"), host.get("chain")))
    neg = {}
    for r in ROWS:
        h = (d.get("health") or {}).get(str(r)) or {}
        g = h.get("gates") or {}
        if not h.get("ok") or not h.get("twin_bitwise") or not all(k in g for k in ("A", "E", "NEG", "L", "S")):
            void.append("HEALTH %d" % r)
            continue
        neg[r] = device_leg(g["NEG"], g["A"], g["L"], g["S"], h.get("neg_o1"))
        if neg[r]["ok"]:
            void.append("VOID-NEGCTL %d (the negative control passed the device leg)" % r)
    return void, neg


def timing_void(d):
    void = []
    for r in ROWS:
        c = (d.get("cases") or {}).get(str(r)) or {}
        e = (d.get("engagement") or {}).get(str(r)) or {}
        for arm in ARMS:
            if us(c, arm) is None or not (e.get(arm) or {}).get("compiled"):
                void.append("COMPILE %s@%d" % (arm, r))
            elif (e.get(arm) or {}).get("engaged") is not True:
                void.append("ENGAGEMENT %s@%d" % (arm, r))
        b, b2 = us(c, "base"), us(c, "base2")
        if b is None or b2 is None:
            void.append("COMPILE base@%d" % r)
        elif abs(b - b2) / min(b, b2) > SPREAD_MAX:
            void.append("SPREAD %d %.3f" % (r, abs(b - b2) / min(b, b2)))
        if r in AA_ROWS and us(c, "W1") is not None and us(c, "W1T") is not None and abs(us(c, "W1") - us(c, "W1T")) > AA_MAX:
            void.append("VOID-AA %d |W1 - W1T| %.3f > %.1f" % (r, abs(us(c, "W1") - us(c, "W1T")), AA_MAX))
    return void


def decide(delta):
    return "KEEP" if all(delta[r] >= b for r, b in BARS.items()) else "NO-GO"


def read(d):
    void, neg = structural(d)
    if void:
        return "VERDICT VOID (" + "; ".join(void[:12]) + ")", []
    lines, legs = [], {}
    for r in ROWS:
        h = d["health"][str(r)]
        g = h["gates"]
        legs[r] = device_leg(g["E"], g["A"], g["L"], g["S"], h.get("o1"))
        lines.append("HEALTH %d: E=W1F vs A=W1 %s %s | negative control %s (G1 over %d)" % (
            r, "PASS" if legs[r]["ok"] else "FAIL", json.dumps(legs[r], sort_keys=True), "FAILS (ok)" if not neg[r]["ok"] else "PASSES",
            neg[r]["G1_over"]))
    bad = [r for r in ROWS if not legs[r]["ok"]]
    if bad:
        return "VERDICT INJMXU BOUND-FAIL (device leg failed at rows %s; timing VOID; bound v2 FAILS)" % bad, lines
    tv = timing_void(d)
    if tv:
        return "VERDICT VOID (" + "; ".join(tv[:12]) + "; device leg PASSED at every row)", lines
    delta = {}
    for r in ROWS:
        c = d["cases"][str(r)]
        delta[r] = us(c, "W1") - us(c, "W1F")
        lines.append("ROW %d: base %.3f W1 %.3f W1F %.3f W1T %.3f base2 %.3f us/call | W1 - W1F %+.3f us/call = %+.3f ms/step x %d | A/A W1 - W1T %+.3f"
                     % (r, us(c, "base"), us(c, "W1"), us(c, "W1F"), us(c, "W1T"), us(c, "base2"), delta[r], delta[r] * CALLS / 1e3, CALLS,
                        us(c, "W1") - us(c, "W1T")))
    v = decide(delta)
    lines.append("INJ-MXU bars: " + ", ".join("%d: %.3f >= %.1f %s" % (r, delta[r], b, "Y" if delta[r] >= b else "N") for r, b in BARS.items())
                 + "; device leg PASS at every row; served clock (no dvfs): the p7 saving is unmeasured")
    return "VERDICT INJMXU %s" % v, lines


def _load(path):
    p = os.path.join(path, "injmb.json") if os.path.isdir(path) else path
    return json.load(open(p))


# ---------------------------------------------------------------- selftest
def _gate(b):
    return struct.unpack(">f", struct.pack(">f", 2.0 * _sig(b / BR)))[0]


def _fixture(w1=30.0, w1f=25.0, w1t=None, spread=0.2, eng=True, clock=WANT_LIBTPU, client=WANT_AT_CLIENT, flip=None, neg_pass=False,
             o1_ok=True, twin=True):
    d = {"plan": {"rows": list(ROWS), "arms": list(ARMS)}, "host": {"chain": [24, 4], "libtpu_final": clock, "libtpu_at_client": client},
         "cases": {}, "health": {}, "engagement": {}}
    for r in ROWS:
        k = r / 448
        d["cases"][str(r)] = {"base": {"us_per_call": w1 * k}, "base2": {"us_per_call": w1 * k + spread}, "W1": {"us_per_call": w1 * k},
                              "W1F": {"us_per_call": w1 * k - (w1 - w1f)}, "W1T": {"us_per_call": w1 * k + (0.1 if w1t is None else w1t)}}
        T = 6
        bs = [[from_ord16(ord16_round(0.3 + 0.17 * i + 0.05 * j)) for j in range(BR)] for i in range(T)]
        A = [[_gate(b) for b in row] for row in bs]
        E = [list(row) for row in A]
        if flip:
            i, j, st = flip
            kk = ord16_round(bs[i][j]) + st
            E[i][j] = _gate(from_ord16(kk))
        NEG = [list(row) for row in A] if neg_pass else [[_gate(bs[i][(j + 1) % BR]) for j in range(BR)] for i in range(T)]
        L = [[b for b in row] for row in bs]
        S = [[3.0 * abs(b) for b in row] for row in bs]
        d["health"][str(r)] = {"ok": True, "twin_bitwise": twin, "o1": {"mixed_bitwise": o1_ok, "h_bitwise": True},
                               "neg_o1": {"mixed_bitwise": True, "h_bitwise": True}, "gates": {"A": A, "E": E, "NEG": NEG, "L": L, "S": S}}
        d["engagement"][str(r)] = {a: {"compiled": True, "engaged": eng} for a in ARMS}
    return d


def selftest():
    bad = 0

    def chk(name, got, want):
        nonlocal bad
        ok = got.startswith(want)
        print("INJMBREAD_SELFTEST %s %s -> %s" % ("ok " if ok else "BAD", name, got[:100]))
        bad += not ok

    f = _fixture
    chk("nominal DELTA 5.0, health pass -> KEEP", read(f())[0], "VERDICT INJMXU KEEP")
    chk("DELTA 3.9 -> NO-GO", read(f(w1f=26.1))[0], "VERDICT INJMXU NO-GO")
    d = f()
    d["cases"]["352"]["W1F"]["us_per_call"] = d["cases"]["352"]["W1"]["us_per_call"] - 3.99
    chk("DELTA at 352 3.99 only -> NO-GO", read(d)[0], "VERDICT INJMXU NO-GO")
    d = f()
    d["cases"]["320"]["W1F"]["us_per_call"] = d["cases"]["320"]["W1"]["us_per_call"] - 3.99
    chk("DELTA at 320 3.99 only -> NO-GO", read(d)[0], "VERDICT INJMXU NO-GO")
    d = f()
    d["cases"]["288"]["W1F"]["us_per_call"] = d["cases"]["288"]["W1"]["us_per_call"] - 3.49
    chk("DELTA at 288 3.49 only -> NO-GO", read(d)[0], "VERDICT INJMXU NO-GO")
    d = f()
    d["cases"]["288"]["W1F"]["us_per_call"] = d["cases"]["288"]["W1"]["us_per_call"] - 3.5
    chk("DELTA at 288 exactly 3.5 -> KEEP (>=)", read(d)[0], "VERDICT INJMXU KEEP")
    chk("decide 352 a hair below -> NO-GO", decide({288: 3.5, 320: 4.0, 352: 4.0 - 1e-12}), "NO-GO")
    chk("one logit +1 ulp (allowed) -> KEEP", read(f(flip=(2, 1, 1)))[0], "VERDICT INJMXU KEEP")
    chk("one logit +3 ulp -> BOUND-FAIL", read(f(flip=(2, 1, 3)))[0], "VERDICT INJMXU BOUND-FAIL")
    chk("one logit +2 ulp -> BOUND-FAIL (allowance ~1 ulp in the bulk)", read(f(flip=(2, 1, 2)))[0], "VERDICT INJMXU BOUND-FAIL")
    chk("mixed not bitwise -> BOUND-FAIL", read(f(o1_ok=False))[0], "VERDICT INJMXU BOUND-FAIL")
    chk("negative control passes -> VOID-NEGCTL", read(f(neg_pass=True))[0], "VERDICT VOID (VOID-NEGCTL")
    chk("twin not bitwise -> VOID HEALTH", read(f(twin=False))[0], "VERDICT VOID (HEALTH")
    chk("A/A 1.2 us at 288-352 -> VOID-AA", read(f(w1t=1.2))[0], "VERDICT VOID (VOID-AA")
    chk("A/A 0.9 -> KEEP", read(f(w1t=0.9))[0], "VERDICT INJMXU KEEP")
    chk("dvfs in final -> VOID-CLOCK", read(f(clock=WANT_LIBTPU + " --xla_tpu_dvfs_p_state=7"))[0], "VERDICT VOID (VOID-CLOCK")
    chk("client = bare 2-flag -> VOID-CLOCK", read(f(client=WANT_LIBTPU))[0], "VERDICT VOID (VOID-CLOCK")
    chk("dvfs at client -> VOID-CLOCK", read(f(client=WANT_AT_CLIENT + " --xla_tpu_dvfs_p_state=7"))[0], "VERDICT VOID (VOID-CLOCK")
    chk("spread 6 % -> VOID", read(f(spread=2.4))[0], "VERDICT VOID (SPREAD")
    chk("not engaged -> VOID", read(f(eng=False))[0], "VERDICT VOID (ENGAGEMENT")
    d = f()
    d["plan"]["arms"] = ["W1", "W1F"]
    chk("plan arms changed -> VOID", read(d)[0], "VERDICT VOID (PLAN")
    d = f(flip=(2, 1, 3))
    d["cases"]["288"]["base2"]["us_per_call"] = 99.0
    chk("bound fails AND spread bad -> BOUND-FAIL (bound outranks timing VOIDs)", read(d)[0], "VERDICT INJMXU BOUND-FAIL")
    lo, hi, em, sa = logit_set(_gate(1.0))
    chk("logit_set of g(1.0) is {1.0}", "ok" if (not em and lo == hi == ord16_round(1.0)) else "bad", "ok")
    chk("off-chain gate -> empty set", "ok" if logit_set(2.0 * _sig((1.0 + 2.0 ** -12) / BR))[2] else "bad", "ok")
    print("INJMBREAD_SELFTEST", "PASS" if not bad else "FAIL (%d)" % bad)
    return 0 if not bad else 1


MUTANTS = [("bar at 352 3.0", "BARS = {288: 3.5, 320: 4.0, 352: 4.0}", "BARS = {288: 3.5, 320: 4.0, 352: 3.0}"),
           ("bar at 288 3.0", "BARS = {288: 3.5, 320: 4.0, 352: 4.0}", "BARS = {288: 3.0, 320: 4.0, 352: 4.0}"),
           ("bars strict >", "    return \"KEEP\" if all(delta[r] >= b for r, b in BARS.items()) else \"NO-GO\"",
            "    return \"KEEP\" if all(delta[r] > b for r, b in BARS.items()) else \"NO-GO\""),
           ("bound leg not required", "    if bad:\n        return \"VERDICT INJMXU BOUND-FAIL", "    if False:\n        return \"VERDICT INJMXU BOUND-FAIL"),
           ("O1 not required", "    rec[\"ok\"] = rec[\"G1\"] and rec[\"G3\"] and rec[\"G4\"] and rec[\"O1\"]", "    rec[\"ok\"] = rec[\"G1\"] and rec[\"G3\"] and rec[\"G4\"]"),
           ("G1 allowance doubled", "            allow = tau + ulp_bf16(abs(l) * INFL + tau)", "            allow = 2 * (tau + ulp_bf16(abs(l) * INFL + tau))"),
           ("negative control not required", "        if neg[r][\"ok\"]:\n", "        if False:\n"),
           ("clock rule dropped", "    if not clock_ok(host):\n", "    if False:\n"),
           ("A/A rule dropped", "abs(us(c, \"W1\") - us(c, \"W1T\")) > AA_MAX:", "False:"),
           ("spread rule dropped", "        elif abs(b - b2) / min(b, b2) > SPREAD_MAX:\n", "        elif False:\n"),
           ("engagement not required", "            elif (e.get(arm) or {}).get(\"engaged\") is not True:\n", "            elif False:\n"),
           ("twin rule dropped", "if not h.get(\"ok\") or not h.get(\"twin_bitwise\") or", "if not h.get(\"ok\") or")]


def mutants():
    import subprocess
    import tempfile
    src = open(__file__).read()
    head, sep, tail = src.partition("\nMUTANTS = [")
    bad = 0
    for name, a, b in MUTANTS:
        assert head.count(a) == 1, name
        fd, p = tempfile.mkstemp(suffix="_injmb_read_mut.py")
        os.close(fd)
        open(p, "w").write(head.replace(a, b) + sep + tail)
        rc = subprocess.run([sys.executable, "-I", p, "--selftest"], capture_output=True, text=True).returncode
        os.unlink(p)
        print("INJMBREAD_MUTANT %s %s" % ("killed  " if rc else "SURVIVED", name))
        bad += not rc
    print("INJMBREAD_MUTANTS", "PASS" if not bad else "FAIL (%d survived)" % bad, "(%d mutants)" % len(MUTANTS))
    return 0 if not bad else 1


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest() | (mutants() if "--mutants" in argv else 0)
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        d = _load(argv[0])
    except (OSError, ValueError) as e:
        print("VERDICT VOID (no JSON: %s)" % e)
        return 2
    v, lines = read(d)
    print(v)
    for ln in lines:
        print(ln)
    return 1 if v.startswith("VERDICT VOID") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

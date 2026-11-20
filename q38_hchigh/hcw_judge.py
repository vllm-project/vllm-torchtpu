# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS CPU-proof judge: applies the proof's pre-stated rows to the numerics and identity JSONs.

usage: hcw_judge.py num <dir of case JSONs>        P4: every planned case ran; N1 / N2 gates (locked prereg); rule-23 flag in
                                                   every record; MD reads NOT bitwise; ME fails the N2 gate; MF fails the N1 gate
       hcw_judge.py ident <dir of ident JSONs>     P5: base == head unset == head WIDE=0 everywhere; WIDE=1 / 2 == base below
                                                   160 rows and != base at 160-496; factor keys head - base == the two new keys
Last line 'HCWJUDGE <num|ident> PASS|FAIL (...)'; exit 0 iff PASS.
"""
import json
import os
import sys

CASES = [f"combine:{r}:{k}" for r in (160, 240, 448) for k in ("random", "correlated", "peaked", "range_ends")]
CASES += ["combine:496:random", "mix:240:random", "mix:240:peaked", "final:240:random", "final:240:peaked"]
MUTANT_CASES = {"MD": "combine:160:random", "ME": "combine:240:random", "MF": "combine:160:random"}
BUCKETS = (16, 64, 112, 128, 144, 160, 192, 240, 288, 384, 448, 496)
OPS = ("mix_inj", "mix_noinj", "combine_inj", "combine_noinj")
NEW_KEYS = {"TPU_QWEN4_HC_DECODE_COMPARE", "TPU_QWEN4_HC_DECODE_WIDE"}


def fname(case, mutant=None):
    return (f"{mutant}_" if mutant else "") + case.replace(":", "_") + ".json"


def judge_num(d):
    fails, lines = [], []
    n1_bit = {}
    for c in CASES:
        f = os.path.join(d, fname(c))
        if not os.path.exists(f):
            fails.append(f"{c} did not run")
            continue
        r = json.load(open(f))
        if not r.get("excess_precision_off"):
            fails.append(f"{c} ran without --xla_allow_excess_precision=false")
        n1, n2 = r["n1"], r["n2"]
        n1_bit[c] = n1["all_bitwise"]
        if not r.get("s_finite"):
            fails.append(f"{c} served output not finite")
        if not n1["gate_ok"]:
            fails.append(f"{c} N1 gate")
        if not n2["gate_ok"]:
            fails.append(f"{c} N2 gate")
        inj = n2.get("inj") or {}
        lines.append(f"HCWJUDGE CASE {c}: N1 gate {n1['gate_ok']} bitwise {n1['bitwise']} mixed max {n1['mixed_tol']['max_diff']:.3g} "
                     f"frac {n1['mixed_tol']['diff_frac']:.2g} | N2 gate {n2['gate_ok']} inj ok {inj.get('ok')} over {inj.get('n_over')} "
                     f"served_ref {inj.get('served_ref_ok')} diff_vs_S {inj.get('n_diff_vs_s')}/{inj.get('n')} chain {inj.get('bf16_chain')} "
                     f"mixed max {n2['mixed_tol']['max_diff']:.3g} frac {n2['mixed_tol']['diff_frac']:.2g} l2 {n2['mixed_l2']['n2']:.3g} "
                     f"vs S {n2['mixed_l2']['s']:.3g} h {n2['h_bitwise']}")
    for m, c in MUTANT_CASES.items():
        f = os.path.join(d, fname(c, m))
        if not os.path.exists(f):
            fails.append(f"mutant {m} did not run")
            continue
        r = json.load(open(f))
        if m == "MD":
            caught = r["excess_precision_off"] and not r["n1"]["all_bitwise"]
        elif m == "ME":
            caught = not r["n2"]["gate_ok"]
        else:
            caught = not r["n1"]["gate_ok"]
        lines.append(f"HCWJUDGE MUTANT {m} at {c}: {'caught' if caught else 'NOT CAUGHT'} (N1 bitwise {r['n1']['all_bitwise']}, "
                     f"N1 gate {r['n1']['gate_ok']}, N2 gate {r['n2']['gate_ok']})")
        if not caught:
            fails.append(f"mutant {m} not caught")
    nb = sum(n1_bit.values())
    lines.append(f"HCWJUDGE N1 CPU bitwise (rule 23: excess precision off + MD control): {nb}/{len(n1_bit)} cases bitwise on all outputs"
                 + ("" if nb == len(n1_bit) else f"; not bitwise: {sorted(c for c, v in n1_bit.items() if not v)} (the device decides)"))
    return fails, lines


def judge_ident(d):
    fails, lines = [], []
    load = lambda n: json.load(open(os.path.join(d, f"{n}.json")))  # noqa: E731
    try:
        base, unset, zero, w1, w2 = (load(n) for n in ("base", "head_unset", "head_0", "head_1", "head_2"))
    except OSError as e:
        return [f"missing ident JSON: {e}"], lines
    for n, x in (("base", base), ("unset", unset), ("0", zero), ("1", w1), ("2", w2)):
        if x["errors"]:
            fails.append(f"{n} lowering errors {x['errors'][:2]}")
    keys = [f"{o}@{t}" for o in OPS for t in BUCKETS]
    for n, x in (("unset", unset), ("0", zero)):
        bad = [k for k in keys if x["hash"].get(k) != base["hash"].get(k)]
        lines.append(f"HCWJUDGE IDENT head {n} vs base: {len(keys) - len(bad)}/{len(keys)} identical")
        if bad:
            fails.append(f"head {n} differs from base at {bad[:6]}")
        if x["names"] != base["names"]:
            fails.append(f"head {n} op names {x['names']} != base {base['names']}")
    for m, x in (("1", w1), ("2", w2)):
        lo = [k for k in keys if int(k.split("@")[1]) < 160]
        hi = [k for k in keys if int(k.split("@")[1]) >= 160]
        lo_bad = [k for k in lo if x["hash"].get(k) != base["hash"].get(k)]
        hi_bad = [k for k in hi if x["hash"].get(k) == base["hash"].get(k) or not x["hash"].get(k + "#wide")]
        lines.append(f"HCWJUDGE IDENT WIDE={m}: below 160 rows {len(lo) - len(lo_bad)}/{len(lo)} == base; 160-496 "
                     f"{len(hi) - len(hi_bad)}/{len(hi)} != base with the _w{m} kernel")
        if lo_bad or hi_bad:
            fails.append(f"WIDE={m} lo {lo_bad[:4]} hi {hi_bad[:4]}")
    extra = set(unset["factor_keys"]) - set(base["factor_keys"])
    gone = set(base["factor_keys"]) - set(unset["factor_keys"])
    lines.append(f"HCWJUDGE IDENT compile env-factor keys head - base = {sorted(extra)}, base - head = {sorted(gone)} (global compile "
                 "hash moves by construction: two new registered keys and new / changed files under vllm_torchtpu/kernels)")
    if extra != NEW_KEYS or gone:
        fails.append(f"factor keys extra {sorted(extra)} gone {sorted(gone)}")
    return fails, lines


def main(argv):
    mode, d = argv
    fails, lines = (judge_num if mode == "num" else judge_ident)(d)
    for ln in lines:
        print(ln)
    print(f"HCWJUDGE {mode} {'PASS' if not fails else 'FAIL'} ({'; '.join(fails[:10]) or 'all rows hold'})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

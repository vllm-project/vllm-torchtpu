# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS numerics harness (CPU, Pallas interpret mode): TPU_QWEN4_HC_DECODE_WIDE modes 1 (N1) and 2 (N2) against
the served hc_decode.py (S, max_tile 512) and Benliang Wang's float64 reference, under the LOCKED rules of
reports/prereg/q38_hchigh_numerics_prereg_LOCKED.md (sha256 bd7e0341...).

Rule 23: the harness runs with --xla_allow_excess_precision=false (it re-execs itself with the flag added to XLA_FLAGS and
asserts the flag before importing jax). Stress kinds from HC-FIX's case hc_decode_cmp.py and the float64 reference from
tests/kernels/test_qwen4_exp_hc_decode_cmp_tpu.py (credit, both unchanged in substance).

usage: hcw_numerics.py --case FORM:ROWS:KIND --out JSON [--module PATH]   (one case per process; --module loads a source
       mutant of hc_decode_wide.py in its place)
       hcw_numerics.py --selftest                                         (comparator / allowance checks, no kernels)
Prints 'HCWNUM CASE <json>' and exits 0 iff the case ran (verdicts are in the JSON; the proof body judges them).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FLAG = "--xla_allow_excess_precision=false"
if "--selftest" not in sys.argv and FLAG not in os.environ.get("XLA_FLAGS", "").split():
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + FLAG).strip()
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

BR, H, LR, EPS = 4, 2560, 320, 1e-6
W = BR * H
EPS_G, RHO = 2.0 ** -20, 2.0 ** -11
KINDS = ("random", "correlated", "peaked", "range_ends")
FORMS = ("combine", "mix", "final")  # combine: prologue + injection; mix: injection, no prologue; final: prologue, no injection


def _bf16(x):
    import ml_dtypes
    return np.asarray(x, np.float64).astype(ml_dtypes.bfloat16).astype(np.float64)


def _sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def inputs(rows, kind, seed):
    """HC-FIX's stress kinds (scratch/hillclimb/hcfix/numerics/cases/hc_decode_cmp.py inputs(), credit)."""
    g = np.random.default_rng(seed)
    n = lambda *s: g.standard_normal(s).astype(np.float32)  # noqa: E731
    residual, block = 4.0 * n(rows, W), n(rows, H)
    injection = 2.0 / (1.0 + np.exp(-n(rows, BR)))
    w, down, up, inj = 0.3 * n(W), 0.02 * n(LR, W), 0.05 * n(W, LR), 0.02 * n(BR, W)
    if kind == "correlated":
        residual = residual[:1] + 1e-2 * residual
        block = block[:1] + 1e-2 * block
    elif kind == "peaked":
        down, up, inj = 8.0 * down, 8.0 * up, 8.0 * inj
    elif kind == "range_ends":
        scales = np.exp2(np.linspace(-12, 12, rows)).astype(np.float32)[:, None]
        residual, block = residual * scales, block * scales
        residual[rows // 2] = 0.0
    elif kind != "random":
        raise ValueError(kind)
    f = lambda a: _bf16(a).astype(np.float32)  # noqa: E731
    return dict(residual=f(residual), block=f(block), injection=injection.astype(np.float32), weight=f(w), down=f(down),
                up=f(up), inject=f(inj))


def reference(x, prologue, inject):
    """Benliang Wang's float64 reference (_reference of test_qwen4_exp_hc_decode_cmp_tpu.py, credit) + the norm it used."""
    w, down, up = (np.asarray(x[k], np.float64) for k in ("weight", "down", "up"))
    h = np.asarray(x["residual"], np.float64)
    if prologue:
        blk, pinj = np.asarray(x["block"], np.float64), np.asarray(x["injection"], np.float64)
        h = np.concatenate([_bf16(h[:, b * H:(b + 1) * H] + blk * pinj[:, b:b + 1]) for b in range(BR)], axis=1)
    norm = []
    for b in range(BR):
        v = h[:, b * H:(b + 1) * H]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + w[b * H:(b + 1) * H])))
    norm = np.concatenate(norm, axis=1)
    z = _bf16(_bf16(norm @ down.T) / BR)
    gate = _bf16(z * _sig(z))
    acc = 0.0
    for b in range(BR):
        acc = acc + _sig(_bf16(gate @ up[b * H:(b + 1) * H].T)) * norm[:, b * H:(b + 1) * H]
    mixed = _bf16(acc / BR)
    L = S = None
    inj_out = None
    if inject:
        iw = np.asarray(x["inject"], np.float64)
        L, S = norm @ iw.T, np.abs(norm) @ np.abs(iw).T
        inj_out = 2.0 * _sig(L / BR)  # g64 = g(L), unrounded (the bound's reference)
    return dict(mixed=mixed, injection=inj_out, h=h, L=L, S=S)


def gate_delta(L, S, inflate=1.0):
    """INJECT-MXU's allowance (LOCKED prereg s4): m = max(|L|, RHO S); u one bf16 ULP at m; delta = u g'(max(|L| - 2u, 0))."""
    m = np.maximum(np.abs(L), RHO * S) * inflate
    e = np.floor(np.log2(np.where(m > 0, m, 1.0)))
    u = np.where(m > 0, np.exp2(np.maximum(e, -126.0) - 7.0), 2.0 ** -133)
    s = _sig(np.maximum(np.abs(L) - 2.0 * u, 0.0) / 4.0)
    return u * s * (1.0 - s) / 2.0


def check_bf16(actual, expected) -> dict:
    """_check_bf16 of tests/kernels/test_qwen4_exp_hc_decode_cmp.py (the served kernel's tolerance), as a record."""
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    d = np.abs(a - e)
    scale = float(np.abs(e).max())
    rec = {"finite": bool(np.isfinite(a).all()), "max_diff": float(d.max()), "scale": scale,
           "l2_rel": float(np.linalg.norm(d) / max(np.linalg.norm(e), 1e-30)), "diff_frac": float((d > 0).mean())}
    rec["ok"] = bool(rec["finite"] and rec["max_diff"] <= 2.0 ** -7 * scale and rec["l2_rel"] <= 1e-3
                     and rec["diff_frac"] <= 0.02)
    return rec


def rel_l2(a, ref):
    return float(np.linalg.norm(np.asarray(a, np.float64) - ref) / max(np.linalg.norm(ref), 1e-30))


def is_bf16_gate(g, tol=4 * 2.0 ** -24) -> bool:
    """Every gate value equals g(b) = 2 sigmoid(b / 4) for some bf16 b (within tol): the bf16 rounding of L is in the chain.
    b is recovered as 4 log(g / (2 - g)) and the nearest bf16 and its two neighbours are tried."""
    import ml_dtypes
    g = np.asarray(g, np.float64)
    inner = (g > 0) & (g < 2)
    b = np.where(inner, BR * np.log(np.where(inner, g, 1.0) / np.where(inner, 2.0 - g, 1.0)), 0.0)
    bits16 = b.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int64)
    best = np.full(g.shape, np.inf)
    for step in (-1, 0, 1):
        cand = np.clip(bits16 + step, 0, 0xFFFF).astype(np.uint16).view(ml_dtypes.bfloat16).astype(np.float64)
        diff = np.abs(2.0 * _sig(cand / BR) - g)
        best = np.minimum(best, np.where(np.isfinite(diff), diff, np.inf))
    return bool(np.where(inner, best <= tol, True).all())


def bits(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32)))


def run_case(form, rows, kind, module_path=None) -> dict:
    import importlib.util

    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import hc_decode as hd
    if module_path:
        spec = importlib.util.spec_from_file_location("hc_decode_wide_mutant", module_path)
        hw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw)
    else:
        from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide as hw
    prologue, inject = form in ("combine", "final"), form in ("combine", "mix")
    seed = 7 * rows + 1 + KINDS.index(kind)
    x = inputs(rows, kind, seed)
    bf = lambda k: jnp.asarray(x[k]).astype(jnp.bfloat16)  # noqa: E731
    args = [bf("residual"), bf("weight"), bf("down"), bf("up"), bf("inject") if inject else None]
    st = dict(branches=BR, hidden=H, eps=EPS, interpret=pltpu.InterpretParams(uninitialized_memory="nan"))
    if prologue:
        st.update(block=bf("block"), prev_injection=jnp.asarray(x["injection"], jnp.float32))
    f64 = lambda o: None if o is None else np.asarray(jnp.asarray(o, jnp.float32), np.float64)  # noqa: E731
    outs = {}
    for arm, call in (("S", lambda: hd.hc_decode_mix(*args, max_tile=512, **st)),
                      ("N1", lambda: hw.hc_decode_mix_wide(*args, mode=1, **st)),
                      ("N2", lambda: hw.hc_decode_mix_wide(*args, mode=2, **st))):
        o = jax.block_until_ready(call())
        o = list(o) + [None] * (3 - len(o))
        if not prologue:
            o = [o[0], o[1], None]
        outs[arm] = dict(mixed=f64(o[0]), injection=f64(o[1]), h=f64(o[2]))
    ref = reference(x, prologue, inject)
    S, N1, N2 = outs["S"], outs["N1"], outs["N2"]
    names = ["mixed"] + (["injection"] if inject else []) + (["h"] if prologue else [])
    rec = {"form": form, "rows": rows, "kind": kind, "seed": seed, "mutant": module_path,
           "xla_flags": os.environ.get("XLA_FLAGS", ""), "excess_precision_off": FLAG in os.environ.get("XLA_FLAGS", "").split()}
    # N1
    n1 = {"bitwise": {k: bits(N1[k], S[k]) for k in names}, "finite": all(np.isfinite(N1[k]).all() for k in names)}
    n1["mixed_tol"] = check_bf16(N1["mixed"], S["mixed"])
    n1["inj_tol"] = bool(np.allclose(N1["injection"], S["injection"], rtol=2e-2, atol=1e-6)) if inject else None
    n1["h_bitwise"] = n1["bitwise"].get("h") if prologue else None
    n1["gate_ok"] = bool(n1["finite"] and n1["mixed_tol"]["ok"] and (n1["inj_tol"] is not False)
                         and (n1["h_bitwise"] is not False))
    n1["all_bitwise"] = all(n1["bitwise"].values())
    rec["n1"] = n1
    # N2
    n2 = {"finite": all(np.isfinite(N2[k]).all() for k in names), "mixed_tol": check_bf16(N2["mixed"], S["mixed"]),
          "mixed_l2": {"n2": rel_l2(N2["mixed"], ref["mixed"]), "s": rel_l2(S["mixed"], ref["mixed"])}}
    n2["mixed_l2"]["ok"] = n2["mixed_l2"]["n2"] <= 2.0 * n2["mixed_l2"]["s"] + 1e-5
    n2["h_bitwise"] = bits(N2["h"], S["h"]) if prologue else None
    if inject:
        g64, delta = ref["injection"], gate_delta(ref["L"], ref["S"])
        es, e2 = np.abs(S["injection"] - g64), np.abs(N2["injection"] - g64)
        s_ok = es <= delta + EPS_G
        e_ok = e2 <= es + delta + EPS_G
        n2["inj"] = {"served_ref_ok": bool(s_ok.all()), "served_ref_over": int((~s_ok).sum()), "ok": bool(e_ok.all()),
                     "n_over": int((~e_ok).sum()), "max_excess": float((e2 - es - delta - EPS_G).max()),
                     "n_diff_vs_s": int((N2["injection"] != S["injection"]).sum()), "n": int(g64.size),
                     "max_abs_vs_s": float(np.abs(N2["injection"] - S["injection"]).max()),
                     "bf16_chain": is_bf16_gate(N2["injection"]), "bf16_chain_s": is_bf16_gate(S["injection"])}
    n2["gate_ok"] = bool(n2["finite"] and n2["mixed_tol"]["ok"] and n2["mixed_l2"]["ok"] and (n2["h_bitwise"] is not False)
                         and (not inject or (n2["inj"]["served_ref_ok"] and n2["inj"]["ok"] and n2["inj"]["bf16_chain"])))
    rec["n2"] = n2
    rec["s_finite"] = all(np.isfinite(S[k]).all() for k in names)
    return rec


def selftest() -> int:
    bad = 0
    g1 = 2.0 ** -7 * _sig((1 - 2.0 ** -6) / 4) * (1 - _sig((1 - 2.0 ** -6) / 4)) / 2
    d = gate_delta(np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0 ** 11, 0.0]))
    ok = abs(d[0] - g1) < 1e-18 and abs(d[1] - 2.0 ** -7 * 0.125) < 1e-18 and d[2] < 1e-39
    print("HCWNUM SELFTEST gate_delta", "ok" if ok else "BAD", list(d))
    bad += not ok
    a = np.array([1.0, 2.0, 3.0], np.float32)
    b = a.copy()
    b[1] = np.nextafter(np.float32(2.0), np.float32(3.0))
    ok = bits(a, a) and not bits(a, b)
    print("HCWNUM SELFTEST bits (one-ulp f32 change reads not equal)", "ok" if ok else "BAD")
    bad += not ok
    import ml_dtypes
    bv = np.array([0.5, -3.0, 0.0, 7.25], np.float64).astype(ml_dtypes.bfloat16).astype(np.float64)
    g = 2.0 * _sig(bv / BR)
    ok = is_bf16_gate(g) and not is_bf16_gate(2.0 * _sig((bv + 0.003) / BR))
    print("HCWNUM SELFTEST bf16-chain detector", "ok" if ok else "BAD")
    bad += not ok
    x = np.random.default_rng(0).standard_normal((64, 2560))
    ok = check_bf16(x, x)["ok"] and not check_bf16(x * 1.02, x)["ok"]
    print("HCWNUM SELFTEST check_bf16", "ok" if ok else "BAD")
    bad += not ok
    print("HCWNUM SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--out")
    ap.add_argument("--module")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert FLAG in os.environ.get("XLA_FLAGS", "").split(), "excess precision must be off (rule 23)"
    form, rows, kind = a.case.split(":")
    assert form in FORMS and kind in KINDS
    rec = run_case(form, int(rows), kind, a.module)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print("HCWNUM CASE " + json.dumps({k: rec[k] for k in ("form", "rows", "kind", "mutant")} | {
        "n1_gate": rec["n1"]["gate_ok"], "n1_bitwise": rec["n1"]["all_bitwise"], "n2_gate": rec["n2"]["gate_ok"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

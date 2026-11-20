# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS microbench (2026-11-20): the decode HC combine-mix kernel at 160-496 rows.

Harness copied from q38 DECODE-L3's HCBIND bench (tests/q38_decl3/tpu_bench_hcbind.py on q38-decl3-hcbind 0626b2461, itself
from HCDEC-XLAHUNT's K3DX bench; credit): the libtpu launch-barrier rule, engagement from the compiled HLO, long-minus-short
chain timing on all 8 TensorCores at once (both cores of a chip stream their weights together, as served), every call with its
own replicated HC weights, call k's combined residual / mixed output / injection gate feeding call k + 1.

Arms (combine prologue + injection gate: the 95-calls-a-step form):
  S512  served hc_decode.hc_decode_mix, TPU_QWEN4_HC_DECODE_TILE_ROWS=512 (one tile: the served S3 / B2 form)  [qwen4exp_hc_decode_combine]
  S256  served hc_decode.hc_decode_mix, max_tile 256 (env-only alternative: two tiles above 256 rows)            [qwen4exp_hc_decode_combine]
  W1    hc_decode_wide mode 1 (served order; bitwise target)                                                     [qwen4exp_hc_decode_combine_w1]
  W2    hc_decode_wide mode 2 (cmp math; NUMERICS-CHANGING)                                                      [qwen4exp_hc_decode_combine_w2]
  WD    hc_decode_wide mode 1, timing "dma" (its copies, no compute)                                             [qwen4exp_hc_decode_combine_w1_dma]
  W1C   hc_decode_wide mode 1, timing "compute" (its compute, no copies)                                         [qwen4exp_hc_decode_combine_w1_compute]
  W2C   hc_decode_wide mode 2, timing "compute"                                                                  [qwen4exp_hc_decode_combine_w2_compute]
  CMP   Benliang Wang's hc_decode_cmp (16-256 rows only; descriptive reference)                                  [qwen4exp_hc_decode_combine_cmp]
Order per row: base (S512 alone), then every arm interleaved in a rotating order, then base2 (S512 alone).
Per-call us = (median long - median short) / (L_LONG - L_SHORT); split halves as a noise figure.

Health, one call per arm on real inputs before any timing (a failure aborts the run): every arm that computes is finite and
equal on all 8 cores. Numerics records (they decide nothing in the bench; the reader applies the pre-stated rules):
  n1_bitwise   W1 == S512 bit for bit on mixed, injection and h (mode 1's device decision)
  n2           W2 against S512 under the locked device bound (injection |g_W2 - g_S| <= delta + 2 EPS_G, L and S from a host
               float64 recompute of the norm from S512's h, m inflated by 1 + 2^-6; mixed within _check_bf16; h bitwise)
  s256_bitwise S256 == S512 bit for bit; cmp_tol CMP within _check_bf16 of S512
Modes: (default) TPU timing; --aot deviceless tpu7x:2x2x1 compile + engagement of every long program; --interpret CPU harness
(8 host devices, flash geometry, one call per arm); --selftest [--mutants] / --barrier-selftest (no devices).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time

try:
    import numpy as np
except ImportError:  # selftests run without numpy; every device mode needs it
    np = None

N = 8
LB = "--xla_tpu_use_enhanced_launch_barrier"
LB_OFF = LB + "=false"
RT = "--xla_enable_custom_call_region_trace"
VMEM_64MIB = 67108864
CFG_ADD = [f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={VMEM_64MIB}",
           f"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes={VMEM_64MIB}",
           "--xla_tpu_use_dynamic_smem_negotiation=true"]
ROWS = (160, 192, 240, 288, 384, 448, 496)
ARMS = ("S512", "S256", "W1", "W2", "WD", "W1C", "W2C", "CMP")
CMP_MAX = 256
L_LONG, L_SHORT = 24, 4
GEO = dict(HC=4, H=2560, LR=320, EPS=1e-6)
SERVED = "qwen4exp_hc_decode_combine"
W1N, W2N = SERVED + "_w1", SERVED + "_w2"
WDN, W1CN, W2CN = SERVED + "_w1_dma", SERVED + "_w1_compute", SERVED + "_w2_compute"
CMPN = SERVED + "_cmp"
# Longest first: every other name starts with SERVED.
_KERNELS = tuple(sorted((SERVED, W1N, W2N, WDN, W1CN, W2CN, CMPN), key=len, reverse=True))
ARM_KERNEL = {"S512": SERVED, "S256": SERVED, "W1": W1N, "W2": W2N, "WD": WDN, "W1C": W1CN, "W2C": W2CN, "CMP": CMPN}
RELAYOUT_RE = re.compile(r"=\s*bf16\[(?:10240,320|320,10240)\]\{[^}]*\}\s+(?:copy|transpose|fusion|copy-start)\(")
EPS_G = 2.0 ** -20
RHO = 2.0 ** -11


def arm_rows(arm, rows):
    """Whether an arm runs at a row count (CMP only up to 256 rows)."""
    return arm != "CMP" or rows <= CMP_MAX


# ---------------------------------------------------------------- libtpu (ARB1 / K3DX / HCBIND rule, copied)
def check_barrier(final: str) -> None:
    toks = [t for t in final.split() if t.split("=")[0] == LB]
    if toks != [LB_OFF]:
        raise SystemExit(f"Q38_BARRIER_ASSERT FAIL: launch-barrier tokens {toks} (want exactly ['{LB_OFF}'])")


def apply_cfg() -> str:
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    names = {a.split("=")[0] for a in CFG_ADD} | {LB, RT}
    kept = [t for t in base.split() if t.split("=")[0] not in names]
    final = " ".join(kept + CFG_ADD + [LB_OFF]).strip()
    check_barrier(final)
    os.environ["LIBTPU_INIT_ARGS"] = final
    return final


def barrier_selftest() -> int:
    bad = 0
    saved = os.environ.get("LIBTPU_INIT_ARGS")
    for base in ("--a=1 " + LB + "=true --b=2", LB + "=true " + LB + "=true", "--a=1", LB_OFF, "", LB, RT + "=true"):
        os.environ["LIBTPU_INIT_ARGS"] = base
        try:
            final = apply_cfg()
            ok = [t for t in final.split() if t.split("=")[0] == LB] == [LB_OFF]
            ok &= not any(t.split("=")[0] == RT for t in final.split())
            ok &= all(a in final.split() for a in CFG_ADD)
        except SystemExit:
            ok = False
        bad += not ok
    for final, want in ((LB + "=true", False), (LB_OFF + " " + LB_OFF, False), ("--a=1", False),
                        (LB_OFF + " " + LB + "=true", False), (LB, False), ("--a=1 " + LB_OFF, True)):
        try:
            check_barrier(final)
            got = True
        except SystemExit:
            got = False
        print(f"Q38_BARRIER_SELFTEST check_barrier({final!r}) -> {'accept' if got else 'refuse'} "
              f"(want {'accept' if want else 'refuse'})")
        bad += got != want
    if saved is None:
        os.environ.pop("LIBTPU_INIT_ARGS", None)
    else:
        os.environ["LIBTPU_INIT_ARGS"] = saved
    print("Q38_BARRIER_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


# ---------------------------------------------------------------- engagement (compiled HLO text)
def call_kernel(ln: str):
    """The kernel a custom-call line IS: the instruction name before ' = ' (Mosaic calls are named after the kernel), else the
    last component of its metadata op_name; longest name first. An operand that is another kernel's output never counts."""
    head = ln.split(" = ", 1)[0].strip()
    head = head[5:].strip() if head.startswith("ROOT ") else head
    head = head.lstrip("%")
    for k in _KERNELS:
        if head.startswith(k) and (len(head) == len(k) or not head[len(k)].isalpha() and head[len(k)] != "_"):
            return k
    m = re.search(r'op_name="([^"]*)"', ln)
    last = m.group(1).rsplit("/", 1)[-1] if m else ""
    for k in _KERNELS:
        if last == k or (last.startswith(k) and not last[len(k)].isalpha() and last[len(k)] != "_"):
            return k
    return None


def kernel_counts(txt: str) -> dict:
    lines = [ln for ln in txt.splitlines() if "custom-call(" in ln]
    kinds = [call_kernel(ln) for ln in lines]
    out = {k: kinds.count(k) for k in _KERNELS}
    out["up_relayout"] = len(RELAYOUT_RE.findall(txt))
    return out


def engaged(arm: str, cnt: dict, layers: int) -> bool:
    want = ARM_KERNEL[arm]
    ok = all(cnt.get(k, -1) == (layers if k == want else 0) for k in _KERNELS)
    ok &= cnt.get("up_relayout", -1) == 0
    return bool(ok)


# ---------------------------------------------------------------- programs
def _stub_vllm_logger():
    import importlib.util
    import logging
    import types
    if importlib.util.find_spec("vllm") is not None:
        return
    vllm = types.ModuleType("vllm")
    vlog = types.ModuleType("vllm.logger")
    vlog.init_logger = logging.getLogger
    vllm.logger = vlog
    sys.modules["vllm"], sys.modules["vllm.logger"] = vllm, vlog


def one_call(arm, prev, block, pinj, p, geo, interpret=False):
    """One HC call of an arm -> (mixed, injection, h)."""
    from vllm_torchtpu.kernels.qwen4_exp import hc_decode as hd
    from vllm_torchtpu.kernels.qwen4_exp import hc_decode_cmp as hcmp
    from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide as hw

    st = dict(branches=geo["HC"], hidden=geo["H"], eps=geo["EPS"], block=block, prev_injection=pinj)
    if interpret:
        from jax.experimental.pallas import tpu as pltpu
        st["interpret"] = pltpu.InterpretParams()
    args = (prev, p["w"], p["down"], p["up"], p["inject"])
    if arm in ("S512", "S256"):
        return hd.hc_decode_mix(*args, max_tile=512 if arm == "S512" else 256, **st)
    if arm == "CMP":
        return hcmp.hc_decode_mix_cmp(*args, **st)
    mode = {"W1": 1, "W2": 2, "WD": 1, "W1C": 1, "W2C": 2}[arm]
    timing = {"WD": "dma", "W1C": "compute", "W2C": "compute"}.get(arm, "full")
    return hw.hc_decode_mix_wide(*args, mode=mode, timing=timing, **st)


def make_chain(arm, geo, layers, interpret=False):
    def chain(prev, block, pinj, ws):
        for li in range(layers):
            m, i, h = one_call(arm, prev, block, pinj, ws[li], geo, interpret)
            prev, block, pinj = h, m, i
        return prev, block
    return chain


def build(arm, geo, mesh, layers, interpret=False):
    import jax
    from jax.sharding import PartitionSpec as P

    chain = make_chain(arm, geo, layers, interpret)

    def body(prev, block, pinj, ws):
        h, m = chain(prev[0], block[0], pinj[0], ws)
        return h[None], m[None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"), P("d"), P("d"), P()), out_specs=(P("d"), P("d")),
                                 check_vma=False))


def build_single(arm, geo, mesh, interpret=False):
    import jax
    from jax.sharding import PartitionSpec as P

    def body(prev, block, pinj, p):
        m, i, h = one_call(arm, prev[0], block[0], pinj[0], p, geo, interpret)
        return m[None], i[None], h[None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"), P("d"), P("d"), P()),
                                 out_specs=(P("d"), P("d"), P("d")), check_vma=False))


def weights(mesh, geo, layers, seed=0):
    """Per call HC weights (distinct per call), replicated on every core, generated on device; up is [hc * hidden, lowrank]
    as the module holds it (the served row-major layout; the engagement rule requires 0 up relayouts)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    HC, H, LR = geo["HC"], geo["H"], geo["LR"]
    W = HC * H

    def make(key):
        out = []
        for li in range(layers):
            k = jax.random.split(jax.random.fold_in(key, li), 4)
            out.append(dict(w=(jax.random.normal(k[0], (W,)) * 0.3).astype(jnp.bfloat16),
                            down=(jax.random.normal(k[1], (LR, W)) * 0.02).astype(jnp.bfloat16),
                            up=(jax.random.normal(k[2], (W, LR)) * 0.05).astype(jnp.bfloat16),
                            inject=(jax.random.normal(k[3], (HC, W)) * 0.02).astype(jnp.bfloat16)))
        return out

    rep = NamedSharding(mesh, P())
    shapes = jax.eval_shape(make, jax.random.PRNGKey(seed))
    return jax.jit(make, out_shardings=jax.tree.map(lambda _: rep, shapes))(jax.random.PRNGKey(seed + 2026))


def weight_specs(mesh, geo, layers):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    HC, H, LR = geo["HC"], geo["H"], geo["LR"]
    W = HC * H
    rep = NamedSharding(mesh, P())

    def s(shape):
        return jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=rep)

    return [dict(w=s((W,)), down=s((LR, W)), up=s((W, LR)), inject=s((HC, W))) for _ in range(layers)]


def inputs(mesh, geo, rows, seed):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    HC, H = geo["HC"], geo["H"]
    W = HC * H
    xs = NamedSharding(mesh, P("d"))
    rng = np.random.default_rng([seed, rows])
    prev = rng.standard_normal((rows, W))
    block = rng.standard_normal((rows, H))
    pinj = rng.uniform(0.2, 1.8, (rows, HC))
    rep = lambda a, dt: jax.device_put(jnp.asarray(np.broadcast_to(a, (N,) + a.shape), dt), xs)  # noqa: E731
    return rep(prev, jnp.bfloat16), rep(block, jnp.bfloat16), rep(pinj, jnp.float32)


def input_specs(mesh, geo, rows):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    HC, H = geo["HC"], geo["H"]
    xs = NamedSharding(mesh, P("d"))
    return (jax.ShapeDtypeStruct((N, rows, HC * H), jnp.bfloat16, sharding=xs),
            jax.ShapeDtypeStruct((N, rows, H), jnp.bfloat16, sharding=xs),
            jax.ShapeDtypeStruct((N, rows, HC), jnp.float32, sharding=xs))


# ---------------------------------------------------------------- numerics (host float64; the locked device forms)
def _bf16(a):
    import ml_dtypes
    return np.asarray(a, np.float64).astype(ml_dtypes.bfloat16).astype(np.float64)


def _sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def check_bf16(actual, expected) -> dict:
    """_check_bf16 of tests/kernels/test_qwen4_exp_hc_decode_cmp.py (the served kernel's tolerance), as a record."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    d = np.abs(actual - expected)
    scale = float(np.abs(expected).max())
    rec = {"finite": bool(np.isfinite(actual).all()), "max_diff": float(d.max()), "scale": scale,
           "l2_rel": float(np.linalg.norm(d) / max(np.linalg.norm(expected), 1e-30)), "diff_frac": float((d > 0).mean())}
    rec["ok"] = bool(rec["finite"] and rec["max_diff"] <= 2.0 ** -7 * scale and rec["l2_rel"] <= 1e-3
                     and rec["diff_frac"] <= 0.02)
    return rec


def gate_delta(L, S, inflate=1.0):
    """INJECT-MXU's allowance (prereg q38_decl3_injmxu s6.2, accepted for N2): m = max(|L|, RHO S) (x inflate), u = one bf16
    ULP at m, delta = u g'(max(|L| - 2u, 0)), g(x) = 2 sigmoid(x / 4), g'(x) = sigmoid(x/4)(1 - sigmoid(x/4)) / 2."""
    m = np.maximum(np.abs(L), RHO * S) * inflate
    e = np.where(m > 0, np.floor(np.log2(np.where(m > 0, m, 1.0))), -133.0 + 7)
    u = np.where(m > 0, np.exp2(np.maximum(e, -126.0) - 7.0), 2.0 ** -133)
    x = np.maximum(np.abs(L) - 2.0 * u, 0.0)
    s = _sig(x / 4.0)
    return u * s * (1.0 - s) / 2.0


def device_n2(h_s, g_s, g_e, w, inj, geo) -> dict:
    """The locked device form for the injection gate: |g_E - g_A| <= delta + 2 EPS_G, L and S from a host float64 recompute of
    the bf16 norm from the served kernel's h, m inflated by 1 + 2^-6."""
    HC, H, EPS = geo["HC"], geo["H"], geo["EPS"]
    h = np.asarray(h_s, np.float64)
    w = np.asarray(w, np.float64)
    inj = np.asarray(inj, np.float64)
    norm = []
    for b in range(HC):
        v = h[:, b * H:(b + 1) * H]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + w[b * H:(b + 1) * H])))
    norm = np.concatenate(norm, axis=1)
    L = norm @ inj.T
    S = np.abs(norm) @ np.abs(inj).T
    allow = gate_delta(L, S, 1.0 + 2.0 ** -6) + 2.0 * EPS_G
    d = np.abs(np.asarray(g_e, np.float64) - np.asarray(g_s, np.float64))
    return {"ok": bool((d <= allow).all()), "max_diff": float(d.max()), "n_diff": int((d > 0).sum()),
            "n_over": int((d > allow).sum()), "max_excess": float((d - allow).max())}


def health(mesh, geo, rows, ws, x, arms, interpret=False) -> dict:
    """One call per arm on real inputs: every computing arm finite and equal on all 8 cores (gate); numerics records."""
    import jax
    prev, block, pinj = x
    out = {}
    for arm in arms:
        if arm in ("WD", "W1C", "W2C") or not arm_rows(arm, rows):
            continue
        f = build_single(arm, geo, mesh, interpret)
        o = jax.device_get(f(prev, block, pinj, ws[0]))
        out[arm] = [np.asarray(np.asarray(a).astype(np.float32), np.float64) for a in o]
    rep = {"finite": {a: bool(all(np.isfinite(v).all() for v in o)) for a, o in out.items()},
           "cores_equal": {a: bool(all(np.array_equal(v[0], v[i]) for v in o for i in range(N))) for a, o in out.items()}}
    rep["ok"] = all(rep["finite"].values()) and all(rep["cores_equal"].values()) and "S512" in out
    s = out.get("S512")
    if s is not None:
        bit = lambda a: [bool(np.array_equal(u[0], v[0])) for u, v in zip(out[a], s)]  # noqa: E731
        if "W1" in out:
            rep["n1_bitwise"] = bit("W1")
        if "S256" in out:
            rep["s256_bitwise"] = bit("S256")
        if "W2" in out:
            p0 = jax.device_get(ws[0])
            rep["n2"] = {"inj": device_n2(s[2][0], s[1][0], out["W2"][1][0], p0["w"].astype(np.float32),
                                          p0["inject"].astype(np.float32), geo),
                         "mixed": check_bf16(out["W2"][0][0], s[0][0]), "h_bitwise": bool(np.array_equal(out["W2"][2][0], s[2][0]))}
            rep["n2"]["ok"] = rep["n2"]["inj"]["ok"] and rep["n2"]["mixed"]["ok"] and rep["n2"]["h_bitwise"]
        if "CMP" in out:
            rep["cmp_tol"] = check_bf16(out["CMP"][0][0], s[0][0])
    return rep


def per_call(long_s, short_s, cl, cs) -> dict:
    ml, ms = statistics.median(long_s), statistics.median(short_s)
    n = len(long_s) // 2
    halves = [None, None]
    if n >= 2:
        f = lambda a, b: round((statistics.median(a) - statistics.median(b)) / (cl - cs), 3)  # noqa: E731
        halves = [f(long_s[:n], short_s[:n]), f(long_s[n:], short_s[n:])]
    return {"us_per_call": round((ml - ms) / (cl - cs), 3), "halves": halves, "long_median_us": round(ml, 2),
            "short_median_us": round(ms, 2), "samples_long_us": [round(v, 2) for v in long_s],
            "samples_short_us": [round(v, 2) for v in short_s]}


# ---------------------------------------------------------------- selftest (no devices)
def _cc(name, rows=448):
    res = f"(bf16[{rows},2560]{{1,0}}, f32[{rows},4]{{1,0}}, bf16[{rows},10240]{{1,0}})"
    return f"  %{name}.7 = {res} custom-call(%p0, %p1), custom_call_target=\"tpu_custom_call\""


def selftest() -> int:
    L = 3
    relay = "  %copy.3 = bf16[10240,320]{0,1} copy(%p5)"
    feeding = "  %fusion.2 = bf16[448,2560]{1,0} fusion(%" + W1N + ".7), kind=kLoop"
    cases = []
    for arm in ARMS:
        k = ARM_KERNEL[arm]
        cases.append((f"{arm} engaged", arm, [_cc(k)] * L, True))
        cases.append((f"{arm} with one call missing", arm, [_cc(k)] * (L - 1), False))
        cases.append((f"{arm} with an up relayout copy", arm, [_cc(k)] * L + [relay], False))
        other = SERVED if k != SERVED else W1N
        cases.append((f"{arm} with a stray {other} call", arm, [_cc(k)] * L + [_cc(other)], False))
    cases += [
        ("W1 with a feeding fusion naming the kernel still engaged (operands never count)", "W1", [_cc(W1N)] * L + [feeding], True),
        ("W1 with an unnamed custom call fed by the kernel still engaged (operands never count)", "W1",
         [_cc(W1N)] * L + ["  %custom-call.9 = (bf16[448,2560]{1,0}) custom-call(%" + W1N + ".7), custom_call_target=\"x\""], True),
        ("S512 with W1-named calls is not engaged (prefix rule)", "S512", [_cc(W1N)] * L, False),
        ("W1 with W1C-named calls is not engaged (prefix rule)", "W1", [_cc(W1CN)] * L, False),
        ("W2 with W2C-named calls is not engaged (prefix rule)", "W2", [_cc(W2CN)] * L, False),
        ("metadata op_name fallback counts", "WD", ["  %custom-call.1 = (bf16[448,2560]{1,0}) custom-call(%a), "
                                                    f"metadata={{op_name=\"jit(f)/{WDN}\"}}"] * L, True),
    ]
    bad = 0
    for label, arm, lines, want in cases:
        got = engaged(arm, kernel_counts("\n".join(lines)), L)
        print(f"Q38_HCW_SELFTEST {'ok ' if got == want else 'BAD'} {label}: got {got} want {want}")
        bad += got != want
    # The locked allowance: one bf16 ULP of L at |L| = 1 is 2^-7; delta = 2^-7 g'(1 - 2^-6); RHO floor; m = 0.
    g1 = 2.0 ** -7 * _sig((1 - 2.0 ** -6) / 4) * (1 - _sig((1 - 2.0 ** -6) / 4)) / 2
    d = [float(v) for v in gate_delta(np.array([1.0, 0.0, 1e-9, 3.0]), np.array([1.0, 2.0 ** 11, 1.0, 0.0]))]
    ok_g = (abs(d[0] - g1) < 1e-15 and abs(d[1] - 2.0 ** -7 * 0.125) < 1e-15
            and d[2] > 0 and abs(d[3] - 2.0 ** -6 * _sig((3 - 2 ** -5) / 4) * (1 - _sig((3 - 2 ** -5) / 4)) / 2) < 1e-15)
    print(f"Q38_HCW_SELFTEST {'ok ' if ok_g else 'BAD'} gate_delta {d}")
    bad += not ok_g
    cfg_ok = ROWS == (160, 192, 240, 288, 384, 448, 496) and ARMS == ("S512", "S256", "W1", "W2", "WD", "W1C", "W2C", "CMP") \
        and (L_LONG, L_SHORT) == (24, 4) and GEO == dict(HC=4, H=2560, LR=320, EPS=1e-6) and CMP_MAX == 256 \
        and (EPS_G, RHO) == (2.0 ** -20, 2.0 ** -11)
    print(f"Q38_HCW_SELFTEST {'ok ' if cfg_ok else 'BAD'} plan rows {ROWS} arms {ARMS} chain {L_LONG}/{L_SHORT} geo {GEO}")
    bad += not cfg_ok
    print("Q38_HCW_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants() -> int:
    """Each mutant weakens one rule; the selftest must fail under every one."""
    g = globals()
    orig = {k: g[k] for k in ("ARM_KERNEL", "call_kernel", "RELAYOUT_RE", "_KERNELS", "gate_delta")}
    muts = {
        "S512 rule names the W1 kernel": ("ARM_KERNEL", {**orig["ARM_KERNEL"], "S512": W1N}),
        "kernel name read from any operand": ("call_kernel", lambda ln: next((k for k in orig["_KERNELS"] if k in ln), None)),
        "plain prefix match, shortest name first": ("call_kernel", lambda ln: next(
            (k for k in sorted(orig["_KERNELS"], key=len) if ln.split(" = ", 1)[0].strip().lstrip("%").startswith(k)), None)),
        "relayout check blind": ("RELAYOUT_RE", re.compile(r"$^")),
        "gate allowance doubled": ("gate_delta", lambda L, S, inflate=1.0: 2 * orig["gate_delta"](L, S, inflate)),
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
        print(f"Q38_HCW_MUTANT {'killed  ' if caught else 'SURVIVED'} {name}")
        bad += not caught
    print("Q38_HCW_MUTANTS", "PASS" if not bad else f"FAIL ({bad} survived)", f"({len(muts)} mutants)")
    return 0 if not bad else 1


# ---------------------------------------------------------------- main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--barrier-selftest"]:
        return barrier_selftest()
    if argv[:1] == ["--selftest"]:
        return selftest() | (mutants() if "--mutants" in argv else 0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="row counts (default 160,192,240,288,384,448,496)")
    ap.add_argument("--arms", help="subset of " + ",".join(ARMS))
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hlo-dir")
    ap.add_argument("--aot", action="store_true", help="deviceless tpu7x:2x2x1 compile + engagement, no timing")
    ap.add_argument("--interpret", action="store_true", help="CPU harness check (8 host devices, one call per arm)")
    ap.add_argument("--l-long", type=int, default=L_LONG)
    ap.add_argument("--l-short", type=int, default=L_SHORT)
    a = ap.parse_args(argv)
    inherited = os.environ.get("LIBTPU_INIT_ARGS", "")
    final = apply_cfg()
    _stub_vllm_logger()
    import vllm_torchtpu  # noqa: F401  before jax and before any client exists (ARB1 rule)
    import jax
    from jax.sharding import Mesh

    at_client = os.environ.get("LIBTPU_INIT_ARGS", "")
    check_barrier(at_client)
    want_src = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "vllm_torchtpu"))
    got_src = os.path.realpath(os.path.dirname(vllm_torchtpu.__file__))
    if got_src != want_src:
        raise SystemExit(f"Q38_SRC_ASSERT FAIL: vllm_torchtpu from {got_src}, want this tree's {want_src}")

    geo = dict(GEO)
    rows_l = [int(r) for r in a.rows.split(",")] if a.rows else list(ROWS)
    arms = [x for x in ARMS if not a.arms or x in a.arms.split(",")]
    if a.aot:
        from jax.experimental import topologies
        devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    else:
        devs = jax.devices()  # the client is created here, after apply_cfg and the vllm_torchtpu import
    assert len(devs) >= N, f"need {N} devices, got {len(devs)}"
    mesh = Mesh(np.array(sorted(devs, key=lambda d: d.id)[:N]), ("d",))
    LL, LS = a.l_long, a.l_short
    res = {"host": {"node": platform.node(), "jax": jax.__version__, "kind": devs[0].device_kind, "aot": a.aot,
                    "interpret": a.interpret, "libtpu_inherited": inherited, "libtpu_final": final,
                    "libtpu_at_client": at_client, "argv": sys.argv, "iters": a.iters, "chain": [LL, LS], "geo": geo,
                    "commit": _commit(), "vllm_torchtpu_file": vllm_torchtpu.__file__},
           "plan": {"rows": rows_l, "arms": arms, "order": "base(S512), all arms rotating, base2(S512)"},
           "cases": {}, "health": {}, "engagement": {}, "errors": [], "ok": False}
    print(f"Q38_LIBTPU_FINAL cfg=hcwide {final}", flush=True)
    print(f"Q38_LIBTPU_AT_CLIENT cfg=hcwide {at_client}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.hlo_dir:
        os.makedirs(a.hlo_dir, exist_ok=True)

    def dump():
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(a.out + ".tmp", a.out)

    def compile_long(arm, rows, specs_or_x, ws_or_spec):
        b = build(arm, geo, mesh, LL, a.interpret)
        comp = b.lower(*specs_or_x, ws_or_spec).compile()
        txt = comp.as_text()
        cnt = None if a.interpret else kernel_counts(txt)
        eng = None if a.interpret else engaged(arm, cnt, LL)
        if a.hlo_dir and not a.interpret:
            with open(os.path.join(a.hlo_dir, f"{arm}_{rows}.hlo.txt"), "w") as f:
                f.write(txt)
        mem = None
        if not a.interpret:
            try:
                ma = comp.memory_analysis()
                mem = {k: int(getattr(ma, k, 0) or 0) for k in ("temp_size_in_bytes", "argument_size_in_bytes")}
            except Exception:  # noqa: BLE001
                mem = None
        return comp, {"compiled": True, "hlo": cnt, "engaged": eng, "mem": mem}

    if a.aot:
        bad = 0
        wspec = weight_specs(mesh, geo, LL)
        for rows in rows_l:
            for arm in arms:
                if not arm_rows(arm, rows):
                    continue
                t0 = time.time()
                try:
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        _, rec = compile_long(arm, rows, input_specs(mesh, geo, rows), wspec)
                    rec["compile_s"] = round(time.time() - t0, 1)
                    bad += not rec["engaged"]
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    rec = {"compiled": False, "error": f"{type(e).__name__}: {str(e)[:1500]}"}
                res["engagement"].setdefault(str(rows), {})[arm] = rec
                print(f"HCW AOT rows {rows} {arm}: {json.dumps(rec)[:700]}", flush=True)
        res["ok"] = bad == 0
        dump()
        print("HCW AOT", "PASS" if not bad else f"FAIL ({bad})", flush=True)
        return 0 if not bad else 1

    ws = weights(mesh, geo, LL, a.seed)
    for rows in rows_l:
        x = inputs(mesh, geo, rows, a.seed)
        try:
            h = health(mesh, geo, rows, ws, x, arms, a.interpret)
        except Exception as e:  # noqa: BLE001
            h = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:1500]}"}
        res["health"][str(rows)] = h
        print(f"HCW HEALTH rows {rows}: {json.dumps(h)}", flush=True)
        if not h.get("ok"):
            res["errors"].append({"rows": rows, "arm": "health", "error": json.dumps(h)[:1500]})
            dump()
            print("HCW ABORT: health failed; no timing", flush=True)
            return 1
        if a.interpret:
            res["cases"][str(rows)] = {}
            dump()
            continue
        progs = {}
        run_arms = [arm for arm in arms if arm_rows(arm, rows)]
        for arm in run_arms:
            t0 = time.time()
            try:
                cl, rec = compile_long(arm, rows, x, ws)
                cs = build(arm, geo, mesh, LS, a.interpret).lower(*x, ws[:LS]).compile()
                o = jax.device_get(cs(*x, ws[:LS]))
                rec["short_finite_fraction"] = float(np.mean([np.isfinite(np.asarray(v, np.float32)).mean() for v in o]))
                rec["compile_s"] = round(time.time() - t0, 1)
                progs[arm] = (cl, cs)
            except Exception as e:  # noqa: BLE001
                rec = {"compiled": False, "error": f"{type(e).__name__}: {str(e)[:1500]}"}
                res["errors"].append({"rows": rows, "arm": arm, "error": rec["error"]})
            res["engagement"].setdefault(str(rows), {})[arm] = rec
            print(f"HCW rows {rows} {arm}: {json.dumps({k: v for k, v in rec.items() if k != 'hlo'})} hlo {rec.get('hlo')}",
                  flush=True)
        ran = [arm for arm in run_arms if arm in progs]
        case = {}

        def time_arms(seq_arms, iters):
            ls, ss = {k: [] for k in seq_arms}, {k: [] for k in seq_arms}
            for it in range(iters):
                for arm in seq_arms[it % len(seq_arms):] + seq_arms[:it % len(seq_arms)]:
                    cl, cs = progs[arm]
                    t0 = time.perf_counter()
                    jax.block_until_ready(cl(*x, ws))
                    ls[arm].append((time.perf_counter() - t0) * 1e6)
                    t0 = time.perf_counter()
                    jax.block_until_ready(cs(*x, ws[:LS]))
                    ss[arm].append((time.perf_counter() - t0) * 1e6)
            return ls, ss

        try:
            for arm in ran:
                for _ in range(3):
                    jax.block_until_ready(progs[arm][0](*x, ws))
                    jax.block_until_ready(progs[arm][1](*x, ws[:LS]))
            if "S512" in ran:
                ls, ss = time_arms(["S512"], a.iters)
                case["base"] = per_call(ls["S512"], ss["S512"], LL, LS)
            ls, ss = time_arms(ran, a.iters)
            for arm in ran:
                case[arm] = per_call(ls[arm], ss[arm], LL, LS)
            if "S512" in ran:
                ls, ss = time_arms(["S512"], a.iters)
                case["base2"] = per_call(ls["S512"], ss["S512"], LL, LS)
            print(f"HCW rows {rows:3d}: " + " ".join(f"{k} {v['us_per_call']:.3f}" for k, v in case.items()) + " us/call",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            res["errors"].append({"rows": rows, "arm": "timing", "error": f"{type(e).__name__}: {str(e)[:1500]}"})
            print(f"HCW rows {rows} timing ERROR {type(e).__name__}: {str(e)[:600]}", flush=True)
        res["cases"][str(rows)] = case
        dump()
    if a.interpret:
        res["ok"] = not res["errors"]
    else:
        res["ok"] = not res["errors"] and all(
            len(res["cases"].get(str(r), {})) == len([x for x in arms if arm_rows(x, r)]) + (2 if "S512" in arms else 0)
            for r in rows_l)
    dump()
    print(f"HCW DONE ok={res['ok']} -> {a.out}", flush=True)
    return 0 if res["ok"] else 1


def _commit():
    try:
        return open("COMMIT").read().strip()
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())

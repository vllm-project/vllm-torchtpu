# SPDX-License-Identifier: Apache-2.0
"""(tpu_bench_injmb.py, q38 INJMXU-BUILD, 2026-11-21: HC-HIGHROWS-2's tpu_bench_w3.py (the T70i W1DC bench; credit) with the arms
W1 / W1F / W1T, the live negative control W1FM, and the bound v2 device leg's inputs recorded per row; generator
scratch/injmxu/car/mk_injmb_bench.py, counted edits B1-B12.)
q38 INJ-MXU microbench: the decode HC combine-mix kernel at 288 / 320 / 352 / 384 / 448 rows, W1 against INJ-MXU.

Harness unchanged from tpu_bench_w3.py: the libtpu launch-barrier rule (LIBTPU_INIT_ARGS PINNED to the served 2-flag decode set +
barrier off, no dvfs), engagement from the compiled HLO, long-minus-short chain timing (24 / 4 calls) on all 8 TensorCores at once,
every call with its own replicated HC weights, call k's combined residual / mixed output / injection gate feeding call k + 1.

Arms (combine prologue + injection gate: the 94-calls-a-step form):
  W1    hc_decode_wide mode 1 (TPU_QWEN4_HC_DECODE_WIDE=1, the served W1 order)                    [qwen4exp_hc_decode_combine_w1]
  W1F   hc_decode_wide mode 1, fold=True (TPU_QWEN4_HC_DECODE_INJECT_FOLD=1, INJ-MXU)              [qwen4exp_hc_decode_combine_w1f]
  W1T   W1 again, compiled and timed as its own arm (the A/A twin)                                  [qwen4exp_hc_decode_combine_w1]
  W1FM  (health only, never timed) the live negative control: W1F with bound v2 mutant M2 (logits from lane 320 + j + 1), the
        tree's hc_decode_wide.py with one counted edit loaded from a temp copy; the reader requires it to FAIL the device leg
Order per row: base (W1 alone), then W1 / W1F / W1T interleaved in a rotating order, then base2 (W1 alone).
Per-call us = (median long - median short) / (L_LONG - L_SHORT); split halves as a noise figure.

Health, one call per arm on real inputs before any timing (a failure aborts the run): every arm finite and equal on all 8 cores and
W1T bitwise W1. Recorded for the reader (reports/prereg/q38_injmxu_bound_v2_LOCKED.md s5, judged by q38_injmxu/injmb_read.py):
E = W1F vs A = W1 mixed and h bitwise, both injection gates, the negative control's gates, and L, S (float64, uninflated) from a
host recompute of the bf16 norm from A's h with call 0's weights.
Modes: (default) TPU timing; --aot deviceless tpu7x:2x2x1 compile + engagement of every long program; --interpret CPU harness
(8 host devices, one call per arm); --selftest [--mutants] / --barrier-selftest (no devices).
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
ROWS = (288, 320, 352, 384, 448)
ARMS = ("W1", "W1F", "W1T")
NEG_ARM = "W1FM"
NEG_EDIT = ("column = _lane_column(inj_tile, lane0 + j)", "column = _lane_column(inj_tile, lane0 + j + 1)")
CMP_MAX = 256
L_LONG, L_SHORT = 24, 4
GEO = dict(HC=4, H=2560, LR=320, EPS=1e-6)
SERVED = "qwen4exp_hc_decode_combine"
W1N, W1FN = SERVED + "_w1", SERVED + "_w1f"
# q38 HC-HIGHROWS-2 W3 microbench: the served B2 / S3 decode LIBTPU (2 flags, default clock) + the barrier off, PINNED: the pod's inherited
# LIBTPU_INIT_ARGS (the bench recipe's 12-flag set with --xla_tpu_dvfs_p_state=7) is ignored (RPAD-HIGHROWS-2 F4 bench rule, credit).
SERVED_LIBTPU = "--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
# Longest first: every other name starts with SERVED.
_KERNELS = tuple(sorted((SERVED, W1N, W1FN), key=len, reverse=True))
ARM_KERNEL = {"W1": W1N, "W1F": W1FN, "W1T": W1N}
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


WANT_LIBTPU = SERVED_LIBTPU + " " + LB_OFF


def apply_cfg() -> str:
    """W3 microbench: LIBTPU_INIT_ARGS pinned to the served 2-flag decode set + the barrier off, whatever the pod inherited."""
    final = WANT_LIBTPU
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
            ok &= final == WANT_LIBTPU and "dvfs_p_state" not in final
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
    if arm in ("W1", "W1T"):
        return hw.hc_decode_mix_wide(*args, mode=1, **st)
    if arm == "W1F":
        return hw.hc_decode_mix_wide(*args, mode=1, fold=True, **st)
    if arm == NEG_ARM:
        return neg_module().hc_decode_mix_wide(*args, mode=1, fold=True, **st)
    raise ValueError(arm)


_NEG = {}


def neg_module():
    """The live negative control: the tree's hc_decode_wide.py with ONE counted edit (NEG_EDIT = bound v2 mutant M2), loaded from a
    temp copy (its kernel keeps the name ..._w1f)."""
    if "m" not in _NEG:
        import importlib.util
        import tempfile

        from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide as hw
        text = open(hw.__file__).read()
        assert text.count(NEG_EDIT[0]) == 1, "negative-control anchor not unique in hc_decode_wide.py"
        fd, p = tempfile.mkstemp(suffix="_hc_decode_wide_neg.py")
        os.close(fd)
        open(p, "w").write(text.replace(NEG_EDIT[0], NEG_EDIT[1]))
        spec = importlib.util.spec_from_file_location("q38_hc_decode_wide_neg", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _NEG["m"] = m
    return _NEG["m"]


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

    return jax.shard_map(body, mesh=mesh, in_specs=(P("d"), P("d"), P("d"), P()), out_specs=(P("d"), P("d")),
                         check_vma=False)


def abstract(tree):
    """Shape, dtype and sharding only (no layout) for every leaf."""
    import jax
    return jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=v.sharding), tree)


def compile_auto(fn, x_or_specs, ws_or_spec, auto=True):
    """Compile a chain with AUTO layouts for the HC weights (INJMXU-2's fix of the bench up-weight relayout, credit: the served
    engine pins the up weight row-major once, so the kernel sees no per-call relayout; an outer jit whose in_shardings carry
    Format(Layout.AUTO, ...) for the weights, lowered on ABSTRACT arguments). The caller lays the arrays into the compiled
    input formats once (lay_args) and calls only the compiled executable. auto=False keeps default layouts (interpret)."""
    import jax
    from jax.experimental.layout import Format, Layout
    xsh = tuple(v.sharding for v in x_or_specs)
    wsh = [{k: Format(Layout.AUTO, v.sharding) if auto else v.sharding for k, v in d.items()} for d in ws_or_spec]
    j = jax.jit(fn, in_shardings=xsh + (wsh,))
    return j.lower(*abstract(tuple(x_or_specs)), abstract(ws_or_spec)).compile()


def lay_args(comp, x, ws):
    """Put the inputs and the weights into the compiled program's input formats (once, outside timing) -> (x, ws)."""
    import jax
    fm = comp.input_formats[0]
    return tuple(jax.device_put(tuple(x), tuple(fm[:3]))), jax.device_put(ws, fm[3])


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


def host_LS(h_a, w, inj, geo):
    """Bound v2 s5: L, S (float64, uninflated; the reader inflates by 1 + 2^-6) from a host recompute of the bf16 norm from A's h."""
    HC, H, EPS = geo["HC"], geo["H"], geo["EPS"]
    h, w, inj = np.asarray(h_a, np.float64), np.asarray(w, np.float64), np.asarray(inj, np.float64)
    norm = []
    for b in range(HC):
        v = h[:, b * H:(b + 1) * H]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + w[b * H:(b + 1) * H])))
    norm = np.concatenate(norm, axis=1)
    return norm @ inj.T, np.abs(norm) @ np.abs(inj).T


def health(mesh, geo, rows, ws, x, arms, interpret=False) -> dict:
    """One call per arm and the negative control on real inputs: every arm finite and equal on all 8 cores and W1T bitwise W1
    (gate); the bound v2 device leg's inputs recorded (the reader judges G1 / G3 / G4 / O1 and the negative control)."""
    import jax
    prev, block, pinj = x
    out = {}
    for arm in list(arms) + [NEG_ARM]:
        f = build_single(arm, geo, mesh, interpret)
        o = jax.device_get(f(prev, block, pinj, ws[0]))
        out[arm] = [np.asarray(np.asarray(v).astype(np.float32), np.float64) for v in o]
    rep = {"finite": {k: bool(all(np.isfinite(v).all() for v in o)) for k, o in out.items()},
           "cores_equal": {k: bool(all(np.array_equal(v[0], v[i]) for v in o for i in range(N))) for k, o in out.items()}}
    A, E, T, M = out.get("W1"), out.get("W1F"), out.get("W1T"), out.get(NEG_ARM)
    bit = lambda u, v: bool(np.array_equal(np.float32(u).view(np.uint32), np.float32(v).view(np.uint32)))  # noqa: E731
    rep["twin_bitwise"] = bool(A is not None and T is not None and all(bit(u[0], v[0]) for u, v in zip(T, A)))
    rep["ok"] = bool(all(rep["finite"].values()) and all(rep["cores_equal"].values()) and rep["twin_bitwise"]
                     and A is not None and E is not None and M is not None)
    if A is not None and E is not None and M is not None:
        p0 = jax.device_get(ws[0])
        L, S = host_LS(A[2][0], np.asarray(p0["w"], np.float32), np.asarray(p0["inject"], np.float32), geo)
        rep["o1"] = {"mixed_bitwise": bit(E[0][0], A[0][0]), "h_bitwise": bit(E[2][0], A[2][0]),
                     "mixed_diff": int((E[0][0] != A[0][0]).sum()), "h_diff": int((E[2][0] != A[2][0]).sum())}
        rep["neg_o1"] = {"mixed_bitwise": bit(M[0][0], A[0][0]), "h_bitwise": bit(M[2][0], A[2][0])}
        rep["gates"] = {"A": A[1][0].tolist(), "E": E[1][0].tolist(), "NEG": M[1][0].tolist(), "L": L.tolist(), "S": S.tolist()}
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
    feeding = "  %fusion.2 = bf16[448,2560]{1,0} fusion(%" + W1FN + ".7), kind=kLoop"
    cases = []
    for arm in ARMS:
        k = ARM_KERNEL[arm]
        cases.append((f"{arm} engaged", arm, [_cc(k)] * L, True))
        cases.append((f"{arm} with one call missing", arm, [_cc(k)] * (L - 1), False))
        cases.append((f"{arm} with an up relayout copy", arm, [_cc(k)] * L + [relay], False))
        other = W1FN if k == W1N else W1N
        cases.append((f"{arm} with a stray {other} call", arm, [_cc(k)] * L + [_cc(other)], False))
        cases.append((f"{arm} with a stray served call", arm, [_cc(k)] * L + [_cc(SERVED)], False))
    cases += [
        ("W1F with a feeding fusion naming the kernel still engaged (operands never count)", "W1F", [_cc(W1FN)] * L + [feeding], True),
        ("W1F with an unnamed custom call fed by the kernel still engaged (operands never count)", "W1F",
         [_cc(W1FN)] * L + ["  %custom-call.9 = (bf16[448,2560]{1,0}) custom-call(%" + W1FN + ".7), custom_call_target=\"x\""], True),
        ("W1 with W1F-named calls is not engaged (prefix rule)", "W1", [_cc(W1FN)] * L, False),
        ("W1F with W1-named calls is not engaged", "W1F", [_cc(W1N)] * L, False),
        ("W1T with W1F-named calls is not engaged", "W1T", [_cc(W1FN)] * L, False),
        ("metadata op_name fallback counts", "W1F", ["  %custom-call.1 = (bf16[448,2560]{1,0}) custom-call(%a), "
                                                     f"metadata={{op_name=\"jit(f)/{W1FN}\"}}"] * L, True),
    ]
    bad = 0
    for label, arm, lines, want in cases:
        got = engaged(arm, kernel_counts("\n".join(lines)), L)
        print(f"Q38_HCW_SELFTEST {'ok ' if got == want else 'BAD'} {label}: got {got} want {want}")
        bad += got != want
    cfg_ok = ROWS == (288, 320, 352, 384, 448) and ARMS == ("W1", "W1F", "W1T") and NEG_ARM == "W1FM" \
        and (L_LONG, L_SHORT) == (24, 4) and GEO == dict(HC=4, H=2560, LR=320, EPS=1e-6) \
        and ARM_KERNEL == {"W1": W1N, "W1F": W1FN, "W1T": W1N}
    print(f"Q38_HCW_SELFTEST {'ok ' if cfg_ok else 'BAD'} plan rows {ROWS} arms {ARMS} neg {NEG_ARM} chain {L_LONG}/{L_SHORT} geo {GEO}")
    bad += not cfg_ok
    kp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "vllm_torchtpu", "kernels", "qwen4_exp",
                      "hc_decode_wide.py")
    try:
        anchor = open(kp).read().count(NEG_EDIT[0]) == 1
    except OSError:
        anchor = False
    print(f"Q38_HCW_SELFTEST {'ok ' if anchor else 'BAD'} negative-control anchor once in the tree's hc_decode_wide.py")
    bad += not anchor
    print("Q38_HCW_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants() -> int:
    """Each mutant weakens one rule; the selftest must fail under every one."""
    g = globals()
    orig = {k: g[k] for k in ("ARM_KERNEL", "call_kernel", "RELAYOUT_RE", "_KERNELS", "NEG_EDIT")}
    muts = {
        "W1 rule names the fold kernel": ("ARM_KERNEL", {**orig["ARM_KERNEL"], "W1": W1FN}),
        "twin rule names the fold kernel": ("ARM_KERNEL", {**orig["ARM_KERNEL"], "W1T": W1FN}),
        "kernel name read from any operand": ("call_kernel", lambda ln: next((k for k in orig["_KERNELS"] if k in ln), None)),
        "plain prefix match, shortest name first": ("call_kernel", lambda ln: next(
            (k for k in sorted(orig["_KERNELS"], key=len) if ln.split(" = ", 1)[0].strip().lstrip("%").startswith(k)), None)),
        "relayout check blind": ("RELAYOUT_RE", re.compile(r"$^")),
        "negative-control anchor stale": ("NEG_EDIT", ("column = _lane_column(inj_tile, lane0 + j)  # stale", NEG_EDIT[1])),
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
    ap.add_argument("--rows", help="row counts (default 288,320,352,384,448)")
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
           "plan": {"rows": rows_l, "arms": arms, "order": "base(W1), W1 / W1F / W1T rotating, base2(W1)"},
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
        comp = compile_auto(build(arm, geo, mesh, LL, a.interpret), specs_or_x, ws_or_spec, auto=not a.interpret)
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
                cs = compile_auto(build(arm, geo, mesh, LS, a.interpret), x, ws[:LS])
                al, as_ = lay_args(cl, x, ws), lay_args(cs, x, ws[:LS])
                o = jax.device_get(cs(*as_[0], as_[1]))
                rec["short_finite_fraction"] = float(np.mean([np.isfinite(np.asarray(v, np.float32)).mean() for v in o]))
                rec["compile_s"] = round(time.time() - t0, 1)
                progs[arm] = (cl, cs, al, as_)
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
                    cl, cs, al, as_ = progs[arm]
                    t0 = time.perf_counter()
                    jax.block_until_ready(cl(*al[0], al[1]))
                    ls[arm].append((time.perf_counter() - t0) * 1e6)
                    t0 = time.perf_counter()
                    jax.block_until_ready(cs(*as_[0], as_[1]))
                    ss[arm].append((time.perf_counter() - t0) * 1e6)
            return ls, ss

        try:
            for arm in ran:
                for _ in range(3):
                    cl, cs, al, as_ = progs[arm]
                    jax.block_until_ready(cl(*al[0], al[1]))
                    jax.block_until_ready(cs(*as_[0], as_[1]))
            if "W1" in ran:
                ls, ss = time_arms(["W1"], a.iters)
                case["base"] = per_call(ls["W1"], ss["W1"], LL, LS)
            ls, ss = time_arms(ran, a.iters)
            for arm in ran:
                case[arm] = per_call(ls[arm], ss[arm], LL, LS)
            if "W1" in ran:
                ls, ss = time_arms(["W1"], a.iters)
                case["base2"] = per_call(ls["W1"], ss["W1"], LL, LS)
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
            len(res["cases"].get(str(r), {})) == len([x for x in arms if arm_rows(x, r)]) + (2 if "W1" in arms else 0)
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

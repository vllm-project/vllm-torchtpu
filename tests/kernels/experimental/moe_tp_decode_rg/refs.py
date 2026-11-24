# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""References for moe_tp_decode and its row-gathered variant, per rank, over "ring items" (one expert x one
128-column I chunk, the kernel's unit; TP16: item = expert).

- native_partial: the incumbent kernel's arithmetic in jnp ops (x per row per 512-col block in bf16, the per-block
  bf16 gate_up chain, coeff-masked mid quantized per row per item, f32 owner accumulation in ascending item order, the
  kernel's own shared expert and combine). The "native reference" of the skill: calibrates the 1e-3 gate.
- rg_partial: the row-gathered ALGORITHM (gather each item's routed rows into a slab, compute on the slab, scatter-add;
  EP: compact accumulator initialised from the gathered owner rows, scatter-set at the expert's last chunk) with
  injectable faults F1..F8. Proves the algebra (it must equal native_partial up to matmul row-independence) and
  measures the fault controls' catch rate before any kernel exists.
- oracle_partial_f32: dequantized weights, unquantized x, exact routing weights, Precision.HIGHEST (TPU tier);
  oracle_partial_f64: the same in numpy float64 (CPU tier, and spot checks of the f32 oracle).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel import _combine
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

F8, BF16, F32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32
FP8_MAX = float(jnp.finfo(F8).max)
QB = 512
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")


# ----------------------------------------------------------------------------------------------- weights / items
def fp8_columns(w):
    s = jnp.max(jnp.abs(w), axis=-2) / FP8_MAX
    s = jnp.where(s == 0, 1.0, s)
    return (w / s[..., None, :]).astype(F8), s


def rank_weights(key, dep):
    """The tests' _rank_weights at a Dep: (w13, w2, s13, s2, sw13, ss13, sw2, ss2) in kernel layout."""
    h, e, i, fs = dep.h, dep.e_l, dep.i, dep.fs
    k13, k2, ks13, ks2 = jax.random.split(key, 4)
    w13, s13 = fp8_columns(jax.random.normal(k13, (e, h, 2 * i), F32) / np.sqrt(h))
    w2, s2 = fp8_columns(jax.random.normal(k2, (e, i, h), F32) / np.sqrt(16 * i))
    sw13, ss13 = fp8_columns(jax.random.normal(ks13, (h, 2 * fs), F32) / np.sqrt(h))
    sw2, ss2 = fp8_columns(jax.random.normal(ks2, (fs, h), F32) / np.sqrt(16 * fs))
    return (w13, w2, s13.reshape(e, 1, 1, 2 * i), s2.reshape(e, 1, 1, h), sw13, ss13, sw2, ss2)


def items_of(w13, w2, s13, s2):
    """Kernel ring items in kernel order (expert-major, chunk-minor): w13 [K, h, 256] (gate chunk | up chunk),
    w2 [K, 128, h], s13 [K, 256], s2 [K, h], local expert id [K]."""
    e, h, two_i = w13.shape
    i = two_i // 2
    ic = min(i, 128)
    nc = i // ic
    g = w13[:, :, :i].reshape(e, h, nc, ic)
    u = w13[:, :, i:].reshape(e, h, nc, ic)
    w13i = jnp.concatenate([g, u], axis=-1).transpose(0, 2, 1, 3).reshape(e * nc, h, 2 * ic)
    w2i = w2.reshape(e, nc, ic, h).reshape(e * nc, ic, h)
    s = s13.reshape(e, two_i)
    s13i = jnp.concatenate([s[:, :i].reshape(e, nc, ic), s[:, i:].reshape(e, nc, ic)], -1).reshape(e * nc, 2 * ic)
    s2i = jnp.repeat(s2.reshape(e, h), nc, axis=0)
    eid = jnp.repeat(jnp.arange(e, dtype=jnp.int32), nc)
    return w13i, w2i, s13i, s2i, eid, nc


# ----------------------------------------------------------------------------------------------- routing
def routing_coef(scores, bias, topk, *, exact=False):
    """[m, E] f32 routing weights as the kernel computes them (top-k of scores + bias; picked unbiased scores over
    their pick-order sum clamped to 1e-20; rounded to bf16), or the exact f32 weights (exact=True, oracle)."""
    _, ids = lax.top_k(scores + bias, topk)
    picked = jnp.take_along_axis(scores, ids, axis=-1)
    wsum = picked[:, :1]
    for j in range(1, topk):
        wsum = wsum + picked[:, j : j + 1]
    w = picked / jnp.maximum(wsum, 1e-20)
    if not exact:
        w = w.astype(BF16).astype(F32)
    return jnp.sum(jax.nn.one_hot(ids, scores.shape[1], dtype=F32) * w[..., None], axis=1)


def quant_blocks(x):
    """_quantize_blocks: x [m, h] bf16 -> (q fp8 [m, h], scales bf16 [m, h/512])."""
    m, h = x.shape
    qs, ss = [], []
    for k in range(0, h, QB):
        block = x[:, k : k + QB]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        qs.append((block * inv).astype(F8))
        ss.append(scale)
    return jnp.concatenate(qs, 1), jnp.concatenate(ss, 1)


# ----------------------------------------------------------------------------------------------- per-item math
def _gate_up(q, sc, w13k, s13k):
    """gmm_v2's per-block bf16 chain on [rows, h] fp8 with bf16 scales [rows, nb]."""
    acc = jnp.zeros((q.shape[0], w13k.shape[1]), BF16)
    ws = s13k.astype(BF16)
    for b in range(q.shape[1] // QB):
        rows = slice(b * QB, (b + 1) * QB)
        part = jnp.matmul(q[:, rows], w13k[rows], preferred_element_type=F32).astype(BF16)
        part = part * sc[:, b : b + 1]
        part = part * ws
        acc = acc + part
    return acc


def _item_y(q, sc, coeff, w13k, w2k, s13k, s2k):
    """One item over a [rows] slab: y [rows, h] f32 (the kernel's finish(), before the owner add)."""
    acc = _gate_up(q, sc, w13k, s13k)
    ic = w2k.shape[0]
    act = jax.nn.silu(acc[:, :ic]) * acc[:, ic:]
    mid = jnp.where(coeff != 0, act.astype(F32) * coeff, 0.0)
    ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
    mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(F8)
    cols = []
    for n in range(0, w2k.shape[1], 512):
        acc2 = jnp.matmul(mq, w2k[:, n : n + 512], preferred_element_type=F32)
        cols.append(acc2 * s2k[n : n + 512] * ms)
    return jnp.concatenate(cols, 1)


def _shared_expert_math(x, sw13, ss13, sw2, ss2):
    """kernel._shared_expert (kernel.py:385-400) on arrays: x per-row quantized by quantize_block."""
    xq, xs = quantize_block(x, axis=1, target_dtype=F8)
    fs = sw2.shape[0]
    gu = jnp.dot(xq, sw13, preferred_element_type=F32)
    gu = (gu * xs * ss13).astype(BF16).astype(F32)
    act = jax.nn.silu(gu[:, :fs]).astype(BF16).astype(F32) * gu[:, fs:]
    act_q, act_scale = quantize_block(act.astype(BF16), axis=1, target_dtype=F8)
    y = jnp.dot(act_q, sw2, preferred_element_type=F32)
    return y * act_scale * ss2


def _shared_and_combine(x, owner, sw13, ss13, sw2, ss2, routed_scale):
    shared = _shared_expert_math(x, sw13, ss13.reshape(1, -1), sw2, ss2.reshape(1, -1))
    return _combine(owner, shared, routed_scale)


# ----------------------------------------------------------------------------------------------- native (incumbent)
@functools.partial(jax.jit, static_argnames=("routed_scale",))
def native_partial(x, coef, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2, *, routed_scale):
    """This rank's bf16 partial (before the all-reduce), the incumbent's arithmetic. coef: [m, E] f32 (kernel
    routing, rows >= n_valid zeroed by the caller)."""
    w13i, w2i, s13i, s2i, eid, _ = items_of(w13, w2, s13, s2)
    q, sc = quant_blocks(x)
    cl = lax.dynamic_slice_in_dim(coef, e0, w13.shape[0], axis=1)  # local experts

    def step(owner, k):
        coeff = lax.dynamic_index_in_dim(cl, eid[k], axis=1, keepdims=True)
        y = _item_y(q, sc, coeff, w13i[k], w2i[k], s13i[k], s2i[k])
        return owner + y, None

    owner, _ = lax.scan(step, jnp.zeros(x.shape, F32), jnp.arange(w13i.shape[0]))
    return _shared_and_combine(x, owner, sw13, ss13, sw2, ss2, routed_scale)


# ----------------------------------------------------------------------------------------------- row-gathered algorithm
@functools.partial(jax.jit, static_argnames=("routed_scale", "fault", "r_slab"))
def rg_partial(x, coef, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2, *, routed_scale, fault=None, r_slab=16):
    """The row-gathered algorithm: per item, gather the rows routed to its expert (ascending) into a slab of m slots
    (slots >= n_e point at a trash row m and carry coeff 0), compute on the slab, scatter-add into owner rows. EP
    (n_chunks > 1): compact accumulator initialised from the gathered owner rows at the expert's first chunk,
    scatter-set at its last. `fault`: one of FAULTS (see module docstring of test_accuracy)."""
    m, h = x.shape
    w13i, w2i, s13i, s2i, eid, nc = items_of(w13, w2, s13, s2)
    q, sc = quant_blocks(x)
    qp = jnp.concatenate([q, jnp.zeros((1, h), q.dtype)], 0)  # + trash row m
    scp = jnp.concatenate([sc, jnp.zeros((1, sc.shape[1]), sc.dtype)], 0)
    cl = lax.dynamic_slice_in_dim(coef, e0, w13.shape[0], axis=1)
    slot = jnp.arange(m)

    def table(col):
        mask = col != 0
        n = jnp.sum(mask)
        order = jnp.argsort(jnp.where(mask, 0, 1) * (m + 1) + slot)  # routed rows first, ascending
        rows = jnp.where(slot < n, order, m)
        return rows, n

    def step(carry, k):
        owner, ycomp = carry
        e = eid[k]
        col_e = (e + 1) % cl.shape[1] if fault == "F6" else e
        rows, n = table(lax.dynamic_index_in_dim(cl, col_e, axis=1, keepdims=False))
        valid = slot < n
        if fault == "F1":
            valid = valid & (slot < n - 1)
        if fault == "F5":
            valid = valid & (slot < r_slab)
        rows = jnp.where(valid, rows, m)
        coeff = jnp.where(valid, jnp.concatenate([cl[:, e], jnp.zeros((1,), F32)])[rows], 0.0)[:, None]
        if fault == "F3":
            coeff = jnp.concatenate([coeff[1:], jnp.zeros((1, 1), F32)], 0)
        qg = qp[rows]
        scg = scp[rows]
        if fault == "F7":
            scg = jnp.broadcast_to(scg[:1], scg.shape)
        y = _item_y(qg, scg, coeff, w13i[k], w2i[k], s13i[k], s2i[k])
        dst = jnp.where(valid, rows + 1, m) if fault == "F2" else rows
        ownp = jnp.concatenate([owner, jnp.zeros((1, h), F32)], 0)
        if nc == 1:
            ownp = ownp.at[dst].add(y)
            if fault == "F8":
                ownp = ownp.at[dst[0]].add(y[0])
            return (ownp[:m], ycomp), None
        first = (k % nc) == 0
        last = (k % nc) == nc - 1
        init = ownp[dst]
        if fault == "F4":
            init = jnp.where(first, ycomp, init)  # stale: previous expert's compact rows
        yc = jnp.where(first, init, ycomp) + y
        if fault == "F8":  # EP: slot 0's chunk contribution accumulated twice
            yc = yc.at[0].add(y[0])
        ownp = jnp.where(last, ownp.at[dst].set(yc), ownp)
        return (ownp[:m], yc), None

    (owner, _), _ = lax.scan(step, (jnp.zeros((m, h), F32), jnp.zeros((m, h), F32)), jnp.arange(w13i.shape[0]))
    return _shared_and_combine(x, owner, sw13, ss13, sw2, ss2, routed_scale)


# ----------------------------------------------------------------------------------------------- oracles
@functools.partial(jax.jit, static_argnames=("routed_scale",))
def oracle_partial_f32(x, coef_exact, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2, *, routed_scale):
    """f32 partial: dequantized weights, unquantized x, exact routing weights, Precision.HIGHEST."""
    hp = lax.Precision.HIGHEST
    xf = x.astype(F32)
    i = w2.shape[1]
    cl = lax.dynamic_slice_in_dim(coef_exact, e0, w13.shape[0], axis=1)

    def step(acc, e):
        w13f = w13[e].astype(F32) * s13[e, 0]
        gu = jnp.dot(xf, w13f, precision=hp)
        act = jax.nn.silu(gu[:, :i]) * gu[:, i:]
        y = jnp.dot(act, w2[e].astype(F32) * s2[e, 0], precision=hp)
        return acc + lax.dynamic_index_in_dim(cl, e, axis=1, keepdims=True) * y, None

    routed, _ = lax.scan(step, jnp.zeros(xf.shape, F32), jnp.arange(w13.shape[0]))
    fs = sw2.shape[0]
    sgu = jnp.dot(xf, sw13.astype(F32) * ss13, precision=hp)
    sact = jax.nn.silu(sgu[:, :fs]) * sgu[:, fs:]
    return jnp.dot(sact, sw2.astype(F32) * ss2, precision=hp) + routed_scale * routed


def oracle_partial_f64(x, coef_exact, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2, *, routed_scale, rows=None):
    """numpy float64 version of oracle_partial_f32 (optionally on a subset of rows)."""
    f = lambda a: np.asarray(jnp.asarray(a).astype(F32)).astype(np.float64)  # noqa: E731
    xf = f(x) if rows is None else f(x)[rows]
    cl = f(coef_exact)[:, e0 : e0 + w13.shape[0]]
    cl = cl if rows is None else cl[rows]
    i = w2.shape[1]
    w13f = f(w13) * f(s13)[:, 0]
    w2f = f(w2) * f(s2)[:, 0]
    routed = np.zeros(xf.shape)
    for e in range(w13.shape[0]):
        if not cl[:, e].any():
            continue
        gu = xf @ w13f[e]
        g = gu[:, :i]
        act = g / (1 + np.exp(-g)) * gu[:, i:]
        routed += cl[:, e : e + 1] * (act @ w2f[e])
    fs = f(sw2).shape[0]
    sgu = xf @ (f(sw13) * f(ss13))
    sg = sgu[:, :fs]
    sact = sg / (1 + np.exp(-sg)) * sgu[:, fs:]
    return sact @ (f(sw2) * f(ss2)) + routed_scale * routed


# ----------------------------------------------------------------------------------------------- metrics
def compare(got, want, n_valid, *, ref=None):
    """Gate metrics of got [m, h] (any float) vs want [>=n_valid, h]; ref: optional native reference [m, h]."""
    g = np.asarray(jnp.asarray(got).astype(F32)).astype(np.float64)
    w = np.asarray(jnp.asarray(want).astype(F32)).astype(np.float64)[:n_valid]
    out = dict(nan=int(np.isnan(g).sum()))
    pad = np.asarray(jnp.asarray(got).astype(F32))[n_valid:]
    out["padding_pos_zero"] = bool(np.all((pad == 0) & ~np.signbit(pad)))
    gv = g[:n_valid]
    if n_valid:
        out["rel_l2"] = float(np.linalg.norm(gv - w) / np.linalg.norm(w))
        out["cos"] = float(np.vdot(gv, w) / (np.linalg.norm(gv) * np.linalg.norm(w)))
        out["worst_row_rel_l2"] = float(np.max(np.linalg.norm(gv - w, axis=1) / np.maximum(np.linalg.norm(w, axis=1), 1e-30)))
    if ref is not None:
        r = np.asarray(jnp.asarray(ref).astype(F32))
        gg = np.asarray(jnp.asarray(got).astype(F32))
        out["bitwise_vs_ref"] = float(np.mean(gg.view(np.uint32) == r.view(np.uint32)))
        rv = r[:n_valid].astype(np.float64)
        out["rel_l2_vs_ref"] = float(np.linalg.norm(gv - rv) / max(np.linalg.norm(rv), 1e-30)) if n_valid else 0.0
    return out


TOL_ORACLE = dict(rel_l2=0.06, cos=0.998)  # as test_moe_tp_decode_kernel.TOLERANCE
TOL_NATIVE = 1e-3  # as EXPECTED_REL_L2


def passes(cmp_oracle, cmp_native=None):
    ok = cmp_oracle["nan"] == 0 and cmp_oracle["padding_pos_zero"]
    if "rel_l2" in cmp_oracle:
        ok = ok and cmp_oracle["rel_l2"] <= TOL_ORACLE["rel_l2"] and cmp_oracle["cos"] >= TOL_ORACLE["cos"]
    if cmp_native is not None and "rel_l2_vs_ref" in cmp_native:
        ok = ok and cmp_native["rel_l2_vs_ref"] <= TOL_NATIVE
    return ok

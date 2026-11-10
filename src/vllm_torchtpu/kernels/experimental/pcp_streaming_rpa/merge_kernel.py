# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TPU_PCP_MERGE_KERNEL: one Pallas pass that LSE-merges the all-RPA partials.

The all-RPA path ends with three partials per local query row: the current
pass, this rank's history over its own pages, and its history over the peer's
pages (computed there and sent back). Each is RPAm output in the kernel's own
layout, o [Hkv, T, G // p, p, D] (the aliased q buffer, bf16, p = 2), with one
LSE per (token, head). The XLA merge converts every partial to f32 in a
separate pass and writes the merged result in another (gla3: f32 reshape 39.0 +
merge fusion 16.6 + row-mask select 5.8 ms per round). This kernel reads each
partial once, merges in f32 in VMEM with the same arithmetic as
``history_rpa.merge_partials_fused``, writes invalid rows as zeros and stores
[T, Hkv * G, D] in the output dtype.

LSE inputs are lane-dense: [Hkv, T * G // 128, 128] f32 whose flat index is
t * G + g, the row order of o (-inf: attended nothing). The valid-row mask is
the same lane-dense shape without the head axis (1.0 valid, 0.0 not).
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_ROWS_U32 = _LANES // 2  # u32 rows of one 128-row bf16 chunk


def merge_kernel_enabled() -> bool:
    """TPU_PCP_MERGE_KERNEL=1 (all-RPA only, default off until measured)."""
    return os.environ.get("TPU_PCP_MERGE_KERNEL", "0") == "1"


def block_tokens(t: int, g: int):
    """Tokens per grid step: the largest of 256 / 128 / 64 dividing ``t`` whose
    LSE block is whole (8-row) tiles; None when no size fits (XLA merge)."""
    for bt in (256, 128, 64):
        if t % bt == 0 and (bt * g) % (_LANES * 8) == 0:
            return bt
    return None


def supported(parts_o) -> bool:
    hkv, t, g2, p, d = parts_o[0].shape
    return (
        p == 2
        and parts_o[0].dtype == jnp.bfloat16
        and d % _LANES == 0
        and block_tokens(t, g2 * p) is not None
    )


def lane_dense(x):
    """[..., T, G] -> [..., T * G // 128, 128] (flat index t * G + g)."""
    *lead, t, g = x.shape
    return x.reshape(*lead, t * g // _LANES, _LANES)


def row_mask_lane_dense(row_valid, g: int):
    """[T] bool -> [T * G // 128, 128] f32 (each token's flag on its G heads)."""
    t = row_valid.shape[0]
    m = jnp.broadcast_to(row_valid.astype(jnp.float32)[:, None], (t, g))
    return m.reshape(t * g // _LANES, _LANES)


def _columns(x):
    """(n, 128) lane-dense rows -> (128, n): column c holds rows c * 128 .. + 127."""
    n = x.shape[0]
    if n == _LANES:
        return x.T
    pad = jnp.zeros((_LANES - n, _LANES), x.dtype)
    return jnp.concatenate([x, pad], axis=0).T[:, :n]


def _merge_kernel(*refs, n_parts: int, g: int, d: int):
    o_refs = refs[:n_parts]
    l_refs = refs[n_parts : 2 * n_parts]
    mask_ref = refs[2 * n_parts]
    out_ref = refs[2 * n_parts + 1]
    # (1, bt, G // 2, 2, D) bf16 -> (bt * G // 2, D) u32: u32 row (t, g // 2)
    # packs heads g and g + 1 (RPAm's own view of this buffer).
    o_u32 = [r.at[0].bitcast(jnp.uint32).reshape(-1, d) for r in o_refs]
    # (bt, G, D) bf16 -> (bt * G // 2, D) u32, same packing.
    out_u32 = out_ref.bitcast(jnp.uint32).reshape(-1, d)
    cols = [_columns(r[0]) for r in l_refs]  # (128, nchunk) each
    mcols = _columns(mask_ref[...])
    nchunk = mcols.shape[1]
    for c in range(nchunk):
        lses = [x[:, c : c + 1] for x in cols]  # (128, 1): rows c * 128 ..
        big = functools.reduce(jnp.maximum, lses)
        big = jnp.where(jnp.isfinite(big), big, 0.0)
        ws = [jnp.exp(x - big) for x in lses]
        den = functools.reduce(jnp.add, ws)
        inv = 1.0 / jnp.where(den > 0, den, 1.0)
        acc = None
        for w, ou in zip(ws, o_u32):
            o = pltpu.bitcast(ou[pl.ds(c * _ROWS_U32, _ROWS_U32), :], jnp.bfloat16)
            wb = jnp.broadcast_to(w, (_LANES, d))
            term = jnp.where(wb > 0, wb * o.astype(jnp.float32), 0.0)
            acc = term if acc is None else acc + term
        res = acc * jnp.broadcast_to(inv, (_LANES, d))
        valid = jnp.broadcast_to(mcols[:, c : c + 1], (_LANES, d)) > 0
        res = jnp.where(valid, res, 0.0).astype(out_ref.dtype)
        out_u32[pl.ds(c * _ROWS_U32, _ROWS_U32), :] = pltpu.bitcast(res, jnp.uint32)


def o_token_major_enabled() -> bool:
    """TPU_PCP_O_TOKEN_MAJOR=1 (default off): the merge kernel writes o as
    [T, Hkv * G * D] -- bf16 rows packed in token pairs, the layout the o
    consumer (vLLM's [T, H * D] view into the sigmoid gate) reads -- instead of
    [T, H, D], whose (H, D) tiles packed in head pairs need a SparseCore retile
    before the consumer."""
    return os.environ.get("TPU_PCP_O_TOKEN_MAJOR", "0") == "1"


def _o_token_major_mutant() -> bool:
    """TPU_PCP_O_TOKEN_MAJOR_MUTANT=1 (tests only): swap the two heads of each
    packed pair, so a bitwise check that cannot fail is caught."""
    return os.environ.get("TPU_PCP_O_TOKEN_MAJOR_MUTANT", "0") == "1"


_TM_ROWS = 32  # tokens per token-major store: two packed bf16 row tiles


def _strided_rows_load(ref2d, first: int, step: int, n: int):
    """Rows first, first + step, ... (n of them) of a (rows, k * 128) ref.

    Mosaic's strided load needs a 128-wide base memref, so the ref is viewed
    as (rows * k, 128) and each 128-lane fold is loaded with stride step * k.
    """
    rows, cols = ref2d.shape
    folds = cols // _LANES
    flat = ref2d.reshape(rows * folds, _LANES)
    parts = [flat[pl.ds(first * folds + i, n, stride=step * folds)] for i in range(folds)]
    return parts[0] if folds == 1 else jnp.concatenate(parts, axis=1)


def _merge_kernel_tm(*refs, n_parts: int, g: int, d: int, mutant: bool):
    """``_merge_kernel`` with a token-major output block (bt, G * D).

    The weights are computed once per block in the column layout of
    ``_merge_kernel`` (row l = (t % tpr) * G + g of column t // tpr, tpr =
    128 // G tokens per lane-dense row) with the same elementwise arithmetic,
    then kept in VMEM scratch. Each step takes one head pair (2j, 2j + 1) over
    32 tokens: a sublane-strided load of the packed partial rows (t, j), whose
    low / high 16 bits are heads 2j / 2j + 1 (bf16 -> f32 is the 16-bit shift),
    the same weighted sum, scale and row mask per element, and one store of
    32 token rows per head into its D-wide column slot.
    """
    o_refs = refs[:n_parts]
    l_refs = refs[n_parts : 2 * n_parts]
    mask_ref = refs[2 * n_parts]
    out_ref = refs[2 * n_parts + 1]
    w_ref = refs[2 * n_parts + 2]
    g2 = g // 2
    tpr = _LANES // g
    bt = out_ref.shape[0]
    cols = [_columns(r[0]) for r in l_refs]  # (128, nchunk) each
    big = functools.reduce(jnp.maximum, cols)
    big = jnp.where(jnp.isfinite(big), big, 0.0)
    ws = [jnp.exp(x - big) for x in cols]
    den = functools.reduce(jnp.add, ws)
    inv = 1.0 / jnp.where(den > 0, den, 1.0)
    nchunk = inv.shape[1]
    # Scratch is 128 lanes wide (strided loads need it); columns >= nchunk are unused.
    for i, w in enumerate(ws):
        w_ref[i, :, pl.ds(0, nchunk)] = w
    w_ref[n_parts, :, pl.ds(0, nchunk)] = inv
    w_ref[n_parts + 1, :, pl.ds(0, nchunk)] = _columns(mask_ref[...])
    # (1, bt, G // 2, 2, D) bf16 -> (bt * G // 2, D) u32: row t * G // 2 + j
    # packs heads 2j (low half) and 2j + 1 (high half).
    o_u32 = [r.at[0].bitcast(jnp.uint32).reshape(-1, d) for r in o_refs]
    cps = _TM_ROWS // tpr  # weight columns per 32-token step

    def rows(v, s):
        """(tpr, 128) weights [t % tpr, t // tpr] -> (32, D): token row t of step s
        carries its value on every lane (one 128-lane broadcast per column, reused
        across the D // 128 lane tiles)."""
        out = []
        for c in range(cps):
            b = jnp.broadcast_to(v[:, s * cps + c : s * cps + c + 1], (tpr, _LANES))
            out.append(b if d == _LANES else jnp.concatenate([b] * (d // _LANES), axis=1))
        return jnp.concatenate(out, axis=0)

    # The valid-row mask is per token (row_mask_lane_dense puts each token's flag
    # on all of its G heads), so head 0's rows serve every head.
    mrow = _strided_rows_load(w_ref.at[n_parts + 1], 0, g, tpr)
    for s in range(bt // _TM_ROWS):
        valid = rows(mrow, s) > 0
        for j in range(g2):
            heads = (2 * j, 2 * j + 1)
            acc = {}
            for p in range(n_parts):
                u = _strided_rows_load(o_u32[p], s * _TM_ROWS * g2 + j, g2, _TM_ROWS)
                lo = jax.lax.bitcast_convert_type(jax.lax.shift_left(u, jnp.uint32(16)), jnp.float32)
                hi = jax.lax.bitcast_convert_type(u & jnp.uint32(0xFFFF0000), jnp.float32)
                for h, o in zip(heads, (hi, lo) if mutant else (lo, hi)):
                    wb = rows(_strided_rows_load(w_ref.at[p], h, g, tpr), s)
                    term = jnp.where(wb > 0, wb * o, 0.0)
                    acc[h] = term if h not in acc else acc[h] + term
            for h in heads:
                res = acc[h] * rows(_strided_rows_load(w_ref.at[n_parts], h, g, tpr), s)
                res = jnp.where(valid, res, 0.0).astype(out_ref.dtype)
                out_ref[pl.ds(s * _TM_ROWS, _TM_ROWS), pl.ds(h * d, d)] = res


def merge_partials_kernel(
    parts_o, parts_lse, mask, out_dtype, *, interpret: bool | None = None, token_major: bool = False
):
    """LSE merge of RPAm partials in their own layout.

    Args:
        parts_o: sequence of [Hkv, T, G // 2, 2, D] bf16 normalized partials.
        parts_lse: sequence of [Hkv, T * G // 128, 128] f32 lane-dense LSEs.
        mask: [T * G // 128, 128] f32 valid-row mask (``row_mask_lane_dense``).
        out_dtype: output dtype (bf16).
        token_major: write [T, Hkv * G * D] (TPU_PCP_O_TOKEN_MAJOR) instead.
            Requires a per-token ``mask`` (``row_mask_lane_dense``: one flag on
            all G heads of a token); the kernel reads head 0's mask rows for
            every head, so a per-head validity mask would be applied wrongly.

    Returns:
        [T, Hkv * G, D] in ``out_dtype``, head h = kv * G + g; invalid rows 0.
        With ``token_major``: the same values as [T, Hkv * G * D].
    """
    hkv, t, g2, p, d = parts_o[0].shape
    g = g2 * p
    bt = block_tokens(t, g)
    if bt is None or not supported(parts_o):
        raise ValueError(f"merge kernel does not support o {parts_o[0].shape} {parts_o[0].dtype}")
    if token_major and (bt % _TM_ROWS or bt * g // _LANES > _LANES):
        raise ValueError(f"token-major merge needs {_TM_ROWS}-token blocks of <= 128 LSE rows, got {bt}")
    n = len(parts_o)
    lrows = bt * g // _LANES
    for x in parts_lse:
        if x.shape != (hkv, t * g // _LANES, _LANES):
            raise ValueError(f"lane-dense LSE shape {x.shape}")
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    o_spec = pl.BlockSpec((1, bt, g2, p, d), lambda i, h: (h, i, 0, 0, 0))
    l_spec = pl.BlockSpec((1, lrows, _LANES), lambda i, h: (h, i, 0))
    m_spec = pl.BlockSpec((lrows, _LANES), lambda i, h: (i, 0))
    if token_major:
        kernel = functools.partial(_merge_kernel_tm, n_parts=n, g=g, d=d, mutant=_o_token_major_mutant())
        out_spec = pl.BlockSpec((bt, g * d), lambda i, h: (i, h))
        out_shape = jax.ShapeDtypeStruct((t, hkv * g * d), out_dtype)
        scratch = [pltpu.VMEM((n + 2, _LANES, _LANES), jnp.float32)]
    else:
        kernel = functools.partial(_merge_kernel, n_parts=n, g=g, d=d)
        out_spec = pl.BlockSpec((bt, g, d), lambda i, h: (i, h, 0))
        out_shape = jax.ShapeDtypeStruct((t, hkv * g, d), out_dtype)
        scratch = None
    return pl.pallas_call(
        kernel,
        grid=(t // bt, hkv),
        in_specs=[o_spec] * n + [l_spec] * n + [m_spec],
        out_specs=out_spec,
        out_shape=out_shape,
        **({} if scratch is None else dict(scratch_shapes=scratch)),
        compiler_params=None if interpret else pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="pcp_merge_kernel",
    )(*parts_o, *[x.astype(jnp.float32) for x in parts_lse], mask.astype(jnp.float32))

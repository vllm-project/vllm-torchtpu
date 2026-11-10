# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E501
"""Gated q split + per-head RMS norm + partial NeoX RoPE in one Pallas call.

Inputs, tiled along rows in VMEM:

  q_gate   [rows, H * 2 * D]  per head: [q | gate]
  k        [rows, Hk * D]
  q_w, k_w [D] f32            the norm weights, already ``1 + w``
  cos_sin  [rows, R]          the rope cache rows: [cos (R/2) | sin (R/2)]

Returns q and k normed in f32 and rotated on their first R dims, and the
pre-sigmoid gate, all in the input dtype.

Fused Pallas kernel for Q/gate split, head-wise Q/K RMSNorm, and partial rotary embeddings.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _norm(x, w, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * w


def _rope(x, cos, sin, rot):
    half = rot // 2
    x1, x2 = x[:, :half], x[:, half:rot]
    parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    if rot < x.shape[1]:
        parts.append(x[:, rot:])
    return jnp.concatenate(parts, axis=1)


def _kernel(
    qg_ref,
    k_ref,
    qw_ref,
    kw_ref,
    cs_ref,
    q_out,
    k_out,
    g_out,
    *,
    heads,
    kv_heads,
    dim,
    rot,
    eps,
):
    half = rot // 2
    cs = cs_ref[...]
    cos = cs[:, :half]
    sin = cs[:, half:rot]
    qw = qw_ref[...]
    kw = kw_ref[...]
    for h in range(heads):
        q = qg_ref[:, h * 2 * dim : h * 2 * dim + dim].astype(jnp.float32)
        g = qg_ref[:, h * 2 * dim + dim : (h + 1) * 2 * dim]
        q = _rope(_norm(q, qw, eps), cos, sin, rot)
        q_out[:, h * dim : (h + 1) * dim] = q.astype(q_out.dtype)
        g_out[:, h * dim : (h + 1) * dim] = g.astype(g_out.dtype)
    for h in range(kv_heads):
        k = k_ref[:, h * dim : (h + 1) * dim].astype(jnp.float32)
        k = _rope(_norm(k, kw, eps), cos, sin, rot)
        k_out[:, h * dim : (h + 1) * dim] = k.astype(k_out.dtype)


def legal_tile(rows: int, cap: int) -> int:
    """The largest multiple of 8 up to ``cap`` dividing ``rows``, else ``rows``.

    A block equal to the full dimension is always legal, which covers row
    counts such as 13 or 36 that have no multiple-of-8 divisor.
    """
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


@functools.partial(
    jax.jit,
    static_argnames=("heads", "kv_heads", "dim", "rot", "eps", "tile", "interpret"),
)
def qk_norm_rope_gate(
    q_gate: jax.Array,
    k: jax.Array,
    q_w: jax.Array,
    k_w: jax.Array,
    cos_sin: jax.Array,
    *,
    heads: int,
    kv_heads: int,
    dim: int,
    rot: int,
    eps: float,
    tile: int = 1024,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens = q_gate.shape[0]
    vmem_tile_cap = max(8, (1024 * 1024) // max(heads * dim, 1))
    tile = legal_tile(tokens, min(tile, vmem_tile_cap))
    kernel = functools.partial(
        _kernel, heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=eps
    )
    out_shape = (
        jax.ShapeDtypeStruct((tokens, heads * dim), q_gate.dtype),
        jax.ShapeDtypeStruct((tokens, kv_heads * dim), k.dtype),
        jax.ShapeDtypeStruct((tokens, heads * dim), q_gate.dtype),
    )
    if tile == tokens:
        vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
        call = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[vmem] * 5,
            out_specs=(vmem, vmem, vmem),
            name="qk_norm_rope_gate",
            interpret=pltpu.InterpretParams() if interpret else False,
        )
    else:
        w_spec = pl.BlockSpec((1, dim), lambda i: (0, 0))
        q_out_spec = pl.BlockSpec((tile, heads * dim), lambda i: (i, 0))
        k_out_spec = pl.BlockSpec((tile, kv_heads * dim), lambda i: (i, 0))
        call = pl.pallas_call(
            kernel,
            grid=(tokens // tile,),
            out_shape=out_shape,
            in_specs=[
                pl.BlockSpec((tile, heads * 2 * dim), lambda i: (i, 0)),
                pl.BlockSpec((tile, kv_heads * dim), lambda i: (i, 0)),
                w_spec,
                w_spec,
                pl.BlockSpec((tile, rot), lambda i: (i, 0)),
            ],
            out_specs=(q_out_spec, k_out_spec, q_out_spec),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name="qk_norm_rope_gate",
            interpret=pltpu.InterpretParams() if interpret else False,
        )
    return call(
        q_gate,
        k,
        q_w.reshape(1, dim).astype(jnp.float32),
        k_w.reshape(1, dim).astype(jnp.float32),
        cos_sin.astype(jnp.float32),
    )


def reference(
    q_gate, k, q_w, k_w, cos_sin, *, heads, kv_heads, dim, rot, eps
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The eager split, Gemma norm per head in f32 and NeoX rope, in jnp."""
    t = q_gate.shape[0]
    qg = q_gate.reshape(t, heads, 2 * dim)
    q, g = qg[:, :, :dim], qg[:, :, dim:]
    kk = k.reshape(t, kv_heads, dim)

    def norm(x, w):
        x = x.astype(jnp.float32)
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)

    half = rot // 2
    cos = cos_sin[:, :half].astype(jnp.float32)[:, None, :]
    sin = cos_sin[:, half:rot].astype(jnp.float32)[:, None, :]

    def rope(x):
        x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    q = rope(norm(q, q_w)).astype(q_gate.dtype).reshape(t, heads * dim)
    kk = rope(norm(kk, k_w)).astype(k.dtype).reshape(t, kv_heads * dim)
    return q, kk, g.reshape(t, heads * dim)


# --------------------------------------------------------------------------------------
# DN3 (R1 push, agent DNS): the same per-head math, reading q_gate and k in place from
# the fused qkv projection output (W6 D1 mode 1, f131db3d3: no sliced copy of q_gate is
# materialized at the torch_tpu jax_op boundary), with two optional changes that keep
# every element identical:
#   rope="roll": the partial NeoX rope as lane rolls of the first 128 lanes with per-lane
#     cos / signed-sin factors built ONCE per row tile, instead of slicing and
#     concatenating 32-lane pieces for every one of the 34 heads. Per element:
#     x1*cos + x2*(-sin) == x1*cos - x2*sin and x2*cos + x1*sin, exactly.
#   tile / vmem_limit: larger row tiles than the 1 MiB-per-input cap above.
# --------------------------------------------------------------------------------------
ROPES = ("concat", "roll", "rollrev")
V2_MUTANTS = ("none", "sign", "qpair")


def _pack_head_pair(lo, hi):
    """bf16 [rows, d] x 2 -> u32 [rows, d] holding the pair as the (2, 1)-packed rows
    (2i, 2i + 1) of a bf16 [2 * rows, d] array (pltpu.bitcast's order): lo in bits 0-15."""
    lo16 = pltpu.bitcast(lo, jnp.uint16).astype(jnp.uint32)
    hi16 = pltpu.bitcast(hi, jnp.uint16).astype(jnp.uint32)
    return lo16 | (hi16 << 16)


def _strided_rows_store(ref2d, first: int, step: int, val):
    """ref2d [R, C] u32 (C % 128 == 0): rows first, first + step, ... <- val [R // step, C]."""
    rows, cols = ref2d.shape
    folds = cols // 128
    flat = ref2d.reshape(rows * folds, 128)
    for i in range(folds):
        flat[pl.ds(first * folds + i, rows // step, step * folds)] = val[:, i * 128 : (i + 1) * 128]


def _v2_kernel(qg_ref, k_ref, qw_ref, kw_ref, cs_ref, q_out, k_out, g_out, *, heads, kv_heads, dim, rot, eps,
               rope, mutant, q_packed=False):
    half = rot // 2
    cs = cs_ref[...]
    cos = cs[:, :half]
    sin = cs[:, half:rot]
    qw = qw_ref[...]
    kw = kw_ref[...]
    if rope == "concat":
        def rope_fn(x):
            return _rope(x, cos, sin, rot)
    else:
        rows = cs.shape[0]
        nsin = sin if mutant == "sign" else -sin
        tail = 128 - rot
        cf = jnp.concatenate([cos, cos, jnp.ones((rows, tail), jnp.float32)], axis=1)
        sf = jnp.concatenate([nsin, sin, jnp.zeros((rows, tail), jnp.float32)], axis=1)
        lane = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
        first = lane < half
        in_rot = lane < rot
        s_lo, s_hi = (128 - half, half) if rope == "roll" else (half, 128 - half)

        def rope_fn(x):
            lo = x[:, :128]
            partner = jnp.where(first, pltpu.roll(lo, s_lo, 1), pltpu.roll(lo, s_hi, 1))
            lo = jnp.where(in_rot, lo * cf + partner * sf, lo)
            return jnp.concatenate([lo, x[:, 128:]], axis=1) if dim > 128 else lo
    if not q_packed:
        for h in range(heads):
            q = qg_ref[:, h * 2 * dim : h * 2 * dim + dim].astype(jnp.float32)
            g = qg_ref[:, h * 2 * dim + dim : (h + 1) * 2 * dim]
            q_out[:, h * dim : (h + 1) * dim] = rope_fn(_norm(q, qw, eps)).astype(q_out.dtype)
            g_out[:, h * dim : (h + 1) * dim] = g.astype(g_out.dtype)
    else:
        # TPU_QK_NORM_ROPE_Q_PACKED: q_out is the RPAm q layout [Hkv, rows, G/2, 2, D] (head
        # h = kv * G + 2 * g2 + p). Its u32 view [rows * G/2, D] has row (t, g2) = heads
        # (2 g2, 2 g2 + 1) of token t, so each head pair is one sublane-strided u32 store.
        def q_head(h):
            q = qg_ref[:, h * 2 * dim : h * 2 * dim + dim].astype(jnp.float32)
            return rope_fn(_norm(q, qw, eps)).astype(q_out.dtype)

        for h in range(heads):
            g = qg_ref[:, h * 2 * dim + dim : (h + 1) * 2 * dim]
            g_out[:, h * dim : (h + 1) * dim] = g.astype(g_out.dtype)
        gq = heads // kv_heads
        rows = qg_ref.shape[0]
        for kv in range(kv_heads):
            q_u32 = q_out.at[kv].bitcast(jnp.uint32).reshape(rows * (gq // 2), dim)
            for g2 in range(gq // 2):
                h0 = kv * gq + 2 * g2
                if mutant == "qpair":  # tests only: the pair's heads swapped
                    pair = _pack_head_pair(q_head(h0 + 1), q_head(h0))
                else:
                    pair = _pack_head_pair(q_head(h0), q_head(h0 + 1))
                _strided_rows_store(q_u32, g2, gq // 2, pair)
    for h in range(kv_heads):
        k = k_ref[:, h * dim : (h + 1) * dim].astype(jnp.float32)
        k_out[:, h * dim : (h + 1) * dim] = rope_fn(_norm(k, kw, eps)).astype(k_out.dtype)


def v2_supported(heads: int, kv_heads: int, dim: int, rot: int, rope: str = "roll") -> bool:
    qg_cols, k_cols = heads * 2 * dim, kv_heads * dim
    ok = k_cols % 128 == 0 and qg_cols % k_cols == 0 and dim % 128 == 0
    return ok and (rope == "concat" or rot <= 128)


@functools.partial(
    jax.jit,
    static_argnames=("heads", "kv_heads", "dim", "rot", "eps", "tile", "rope", "vmem_limit", "mutant", "interpret",
                     "name", "q_packed"),
)
def qk_norm_rope_gate_v2(
    qkv: jax.Array,
    q_w: jax.Array,
    k_w: jax.Array,
    cos_sin: jax.Array,
    *,
    heads: int,
    kv_heads: int,
    dim: int,
    rot: int,
    eps: float,
    tile: int = 0,
    rope: str = "roll",
    vmem_limit: int = 0,
    mutant: str = "none",
    interpret: bool = False,
    name: str | None = None,
    q_packed: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``qk_norm_rope_gate`` on ``qkv`` [rows, H * 2 * D + 2 * Hk * D] read in place.
    tile 0 keeps the original 1 MiB-per-input cap; vmem_limit 0 the default limit.
    ``q_packed`` (TPU_QK_NORM_ROPE_Q_PACKED): q comes back as RPAm's packed kv-major
    [Hk, rows, G/2, 2, D] instead of [rows, H * D] (same values, permuted)."""
    assert rope in ROPES and mutant in V2_MUTANTS, (rope, mutant)
    assert v2_supported(heads, kv_heads, dim, rot, rope), (heads, kv_heads, dim, rot)
    tokens = qkv.shape[0]
    qg_cols, k_cols = heads * 2 * dim, kv_heads * dim
    assert qkv.shape[1] >= qg_cols + k_cols, (qkv.shape, qg_cols, k_cols)
    if tile:
        tile = legal_tile(tokens, tile)
    else:
        tile = legal_tile(tokens, min(1024, max(8, (1024 * 1024) // max(heads * dim, 1))))
    if q_packed:
        assert heads % kv_heads == 0 and (heads // kv_heads) % 2 == 0, (heads, kv_heads)
    kernel = functools.partial(_v2_kernel, heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=eps, rope=rope,
                               mutant=mutant, q_packed=q_packed)
    gq = heads // kv_heads
    q_shape = (kv_heads, tokens, gq // 2, 2, dim) if q_packed else (tokens, heads * dim)
    q_spec = (pl.BlockSpec((kv_heads, tile, gq // 2, 2, dim), lambda i: (0, i, 0, 0, 0)) if q_packed
              else pl.BlockSpec((tile, heads * dim), lambda i: (i, 0)))
    w_spec = pl.BlockSpec((1, dim), lambda i: (0, 0))
    params = dict(dimension_semantics=("parallel",))
    if vmem_limit:
        params["vmem_limit_bytes"] = vmem_limit
    call = pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        out_shape=(
            jax.ShapeDtypeStruct(q_shape, qkv.dtype),
            jax.ShapeDtypeStruct((tokens, k_cols), qkv.dtype),
            jax.ShapeDtypeStruct((tokens, heads * dim), qkv.dtype),
        ),
        in_specs=[
            pl.BlockSpec((tile, qg_cols), lambda i: (i, 0)),
            pl.BlockSpec((tile, k_cols), lambda i: (i, qg_cols // k_cols)),
            w_spec,
            w_spec,
            pl.BlockSpec((tile, rot), lambda i: (i, 0)),
        ],
        out_specs=(
            q_spec,
            pl.BlockSpec((tile, k_cols), lambda i: (i, 0)),
            pl.BlockSpec((tile, heads * dim), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(**params),
        name=name or "qk_norm_rope_gate_v2_%s_t%d%s" % (rope, tile, "" if mutant == "none" else "_mut" + mutant),
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    return call(
        qkv,
        qkv,
        q_w.reshape(1, dim).astype(jnp.float32),
        k_w.reshape(1, dim).astype(jnp.float32),
        cos_sin.astype(jnp.float32),
    )

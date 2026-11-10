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

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

import functools
import math
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES

# Grid token block at large buckets. Each step pushes the three resident fn
# chunks into the MXU once for the whole block, so a bigger block amortizes
# that and the per-step overhead over more tokens.
TOKEN_BLOCK = 128
# Token block at small (decode-size) buckets, and the multiple blocks round to.
MIN_TOKEN_BLOCK = 32
# A bucket is split into at least this many grid steps, so the next block's
# DMA still hides behind compute; smaller buckets keep MIN_TOKEN_BLOCK.
MIN_GRID_STEPS = 4
# Recombine/collapse tile. A bf16 vreg as loaded from VMEM holds 8 rows x
# 256 lanes, which unpacks into two f32 vregs; at this size the tile's inputs,
# its 20 lane-broadcast mix weights and its accumulators fit the registers.
TILE_ROWS = 8
TILE_LANES = 256


def _recombine_tile(
    x_ref, res_ref, newres_ref, post_b, comb_b, sq, rows, c, *, hc_mult, hidden_size
):
    """All new streams of one tile at column ``c``, from one load of x and of
    each old stream. Adds their squares to the running squared sum ``sq``
    (None to start one) and returns it."""
    slab = post_b[0].shape[1]

    def cols(stream):
        return pl.ds(stream * hidden_size + c, slab)

    # bf16 -> f32 casts are exact, so results are unchanged.
    x = x_ref[rows, pl.ds(c, slab)].astype(jnp.float32)
    old = [res_ref[rows, cols(i)].astype(jnp.float32) for i in range(hc_mult)]
    for j in range(hc_mult):
        acc = post_b[j] * x
        for i in range(hc_mult):
            acc = acc + comb_b[i * hc_mult + j] * old[i]
        # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
        # bf16 residual that post stored, and parity requires matching it.
        newres_ref[rows, cols(j)] = acc.astype(jnp.bfloat16)
        # Square the stored bf16 value. Read back rather than re-converted
        # from `acc`, its squares compile to multiply-accumulates of the
        # packed bf16 vreg into f32 (exact products), as in a whole-row sum.
        new = newres_ref[rows, cols(j)].astype(jnp.float32)
        sq = new * new if sq is None else sq + new * new
    return sq


def _collapse_tile(newres_ref, layer_ref, pre_b, rows, c, *, hc_mult, hidden_size):
    """The layer input of one tile at column ``c``, from the stored new
    streams."""
    slab = pre_b[0].shape[1]

    def stream(i):
        return newres_ref[rows, pl.ds(i * hidden_size + c, slab)].astype(jnp.float32)

    lay = pre_b[0] * stream(0)
    for i in range(1, hc_mult):
        lay = lay + pre_b[i] * stream(i)
    layer_ref[rows, pl.ds(c, slab)] = lay.astype(jnp.bfloat16)


def _lane_broadcast(cols, width):
    """Each (rows, 1) column of ``cols`` broadcast to ``width`` lanes."""
    return [
        jnp.broadcast_to(cols[:, k : k + 1], (cols.shape[0], width))
        for k in range(cols.shape[1])
    ]


def _select_token_block(num_tokens: int, token_block_size: int) -> tuple[int, int]:
    """(token block, padded tokens): ``token_block_size`` at large buckets,
    else MIN_GRID_STEPS blocks of at least MIN_TOKEN_BLOCK tokens."""
    # The block is sized by the bucket alone and fits the budget at
    # DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    per_step = utils.round_up(num_tokens, utils.SUBLANE) // MIN_GRID_STEPS
    block = min(
        token_block_size,
        max(MIN_TOKEN_BLOCK, per_step // MIN_TOKEN_BLOCK * MIN_TOKEN_BLOCK),
    )
    return utils.select_token_block(num_tokens, block)


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    layer_ref,
    sqrsum_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """One token block: recombine, GEMM, collapse, gates.

    Recombine and collapse are elementwise per token, so they run in tiles
    of TILE_ROWS x TILE_LANES that load their inputs once and store as they
    go: each f32 intermediate lives for one tile instead of spanning the
    block's whole width. The squared sum accumulates tile by tile, and the
    GEMM reads the stored new streams back. The post and comb gates,
    Sinkhorn included, are computed from the GEMM's result in-block, so the
    call emits post_mix and comb_mix rather than raw mixes.
    """
    tb = x_ref.shape[0]
    m, h = hc_mult, hidden_size
    slab = math.gcd(TILE_LANES, h)
    tile_kw = dict(hc_mult=m, hidden_size=h)
    row_groups = [pl.ds(r, TILE_ROWS) for r in range(0, tb, TILE_ROWS)]

    for rows in row_groups:
        post_b = _lane_broadcast(post_ref[rows, :], slab)  # per stream j
        comb_b = _lane_broadcast(comb_ref[rows, :], slab)  # row-major (i, j)
        sq = None
        for c in range(0, h, slab):
            sq = _recombine_tile(
                x_ref, res_ref, newres_ref, post_b, comb_b, sq, rows, c, **tile_kw
            )
        sqrsum_ref[rows, :] = jnp.sum(sq, axis=-1, keepdims=True)

    acc = utils.dot_fn_chunks(newres_ref[...], fn_ref[...], hb_ref.shape[1])
    sqr = sqrsum_ref[...]

    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    for rows in row_groups:
        pre_b = _lane_broadcast(pre_mix[rows.start : rows.start + TILE_ROWS], slab)
        for c in range(0, h, slab):
            _collapse_tile(newres_ref, layer_ref, pre_b, rows, c, **tile_kw)

    post_t, comb_t = utils.mhc_post_comb_gates_t(
        acc.T,
        sqr.T,
        sc_ref[0, 1],
        sc_ref[0, 2],
        hb_ref[...].T,
        hc_mult=m,
        hidden_size=h,
        rms_eps=rms_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    post_out_ref[...] = post_t.T
    comb_out_ref[...] = comb_t.T


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
    ),
)
def fused_post_pre_gates(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn_chunks: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = TOKEN_BLOCK,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM,
    gates and in-block collapse.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn_chunks: (3 * hc_mult3, hc_mult * hidden_size), bfloat16 — next
            sublayer's gate projection, split by ``utils.stack_fn3``.
        hc_scale: (3,), float32 — pre/post/comb logit scales.
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon for the in-block gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat: static; the
            post and comb gate constants of ``utils.mhc_pre_gates``.
        token_block_size: most tokens per grid step. Below it, a bucket
            is split into MIN_GRID_STEPS blocks, but none smaller than
            MIN_TOKEN_BLOCK. 256 still fits VMEM at DeepSeek-V4 shapes;
            512 does not.
        gemm_precision: "highest" (3-chunk exact decomposition, the
            default) or "default" (hi-chunk pass only; every gate and
            the collapse then see the hi-only mixes).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post_mix: (num_tokens, hc_mult), float32.
        comb_mix: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        layer2d: (num_tokens, hidden_size), bfloat16.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype
    assert fn_chunks.dtype == jnp.bfloat16, fn_chunks.dtype

    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn_chunks.shape[0] // 3
    assert res2d.shape == (num_tokens, hc_mult * hidden_size)
    assert fn_chunks.shape == (3 * hc_mult3, hc_hidden), (fn_chunks.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    tb, padded_tokens = _select_token_block(num_tokens, token_block_size)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    if gemm_precision == "default":
        fn_chunks = fn_chunks[:hc_mult3]
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    new_res, post_mix, comb_mix, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec(fn_chunks.shape, lambda i: (0, 0)),
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult * hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=[pltpu.VMEM((tb, 1), jnp.float32)],
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, fn_chunks, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, post_mix, comb_mix, layer2d),
    )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_chunks: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = TOKEN_BLOCK,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: one Pallas post+GEMM+gates+collapse kernel.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    new_res2d, post_mix_cur, comb_mix_cur, layer_input_cur = fused_post_pre_gates(
        x2d,
        res2d,
        post2d,
        comb2d,
        fn_chunks,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        token_block_size=token_block_size,
        gemm_precision=gemm_precision,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )


def _head_collapse_tile(newres_ref, out_ref, pre_b, rows, c, *, hc_mult, hidden_size):
    """The head output of one tile at column ``c``. Streams add pairwise,
    ``i`` with ``i + hc_mult / 2`` first, the order XLA's reduce over the
    streams uses, rather than ``_collapse_tile``'s running sum."""
    slab = pre_b[0].shape[1]
    terms = [
        pre_b[i]
        * newres_ref[rows, pl.ds(i * hidden_size + c, slab)].astype(jnp.float32)
        for i in range(hc_mult)
    ]
    while len(terms) > 1:
        half = len(terms) // 2
        terms = [terms[i] + terms[i + half] for i in range(half)]
    out_ref[rows, pl.ds(c, slab)] = terms[0].astype(jnp.bfloat16)


def _post_head_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    out_ref,
    newres_ref,
    norm_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_eps,
):
    """One token block: recombine, RMS norm, gate GEMM, collapse.

    The settled streams stay in VMEM. As the head computes its gates, each
    token's flat streams are RMS-normalized in f32 and the projection takes
    them, and fn, as bf16 in one MXU pass.
    """
    tb = x_ref.shape[0]
    m, h = hc_mult, hidden_size
    slab = math.gcd(TILE_LANES, h)
    tile_kw = dict(hc_mult=m, hidden_size=h)
    row_groups = [pl.ds(r, TILE_ROWS) for r in range(0, tb, TILE_ROWS)]

    for rows in row_groups:
        post_b = _lane_broadcast(post_ref[rows, :], slab)
        comb_b = _lane_broadcast(comb_ref[rows, :], slab)
        sq = None
        for c in range(0, h, slab):
            sq = _recombine_tile(
                x_ref, res_ref, newres_ref, post_b, comb_b, sq, rows, c, **tile_kw
            )
        inv_rms = jax.lax.rsqrt(jnp.sum(sq, axis=-1, keepdims=True) / (m * h) + rms_eps)
        inv_rms_b = jnp.broadcast_to(inv_rms, (TILE_ROWS, slab))
        for c in range(0, m * h, slab):
            new = newres_ref[rows, pl.ds(c, slab)].astype(jnp.float32)
            norm_ref[rows, pl.ds(c, slab)] = (new * inv_rms_b).astype(jnp.bfloat16)

    dn = (((1,), (1,)), ((), ()))
    logits = jax.lax.dot_general(
        norm_ref[...], fn_ref[...], dn, preferred_element_type=jnp.float32
    )
    # The sigmoid as torch's lowers it, 1 / (1 + exp(-z)).
    pre_mix = 1.0 / (1.0 + jnp.exp(-(logits * sc_ref[0, 0] + hb_ref[...]))) + hc_eps
    for rows in row_groups:
        pre_b = _lane_broadcast(pre_mix[rows.start : rows.start + TILE_ROWS], slab)
        for c in range(0, h, slab):
            _head_collapse_tile(newres_ref, out_ref, pre_b, rows, c, **tile_kw)


@functools.partial(
    jax.jit,
    static_argnames=("rms_eps", "hc_eps", "token_block_size", "vmem_limit_bytes"),
)
def mhc_post_head(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    head_fn: jax.Array,
    head_scale: jax.Array,
    head_base: jax.Array,
    rms_eps: float,
    hc_eps: float,
    *,
    token_block_size: int = TOKEN_BLOCK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """The last layer's post and the model head's stream collapse as one
    kernel; the settled streams never reach HBM.

    The head is ``sum_i pre_mix[:, i] * stream_i`` with
    ``pre_mix = sigmoid((rms_norm(flat streams) @ head_fn.T) * head_scale
    + head_base) + hc_eps``.

    Args:
        x: (..., hidden_size), bfloat16 — the last sublayer's output.
        residual: (..., hc_mult, hidden_size), bfloat16.
        post_layer_mix: (..., hc_mult, 1), float32.
        comb_res_mix: (..., hc_mult, hc_mult), float32.
        head_fn: (hc_mult, hc_mult * hidden_size), bfloat16 — the head's
            gate projection, rounded to bf16.
        head_scale: (1,), float32.
        head_base: (hc_mult,), float32.
        rms_eps: static; RMS-norm epsilon of the gates.
        hc_eps: static; additive floor on the sigmoid gates.

    Returns:
        (..., hidden_size), bfloat16.
    """
    assert residual.dtype == jnp.bfloat16, residual.dtype
    assert head_fn.dtype == jnp.bfloat16, head_fn.dtype

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    hc_hidden = hc_mult * hidden_size
    assert head_fn.shape == (hc_mult, hc_hidden), (head_fn.shape, residual.shape)
    # The collapse adds the streams pairwise (_head_collapse_tile).
    assert hc_mult & (hc_mult - 1) == 0, hc_mult

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_hidden)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    num_tokens = x2d.shape[0]

    tb, padded_tokens = _select_token_block(num_tokens, token_block_size)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)
    sc2d = head_scale.reshape(1, 1).astype(jnp.float32)
    hb2d = head_base.reshape(1, hc_mult).astype(jnp.float32)

    out = pl.pallas_call(
        functools.partial(
            _post_head_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec(head_fn.shape, lambda i: (0, 0)),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tb, hc_hidden), jnp.bfloat16),
            pltpu.VMEM((tb, hc_hidden), jnp.bfloat16),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
    )(x2d, res2d, post2d, comb2d, head_fn, sc2d, hb2d)

    (out,) = utils.trim_to(num_tokens, out)
    return out.reshape(*outer_shape, hidden_size)

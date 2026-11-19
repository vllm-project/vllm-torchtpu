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
"""Fused Pallas mHC kernels for GLM 5.3 Flash.

Adapted from `vllm_torchtpu/kernels/deepseek_v4/mhc/` (`utils.py`,
`pre_kernel.py`, and `fused_post_pre_kernel.py`) with three GLM 5.3 Flash
specific changes:
1. The 3 chunk BF16 `fn` decomposition (`fn_hi`, `fn_mid`, `fn_lo`) is
   precomputed once at weight load and passed directly into `pl.pallas_call`,
   so no `split_fn3` XLA fusion runs per mHC site per step.
2. The float32 Sinkhorn gates (`post_mix` and `comb_mix`) are computed in
   kernel over the VMEM resident mix logits, replacing thousands of small
   Sinkhorn reduction, broadcast and slice ops per step.
3. The weighted sublayer RMSNorm is applied in kernel to the collapsed stream
   before it is written to HBM, saving an HBM round trip per mHC site.

This module is pure JAX; the torch-side helpers (e.g. the eager CPU split of
`fn`) live in `layers/adapter/custom_ops/glm53_mhc_tpu.py`.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# TPU TensorCore sublane count is 8; bf16 (8, 128)(2, 1) tiles with a 128
# aligned minor axis are complete at 8 token rows.
SUBLANE = 8

# Explicit scoped VMEM budget matching the repository convention.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

_FUSED_LARGE_TOKEN_BLOCK = 128


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
    """Grid token block and the token count padded to a multiple of it."""
    if num_tokens <= SUBLANE:
        tb = SUBLANE
    elif num_tokens <= 16:
        tb = min(token_block_size, 16)
    else:
        tb = min(token_block_size, round_up(num_tokens, 16))
    if vmem_need is not None:
        while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
            tb //= 2
    return tb, round_up(num_tokens, tb)


def select_fused_token_block(num_tokens: int) -> int:
    """Preferred token block for the fused post and pre kernel."""
    if num_tokens >= _FUSED_LARGE_TOKEN_BLOCK and (
        num_tokens % _FUSED_LARGE_TOKEN_BLOCK == 0
    ):
        return _FUSED_LARGE_TOKEN_BLOCK
    return 32


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Zero pad the leading token axis of each array to `padded_tokens`."""
    pad = padded_tokens - arrays[0].shape[0]
    if pad == 0:
        return arrays
    return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Inverse of `pad_to`: drop the padded trailing token rows."""
    if arrays[0].shape[0] == num_tokens:
        return arrays
    return tuple(a[:num_tokens] for a in arrays)


def split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Three chunk BF16 split of float32 `fn` (8 mantissa bits each)."""
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return (
        fn_hi.astype(jnp.bfloat16),
        fn_mid.astype(jnp.bfloat16),
        fn_lo.astype(jnp.bfloat16),
    )


def _gates_collapse_norm_in_block(
    x_bf: jax.Array,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    nw_ref,
    post_out_ref,
    comb_out_ref,
    normed_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    gemm_precision: str,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """Compute mix GEMM, Sinkhorn gates, stream collapse, and sublayer RMSNorm."""
    m, h = hc_mult, hidden_size
    dn = (((1,), (1,)), ((), ()))
    acc = jax.lax.dot_general(
        x_bf, fn_hi_ref[...], dn, preferred_element_type=jnp.float32
    )
    if gemm_precision == "highest":
        acc = acc + jax.lax.dot_general(
            x_bf, fn_mid_ref[...], dn, preferred_element_type=jnp.float32
        )
        acc = acc + jax.lax.dot_general(
            x_bf, fn_lo_ref[...], dn, preferred_element_type=jnp.float32
        )

    xf = x_bf.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    inv_rms = jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    mixes = acc * inv_rms
    hb = hb_ref[...]

    pre_logits = mixes[:, :m] * sc_ref[0, 0] + hb[:, :m]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    # Collapse the residual streams and apply the weighted sublayer RMSNorm
    # before the Sinkhorn loop so `xf` is dead before `fori_loop` and zero
    # stream VREGs remain live across the loop.
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    # Round collapsed stream to bfloat16 before RMSNorm to match the unfused
    # boundary between `mhc_pre` and `_rms_norm_weighted_jax`.
    lay_bf = lay.astype(jnp.bfloat16)
    lf = lay_bf.astype(jnp.float32)
    var = jnp.mean(lf * lf, axis=-1, keepdims=True)
    lf = lf * jax.lax.rsqrt(var + rms_eps)
    nw = nw_ref[...]
    normed_out_ref[...] = (lf.astype(nw.dtype) * nw).astype(jnp.bfloat16)

    post_logits = mixes[:, m : 2 * m] * sc_ref[0, 1] + hb[:, m : 2 * m]
    post_out_ref[...] = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * m :] * sc_ref[0, 2] + hb[:, 2 * m :]
    nt = x_bf.shape[0] // SUBLANE
    # Pack the `nt = tb // 8` sublane tiles of each matrix entry across lanes
    # into shape `(8, nt)` so each of the 16 entries occupies a single
    # `(8, 128)` VREG regardless of `tb`. Inside `fori_loop` every iteration
    # executes on 16 VREGs with zero cross lane reductions.
    if nt == 1:
        c = [
            [
                jnp.sum(
                    comb_logits[:, i * m + j : i * m + j + 1],
                    axis=-1,
                    keepdims=True,
                )
                for j in range(m)
            ]
            for i in range(m)
        ]
    else:
        c = [
            [
                jnp.concatenate(
                    [
                        comb_logits[
                            g * SUBLANE : (g + 1) * SUBLANE,
                            i * m + j : i * m + j + 1,
                        ]
                        for g in range(nt)
                    ],
                    axis=1,
                )
                for j in range(m)
            ]
            for i in range(m)
        ]
    for i in range(m):
        row_max = c[i][0]
        for j in range(1, m):
            row_max = jnp.maximum(row_max, c[i][j])
        exps = [jnp.exp(c[i][j] - row_max) for j in range(m)]
        row_sum = exps[0]
        for j in range(1, m):
            row_sum = row_sum + exps[j]
        c[i] = [exps[j] / row_sum + hc_sinkhorn_eps for j in range(m)]

    for j in range(m):
        col_sum = c[0][j]
        for i in range(1, m):
            col_sum = col_sum + c[i][j]
        col_denom = col_sum + hc_sinkhorn_eps
        for i in range(m):
            c[i][j] = c[i][j] / col_denom

    def _sinkhorn_body(_, state):
        grid_c = [[state[i * m + j] for j in range(m)] for i in range(m)]
        for i in range(m):
            row_sum = grid_c[i][0]
            for j in range(1, m):
                row_sum = row_sum + grid_c[i][j]
            row_denom = row_sum + hc_sinkhorn_eps
            for j in range(m):
                grid_c[i][j] = grid_c[i][j] / row_denom
        for j in range(m):
            col_sum = grid_c[0][j]
            for i in range(1, m):
                col_sum = col_sum + grid_c[i][j]
            col_denom = col_sum + hc_sinkhorn_eps
            for i in range(m):
                grid_c[i][j] = grid_c[i][j] / col_denom
        return tuple(grid_c[i][j] for i in range(m) for j in range(m))

    flat_c = tuple(c[i][j] for i in range(m) for j in range(m))
    if sinkhorn_repeat > 1:
        flat_c = jax.lax.fori_loop(0, sinkhorn_repeat - 1, _sinkhorn_body, flat_c)
    if nt == 1:
        comb_out_ref[...] = jnp.concatenate(flat_c, axis=1)
    else:
        tile_rows = [
            jnp.concatenate([flat_c[k][:, g : g + 1] for k in range(m * m)], axis=1)
            for g in range(nt)
        ]
        comb_out_ref[...] = jnp.concatenate(tile_rows, axis=0)


def _pre_norm_kernel(
    x_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    nw_ref,
    post_out_ref,
    comb_out_ref,
    normed_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    gemm_precision: str,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """Standalone pre + Sinkhorn + collapse + weighted RMSNorm kernel."""
    _gates_collapse_norm_in_block(
        x_ref[...],
        fn_hi_ref,
        fn_mid_ref,
        fn_lo_ref,
        sc_ref,
        hb_ref,
        nw_ref,
        post_out_ref,
        comb_out_ref,
        normed_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        gemm_precision=gemm_precision,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


def _fused_post_pre_norm_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    nw_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    gemm_precision: str,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """Fused post recombine + pre mix GEMM + Sinkhorn + collapse + RMSNorm."""
    post = post_ref[...]
    comb = comb_ref[...]
    x_bf = x_ref[...]
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    streams_bf = []
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)

    g_bf = jnp.concatenate(streams_bf, axis=1)
    _gates_collapse_norm_in_block(
        g_bf,
        fn_hi_ref,
        fn_mid_ref,
        fn_lo_ref,
        sc_ref,
        hb_ref,
        nw_ref,
        post_out_ref,
        comb_out_ref,
        normed_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        gemm_precision=gemm_precision,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


def mhc_pre_norm_fused(
    residual: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 64,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused Pallas `mhc_pre` + Sinkhorn gates + weighted sublayer RMSNorm.

    Args:
        residual: `(T, hc_mult * hidden_size)` or `(..., hc_mult, hidden_size)`
            in bfloat16.
        fn_hi, fn_mid, fn_lo: Precomputed `(hc_mult3, hc_mult * hidden_size)`
            bfloat16 chunks of `fn`.
        hc_scale: `(3,)` float32 scale for pre, post, and comb logits.
        hc_base: `(hc_mult3,)` float32 bias for pre, post, and comb logits.
        norm_weight: `(hidden_size,)` sublayer RMSNorm weight.
        rms_eps: RMSNorm epsilon.
        hc_pre_eps: Additive epsilon on sigmoid pre gates.
        hc_sinkhorn_eps: Additive epsilon in Sinkhorn normalization.
        hc_post_mult_value: Multiplier on sigmoid post gates (2.0 for GLM 5.3).
        sinkhorn_repeat: Number of Sinkhorn normalization iterations.
    """
    assert residual.dtype == jnp.bfloat16, residual.dtype
    assert fn_hi.dtype == jnp.bfloat16, fn_hi.dtype
    assert fn_mid.dtype == jnp.bfloat16, fn_mid.dtype
    assert fn_lo.dtype == jnp.bfloat16, fn_lo.dtype

    hidden_size = norm_weight.shape[-1]
    hc_mult3, hc_hidden = fn_hi.shape
    hc_mult = hc_hidden // hidden_size
    assert hc_mult * (hc_mult + 2) == hc_mult3, (hc_mult, hc_mult3)
    assert fn_mid.shape == (hc_mult3, hc_hidden)
    assert fn_lo.shape == (hc_mult3, hc_hidden)

    if residual.ndim == 2:
        outer_shape = (residual.shape[0],)
        x2d = residual
    else:
        outer_shape = residual.shape[:-2]
        x2d = residual.reshape(-1, hc_hidden)
    num_tokens = x2d.shape[0]
    assert x2d.shape == (num_tokens, hc_hidden), (x2d.shape, hc_hidden)

    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)
    nw2d = norm_weight.reshape(1, hidden_size)

    def _vmem_need(tb: int) -> int:
        return 2 * (
            tb * hc_hidden * (2 * 2 + 4)
            + tb * hidden_size * 2 * 2
            + 3 * hc_mult3 * hc_hidden * 2
        )

    tb, padded_tokens = select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    (x2d_padded,) = pad_to(padded_tokens, x2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )
    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    post2d, comb2d, normed2d = pl.pallas_call(
        functools.partial(
            _pre_norm_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
            pl.BlockSpec((1, hidden_size), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult * hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
        interpret=interpret,
    )(x2d_padded, fn_hi, fn_mid, fn_lo, sc2d, hb2d, nw2d)

    post2d, comb2d, normed2d = trim_to(num_tokens, post2d, comb2d, normed2d)
    return (
        post2d.reshape(*outer_shape, hc_mult, 1),
        comb2d.reshape(*outer_shape, hc_mult, hc_mult),
        normed2d.reshape(*outer_shape, hidden_size),
    )


def mhc_fused_post_pre_norm(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int | None = None,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused Pallas `mhc_post` + `mhc_pre` + Sinkhorn + weighted RMSNorm."""
    assert residual.dtype == jnp.bfloat16, residual.dtype
    assert fn_hi.dtype == jnp.bfloat16, fn_hi.dtype
    assert fn_mid.dtype == jnp.bfloat16, fn_mid.dtype
    assert fn_lo.dtype == jnp.bfloat16, fn_lo.dtype

    hidden_size = norm_weight.shape[-1]
    hc_mult3, hc_hidden = fn_hi.shape
    hc_mult = hc_hidden // hidden_size
    assert hc_mult * (hc_mult + 2) == hc_mult3, (hc_mult, hc_mult3)
    assert fn_mid.shape == (hc_mult3, hc_hidden)
    assert fn_lo.shape == (hc_mult3, hc_hidden)
    assert gemm_precision in ("highest", "default"), gemm_precision

    residual_orig_shape = residual.shape
    if residual.ndim == 2:
        outer_shape = (residual.shape[0],)
        res2d = residual
    else:
        outer_shape = residual.shape[:-2]
        res2d = residual.reshape(-1, hc_hidden)

    num_tokens = res2d.shape[0]
    x2d = x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(num_tokens, hc_mult)
    comb2d = comb_res_mix.reshape(num_tokens, hc_mult * hc_mult)
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype

    if token_block_size is None:
        token_block_size = select_fused_token_block(num_tokens)

    tb, padded_tokens = select_token_block(num_tokens, token_block_size)
    x2d_pad, res2d_pad, post2d_pad, comb2d_pad = pad_to(
        padded_tokens, x2d, res2d, post2d, comb2d
    )

    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)
    nw2d = norm_weight.reshape(1, hidden_size)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )
    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    new_res2d, post_cur2d, comb_cur2d, normed2d = pl.pallas_call(
        functools.partial(
            _fused_post_pre_norm_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
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
            resident,
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
            pl.BlockSpec((1, hidden_size), lambda i: (0, 0)),
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
        compiler_params=compiler_params,
        interpret=interpret,
    )(
        x2d_pad,
        res2d_pad,
        post2d_pad,
        comb2d_pad,
        fn_hi,
        fn_mid,
        fn_lo,
        sc2d,
        hb2d,
        nw2d,
    )

    new_res2d, post_cur2d, comb_cur2d, normed2d = trim_to(
        num_tokens, new_res2d, post_cur2d, comb_cur2d, normed2d
    )
    return (
        new_res2d.reshape(residual_orig_shape),
        post_cur2d.reshape(*outer_shape, hc_mult, 1),
        comb_cur2d.reshape(*outer_shape, hc_mult, hc_mult),
        normed2d.reshape(*outer_shape, hidden_size),
    )

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
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES

# Most tokens per grid step of the mixes + collapse kernel. Each step
# pushes the resident fn chunks into the MXU once, so a bigger block
# amortizes that over more tokens.
COLLAPSE_TOKEN_BLOCK = 128
# Its token block at small (decode-size) buckets, and the multiple blocks
# round to.
COLLAPSE_MIN_TOKEN_BLOCK = 64
# A bucket is split into at least this many grid steps, so the next block's
# DMA still hides behind compute; smaller buckets keep the minimum block.
COLLAPSE_MIN_GRID_STEPS = 4


def _mixes_kernel(x_ref, fn_ref, mixes_ref, sqrsum_ref):
    """One token block: exact GEMM against the resident stacked fn chunks.

    x is bf16 (its low chunks are zero), so its products with fn's three
    bf16 chunks reproduce HIGHEST.
    """
    x = x_ref[...]
    mixes_ref[...] = utils.dot_fn_chunks(x, fn_ref[...], mixes_ref.shape[1])
    xf = x.astype(jnp.float32)  # f32 needed only for the squared sum
    sqrsum_ref[...] = jnp.sum(xf * xf, axis=-1, keepdims=True)


def _mixes_collapse_kernel(
    x_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    post_ref,
    comb_ref,
    layer_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """Mixes kernel plus the in-block collapse and gates.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. The post and comb gates, Sinkhorn included, are
    per-token too and are emitted in place of the raw mixes.
    """
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    acc = utils.dot_fn_chunks(x, fn_ref[...], hb_ref.shape[1])
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    # Same op order as utils.mhc_pre_gates for bit-level agreement.
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)

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
    post_ref[...] = post_t.T
    comb_ref[...] = comb_t.T


@functools.partial(jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_pre_mixes(
    x2d: jax.Array,
    fn: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """Pallas replacement for the mix GEMM + squared sum of mhc_pre.

    Args:
        x2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        fn: (hc_mult3, hc_mult * hidden_size), float32.
        token_block_size: tokens per grid step. The default (64) keeps the
            worst-case VMEM footprint (bf16 block + f32 cast + resident fn)
            around 14 MB at DeepSeek-V4 shapes.

    Returns:
        mixes: (num_tokens, hc_mult3), float32.
        sqrsum: (num_tokens, 1), float32.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block, transient f32 copy for the squared
        # sum, resident bf16 fn chunks; x2 headroom for spills/scratch
        # (quantized_matmul-style safety factor).
        return 2 * (tb * hc_hidden * (2 * 2 + 4) + 3 * hc_mult3 * hc_hidden * 2)

    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # Zero rows are safe: they produce zero mixes/sqrsum and are sliced
    # off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)

    fn_chunks = utils.stack_fn3(fn)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec(fn_chunks.shape, lambda i: (0, 0))
    mixes, sqrsum = pl.pallas_call(
        _mixes_kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_mult3), lambda i: (i, 0)),
            pl.BlockSpec((tb, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_mult3), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
        ],
        compiler_params=compiler_params,
    )(x2d, fn_chunks)

    return cast(tuple[jax.Array, jax.Array], utils.trim_to(num_tokens, mixes, sqrsum))


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "token_block_size",
        "vmem_limit_bytes",
    ),
)
def mhc_pre_mixes_collapse(
    x2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = COLLAPSE_TOKEN_BLOCK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse and gates: all of ``mhc_pre``.

    ``token_block_size`` is the most tokens per grid step; below it, a
    bucket is split into COLLAPSE_MIN_GRID_STEPS blocks, but none smaller
    than COLLAPSE_MIN_TOKEN_BLOCK.

    Returns (post_mix (T, hc_mult) f32; comb_mix (T, hc_mult * hc_mult)
    f32, row-major (i, j); layer_input2d (T, hidden_size) bf16).
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)
    # hc_mult3 = hc_mult * (hc_mult + 2) -> unique positive hc_mult.
    hc_mult = next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)
    hidden_size = hc_hidden // hc_mult

    fn_chunks = utils.stack_fn3(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block + layer output, transient f32
        # copy, resident bf16 fn chunks; x2 spill/scratch headroom.
        return 2 * (
            tb * hc_hidden * (2 * 2 + 4)
            + tb * hidden_size * 2 * 2
            + 3 * hc_mult3 * hc_hidden * 2
        )

    per_step = utils.round_up(num_tokens, utils.SUBLANE) // COLLAPSE_MIN_GRID_STEPS
    block = min(
        token_block_size,
        max(
            COLLAPSE_MIN_TOKEN_BLOCK,
            per_step // COLLAPSE_MIN_TOKEN_BLOCK * COLLAPSE_MIN_TOKEN_BLOCK,
        ),
    )
    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        block,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # Zero rows are safe: they produce zero layer rows and finite gates,
    # and are sliced off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec(fn_chunks.shape, lambda i: (0, 0))
    post_mix, comb_mix, layer2d = pl.pallas_call(
        functools.partial(
            _mixes_collapse_kernel,
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
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
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
    )(x2d, fn_chunks, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, post_mix, comb_mix, layer2d),
    )


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = COLLAPSE_TOKEN_BLOCK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` as one Pallas kernel: mix GEMM, collapse and gates.

    Same contract as vLLM's ``mhc_pre_torch``:
    residual (..., hc_mult, hidden_size) bf16 in; post_mix (..., hc_mult, 1)
    f32, comb_mix (..., hc_mult, hc_mult) f32, layer_input (..., hidden_size)
    bf16 out.
    """
    assert residual.dtype == jnp.bfloat16
    assert fn.dtype == jnp.float32

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)

    post_mix, comb_mix, layer2d = mhc_pre_mixes_collapse(
        x2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )

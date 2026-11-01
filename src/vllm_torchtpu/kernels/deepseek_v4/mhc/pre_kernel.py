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


def _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref):
    """3-pass exact GEMM: x is bf16 (its low chunks are zero), so three
    bf16 passes against the resident fn chunks reproduce HIGHEST. The
    dimension numbers contract x dim 1 with fn dim 1 (i.e. x @ fn.T) so
    fn needs no transpose on the host side."""
    dn = (((1,), (1,)), ((), ()))
    acc = jax.lax.dot_general(x, fn_hi_ref[...], dn, preferred_element_type=jnp.float32)
    acc = acc + jax.lax.dot_general(
        x, fn_mid_ref[...], dn, preferred_element_type=jnp.float32
    )
    acc = acc + jax.lax.dot_general(
        x, fn_lo_ref[...], dn, preferred_element_type=jnp.float32
    )
    return acc


def _mixes_kernel(x_ref, fn_hi_ref, fn_mid_ref, fn_lo_ref, mixes_ref, sqrsum_ref):
    """One token block: 3-pass exact GEMM against resident fn chunks."""
    x = x_ref[...]
    mixes_ref[...] = _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    xf = x.astype(jnp.float32)  # f32 needed only for the squared sum
    sqrsum_ref[...] = jnp.sum(xf * xf, axis=-1, keepdims=True)


def _mixes_collapse_kernel(
    x_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    post_ref,
    comb_ref,
    layer_ref,
    logits_ref,
    logits_t_ref,
    gates_t_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """Mixes kernel plus the in-block collapse and the post/comb gates.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. The last block runs the Sinkhorn/post/comb gates
    for every token.
    """
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    acc = _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    # Same op order as utils.mhc_pre_gates for bit-level agreement.
    scaled = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)
    utils.post_comb_gates_epilogue(
        scaled,
        sc_ref,
        hb_ref,
        post_ref,
        comb_ref,
        logits_ref,
        logits_t_ref,
        gates_t_ref,
        hc_mult=m,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


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

    fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    mixes, sqrsum = pl.pallas_call(
        _mixes_kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
            resident,
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
    )(x2d, fn_hi, fn_mid, fn_lo)

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
def mhc_pre_gates_collapse(
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
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse + gates: all of ``mhc_pre`` in one kernel.

    Returns (post_mix2d (T, hc_mult) f32; comb_mix2d (T, hc_mult**2) f32,
    row-major (i, j); layer_input2d (T, hidden_size) bf16).
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)
    # hc_mult3 = hc_mult * (hc_mult + 2) -> unique positive hc_mult.
    hc_mult = next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)
    hidden_size = hc_hidden // hc_mult

    fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
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

    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # Zero rows are safe: they produce zero mixes/sqrsum/layer rows and
    # are sliced off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)

    # The last block's gates read every block's logits.
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    gate_specs, gate_shapes, gate_scratch = utils.post_comb_gates_specs(
        padded_tokens, hc_mult
    )
    post2d, comb2d, layer2d = pl.pallas_call(
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
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            *gate_specs,
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            *gate_shapes,
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=gate_scratch,
        compiler_params=compiler_params,
    )(x2d, fn_hi, fn_mid, fn_lo, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, post2d, comb2d, layer2d),
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
    token_block_size: int = 64,
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

    post_mix, comb_mix, layer2d = mhc_pre_gates_collapse(
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

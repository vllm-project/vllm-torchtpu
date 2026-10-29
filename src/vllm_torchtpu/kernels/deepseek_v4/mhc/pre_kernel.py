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

from vllm_torchtpu.kernels.deepseek_v4.mhc import gates_kernel, utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES

# Tokens per grid step of the single-stream pre: its (tb, hidden) block and
# K=hidden GEMM are a quarter of the full pre's, so it takes larger blocks.
SINGLE_STREAM_TOKEN_BLOCK = 256


def _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref):
    """3-pass exact GEMM: x is bf16 (its low chunks are zero), so three
    bf16 passes against the resident fn chunks reproduce HIGHEST. The
    dimension numbers contract x dim 1 with fn dim 1 (i.e. x @ fn.T) so
    fn needs no transpose on the host side. The three chunks are stacked
    into one N=3*hc_mult3 GEMM so x streams through the MXU once."""
    dn = (((1,), (1,)), ((), ()))
    fn_all = jnp.concatenate([fn_hi_ref[...], fn_mid_ref[...], fn_lo_ref[...]])
    acc3 = jax.lax.dot_general(x, fn_all, dn, preferred_element_type=jnp.float32)
    n = fn_hi_ref.shape[0]
    return (acc3[:, :n] + acc3[:, n : 2 * n]) + acc3[:, 2 * n :]


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
    mix_a_ref,
    mix_b_ref,
    layer_ref,
    *scratch,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    gates,
    single_stream,
):
    """Mixes kernel plus the in-block collapse.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. Without ``gates``, the Sinkhorn/post/comb gates
    stay in XLA on the raw mixes and squared sums (``mix_a_ref``/
    ``mix_b_ref``); with them, the kernel stages the gate logits and, on
    its last step, writes the post and comb gates there instead.

    With ``single_stream``, the (tb, hidden_size) block is one stream
    standing for all hc_mult identical ones (the embedding before any
    post) and fn arrives folded over the streams
    (``utils.fold_fn_streams``): the GEMM, the squared sum and the
    collapse run once on the stream instead of on hc_mult copies of it.
    """
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    acc = _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    if single_stream:
        sqr = sqr * m
    if gates is None:
        mix_a_ref[...] = acc
        mix_b_ref[...] = sqr
    else:
        gates_kernel.stage_logits(
            scratch[0],
            acc,
            sqr,
            sc_ref,
            hb_ref,
            hc_mult=m,
            hidden_size=h,
            rms_eps=rms_eps,
        )
    # Same op order as utils.mhc_pre_gates for bit-level agreement.
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    if single_stream:
        weight = pre_mix[:, 0:1]
        for i in range(1, m):
            weight = weight + pre_mix[:, i : i + 1]
        lay = weight * xf
    else:
        lay = pre_mix[:, 0:1] * xf[:, :h]
        for i in range(1, m):
            lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)

    if gates is not None:
        pl.when(pl.program_id(0) == pl.num_programs(0) - 1)(
            lambda: gates_kernel.gate_staged(
                *scratch, mix_a_ref, mix_b_ref, hc_mult=m, consts=gates
            )
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
        "token_block_size",
        "vmem_limit_bytes",
        "gates",
        "fn_folded",
    ),
)
def mhc_pre_mixes_collapse(
    x2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates: gates_kernel.GateConsts | None = None,
    fn_folded: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse: the full memory-bound half of ``mhc_pre``.

    ``x2d`` is (T, hc_mult * hidden_size); with ``fn_folded``, it is
    (T, hidden_size), the one stream all hc_mult streams are copies of,
    and ``fn`` is folded over the streams (``utils.fold_fn_streams``).

    Returns (mixes (T, hc_mult3) f32 — raw, unscaled, for the XLA gates;
    sqrsum (T, 1) f32; layer_input2d (T, hidden_size) bf16). With
    ``gates``, post_mix (T, hc_mult) and comb_mix (T, hc_mult**2), row-major
    (i, j), replace mixes and sqrsum. ``fn`` may be given as its bf16
    chunks, see ``utils.fn_chunk_operands``.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    hc_mult3, hc_hidden = fn.shape[-2:]
    hc_mult = utils.hc_mult_of(hc_mult3)
    hidden_size = hc_hidden if fn_folded else hc_hidden // hc_mult
    num_tokens, x_width = x2d.shape
    assert x_width == hc_hidden, (fn.shape, x2d.shape)

    fn_chunks, fn_specs = utils.fn_chunk_operands(fn)
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

    mix_out = gates_kernel.mix_outputs(tb, padded_tokens, hc_mult, gates)
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=(mix_out.semantics,),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    mixes, sqrsum, layer2d = pl.pallas_call(
        functools.partial(
            _mixes_collapse_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
            single_stream=fn_folded,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, x_width), lambda i: (i, 0)),
            *fn_specs,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[*mix_out.specs, pl.BlockSpec((tb, hidden_size), lambda i: (i, 0))],
        out_shape=[
            *mix_out.shapes,
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=mix_out.scratch,
        compiler_params=compiler_params,
    )(x2d, *fn_chunks, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, mixes, sqrsum, layer2d),
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
    """``mhc_pre`` with the mix GEMM, the collapse and the gates/Sinkhorn
    in one Pallas kernel.

    Same contract as vLLM's ``mhc_pre_torch``:
    residual (..., hc_mult, hidden_size) bf16 in; post_mix (..., hc_mult, 1)
    f32, comb_mix (..., hc_mult, hc_mult) f32, layer_input (..., hidden_size)
    bf16 out.
    """
    assert residual.dtype == jnp.bfloat16

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
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
        gates=gates_kernel.GateConsts(
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )


def mhc_pre_single_stream(
    x: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = SINGLE_STREAM_TOKEN_BLOCK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` of hc_mult identical streams, given the one stream.

    x: (..., hidden_size) bf16 -- what ``mhc_pre`` would see repeated
    hc_mult times on the stream axis. fn: the pre's fn folded over the
    streams, (hc_mult3, hidden_size) f32 or its (3, hc_mult3, hidden_size)
    bf16 chunks (``utils.fold_fn_streams``). Same outputs as ``mhc_pre``
    up to f32 reassociation: the stream sums are taken in fn and in the
    gate weights instead of over hc_mult copies of x.
    """
    assert x.dtype == jnp.bfloat16

    outer_shape = x.shape[:-1]
    hidden_size = x.shape[-1]
    hc_mult = utils.hc_mult_of(fn.shape[-2])
    post_mix, comb_mix, layer2d = mhc_pre_mixes_collapse(
        x.reshape(-1, hidden_size),
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
        gates=gates_kernel.GateConsts(
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        fn_folded=True,
    )
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )

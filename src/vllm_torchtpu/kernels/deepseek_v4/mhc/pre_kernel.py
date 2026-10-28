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
    *refs,
    n_fn,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    gates,
    norm_eps,
):
    """Mixes kernel plus the in-block collapse.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. The Sinkhorn/post/comb gates stay in
    XLA on the raw ``mixes`` output (gate_a/gate_b = mixes/sqrsum), unless
    ``gates`` is set: then they run in-block too (gate_a/gate_b =
    post_mix/comb_mix). refs: [norm_w_ref] (only with ``norm_eps``), then
    the outputs.

    A (tb, hidden_size) block is one stream standing for all hc_mult
    identical ones (the embedding before any post); it is repeated in VMEM
    so the GEMM and the sums see the values the repeated HBM copy holds.
    """
    fn_refs, (sc_ref, hb_ref), refs = refs[:n_fn], refs[n_fn : n_fn + 2], refs[n_fn + 2 :]
    *norm_w, gate_a_ref, gate_b_ref, layer_ref = refs
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    if x.shape[1] == h:
        x = jnp.concatenate([x] * m, axis=1)
    acc = utils.mix_gemm(x, fn_refs, m * (m + 2), "highest")
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    if gates is None:
        gate_a_ref[...] = acc
        gate_b_ref[...] = sqr
        # Same op order as utils.mhc_pre_gates for bit-level agreement.
        scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
        pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    else:
        pre_mix, gate_a_ref[...], gate_b_ref[...] = utils.gates_in_block(
            acc,
            sqr,
            sc_ref,
            hb_ref,
            hc_mult=m,
            hidden_size=h,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
        )
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = utils.finish_layer_input(lay, *norm_w or [None], norm_eps)


@functools.partial(jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_pre_mixes(
    x2d: jax.Array,
    fn: jax.Array | utils.SplitFn,
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
        grid=(pl.cdiv(padded_tokens, tb),),
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
        "norm_eps",
    ),
)
def mhc_pre_mixes_collapse(
    x2d: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates: utils.GateConsts | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse: the full memory-bound half of ``mhc_pre``.

    ``x2d`` is (T, hc_mult * hidden_size), or (T, hidden_size) when all
    hc_mult streams are the same single stream.

    Returns (mixes (T, hc_mult3) f32 — raw, unscaled, for the XLA gates;
    sqrsum (T, 1) f32; layer_input2d (T, hidden_size) bf16). With
    ``gates``, mixes/sqrsum are replaced by the in-block post_mix
    (T, hc_mult) and comb_mix (T, hc_mult**2). With ``norm_weight``/
    ``norm_eps``, layer_input2d is RMS-normalized.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    hc_mult3, hc_hidden = fn.shape
    # hc_mult3 = hc_mult * (hc_mult + 2) -> unique positive hc_mult.
    hc_mult = next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)
    hidden_size = hc_hidden // hc_mult
    num_tokens, x_width = x2d.shape
    assert x_width in (hc_hidden, hidden_size), (fn.shape, x2d.shape)

    fn_ops = utils.fn_operands(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block + layer output, transient f32
        # copy, resident bf16 fn chunks; x2 spill/scratch headroom. Also
        # bounds a one-stream block: its x is a quarter the size, plus the
        # repeated bf16 copy.
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
        whole_small_batch=gates is not None,
    )
    # Zero rows are safe: they produce zero mixes/sqrsum/layer rows and
    # are sliced off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    assert (norm_weight is None) == (norm_eps is None)
    norm_args = [] if norm_weight is None else [norm_weight.reshape(1, hidden_size)]
    norm_specs = [pl.BlockSpec((1, hidden_size), lambda i: (0, 0)) for _ in norm_args]

    gate_widths = (hc_mult3, 1) if gates is None else (hc_mult, hc_mult * hc_mult)
    gate_a, gate_b, layer2d = pl.pallas_call(
        functools.partial(
            _mixes_collapse_kernel,
            n_fn=len(fn_ops),
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
            norm_eps=norm_eps,
        ),
        grid=(pl.cdiv(padded_tokens, tb),),
        in_specs=[
            pl.BlockSpec((tb, x_width), lambda i: (i, 0)),
            *(pl.BlockSpec(f.shape, lambda i: (0, 0)) for f in fn_ops),
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
            *norm_specs,
        ],
        out_specs=[
            *(pl.BlockSpec((tb, w), lambda i: (i, 0)) for w in gate_widths),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            *(
                jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32)
                for w in gate_widths
            ),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, *fn_ops, sc2d, hb2d, *norm_args)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, gate_a, gate_b, layer2d),
    )


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
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
    gates_in_kernel: bool = False,
    gates_max_tokens: int | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` with the mix GEMM AND the collapse in Pallas; only the
    tiny gates/Sinkhorn stay in XLA (or, with ``gates_in_kernel`` and up to
    ``utils.GATES_IN_KERNEL_MAX_TOKENS`` tokens, they move into the kernel).

    Same contract as vLLM's ``mhc_pre_torch``:
    residual (..., hc_mult, hidden_size) bf16 in; post_mix (..., hc_mult, 1)
    f32, comb_mix (..., hc_mult, hc_mult) f32, layer_input (..., hidden_size)
    bf16 out. The gates recompute ``pre_mix`` on the raw mixes; its cost is
    a few tiny VPU ops and keeping ``utils.mhc_pre_gates`` shared and
    unsliced is worth more than removing them.
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)
    return _pre_from_2d(
        x2d,
        outer_shape,
        hc_mult,
        hidden_size,
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
        gates_in_kernel=gates_in_kernel,
        gates_max_tokens=gates_max_tokens,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def mhc_pre_single_stream(
    x: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
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
    gates_in_kernel: bool = False,
    gates_max_tokens: int | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` of hc_mult identical streams, given the one stream.

    x: (..., hidden_size) bf16 -- what ``mhc_pre`` would see repeated
    hc_mult times on the stream axis, without the repeated copy in HBM.
    Same outputs up to the gates' last f32 ulp: the one-stream block is
    tiled differently by Mosaic (bit-identical in interpret mode).
    """
    assert x.dtype == jnp.bfloat16

    outer_shape = x.shape[:-1]
    hidden_size = x.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    return _pre_from_2d(
        x.reshape(-1, hidden_size),
        outer_shape,
        hc_mult,
        hidden_size,
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
        gates_in_kernel=gates_in_kernel,
        gates_max_tokens=gates_max_tokens,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def _pre_from_2d(
    x2d: jax.Array,
    outer_shape: tuple[int, ...],
    hc_mult: int,
    hidden_size: int,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int,
    vmem_limit_bytes: int,
    gates_in_kernel: bool,
    gates_max_tokens: int | None,
    norm_weight: jax.Array | None,
    norm_eps: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The kernel plus gates of ``mhc_pre`` on flattened streams."""
    gates = utils.gate_consts(
        x2d.shape[0],
        gates_in_kernel,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_max_tokens,
    )
    gate_a, gate_b, layer2d = mhc_pre_mixes_collapse(
        x2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
        gates=gates,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    if gates is not None:
        post_mix, comb_mix = gate_a, gate_b
    else:
        _, post_mix, comb_mix = utils.mhc_pre_gates(
            gate_a,
            gate_b,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )

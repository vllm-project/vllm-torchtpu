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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES

# Scoped VMEM for decode-sized seams (see VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS).
LOW_VMEM_LIMIT_BYTES = 32 * 1024 * 1024
LOW_VMEM_CAPS = (0, 1, 2, 4, 8, 16, 32)


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    *refs,
    n_fn,
    hc_mult,
    hidden_size,
    gemm_precision,
    rms_eps,
    hc_pre_eps,
    gates,
    norm_eps,
):
    """One token block: recombine, store once, 3-pass GEMM, collapse.

    refs: [norm_w_ref] (only with ``norm_eps``), then the outputs.
    gate_a/gate_b are (mixes, sqrsum) when ``gates`` is None, else the
    in-block (post_mix, comb_mix).
    """
    fn_refs, (sc_ref, hb_ref), refs = refs[:n_fn], refs[n_fn : n_fn + 2], refs[n_fn + 2 :]
    *norm_w, newres_ref, gate_a_ref, gate_b_ref, layer_ref = refs
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    # Keep the streams (and x) in bf16 and upcast per use: transient f32
    # values instead of 4 held f32 copies halves the resident working set,
    # leaving VMEM headroom for the pipeline to prefetch the next block.
    # bf16 -> f32 casts are exact, so results are unchanged.
    x_bf = x_ref[...]  # (tb, hidden) bf16
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    streams_bf = []
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
        # bf16 residual that post stored, and parity requires matching it.
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)

    g_bf = jnp.concatenate(streams_bf, axis=1)  # (tb, hc_mult * hidden_size)
    acc = utils.mix_gemm(
        g_bf, fn_refs, hc_mult * (hc_mult + 2), gemm_precision
    )
    gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)

    m, h = hc_mult, hidden_size
    if gates is None:
        gate_a_ref[...] = acc
        gate_b_ref[...] = sqr
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
    lay = pre_mix[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    layer_ref[...] = utils.finish_layer_input(lay, *norm_w or [None], norm_eps)


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
        "gates",
        "norm_eps",
        "small_block",
        "fn_chunks",
        "fn_hbm",
    ),
)
def fused_post_pre_mixes(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates: utils.GateConsts | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
    small_block: int = utils.SUBLANE,
    fn_chunks: int = 3,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM
    and in-block collapse.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales (the kernel
            uses only the pre entry; the rest feed the XLA gates).
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon for the in-block pre gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        token_block_size: tokens per grid step. Default 32 keeps the
            worst-case VMEM footprint (f32 old streams + bf16 blocks +
            the three resident bf16 fn chunks) near 12 MB at DeepSeek-V4
            shapes; 64 also fits.
        gemm_precision: "highest" (3-pass exact decomposition, the
            default) or "default" (hi-chunk pass only; pre_mix and the
            collapse then also see the hi-only mixes).
        gates: when set, the post/comb gates (incl. Sinkhorn) are computed
            in-block and returned in place of (mixes, sqrsum).
        norm_weight, norm_eps: when set, layer2d is returned RMS-normalized
            with this (hidden_size,) weight (see utils.finish_layer_input).
        small_block: with in-kernel gates, a batch of at most this many
            tokens (8) runs as one block of exactly its rows instead of the
            16-row bf16 tile (see the call site).
        fn_chunks: with a stacked fn (``utils.CatFn``), how many of its bf16
            chunks the kernel reads: 3 (exact f32 product) or 2 (hi + mid,
            16 mantissa bits: the lo rows are never DMA'd or pushed).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        mixes: (num_tokens, hc_mult3), float32 — raw, unscaled, for the
            XLA gates.
        sqrsum: (num_tokens, 1), float32.
        layer2d: (num_tokens, hidden_size), bfloat16.
        With ``gates``, mixes/sqrsum are replaced by post_mix
        (num_tokens, hc_mult) and comb_mix (num_tokens, hc_mult**2), f32.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype

    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn.shape[0]
    assert res2d.shape == (num_tokens, hc_mult * hidden_size)
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    # This kernel keeps a fixed block: its default (32) already fits the
    # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    tb, padded_tokens = utils.select_token_block(
        num_tokens, token_block_size, whole_small_batch=gates is not None
    )
    if gates is not None and num_tokens <= small_block < tb:
        # Decode batches of <= 8 rows: one block of exactly the array's rows
        # (legal as it equals the full dims). The body's f32 VPU work is per
        # 8-row vreg, so the 16-row overhang block doubled it for nothing:
        # T=8 seam 6.42 -> 5.78 us (tpu7x, served shapes).
        tb, padded_tokens = num_tokens, num_tokens
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    fn_ops = utils.fn_operands(fn)
    if fn_hbm:
        # Keep XLA from prefetching fn into VMEM ahead of the call: where the
        # op before the seam leaves no room to hide that copy, the seam's own
        # pipeline streaming fn from HBM costs less than the exposed copy.
        fn_ops = tuple(pltpu.with_memory_space_constraint(f, pltpu.HBM) for f in fn_ops)
    assert fn_chunks in (2, 3), fn_chunks
    # Stacked fn with fn_chunks=2: the BlockSpec covers only the hi and mid
    # chunk rows of the (3 * M3, K) array, so the lo chunk (a third of the
    # kernel's dominant bytes and MXU weight pushes) is never read.
    fn_rows = fn_chunks * hc_mult3 if len(fn_ops) == 1 else None
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    assert (norm_weight is None) == (norm_eps is None)
    norm_args = [] if norm_weight is None else [norm_weight.reshape(1, hidden_size)]
    norm_specs = [pl.BlockSpec((1, hidden_size), lambda i: (0, 0)) for _ in norm_args]

    gate_widths = (hc_mult3, 1) if gates is None else (hc_mult, hc_mult * hc_mult)
    new_res, gate_a, gate_b, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            n_fn=len(fn_ops),
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
            norm_eps=norm_eps,
        ),
        grid=(pl.cdiv(padded_tokens, tb),),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            *(
                pl.BlockSpec((fn_rows or f.shape[0], f.shape[1]), lambda i: (0, 0))
                for f in fn_ops
            ),
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
            *norm_specs,
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            *(pl.BlockSpec((tb, w), lambda i: (i, 0)) for w in gate_widths),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            *(
                jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32)
                for w in gate_widths
            ),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, *fn_ops, sc2d, hb2d, *norm_args)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, gate_a, gate_b, layer2d),
    )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int | None = None,
    gates_in_kernel: bool = False,
    gates_max_tokens: int | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
    small_block: int | None = None,
    fn_chunks: int | None = None,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: Pallas post+GEMM+collapse kernel, gates in an XLA
    epilogue or (``gates_in_kernel``, up to ``gates_max_tokens`` tokens --
    default ``utils.GATES_IN_KERNEL_MAX_TOKENS``, 0 = no cap) in the kernel.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur);
    with ``norm_weight``/``norm_eps``, layer_input_cur is RMS-normalized.
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    if vmem_limit_bytes is None:
        # A seam of at most VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS rows needs
        # well under 32 MiB, and the default 100 MiB (clamped to 64 MiB -
        # 64 KiB) scoped reservation makes XLA evict the neighbours' VMEM
        # buffers around every call (the all-reduce output it consumes, the
        # next sublayer's prefetched weights) and stops it prefetching this
        # call's own operands.
        cap = envs.VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS
        if cap not in LOW_VMEM_CAPS:
            raise ValueError(
                f"VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS={cap}: must be 0 or a power "
                f"of 2 up to {LOW_VMEM_CAPS[-1]} (decode-sized seams fit 32 MiB)"
            )
        low = cap > 0 and x2d.shape[0] <= cap
        vmem_limit_bytes = LOW_VMEM_LIMIT_BYTES if low else DEFAULT_VMEM_LIMIT_BYTES
        # The fn prefetch the knob below avoids only happens with the low
        # reservation; larger seams keep the default placement.
        fn_hbm = fn_hbm and low
    if small_block is None:
        small_block = envs.VLLM_TPU_DSV4_MHC_SMALL_BLOCK
    if fn_chunks is None:
        fn_chunks = envs.VLLM_TPU_DSV4_MHC_FN_CHUNKS
    gates = utils.gate_consts(
        x2d.shape[0],
        gates_in_kernel,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_max_tokens,
    )
    new_res2d, gate_a, gate_b, layer_input_cur = fused_post_pre_mixes(
        x2d,
        res2d,
        post2d,
        comb2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        token_block_size=token_block_size,
        gemm_precision=gemm_precision,
        vmem_limit_bytes=vmem_limit_bytes,
        gates=gates,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        small_block=small_block,
        fn_chunks=fn_chunks,
        fn_hbm=fn_hbm,
    )
    if gates is not None:
        post_mix_cur, comb_mix_cur = gate_a, gate_b
    else:
        _, post_mix_cur, comb_mix_cur = utils.mhc_pre_gates(
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
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )

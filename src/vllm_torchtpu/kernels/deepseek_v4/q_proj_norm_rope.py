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
"""The query's fp8 projection with its per-head RMSNorm and RoPE fused in.

Computes what ``fused_quantize_matmul`` followed by ``rope.qnorm_rope``
compute, but the un-normalized bf16 query never goes to HBM and back. The
matmul is ``fused_quantize_matmul``'s: the first grid step quantizes the
resident activation and every step multiplies it by one output tile of the
weight, here a whole number of heads. Each head's columns are rounded to bf16
as the plain matmul stores them, then normed and rotated by ``qnorm_rope``'s
own ops. Each token's cos/sin lanes come in already gathered: the o-projection's
inverse-rotation rows, which the first step turns into the forward rotation
once for all steps.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import LANE, _qnorm_rope_rows
from vllm_torchtpu.kernels.quantized_matmul import fused_quantize_matmul
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    get_device_vmem_limit,
)


def _tile_n(m: int, k: int, n: int, itemsize: int, head_dim: int) -> int | None:
    """``fused_quantize_matmul``'s output tile width, in whole heads."""
    # Each token's cos and sin lanes, f32.
    cos_sin_bytes = m * 2 * LANE * 4
    return fused_quantize_matmul.output_tile_n(
        m,
        k,
        n,
        itemsize,
        get_device_vmem_limit(),
        multiple_of=head_dim,
        extra_vmem_bytes=cos_sin_bytes,
    )


def applies(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    """Whether ``q_proj_norm_rope`` takes these operands, the projection's
    output being heads of ``head_dim`` side by side with their last
    ``rotary_dim`` channels roped."""
    m, k = x.shape
    n = w_q.shape[1]
    return (
        fused_quantize_matmul.applies(x, w_q, w_scale)
        and head_dim % LANE == 0
        and n % head_dim == 0
        and rotary_dim % 2 == 0
        and rotary_dim <= LANE
        and _tile_n(m, k, n, x.dtype.itemsize, head_dim) is not None
    )


def _kernel(
    # inputs
    x_hbm_ref: jax.Ref,  # (M, K) bf16, HBM
    w_ref: jax.Ref,  # (K, tile_n) fp8
    w_scale_ref: jax.Ref,  # (N,) f32, resident
    cos_sin_hbm_ref: jax.Ref,  # (M, 2 * LANE) f32, HBM
    # outputs
    out_ref: jax.Ref,  # (M, tile_n)
    *refs: jax.Ref,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    return_quantized: bool,
):
    if return_quantized:
        # The (M, K) fp8 and (M, 1) f32 outputs, and their (2,) DMA semaphores.
        x_q_out_ref, x_scale_out_ref, *refs, out_sems = refs
    # cos_sin_ref: (M, 2 * LANE) f32, each token's forward cos and sin lanes.
    x_ref, x_q_ref, x_scale_ref, sems, cos_sin_ref, cos_sin_sem = refs
    step = pl.program_id(0)

    def quantized_copies():
        return [
            pltpu.make_async_copy(x_q_ref, x_q_out_ref, out_sems.at[0]),
            pltpu.make_async_copy(x_scale_ref, x_scale_out_ref, out_sems.at[1]),
        ]

    @pl.when(step == 0)
    def _():
        copies = fused_quantize_matmul.start_activation_copies(x_hbm_ref, x_ref, sems)
        # Lands while the activation is quantized.
        cos_sin_copy = pltpu.make_async_copy(cos_sin_hbm_ref, cos_sin_ref, cos_sin_sem)
        cos_sin_copy.start()
        fused_quantize_matmul.quantize_activation(x_ref, x_q_ref, x_scale_ref, copies)
        if return_quantized:
            # Written back under the remaining steps' matmuls.
            for copy in quantized_copies():
                copy.start()
        cos_sin_copy.wait()
        # The inverse rotation's sin is the forward one negated on the roped
        # lanes; the NoPE lanes keep the forward rotation's +0. This is exactly
        # ``_cos_sin_lanes(..., inverse=False)``'s sin, so ``qnorm_rope``'s.
        inverse_sin = cos_sin_ref[:, LANE:]
        lane = jax.lax.broadcasted_iota(jnp.int32, inverse_sin.shape, 1)
        cos_sin_ref[:, LANE:] = jnp.where(lane >= LANE - rotary_dim, -inverse_sin, 0.0)

    tile_n = out_ref.shape[1]

    def project(h):
        """fused_quantize_matmul's dot, scaling and bf16 store of head h."""
        cols = pl.ds(h * head_dim, head_dim)
        acc = jax.lax.dot_general(
            x_q_ref[...],
            w_ref[:, cols],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        acc = acc * x_scale_ref[...]
        first = pl.multiple_of(step * tile_n + h * head_dim, LANE)
        acc = acc * w_scale_ref[pl.ds(first, head_dim)][None, :]
        return acc.astype(out_ref.dtype)

    num_heads = tile_n // head_dim
    # Head h + 1's dot is issued ahead of head h's norm and rotation, so the
    # MXU has work while the VPU runs the epilogue.
    q_next = project(0)
    for h in range(num_heads):
        q = q_next
        if h + 1 < num_heads:
            q_next = project(h + 1)
        # The bf16 query read back as qnorm_rope does.
        y = _qnorm_rope_rows(
            q.astype(jnp.float32),
            cos_sin_ref[:, :LANE],
            cos_sin_ref[:, LANE:],
            eps=eps,
        )
        out_ref[:, pl.ds(h * head_dim, head_dim)] = y.astype(out_ref.dtype)

    if return_quantized:

        @pl.when(step == pl.num_programs(0) - 1)
        def _():
            for copy in quantized_copies():
                copy.wait()


@functools.partial(
    jax.jit, static_argnames=("head_dim", "rotary_dim", "eps", "return_quantized")
)
def q_proj_norm_rope(
    x: jax.Array,  # [M, K] bf16
    w_q: jax.Array,  # [K, N] fp8
    w_scale: jax.Array,  # [N]
    cos_sin: jax.Array,  # [M, 2 * LANE] float32
    *,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    return_quantized: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """``qnorm_rope(fused_quantize_matmul(x, w_q, w_scale).reshape(M, -1,
    head_dim), positions, cos_sin_cache, eps=eps).reshape(M, N)`` where
    ``applies`` holds.

    ``cos_sin`` is ``o_projection.gather_cos_sin(positions, table)`` for
    ``table = o_projection.cos_sin_table(cos_sin_cache, inverse=True, ...)``:
    the rows the o-projection reads, so one gather serves both.

    With ``return_quantized``, also returns the fp8 activation and its (M, 1)
    row scales, as ``fused_quantize_matmul`` does.
    """
    assert applies(x, w_q, w_scale, head_dim, rotary_dim), (
        x.shape,
        w_q.shape,
        head_dim,
        rotary_dim,
    )
    m, k = x.shape
    n = w_q.shape[1]
    assert cos_sin.shape == (m, 2 * LANE) and cos_sin.dtype == jnp.float32
    tile_n = _tile_n(m, k, n, x.dtype.itemsize, head_dim)
    assert tile_n is not None
    w_scale = w_scale.astype(jnp.float32)
    out_shape = [jax.ShapeDtypeStruct((m, n), x.dtype)]
    out_specs = [pl.BlockSpec((m, tile_n), lambda j: (0, j))]
    scratch_shapes = [
        pltpu.VMEM((m, k), x.dtype),
        pltpu.VMEM((m, k), w_q.dtype),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.SemaphoreType.DMA((m // fused_quantize_matmul.SUB_M,)),
        pltpu.VMEM((m, 2 * LANE), jnp.float32),
        pltpu.SemaphoreType.DMA(()),
    ]
    if return_quantized:
        out_shape += [
            jax.ShapeDtypeStruct((m, k), w_q.dtype),
            jax.ShapeDtypeStruct((m, 1), jnp.float32),
        ]
        out_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
        scratch_shapes.append(pltpu.SemaphoreType.DMA((2,)))
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            return_quantized=return_quantized,
        ),
        out_shape=out_shape,
        grid=(n // tile_n,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((k, tile_n), lambda j: (0, j)),
            # Whole: a rank-1 block must be a power of two or 1024-aligned.
            pl.BlockSpec((n,), lambda j: (0,)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            # Steps after the first read the activation and cos/sin the first
            # one wrote.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=get_device_vmem_limit(),
        ),
        name="q_proj_norm_rope",
    )(x, w_q, w_scale, cos_sin)
    return tuple(outs) if return_quantized else outs[0]

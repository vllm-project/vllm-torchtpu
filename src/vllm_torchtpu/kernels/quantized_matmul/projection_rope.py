# Copyright 2025 Google LLC
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

"""FP8 projection and interleaved RoPE with pipelined weights and output.

Preserves per-row activation quantization, scale order, and BF16 rounding
before RoPE. The specialization targets the measured v7 decode geometry.
Positions must index valid rows of the supplied rotary cache.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import _cos_sin_lanes, _lane_gather
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    PROJECTION_ROPE_TUNING,
    get_tpu_version,
)

_WEIGHT_COLUMNS, _OUTPUT_ROWS, _VMEM_BUDGET = PROJECTION_ROPE_TUNING[
    (7, 288, 1280, 32768)
]


def is_supported(
    x: jax.Array,
    w: jax.Array,
    scale: jax.Array,
    pos: jax.Array,
    cache: jax.Array,
) -> bool:
    """Whether concrete lowering inputs match the qualified specialization."""
    return (
        x.shape == (288, 1280)
        and w.shape == (1280, 32768)
        and scale.shape == (32768,)
        and pos.shape == (288,)
        and cache.shape == (98816, 64)
        and x.dtype == jnp.bfloat16
        and w.dtype == jnp.float8_e4m3fn
        and scale.dtype == jnp.float32
        and pos.dtype == jnp.int32
        and cache.dtype == jnp.float32
        and get_tpu_version() == 7
    )


def quant_kernel(x, q, scales):
    @pl.loop(0, 9, unroll=False)
    def rows(i):
        a = x[pl.ds(i * 32, 32), :].astype(jnp.float32)
        maximum = jnp.max(jnp.abs(a), axis=1, keepdims=True)
        scale = maximum * jnp.float32(1.0 / 448.0)
        scale = jnp.where(scale == 0, 1.0, scale)
        q[pl.ds(i * 32, 32), :] = jnp.clip(a / scale, -448.0, 448.0).astype(
            jnp.float8_e4m3fn
        )
        scales[pl.ds(i * 32, 32), :] = scale


def kernel(
    pos,
    x,
    w,
    wscale_hbm,
    cache,
    out,
    weights,
    wscale,
    gathered,
    matrix,
    obuf,
    coeff,
    wsems,
    ssem,
    gsem,
    osems,
    q,
    xscale,
    xbuf,
    xsem,
):
    def weight_issue(i):
        with jax.named_scope("fused_weight_issue"):
            pltpu.make_async_copy(
                w.at[:, pl.ds(i * _WEIGHT_COLUMNS, _WEIGHT_COLUMNS)],
                weights.at[i % 2],
                wsems.at[i % 2],
            ).start()

    with jax.named_scope("fused_x_issue"):
        pltpu.make_async_copy(x, xbuf, xsem).start()
    weight_issue(0)
    with jax.named_scope("fused_scale_issue"):
        pltpu.make_async_copy(wscale_hbm, wscale, ssem).start()
    with jax.named_scope("fused_cache_issue"):

        @pl.loop(0, 288, unroll=False)
        def gather(row):
            pltpu.make_async_copy(
                cache.at[pos[row], :], gathered.at[row, :], gsem
            ).start()

    with jax.named_scope("fused_x_wait"):
        pltpu.make_async_copy(x, xbuf, xsem).wait()
    with jax.named_scope("fused_quant_compute"):
        quant_kernel(xbuf, q, xscale)
    with jax.named_scope("fused_cache_wait"):
        pltpu.make_async_copy(gathered, gathered, gsem).wait()
    weight_issue(1)
    with jax.named_scope("fused_coeff_prepare"):

        @pl.loop(0, 6, unroll=False)
        def coefficients(t):
            cos, sin = _cos_sin_lanes(
                gathered[pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :],
                rotary_dim=64,
                inverse=False,
            )
            sin = sin * jnp.where(jnp.arange(128) % 2 == 0, -1.0, 1.0)
            coeff[0, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :] = jnp.broadcast_to(
                cos[:, None, :], (_OUTPUT_ROWS, 16, 128)
            )
            coeff[1, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :] = jnp.broadcast_to(
                sin[:, None, :], (_OUTPUT_ROWS, 16, 128)
            )

    with jax.named_scope("fused_scale_wait"):
        pltpu.make_async_copy(wscale_hbm, wscale, ssem).wait()

    @pl.loop(0, 4, unroll=False)
    def window(i):
        slot = i % 2
        with jax.named_scope("fused_weight_wait"):
            pltpu.make_async_copy(
                w.at[:, pl.ds(i * _WEIGHT_COLUMNS, _WEIGHT_COLUMNS)],
                weights.at[slot],
                wsems.at[slot],
            ).wait()
        with jax.named_scope("fused_output_recycle"):

            @pl.when(i >= 2)
            def recycle():
                pltpu.make_async_copy(
                    obuf.at[slot], out.at[:, pl.ds((i - 2) * 16, 16), :], osems.at[slot]
                ).wait()

        with jax.named_scope("fused_mxu_scale_bf16"):
            z = jax.lax.dot_general(
                q[...],
                weights[slot, ...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            z = z * xscale[...]
            z = z * wscale[pl.ds(i * _WEIGHT_COLUMNS, _WEIGHT_COLUMNS)][None, :]
            matrix[...] = z.astype(jnp.bfloat16)

        # Current window no longer reads its weight buffer after the dot;
        # start the next generation before the VPU layout/RoPE work.
        @pl.when(i + 2 < 4)
        def refill():
            weight_issue(i + 2)

        with jax.named_scope("fused_layout_rope"):

            @pl.loop(0, 6, unroll=False)
            def rows(t):
                x = matrix[pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :].reshape(
                    _OUTPUT_ROWS, 16, 512
                )
                tail = x[:, :, 384:].astype(jnp.float32)
                rot = _lane_gather(tail, jnp.arange(128) ^ 1)
                y = (
                    tail * coeff[0, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :]
                    + rot * coeff[1, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :]
                ).astype(jnp.bfloat16)
                obuf[slot, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :] = (
                    jnp.concatenate((x[:, :, :384], y), axis=2)
                )
                with jax.named_scope("fused_output_tile_issue"):
                    pltpu.make_async_copy(
                        obuf.at[slot, pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), :, :],
                        out.at[
                            pl.ds(t * _OUTPUT_ROWS, _OUTPUT_ROWS), pl.ds(i * 16, 16), :
                        ],
                        osems.at[slot],
                    ).start()

    with jax.named_scope("fused_final_drain"):
        for i in (2, 3):
            pltpu.make_async_copy(
                obuf.at[i % 2], out.at[:, pl.ds(i * 16, 16), :], osems.at[i % 2]
            ).wait()


def chain(
    x: jax.Array, w: jax.Array, scale: jax.Array, pos: jax.Array, cache: jax.Array
) -> jax.Array:
    if not is_supported(x, w, scale, pos, cache):
        raise ValueError("FP8 projection/RoPE requires the qualified v7 decode inputs")
    x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
    w = pltpu.with_memory_space_constraint(w, pltpu.HBM)
    scale = pltpu.with_memory_space_constraint(scale, pltpu.HBM)
    cache = pltpu.with_memory_space_constraint(cache, pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((288, 64, 512), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, 1280, _WEIGHT_COLUMNS), w.dtype),
                pltpu.VMEM((32768,), jnp.float32),
                pltpu.VMEM((288, 64), jnp.float32),
                pltpu.VMEM((288, _WEIGHT_COLUMNS), jnp.bfloat16),
                pltpu.VMEM((2, 288, 16, 512), jnp.bfloat16),
                pltpu.VMEM((2, 288, 16, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((288, 1280), w.dtype),
                pltpu.VMEM((288, 1), jnp.float32),
                pltpu.VMEM((288, 1280), x.dtype),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_VMEM_BUDGET, disable_bounds_checks=True
        ),
        name="fused_quant_projection_rope_output48_all",
    )(pos, x, w, scale, cache)

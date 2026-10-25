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

"""Per-channel FP8 matmul with fused row quantization and buffered DMA.

Preserves util.xla_quantized_matmul's full-row FP32 quantization and scale
order. The measured envelope uses K-major weights without another weight copy.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    BUFFERED_MATMUL_TUNING,
    get_tpu_version,
)


def is_supported(x: jax.Array, w: jax.Array, scale: jax.Array) -> bool:
    """Whether concrete lowering inputs match the measured specialization."""
    return (
        x.ndim == 2
        and x.shape[0] in (240, 288)
        and x.shape[1] == 8192
        and w.shape == (8192, 5120)
        and scale.shape == (5120,)
        and x.dtype == jnp.bfloat16
        and w.dtype == jnp.float8_e4m3fn
        and scale.dtype == jnp.float32
        and get_tpu_version() == 7
    )


def _quantize_rows(x, q, scale, *, rows=32):
    def block(start, size):
        a = x[pl.ds(start, size), :].astype(jnp.float32)
        maximum = jnp.max(jnp.abs(a), axis=1, keepdims=True)
        s = maximum * jnp.float32(1.0 / 448.0)
        s = jnp.where(s == 0, 1.0, s)
        q[pl.ds(start, size), :] = jnp.clip(a / s, -448.0, 448.0).astype(q.dtype)
        scale[pl.ds(start, size), :] = s

    @pl.loop(0, x.shape[0] // rows, unroll=False)
    def body(i):
        block(i * rows, rows)

    if x.shape[0] % rows:
        block(x.shape[0] // rows * rows, x.shape[0] % rows)


def _kernel(
    x, w, s, out, xb, q, xs, wb, sb, ob, xsem, wsem, ssem, osem, *, width, quant_rows
):
    windows = w.shape[1] // width

    def weight(i):
        return pltpu.make_async_copy(
            w.at[:, pl.ds(i * width, width)], wb.at[i % 2], wsem.at[i % 2]
        )

    def output(i):
        return pltpu.make_async_copy(
            ob.at[i % 2], out.at[:, pl.ds(i * width, width)], osem.at[i % 2]
        )

    with jax.named_scope("ob_input_issue"):
        pltpu.make_async_copy(x, xb, xsem).start()
        weight(0).start()
        pltpu.make_async_copy(s, sb, ssem).start()
    with jax.named_scope("ob_input_wait"):
        pltpu.make_async_copy(x, xb, xsem).wait()
    with jax.named_scope("ob_quant"):
        _quantize_rows(xb, q, xs, rows=quant_rows)
    weight(1).start()
    pltpu.make_async_copy(s, sb, ssem).wait()

    @pl.loop(0, windows, unroll=False)
    def body(i):
        with jax.named_scope("ob_weight_wait"):
            weight(i).wait()
        with jax.named_scope("ob_output_recycle"):

            @pl.when(i >= 2)
            def recycle():
                output(i - 2).wait()

        with jax.named_scope("ob_gemm_scale"):
            z = jax.lax.dot_general(
                q[...],
                wb[i % 2, ...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            z = (z * xs[...]) * sb[pl.ds(i * width, width)][None, :]
            ob[i % 2, ...] = z.astype(out.dtype)

        @pl.when(i + 2 < windows)
        def refill():
            weight(i + 2).start()

        with jax.named_scope("ob_output_issue"):
            output(i).start()

    # Recycle waits cover earlier generations; drain the last two stores
    # before returning their backing scratch to the caller.
    with jax.named_scope("ob_drain"):
        output(windows - 2).wait()
        output(windows - 1).wait()


def buffered_matmul(x: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
    if not is_supported(x, w, scale):
        raise ValueError("Buffered FP8 matmul requires qualified v7 decode inputs")
    width, quant_rows, vmem_budget = BUFFERED_MATMUL_TUNING[
        (get_tpu_version(), x.shape[0], x.shape[1], w.shape[1])
    ]
    t, k = x.shape
    n = w.shape[1]
    x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
    w = pltpu.with_memory_space_constraint(w, pltpu.HBM)
    scale = pltpu.with_memory_space_constraint(scale, pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_kernel, width=width, quant_rows=quant_rows),
        out_shape=jax.ShapeDtypeStruct((t, n), x.dtype),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 3,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((t, k), x.dtype),
            pltpu.VMEM((t, k), w.dtype),
            pltpu.VMEM((t, 1), jnp.float32),
            pltpu.VMEM((2, k, width), w.dtype),
            pltpu.VMEM((n,), jnp.float32),
            pltpu.VMEM((2, t, width), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_budget),
        name="buffered_per_channel_matmul",
    )(x, w, scale)

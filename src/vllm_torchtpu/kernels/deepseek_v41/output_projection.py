# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeepSeek-V4.1 attention output projection: wo_a, then wo_b.

Computes ``buffered_matmul(wo_a_projection(x, wa, sa, None), wb, sb)`` in one
call. The wo_a result stays in VMEM instead of a write to HBM and a read back,
and the wo_b weight stream starts while the last wo_a groups compute.
Arithmetic and rounding match the two buffered kernels: wo_a quantizes each
group row of 4096 to FP8 with a BF16 scale; wo_b quantizes each full row of
8192 with an FP32 scale, whose amax the wo_a epilogues collect.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

HEADS_PER_GROUP = 8
# wo_a groups (x and weight slice) in flight.
GROUP_DEPTH = 3
# wo_b output columns per weight window.
WINDOW = 1024
# wo_b weight windows in flight. Window i + WINDOW_DEPTH refills after window
# i's matmul, so each refill has WINDOW_DEPTH - 1 windows of compute to land.
WINDOW_DEPTH = 3
QUANT_ROWS = 32
# wo_a loop iterations at which the first WINDOW_DEPTH wo_b windows start;
# the last iteration index means just before the final group's matmul. v7
# T288 XPlane, 100 interleaved calls: (5, 6, 7) 39.09 us, (4, 5, 6) 39.90 us,
# (3, 4, 5) 42.52 us; earlier starts take bandwidth from the wo_a groups.
B_ISSUE = (5, 6, 7)
# Weight streams yield to activation and scale copies.
WEIGHT_PRIORITY = 1
VMEM_LIMIT_BYTES = 60 * 1024**2


def is_supported(
    x: jax.Array, wa: jax.Array, sa: jax.Array, wb: jax.Array, sb: jax.Array
) -> bool:
    """Whether inputs match the measured v7 decode envelope."""
    return (
        x.ndim == 3
        and x.shape[0] in (240, 288)
        and x.shape[1:] == (64, 512)
        and wa.shape == (4096, 8192)
        and sa.shape == (8192,)
        and wb.shape == (8192, 5120)
        and sb.shape == (5120,)
        and x.dtype == jnp.bfloat16
        and wa.dtype == jnp.float8_e4m3fn
        and wb.dtype == jnp.float8_e4m3fn
        and sa.dtype == jnp.float32
        and sb.dtype == jnp.float32
        and get_tpu_version() == 7
    )


def _kernel(
    # HBM inputs and output
    x,  # [T, 64, 512] bf16
    wa,  # [4096, 8192] fp8
    sa,  # [1, 8192] f32
    wb,  # [8192, 5120] fp8
    sb,  # [5120] f32
    out,  # [T, 5120] bf16
    # VMEM scratch
    xb,  # [GROUP_DEPTH, T, 8, 512] bf16
    wab,  # [GROUP_DEPTH, 4096, 1024] fp8
    qa,  # [2, T, 4096] fp8, quantized wo_a input of groups g and g + 1
    ia,  # [2, T, 1] f32, their BF16 inverse scales
    y,  # [T, 8192] bf16, wo_a output
    ya,  # [T, 1] f32, running row amax of y
    qb,  # [T, 8192] fp8, quantized wo_b input
    xs,  # [T, 1] f32, its row scales
    wbb,  # [WINDOW_DEPTH, 8192, WINDOW] fp8
    sav,  # [1, 8192] f32
    sbv,  # [5120] f32
    ob,  # [2, T, WINDOW] bf16
    xsem,
    wsem,
    bsem,
    osem,
    ssem,
    *,
    b_issue: tuple[int, ...],
    weight_priority: int,
):
    tokens = x.shape[0]
    groups = x.shape[1] // HEADS_PER_GROUP
    width = wa.shape[1] // groups
    windows = wb.shape[1] // WINDOW

    def x_copy(g):
        return pltpu.make_async_copy(
            x.at[:, pl.ds(g * HEADS_PER_GROUP, HEADS_PER_GROUP), :],
            xb.at[g % GROUP_DEPTH],
            xsem.at[g % GROUP_DEPTH],
        )

    def wa_copy(g):
        return pltpu.make_async_copy(
            wa.at[:, pl.ds(g * width, width)],
            wab.at[g % GROUP_DEPTH],
            wsem.at[g % GROUP_DEPTH],
        )

    def wb_copy(i):
        return pltpu.make_async_copy(
            wb.at[:, pl.ds(i * WINDOW, WINDOW)],
            wbb.at[i % WINDOW_DEPTH],
            bsem.at[i % WINDOW_DEPTH],
        )

    def out_copy(i):
        return pltpu.make_async_copy(
            ob.at[i % 2], out.at[:, pl.ds(i * WINDOW, WINDOW)], osem.at[i % 2]
        )

    sa_copy = pltpu.make_async_copy(sa, sav, ssem.at[0])
    sb_copy = pltpu.make_async_copy(sb, sbv, ssem.at[1])

    def quantize_a(g):
        # Same operations as wo_a_projection's quantized _kernel body.
        v = xb[g % GROUP_DEPTH].reshape(tokens, -1)
        amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
        inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
            jnp.bfloat16
        )
        qa[g % 2] = (v * inv).astype(qa.dtype)
        ia[g % 2] = inv.astype(jnp.float32)

    def project_a(g):
        partial = jax.lax.dot_general(
            qa[g % 2],
            wab[g % GROUP_DEPTH],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        start = g * width if isinstance(g, int) else pl.multiple_of(g * width, width)
        cols = pl.ds(start, width)
        v = ((partial * (1.0 / ia[g % 2])) * sav[:, cols]).astype(y.dtype)
        y[:, cols] = v
        # wo_b's row amax, exact: max is order independent.
        ya[...] = jnp.maximum(
            ya[...], jnp.max(jnp.abs(v.astype(jnp.float32)), axis=1, keepdims=True)
        )

    def issue_b(g):
        for i, at in enumerate(b_issue):
            if isinstance(g, int):
                if g == at:
                    wb_copy(i).start(priority=weight_priority)
            else:

                @pl.when(g == at)
                def _(i=i):
                    wb_copy(i).start(priority=weight_priority)

    with jax.named_scope("op_start"):
        sa_copy.start()
        for g in range(GROUP_DEPTH):
            x_copy(g).start()
        sb_copy.start()
        for g in range(GROUP_DEPTH):
            wa_copy(g).start(priority=weight_priority)
        ya[...] = jnp.zeros(ya.shape, ya.dtype)
        x_copy(0).wait()
        quantize_a(0)
        sa_copy.wait()

    # Iteration g quantizes group g + 1 and multiplies group g; the two are
    # independent, so their vector and MXU work can interleave.
    @pl.loop(0, groups - 1, unroll=False)
    def _(g):
        with jax.named_scope("op_a_group"):
            x_copy(g + 1).wait()
            quantize_a(g + 1)
            wa_copy(g).wait()
            project_a(g)

        @pl.when(g + GROUP_DEPTH < groups)
        def _():
            x_copy(g + GROUP_DEPTH).start()
            wa_copy(g + GROUP_DEPTH).start(priority=weight_priority)

        issue_b(g)

    with jax.named_scope("op_a_last"):
        issue_b(groups - 1)
        wa_copy(groups - 1).wait()
        project_a(groups - 1)

    # Same operations and order as buffered._quantize_rows.
    with jax.named_scope("op_b_quant"):
        s = ya[...] * jnp.float32(1.0 / 448.0)
        xs[...] = jnp.where(s == 0, 1.0, s)

        def quantize_rows(start, size):
            rows = pl.ds(start, size)
            scale = xs[rows]
            for c in range(0, y.shape[1], width):
                cols = pl.ds(c, width)
                a = y[rows, cols].astype(jnp.float32)
                qb[rows, cols] = jnp.clip(a / scale, -448.0, 448.0).astype(qb.dtype)

        @pl.loop(0, tokens // QUANT_ROWS, unroll=False)
        def _(i):
            quantize_rows(pl.multiple_of(i * QUANT_ROWS, QUANT_ROWS), QUANT_ROWS)

        if tokens % QUANT_ROWS:
            quantize_rows(tokens // QUANT_ROWS * QUANT_ROWS, tokens % QUANT_ROWS)
        sb_copy.wait()

    # Same operations as buffered._kernel's window loop.
    @pl.loop(0, windows, unroll=False)
    def _(i):
        with jax.named_scope("op_b_window"):
            wb_copy(i).wait()

            @pl.when(i >= 2)
            def _():
                out_copy(i - 2).wait()

            z = jax.lax.dot_general(
                qb[...],
                wbb[i % WINDOW_DEPTH, ...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            z = (z * xs[...]) * sbv[pl.ds(i * WINDOW, WINDOW)][None, :]
            ob[i % 2, ...] = z.astype(out.dtype)

        @pl.when(i + WINDOW_DEPTH < windows)
        def _():
            wb_copy(i + WINDOW_DEPTH).start(priority=weight_priority)

        out_copy(i).start()

    with jax.named_scope("op_drain"):
        out_copy(windows - 2).wait()
        out_copy(windows - 1).wait()


@functools.partial(jax.jit, static_argnames=("b_issue", "weight_priority"))
def fused_output_projection(
    x: jax.Array,  # [T, 64, 512] bf16
    wa: jax.Array,  # [4096, 8192] fp8
    sa: jax.Array,  # [8192] f32
    wb: jax.Array,  # [8192, 5120] fp8
    sb: jax.Array,  # [5120] f32
    *,
    b_issue: tuple[int, ...] = B_ISSUE,
    weight_priority: int = WEIGHT_PRIORITY,
) -> jax.Array:
    """``wo_b(wo_a(x))`` for the qualified decode shapes, see ``is_supported``."""
    if not is_supported(x, wa, sa, wb, sb):
        raise ValueError("Fused output projection requires qualified v7 inputs")
    tokens, heads, head_dim = x.shape
    reduction, features = wa.shape
    groups = heads // HEADS_PER_GROUP
    width = features // groups
    k, n = wb.shape
    assert len(b_issue) == WINDOW_DEPTH and all(0 <= at < groups for at in b_issue)
    hbm = pltpu.HBM
    # Ordinary JAX primitives cannot consume Pallas memory-space annotations.
    # Reshape before constraining the array at the custom-call boundary.
    sa = sa.reshape(1, features)
    x, wa, sa, wb, sb = (
        pltpu.with_memory_space_constraint(a, hbm) for a in (x, wa, sa, wb, sb)
    )
    f32, fp8 = jnp.float32, wa.dtype
    return pl.pallas_call(
        functools.partial(_kernel, b_issue=b_issue, weight_priority=weight_priority),
        out_shape=jax.ShapeDtypeStruct((tokens, n), x.dtype),
        in_specs=(pl.BlockSpec(memory_space=hbm),) * 5,
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=(
            pltpu.VMEM((GROUP_DEPTH, tokens, HEADS_PER_GROUP, head_dim), x.dtype),
            pltpu.VMEM((GROUP_DEPTH, reduction, width), fp8),
            pltpu.VMEM((2, tokens, reduction), fp8),
            pltpu.VMEM((2, tokens, 1), f32),
            pltpu.VMEM((tokens, features), x.dtype),
            pltpu.VMEM((tokens, 1), f32),
            pltpu.VMEM((tokens, k), fp8),
            pltpu.VMEM((tokens, 1), f32),
            pltpu.VMEM((WINDOW_DEPTH, k, WINDOW), fp8),
            pltpu.VMEM((1, features), f32),
            pltpu.VMEM((n,), f32),
            pltpu.VMEM((2, tokens, WINDOW), x.dtype),
            pltpu.SemaphoreType.DMA((GROUP_DEPTH,)),
            pltpu.SemaphoreType.DMA((GROUP_DEPTH,)),
            pltpu.SemaphoreType.DMA((WINDOW_DEPTH,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
        name="fused_output_projection",
    )(x, wa, sa, wb, sb)

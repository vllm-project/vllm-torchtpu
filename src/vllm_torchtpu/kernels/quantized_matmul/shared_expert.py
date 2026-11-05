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

"""Per-channel FP8 SwiGLU MLP (gate/up, clamped SiLU-and-mul, down) as one kernel.

Computes what the XLA path computes op for op: ``xla_quantized_matmul`` for
both projections (full-row FP32 activation quantization, f32 accumulation,
row scale then column scale, bf16 output) and ``SiluAndMulWithClamp`` as torch
lowers it (bf16 clamps, f32 ``1 / (1 + exp(-x))`` rounded to bf16, bf16
products). Both weights stream through VMEM in column windows; the
intermediate never leaves VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    SHARED_EXPERT_TUNING,
    get_tpu_version,
)

_FP8_MAX = 448.0


def is_supported(
    x: jax.Array,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
) -> bool:
    """Whether concrete lowering inputs match a measured specialization."""
    if x.ndim != 2 or w13.ndim != 2 or w2.ndim != 2:
        return False
    t, k = x.shape
    inter = w2.shape[0]
    return (
        w13.shape == (k, 2 * inter)
        and w2.shape[1] == k
        and s13.shape == (2 * inter,)
        and s2.shape == (k,)
        and x.dtype == jnp.bfloat16
        and w13.dtype == jnp.float8_e4m3fn
        and w2.dtype == jnp.float8_e4m3fn
        and s13.dtype == jnp.float32
        and s2.dtype == jnp.float32
        and (get_tpu_version(), t, k, inter) in SHARED_EXPERT_TUNING
    )


def _row_scale(maximum):
    s = maximum * jnp.float32(1.0 / _FP8_MAX)
    return jnp.where(s == 0, 1.0, s)


def _quantize_rows(x, q, scale, *, rows):
    def block(start, size):
        a = x[pl.ds(start, size), :].astype(jnp.float32)
        s = _row_scale(jnp.max(jnp.abs(a), axis=1, keepdims=True))
        q[pl.ds(start, size), :] = jnp.clip(a / s, -_FP8_MAX, _FP8_MAX).astype(q.dtype)
        scale[pl.ds(start, size), :] = s

    @pl.loop(0, x.shape[0] // rows, unroll=False)
    def body(i):
        block(i * rows, rows)

    if x.shape[0] % rows:
        block(x.shape[0] // rows * rows, x.shape[0] % rows)


def silu_and_mul_with_clamp(gate, up, limit):
    """vLLM's SiluAndMulWithClamp (alpha 1, beta 0) in torch's lowered op order."""
    gate = jnp.minimum(gate, jnp.asarray(limit, gate.dtype))
    up = jnp.clip(up, jnp.asarray(-limit, up.dtype), jnp.asarray(limit, up.dtype))
    sig = 1.0 / (jnp.exp(-gate.astype(jnp.float32)) + 1.0)
    return (gate * sig.astype(gate.dtype)) * up


def _kernel(
    x,
    w13,
    s13,
    w2,
    s2,
    out,
    xb,
    q,
    xs,
    h,
    hmax,
    hq,
    hs,
    gb,
    ub,
    s13b,
    w2b,
    s2b,
    ob,
    zb,
    zd,
    xsem,
    ssem,
    gsem,
    usem,
    w2sem,
    osem,
    *,
    width,
    down_width,
    depth,
    split,
    quant_rows,
    limit,
):
    inter = w2.shape[0]
    n_up = inter // width
    n_down = w2.shape[1] // down_width
    n_stream = n_up + n_down

    # One weight stream, gate/up windows then down windows, `depth` deep; each
    # window moves as `split` row-chunk DMAs so more descriptors are in flight.
    def chunks(src, dst, sem):
        rows = src.shape[0] // split
        return tuple(
            pltpu.make_async_copy(
                src.at[pl.ds(c * rows, rows)], dst.at[pl.ds(c * rows, rows)], sem
            )
            for c in range(split)
        )

    def stream(i):
        if i < n_up:
            slot = i % depth
            return chunks(
                w13.at[:, pl.ds(i * width, width)], gb.at[slot], gsem.at[slot]
            ) + chunks(
                w13.at[:, pl.ds(inter + i * width, width)], ub.at[slot], usem.at[slot]
            )
        j = i - n_up
        slot = j % depth
        return chunks(
            w2.at[:, pl.ds(j * down_width, down_width)], w2b.at[slot], w2sem.at[slot]
        )

    def start(i):
        if i < n_stream:
            for c in stream(i):
                c.start()

    def wait(i):
        for c in stream(i):
            c.wait()

    def output(j):
        return pltpu.make_async_copy(
            ob.at[j % 2], out.at[:, pl.ds(j * down_width, down_width)], osem.at[j % 2]
        )

    scales = (
        pltpu.make_async_copy(s13, s13b, ssem.at[0]),
        pltpu.make_async_copy(s2, s2b, ssem.at[1]),
    )
    with jax.named_scope("se_input_issue"):
        pltpu.make_async_copy(x, xb, xsem).start()
        for i in range(depth):
            start(i)
        for c in scales:
            c.start()
    with jax.named_scope("se_input_wait"):
        pltpu.make_async_copy(x, xb, xsem).wait()
    with jax.named_scope("se_quant_x"):
        _quantize_rows(xb, q, xs, rows=quant_rows)
    for c in scales:
        c.wait()

    def dot(a, b):
        return jax.lax.dot_general(
            a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

    # Software-pipelined: window i + 1's matmuls are issued before window i's
    # epilogue, in one basic block, so the scheduler overlaps MXU and VPU work.
    # The raw f32 products park in VMEM between the two (exact).
    def gate_up_dots(i):
        wait(i)
        zb[i % 2, 0] = dot(q[...], gb[i % depth])
        zb[i % 2, 1] = dot(q[...], ub[i % depth])
        start(i + depth)

    def gate_up_epilogue(i):
        zs = [
            ((zb[i % 2, c] * xs[...]) * s13b[pl.ds(col, width)][None, :]).astype(
                x.dtype
            )
            for c, col in enumerate((i * width, inter + i * width))
        ]
        a = silu_and_mul_with_clamp(*zs, limit)
        h[:, pl.ds(i * width, width)] = a
        m = jnp.max(jnp.abs(a.astype(jnp.float32)), axis=1, keepdims=True)
        hmax[...] = m if i == 0 else jnp.maximum(hmax[...], m)

    def down_dot(j):
        wait(n_up + j)
        zd[j % 2] = dot(hq[...], w2b[j % depth])
        start(n_up + j + depth)

    def down_epilogue(j):
        if j >= 2:
            output(j - 2).wait()
        z = (zd[j % 2] * hs[...]) * s2b[pl.ds(j * down_width, down_width)][None, :]
        ob[j % 2] = z.astype(out.dtype)
        output(j).start()

    with jax.named_scope("se_gate_up"):
        gate_up_dots(0)
        for i in range(n_up):
            if i + 1 < n_up:
                gate_up_dots(i + 1)
            gate_up_epilogue(i)

    with jax.named_scope("se_quant_h"):
        s = _row_scale(hmax[...])
        hs[...] = s
        hq[...] = jnp.clip(h[...].astype(jnp.float32) / s, -_FP8_MAX, _FP8_MAX).astype(
            hq.dtype
        )

    with jax.named_scope("se_down"):
        down_dot(0)
        for j in range(n_down):
            if j + 1 < n_down:
                down_dot(j + 1)
            down_epilogue(j)

    with jax.named_scope("se_drain"):
        for j in range(max(n_down - 2, 0), n_down):
            output(j).wait()


def fused_shared_expert(
    x: jax.Array,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    limit: float,
    tuning: tuple[int, int, int, int, int, int] | None = None,
) -> jax.Array:
    """``down(silu_and_mul_with_clamp(gate_up(x)))`` for per-channel FP8 weights.

    `w13` is the merged gate/up weight `[k, 2 * inter]` (gate columns first),
    `w2` the down weight `[inter, k]`, both (k, n) with per-output-channel f32
    scales. `tuning` = (width, down_width, depth, split, quant_rows,
    vmem_bytes) overrides the table (for measurement only).
    """
    if not is_supported(x, w13, s13, w2, s2):
        raise ValueError("Fused shared expert requires qualified v7 decode inputs")
    t, k = x.shape
    inter = w2.shape[0]
    if tuning is None:
        tuning = SHARED_EXPERT_TUNING[(get_tpu_version(), t, k, inter)]
    width, down_width, depth, split, quant_rows, vmem_budget = tuning
    if inter % width or k % down_width or inter % split or k % split:
        raise ValueError(f"{tuning=} does not tile {inter=} / {k=}")
    x, w13, s13, w2, s2 = (
        pltpu.with_memory_space_constraint(a, pltpu.HBM) for a in (x, w13, s13, w2, s2)
    )
    return pl.pallas_call(
        functools.partial(
            _kernel,
            width=width,
            down_width=down_width,
            depth=depth,
            split=split,
            quant_rows=quant_rows,
            limit=limit,
        ),
        out_shape=jax.ShapeDtypeStruct((t, k), x.dtype),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 5,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((t, k), x.dtype),  # xb
            pltpu.VMEM((t, k), w13.dtype),  # q
            pltpu.VMEM((t, 1), jnp.float32),  # xs
            pltpu.VMEM((t, inter), x.dtype),  # h
            pltpu.VMEM((t, 1), jnp.float32),  # hmax
            pltpu.VMEM((t, inter), w2.dtype),  # hq
            pltpu.VMEM((t, 1), jnp.float32),  # hs
            pltpu.VMEM((depth, k, width), w13.dtype),  # gb
            pltpu.VMEM((depth, k, width), w13.dtype),  # ub
            pltpu.VMEM((2 * inter,), jnp.float32),  # s13b
            pltpu.VMEM((depth, inter, down_width), w2.dtype),  # w2b
            pltpu.VMEM((k,), jnp.float32),  # s2b
            pltpu.VMEM((2, t, down_width), x.dtype),  # ob
            pltpu.VMEM((2, 2, t, width), jnp.float32),  # zb
            pltpu.VMEM((2, t, down_width), jnp.float32),  # zd
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((depth,)),
            pltpu.SemaphoreType.DMA((depth,)),
            pltpu.SemaphoreType.DMA((depth,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_budget),
        name="fused_shared_expert_fp8",
    )(x, w13, s13, w2, s2)

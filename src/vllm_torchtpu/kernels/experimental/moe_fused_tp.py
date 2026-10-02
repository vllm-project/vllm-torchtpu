# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE for tensor-parallel experts at decode-sized token counts.

Under MoE tensor parallelism every rank holds all experts with the
intermediate dimension sharded, so each expert's local slice is small (for
Qwen3.5-397B at TP8: i = 128, 1.5 MiB of FP8 per expert) and rows per expert
at decode are ~1-3. The GMM path pays a fixed cost per active expert in each
of its two kernels plus a permute/sort/combine around them.

This kernel visits each active expert once. Its w13/w2 slices stream from HBM
through a ring of `num_buffers` VMEM slots while the previous expert computes
on ALL tokens in their original order; the route weight zeroes rows the
expert does not own. The first matmul of expert j+1 is issued in step j, ahead
of expert j's elementwise chain and second matmul, so the MXU stays busy while
the VPU quantizes. Nothing is permuted and no intermediate goes to HBM.

Numerics: hidden states are quantized to FP8 once per call with one scale per
row; the activated intermediate is quantized per row per expert; both
matmuls are FP8 x FP8 with FP32 accumulation into an FP32 VMEM accumulator.
Router top-k, the shared expert and the TP all-reduce stay with the caller.

Derived from moe_dense_expert.py, which targets i % 512 == 0 (EP layouts).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_F8 = jnp.float8_e4m3fn
_F8_MAX = 448.0
# Output columns per second-matmul chunk; bounds the live f32 result.
_N_CHUNK = 512
# Weight ring depth. One slot holds the expert whose first matmul is in
# flight, so num_buffers - 2 transfers run ahead of compute.
_NUM_BUFFERS = 6


def can_use_fused_tp(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Arithmetic/layout support only; the token cap is applied by the caller."""
    if activation != "silu" or rhs_quant_dtype not in (None, _F8):
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    _, h = hidden.shape
    e, i, out = w2.shape
    if min(h, i, e) <= 0 or i % 128 or h % _N_CHUNK or out != h:
        return False
    if w1.shape != (e, h, 2 * i):
        return False
    if hidden.dtype != jnp.bfloat16 or w1.dtype != _F8 or w2.dtype != _F8:
        return False
    return (
        s1.shape == (e, 1, 1, 2 * i)
        and s2.shape == (e, 1, 1, h)
        and s1.dtype == jnp.float32
        and s2.dtype == jnp.float32
    )


def _kernel(
    active_ref,
    x_ref,
    coeff_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    out_ref,
    w1_ref,
    w2_ref,
    s1_ref,
    s2_ref,
    xq_ref,
    xs_ref,
    owner_ref,
    ids_ref,
    sem_ref,
    *,
    m,
    h,
    i,
    experts,
    nb,
):
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    x = x_ref[...].astype(jnp.float32)
    xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / _F8_MAX
    xq_ref[...] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(_F8)
    xs_ref[...] = jnp.broadcast_to(xsc, (m, 2 * i))
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = ids_ref[step], step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(s1_hbm.at[e, 0], s1_ref.at[slot], sem_ref.at[slot, 2]),
            pltpu.make_async_copy(s2_hbm.at[e, 0], s2_ref.at[slot], sem_ref.at[slot, 3]),
        ]

    def gate_up(slot):
        return jnp.matmul(xq_ref[...], w1_ref[slot], preferred_element_type=jnp.float32)

    def finish(step, slot, acc1):
        e = ids_ref[step]
        coeff = jnp.sum(jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True)
        acc1 = acc1 * xs_ref[...] * s1_ref[slot]
        # Mask, not multiply: a padded row with non-finite hidden would give
        # NaN * 0 = NaN. Rows this expert does not own contribute exactly 0.
        mid = jnp.where(coeff != 0, jax.nn.silu(acc1[:, :i]) * acc1[:, i:] * coeff, 0.0)
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / _F8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(_F8)
        for n in range(0, h, _N_CHUNK):
            acc2 = jnp.matmul(
                mq, w2_ref[slot, :, n : n + _N_CHUNK], preferred_element_type=jnp.float32
            )
            owner_ref[:, n : n + _N_CHUNK] += acc2 * s2_ref[slot, :, n : n + _N_CHUNK] * ms

    @pl.when(count > 0)
    def run_experts():
        for initial in range(nb):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < count)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            # Issued before this expert's elementwise chain so the MXU and VPU
            # overlap. On the last step it reads a stale slot; the result is
            # dropped.
            acc1_next = gate_up(nxt % nb)
            finish(step, step % nb, acc1)

            # This expert's slot is free once its second matmul has read it.
            @pl.when(step + nb < count)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(0, count, expert_body, gate_up(0))

    out_ref[...] = owner_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("num_buffers",))
def fused_moe_tp(
    hidden, w1, w2, s1, s2, topk_weights, topk_ids, num_buffers=_NUM_BUFFERS
):
    """Routed-expert contribution of one TP rank, BF16 [tokens, hidden].

    Same contract as fused_moe_func(use_ep=False): scales are [E, 1, 1, N],
    padded tokens carry zero routing weight, and the caller owns top-k, the
    shared expert and the all-reduce. Callers must check can_use_fused_tp.
    """
    tokens, h = hidden.shape
    experts, i, _ = w2.shape
    # The bucket ladder includes 4 and 8; pad to the BF16 sublane tile.
    pad = (-tokens) % 16
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
    m = tokens + pad
    # Dense route coefficients [tokens, experts]. Unroll the static top-k so
    # no [M, K, E] reduction is formed (see moe_dense_expert.py).
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(topk_ids.shape[1]):
        selected = topk_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    nb = num_buffers
    out = pl.pallas_call(
        functools.partial(_kernel, m=m, h=h, i=i, experts=experts, nb=nb),
        out_shape=jax.ShapeDtypeStruct((m, h), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * i), _F8),
                pltpu.VMEM((nb, i, h), _F8),
                pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((m, h), _F8),
                pltpu.VMEM((m, 2 * i), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name=f"fused_moe_tp-e_{experts}-m_{m}-h_{h}-i_{i}-buffers_{nb}",
    )(active, hidden, coeff, w1, w2, s1, s2)
    return out[:tokens] if pad else out

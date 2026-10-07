# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Experimental small-batch FP8 MoE without token permutation.

Each nonempty local expert is visited once. Its two weights are DMA'd once,
all tokens are evaluated in their original order, and weighted outputs are
accumulated in FP32 VMEM. Neither intermediate activations nor per-expert
outputs are written to HBM. The caller still owns top-k and EP collectives.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    silu_and_mul_with_clamp,
    swigluoai,
)

_MAX_EXPERT_BUFFERS = 3
_SUPPORTED_ACTIVATIONS = ("silu", "silu_and_mul_with_clamp", "swigluoai")


def _select_expert_buffers(
    m,
    h,
    i,
    experts,
    vmem_capacity_bytes,
    packing_factor=1,
    s1_blocks=1,
    s2_blocks=1,
):
    """Choose up to three slots, reserving 20% for padding and temporaries.

    Count explicit VMEM inputs, output and scratch, including quantization
    scales expanded over 128 lanes. As in RPA, this is a sizing heuristic;
    the compiler still decides resource feasibility, even for one slot.
    """
    if packing_factor == 1 and s1_blocks == 1 and s2_blocks == 1:
        expert_bytes = 3 * h * i + 4 * (2 * i + h)
    else:
        expert_bytes = (3 * h * i) // packing_factor + 32 * (
            s1_blocks * 2 * i + s2_blocks * h
        )
    # Hidden input/output, quantized hidden, and FP32 output accumulator.
    fixed_bytes = m * h * (2 + 2 + 1 + 4)
    # BF16 activation, FP8 activation, and the two BF16 quantization scales.
    fixed_bytes += m * i * (2 + 1) + (h // 512 + i // 512) * m * 128 * 2
    fixed_bytes += m * pl.cdiv(experts, 128) * 128 * 4
    budget = vmem_capacity_bytes * 4 // 5
    return max(1, min(_MAX_EXPERT_BUFFERS, (budget - fixed_bytes) // expert_bytes))


def _packing_factor_for(w1, w2, rhs_quant_dtype):
    if rhs_quant_dtype in (None, jnp.float8_e4m3fn):
        if w1.dtype == jnp.float8_e4m3fn and w2.dtype == jnp.float8_e4m3fn:
            return 1
        return None
    if rhs_quant_dtype == jnp.float4_e2m1fn:
        if w1.dtype == jnp.uint8 and w2.dtype == jnp.uint8:
            return 2
        return None
    return None


def can_use_dense_expert(
    hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype, skip_padded_tokens
):
    """Check arithmetic/layout support; the compiler owns resource limits.

    Zero routing coefficients already suppress padded tokens, so either
    skip_padded_tokens setting is supported. There is no prefill/decode
    inference here: the MoE entry applies its configurable token threshold.
    """
    if activation not in _SUPPORTED_ACTIVATIONS:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    packing_factor = _packing_factor_for(w1, w2, rhs_quant_dtype)
    if packing_factor is None:
        return False
    m, h = hidden.shape
    e, i_packed, out = w2.shape
    i = i_packed * packing_factor
    if min(m, h, i, e) <= 0:
        return False
    if (packing_factor == 1 and m % 16) or h % 512 or i % 512:
        return False
    if out != h or w1.shape != (e, h // packing_factor, 2 * i):
        return False
    if hidden.dtype != jnp.bfloat16:
        return False
    valid_s1 = s1.shape in ((e, 1, 1, 2 * i), (e, h // 512, 1, 2 * i))
    valid_s2 = s2.shape in ((e, 1, 1, h), (e, i // 512, 1, h))
    return valid_s1 and valid_s2 and s1.dtype == jnp.float32 and s2.dtype == jnp.float32


def _quantize(x, q_ref, scale_ref):
    """Same BF16 scale/inverse and 512-wide FP8 quantization as gmm_v2."""
    # Keep K blocks first so each scale read is a contiguous [tokens, 128]
    # plane; placing K between tokens and lanes causes expensive relayouts.
    for k in range(0, x.shape[1], 512):
        block = x[:, k : k + 512]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
        inv = jnp.where(scale == 0, 0, 1 / scale)
        q_ref[:, k : k + 512] = (block * inv).astype(jnp.float8_e4m3fn)
        scale_ref[k // 512, :, :] = jnp.broadcast_to(scale, (x.shape[0], 128))


def _apply_activation(gate, up, activation):
    if activation == "silu":
        return jax.nn.silu(gate) * up
    if activation == "silu_and_mul_with_clamp":
        return silu_and_mul_with_clamp(gate, up)
    if activation == "swigluoai":
        return swigluoai(gate, up)
    raise NotImplementedError(f"Unsupported activation: {activation}")


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
    a_ref,
    aq_ref,
    asc_ref,
    owner_ref,
    active_ids_ref,
    sem_ref,
    *,
    m,
    h,
    i,
    experts,
    num_buffers,
    packing_factor=1,
    s1_blocks=1,
    s2_blocks=1,
    activation="silu",
    rhs_quant_dtype=None,
):
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    _quantize(x_ref[...], xq_ref, xs_ref)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compute(e, w1_ref, w2_ref, s1_ref, s2_ref):
        coeff = jnp.sum(
            jnp.where(expert_columns == e, coeff_ref[...], 0), axis=1, keepdims=True
        )
        # Gate/up projections in interleaved 128-column pairs, matching
        # the original fused GMM's arithmetic and BF16 rounding points.
        for n in range(0, i, 128):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, h, 512):
                k_lo = k // packing_factor
                k_hi = (k + 512) // packing_factor
                rhs = jnp.concatenate(
                    (
                        w1_ref[k_lo:k_hi, n : n + 128],
                        w1_ref[k_lo:k_hi, i + n : i + n + 128],
                    ),
                    axis=1,
                )
                if packing_factor > 1:
                    rhs = pltpu.bitcast(rhs, rhs_quant_dtype).astype(jnp.float8_e4m3fn)
                part = jnp.matmul(
                    xq_ref[:, k : k + 512], rhs, preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
                xs = xs_ref[k // 512, :, :]
                part *= jnp.concatenate((xs, xs), axis=1)
                s1_slice = s1_ref if s1_blocks == 1 else s1_ref[k // 512]
                scale = jnp.concatenate(
                    (
                        s1_slice[:, n : n + 128],
                        s1_slice[:, i + n : i + n + 128],
                    ),
                    axis=1,
                )
                part *= scale.astype(jnp.bfloat16)
                acc += part
            a_ref[:, n : n + 128] = _apply_activation(
                acc[:, :128], acc[:, 128:], activation
            )
        _quantize(a_ref[...], aq_ref, asc_ref)
        for n in range(0, h, 256):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, i, 512):
                k_lo = k // packing_factor
                k_hi = (k + 512) // packing_factor
                rhs = w2_ref[k_lo:k_hi, n : n + 256]
                if packing_factor > 1:
                    rhs = pltpu.bitcast(rhs, rhs_quant_dtype).astype(jnp.float8_e4m3fn)
                part = jnp.matmul(
                    aq_ref[:, k : k + 512],
                    rhs,
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                asc = asc_ref[k // 512, :, :]
                part *= jnp.concatenate((asc, asc), axis=1)
                s2_slice = (
                    s2_ref[:, n : n + 256]
                    if s2_blocks == 1
                    else s2_ref[k // 512, :, n : n + 256]
                )
                part *= s2_slice.astype(jnp.bfloat16)
                acc += part
            # Mask before accumulation; zero routing must suppress even
            # non-finite unused expert outputs, rather than multiply NaN*0.
            weighted = jnp.where(coeff != 0, acc.astype(jnp.float32) * coeff, 0)
            owner_ref[:, n : n + 256] += weighted

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = active_ids_ref[step], step % num_buffers
        s1_src = s1_hbm.at[e, 0] if s1_blocks == 1 else s1_hbm.at[e]
        s2_src = s2_hbm.at[e, 0] if s2_blocks == 1 else s2_hbm.at[e]
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(s1_src, s1_ref.at[slot], sem_ref.at[slot, 2]),
            pltpu.make_async_copy(s2_src, s2_ref.at[slot], sem_ref.at[slot, 3]),
        ]

    @pl.when(count > 0)
    def run_experts():
        for initial in range(num_buffers - 1):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        def expert_body(step, _):
            if num_buffers == 1:
                for copy in transfers(step):
                    copy.start()
            for copy in transfers(step):
                copy.wait()

            if num_buffers > 1:

                @pl.when(step + num_buffers - 1 < count)
                def prefetch():
                    for copy in transfers(step + num_buffers - 1):
                        copy.start()

            slot = step % num_buffers
            compute(
                active_ids_ref[step],
                w1_ref.at[slot],
                w2_ref.at[slot],
                s1_ref.at[slot],
                s2_ref.at[slot],
            )
            return None

        # Prefetch up to num_buffers - 1 experts ahead. With one slot, each
        # expert finishes computing before the next DMA overwrites its inputs.
        # Compacting the local-expert active bitmap uses only scalar SMEM ops.
        lax.fori_loop(0, count, expert_body, None)

    out_ref[...] = owner_ref[...].astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("activation", "rhs_quant_dtype"))
def dense_expert_moe(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    local_ids,
    *,
    activation="silu",
    rhs_quant_dtype=None,
):
    """Local routed contribution, same [tokens, hidden]/BF16 output as GMM.

    IDs outside this rank must be -1. Scales use [E,blocks,1,N]. Callers must
    first check can_use_dense_expert. Shared experts and collectives stay
    outside this kernel, exactly as for fused_moe_func's standard path.
    """
    orig_m, h = hidden.shape
    pad_m = (-orig_m) % 16
    m = orig_m + pad_m
    packing_factor = _packing_factor_for(w1, w2, rhs_quant_dtype)
    assert packing_factor is not None
    experts, i_packed, _ = w2.shape
    i = i_packed * packing_factor
    s1_blocks = s1.shape[1]
    s2_blocks = s2.shape[1]
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    if packing_factor == 1 and s1_blocks == 1 and s2_blocks == 1:
        num_buffers = _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes)
    else:
        num_buffers = _select_expert_buffers(
            m,
            h,
            i,
            experts,
            vmem_capacity_bytes,
            packing_factor,
            s1_blocks,
            s2_blocks,
        )
    # Dense route coefficients are only [tokens, local experts], not expert
    # output tensors. Accumulation handles repeated IDs without reloading an
    # expert. Unroll the static top-k dimension to avoid the [M,K,E] reduction
    # whose layout copies can be offloaded to SparseCore and delay all-gather.
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    coeff = jnp.zeros((orig_m, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    if pad_m:
        hidden = jnp.pad(hidden, ((0, pad_m), (0, 0)))
    coeff = jnp.pad(coeff, ((0, pad_m), (0, (-experts) % 128)))
    s1_scratch_shape = (
        (num_buffers, 1, 2 * i)
        if s1_blocks == 1
        else (num_buffers, s1_blocks, 1, 2 * i)
    )
    s2_scratch_shape = (
        (num_buffers, 1, h) if s2_blocks == 1 else (num_buffers, s2_blocks, 1, h)
    )
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            num_buffers=num_buffers,
            packing_factor=packing_factor,
            s1_blocks=s1_blocks,
            s2_blocks=s2_blocks,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((num_buffers, h // packing_factor, 2 * i), w1.dtype),
                pltpu.VMEM((num_buffers, i // packing_factor, h), w2.dtype),
                pltpu.VMEM(s1_scratch_shape, jnp.float32),
                pltpu.VMEM(s2_scratch_shape, jnp.float32),
                pltpu.VMEM((m, h), jnp.float8_e4m3fn),
                pltpu.VMEM((h // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.float8_e4m3fn),
                pltpu.VMEM((i // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((num_buffers, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity_bytes),
        name=(f"dense_expert_moe-e_{experts}-m_{m}-h_{h}-i_{i}-buffers_{num_buffers}"),
    )(active, hidden, coeff, w1, w2, s1, s2)
    return out[:orig_m]

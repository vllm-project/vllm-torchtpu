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


def can_use_dense_expert(hidden, w1, w2, s1, s2, b1, b2, *, activation,
                         rhs_quant_dtype, skip_padded_tokens):
    """Conservative, static eligibility check; other calls keep standard GMM."""
    if activation != 'silu' or rhs_quant_dtype is not None or skip_padded_tokens:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    m, h = hidden.shape
    e, i, out = w2.shape
    if not (0 < m <= 256 and m % 16 == 0 and 0 < e <= 64
            and h in (512, 1024, 2048, 4096) and i in (512, 1024) and out == h
            and w1.shape == (e, h, 2 * i)):
        return False
    if (hidden.dtype != jnp.bfloat16 or w1.dtype != jnp.float8_e4m3fn
            or w2.dtype != jnp.float8_e4m3fn):
        return False
    if (s1.shape != (e, 1, 1, 2 * i) or s2.shape != (e, 1, 1, h)
            or s1.dtype != jnp.float32 or s2.dtype != jnp.float32):
        return False
    try:
        return pltpu.get_tpu_info().vmem_capacity_bytes >= 64 * 1024 * 1024
    except ValueError:
        return False


def _quantize(x, q_ref, scale_ref):
    """Same BF16 scale/inverse and 512-wide FP8 quantization as gmm_v2."""
    # Keep K blocks first so each scale read is a contiguous [tokens, 128]
    # plane; placing K between tokens and lanes causes expensive relayouts.
    for k in range(0, x.shape[1], 512):
        block = x[:, k:k + 512]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
        inv = jnp.where(scale == 0, 0, 1 / scale)
        q_ref[:, k:k + 512] = (block * inv).astype(jnp.float8_e4m3fn)
        scale_ref[k // 512, :, :] = jnp.broadcast_to(scale, (x.shape[0], 128))


def _kernel(active_ref, x_ref, coeff_ref, w1_hbm, w2_hbm, s1_hbm, s2_hbm,
            out_ref, w1_ref, w2_ref, s1_ref, s2_ref, xq_ref, xs_ref, a_ref,
            aq_ref, asc_ref, owner_ref, active_ids_ref, sem_ref, *, m, h, i,
            experts):
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    _quantize(x_ref[...], xq_ref, xs_ref)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compute(e, w1_ref, w2_ref, s1_ref, s2_ref):
        coeff = jnp.sum(jnp.where(expert_columns == e, coeff_ref[...], 0),
                        axis=1,
                        keepdims=True)
        # Gate/up projections in interleaved 128-column pairs, matching
        # the original fused GMM's arithmetic and BF16 rounding points.
        for n in range(0, i, 128):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            scale = jnp.concatenate(
                (s1_ref[:, n:n + 128], s1_ref[:, i + n:i + n + 128]), axis=1)
            for k in range(0, h, 512):
                rhs = jnp.concatenate(
                    (w1_ref[k:k + 512, n:n + 128], w1_ref[k:k + 512,
                                                          i + n:i + n + 128]),
                    axis=1)
                part = jnp.matmul(xq_ref[:, k:k + 512],
                                  rhs,
                                  preferred_element_type=jnp.float32).astype(
                                      jnp.bfloat16)
                xs = xs_ref[k // 512, :, :]
                part *= jnp.concatenate((xs, xs), axis=1)
                part *= scale.astype(jnp.bfloat16)
                acc += part
            a_ref[:, n:n + 128] = jax.nn.silu(acc[:, :128]) * acc[:, 128:]
        _quantize(a_ref[...], aq_ref, asc_ref)
        for n in range(0, h, 256):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, i, 512):
                part = jnp.matmul(aq_ref[:, k:k + 512],
                                  w2_ref[k:k + 512, n:n + 256],
                                  preferred_element_type=jnp.float32).astype(
                                      jnp.bfloat16)
                asc = asc_ref[k // 512, :, :]
                part *= jnp.concatenate((asc, asc), axis=1)
                part *= s2_ref[:, n:n + 256].astype(jnp.bfloat16)
                acc += part
            # Mask before accumulation; zero routing must suppress even
            # non-finite unused expert outputs, rather than multiply NaN*0.
            weighted = jnp.where(coeff != 0,
                                 acc.astype(jnp.float32) * coeff, 0)
            owner_ref[:, n:n + 256] += weighted

    def compact(e, count):

        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = active_ids_ref[step], step % 2
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot],
                                  sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot],
                                  sem_ref.at[slot, 1]),
            pltpu.make_async_copy(s1_hbm.at[e, 0], s1_ref.at[slot],
                                  sem_ref.at[slot, 2]),
            pltpu.make_async_copy(s2_hbm.at[e, 0], s2_ref.at[slot],
                                  sem_ref.at[slot, 3]),
        ]

    @pl.when(count > 0)
    def run_experts():
        for copy in transfers(jnp.int32(0)):
            copy.start()

        def expert_body(step, _):
            for copy in transfers(step):
                copy.wait()

            @pl.when(step + 1 < count)
            def prefetch():
                for copy in transfers(step + 1):
                    copy.start()

            slot = step % 2
            compute(active_ids_ref[step], w1_ref.at[slot], w2_ref.at[slot],
                    s1_ref.at[slot], s2_ref.at[slot])
            return None

        # Double-buffered weights overlap the next expert DMA with compute.
        # Compacting the 64-element active bitmap uses only scalar SMEM ops.
        lax.fori_loop(0, count, expert_body, None)

    out_ref[...] = owner_ref[...].astype(jnp.bfloat16)


@jax.jit
def dense_expert_moe(hidden, w1, w2, s1, s2, topk_weights, local_ids):
    """Local routed contribution, same [tokens, hidden]/BF16 output as GMM.

    IDs outside this rank must be -1. Scales use [E,1,1,N]. Callers must
    first check can_use_dense_expert. Shared experts and collectives stay
    outside this kernel, exactly as for fused_moe_func's standard path.
    """
    m, h = hidden.shape
    experts, i, _ = w2.shape
    # Dense route coefficients are only [tokens, local experts], not expert
    # output tensors. Sum handles repeated IDs without reloading an expert.
    selector = jax.nn.one_hot(local_ids, experts, dtype=jnp.float32)
    coeff = jnp.sum(selector * topk_weights.astype(jnp.float32)[..., None],
                    axis=1)
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, 128 - experts)))
    return pl.pallas_call(
        functools.partial(_kernel, m=m, h=h, i=i, experts=experts),
        out_shape=jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((2, h, 2 * i), jnp.float8_e4m3fn),
                pltpu.VMEM((2, i, h), jnp.float8_e4m3fn),
                pltpu.VMEM((2, 1, 2 * i), jnp.float32),
                pltpu.VMEM((2, 1, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float8_e4m3fn),
                pltpu.VMEM((h // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.float8_e4m3fn),
                pltpu.VMEM((i // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts, ), jnp.int32),
                pltpu.SemaphoreType.DMA((2, 4)),
            ]),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 *
                                             1024),
        name=f'dense_expert_moe-e_{experts}-m_{m}-h_{h}-i_{i}',
    )(active, hidden, coeff, w1, w2, s1, s2)

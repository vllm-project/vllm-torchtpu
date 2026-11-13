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
"""Experimental small-batch MoE without token permutation.

Each nonempty local expert is visited once. Its two weights are DMA'd once,
all tokens are evaluated in their original order, and weighted outputs are
accumulated in FP32 VMEM. Neither intermediate activations nor per-expert
outputs are written to HBM. The caller still owns top-k and EP collectives.

Two weight formats: FP8 with per-channel scales (W8A8, ``dense_expert_moe``)
and native FP4 with K-block scales (W4A16, ``dense_expert_moe_fp4``).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import swigluoai

_MAX_EXPERT_BUFFERS = 3


def _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes):
    """Choose up to three slots, reserving 20% for padding and temporaries.

    Count explicit VMEM inputs, output and scratch, including quantization
    scales expanded over 128 lanes. As in RPA, this is a sizing heuristic;
    the compiler still decides resource feasibility, even for one slot.
    """
    expert_bytes = 3 * h * i + 4 * (2 * i + h)
    # Hidden input/output, quantized hidden, and FP32 output accumulator.
    fixed_bytes = m * h * (2 + 2 + 1 + 4)
    # BF16 activation, FP8 activation, and the two BF16 quantization scales.
    fixed_bytes += m * i * (2 + 1) + (h // 512 + pl.cdiv(i, 512)) * m * 128 * 2
    fixed_bytes += m * pl.cdiv(experts, 128) * 128 * 4
    budget = vmem_capacity_bytes * 4 // 5
    return max(1, min(_MAX_EXPERT_BUFFERS, (budget - fixed_bytes) // expert_bytes))


def can_use_dense_expert(
    hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype, skip_padded_tokens
):
    """Check arithmetic/layout support; the compiler owns resource limits.

    Zero routing coefficients already suppress padded tokens, so either
    skip_padded_tokens setting is supported. There is no prefill/decode
    inference here: the MoE entry applies its configurable token threshold.
    """
    if activation != "silu":
        return False
    if rhs_quant_dtype not in (None, jnp.float8_e4m3fn):
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    m, h = hidden.shape
    e, i, out = w2.shape
    if min(m, h, i, e) <= 0:
        return False
    if m % 16 or h % 512 or i % 128:
        return False
    if out != h or w1.shape != (e, h, 2 * i):
        return False
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
    ):
        return False
    if not (
        s1.shape == (e, 1, 1, 2 * i)
        and s2.shape == (e, 1, 1, h)
        and s1.dtype == jnp.float32
        and s2.dtype == jnp.float32
    ):
        return False
    # Per-channel GMM quantizes activations only with native FP8 support and
    # a scale block at least one MXU column wide, for both contracting axes.
    tpu_info = pltpu.get_tpu_info()
    return tpu_info.fp8_ops_per_second > 0 and min(h, i) >= tpu_info.mxu_column_size


def _quantize(x, q_ref, scale_ref):
    """Quantize activations to FP8 in up-to-512-wide blocks with BF16 scales."""
    # Keep K blocks first so each scale read is a contiguous [tokens, 128]
    # plane; placing K between tokens and lanes causes expensive relayouts.
    for k in range(0, x.shape[1], 512):
        block = x[:, k : k + 512]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
        inv = jnp.where(scale == 0, 0, 1 / scale)
        q_ref[:, k : k + 512] = (block * inv).astype(jnp.float8_e4m3fn)
        scale_ref[k // 512, :, :] = jnp.broadcast_to(scale, (x.shape[0], 128))


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
            scale = jnp.concatenate(
                (s1_ref[:, n : n + 128], s1_ref[:, i + n : i + n + 128]), axis=1
            )
            for k in range(0, h, 512):
                rhs = jnp.concatenate(
                    (
                        w1_ref[k : k + 512, n : n + 128],
                        w1_ref[k : k + 512, i + n : i + n + 128],
                    ),
                    axis=1,
                )
                part = jnp.matmul(
                    xq_ref[:, k : k + 512], rhs, preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
                xs = xs_ref[k // 512, :, :]
                part *= jnp.concatenate((xs, xs), axis=1)
                part *= scale.astype(jnp.bfloat16)
                acc += part
            a_ref[:, n : n + 128] = jax.nn.silu(acc[:, :128]) * acc[:, 128:]
        _quantize(a_ref[...], aq_ref, asc_ref)
        for n in range(0, h, 256):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, i, 512):
                part = jnp.matmul(
                    aq_ref[:, k : k + 512],
                    w2_ref[k : k + 512, n : n + 256],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                asc = asc_ref[k // 512, :, :]
                part *= jnp.concatenate((asc, asc), axis=1)
                part *= s2_ref[:, n : n + 256].astype(jnp.bfloat16)
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
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_ref.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, 0], s2_ref.at[slot], sem_ref.at[slot, 3]
            ),
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


@jax.jit
def dense_expert_moe(hidden, w1, w2, s1, s2, topk_weights, local_ids):
    """Local routed contribution, same [tokens, hidden]/BF16 output as GMM.

    IDs outside this rank must be -1. Scales use [E,1,1,N]. Callers must
    first check can_use_dense_expert. Shared experts and collectives stay
    outside this kernel, exactly as for fused_moe_func's standard path.
    """
    m, h = hidden.shape
    experts, i, _ = w2.shape
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    num_buffers = _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes)
    # Dense route coefficients are only [tokens, local experts], not expert
    # output tensors. Accumulation handles repeated IDs without reloading an
    # expert. Unroll the static top-k dimension to avoid the [M,K,E] reduction
    # whose layout copies can be offloaded to SparseCore and delay all-gather.
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    return pl.pallas_call(
        functools.partial(
            _kernel, m=m, h=h, i=i, experts=experts, num_buffers=num_buffers
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
                pltpu.VMEM((num_buffers, h, 2 * i), jnp.float8_e4m3fn),
                pltpu.VMEM((num_buffers, i, h), jnp.float8_e4m3fn),
                pltpu.VMEM((num_buffers, 1, 2 * i), jnp.float32),
                pltpu.VMEM((num_buffers, 1, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float8_e4m3fn),
                pltpu.VMEM((h // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.float8_e4m3fn),
                pltpu.VMEM((pl.cdiv(i, 512), m, 128), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((num_buffers, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity_bytes),
        name=(f"dense_expert_moe-e_{experts}-m_{m}-h_{h}-i_{i}-buffers_{num_buffers}"),
    )(active, hidden, coeff, w1, w2, s1, s2)


_FP4 = jnp.float4_e2m1fn
_FP8 = jnp.float8_e4m3fn
_FP4_ACTIVATIONS = {
    "silu": lambda gate, up: jax.nn.silu(gate) * up,
    "swigluoai": swigluoai,
}
# Down-projection output columns per accumulator: bounds the live f32 tile.
_FP4_DOWN_COLS = 2048
_MAX_FP4_EXPERT_BUFFERS = 4


def _fp4_expert_bytes(h, i, bk1, bk2):
    """VMEM of one FP4 expert slot. A [blocks, 1, N] f32 scale pads its unit
    sublane dimension to a full 8-row tile."""
    return h * i + 4 * 8 * (h // bk1 * 2 * i + i // bk2 * h)


def _select_fp4_expert_buffers(m, h, i, experts, bk1, bk2, vmem_capacity_bytes):
    """Choose up to four slots within 80% of VMEM, or 0 if two do not fit."""
    fixed_bytes = m * h * (2 + 2 + 4) + m * pl.cdiv(experts, 128) * 128 * 4
    budget = vmem_capacity_bytes * 4 // 5
    slots = (budget - fixed_bytes) // _fp4_expert_bytes(h, i, bk1, bk2)
    return min(_MAX_FP4_EXPERT_BUFFERS, slots) if slots >= 2 else 0


def can_use_dense_expert_fp4(
    hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype
):
    """W4A16 arithmetic/layout support for ``dense_expert_moe_fp4``.

    Weights are native FP4, K-major, with one f32 scale per K block per output
    column (gmm_v2's ``[E, K // block, 1, N]`` layout); activations stay bf16
    and the FP4 values are fed to the MXU as FP8, as gmm_v2 does at decode.
    """
    if activation not in _FP4_ACTIVATIONS or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    if s1.ndim != 4 or s2.ndim != 4:
        return False
    m, h = hidden.shape
    e, i, out = w2.shape
    if min(m, h, i, e) <= 0 or out != h or w1.shape != (e, h, 2 * i):
        return False
    if hidden.dtype != jnp.bfloat16 or w1.dtype != _FP4 or w2.dtype != _FP4:
        return False
    if s1.dtype != jnp.float32 or s2.dtype != jnp.float32:
        return False
    if h % s1.shape[1] or i % s2.shape[1]:
        return False
    bk1, bk2 = h // s1.shape[1], i // s2.shape[1]
    if s1.shape != (e, h // bk1, 1, 2 * i) or s2.shape != (e, i // bk2, 1, h):
        return False
    tpu_info = pltpu.get_tpu_info()
    lanes = tpu_info.num_lanes
    if h % _FP4_DOWN_COLS or i % lanes or bk1 % lanes or bk2 % lanes:
        return False
    if not tpu_info.is_matmul_supported(jnp.bfloat16, _FP8):
        return False
    m_pad = pl.cdiv(m, 16) * 16
    return (
        _select_fp4_expert_buffers(
            m_pad, h, i, e, bk1, bk2, tpu_info.vmem_capacity_bytes
        )
        > 0
    )


def _kernel_fp4(
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
    owner_ref,
    active_ids_ref,
    sem_ref,
    *,
    m,
    h,
    i,
    experts,
    num_buffers,
    bk1,
    bk2,
    activation,
):
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
    act_fn = _FP4_ACTIVATIONS[activation]

    def compute(e, slot):
        coeff = jnp.sum(
            jnp.where(expert_columns == e, coeff_ref[...], 0), axis=1, keepdims=True
        )
        # gmm_v2's decode arithmetic: per K block an FP8-fed matmul into f32,
        # scaled per output column and summed in f32; bf16 between the two
        # projections and at the down projection's output.
        acc = jnp.zeros((m, 2 * i), jnp.float32)
        for b in range(h // bk1):
            rhs = w1_ref[slot, b * bk1 : (b + 1) * bk1, :].astype(_FP8)
            part = jnp.matmul(
                x_ref[:, b * bk1 : (b + 1) * bk1],
                rhs,
                preferred_element_type=jnp.float32,
            )
            acc += part * s1_ref[slot, b]
        act = act_fn(acc[:, :i], acc[:, i:]).astype(jnp.bfloat16)
        for n in range(0, h, _FP4_DOWN_COLS):
            acc2 = jnp.zeros((m, _FP4_DOWN_COLS), jnp.float32)
            for b in range(i // bk2):
                rhs = w2_ref[
                    slot, b * bk2 : (b + 1) * bk2, n : n + _FP4_DOWN_COLS
                ].astype(_FP8)
                part = jnp.matmul(
                    act[:, b * bk2 : (b + 1) * bk2],
                    rhs,
                    preferred_element_type=jnp.float32,
                )
                acc2 += part * s2_ref[slot, b, :, n : n + _FP4_DOWN_COLS]
            down = acc2.astype(jnp.bfloat16).astype(jnp.float32)
            # Zero routing must suppress even non-finite unused expert outputs.
            owner_ref[:, n : n + _FP4_DOWN_COLS] += jnp.where(
                coeff != 0, down * coeff, 0
            )

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = active_ids_ref[step], step % num_buffers
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(s1_hbm.at[e], s1_ref.at[slot], sem_ref.at[slot, 2]),
            pltpu.make_async_copy(s2_hbm.at[e], s2_ref.at[slot], sem_ref.at[slot, 3]),
        ]

    @pl.when(count > 0)
    def run_experts():
        for initial in range(num_buffers - 1):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        def expert_body(step, _):
            for copy in transfers(step):
                copy.wait()

            @pl.when(step + num_buffers - 1 < count)
            def prefetch():
                for copy in transfers(step + num_buffers - 1):
                    copy.start()

            compute(active_ids_ref[step], step % num_buffers)
            return None

        lax.fori_loop(0, count, expert_body, None)

    out_ref[...] = owner_ref[...].astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("activation",))
def dense_expert_moe_fp4(
    hidden, w1, w2, s1, s2, topk_weights, local_ids, *, activation
):
    """Local routed contribution of FP4 experts, [tokens, hidden] bf16 as GMM.

    IDs outside this rank must be -1. Callers must first check
    can_use_dense_expert_fp4. Shared experts and collectives stay outside.
    """
    num_tokens, h = hidden.shape
    experts, i, _ = w2.shape
    bk1, bk2 = h // s1.shape[1], i // s2.shape[1]
    # bf16 rows tile by 16; padded rows carry no route and stay zero.
    m = pl.cdiv(num_tokens, 16) * 16
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    num_buffers = _select_fp4_expert_buffers(
        m, h, i, experts, bk1, bk2, vmem_capacity_bytes
    )
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    coeff = jnp.zeros((num_tokens, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, m - num_tokens), (0, (-experts) % 128)))
    hidden = jnp.pad(hidden, ((0, m - num_tokens), (0, 0)))
    out = pl.pallas_call(
        functools.partial(
            _kernel_fp4,
            m=m,
            h=h,
            i=i,
            experts=experts,
            num_buffers=num_buffers,
            bk1=bk1,
            bk2=bk2,
            activation=activation,
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
                pltpu.VMEM((num_buffers, h, 2 * i), _FP4),
                pltpu.VMEM((num_buffers, i, h), _FP4),
                pltpu.VMEM((num_buffers, h // bk1, 1, 2 * i), jnp.float32),
                pltpu.VMEM((num_buffers, i // bk2, 1, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((num_buffers, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity_bytes),
        name=(
            f"dense_expert_moe_fp4-e_{experts}-m_{m}-h_{h}-i_{i}-bk_{bk1}_{bk2}"
            f"-act_{activation}-buffers_{num_buffers}"
        ),
    )(active, hidden, coeff, w1, w2, s1, s2)
    return out[:num_tokens]

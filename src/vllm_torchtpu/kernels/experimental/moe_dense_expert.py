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
"""Experimental small-batch FP8 MoE without token permutation.

Each nonempty local expert is visited once. Its two weights are DMA'd once,
all tokens are evaluated in their original order, and weighted outputs are
accumulated in FP32 VMEM. Neither intermediate activations nor per-expert
outputs are written to HBM. The caller still owns top-k and EP collectives.

The arithmetic is gmm_v2's for per-channel FP8 weights: activations are
quantized per row in 512-column blocks (a narrower last block when the
contracting size is not a multiple of 512), each block's product is rounded
to BF16, scaled and summed in BF16, and the gated activation (SiLU or tanh
GELU) is gmm_v2's own.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act

_MAX_EXPERT_BUFFERS = 3
_ACTIVATIONS = ("silu", "gelu_tanh")
# Room for Mosaic's own temporaries beyond the counted buffers. Deviceless
# tpu7x compiles of Gemma4-26B EP2 experts need at most the counted bytes plus
# 1 MiB at 16-256 tokens; the limit compiles at 16-1024 tokens, and at 256
# tokens of hidden/inter 4096/1024 and 5376/512.
_VMEM_HEADROOM_BYTES = 4 * 1024 * 1024


def _buffer_bytes(m, h, i):
    """(bytes per expert slot, bytes of the other VMEM inputs, output, scratch).

    Quantization scales count expanded over 128 lanes.
    """
    expert_bytes = 3 * h * i + 4 * (2 * i + h)
    # Hidden input/output, quantized hidden, and FP32 output accumulator.
    fixed_bytes = m * h * (2 + 2 + 1 + 4)
    # BF16 activation, FP8 activation, and the two BF16 quantization scales.
    fixed_bytes += m * i * (2 + 1) + (pl.cdiv(h, 512) + pl.cdiv(i, 512)) * m * 128 * 2
    # Route ids and weights, [tokens, top-k] padded to 128 lanes.
    fixed_bytes += 2 * m * 128 * 4
    return expert_bytes, fixed_bytes


def _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes):
    """Choose up to three slots, reserving 20% for padding and temporaries.

    As in RPA, this is a sizing heuristic; the compiler still decides
    resource feasibility, even for one slot.
    """
    expert_bytes, fixed_bytes = _buffer_bytes(m, h, i)
    budget = vmem_capacity_bytes * 4 // 5
    return max(1, min(_MAX_EXPERT_BUFFERS, (budget - fixed_bytes) // expert_bytes))


def _vmem_limit_bytes(m, h, i, num_buffers, vmem_capacity_bytes):
    """The kernel's buffers plus headroom, not all of VMEM.

    XLA keeps its own prefetched operands, such as the dense MLP weights the
    layer reads next, in the VMEM this call leaves free while it runs.
    """
    expert_bytes, fixed_bytes = _buffer_bytes(m, h, i)
    need = fixed_bytes + num_buffers * expert_bytes + _VMEM_HEADROOM_BYTES
    return min(vmem_capacity_bytes, need)


def can_use_dense_expert(
    hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype, skip_padded_tokens
):
    """Check arithmetic/layout support; the compiler owns resource limits.

    Zero routing coefficients already suppress padded tokens, so either
    skip_padded_tokens setting is supported. There is no prefill/decode
    inference here: the MoE entry applies its configurable token threshold.
    """
    if activation not in _ACTIVATIONS:
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
    # The down projection writes 256-column output tiles.
    if m % 16 or h % 256 or i % 128:
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
    route_ids_smem,
    route_weight_bits_smem,
    start_smem,
    x_ref,
    ids_ref,
    weights_ref,
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
    active_ref,
    active_ids_ref,
    sem_ref,
    *,
    m,
    h,
    i,
    experts,
    num_buffers,
    activation,
):
    start = start_smem[0]

    def compute(e, w1_ref, w2_ref, s1_ref, s2_ref):
        # Each token's weight for expert e (a token routes to an expert once,
        # so at most one term is nonzero and the sum is exact).
        coeff = jnp.sum(
            jnp.where(ids_ref[...] - start == e, weights_ref[...], 0.0),
            axis=1,
            keepdims=True,
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
            a_ref[:, n : n + 128] = apply_gated_act(
                acc[:, :128], acc[:, 128:], activation
            )
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

    # Scalar routing: the local experts any route with a nonzero weight hits
    # (routes to other ranks' experts fall outside [0, experts)).
    def clear(e, carry):
        active_ref[e] = 0
        return carry

    lax.fori_loop(0, experts, clear, 0)

    def mark(route, carry):
        e = route_ids_smem[route] - start
        live = (e >= 0) & (e < experts)
        live &= (route_weight_bits_smem[route] & 0x7FFFFFFF) != 0

        @pl.when(live)
        def record():
            active_ref[e] = 1

        return carry

    lax.fori_loop(0, route_ids_smem.shape[0], mark, 0)

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

    for initial in range(num_buffers - 1):

        @pl.when(count > initial)
        def prime(*, initial=initial):
            for copy in transfers(jnp.int32(initial)):
                copy.start()

    # Under the first experts' weight copies.
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    _quantize(x_ref[...], xq_ref, xs_ref)

    @pl.when(count > 0)
    def run_experts():
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


@functools.partial(jax.jit, static_argnames=("activation",))
def dense_expert_moe(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    topk_ids,
    activation="silu",
    experts_start=None,
):
    """Local routed contribution, same [tokens, hidden]/BF16 output as GMM.

    ``topk_ids`` are this rank's local expert ids (other ranks' routes -1),
    or global ids with ``experts_start``, the first global id of this rank's
    experts. Routes with a zero weight are skipped; weights are rounded to
    the hidden dtype as the GMM path pins them. Scales use [E,1,1,N].
    Callers must first check can_use_dense_expert. Shared experts and
    collectives stay outside this kernel, exactly as for fused_moe_func's
    standard path.
    """
    m, h = hidden.shape
    experts, i, _ = w2.shape
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    num_buffers = _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes)
    vmem_limit_bytes = _vmem_limit_bytes(m, h, i, num_buffers, vmem_capacity_bytes)
    # The routing is resolved inside the kernel: XLA only rounds the weights
    # and hands over the [tokens, top-k] ids, so no per-expert table or
    # active mask is built (and no EP remap) between the router and here.
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    ids = topk_ids.astype(jnp.int32)
    start = (
        jnp.zeros((1,), jnp.int32)
        if experts_start is None
        else jnp.reshape(experts_start, (1,)).astype(jnp.int32)
    )
    return pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            num_buffers=num_buffers,
            activation=activation,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
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
                pltpu.VMEM((pl.cdiv(h, 512), m, 128), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.float8_e4m3fn),
                pltpu.VMEM((pl.cdiv(i, 512), m, 128), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((num_buffers, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name=(
            f"dense_expert_moe-e_{experts}-m_{m}-h_{h}-i_{i}-{activation}"
            f"-buffers_{num_buffers}"
        ),
    )(
        ids.reshape(-1),
        lax.bitcast_convert_type(weights, jnp.int32).reshape(-1),
        start,
        hidden,
        ids,
        weights,
        w1,
        w2,
        s1,
        s2,
    )

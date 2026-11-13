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
"""Small-batch NVFP4 (W4A16) MoE without token permutation.

The decode-sized counterpart of moe_dense_expert for fp4 experts with fp32
[128, 1] block scales. Each nonempty local expert is streamed once, in units of
`chunk` intermediate columns (its gate, up and down slices), through a ring of
VMEM buffers; every token is evaluated in its original order and the weighted
expert outputs accumulate in fp32 VMEM. This replaces the routing sort, the
one-hot permute, the grouped GEMM and the combine with one call. The fp4
weights feed the MXU as fp8 (exact) against bf16 activations, and each 128-row
block scale is applied to the block's partial product, as gmm_v2's decode
tiles do. Rounding follows the GMM path: the activation is rounded to bf16
before the down projection, each expert's output to bf16 before its routing
weight is applied.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane

_QUANT_BLOCK = 128
_NUM_BUFFERS = 3
_ACTIVATIONS = ("silu", "swigluoai")


def _chunk(i: int) -> int:
    for c in (1024, 512, 384, 256, 128):
        if i % c == 0:
            return c
    return 0


def can_use_dense_expert_fp4(hidden, w1, w2, s1, s2, *, activation) -> bool:
    """Layout and arithmetic support; the token threshold is the caller's."""
    if activation not in _ACTIVATIONS or hidden.ndim != 2:
        return False
    m, h = hidden.shape
    if w1.ndim != 3 or w2.ndim != 3 or s1 is None or s2 is None:
        return False
    e, i, h2 = w2.shape
    fp4 = jnp.float4_e2m1fn
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != fp4
        or w2.dtype != fp4
        or h2 != h
        or w1.shape != (e, h, 2 * i)
        or min(m, h, i, e) <= 0
        or h % _QUANT_BLOCK
        or _chunk(i) == 0
    ):
        return False
    if s1.shape != (e, h // _QUANT_BLOCK, 1, 2 * i) or s2.shape != (
        e,
        i // _QUANT_BLOCK,
        1,
        h,
    ):
        return False
    if s1.dtype != jnp.float32 or s2.dtype != jnp.float32:
        return False
    return pltpu.get_tpu_info().is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn)


def _kernel(
    active_ref,  # SMEM [E] (scalar prefetch)
    x_ref,  # VMEM [m, h] bf16
    coeff_ref,  # VMEM [m, E_pad] f32
    w1_hbm,  # [E, h, 2i] fp4
    w2_hbm,  # [E, i, h] fp4
    s1_hbm,  # [E, h / 128, 1, 2i] f32
    s2_hbm,  # [E, i / 128, 1, h] f32
    out_ref,  # VMEM [m, h] bf16
    wg_ref,
    wu_ref,
    wd_ref,
    sg_ref,
    su_ref,
    sd_ref,
    expert_acc_ref,  # VMEM [m, h] f32: the current expert's down projection
    acc_ref,  # VMEM [m, h] f32: the routing-weighted sum
    active_ids_ref,  # SMEM [E]
    sem_ref,
    *,
    h,
    i,
    chunk,
    experts,
    activation,
):
    fp8 = jnp.float8_e4m3fn
    num_chunks = i // chunk
    acc_ref[...] = jnp.zeros_like(acc_ref)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    units = count * num_chunks

    def transfers(unit):
        e = active_ids_ref[unit // num_chunks]
        col = pl.multiple_of((unit % num_chunks) * chunk, chunk)
        blk = (unit % num_chunks) * (chunk // _QUANT_BLOCK)
        slot = unit % _NUM_BUFFERS
        sem = sem_ref.at[slot]
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(col, chunk)], wg_ref.at[slot], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(i + col, chunk)], wu_ref.at[slot], sem.at[1]
            ),
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(col, chunk), :], wd_ref.at[slot], sem.at[2]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(col, chunk)], sg_ref.at[slot], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(i + col, chunk)], su_ref.at[slot], sem.at[4]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, pl.ds(blk, chunk // _QUANT_BLOCK), :],
                sd_ref.at[slot],
                sem.at[5],
            ),
        ]

    def compute(unit):
        slot = unit % _NUM_BUFFERS
        c = unit % num_chunks
        x = x_ref[...]
        gate = jnp.zeros((x.shape[0], chunk), jnp.float32)
        up = jnp.zeros((x.shape[0], chunk), jnp.float32)
        for k in range(0, h, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            xk = x[:, k : k + _QUANT_BLOCK]
            gate += (
                jnp.matmul(
                    xk,
                    wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * sg_ref[slot, b]
            )
            up += (
                jnp.matmul(
                    xk,
                    wu_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * su_ref[slot, b]
            )
        act = apply_act_fn(interleave_lane(gate, up), activation).astype(x.dtype)
        down = jnp.zeros((x.shape[0], h), jnp.float32)
        for k in range(0, chunk, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            down += (
                jnp.matmul(
                    act[:, k : k + _QUANT_BLOCK],
                    wd_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * sd_ref[slot, b]
            )

        @pl.when(c == 0)
        def first():
            expert_acc_ref[...] = down

        @pl.when(c > 0)
        def rest():
            expert_acc_ref[...] += down

        @pl.when(c == num_chunks - 1)
        def finish():
            e = active_ids_ref[unit // num_chunks]
            coeff = jnp.sum(
                jnp.where(expert_columns == e, coeff_ref[...], 0),
                axis=1,
                keepdims=True,
            )
            out_e = expert_acc_ref[...].astype(out_ref.dtype).astype(jnp.float32)
            # Zero routing must suppress even a non-finite unused output.
            acc_ref[...] += jnp.where(coeff != 0, out_e * coeff, 0)

    @pl.when(units > 0)
    def run():
        for initial in range(_NUM_BUFFERS - 1):

            @pl.when(units > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        def body(unit, carry):
            for copy in transfers(unit):
                copy.wait()

            @pl.when(unit + _NUM_BUFFERS - 1 < units)
            def prefetch():
                for copy in transfers(unit + _NUM_BUFFERS - 1):
                    copy.start()

            compute(unit)
            return carry

        lax.fori_loop(0, units, body, 0)

    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("activation",))
def dense_expert_moe_fp4(
    hidden, w1, w2, s1, s2, topk_weights, local_ids, *, activation="swigluoai"
):
    """Local routed contribution of fp4 experts, [tokens, hidden] bf16, as
    fused_moe_func's GMM path returns it. IDs outside this rank must be -1.
    Callers check can_use_dense_expert_fp4 first."""
    m, h = hidden.shape
    experts, i, _ = w2.shape
    chunk = _chunk(i)
    sublane = 16
    pad = -m % sublane
    x = jnp.where(jnp.isfinite(hidden), hidden, 0)
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    weights = jnp.where(jnp.isfinite(weights), weights, 0)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    lanes = pltpu.get_tpu_info().num_lanes
    coeff = jnp.pad(coeff, ((0, pad), (0, (-experts) % lanes)))
    x = jnp.pad(x, ((0, pad), (0, 0)))
    mp = m + pad
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            h=h,
            i=i,
            chunk=chunk,
            experts=experts,
            activation=activation,
        ),
        out_shape=jax.ShapeDtypeStruct((mp, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, chunk, h), w2.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, chunk // _QUANT_BLOCK, 1, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((_NUM_BUFFERS, 6)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name=f"dense_expert_moe_fp4-e_{experts}-m_{mp}-h_{h}-i_{i}-c_{chunk}",
    )(
        active,
        x,
        coeff,
        w1,
        w2,
        # 4-D as the GMM path holds them: a reshape that drops the size-1
        # axis is a relayout copy of every scale on each call.
        s1,
        s2,
    )
    return out[:m]

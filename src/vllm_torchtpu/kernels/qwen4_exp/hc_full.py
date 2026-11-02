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
"""Optional prefill HyperConnection mixer with resident weights and two tiles.

Normalization and SiLU use FP32 internally, matching upstream NVIDIA HC.
Projection and stored outputs remain BF16. The kernel stages the
normalization and both projection weights once, overlaps the next input tile
with the current computation, and drains output tiles only before reuse.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    HcHardware,
    supports_full_mix,
)


def _kernel(
    x_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    mixed_hbm,
    norm_hbm,
    x_vm,
    weight_vm,
    down_vm,
    up_vm,
    mixed_vm,
    norm_vm,
    x_sems,
    weight_sem,
    down_sem,
    up_sem,
    mixed_sems,
    norm_sems,
    *,
    token_tile,
    tokens,
    hidden,
    branches,
    eps,
):
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(down_hbm, down_vm, down_sem)
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    weight_dma.start()
    down_dma.start()
    up_dma.start()
    pltpu.make_async_copy(
        x_hbm.at[pl.ds(0, token_tile)], x_vm.at[0], x_sems.at[0]
    ).start()

    def one_step(step, parity, weight):
        first = weight is None
        # The previous step started this input copy. Wait at its first read.
        pltpu.make_async_copy(
            x_vm.at[parity], x_vm.at[parity], x_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_input():
            pltpu.make_async_copy(
                x_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                x_vm.at[1 - parity],
                x_sems.at[1 - parity],
            ).start()

        if first:
            weight_dma.wait()
            weight = weight_vm[...].astype(jnp.float32)

        @pl.when(step >= 2)
        def _wait_prior_norm_output():
            pltpu.make_async_copy(
                norm_vm.at[parity], norm_vm.at[parity], norm_sems.at[parity]
            ).wait()

        # Bound normalization FP32 work to one32-row branch strip. This does
        # not bound the later full-width logits/weighted epilogue footprint.
        def normalize_rows(index, carry):
            offset = index * 32
            for branch in range(branches):
                start = branch * hidden
                values = x_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                variance = jnp.mean(values * values, axis=-1, keepdims=True)
                normalized = values * jax.lax.rsqrt(variance + eps)
                scaled = normalized * weight[None, start : start + hidden]
                norm_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)] = (
                    normalized + scaled
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
        norm = norm_vm[parity]
        pltpu.make_async_copy(
            norm_vm.at[parity],
            norm_hbm.at[pl.ds(step * token_tile, token_tile)],
            norm_sems.at[parity],
        ).start()

        if first:
            down_dma.wait()

        down = jax.lax.dot_general(
            norm,
            down_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before NVIDIA's FP32 sigmoid/mix.
        logits = (
            jax.lax.dot_general(
                gate,
                up_vm[...],
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        weighted = jax.nn.sigmoid(logits) * norm.astype(jnp.float32)
        mixed = weighted[:, :hidden]
        for branch in range(1, branches):
            start = branch * hidden
            mixed = mixed + weighted[:, start : start + hidden]

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            mixed_vm.at[parity],
            mixed_hbm.at[pl.ds(step * token_tile, token_tile)],
            mixed_sems.at[parity],
        ).start()
        return weight

    # Peel the first pair so weight waits/cast are static first-tile work.
    # Down/up waits stay at their first reads, overlapping normalization.
    tiles = tokens // token_tile
    weight = one_step(0, 0, None)
    if tiles > 1:
        one_step(1, 1, weight)

    def step_pair(pair, carry):
        one_step(2 * pair, 0, weight)
        one_step(2 * pair + 1, 1, weight)
        return carry

    jax.lax.fori_loop(1, tiles // 2, step_pair, jnp.int32(0))
    if tiles > 1 and tiles % 2:
        one_step(tiles - 1, 0, weight)
    # A one-tile call never started parity1; do not wait its semaphore.
    for parity in range(min(tiles, 2)):
        pltpu.make_async_copy(
            norm_vm.at[parity], norm_vm.at[parity], norm_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def _full_mix(x, weight, down, up, *, branches, hidden, eps, token_tile, hardware=None):
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError(
            "HC full mix requires positive branches and 128-aligned hidden"
        )
    if token_tile <= 0 or token_tile % 128:
        raise ValueError("HC full mix requires positive 128-aligned token tiles")
    tokens, width = x.shape
    lowrank = down.shape[0]
    if (
        tokens == 0
        or lowrank == 0
        or tokens % token_tile
        or width != branches * hidden
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up))
    ):
        raise ValueError("HC full mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_full_mix(branches, hidden, lowrank, token_tile, hardware):
        raise ValueError("HC full mix requires a supported layout and VMEM budget")
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * 4,
            out_specs=(hbm,) * 2,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.VMEM((width,), jnp.bfloat16),
                pltpu.VMEM((lowrank, width), jnp.bfloat16),
                pltpu.VMEM((width, lowrank), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # Two GEMMs at two FLOPs/MAC; normalization, SiLU and branch mixing.
        # Count each logical rsqrt/sigmoid as one transcendental operation.
        # Weights are read once; input and both outputs once per token. These
        # unpadded logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * width * lowrank
            + tokens * (7 * width + branches + 2 * lowrank),
            transcendentals=tokens * (branches + lowrank + width),
            bytes_accessed=sum(a.size * a.dtype.itemsize for a in (x, weight, down, up))
            + tokens * (width + hidden) * x.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_branchstore32_m{token_tile}",
    )(x, weight, down, up)

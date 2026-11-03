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
"""Prefill HyperConnection mix that also projects the injection gates.

The full-width normalized hyper input never reaches HBM on this path: the
kernel normalizes each staged input tile in place, projects the low-rank
mixing gates and the per-branch injection logits from that vector-memory
value, and stores only the mixed block input plus the [tokens, branches]
logits that ``combine`` turns into FP32 gates. The down and inject
projections contract the same normalized BF16 value over the full width, so
their live HBM weights are DMA-staged into two nonoverlapping row ranges of
one resident plane and consumed by a single fullK FP32 dot: the inject rows
start at the first hardware row boundary at or after the down rows, the
boundary rows in between are written to zero before the first read, and each
projection's columns keep their own BF16 store boundary. Stores keep the
checkpoint BF16 boundaries of the upstream kernels: the injection logits
round to BF16 exactly where a standalone linear over the stored norm would.

The broadcast variant serves the chain's first mix, whose residual arrives
un-expanded: every branch slice of the expanded hyper input would repeat the
same [tokens, hidden] value, so the kernel stages that value once, normalizes
one strip per 32 rows exactly as the expanded input would be normalized, and
writes each branch's affine-scaled normalized value into the full-width dot
operand. No expanded plane is ever materialized. The staged values are the
same words the expanded plane would carry, but the variance reduces over one
``hidden`` strip instead of each branch's repeated slice; that is
mathematically the same normalization while FP32 sum order can legitimately
change output words, so the qualification oracles are the finite/L2/cosine
tolerances, not cross-implementation bit equality.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_common import (
    invalid_mix_rows,
    invalid_mix_weights,
    project_up,
    run_mix_tiles,
    validate_mix_geometry,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    HcHardware,
    packed_projection_rows,
    supports_inject_broadcast_mix,
    supports_inject_mix,
)


def _kernel(
    x_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    inject_hbm,
    mixed_hbm,
    logits_hbm,
    x_vm,
    weight_vm,
    packed_vm,
    up_vm,
    mixed_vm,
    logits_vm,
    x_sems,
    weight_sem,
    down_sem,
    up_sem,
    inject_sem,
    mixed_sems,
    logits_sems,
    *,
    token_tile,
    tokens,
    hidden,
    branches,
    eps,
    lowrank,
    inject_row,
):
    gap_rows = inject_row - lowrank
    # The two weight ranges tile one resident plane: down rows [0, lowrank)
    # and inject rows [inject_row, inject_row + branches) never overlap.
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(
        down_hbm, packed_vm.at[pl.ds(0, lowrank)], down_sem
    )
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    inject_dma = pltpu.make_async_copy(
        inject_hbm, packed_vm.at[pl.ds(inject_row, branches)], inject_sem
    )
    weight_dma.start()
    down_dma.start()
    up_dma.start()
    inject_dma.start()
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
        def _wait_prior_logits_output():
            pltpu.make_async_copy(
                logits_vm.at[parity], logits_vm.at[parity], logits_sems.at[parity]
            ).wait()

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        # Normalize one 32-row branch strip straight back into the staged
        # input tile: the original input rows have no reader after the strip's
        # variance is formed, so no separate full-width plane is needed.
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
                x_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)] = (
                    normalized + scaled
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
        norm = x_vm[parity]

        if first:
            # Both packed ranges land before the single projection read.
            down_dma.wait()
            inject_dma.wait()
            if gap_rows:
                # The boundary rows belong to no DMA range; define them so the
                # packed dot never reads undefined vector memory.
                packed_vm[pl.ds(lowrank, gap_rows)] = jnp.zeros(
                    (gap_rows, branches * hidden), jnp.bfloat16
                )

        # One fullK FP32 dot serves both projections: the down columns and the
        # injection columns round through BF16 exactly where the separate
        # linears over the stored normalized value would. Injection logits
        # round at the same BF16 boundary; combine applies FP32 sigmoid.
        projected = jax.lax.dot_general(
            norm,
            packed_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        down = projected[:, :lowrank].astype(jnp.bfloat16)
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        logits = projected[:, inject_row : inject_row + branches].astype(jnp.bfloat16)
        logits_vm[parity] = logits
        pltpu.make_async_copy(
            logits_vm.at[parity],
            logits_hbm.at[pl.ds(step * token_tile, token_tile)],
            logits_sems.at[parity],
        ).start()

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before NVIDIA's FP32 sigmoid/mix.
        logits_up = project_up(gate, up_vm[...])
        weighted = jax.nn.sigmoid(logits_up) * norm.astype(jnp.float32)
        mixed = weighted[:, :hidden]
        for branch in range(1, branches):
            start = branch * hidden
            mixed = mixed + weighted[:, start : start + hidden]

        mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            mixed_vm.at[parity],
            mixed_hbm.at[pl.ds(step * token_tile, token_tile)],
            mixed_sems.at[parity],
        ).start()
        return weight

    # Peel the first pair so weight waits/cast are static first-tile work.
    tiles = tokens // token_tile
    run_mix_tiles(one_step, tiles)
    # A one-tile call never started parity1; do not wait its semaphore.
    for parity in range(min(tiles, 2)):
        pltpu.make_async_copy(
            logits_vm.at[parity], logits_vm.at[parity], logits_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def full_mix_inject(
    x,
    weight,
    down,
    up,
    inject,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    hardware=None,
):
    """Mix with in-kernel injection projection.

    Args:
        x: BF16 [tokens, branches * hidden] hyper input.
        weight: BF16 [branches * hidden] per-branch norm affine weight.
        down: BF16 [lowrank, branches * hidden] down projection.
        up: BF16 [branches * hidden, lowrank] up projection.
        inject: BF16 [branches, branches * hidden] injection projection.
        branches: HC branch count derived from the tensors.
        hidden: Hidden size per branch.
        eps: RMS norm epsilon.
        token_tile: Token rows per step, multiple of 128.
        hardware: Optional immutable layout facts.

    Returns:
        (mixed [tokens, hidden] BF16, logits [tokens, branches] BF16).

    Raises:
        ValueError: Geometry, operand shapes, dtypes, or resources unsupported.
    """
    validate_mix_geometry(branches, hidden, token_tile, "HC inject mix")
    tokens, width = x.shape
    lowrank = down.shape[0]
    if (
        invalid_mix_rows(tokens, lowrank, token_tile, width, branches * hidden)
        or invalid_mix_weights(weight, down, up, width, lowrank)
        or inject.shape != (branches, width)
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up, inject))
    ):
        raise ValueError("HC inject mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_inject_mix(branches, hidden, lowrank, token_tile, hardware):
        raise ValueError("HC inject mix requires a supported layout and VMEM budget")
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
            lowrank=lowrank,
            inject_row=inject_row,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * 5,
            out_specs=(hbm,) * 2,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.VMEM((width,), jnp.bfloat16),
                pltpu.VMEM((packed_rows, width), jnp.bfloat16),
                pltpu.VMEM((width, lowrank), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, branches), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # The up GEMM and the packed down+inject dot at two FLOPs/MAC, with
        # the normalization, SiLU and branch mixing elementwise work. Weights
        # are read once; the input, mixed output and logits once per token.
        # These unpadded logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=2 * tokens * width * (lowrank + packed_rows)
            + tokens * (7 * width + branches + 2 * lowrank + branches),
            transcendentals=tokens * (branches + lowrank + width),
            bytes_accessed=sum(
                a.size * a.dtype.itemsize for a in (x, weight, down, up, inject)
            )
            + tokens * (hidden + branches) * x.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_inject_m{token_tile}",
    )(x, weight, down, up, inject)


def _broadcast_kernel(
    x_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    inject_hbm,
    mixed_hbm,
    logits_hbm,
    input_vm,
    norm_vm,
    weight_vm,
    packed_vm,
    up_vm,
    mixed_vm,
    logits_vm,
    input_sems,
    weight_sem,
    down_sem,
    up_sem,
    inject_sem,
    mixed_sems,
    logits_sems,
    *,
    token_tile,
    tokens,
    hidden,
    branches,
    eps,
    lowrank,
    inject_row,
):
    gap_rows = inject_row - lowrank
    # The two weight ranges tile one resident plane: down rows [0, lowrank)
    # and inject rows [inject_row, inject_row + branches) never overlap.
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(
        down_hbm, packed_vm.at[pl.ds(0, lowrank)], down_sem
    )
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    inject_dma = pltpu.make_async_copy(
        inject_hbm, packed_vm.at[pl.ds(inject_row, branches)], inject_sem
    )
    weight_dma.start()
    down_dma.start()
    up_dma.start()
    inject_dma.start()
    pltpu.make_async_copy(
        x_hbm.at[pl.ds(0, token_tile)], input_vm.at[0], input_sems.at[0]
    ).start()

    def one_step(step, parity, weight):
        first = weight is None
        # The previous step started this input copy. Wait at its first read.
        pltpu.make_async_copy(
            input_vm.at[parity], input_vm.at[parity], input_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_input():
            pltpu.make_async_copy(
                x_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                input_vm.at[1 - parity],
                input_sems.at[1 - parity],
            ).start()

        if first:
            weight_dma.wait()
            weight = weight_vm[...].astype(jnp.float32)

        @pl.when(step >= 2)
        def _wait_prior_logits_output():
            pltpu.make_async_copy(
                logits_vm.at[parity], logits_vm.at[parity], logits_sems.at[parity]
            ).wait()

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        # The caller's residual is un-expanded: every branch slice of the
        # expanded plane carries this same staged tile. Normalize one 32-row
        # strip once and write each branch's affine-scaled value into the
        # full-width dot operand, which no DMA ever targets.
        def normalize_rows(index, carry):
            offset = index * 32
            values = input_vm[parity, pl.ds(offset, 32), pl.ds(0, hidden)].astype(
                jnp.float32
            )
            variance = jnp.mean(values * values, axis=-1, keepdims=True)
            normalized = values * jax.lax.rsqrt(variance + eps)
            for branch in range(branches):
                start = branch * hidden
                scaled = normalized * weight[None, start : start + hidden]
                norm_vm[pl.ds(offset, 32), pl.ds(start, hidden)] = (
                    normalized + scaled
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
        norm = norm_vm[...]

        if first:
            # Both packed ranges land before the single projection read.
            down_dma.wait()
            inject_dma.wait()
            if gap_rows:
                # The boundary rows belong to no DMA range; define them so the
                # packed dot never reads undefined vector memory.
                packed_vm[pl.ds(lowrank, gap_rows)] = jnp.zeros(
                    (gap_rows, branches * hidden), jnp.bfloat16
                )

        # One fullK FP32 dot serves both projections: the down columns and the
        # injection columns round through BF16 exactly where the separate
        # linears over the stored normalized value would. Injection logits
        # round at the same BF16 boundary; combine applies FP32 sigmoid.
        projected = jax.lax.dot_general(
            norm,
            packed_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        down = projected[:, :lowrank].astype(jnp.bfloat16)
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        logits = projected[:, inject_row : inject_row + branches].astype(jnp.bfloat16)
        logits_vm[parity] = logits
        pltpu.make_async_copy(
            logits_vm.at[parity],
            logits_hbm.at[pl.ds(step * token_tile, token_tile)],
            logits_sems.at[parity],
        ).start()

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before NVIDIA's FP32 sigmoid/mix.
        logits_up = project_up(gate, up_vm[...])
        weighted = jax.nn.sigmoid(logits_up) * norm.astype(jnp.float32)
        mixed = weighted[:, :hidden]
        for branch in range(1, branches):
            start = branch * hidden
            mixed = mixed + weighted[:, start : start + hidden]

        mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            mixed_vm.at[parity],
            mixed_hbm.at[pl.ds(step * token_tile, token_tile)],
            mixed_sems.at[parity],
        ).start()
        return weight

    # Peel the first pair so weight waits/cast are static first-tile work.
    tiles = tokens // token_tile
    run_mix_tiles(one_step, tiles)
    # A one-tile call never started parity1; do not wait its semaphore.
    for parity in range(min(tiles, 2)):
        pltpu.make_async_copy(
            logits_vm.at[parity], logits_vm.at[parity], logits_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def full_mix_inject_broadcast(
    x,
    weight,
    down,
    up,
    inject,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    hardware=None,
):
    """Mix an un-expanded broadcast residual while projecting injection gates.

    The caller's residual is the ``[tokens, hidden]`` value whose expanded
    hyper input would repeat once per branch, so every branch slice of the
    expanded plane carries the same words. This kernel stages that un-expanded
    tile once, normalizes one strip per 32 rows exactly as the expanded input
    would be normalized (the variance covers the same values), writes each
    branch's affine-scaled normalized value into the full-width dot operand,
    and then follows the inject-projecting mix exactly. The staged values are
    the same words the expanded plane would carry; the qualification oracle is
    the finite/L2/cosine tolerance against the reference over the expanded
    value, because the reduced variance sum order can legitimately change
    output words relative to ``full_mix_inject``.

    Args:
        x: BF16 [tokens, hidden] un-expanded residual (one branch slice).
        weight: BF16 [branches * hidden] per-branch norm affine weight.
        down: BF16 [lowrank, branches * hidden] down projection.
        up: BF16 [branches * hidden, lowrank] up projection.
        inject: BF16 [branches, branches * hidden] injection projection.
        branches: HC branch count derived from the tensors.
        hidden: Hidden size per branch.
        eps: RMS norm epsilon.
        token_tile: Token rows per step, multiple of 128.
        hardware: Optional immutable layout facts.

    Returns:
        (mixed [tokens, hidden] BF16, logits [tokens, branches] BF16).

    Raises:
        ValueError: Geometry, operand shapes, dtypes, or resources unsupported.
    """
    validate_mix_geometry(
        branches, hidden, token_tile, "HC broadcast inject mix", positive_tile=False
    )
    tokens, width = x.shape
    lowrank = down.shape[0]
    if (
        invalid_mix_rows(tokens, lowrank, token_tile, width, hidden)
        or invalid_mix_weights(weight, down, up, branches * hidden, lowrank)
        or inject.shape != (branches, branches * hidden)
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up, inject))
    ):
        raise ValueError("HC broadcast inject mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_inject_broadcast_mix(
        branches, hidden, lowrank, token_tile, hardware
    ):
        raise ValueError(
            "HC broadcast inject mix requires a supported layout and VMEM budget"
        )
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)
    return pl.pallas_call(
        functools.partial(
            _broadcast_kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
            lowrank=lowrank,
            inject_row=inject_row,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * 5,
            out_specs=(hbm,) * 2,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((token_tile, branches * hidden), jnp.bfloat16),
                pltpu.VMEM((branches * hidden,), jnp.bfloat16),
                pltpu.VMEM((packed_rows, branches * hidden), jnp.bfloat16),
                pltpu.VMEM((branches * hidden, lowrank), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, branches), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # Identical arithmetic to the inject mix over the expanded residual:
        # the same two GEMMs at two FLOPs/MAC, the same normalization,
        # SiLU and branch mixing elementwise work, and the same weight and
        # output traffic. Only the input read shrinks to the un-expanded tile
        # the caller actually owns. These unpadded logical bytes do not claim
        # exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=2 * tokens * branches * hidden * (lowrank + packed_rows)
            + tokens * (7 * branches * hidden + branches + 2 * lowrank + branches),
            transcendentals=tokens * (branches + lowrank + branches * hidden),
            bytes_accessed=sum(
                a.size * a.dtype.itemsize for a in (x, weight, down, up, inject)
            )
            + tokens * (hidden + branches) * x.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_inject_bcast_m{token_tile}",
    )(x, weight, down, up, inject)

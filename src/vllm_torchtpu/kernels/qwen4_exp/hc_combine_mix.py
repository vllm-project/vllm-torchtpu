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
"""Prefill HC boundary kernels: one owner's combine fused with the next mix.

The unfused boundary round-trips the full-width hyper input through HBM twice:
``combine`` writes ``residual + block * gates`` and the following owner's mix
immediately reads it back only to normalize and project it. This kernel stages
the residual tile once, forms the combined value in vector memory, streams that
one combined tile to HBM (the following owner's ``combine`` still needs it as
its residual operand), and consumes the same vector-memory value for the
per-branch normalization and the three projections of the inject-projecting
mix. The injection gates are formed in kernel from the prior owner's BF16
logits, so the eager sigmoid chain between the two kernels disappears too.

The down and inject projections contract the same normalized BF16 value over
the full width, so their live HBM weights are DMA-staged into two
nonoverlapping row ranges of one resident plane and consumed by a single
fullK FP32 dot: the inject rows start at the first hardware row boundary at
or after the down rows, the boundary rows in between are written to zero
before the first read, and each projection's columns keep their own BF16
store boundary. Store boundaries match the unfused chain exactly: the combined
value and the normalized input round through BF16, the injection logits round
through BF16 where a standalone linear over the stored combined value would,
and the gate sigmoid runs in FP32 over those stored words.

The terminal variant fuses the final owner's combine with a terminal mix that
has no combine and no inject projection of its own: the combined value has no
downstream reader, so it is formed in place over the staged residual tile (no
HBM stream, no combined ring) and that tile then carries the normalized value
for the down and up projections and the strip-scheduled branch reduction.

The broadcast variant serves the chain's first boundary, whose prior owner's
residual arrives un-expanded: every branch slice of the expanded hyper input
would repeat the same [tokens, hidden] value, so the kernel stages that value
once, forms each branch's combined value from it exactly as the expanded
residual would, and moves the normalized dot operand to a single full-width
plane that no DMA targets and no later tile reads. No expanded plane is ever
materialized. The staged values are the same words the expanded plane would
carry; the qualification oracle is the finite/L2/cosine tolerance against the
reference over the expanded value, because the reduced variance sum order can
legitimately change output words relative to the expanded-input kernel.
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
    supports_combine_broadcast_mix,
    supports_combine_mix,
    supports_terminal_mix,
)


def _kernel(
    residual_hbm,
    block_hbm,
    inlogits_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    inject_hbm,
    combined_hbm,
    mixed_hbm,
    outlogits_hbm,
    residual_vm,
    block_vm,
    inlogits_vm,
    weight_vm,
    packed_vm,
    up_vm,
    combined_vm,
    mixed_vm,
    outlogits_vm,
    residual_sems,
    block_sems,
    inlogits_sems,
    weight_sem,
    down_sem,
    up_sem,
    inject_sem,
    combined_sem,
    mixed_sems,
    outlogits_sems,
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
        residual_hbm.at[pl.ds(0, token_tile)], residual_vm.at[0], residual_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        block_hbm.at[pl.ds(0, token_tile)], block_vm.at[0], block_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        inlogits_hbm.at[pl.ds(0, token_tile)],
        inlogits_vm.at[0],
        inlogits_sems.at[0],
    ).start()

    def one_step(step, parity, scale):
        first = scale is None
        # The previous step started these input copies. Wait at first read.
        pltpu.make_async_copy(
            residual_vm.at[parity], residual_vm.at[parity], residual_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            block_vm.at[parity], block_vm.at[parity], block_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            inlogits_vm.at[parity], inlogits_vm.at[parity], inlogits_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_inputs():
            pltpu.make_async_copy(
                residual_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                residual_vm.at[1 - parity],
                residual_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                block_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                block_vm.at[1 - parity],
                block_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                inlogits_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                inlogits_vm.at[1 - parity],
                inlogits_sems.at[1 - parity],
            ).start()

        if first:
            weight_dma.wait()
            # The affine scale (1 + weight) is formed once per call, after the
            # live weight DMA completes: the following owner's public sequence
            # multiplies each normalized value by this FP32 scale, so the strip
            # loop needs no per-cell multiply-add pair at the norm store.
            scale = 1.0 + weight_vm[...].astype(jnp.float32)

        # The single combined plane is reused every tile; its previous output
        # copy drains while this tile normalizes and runs its projections.
        @pl.when(step >= 1)
        def _wait_prior_combined_output():
            pltpu.make_async_copy(combined_vm, combined_vm, combined_sem).wait()

        # Combine: FP32 gates over the prior owner's BF16 logits, FP32
        # residual-plus-block arithmetic, one BF16 store boundary.
        gates = 2.0 * jax.nn.sigmoid(inlogits_vm[parity].astype(jnp.float32) / branches)
        block = block_vm[parity].astype(jnp.float32)
        for branch in range(branches):
            start = branch * hidden
            combined_vm[:, pl.ds(start, hidden)] = (
                residual_vm[parity, pl.ds(0, token_tile), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                + block * gates[:, branch : branch + 1]
            ).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            combined_vm,
            combined_hbm.at[pl.ds(step * token_tile, token_tile)],
            combined_sem,
        ).start()

        @pl.when(step >= 2)
        def _wait_prior_logits_output():
            pltpu.make_async_copy(
                outlogits_vm.at[parity],
                outlogits_vm.at[parity],
                outlogits_sems.at[parity],
            ).wait()

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        # Normalize one 32-row branch strip of the combined value back into
        # the staged residual tile, whose raw rows have no reader left. The
        # normalized value reads the same BF16 words the HBM copy carries.
        def normalize_rows(index, carry):
            offset = index * 32
            for branch in range(branches):
                start = branch * hidden
                values = combined_vm[pl.ds(offset, 32), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                variance = jnp.mean(values * values, axis=-1, keepdims=True)
                normalized = values * jax.lax.rsqrt(variance + eps)
                residual_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)] = (
                    normalized * scale[None, start : start + hidden]
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
        norm = residual_vm[parity]

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
        # linears over the stored normalized value would.
        projected = jax.lax.dot_general(
            norm,
            packed_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        down = projected[:, :lowrank].astype(jnp.bfloat16)
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        # Injection logits round at the same BF16 boundary as a linear over
        # the stored combined value; the next combine applies FP32 sigmoid.
        logits = projected[:, inject_row : inject_row + branches].astype(jnp.bfloat16)
        outlogits_vm[parity] = logits
        pltpu.make_async_copy(
            outlogits_vm.at[parity],
            outlogits_hbm.at[pl.ds(step * token_tile, token_tile)],
            outlogits_sems.at[parity],
        ).start()

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before the FP32 sigmoid/mix. The up
        # projection and the branch reduction run one branch strip at a time:
        # each strip's dot output, sigmoid-weighted product, and the branch
        # accumulator hold [tile, hidden] planes reused strip-to-strip, so no
        # full-width FP32 gate plane is ever materialized. Every output word
        # matches the single wide projection (each element rounds once from
        # the same FP32 accumulator, and the branch additions keep the same
        # left-to-right order).
        logits_up = project_up(gate, up_vm[pl.ds(0, hidden), ...])
        mixed = jax.nn.sigmoid(logits_up) * residual_vm[
            parity, pl.ds(0, token_tile), pl.ds(0, hidden)
        ].astype(jnp.float32)
        for branch in range(1, branches):
            start = branch * hidden
            logits_up = project_up(gate, up_vm[pl.ds(start, hidden), ...])
            weighted = jax.nn.sigmoid(logits_up) * residual_vm[
                parity, pl.ds(0, token_tile), pl.ds(start, hidden)
            ].astype(jnp.float32)
            mixed = mixed + weighted

        mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            mixed_vm.at[parity],
            mixed_hbm.at[pl.ds(step * token_tile, token_tile)],
            mixed_sems.at[parity],
        ).start()
        return scale

    # Peel the first pair so the weight wait and the one-time affine-scale
    # formation are static first-tile work.
    tiles = tokens // token_tile
    run_mix_tiles(one_step, tiles)
    # A one-tile call never started parity1; do not wait its semaphores.
    for parity in range(min(tiles, 2)):
        pltpu.make_async_copy(
            outlogits_vm.at[parity], outlogits_vm.at[parity], outlogits_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()
    pltpu.make_async_copy(combined_vm, combined_vm, combined_sem).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def combine_mix_inject(
    residual,
    block,
    in_logits,
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
    """Fuse one owner's combine with the following owner's inject mix.

    Args:
        residual: BF16 [tokens, branches * hidden] prior hyper input.
        block: BF16 [tokens, hidden] block output of the prior owner.
        in_logits: BF16 [tokens, branches] prior owner's injection logits.
        weight: BF16 [branches * hidden] following owner's norm affine.
        down: BF16 [lowrank, branches * hidden] following down projection.
        up: BF16 [branches * hidden, lowrank] following up projection.
        inject: BF16 [branches, branches * hidden] following injection weight.
        branches: HC branch count derived from the tensors.
        hidden: Hidden size per branch.
        eps: RMS norm epsilon of the following owner.
        token_tile: Token rows per step, multiple of 128.
        hardware: Optional immutable layout facts.

    Returns:
        (combined [tokens, branches * hidden] BF16,
         mixed [tokens, hidden] BF16,
         logits [tokens, branches] BF16).

    Raises:
        ValueError: Geometry, operand shapes, dtypes, or resources unsupported.
    """
    validate_mix_geometry(branches, hidden, token_tile, "HC combine mix")
    tokens, width = residual.shape
    lowrank = down.shape[0]
    if (
        invalid_mix_rows(tokens, lowrank, token_tile, width, branches * hidden)
        or block.shape != (tokens, hidden)
        or in_logits.shape != (tokens, branches)
        or invalid_mix_weights(weight, down, up, width, lowrank)
        or inject.shape != (branches, width)
        or any(
            a.dtype != jnp.bfloat16
            for a in (residual, block, in_logits, weight, down, up, inject)
        )
    ):
        raise ValueError("HC combine mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_combine_mix(branches, hidden, lowrank, token_tile, hardware):
        raise ValueError("HC combine mix requires a supported layout and VMEM budget")
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
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
            in_specs=(hbm,) * 7,
            out_specs=(hbm,) * 3,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.VMEM((width,), jnp.bfloat16),
                pltpu.VMEM((packed_rows, width), jnp.bfloat16),
                pltpu.VMEM((width, lowrank), jnp.bfloat16),
                pltpu.VMEM((token_tile, width), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, branches), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # The combine adds one FP32 multiply-add per combined element and one
        # sigmoid per token/branch over the mix's two GEMMs and projections.
        # Reads cover every operand once; writes cover the combined residual,
        # the mixed block input, and the next logits once per token. These
        # unpadded logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * width * lowrank
            + 2 * tokens * width * branches
            + 2 * tokens * width
            + tokens * (7 * width + 2 * branches + 2 * lowrank),
            transcendentals=tokens * (2 * branches + lowrank + width),
            bytes_accessed=sum(
                a.size * a.dtype.itemsize
                for a in (residual, block, in_logits, weight, down, up, inject)
            )
            + tokens * (width + hidden + branches) * residual.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_combine_mix_m{token_tile}",
    )(residual, block, in_logits, weight, down, up, inject)


def _broadcast_kernel(
    residual_hbm,
    block_hbm,
    inlogits_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    inject_hbm,
    combined_hbm,
    mixed_hbm,
    outlogits_hbm,
    input_vm,
    norm_vm,
    block_vm,
    inlogits_vm,
    weight_vm,
    packed_vm,
    up_vm,
    combined_vm,
    mixed_vm,
    outlogits_vm,
    input_sems,
    block_sems,
    inlogits_sems,
    weight_sem,
    down_sem,
    up_sem,
    inject_sem,
    combined_sem,
    mixed_sems,
    outlogits_sems,
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
        residual_hbm.at[pl.ds(0, token_tile)], input_vm.at[0], input_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        block_hbm.at[pl.ds(0, token_tile)], block_vm.at[0], block_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        inlogits_hbm.at[pl.ds(0, token_tile)],
        inlogits_vm.at[0],
        inlogits_sems.at[0],
    ).start()

    def one_step(step, parity, weight):
        first = weight is None
        # The previous step started these input copies. Wait at first read.
        pltpu.make_async_copy(
            input_vm.at[parity], input_vm.at[parity], input_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            block_vm.at[parity], block_vm.at[parity], block_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            inlogits_vm.at[parity], inlogits_vm.at[parity], inlogits_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_inputs():
            pltpu.make_async_copy(
                residual_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                input_vm.at[1 - parity],
                input_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                block_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                block_vm.at[1 - parity],
                block_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                inlogits_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                inlogits_vm.at[1 - parity],
                inlogits_sems.at[1 - parity],
            ).start()

        if first:
            weight_dma.wait()
            weight = weight_vm[...].astype(jnp.float32)

        # The single combined plane is reused every tile; its previous output
        # copy drains while this tile normalizes and runs its projections.
        @pl.when(step >= 1)
        def _wait_prior_combined_output():
            pltpu.make_async_copy(combined_vm, combined_vm, combined_sem).wait()

        # Combine: FP32 gates over the prior owner's BF16 logits, FP32
        # residual-plus-block arithmetic, one BF16 store boundary. The prior
        # owner's residual arrives un-expanded — every branch slice of the
        # expanded plane carries this same staged tile — so each branch reads
        # the one [token_tile, hidden] ring directly.
        gates = 2.0 * jax.nn.sigmoid(inlogits_vm[parity].astype(jnp.float32) / branches)
        block = block_vm[parity].astype(jnp.float32)
        for branch in range(branches):
            start = branch * hidden
            combined_vm[:, pl.ds(start, hidden)] = (
                input_vm[parity, pl.ds(0, token_tile), pl.ds(0, hidden)].astype(
                    jnp.float32
                )
                + block * gates[:, branch : branch + 1]
            ).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            combined_vm,
            combined_hbm.at[pl.ds(step * token_tile, token_tile)],
            combined_sem,
        ).start()

        @pl.when(step >= 2)
        def _wait_prior_logits_output():
            pltpu.make_async_copy(
                outlogits_vm.at[parity],
                outlogits_vm.at[parity],
                outlogits_sems.at[parity],
            ).wait()

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        # Normalize one 32-row branch strip of the combined value into the
        # full-width dot operand, which no DMA ever targets and which no later
        # tile reads, so a single plane serves every tile. The normalized
        # value reads the same BF16 words the HBM copy carries.
        def normalize_rows(index, carry):
            offset = index * 32
            for branch in range(branches):
                start = branch * hidden
                values = combined_vm[pl.ds(offset, 32), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                variance = jnp.mean(values * values, axis=-1, keepdims=True)
                normalized = values * jax.lax.rsqrt(variance + eps)
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
        # linears over the stored normalized value would.
        projected = jax.lax.dot_general(
            norm,
            packed_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        down = projected[:, :lowrank].astype(jnp.bfloat16)
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        # Injection logits round at the same BF16 boundary as a linear over
        # the stored combined value; the next combine applies FP32 sigmoid.
        logits = projected[:, inject_row : inject_row + branches].astype(jnp.bfloat16)
        outlogits_vm[parity] = logits
        pltpu.make_async_copy(
            outlogits_vm.at[parity],
            outlogits_hbm.at[pl.ds(step * token_tile, token_tile)],
            outlogits_sems.at[parity],
        ).start()

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before the FP32 sigmoid/mix. The up
        # projection and the branch reduction run one branch strip at a time:
        # each strip's dot output, sigmoid-weighted product, and the branch
        # accumulator hold [tile, hidden] planes reused strip-to-strip, so no
        # full-width FP32 gate plane is ever materialized. Every output word
        # matches the single wide projection (each element rounds once from
        # the same FP32 accumulator, and the branch additions keep the same
        # left-to-right order).
        logits_up = project_up(gate, up_vm[pl.ds(0, hidden), ...])
        mixed = jax.nn.sigmoid(logits_up) * norm_vm[
            pl.ds(0, token_tile), pl.ds(0, hidden)
        ].astype(jnp.float32)
        for branch in range(1, branches):
            start = branch * hidden
            logits_up = project_up(gate, up_vm[pl.ds(start, hidden), ...])
            weighted = jax.nn.sigmoid(logits_up) * norm_vm[
                pl.ds(0, token_tile), pl.ds(start, hidden)
            ].astype(jnp.float32)
            mixed = mixed + weighted

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
    # A one-tile call never started parity1; do not wait its semaphores.
    for parity in range(min(tiles, 2)):
        pltpu.make_async_copy(
            outlogits_vm.at[parity], outlogits_vm.at[parity], outlogits_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()
    pltpu.make_async_copy(combined_vm, combined_vm, combined_sem).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def combine_mix_inject_broadcast(
    residual,
    block,
    in_logits,
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
    """Fuse one owner's broadcast-residual combine with the following mix.

    The prior owner's residual arrives un-expanded — every branch slice of the
    expanded hyper input would repeat the same ``[tokens, hidden]`` value — so
    this kernel stages that value once instead of the full-width plane, forms
    each branch's combined value from it exactly as the expanded residual
    would (the same FP32 add of the same words, one BF16 store boundary), and
    follows the boundary kernel from there: the combined tile streams to HBM
    for the following owner's ``combine`` while the same vector-memory value
    feeds the per-branch normalization and the three projections of the
    inject-projecting mix. The staged values are the same words the expanded
    plane would carry; the qualification oracle is the finite/L2/cosine
    tolerance against the reference over the expanded value, because the
    reduced variance sum order can legitimately change output words relative
    to ``combine_mix_inject``.

    Args:
        residual: BF16 [tokens, hidden] prior owner's un-expanded residual.
        block: BF16 [tokens, hidden] block output of the prior owner.
        in_logits: BF16 [tokens, branches] prior owner's injection logits.
        weight: BF16 [branches * hidden] following owner's norm affine.
        down: BF16 [lowrank, branches * hidden] following down projection.
        up: BF16 [branches * hidden, lowrank] following up projection.
        inject: BF16 [branches, branches * hidden] following injection weight.
        branches: HC branch count derived from the tensors.
        hidden: Hidden size per branch.
        eps: RMS norm epsilon of the following owner.
        token_tile: Token rows per step, multiple of 128.
        hardware: Optional immutable layout facts.

    Returns:
        (combined [tokens, branches * hidden] BF16,
         mixed [tokens, hidden] BF16,
         logits [tokens, branches] BF16).

    Raises:
        ValueError: Geometry, operand shapes, dtypes, or resources unsupported.
    """
    validate_mix_geometry(
        branches, hidden, token_tile, "HC broadcast combine mix", positive_tile=False
    )
    tokens, residual_width = residual.shape
    lowrank = down.shape[0]
    if (
        invalid_mix_rows(tokens, lowrank, token_tile, residual_width, hidden)
        or block.shape != (tokens, hidden)
        or in_logits.shape != (tokens, branches)
        or invalid_mix_weights(weight, down, up, branches * hidden, lowrank)
        or inject.shape != (branches, branches * hidden)
        or any(
            a.dtype != jnp.bfloat16
            for a in (residual, block, in_logits, weight, down, up, inject)
        )
    ):
        raise ValueError("HC broadcast combine mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_combine_broadcast_mix(
        branches, hidden, lowrank, token_tile, hardware
    ):
        raise ValueError(
            "HC broadcast combine mix requires a supported layout and VMEM budget"
        )
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
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
            in_specs=(hbm,) * 7,
            out_specs=(hbm,) * 3,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((token_tile, branches * hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.VMEM((branches * hidden,), jnp.bfloat16),
                pltpu.VMEM((packed_rows, branches * hidden), jnp.bfloat16),
                pltpu.VMEM((branches * hidden, lowrank), jnp.bfloat16),
                pltpu.VMEM((token_tile, branches * hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, branches * hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, branches), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # Identical arithmetic to the boundary kernel over the expanded
        # residual: the same combine multiply-add, the same two GEMMs, the
        # same sigmoid and strip-scheduled reduction. Only the residual read
        # shrinks to the un-expanded tile the caller actually owns. These
        # unpadded logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * branches * hidden * lowrank
            + 2 * tokens * branches * hidden * branches
            + 2 * tokens * branches * hidden
            + tokens * (7 * branches * hidden + 2 * branches + 2 * lowrank),
            transcendentals=tokens * (2 * branches + lowrank + branches * hidden),
            bytes_accessed=sum(
                a.size * a.dtype.itemsize
                for a in (residual, block, in_logits, weight, down, up, inject)
            )
            + tokens
            * (branches * hidden + hidden + branches)
            * residual.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_combine_bcast_m{token_tile}",
    )(residual, block, in_logits, weight, down, up, inject)


def _terminal_kernel(
    residual_hbm,
    block_hbm,
    inlogits_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    mixed_hbm,
    residual_vm,
    block_vm,
    inlogits_vm,
    weight_vm,
    down_vm,
    up_vm,
    mixed_vm,
    residual_sems,
    block_sems,
    inlogits_sems,
    weight_sem,
    down_sem,
    up_sem,
    mixed_sems,
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
        residual_hbm.at[pl.ds(0, token_tile)], residual_vm.at[0], residual_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        block_hbm.at[pl.ds(0, token_tile)], block_vm.at[0], block_sems.at[0]
    ).start()
    pltpu.make_async_copy(
        inlogits_hbm.at[pl.ds(0, token_tile)],
        inlogits_vm.at[0],
        inlogits_sems.at[0],
    ).start()

    def one_step(step, parity, weight):
        first = weight is None
        # The previous step started these input copies. Wait at first read.
        pltpu.make_async_copy(
            residual_vm.at[parity], residual_vm.at[parity], residual_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            block_vm.at[parity], block_vm.at[parity], block_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            inlogits_vm.at[parity], inlogits_vm.at[parity], inlogits_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_inputs():
            pltpu.make_async_copy(
                residual_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                residual_vm.at[1 - parity],
                residual_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                block_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                block_vm.at[1 - parity],
                block_sems.at[1 - parity],
            ).start()
            pltpu.make_async_copy(
                inlogits_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                inlogits_vm.at[1 - parity],
                inlogits_sems.at[1 - parity],
            ).start()

        if first:
            weight_dma.wait()
            weight = weight_vm[...].astype(jnp.float32)

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            # This step rewrites the parity buffer whose output copy the step
            # two tiles back started; that drain must finish first.
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        # Combine: FP32 gates over the prior owner's BF16 logits, FP32
        # residual-plus-block arithmetic, one BF16 store boundary. The staged
        # residual rows have no later reader and no HBM combined stream exists
        # on the terminal path, so the combined value replaces them in place.
        gates = 2.0 * jax.nn.sigmoid(inlogits_vm[parity].astype(jnp.float32) / branches)
        block = block_vm[parity].astype(jnp.float32)
        for branch in range(branches):
            start = branch * hidden
            residual_vm[parity, pl.ds(0, token_tile), pl.ds(start, hidden)] = (
                residual_vm[parity, pl.ds(0, token_tile), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                + block * gates[:, branch : branch + 1]
            ).astype(jnp.bfloat16)

        # Normalize one 32-row branch strip of the combined value in place:
        # the strip's rows have no reader after its variance is formed, and
        # the normalized value reads the same BF16 words an HBM round trip
        # of the combined tensor would carry.
        def normalize_rows(index, carry):
            offset = index * 32
            for branch in range(branches):
                start = branch * hidden
                values = residual_vm[
                    parity, pl.ds(offset, 32), pl.ds(start, hidden)
                ].astype(jnp.float32)
                variance = jnp.mean(values * values, axis=-1, keepdims=True)
                normalized = values * jax.lax.rsqrt(variance + eps)
                scaled = normalized * weight[None, start : start + hidden]
                residual_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)] = (
                    normalized + scaled
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
        norm = residual_vm[parity]

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

        # Match the BF16 up Linear output before the FP32 sigmoid/mix. The up
        # projection and the branch reduction run one branch strip at a time:
        # each strip's dot output, sigmoid-weighted product, and the branch
        # accumulator hold [tile, hidden] planes reused strip-to-strip, so no
        # full-width FP32 gate plane is ever materialized. Every output word
        # matches the single wide projection (each element rounds once from
        # the same FP32 accumulator, and the branch additions keep the same
        # left-to-right order).
        logits_up = project_up(gate, up_vm[pl.ds(0, hidden), ...])
        mixed = jax.nn.sigmoid(logits_up) * residual_vm[
            parity, pl.ds(0, token_tile), pl.ds(0, hidden)
        ].astype(jnp.float32)
        for branch in range(1, branches):
            start = branch * hidden
            logits_up = project_up(gate, up_vm[pl.ds(start, hidden), ...])
            weighted = jax.nn.sigmoid(logits_up) * residual_vm[
                parity, pl.ds(0, token_tile), pl.ds(start, hidden)
            ].astype(jnp.float32)
            mixed = mixed + weighted

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
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "token_tile", "hardware"),
)
def combine_mix_terminal(
    residual,
    block,
    in_logits,
    weight,
    down,
    up,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    hardware=None,
):
    """Fuse the final owner's combine with the terminal mix.

    Args:
        residual: BF16 [tokens, branches * hidden] prior hyper input.
        block: BF16 [tokens, hidden] block output of the prior owner.
        in_logits: BF16 [tokens, branches] prior owner's injection logits.
        weight: BF16 [branches * hidden] terminal owner's norm affine.
        down: BF16 [lowrank, branches * hidden] terminal down projection.
        up: BF16 [branches * hidden, lowrank] terminal up projection.
        branches: HC branch count derived from the tensors.
        hidden: Hidden size per branch.
        eps: RMS norm epsilon of the terminal owner.
        token_tile: Token rows per step, multiple of 128.
        hardware: Optional immutable layout facts.

    Returns:
        mixed [tokens, hidden] BF16.

    Raises:
        ValueError: Geometry, operand shapes, dtypes, or resources unsupported.
    """
    validate_mix_geometry(branches, hidden, token_tile, "HC terminal mix")
    tokens, width = residual.shape
    lowrank = down.shape[0]
    if (
        invalid_mix_rows(tokens, lowrank, token_tile, width, branches * hidden)
        or block.shape != (tokens, hidden)
        or in_logits.shape != (tokens, branches)
        or invalid_mix_weights(weight, down, up, width, lowrank)
        or any(
            a.dtype != jnp.bfloat16
            for a in (residual, block, in_logits, weight, down, up)
        )
    ):
        raise ValueError("HC terminal mix requires full BF16 token tiles")
    if hardware is None:
        hardware = HcHardware.from_tpu_info(pltpu.get_tpu_info())
    if not supports_terminal_mix(branches, hidden, lowrank, token_tile, hardware):
        raise ValueError("HC terminal mix requires a supported layout and VMEM budget")
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    (mixed,) = pl.pallas_call(
        functools.partial(
            _terminal_kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * 6,
            out_specs=(hbm,) * 1,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),
                pltpu.VMEM((width,), jnp.bfloat16),
                pltpu.VMEM((lowrank, width), jnp.bfloat16),
                pltpu.VMEM((width, lowrank), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=hardware.vmem_bytes),
        # The combine adds one FP32 multiply-add per combined element and one
        # sigmoid per token/branch over the terminal mix's two GEMMs. Reads
        # cover every operand once; writes cover the mixed output once per
        # token — the combined value never reaches HBM. These unpadded
        # logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * width * lowrank
            + 2 * tokens * width
            + tokens * (7 * width + 2 * branches + 2 * lowrank),
            transcendentals=tokens * (branches + lowrank + width),
            bytes_accessed=sum(
                a.size * a.dtype.itemsize
                for a in (residual, block, in_logits, weight, down, up)
            )
            + tokens * hidden * residual.dtype.itemsize,
        ),
        name=f"qwen4exp_hc_combine_mix_terminal_m{token_tile}",
    )(residual, block, in_logits, weight, down, up)
    # The kernel has one output plane; the public contract is the bare tensor.
    return mixed

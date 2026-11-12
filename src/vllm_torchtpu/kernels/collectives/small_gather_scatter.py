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
"""Small Pallas reduce-scatter for the data-parallel MoE combine.

Every core pushes the partial sums for each rank's rows to that rank's
mesh core. Each destination sums its arrivals in float32 and rounds once
to the input dtype. On dual-core TPU chips, an opt-in two-stage mode
first exchanges and reduces partner-rank blocks between the two cores of
each physical chip before pushing one reduced block per remote chip over
the inter-chip interconnect, halving cross-chip traffic.

Input per core: [cores * rows, hidden], with blocks in rank order.
Output per core: [rows, hidden], the sum for that core's rank.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.small_all_reduce import (
    _barrier,
    _push,
    _split,
    _wait_bytes_of,
)

LANE = 128  # the lane tile: a block's width must sit on it
# Distinct barrier groups from the all-reduce kernel's (5).
SCATTER_COLLECTIVE_ID = 7
# Without `vmem_limit_bytes` on `pltpu.CompilerParams`, XLA caps Mosaic scoped
# VMEM at 32 MiB even on 64 MiB chips. Keep a 25% margin under that ceiling for
# float32 accumulation temporaries and Mosaic register allocation.
_DEFAULT_SCOPED_VMEM_LIMIT_BYTES = 32 * 1024 * 1024
_HIERARCHICAL_VMEM_BUDGET_FRACTION = 0.75


def _check(hidden):
    if hidden % LANE:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANE} lanes")


def _supports_hierarchical_core_count(num_cores: int) -> bool:
    num_chips = num_cores // 2
    return num_cores >= 4 and num_cores % 2 == 0 and (num_chips & (num_chips - 1)) == 0


def _hierarchical_fits_vmem(num_cores: int, rows: int, hidden: int, dtype) -> bool:
    """Whether the two-stage buffers fit within XLA's default scoped VMEM cap."""
    try:
        vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # noqa: BLE001 - CPU interpret mode has no TPU info
        return True
    scoped_cap_bytes = min(vmem_bytes, _DEFAULT_SCOPED_VMEM_LIMIT_BYTES)
    itemsize = jnp.dtype(dtype).itemsize
    # Input (n), output (1), stage1_send (n/2), stage1_recv reused in place for
    # stage2 local partials (n/2), and stage2_recv (n/2 - 1) total (5*n)//2
    # blocks.
    total_blocks = (5 * num_cores) // 2
    needed_bytes = total_blocks * rows * hidden * itemsize
    return needed_bytes <= int(scoped_cap_bytes * _HIERARCHICAL_VMEM_BUDGET_FRACTION)


def _call(kernel, x, out_rows, hidden, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, hidden), x.dtype),
        in_specs=rank_specs + [pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)


def _invoke_reduce_scatter(
    kernel,
    x,
    *,
    rows: int,
    hidden: int,
    rank,
    my_rank,
    core_of_rank,
    include_my_rank: bool,
    scratch,
    collective_id: int,
    base_name: str,
):
    if core_of_rank is None:
        return _call(kernel, x, rows, hidden, rank, scratch, collective_id, base_name)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    rank_specs = [smem] if rank is not None else []
    perm_specs = [smem, smem] if include_my_rank else [smem]
    perm_args = (
        (my_rank.astype(jnp.int32), core_of_rank.astype(jnp.int32))
        if include_my_rank
        else (core_of_rank.astype(jnp.int32),)
    )
    args = (() if rank is None else (rank.astype(jnp.int32),)) + perm_args + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        in_specs=rank_specs + perm_specs + [pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"{base_name}_permuted",
    )(*args)


def small_reduce_scatter(
    x,
    *,
    num_cores,
    axis="d",
    rank=None,
    my_rank=None,
    core_of_rank=None,
    dict_ids=None,
    collective_id=SCATTER_COLLECTIVE_ID,
    hierarchical: bool = False,
):
    """[num_cores * rows, hidden] on each core (block r = its partial sums
    for RANK r's rows, the order the gather placed them in) -> [rows,
    hidden]: the float32 sum over the cores of this core's own rank's block,
    rounded once to x's type (or once per stage when `hierarchical=True`);
    inside a shard_map.

    The kernel addresses cores by mesh index while the blocks are in rank
    order. The mesh core order may differ from the rank order. `core_of_rank`
    [num_cores] int32 (SMEM) is the mesh core that serves each rank and
    `my_rank` [1] int32 (SMEM) this core's rank; both None means the
    identity (mesh order is rank order). Block r goes to core
    core_of_rank[r]; every core sums the blocks it receives for its rank.

    When `hierarchical=True` on a mesh whose adjacent device pairs
    `(2*c, 2*c + 1)` share dual-core chips across a power-of-two chip count
    (`>= 2`), Stage 1 exchanges partner blocks between on-chip mate cores and
    reduces in float32 to `x.dtype`, and Stage 2 pushes one reduced block per
    remote chip across the inter-chip links and reduces in float32 to
    `x.dtype`. Shapes whose two-stage buffers exceed the scoped VMEM budget
    fall back to the one-stage kernel.
    """
    total, hidden = x.shape
    _check(hidden)
    n = num_cores
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    rows = total // n
    if dict_ids is None:
        dict_ids = rank is not None
    if (my_rank is None) != (core_of_rank is None):
        raise ValueError("my_rank and core_of_rank come together")
    permuted = core_of_rank is not None

    if hierarchical and not _supports_hierarchical_core_count(n):
        raise ValueError(
            f"hierarchical reduce-scatter requires an even core count >= 4 "
            f"with a power-of-two chip count, got num_cores={n}"
        )

    if rows % 8:
        # DMA slices need whole row tiles. Pad each rank's block separately
        # so the padding does not move rows between destination ranks.
        padded_rows = pl.cdiv(rows, 8) * 8
        x = jnp.pad(
            x.reshape(n, rows, hidden), ((0, 0), (0, padded_rows - rows), (0, 0))
        )
        return small_reduce_scatter(
            x.reshape(n * padded_rows, hidden),
            num_cores=n,
            axis=axis,
            rank=rank,
            my_rank=my_rank,
            core_of_rank=core_of_rank,
            dict_ids=dict_ids,
            collective_id=collective_id,
            hierarchical=hierarchical,
        )[:rows]

    use_hierarchical = bool(hierarchical) and _hierarchical_fits_vmem(
        n, rows, hidden, x.dtype
    )

    if use_hierarchical:
        num_chips = n // 2

        def hierarchical_kernel(*refs):
            me, refs = _split(refs, rank is not None, axis)
            if permuted:
                core_ref, *refs = refs
                (
                    x_ref,
                    o_ref,
                    rank_of_core_smem,
                    stage1_send,
                    stage1_recv,
                    stage2_recv,
                    stage1_send_sem,
                    stage1_recv_sem,
                    stage2_send_sem,
                    stage2_recv_sem,
                ) = refs
                for candidate_rank in range(n):
                    rank_of_core_smem[core_ref[candidate_rank]] = jnp.int32(
                        candidate_rank
                    )
            else:
                rank_of_core_smem = None
                (
                    x_ref,
                    o_ref,
                    stage1_send,
                    stage1_recv,
                    stage2_recv,
                    stage1_send_sem,
                    stage1_recv_sem,
                    stage2_send_sem,
                    stage2_recv_sem,
                ) = refs

            mate_core = lax.bitwise_xor(me, jnp.int32(1))
            my_chip = lax.shift_right_logical(me, jnp.int32(1))
            my_slot_on_chip = lax.bitwise_and(me, jnp.int32(1))
            mate_slot_on_chip = jnp.int32(1) - my_slot_on_chip

            def rank_for_core(target_core):
                if not permuted:
                    return target_core
                assert rank_of_core_smem is not None
                return rank_of_core_smem[target_core]

            # Pack Stage 1 send buffer before the barrier so local VMEM copies
            # overlap with barrier arrival skew across ranks.
            for chip_idx in range(num_chips):
                mate_target_core = jnp.int32(2 * chip_idx) + mate_slot_on_chip
                mate_rank = rank_for_core(mate_target_core)
                mate_row_offset = pl.multiple_of(mate_rank * rows, 8)
                stage1_send[chip_idx] = x_ref[pl.ds(mate_row_offset, rows), :]

            _barrier(n, axis, dict_ids)
            _push(
                stage1_send,
                stage1_recv,
                stage1_send_sem,
                stage1_recv_sem,
                mate_core,
                axis,
                dict_ids,
            )

            _wait_bytes_of(stage1_recv, stage1_recv_sem)
            for chip_idx in range(num_chips):
                self_target_core = jnp.int32(2 * chip_idx) + my_slot_on_chip
                self_rank = rank_for_core(self_target_core)
                self_row_offset = pl.multiple_of(self_rank * rows, 8)
                chip_partial = x_ref[pl.ds(self_row_offset, rows), :].astype(
                    jnp.float32
                ) + stage1_recv[chip_idx].astype(jnp.float32)
                stage1_recv[chip_idx] = chip_partial.astype(x.dtype)

            for peer_idx in range(num_chips - 1):
                delta = peer_idx + 1
                peer_chip = lax.bitwise_xor(my_chip, jnp.int32(delta))
                peer_core = lax.bitwise_xor(me, jnp.int32(2 * delta))
                _push(
                    stage1_recv.at[peer_chip],
                    stage2_recv.at[peer_idx],
                    stage2_send_sem,
                    stage2_recv_sem,
                    peer_core,
                    axis,
                    dict_ids,
                )

            acc = stage1_recv[my_chip].astype(jnp.float32)
            _wait_bytes_of(stage2_recv, stage2_recv_sem)
            for peer_idx in range(num_chips - 1):
                acc = acc + stage2_recv[peer_idx].astype(jnp.float32)
            o_ref[...] = acc.astype(o_ref.dtype)
            _wait_bytes_of(stage1_send, stage1_send_sem)
            _wait_bytes_of(stage2_recv, stage2_send_sem)

        smem_scratch = [pltpu.SMEM((n,), jnp.int32)] if permuted else []
        hierarchical_scratch = smem_scratch + [
            pltpu.VMEM((num_chips, rows, hidden), x.dtype),
            pltpu.VMEM((num_chips, rows, hidden), x.dtype),
            pltpu.VMEM((num_chips - 1, rows, hidden), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ]
        return _invoke_reduce_scatter(
            hierarchical_kernel,
            x,
            rows=rows,
            hidden=hidden,
            rank=rank,
            my_rank=my_rank,
            core_of_rank=core_of_rank,
            include_my_rank=False,
            scratch=hierarchical_scratch,
            collective_id=collective_id,
            base_name=f"small_reduce_scatter_hier_r{rows}",
        )

    def kernel(*refs):
        me, refs = _split(refs, rank is not None, axis)
        if permuted:
            my_rank_ref, core_ref, *refs = refs
            mine = my_rank_ref[0]
        else:
            mine = me
        x_ref, o_ref, slices, send_sem, recv_sem = refs
        _barrier(n, axis, dict_ids)
        # block r (rank r's rows) into slot `me` of the slices of the core
        # that serves rank r
        for r in range(n):

            @pl.when(jnp.int32(r) != mine)
            def _(r=r):
                dest = core_ref[r] if permuted else jnp.int32(r)
                _push(
                    x_ref.at[pl.ds(r * rows, rows)],
                    slices.at[me],
                    send_sem,
                    recv_sem,
                    dest,
                    axis,
                    dict_ids,
                )

        # my own rank's block: a static row slice picked by a scalar compare
        for r in range(n):

            @pl.when(jnp.int32(r) == mine)
            def _(r=r):
                slices[pl.ds(me, 1)] = x_ref[pl.ds(r * rows, rows), :][None]

        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        acc = slices[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j].astype(jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)
        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return _invoke_reduce_scatter(
        kernel,
        x,
        rows=rows,
        hidden=hidden,
        rank=rank,
        my_rank=my_rank,
        core_of_rank=core_of_rank,
        include_my_rank=True,
        scratch=scratch,
        collective_id=collective_id,
        base_name=f"small_reduce_scatter_r{rows}",
    )


def reference_scatter(blocks, core):
    """[cores, cores * rows, hidden] -> core `core`'s output: the float32 sum
    over the cores of block `core`, rounded once."""
    rows = blocks.shape[1] // blocks.shape[0]
    mine = blocks[:, core * rows : (core + 1) * rows, :]
    return jnp.sum(mine.astype(jnp.float32), axis=0).astype(blocks.dtype)

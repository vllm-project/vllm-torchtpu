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
"""TPU_MOE_PREFILL_COMBINE_SCATTER: the prefill MoE combine of one combine chunk
and its reduce-scatter over the 16 cores of a 2x2x2 slice in one kernel.

Today the combine sums each token's expert rows on the TensorCore and writes the
chunk's [T, hidden] partial to HBM; XLA's reduce-scatter then runs on the
SparseCore, and the last chunk's waits with the TensorCore idle. Here the
combine computes the chunk one row block at a time (block m: the T / 16 rows
that mesh index m keeps) and each block enters the rotated halving-doubling of
rotated_collectives.reduce_scatter the moment it is summed:

  - blocks of the chip-mate's half (same chip offset, other core) go to the
    mate as soon as they are combined;
  - blocks of this core's half are added to the mate's partial as soon as both
    are there, and each continues through the three chip phases at once: sent
    on to the axis partner, or summed with the partner's contribution and
    carried to the next phase.

The blocks are combined by chip offset (bit 0: x, bit 1: y, bit 2: z) in
CHIP_ORDER, every offset before each offset whose axes are a subset of its own,
and the mate's half before this core's half at each offset. So every wait is
for a block its peer combined earlier, and the chip links carry data from the
second block on, beside the combine. The combine and the reduce-scatter add the
same bf16 terms in the same order as the combine followed by
rotated_collectives.reduce_scatter: the output is bitwise that pair's.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import rotated_collectives as rc
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import COMBINE_CHUNK, COMBINE_GROUP

CORES = rc.CORES
PARTS = len(rc.ROTATIONS)
LANES = rc.LANES
# Barrier semaphore group of this kernel; rotated_collectives uses 15, row_all_gather 17.
COLLECTIVE_ID = 19
CHIP_ORDER = (7, 6, 5, 3, 4, 2, 1, 0)
AXIS_BIT = {"x": 1, "y": 2, "z": 4}
SLOTS = rc._SLOTS
BASE = rc._BASE

# Table columns per mesh index.
_MATE = 0
_PARTNER = 1
_BLOCK = _PARTNER + PARTS * 3
_SLOT = _BLOCK + 2 * len(CHIP_ORDER)
TABLE_WIDTH = 64


def slot_of(part, offset):
    """The reduce-scatter slot of part `part` that holds the block of the chip
    `offset` away: bit t set where the offset crosses the part's t-th axis."""
    order = rc.ROTATIONS[part]
    return sum(1 << t for t in range(3) if offset & AXIS_BIT[order[t]])


def scatter_table(places):
    """int32 [CORES * TABLE_WIDTH]: per mesh index its chip-mate, its partner per
    (part, chip phase), the row block it combines at each position (the mate's
    half then its own, per offset of CHIP_ORDER) and the slot each part of an
    offset's block takes."""
    layout = rc.Layout(places)
    rows = []
    for m in range(CORES):
        coords, core = layout.places[m]
        mate = layout.peer(m, "c")
        partners = [layout.peer(m, order[k]) for order in rc.ROTATIONS for k in range(3)]
        blocks = []
        for half in (1 - core, core):
            for offset in CHIP_ORDER:
                chip = tuple(c ^ (offset >> a & 1) for a, c in enumerate(coords))
                blocks.append(layout.index[(chip, half)])
        slots = [slot_of(j, offset) for offset in CHIP_ORDER for j in range(PARTS)]
        row = [mate] + partners + blocks + slots
        rows.append(row + [0] * (TABLE_WIDTH - len(row)))
    return np.asarray(rows, dtype=np.int32).reshape(-1)


def plan(tokens, hidden):
    """(tokens per combine step, row parities, steps of row lead) for a chunk
    of `tokens` tokens, or None where the kernel does not apply: whole 16-row
    blocks, three column parts of whole lane tiles, and a block of whole steps
    that the parities divide."""
    if tokens % (CORES * COMBINE_GROUP) or hidden % (PARTS * LANES):
        return None
    rows = tokens // CORES
    for tb in (32, 16):
        if rows % tb or tb % COMBINE_CHUNK:
            continue
        steps = rows // tb
        for parities in (4, 5, 3, 2):
            if not steps % parities:
                return tb, parities, min(2, parities - 1)
    return None


def supports(tokens, hidden):
    return plan(tokens, hidden) is not None


@functools.cache
def build(*, tokens, topk, hidden, axis, interpret=False):
    """The kernel for one combine chunk of `tokens` tokens: (slab, coeff [tokens,
    topk], slab_row [tokens, topk], rank int32 [1], table) -> [tokens / 16,
    hidden] bf16, this core's rows of the chunk summed over the cores."""
    tb, parities, lead = plan(tokens, hidden)
    r = tokens // CORES
    W = hidden // PARTS
    spb = r // tb
    steps = 2 * len(CHIP_ORDER) * spb
    gt = COMBINE_GROUP
    ch = COMBINE_CHUNK
    lane_blocks = hidden // LANES
    RC = 16
    CC = 512 if W % 512 == 0 else LANES

    def slot(buf, j, s):
        return buf.at[j * SLOTS + s]

    def add_into(dst, src):
        """dst += src over [r, W] bf16 VMEM blocks: float32 add, rounded to bf16."""

        def rows(a, carry):
            rs = pl.ds(pl.multiple_of(a * RC, RC), RC)

            def cols(b, carry2):
                cs = pl.ds(pl.multiple_of(b * CC, CC), CC)
                dst[rs, cs] = (
                    dst[rs, cs].astype(jnp.float32) + src[rs, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                return carry2

            return lax.fori_loop(0, W // CC, cols, carry)

        lax.fori_loop(0, r // RC, rows, 0)

    def kernel(
        rank_ref,
        t_ref,
        pos_sm,
        coef_sm,
        arr_hbm,
        o_ref,
        rows_vm,
        tile_vm,
        blk_vm,
        buf,
        row_sems,
        msend,
        send,
        recv,
        osem,
    ):
        base = rank_ref[0] * TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        mate = at(_MATE)
        rc._barrier([mate] + [at(_PARTNER + k) for k in range(3)], axis, True)

        def remote(src, dst, s, rv, p):
            pltpu.make_async_remote_copy(
                src, dst, s, rv, device_id={axis: p}, device_id_type=rc.MESH
            ).start()

        def step_token(g):
            """First token in the chunk of global combine step g."""
            position = g // spb
            return at(_BLOCK + position % 2 * len(CHIP_ORDER) + position // 2) * r + g % spb * tb

        def start_row(pidx, parity, i, k):
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[pidx]],
                rows_vm.at[parity, k * tb + i],
                row_sems.at[parity],
            ).start(priority=k % 2)

        def one_step(g, parity, half, sub):
            nxt = step_token(jnp.minimum(g + lead, steps - 1))
            first = step_token(g)
            npar = (parity + lead) % parities
            pltpu.make_async_copy(
                rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
            ).wait()

            def one_chunk(c, carry):
                pn, pb, r0 = carry
                for u in range(ch):
                    for k in range(topk):
                        start_row(pn + (u * topk + k), npar, r0 + u, k)
                    acc = None
                    for k in range(topk):
                        term = (
                            rows_vm[parity, k * tb + r0 + u].astype(jnp.float32)
                            * coef_sm[pb + (u * topk + k)]
                        )
                        acc = term if acc is None else acc + term
                    tile_vm[r0 + u] = acc.astype(jnp.bfloat16)
                return pn + ch * topk, pb + ch * topk, r0 + ch

            lax.fori_loop(0, tb // ch, one_chunk, (nxt * topk, first * topk, jnp.int32(0)))
            for grp in range(tb // gt):
                row = pl.multiple_of(sub * tb + grp * gt, gt)
                blk_vm[half, pl.ds(row, gt)] = tile_vm[pl.ds(grp * gt, gt)].reshape(gt, hidden)

        def combine_block(position, half):
            """Combine the block at `position` into blk_vm[half]."""

            def group(q, carry):
                for p in range(parities):
                    sub = q * parities + p
                    one_step(position * spb + sub, p, half, sub)
                return carry

            lax.fori_loop(0, spb // parities, group, 0)

        def part(ref, j):
            return ref.at[:, pl.ds(j * W, W)]

        def one_offset(t, carry):
            @pl.when(t > 0)
            def _():
                rc._wait(blk_vm.at[0], msend)

            combine_block(2 * t, 0)
            for j in range(PARTS):
                i = at(_SLOT + t * PARTS + j)
                remote(part(blk_vm.at[0], j), slot(buf, j, i), msend, recv.at[j * SLOTS + i], mate)
            combine_block(2 * t + 1, 1)
            for j in range(PARTS):
                i = at(_SLOT + t * PARTS + j)
                rc._wait(slot(buf, j, i), recv.at[j * SLOTS + i])
                add_into(slot(buf, j, i), part(blk_vm.at[1], j))
                alive = jnp.int32(1)
                for k in (2, 1, 0):
                    bit = (i >> k) & 1
                    partner = at(_PARTNER + j * 3 + k)

                    @pl.when((alive == 1) & (bit == 1))
                    def _():
                        dst = BASE[k] + i - 2**k
                        remote(slot(buf, j, i), slot(buf, j, dst), send, recv.at[j * SLOTS + dst], partner)

                    @pl.when((alive == 1) & (bit == 0))
                    def _():
                        s = BASE[k] + i
                        rc._wait(slot(buf, j, s), recv.at[j * SLOTS + s])
                        add_into(slot(buf, j, i), slot(buf, j, s))

                    alive = alive * (1 - bit)
            return carry

        for p in range(lead):
            first = step_token(jnp.int32(p))
            for u in range(tb):
                for k in range(topk):
                    start_row((first + u) * topk + k, p, u, k)
        lax.fori_loop(0, len(CHIP_ORDER), one_offset, 0)
        outs = [
            pltpu.make_async_copy(slot(buf, j, 0), part(o_ref, j), osem) for j in range(PARTS)
        ]
        for c in outs:
            c.start()
        for p in range(lead):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
        rc._wait(blk_vm.at[0], msend)
        rc._wait(arr_hbm.at[pl.ds(0, 7 * r)], send)
        for c in outs:
            c.wait()

    need = 2 * (
        PARTS * SLOTS * r * W + 2 * r * hidden + parities * topk * tb * hidden + tb * hidden
    )
    info = pltpu.get_tpu_info() if not interpret else None
    limit = v2_host.vmem_limit(info) if not interpret else 63 << 20
    kw = {"interpret": pltpu.InterpretParams()} if interpret else {}
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=(
                pltpu.VMEM((parities, topk * tb, lane_blocks, LANES), jnp.bfloat16),
                pltpu.VMEM((tb, lane_blocks, LANES), jnp.bfloat16),
                pltpu.VMEM((2, r, hidden), jnp.bfloat16),
                pltpu.VMEM((PARTS * SLOTS, r, W), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((parities,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((PARTS * SLOTS,)),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((r, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=min(limit, need + (8 << 20)),
            disable_bounds_checks=True,
        ),
        name=f"moe_combine_scatter_t{tokens}_k{topk}_b{tb}",
        **kw,
    )

    def fn(slab, coeff, slab_row, rank, table):
        assert slab_row.shape == (tokens, topk) and coeff.shape == (tokens, topk)
        return call(
            rank.reshape(1).astype(jnp.int32),
            table,
            slab_row.astype(jnp.int32).reshape(-1),
            coeff.astype(jnp.float32).reshape(-1),
            slab,
        )

    return fn


def combine_scatter(slab, coeff, slab_row, rank, table, *, axis, interpret=False):
    """This core's rows (block `rank`) of the chunk's combine summed over the
    cores of `axis`: `lax.psum_scatter(tiled=True)` of the combine's output,
    in one kernel. slab [rows, hidden / 128, 128] bf16, coeff and slab_row
    [tokens, topk], rank int32 [1], table `scatter_table`."""
    tokens, topk = slab_row.shape
    hidden = slab.shape[1] * slab.shape[2]
    return build(tokens=tokens, topk=topk, hidden=hidden, axis=axis, interpret=interpret)(
        slab, coeff, slab_row, rank, table
    )

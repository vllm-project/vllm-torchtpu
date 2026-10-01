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
"""The streamed step of the FFN kernel: an expert too large for two
whole-expert weight slots, in column blocks of the intermediate. The
weight slots hold one block of one expert at a time (its up columns with
their scales and bias, its down rows), a group of the expert's rows and
their scale windows stay resident, double-buffered across groups, with
the group's float32 running sum in VMEM; every block adds its partial
down product for every tile of the group, and the last block finishes
each tile into the same staging, commit and push as the whole-expert
step. The refill sequence is (expert, group, block) in visit order,
prefetched weight_prefetch units ahead, so an expert boundary needs no
special case; each group reads the expert's weights once."""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import ffn_body as body
from vllm_torchtpu.kernels.fused_moe.v3.layout import FP4_PER_WORD, ROW_BLOCK


def unit_copies(ctx, expert, blk, slot):
    """The refill of weight slot `slot` with column block `blk` of
    `expert`: the up columns of the block (both halves with a gate) with
    their scales and bias on the w1 semaphore, the down rows of the block
    (and their fp4 block scales) on the w2 semaphore. Every copy is a whole
    slice of the slot, so the waits count the same bytes the starts
    issue."""
    b, operands, scratch, sems = ctx.b, ctx.operands, ctx.scratch, ctx.sems
    col = pl.multiple_of(blk * b.stream_block, b.stream_block)
    halves = [(0, col)] if b.no_gate else [(0, col), (b.stream_block, b.inter + col)]
    copies = []
    for dst_col, src_col in halves:
        copies.append(
            pltpu.make_async_copy(
                operands.w1.at[expert, :, pl.ds(src_col, b.stream_block)],
                scratch.w1.at[slot, :, pl.ds(dst_col, b.stream_block)],
                sems.w1.at[slot],
            )
        )
        if b.form.has_scales:
            src = (
                operands.w1_scales.at[expert, :, pl.ds(src_col, b.stream_block)]
                if b.form.block_scaled
                else operands.w1_scales.at[
                    pl.ds(expert, 1), pl.ds(src_col, b.stream_block)
                ]
            )
            copies.append(
                pltpu.make_async_copy(
                    src,
                    scratch.w1_scales.at[slot, :, pl.ds(dst_col, b.stream_block)],
                    sems.w1.at[slot],
                )
            )
        if b.has_w1_bias:
            copies.append(
                pltpu.make_async_copy(
                    operands.w1_bias.at[
                        pl.ds(expert, 1), pl.ds(src_col, b.stream_block)
                    ],
                    scratch.w1_bias.at[slot, :, pl.ds(dst_col, b.stream_block)],
                    sems.w1.at[slot],
                )
            )
    packed = FP4_PER_WORD if b.form.packed else 1
    row = pl.multiple_of(blk * (b.stream_block // packed), b.stream_block // packed)
    copies.append(
        pltpu.make_async_copy(
            operands.w2.at[expert, pl.ds(row, b.stream_block // packed)],
            scratch.w2.at[slot],
            sems.w2.at[slot],
        )
    )
    if b.form.block_scaled:
        scale_rows = b.stream_block // b.weight_block
        scale_row = pl.multiple_of(blk * scale_rows, scale_rows)
        copies.append(
            pltpu.make_async_copy(
                operands.w2_scales.at[expert, pl.ds(scale_row, scale_rows)],
                scratch.w2_scales.at[slot],
                sems.w2.at[slot],
            )
        )
    return copies


def start_unit(ctx, expert, blk, slot):
    """Start the weight copies of one (expert, block) unit into a slot."""
    for copy in unit_copies(ctx, expert, blk, slot):
        copy.start(priority=ctx.b.weight_dma_priority)


def wait_unit(ctx, expert, blk, slot):
    """Wait for the weight copies of one (expert, block) unit."""
    for copy in unit_copies(ctx, expert, blk, slot):
        copy.wait()


def groups_of(ctx, expert):
    """Row groups of a streamed expert (at least one: an expert in the
    visit order has rows)."""
    return -(-ctx.tables.expert_rows[expert] // ctx.b.stream_rows)


def units_ahead(ctx, visit_i, g, blk, steps):
    """The (visit index, group, block) `steps` refill units after
    (visit_i, g, blk) in the streamed order: blocks within a group, then
    the expert's next group (the same blocks again), then the next
    expert's first group. Past the last expert the visit index reaches
    n_visit, which the callers test."""
    n_blocks = ctx.b.n_stream_blocks
    for _ in range(steps):
        blk = blk + 1
        wrap = blk == n_blocks
        blk = jnp.where(wrap, 0, blk)
        g = jnp.where(wrap, g + 1, g)
        expert = ctx.tables.visit_order[
            jnp.clip(visit_i, 0, jnp.maximum(ctx.n_visit - 1, 0))
        ]
        advance = jnp.logical_and(wrap, g == groups_of(ctx, expert))
        g = jnp.where(advance, 0, g)
        visit_i = jnp.where(advance, visit_i + 1, visit_i)
    return visit_i, g, blk


def start_units_ahead(ctx, visit_i, g, blk, unit_count):
    """Refills, if the sequence has one, the unit weight_prefetch units
    after (visit_i, g, blk), unit number unit_count, into the slot that
    unit takes."""
    b = ctx.b
    ahead_i, _, ahead_blk = units_ahead(ctx, visit_i, g, blk, b.weight_prefetch)

    @pl.when(ahead_i < ctx.n_visit)
    def _():
        start_unit(
            ctx,
            ctx.tables.visit_order[ahead_i],
            ahead_blk,
            lax.rem(unit_count + b.weight_prefetch, jnp.int32(b.weight_slots)),
        )


def first_refills(ctx):
    """The first weight refills of the streamed sequence, each guarded on
    the visit count."""
    for u in range(ctx.b.weight_prefetch):
        ahead_i, _, ahead_blk = units_ahead(
            ctx, jnp.int32(0), jnp.int32(0), jnp.int32(-1), u + 1
        )

        @pl.when(ahead_i < ctx.n_visit)
        def _(u=u, ahead_i=ahead_i, ahead_blk=ahead_blk):
            start_unit(ctx, ctx.tables.visit_order[ahead_i], ahead_blk, u)


def group_rows(ctx, expert, g):
    """The live rows of the expert's g-th row group."""
    return jnp.minimum(
        ctx.tables.expert_rows[expert] - g * ctx.b.stream_rows, ctx.b.stream_rows
    )


def following(ctx, visit_i, g):
    """(visit index, group, valid) of the group issued after group g of the
    expert at visit_i: the expert's next group, or the next expert's first."""
    expert = ctx.tables.visit_order[visit_i]
    last = g + 1 >= groups_of(ctx, expert)
    visit_next = jnp.where(last, visit_i + 1, visit_i)
    return visit_next, jnp.where(last, 0, g + 1), visit_next < ctx.n_visit


def start_group_window(ctx, visit_i, g, buf):
    """Starts group g's token window (the expert at visit_i) into buffer
    buf's window."""
    b, tables, scratch, sems = ctx.b, ctx.tables, ctx.scratch, ctx.sems
    expert = tables.visit_order[jnp.minimum(visit_i, b.experts_per_shard - 1)]
    body.start_token_window(
        ctx,
        tables.expert_base[expert] + g * b.stream_rows,
        scratch.token_window.at[buf],
        sems.token_window.at[buf],
    )


def issue_group(ctx, expert, g, buf):
    """Issues group g of `expert` (its tiles' rows and scale windows) into
    group buffer `buf`."""
    b, tables, scratch, sems = ctx.b, ctx.tables, ctx.scratch, ctx.sems
    live = group_rows(ctx, expert, g)
    base = tables.expert_base[expert] + g * b.stream_rows
    window = scratch.token_window.at[buf]
    body.wait_token_window(window, sems.token_window.at[buf])
    offset = body.window_offset(ctx, base, window)
    for t in range(b.tiles_per_group):

        @pl.when(live > t * b.tile_rows)
        def _(t=t):
            body.issue_rows(
                ctx,
                base + t * b.tile_rows,
                jnp.minimum(live - t * b.tile_rows, b.tile_rows) // ROW_BLOCK,
                lambda i, t=t: scratch.rows.at[buf, t * b.tile_rows + i],
                lambda plane, t=t: scratch.row_scales.at[buf, t, plane],
                sems.rows.at[buf],
                body.window_tokens(window, offset + t * b.tile_rows),
            )


def issue_first(ctx):
    """The prologue's row fetch: the first expert's first group."""

    @pl.when(ctx.n_visit > 0)
    def _():
        start_group_window(ctx, jnp.int32(0), jnp.int32(0), jnp.int32(0))
        issue_group(ctx, ctx.tables.visit_order[0], jnp.int32(0), jnp.int32(0))
        visit_1, g_1, ok_1 = following(ctx, jnp.int32(0), jnp.int32(0))

        @pl.when(ok_1)
        def _():
            start_group_window(ctx, visit_1, g_1, jnp.int32(1))


def wait_group(ctx, live, n_tiles, buf):
    """Wait for a row group's rows and row scales to land in buf."""
    scratch, sems = ctx.scratch, ctx.sems
    pltpu.make_async_copy(
        scratch.rows.at[buf, pl.ds(0, live)],
        scratch.rows.at[buf, pl.ds(0, live)],
        sems.rows.at[buf],
    ).wait()
    pltpu.make_async_copy(
        scratch.row_scales.at[buf, pl.ds(0, n_tiles)],
        scratch.row_scales.at[buf, pl.ds(0, n_tiles)],
        sems.rows.at[buf],
    ).wait()


def group_tile(ctx, expert, g, buf, t):
    """Tile t of the group as (rows [tile_rows, hidden], row scales
    [tile_rows, planes], its first routed row)."""
    b, scratch = ctx.b, ctx.scratch
    row_base = ctx.tables.expert_base[expert] + g * b.stream_rows + t * b.tile_rows
    lo = pl.multiple_of(t * b.tile_rows, b.tile_rows)
    rows = scratch.rows[buf, pl.ds(lo, b.tile_rows)].reshape(b.tile_rows, b.hidden)
    scales = jnp.concatenate(
        [
            body.tile_scale_column(
                scratch.row_scales[buf, t, plane], row_base, b.tile_rows
            )
            for plane in range(b.scale_planes)
        ],
        axis=1,
    )
    return rows, scales, lo


def streamed_group(ctx, expert, visit_i, g, buf, n_tiles, carry):
    """One group of an expert: every column block over every tile of the
    group into the running sum; the last block finishes each tile into the
    result staging and commits it. The carry is the staging carry plus the
    refill unit counter."""
    b, scratch = ctx.b, ctx.scratch
    n_blocks = b.n_stream_blocks
    *staging, unit_count = carry

    def refill(blk, unit_count):
        """Waits block blk's slot and issues the refill ahead."""
        slot = lax.rem(unit_count, jnp.int32(b.weight_slots))
        wait_unit(ctx, expert, blk, slot)
        start_units_ahead(ctx, visit_i, g, blk, unit_count)
        return slot

    def block_step(blk, unit_count):
        slot = refill(blk, unit_count)

        def tile_sum(t, c):
            rows, scales, lo = group_tile(ctx, expert, g, buf, t)
            part, _ = body.down_product(ctx, expert, slot, rows, scales)

            @pl.when(blk == 0)
            def _():
                scratch.acc[pl.ds(lo, b.tile_rows), :] = part

            @pl.when(blk > 0)
            def _():
                scratch.acc[pl.ds(lo, b.tile_rows), :] = (
                    scratch.acc[pl.ds(lo, b.tile_rows), :] + part
                )

            return c

        lax.fori_loop(0, n_tiles, tile_sum, jnp.int32(0))
        return unit_count + 1

    # Every block but the last only accumulates; the last block also
    # finishes and stages each tile, which threads the staging carry.
    unit_count = lax.fori_loop(0, n_blocks - 1, block_step, unit_count)
    last_slot = refill(jnp.int32(n_blocks - 1), unit_count)

    def last_tile(t, c):
        rows, scales, lo = group_tile(ctx, expert, g, buf, t)
        part, _ = body.down_product(ctx, expert, last_slot, rows, scales)
        acc = part
        if n_blocks > 1:
            acc = scratch.acc[pl.ds(lo, b.tile_rows), :] + part
        result, result_scales = body.finish_rows(ctx, expert, acc, None)
        slot, pending_here = body.slot_and_pending(ctx, c)

        @pl.when(pending_here > 0)
        def _():
            body.wait_commits(ctx, slot, pending_here)

        live = jnp.minimum(group_rows(ctx, expert, g) - t * b.tile_rows, b.tile_rows)
        live_blocks = live // ROW_BLOCK
        tile_block_base = (g * b.stream_rows + t * b.tile_rows) // ROW_BLOCK
        body.stage_tile(ctx, slot, result, result_scales)
        body.commit_tile(ctx, expert, slot, tile_block_base, live_blocks)
        return (c[0] + 1,) + tuple(
            jnp.where(slot == i, live_blocks, c[1 + i]) for i in range(b.result_slots)
        )

    staging = lax.fori_loop(0, n_tiles, last_tile, tuple(staging))
    return (*staging, unit_count + 1)


def expert_step(ctx, expert, visit_i, carry):
    """One streamed expert: its row groups in turn, each group's rows
    waited, the next group's (or the next expert's first) issued into the
    other buffer, then the blocks."""
    b, tables = ctx.b, ctx.tables
    *staging, unit_count, group_count = carry

    def group_step(g, c):
        *staging, unit_count, group_count = c
        buf = lax.rem(group_count, jnp.int32(2))
        live = group_rows(ctx, expert, g)
        n_tiles = -(-live // b.tile_rows)
        wait_group(ctx, live, n_tiles, buf)
        other = 1 - buf

        @pl.when(g + 1 < groups_of(ctx, expert))
        def _():
            issue_group(ctx, expert, g + 1, other)

        @pl.when(
            jnp.logical_and(g + 1 == groups_of(ctx, expert), visit_i + 1 < ctx.n_visit)
        )
        def _():
            issue_group(ctx, tables.visit_order[visit_i + 1], jnp.int32(0), other)

        # The window of the group after the one just issued, into this
        # group's buffer (its own window was consumed when it was issued).
        visit_1, g_1, ok_1 = following(ctx, visit_i, g)
        visit_2, g_2, ok_2 = following(ctx, jnp.minimum(visit_1, ctx.n_visit - 1), g_1)

        @pl.when(jnp.logical_and(ok_1, ok_2))
        def _():
            start_group_window(ctx, visit_2, g_2, buf)

        *staging, unit_count = streamed_group(
            ctx, expert, visit_i, g, buf, n_tiles, (*staging, unit_count)
        )
        return (*staging, unit_count, group_count + 1)

    return lax.fori_loop(
        0, groups_of(ctx, expert), group_step, (*staging, unit_count, group_count)
    )


def visit(ctx, visit_i, carry):
    """One visit of the streamed build: the expert's groups, the drain,
    the push. The carry is the staging carry, the unit counter and the
    group counter."""
    expert = ctx.tables.visit_order[visit_i]
    *staging, unit_count, group_count = expert_step(ctx, expert, visit_i, carry)
    staging = body.drain_commits(ctx, tuple(staging))
    body.push_expert(ctx, expert)
    return (*staging, unit_count, group_count)


def initial_carry(ctx):
    """The loop carry at start: tiles, pending commits, units, groups."""
    return (
        (jnp.int32(0),)
        + (jnp.int32(0),) * ctx.b.result_slots
        + (jnp.int32(0), jnp.int32(0))
    )  # units, groups

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
"""The whole-expert step of the FFN kernel: an expert's two matrices held
in one weight slot, its rows streamed a tile at a time through the result
slots, each tile computed, staged and committed as it lands."""
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import ffn
from . import ffn_body as body
from .layout import LANES, ROW_BLOCK


def w1_copy(ctx, expert, slot):
    """The async copy of an expert's first weight into a slot."""
    return pltpu.make_async_copy(ctx.operands.w1.at[expert],
                                 ctx.scratch.w1.at[slot],
                                 ctx.sems.w1.at[slot],
                                 priority=ctx.b.weight_dma_priority,
                                 strict_ordering=False)


def w2_copy(ctx, expert, slot):
    """The async copy of an expert's second weight into a slot."""
    return pltpu.make_async_copy(ctx.operands.w2.at[expert],
                                 ctx.scratch.w2.at[slot],
                                 ctx.sems.w2.at[slot],
                                 priority=ctx.b.weight_dma_priority,
                                 strict_ordering=False)


def first_refills(ctx):
    """The first weight refills, each guarded on the visit count like the
    head waits. The range is clamped so a one-expert shard never reads
    visit_order[1]."""
    b = ctx.b
    for i in range(min(b.weight_prefetch, b.experts_per_shard)):

        @pl.when(jnp.int32(i) < ctx.n_visit)
        def _(i=i):
            expert = ctx.tables.visit_order[i]
            w1_copy(ctx, expert, i).start()
            w2_copy(ctx, expert, i).start()


def start_tile_window(ctx, row_base, window_ref, sem):
    """Starts the tile's window of the routed-row token table (none with
    sorted rows: the fetch reads no tokens)."""
    if ctx.b.sorted_rows:
        return
    body.start_token_window(ctx, row_base, window_ref, sem)


def issue_tile_rows(ctx, row_base, live_blocks, slot, window_ref, sem):
    """Issues one tile's row stream and scale window into `slot`, its
    tokens read from `window_ref`, whose copy was started one tile (or
    one expert) earlier and is waited here. With sorted rows the tile is
    one copy of consecutive rows from row_base, followed by the scale
    window copies."""
    b = ctx.b
    if b.sorted_rows:
        rows = live_blocks * ROW_BLOCK
        pltpu.make_async_copy(ctx.operands.tokens.at[pl.ds(row_base, rows)],
                              ctx.scratch.rows.at[slot, pl.ds(0, rows)],
                              ctx.sems.rows.at[slot],
                              strict_ordering=False).start()
        for plane in range(b.scale_planes):
            pltpu.make_async_copy(
                ctx.operands.token_scales.at[plane,
                                             pl.ds(row_base //
                                                   LANES, b.window_rows)],
                ctx.scratch.row_scales.at[slot, plane],
                ctx.sems.rows.at[slot],
                strict_ordering=False).start()
        return
    body.wait_token_window(window_ref, sem)
    offset = body.window_offset(ctx, row_base, window_ref)
    body.issue_rows(ctx, row_base, live_blocks,
                    lambda i: ctx.scratch.rows.at[slot, i],
                    lambda plane: ctx.scratch.row_scales.at[slot, plane],
                    ctx.sems.rows.at[slot],
                    body.window_tokens(window_ref, offset))


def start_head_windows(ctx, expert):
    """Starts the windows of an expert's first row_fetch_ahead tiles into
    the head windows (consumed by issue_first_tiles for that expert)."""
    b, tables, scratch, sems = ctx.b, ctx.tables, ctx.scratch, ctx.sems
    if b.sorted_rows:
        return
    for j in range(b.row_fetch_ahead):

        @pl.when(tables.expert_rows[expert] > j * b.tile_rows)
        def _(j=j):
            start_tile_window(ctx,
                              tables.expert_base[expert] + j * b.tile_rows,
                              scratch.token_head.at[j], sems.token_head.at[j])


def wait_tile_rows(ctx, live_rows, slot):
    """Waits one tile's rows and scale windows (byte-counted, so the two
    waits are exact under any landing order; the second counts every
    plane's window)."""
    scratch, sems = ctx.scratch, ctx.sems
    pltpu.make_async_copy(scratch.rows.at[slot, pl.ds(0, live_rows)],
                          scratch.rows.at[slot, pl.ds(0, live_rows)],
                          sems.rows.at[slot],
                          strict_ordering=False).wait()
    pltpu.make_async_copy(scratch.row_scales.at[slot],
                          scratch.row_scales.at[slot],
                          sems.rows.at[slot],
                          strict_ordering=False).wait()


def issue_first_tiles(ctx, expert, slot_of):
    """Issues an expert's first row_fetch_ahead tiles, tile j into result
    slot slot_of(j)."""
    b, tables, scratch, sems = ctx.b, ctx.tables, ctx.scratch, ctx.sems
    for j in range(b.row_fetch_ahead):

        @pl.when(tables.expert_rows[expert] > j * b.tile_rows)
        def _(j=j):
            issue_tile_rows(
                ctx, tables.expert_base[expert] + j * b.tile_rows,
                jnp.minimum(tables.expert_rows[expert] - j * b.tile_rows,
                            b.tile_rows) // ROW_BLOCK, slot_of(j),
                scratch.token_head.at[j], sems.token_head.at[j])


def issue_first(ctx):
    """The prologue's row fetch: the first expert's first tiles."""

    @pl.when(ctx.n_visit > 0)
    def _():
        start_head_windows(ctx, ctx.tables.visit_order[0])
        issue_first_tiles(ctx, ctx.tables.visit_order[0],
                          lambda j: jnp.int32(j))


def tile_step(ctx, expert, weight_slot, rows, expert_first_row, n_tiles, t,
              carry):
    """Computes tile t of the expert, stages it and commits it."""
    b, scratch = ctx.b, ctx.scratch
    tile_count, *pending = carry
    row_base = expert_first_row + t * b.tile_rows
    live_rows = jnp.minimum(rows - t * b.tile_rows, b.tile_rows)
    live_blocks = live_rows // ROW_BLOCK
    tile_block_base = t * b.tile_blocks
    slot = lax.rem(tile_count, jnp.int32(b.result_slots))
    # The slot's last committed block count. Zero after a drain.
    pending_here = pending[0]
    for i in range(1, b.result_slots):
        pending_here = jnp.where(slot == i, pending[i], pending_here)

    @pl.when(pending_here > 0)
    def _():
        body.wait_commits(ctx, slot, pending_here)

    wait_tile_rows(ctx, live_rows, slot)
    ahead_slot = lax.rem(tile_count + jnp.int32(b.row_fetch_ahead),
                         jnp.int32(b.result_slots))

    @pl.when(t + b.row_fetch_ahead < n_tiles)
    def _():
        next_rows = jnp.minimum(rows - (t + b.row_fetch_ahead) * b.tile_rows,
                                b.tile_rows)
        issue_tile_rows(ctx, row_base + b.row_fetch_ahead * b.tile_rows,
                        next_rows // ROW_BLOCK, ahead_slot,
                        scratch.token_window.at[ahead_slot],
                        ctx.sems.token_window.at[ahead_slot])

    # The window of the tile issued next step, into the slot whose window
    # was consumed when this tile's rows were issued (result_slots >
    # row_fetch_ahead keeps the two apart).
    next_slot = lax.rem(tile_count + jnp.int32(b.row_fetch_ahead + 1),
                        jnp.int32(b.result_slots))

    @pl.when(t + b.row_fetch_ahead + 1 < n_tiles)
    def _():
        start_tile_window(ctx,
                          row_base + (b.row_fetch_ahead + 1) * b.tile_rows,
                          scratch.token_window.at[next_slot],
                          ctx.sems.token_window.at[next_slot])

    tile = scratch.rows[slot].reshape(b.tile_rows, b.hidden)
    row_scales = jnp.concatenate([
        body.tile_scale_column(scratch.row_scales[slot, plane], row_base,
                               b.tile_rows) for plane in range(b.scale_planes)
    ],
                                 axis=1)
    acc2, mid_scales = body.down_product(ctx, expert, weight_slot, tile,
                                         row_scales)
    result, result_scales = body.finish_rows(ctx, expert, acc2, mid_scales)
    body.stage_tile(ctx, slot, result, result_scales)
    body.commit_tile(ctx, expert, slot, tile_block_base, live_blocks)
    return (tile_count + 1, ) + tuple(
        jnp.where(slot == i, live_blocks, pending[i])
        for i in range(b.result_slots))


def expert_step(ctx, expert, visit_i, carry):
    """One expert: its tiles, then the next expert's first tiles."""
    b, tables = ctx.b, ctx.tables
    weight_slot = lax.rem(visit_i, jnp.int32(b.weight_slots))
    w1_copy(ctx, expert, weight_slot).wait()
    w2_copy(ctx, expert, weight_slot).wait()
    if b.intermediate == "bf16":  # the down matmul's bf16 copy of this expert's w2
        ctx.scratch.w2_bf16[...] = ctx.scratch.w2[weight_slot].astype(
            jnp.bfloat16)
    if ctx.scratch.w1_bf16 is not None:  # bf16 token rows: the up matmul's copy of w1
        ctx.scratch.w1_bf16[...] = ctx.scratch.w1[weight_slot].astype(
            jnp.bfloat16)
    if b.form.packed:  # this expert's four-bit slabs widened to fp8, once
        ctx.scratch.w1_fp8[...] = ffn.widen_packed_slab(
            ctx.scratch.w1, weight_slot, b.form.weight_dtype)
        ctx.scratch.w2_fp8[...] = ffn.widen_packed_slab(
            ctx.scratch.w2, weight_slot, b.form.weight_dtype)

    @pl.when(visit_i + b.weight_prefetch < ctx.n_visit)
    def _():
        refill_slot = lax.rem(visit_i + b.weight_prefetch,
                              jnp.int32(b.weight_slots))
        ahead = tables.visit_order[visit_i + b.weight_prefetch]
        w1_copy(ctx, ahead, refill_slot).start()
        w2_copy(ctx, ahead, refill_slot).start()

    rows = tables.expert_rows[expert]
    expert_first_row = tables.expert_base[expert]
    n_tiles = -(-rows // b.tile_rows)
    # Token windows ahead: this expert's tile row_fetch_ahead (its first
    # tiles' windows came with the head windows), and the next expert's
    # first tiles, started here so they land during this expert's tiles.
    tile_count = carry[0]
    first_slot = lax.rem(tile_count + jnp.int32(b.row_fetch_ahead),
                         jnp.int32(b.result_slots))

    @pl.when(n_tiles > b.row_fetch_ahead)
    def _():
        start_tile_window(ctx,
                          expert_first_row + b.row_fetch_ahead * b.tile_rows,
                          ctx.scratch.token_window.at[first_slot],
                          ctx.sems.token_window.at[first_slot])

    @pl.when(visit_i + 1 < ctx.n_visit)
    def _():
        start_head_windows(ctx, tables.visit_order[visit_i + 1])

    carry = lax.fori_loop(
        0, n_tiles, lambda t, c: tile_step(ctx, expert, weight_slot, rows,
                                           expert_first_row, n_tiles, t, c),
        carry)

    tiles_done = carry[0]

    @pl.when(visit_i + 1 < ctx.n_visit)
    def _():
        issue_first_tiles(
            ctx, tables.visit_order[visit_i + 1], lambda j: lax.rem(
                tiles_done + jnp.int32(j), jnp.int32(b.result_slots)))

    return carry


def visit(ctx, visit_i, carry):
    """One visit of the whole-expert build: the expert, the drain, the
    push."""
    expert = ctx.tables.visit_order[visit_i]
    carry = expert_step(ctx, expert, visit_i, carry)
    carry = body.drain_commits(ctx, carry)
    body.push_expert(ctx, expert)
    return carry


def initial_carry(ctx):
    """The loop carry at start: tiles and pending commits per slot."""
    return (jnp.int32(0), ) + (jnp.int32(0), ) * ctx.b.result_slots

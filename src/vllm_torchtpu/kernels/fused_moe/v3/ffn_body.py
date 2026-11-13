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
"""The FFN kernel's body, shared between its two expert steps: the build
record every step reads, the context that carries the refs, and the
plumbing both steps use (row streams, commits, staging, the down
product, the push, the drains).

ffn_expert_whole.py and ffn_expert_streamed.py hold the two expert
steps; ffn_kernel.py builds the program and runs one of them.
"""
import dataclasses

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import ffn
from .layout import FP4_PER_WORD, LANES, MESH_AXIS, ROW_BLOCK, device_coords
from .routing_tables import COUNT_ROWS

# Rows of the count table, in the order routing_tables.shard_counts writes
# them. The kernel closes each one's sum over the shards on its scalar core.
(COUNT_SEND_ROWS, COUNT_RECEIVE_ROWS, COUNT_SEND_ALIGNED_ROWS,
 COUNT_RECEIVE_ALIGNED_ROWS, COUNT_VISITS) = range(COUNT_ROWS)


@dataclasses.dataclass(frozen=True)
class Build:
    """Everything static a program body reads: the shape, the format, the
    depths, the rounding, the streaming and the geometry of one built
    kernel."""
    tile_rows: int
    hidden: int
    inter: int
    num_shards: int
    experts_per_shard: int
    form: object
    weight_block: object
    has_w1_bias: bool
    has_w2_bias: bool
    no_gate: bool
    activation: str
    swiglu_limit: float
    pallas_transport: bool
    weight_slots: int
    weight_prefetch: int
    result_slots: int
    row_fetch_ahead: int
    activation_block: int
    token_block: int
    result_dtype: object
    elementwise: object
    has_result_scales: bool
    stream_block: int
    stream_rows: int
    activation_columns_per_pass: int
    weight_dma_priority: int
    sorted_rows: bool  # each tile fetched as one copy from rows sorted into expert order (no token windows)
    intermediate: str  # "fp8" (requantized between the matmuls) or "bf16"
    lane_blocks: int
    result_blocks: int
    tile_blocks: int
    scale_planes: int
    scale_lanes: int
    window_rows: int
    mesh_axes: tuple = (
        MESH_AXIS, )  # the mesh's axes; MESH_AXIS is the expert-parallel one
    token_dtype: object = jnp.float8_e4m3fn  # the token rows' element type on the wire

    @property
    def rows_bf16(self):
        """bf16 token rows (Config.token_rows): the up matmul runs in bf16."""
        return jnp.dtype(self.token_dtype) == jnp.dtype(jnp.bfloat16)

    @property
    def n_stream_blocks(self):
        """Streamed blocks the intermediate width splits into."""
        return self.inter // self.stream_block if self.stream_block else 1

    @property
    def tiles_per_group(self):
        """Row tiles per streamed row group (0 with whole experts)."""
        return (self.stream_rows // self.tile_rows) if self.stream_block else 0


@dataclasses.dataclass(frozen=True)
class Context:
    """One program invocation: the build and its refs, plus the two scalars
    every step reads (the visit count and this shard's index)."""
    b: Build
    tables: object
    operands: object
    outputs: object
    scratch: object
    sems: object
    n_visit: object
    me: object


def wait_rows(sem, ref, rows):
    """Waits until `rows` rows' worth of DMAs on `sem` have landed."""
    pltpu.make_async_copy(ref.at[pl.ds(0, rows)],
                          ref.at[pl.ds(0, rows)],
                          sem,
                          strict_ordering=False).wait()


def and_nonempty(predicate, count):
    """`predicate` and a nonzero count: a zero-length DMA never issues."""
    return jnp.logical_and(predicate, count > 0)


def tile_scale_column(window, row_base, tile_rows):
    """One tile's activation row scales as the [tile_rows, 1] f32 column
    the FFN takes, out of the [sublanes, LANES] window of the scale table
    its rows fall in (row r is at sublane r // LANES, lane r % LANES).

    A tile's first row is block-aligned but not LANES-aligned, so the tile
    straddles two sublanes: the window's lanes are rotated by the first
    row's lane and each sublane is selected against the next, which puts
    the tile's rows in order on one sublane. The column is that sublane
    read the other way round, one lane row at a time (the single-row fold
    the layout pass supports at every tile height).
    """
    window = pltpu.bitcast(window, jnp.float32)
    resid = lax.rem(row_base, jnp.int32(LANES))
    rotated = pltpu.roll(window, LANES - resid, 1)
    lane = lax.broadcasted_iota(jnp.int32, rotated[:-1].shape, 1)
    rows = jnp.where(lane < LANES - resid, rotated[:-1], rotated[1:])
    return jnp.concatenate([rows[i][:, None] for i in range(rows.shape[0])],
                           axis=0)[:tile_rows]


def count(ctx, row):
    """Closes one count's sum over the shards on the scalar core."""
    total = ctx.tables.counts[row, 0]
    for d in range(1, ctx.b.num_shards):
        total = total + ctx.tables.counts[row, d]
    return total


def resident_copy(ctx, src, dst):
    """Copy src to dst on the resident semaphore and wait for it. A source with more leading entries than the
    destination (a placement's copy slots after the shard's primaries) contributes its head only."""
    if src.shape[0] != dst.shape[0]:
        src = src.at[pl.ds(0, dst.shape[0])]
    copy = pltpu.make_async_copy(src,
                                 dst,
                                 ctx.sems.copy,
                                 strict_ordering=False)
    copy.start()
    copy.wait()


def resident_copies(ctx):
    """The prologue's resident tables: the whole-expert build keeps every
    expert's up scales and bias resident; a streamed build carries them
    per weight slot and keeps only the down scales and bias."""
    b = ctx.b
    if b.form.has_scales and not (b.stream_block and b.form.block_scaled):
        if not b.stream_block:
            resident_copy(ctx, ctx.operands.w1_scales, ctx.scratch.w1_scales)
        resident_copy(ctx, ctx.operands.w2_scales, ctx.scratch.w2_scales)
    if b.has_w1_bias and not b.stream_block:
        resident_copy(ctx, ctx.operands.w1_bias, ctx.scratch.w1_bias)
    if b.has_w2_bias:
        resident_copy(ctx, ctx.operands.w2_bias, ctx.scratch.w2_bias)


def upcast_blocks(ctx, slot):
    """(w1_block, w2_block): the fp8 readers of one contraction block of the
    current expert's weights: from the slabs widened once per expert in a
    whole-expert build, or widened here per tile from packed weight slot
    `slot` in a streamed one."""
    block = ctx.b.weight_block
    if ctx.scratch.w1_fp8 is not None:
        return (lambda blk: ctx.scratch.w1_fp8[pl.ds(blk * block, block), :],
                lambda blk: ctx.scratch.w2_fp8[pl.ds(blk * block, block), :])
    if not ctx.b.form.packed:
        # Block-scaled fp8: the weights sit in their slot as the matmul
        # reads them.
        return (lambda blk: ctx.scratch.w1[
            slot, pl.ds(blk * block, block), :], lambda blk: ctx.scratch.w2[
                slot, pl.ds(blk * block, block), :])
    packed_rows = block // FP4_PER_WORD
    dtype = ctx.b.form.weight_dtype
    return (lambda blk: ffn.upcast_packed_block_to_fp8(
        ctx.scratch.w1, slot, packed_rows, blk, dtype),
            lambda blk: ffn.upcast_packed_block_to_fp8(
                ctx.scratch.w2, slot, packed_rows, blk, dtype))


# ---- the row streams ----
def token_window_blocks(n_rows, table_blocks):
    """Blocks of LANES rows a scalar-memory window needs to hold n_rows of
    the routed-row token table from any ROW_BLOCK-aligned row: the rows'
    whole blocks and one more for the row's offset in its block, at most
    the table itself. The table travels as [blocks, 1, LANES]: its block
    axis is untiled, so a copy may start at any block."""
    return min(-(-n_rows // LANES) + 1, table_blocks)


def window_start(ctx, row_base, window_ref):
    """The table block a window for the rows from row_base copies from:
    row_base's block, pulled back where the window would run past the
    table (the rows themselves never do: the routed rows carry a tile of
    tail slack)."""
    table_blocks = ctx.operands.token_of_row.shape[0]
    return jnp.minimum(row_base // LANES, table_blocks - window_ref.shape[0])


def start_token_window(ctx, row_base, window_ref, sem):
    """Starts the copy of the token table's window covering the rows from
    row_base (the window's whole length, one descriptor)."""
    start = window_start(ctx, row_base, window_ref)
    pltpu.make_async_copy(ctx.operands.token_of_row.at[pl.ds(
        start, window_ref.shape[0])],
                          window_ref,
                          sem,
                          strict_ordering=False).start()


def wait_token_window(window_ref, sem):
    """Waits the window's copy (byte-counted on the window itself)."""
    pltpu.make_async_copy(window_ref, window_ref, sem,
                          strict_ordering=False).wait()


def window_offset(ctx, row_base, window_ref):
    """Where the rows from row_base start inside their window, in rows
    (window_tokens reads from there)."""
    return row_base - window_start(ctx, row_base, window_ref) * LANES


def window_tokens(window_ref, offset):
    """The tokens of the rows from window row `offset`, by ROW_BLOCK block:
    tokens_of(i)(r) is row r of block i. Blocks and the offset are whole
    ROW_BLOCKs and LANES is a multiple of ROW_BLOCK, so a block's rows
    share one window row, located once per block."""

    def tokens_of(i):
        k = offset + i * ROW_BLOCK
        block, lane = k // LANES, k % LANES
        return lambda r: window_ref[block, 0, lane + r]

    return tokens_of


def issue_rows(ctx, row_base, live_blocks, row_ref_of, window_ref_of, sem,
               tokens_of):
    """Issues one tile's row stream (row i of the tile into row_ref_of(i))
    and its scale windows (plane p into window_ref_of(p)) on `sem`.
    tokens_of(i)(r) is the token of row r of the tile's block i, read from
    the tile's scalar-memory window of the routed-row token table."""
    b = ctx.b

    def issue_block(i, carry):
        token_of = tokens_of(i)
        for r in range(ROW_BLOCK):
            pltpu.make_async_copy(ctx.operands.tokens.at[token_of(r)],
                                  row_ref_of(i * ROW_BLOCK + r),
                                  sem,
                                  strict_ordering=False).start()
        return carry

    lax.fori_loop(0, live_blocks, issue_block, jnp.int32(0))
    for plane in range(b.scale_planes):
        pltpu.make_async_copy(
            ctx.operands.token_scales.at[plane,
                                         pl.ds(row_base //
                                               LANES, b.window_rows)],
            window_ref_of(plane),
            sem,
            strict_ordering=False).start()


# ---- commits and staging ----
def wait_commits(ctx, slot, live_blocks):
    """Wait for a slot's committed result rows and their scales."""
    wait_rows(ctx.sems.commit.at[slot], ctx.outputs.outgoing,
              live_blocks * ROW_BLOCK)
    if ctx.b.has_result_scales:
        wait_rows(ctx.sems.commit_scales.at[slot], ctx.outputs.outgoing_scales,
                  live_blocks)


def commit_tile(ctx, expert, slot, tile_block_base, live_blocks):
    """Commits the tile's part of each (expert, destination) run."""
    tables, scratch, outputs, sems = (ctx.tables, ctx.scratch, ctx.outputs,
                                      ctx.sems)
    for d in range(ctx.b.num_shards):
        start = tables.run_start[expert, d]
        length = tables.run_blocks[expert, d]
        lo = jnp.maximum(start, tile_block_base)
        hi = jnp.minimum(start + length, tile_block_base + live_blocks)
        overlap = jnp.maximum(hi - lo, 0)
        lo = jnp.minimum(lo, hi)  # an empty run leaves lo unclamped
        src_block = lo - tile_block_base
        dst_block = tables.outgoing_offset[expert, d] + (lo - start)

        @pl.when(overlap > 0)
        def _(src_block=src_block, dst_block=dst_block, overlap=overlap):
            pltpu.make_async_copy(
                scratch.result_rows.at[slot,
                                       pl.ds(src_block * ROW_BLOCK, overlap *
                                             ROW_BLOCK)],
                outputs.outgoing.at[pl.ds(dst_block * ROW_BLOCK,
                                          overlap * ROW_BLOCK)],
                sems.commit.at[slot],
                strict_ordering=False).start()
            if ctx.b.has_result_scales:
                pltpu.make_async_copy(
                    scratch.result_scales.at[slot,
                                             pl.ds(src_block, overlap)],
                    outputs.outgoing_scales.at[pl.ds(dst_block, overlap)],
                    sems.commit_scales.at[slot],
                    strict_ordering=False).start()


def stage_tile(ctx, slot, result, result_scales):
    """Stores a tile's finished rows (and scales) into result slot `slot`,
    the rows padded with zero blocks to the result width."""
    b = ctx.b
    result = result.reshape(b.tile_rows, b.lane_blocks, LANES)
    if b.result_blocks != b.lane_blocks:
        result = jnp.concatenate([
            result,
            jnp.zeros((b.tile_rows, b.result_blocks - b.lane_blocks, LANES),
                      result.dtype)
        ],
                                 axis=1)
    if b.has_result_scales:
        # The tile's scales, a [tile rows, 1] column, packed a block's
        # ROW_BLOCK words to a lane row: [tile blocks, 1, ROW_BLOCK].
        result_scales = result_scales.reshape(b.tile_blocks, 1, ROW_BLOCK)
    for i in range(b.result_slots):  # store slots are static

        @pl.when(slot == i)
        def _(i=i):
            ctx.scratch.result_rows[i] = result
            if b.has_result_scales:
                ctx.scratch.result_scales[i] = result_scales


def slot_and_pending(ctx, carry):
    """The result slot the next tile stages in and the block count its last
    commit is still draining (zero after a drain)."""
    tile_count, *pending = carry
    slot = lax.rem(tile_count, jnp.int32(ctx.b.result_slots))
    pending_here = pending[0]
    for i in range(1, ctx.b.result_slots):
        pending_here = jnp.where(slot == i, pending[i], pending_here)
    return slot, pending_here


def drain_commits(ctx, carry):
    """Wait for every result slot with commits still pending."""
    tile_count, *pending = carry
    for i in range(ctx.b.result_slots):

        @pl.when(pending[i] > 0)
        def _(i=i):
            wait_commits(ctx, i, pending[i])

    return (tile_count, ) + (jnp.int32(0), ) * ctx.b.result_slots


# ---- the arithmetic of one tile ----
def up_scales_and_bias(ctx, expert, weight_slot):
    """(w1_scales, w1_bias) of the expert, or of the column block in
    `weight_slot` for a streamed build: [1, cols] each (fp4 scales
    [blocks, cols]), None where absent."""
    b, scratch = ctx.b, ctx.scratch
    if b.stream_block:
        scales = scratch.w1_scales[weight_slot] if b.form.has_scales else None
        bias = scratch.w1_bias[weight_slot] if b.has_w1_bias else None
        return scales, bias
    scales = None
    if b.form.block_scaled:
        scales = scratch.w1_scales[expert]
    elif b.form.has_scales:
        scales = scratch.w1_scales[pl.ds(expert, 1), :]
    bias = scratch.w1_bias[pl.ds(expert, 1), :] if b.has_w1_bias else None
    return scales, bias


def down_product(ctx, expert, weight_slot, rows, row_scales):
    """One tile's rows through the FFN of the weights in `weight_slot` (an
    expert, or one column block of it): the down matmul's accumulator and,
    for the whole-row rounding, the intermediate row scales not yet
    applied."""
    b, scratch = ctx.b, ctx.scratch
    w1_scales, w1_bias = up_scales_and_bias(ctx, expert, weight_slot)
    # bf16 token rows against fp8 weights: the up matmul runs in bf16 on
    # the expert's widened copy of w1 with per-channel scales applied
    # after the matmul (block-scaled forms are unsupported).
    w1_widened = b.rows_bf16 and b.form.has_scales and not b.form.block_scaled
    w1_up = scratch.w1_bf16[...] if w1_widened else scratch.w1[weight_slot]
    if b.form.block_scaled:
        w1_block, w2_block = upcast_blocks(ctx, weight_slot)
        w2_scales = (scratch.w2_scales[weight_slot]
                     if b.stream_block else scratch.w2_scales[expert])
        return ffn.expert_ffn_fp4(
            rows,
            row_scales,
            w1_block,
            w2_block,
            w1_scales,
            w2_scales,
            block=b.weight_block,
            activation=b.activation,
            w1_bias=w1_bias,
            no_gate=b.no_gate,
            activation_block=b.activation_block,
            elementwise=b.elementwise,
            swiglu_limit=b.swiglu_limit,
            columns_per_pass=b.activation_columns_per_pass,
            token_block=b.token_block)
    if b.intermediate == "bf16":
        return ffn.expert_ffn_bf16_intermediate(
            rows,
            row_scales,
            w1_up,
            scratch.w2_bf16[...],
            w1_scales,
            activation=b.activation,
            w1_bias=w1_bias,
            no_gate=b.no_gate,
            swiglu_limit=b.swiglu_limit,
            token_block=b.token_block,
            weights_bf16=not b.form.has_scales)
    if not b.form.has_scales:
        return ffn.expert_ffn_bf16(
            rows,
            row_scales,
            scratch.w1[weight_slot],
            scratch.w2[weight_slot],
            activation=b.activation,
            w1_bias=w1_bias,
            no_gate=b.no_gate,
            block=b.activation_block,
            elementwise=b.elementwise,
            swiglu_limit=b.swiglu_limit,
            columns_per_pass=b.activation_columns_per_pass,
            token_block=b.token_block)
    return ffn.expert_ffn_fp8(rows,
                              row_scales,
                              w1_up,
                              scratch.w2[weight_slot],
                              w1_scales,
                              activation=b.activation,
                              w1_bias=w1_bias,
                              no_gate=b.no_gate,
                              block=b.activation_block,
                              elementwise=b.elementwise,
                              swiglu_limit=b.swiglu_limit,
                              columns_per_pass=b.activation_columns_per_pass,
                              token_block=b.token_block,
                              mask_non_finite=False)


def finish_rows(ctx, expert, acc2, mid_scales):
    """The down accumulator as result rows: the expert's down scales (fp8
    weights) and bias applied, then the rows quantized."""
    b, scratch = ctx.b, ctx.scratch
    w2_scales = (scratch.w2_scales[pl.ds(expert, 1), :]
                 if b.form.has_scales and not b.form.block_scaled else None)
    w2_bias = scratch.w2_bias[pl.ds(expert, 1), :] if b.has_w2_bias else None
    return ffn.result_rows(acc2,
                           mid_scales,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           dtype=b.result_dtype,
                           elementwise=b.elementwise,
                           mask_non_finite=False)


# ---- the push and the drains ----
def push_expert(ctx, expert):
    """Pushes the expert's runs to their shards. The own-shard run is
    copied locally (it is read only after the drain)."""
    b, tables, outputs, sems = ctx.b, ctx.tables, ctx.outputs, ctx.sems
    for d in range(b.num_shards):
        blocks = tables.push_blocks[expert, d]
        src_block = tables.outgoing_offset[expert, d]
        dst_block = tables.push_destination[expert, d]
        is_me = jnp.int32(d) == ctx.me

        @pl.when(and_nonempty(jnp.logical_not(is_me), blocks))
        def _():
            rows = tables.push_rows[expert, d]

            def push_run():
                pltpu.make_async_remote_copy(
                    src_ref=outputs.outgoing.at[pl.ds(
                        tables.outgoing_offset[expert, d] * ROW_BLOCK, rows)],
                    dst_ref=outputs.arrivals.at[pl.ds(
                        tables.arrival_offset[expert, d], rows)],
                    send_sem=sems.push_send,
                    recv_sem=sems.push_receive,
                    device_id=device_coords(jnp.int32(d), b.mesh_axes),
                    device_id_type=pl.DeviceIdType.MESH,
                    strict_ordering=False).start()

            pl.when(rows > 0)(push_run)  # a run can be empty
            if b.has_result_scales:
                pltpu.make_async_remote_copy(
                    src_ref=outputs.outgoing_scales.at[pl.ds(
                        src_block, blocks)],
                    dst_ref=outputs.arrival_scales.at[pl.ds(dst_block,
                                                            blocks)],
                    send_sem=sems.push_send_scales,
                    recv_sem=sems.push_receive_scales,
                    device_id=device_coords(jnp.int32(d), b.mesh_axes),
                    device_id_type=pl.DeviceIdType.MESH,
                    strict_ordering=False).start()

        @pl.when(and_nonempty(is_me, blocks))
        def _():
            pltpu.make_async_copy(
                outputs.outgoing.at[pl.ds(src_block * ROW_BLOCK,
                                          blocks * ROW_BLOCK)],
                outputs.arrivals.at[pl.ds(dst_block * ROW_BLOCK,
                                          blocks * ROW_BLOCK)],
                sems.local_hop,
                strict_ordering=False).start()
            if b.has_result_scales:
                pltpu.make_async_copy(
                    outputs.outgoing_scales.at[pl.ds(src_block, blocks)],
                    outputs.arrival_scales.at[pl.ds(dst_block, blocks)],
                    sems.local_hop_scales,
                    strict_ordering=False).start()


def drain_transport(ctx):
    """Consumes every push and the commits the per-tile waits left."""
    b, tables, outputs, sems = ctx.b, ctx.tables, ctx.outputs, ctx.sems
    wait_rows(sems.push_send, outputs.outgoing, count(ctx, COUNT_SEND_ROWS))
    wait_rows(sems.push_receive, outputs.arrivals,
              count(ctx, COUNT_RECEIVE_ROWS))
    if b.has_result_scales:
        wait_rows(sems.push_send_scales, outputs.outgoing_scales,
                  count(ctx, COUNT_SEND_ALIGNED_ROWS) // ROW_BLOCK)
        wait_rows(sems.push_receive_scales, outputs.arrival_scales,
                  count(ctx, COUNT_RECEIVE_ALIGNED_ROWS) // ROW_BLOCK)
    own_blocks = tables.push_blocks[0, ctx.me]
    for g in range(1, b.experts_per_shard):
        own_blocks = own_blocks + tables.push_blocks[g, ctx.me]
    wait_rows(sems.local_hop, outputs.arrivals, own_blocks * ROW_BLOCK)
    if b.has_result_scales:
        wait_rows(sems.local_hop_scales, outputs.arrival_scales, own_blocks)

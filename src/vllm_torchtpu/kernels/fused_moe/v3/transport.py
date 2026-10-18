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
"""The pallas mode's token exchange: one DMA program carried by three
kernels, on a power-of-two mesh of 1 to MAX_SHARDS shards.

Layout. Shards pair up: shard s and s ^ 1 are mates, and the even shard is
the pair's representative. The representatives form a hypercube of n =
log2(shards) - 1 dimensions, and dimension d joins representative r to
r ^ (2 << d). A pair's rows are one block of the gathered buffer, and every
block reaches every representative along shortest paths in n rounds. Round
1 sends the pair's own block on every link. Round t forwards each block
that arrived in round t - 1 one link further, so a block at distance t
arrives in t equal parts over its t shortest paths, the part on a path
fixed by the rank of the path's last link among the block's links. Every
link carries the same bytes in every round, C(n - 1, t - 1) / t blocks in
round t, and (2^n - 1) / n blocks in all, the least any all-gather over the
links can carry. A block travels as PIECES equal pieces so a forward starts
as soon as the piece it needs has landed; pieces on one link land in the
order they were issued.

Mates. During token preparation the representative and its mate exchange
their own rows a tile at a time. After each round the representative copies the
blocks that round brought to its mate, and the mate waits for those copies
before its first tile.

Kernels. start_transport is its own Pallas call behind the router: a
barrier, the shard's rows prepared a tile at a time in the configured wire
format and shared with the mate, the routing message sent to every shard, round 1
issued. It returns the gathered-rows buffer and its semaphores with the
copies in flight (the DMA engine keeps moving bytes across a call boundary
and the semaphores keep their counts). The shard-tables kernel forwards the
middle rounds on its last grid step (forward_middle_rounds). The FFN
kernel's prologue waits for every copy (finish) before its first tile.

The schedule is static: sends(), arrivals() and the wait bookkeeping are
plain Python over relative addresses, so a test simulates the whole exchange
without a device, and the kernels only turn relative addresses into peers
with one XOR of the shard index.
"""
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import layout
from .layout import (COLLECTIVE_ID_TRANSPORT, LANES, MESH_AXIS, ROW_BLOCK,
                     SUBLANES, device_coords)
from .rowquant import FP8, quantize_rows

# The maximum supported mesh width.
MAX_SHARDS = 32
# The routing message travels as dense (SUBLANES, LANES) tiles of int32.
MESSAGE_LANES = LANES
# Rows per quantization tile in the start call: a whole (LANES, LANES)
# transpose turns a tile's scale column into one routing message row.
QUANT_TILE_ROWS = LANES
# Receive semaphore 0 carries the mate's tiles during quantization.
MATE_TILE_SEMAPHORE = 0


def check_width(num_shards):
    """Raises ValueError unless the mesh width is a power of two in
    [1, MAX_SHARDS]."""
    if (num_shards < 1 or num_shards > MAX_SHARDS
            or num_shards & (num_shards - 1)):
        raise ValueError(
            f"the pallas mode's transport takes a power-of-two mesh of 1 to "
            f"{MAX_SHARDS} shards. This mesh has {num_shards} shards")


def dimensions(num_shards):
    """Hypercube dimensions of the representatives: log2(width) - 1, and 0
    for the single shard, which has no mate and no exchange."""
    check_width(num_shards)
    return max(num_shards.bit_length() - 2, 0)


def pieces(num_shards):
    """Pieces a block travels in: twice the least common multiple of the
    part counts 1..n, so every round's parts are whole pieces and round 1
    lands in halves. 2 for two shards, where no block travels."""
    n = dimensions(num_shards)
    return 2 * math.lcm(*range(1, n + 1)) if n else 2


def receive_semaphores(num_shards):
    """The mate's tile semaphore, one per (round, link), one per round of
    mate copies."""
    n = dimensions(num_shards)
    return 1 + n * n + n


def token_multiple(num_shards):
    """Tokens per shard divide into pieces only when they are a multiple of
    this: half the piece count of the width."""
    return pieces(num_shards) // 2


def blocks_at(n, distance):
    """Relative addresses (n bits, one per dimension) of the pairs `distance`
    links from a representative, ascending."""
    return [r for r in range(1, 1 << n) if bin(r).count("1") == distance]


def _rank(r, link):
    """The rank of `link` among the dimensions of block r (which has it)."""
    return [d for d in range(n_bits(r)) if r >> d & 1].index(link)


def n_bits(r):
    """The bit length of r."""
    return r.bit_length()


def sends(n, t, n_pieces):
    """Round t of an n-dimensional exchange, from the sender's side:
    (link, relative block, pieces) in issue order. Round 1 sends the own
    block (relative 0) whole on every link. Round t >= 2 forwards every
    block at distance t - 1 on every link it has not travelled, as the part
    ranked by that link among the destination's links, of t parts."""
    if t == 1:
        return [(link, 0, range(n_pieces)) for link in range(n)]
    out = []
    for link in range(n):
        for r in blocks_at(n, t - 1):
            if r >> link & 1:
                continue
            rank = _rank(r | 1 << link, link)
            width = n_pieces // t
            out.append((link, r, range(rank * width, (rank + 1) * width)))
    return out


def arrivals(n, t, n_pieces, link):
    """Round t pieces landing on `link`, in landing order: (relative block,
    piece). The neighbour's relative addresses are this shard's with the
    link's bit flipped."""
    return [(r ^ 1 << link, p)
            for (sender_link, r, rng) in sends(n, t, n_pieces)
            if sender_link == link for p in rng]


def landing_link(r, p, n_pieces):
    """The link piece p of block r (at distance popcount(r)) landed on: the
    link ranked by the part the piece belongs to."""
    distance = bin(r).count("1")
    links = [d for d in range(n_bits(r)) if r >> d & 1]
    return links[p // (n_pieces // distance)]


class Exchange:
    """Every slice, peer and semaphore of one shard's part of the exchange,
    from the traced shard index."""

    def __init__(self,
                 gathered_rows,
                 send_sem,
                 receive_sems,
                 shard,
                 rows_per_shard,
                 num_shards,
                 mesh_axes=(MESH_AXIS, )):
        self.mesh_axes = mesh_axes
        self.n = dimensions(num_shards)
        self.n_pieces = pieces(num_shards)
        self.gathered_rows, self.send_sem, self.receive_sems = (gathered_rows,
                                                                send_sem,
                                                                receive_sems)
        self.shard, self.rows = shard, rows_per_shard
        self.mate = shard ^ 1
        self.representative = shard & jnp.int32(-2)
        self.block_rows = 2 * rows_per_shard
        if self.block_rows % self.n_pieces:
            raise ValueError(
                f"{rows_per_shard} rows per shard do not split into "
                f"{self.n_pieces} pieces per block")
        self.piece_rows = self.block_rows // self.n_pieces

    def peer(self, link):
        """The shard reached over the given link."""
        return self.representative ^ (2 << link)

    def block_start(self, relative):
        """The first row of the block a relative shard's rows occupy."""
        return (self.representative ^ (relative << 1)) * self.rows

    def block(self, relative):
        """Rows of the block `relative` links away (0 = the own pair)."""
        return pl.ds(self.block_start(relative), self.block_rows)

    def piece(self, relative, p):
        """The row slice of piece p of a relative shard's block."""
        return pl.ds(
            self.block_start(relative) + p * self.piece_rows, self.piece_rows)

    def round_sem(self, t, link):
        """The receive semaphore of round t on a link."""
        return self.receive_sems.at[1 + (t - 1) * self.n + link]

    def mate_sem(self, t):
        """The receive semaphore of round t's mate exchange."""
        return self.receive_sems.at[1 + self.n * self.n + (t - 1)]

    def remote_copy(self, rows, sem, peer, src=None):
        """gathered rows [rows] to the same rows on `peer` (or from `src`)."""
        return pltpu.make_async_remote_copy(
            src_ref=self.gathered_rows.at[rows] if src is None else src,
            dst_ref=self.gathered_rows.at[rows],
            send_sem=self.send_sem,
            recv_sem=sem,
            device_id=device_coords(peer, self.mesh_axes),
            device_id_type=pl.DeviceIdType.MESH,
            strict_ordering=False)

    def wait_landed(self, rows, sem):
        """Wait for `rows` of the gathered rows to land on `sem`. Under the
        torch bridge (layout.SHARD_INDEX_OPERAND) the wait goes on a local
        descriptor of the same rows and semaphore: the bridge's compile
        refuses a wait on a descriptor that
        carries a peer address ("Number of device ids must match the number
        of mesh axes ... 0D mesh"). Other callers use the remote
        descriptor's receive wait."""
        if layout.SHARD_INDEX_OPERAND:
            pltpu.make_async_copy(self.gathered_rows.at[rows],
                                  self.gathered_rows.at[rows],
                                  sem,
                                  strict_ordering=False).wait()
        else:
            self.remote_copy(rows, sem, self.mate).wait_recv()

    def representative_only(self, body):
        """Run body on the even shards only."""
        pl.when((self.shard & 1) == 0)(body)

    def mate_only(self, body):
        """Run body on the odd shards only."""
        pl.when((self.shard & 1) == 1)(body)


class _Landed:
    """Trace-time bookkeeping of the pieces waited for, per round and link,
    in landing order."""

    def __init__(self, exchange):
        self.x = exchange
        self.waited = {}

    def wait(self, t, link, count):
        """Waits until `count` pieces of round t have landed on `link`."""
        order = arrivals(self.x.n, t, self.x.n_pieces, link)
        done = self.waited.get((t, link), 0)
        for r, p in order[done:count]:
            self.x.wait_landed(self.x.piece(r, p), self.x.round_sem(t, link))
        self.waited[(t, link)] = max(done, count)

    def ensure(self, t, r, p):
        """Waits until piece p of block r, landed in round t, is there."""
        link = landing_link(r, p, self.x.n_pieces)
        order = arrivals(self.x.n, t, self.x.n_pieces, link)
        self.wait(t, link, order.index((r, p)) + 1)

    def drain(self, t):
        """Waits for every piece of round t."""
        for link in range(self.x.n):
            self.wait(t, link, len(arrivals(self.x.n, t, self.x.n_pieces,
                                            link)))


def issue_first_round(x):
    """Start call: the pair's block to every neighbour, piece by piece."""
    if not x.n:
        return

    @x.representative_only
    def _():
        for link, r, rng in sends(x.n, 1, x.n_pieces):
            for p in rng:
                x.remote_copy(x.piece(r, p), x.round_sem(1, link),
                              x.peer(link)).start()


def forward_middle_rounds(x):
    """Shard-tables kernel: rounds 2..n, each piece forwarded as it lands,
    and the blocks of rounds 1..n-1 copied to the mate once whole."""
    if x.n < 2:
        return

    @x.representative_only
    def _():
        landed = _Landed(x)
        for t in range(2, x.n + 1):
            for link, r, rng in sends(x.n, t, x.n_pieces):
                for p in rng:
                    landed.ensure(t - 1, r, p)
                    x.remote_copy(x.piece(r, p), x.round_sem(t, link),
                                  x.peer(link)).start()
            landed.drain(t - 1)
            for r in blocks_at(x.n, t - 1):
                x.remote_copy(x.block(r), x.mate_sem(t - 1), x.mate).start()


def finish(x):
    """FFN kernel prologue: the last round's arrivals, their copies to the
    mate, every send still in flight; the mate waits for its copies."""
    if not x.n:
        return

    @x.representative_only
    def _():
        landed = _Landed(x)
        landed.drain(x.n)
        for r in blocks_at(x.n, x.n):
            x.remote_copy(x.block(r), x.mate_sem(x.n), x.mate).start()
        for t in range(1, x.n + 1):
            for link, r, rng in sends(x.n, t, x.n_pieces):
                for p in rng:
                    x.remote_copy(x.piece(r, p), x.round_sem(t, link),
                                  x.peer(link)).wait_send()
            for r in blocks_at(x.n, t):
                x.remote_copy(x.block(r), x.mate_sem(t), x.mate).wait_send()

    @x.mate_only
    def _():
        for t in range(1, x.n + 1):
            for r in blocks_at(x.n, t):
                x.wait_landed(x.block(r), x.mate_sem(t))


def finish_call(gathered_rows,
                send_sem,
                receive_sems,
                after,
                *,
                num_shards,
                rows_per_shard,
                mesh_axes=(MESH_AXIS, ),
                shard_index=None):
    """The exchange's finish as a program of its own: waits the landing of
    every arrival on this shard and the mate copies, and returns a
    completion word ([1, LANES] int32, ones). The FFN kernel does this in
    its prologue when it takes the landing buffer itself; a build that
    reads sorted rows needs the landing complete before the sort's gather
    reads the buffer, and the gather's index is made to depend on the
    word. The buffer is read here, never returned: returning it aliased
    would have the compiler snapshot it, and the snapshot is taken before
    the arrivals land. `after` is an array the call only depends on: the
    layer hands in a slice of the scatter's output, so the finish runs
    once the scatter's wait is over, by which time the arrivals have
    landed and the finish waits for nothing."""

    def kernel(*refs):
        if shard_index is not None:
            shard_ref, *refs = refs
            shard = shard_ref[0]
        else:
            shard = lax.axis_index(MESH_AXIS)
        rows_ref, send_ref, receive_ref, after_ref, done_ref = refs
        del after_ref  # an ordering operand
        finish(
            Exchange(rows_ref,
                     send_ref,
                     receive_ref,
                     shard,
                     rows_per_shard,
                     num_shards,
                     mesh_axes=mesh_axes))
        done_ref[...] = jnp.ones((1, LANES), jnp.int32)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, LANES), jnp.int32),
        in_specs=([pl.BlockSpec(
            memory_space=pltpu.SMEM)] if shard_index is not None else []) +
        [hbm, sem, sem, hbm],
        out_specs=vmem,
        compiler_params=pltpu.CompilerParams(
            has_side_effects=layout.SIDE_EFFECTS),
        name="fused_ep_transport_finish",
    )(*(() if shard_index is None else (shard_index, )), gathered_rows,
      send_sem, receive_sems, after)


def all_shards_barrier(num_shards, mesh_axes=(MESH_AXIS, )):
    """A barrier over every shard (peers are not only neighbours)."""
    barrier = pltpu.get_barrier_semaphore()
    for i in range(num_shards):
        pl.semaphore_signal(barrier,
                            inc=1,
                            device_id=device_coords(jnp.int32(i), mesh_axes),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, num_shards)


def scale_row_offset(pairs_per_shard):
    """The routing message row where a shard's row scales begin: after its
    routing indices (words [0, pairs_per_shard), slot-major, the padding
    pairs past tokens * top_k routed nowhere), on the next whole row."""
    return -(-pairs_per_shard // MESSAGE_LANES)


def scale_planes(hidden, activation_block):
    """Scale words per token row: one, or hidden // activation_block."""
    return hidden // activation_block if activation_block else 1


def message_rows(tokens_per_shard, pairs_per_shard, n_scales=1):
    """Rows of the [rows, MESSAGE_LANES] int32 routing message one shard
    sends: the routing indices of `pairs_per_shard` pairs (tokens * top_k
    real ones, the rest routed nowhere), then one row per (scale plane,
    quantization tile) holding that tile's tokens' scale words,
    plane-major; padded to whole (SUBLANES, LANES) tiles so each shard's
    block of the gathered rows is a tile-aligned slice."""
    rows = scale_row_offset(
        pairs_per_shard) + n_scales * quant_tiles(tokens_per_shard)
    return -(-rows // SUBLANES) * SUBLANES


def quant_tile_rows(tokens_per_shard):
    """Rows per quantization tile: QUANT_TILE_ROWS, or the shard's whole
    block when that is smaller. The last tile of a shard whose rows are
    not a whole number of tiles is partial."""
    return min(tokens_per_shard, QUANT_TILE_ROWS)


def quant_tiles(tokens_per_shard):
    """Quantization tiles a shard's rows take, the last one partial when
    the rows are not a whole number of QUANT_TILE_ROWS."""
    return -(-tokens_per_shard // quant_tile_rows(tokens_per_shard))


def start_transport(rows,
                    routing_message,
                    *,
                    scale_base,
                    num_shards,
                    scale_dtype=jnp.float32,
                    activation_block=0,
                    elementwise=jnp.float32,
                    mesh_axes=(MESH_AXIS, ),
                    token_dtype=FP8,
                    shard_index=None):
    """Quantizes the shard's rows, issues the exchange, exchanges the
    messages.

    Args:
      rows: [tokens_per_shard, hidden] bf16, the shard's token rows.
      routing_message: [message_rows, MESSAGE_LANES] int32 with the routing
        indices in place and the scale rows (from `scale_base` on) zero.
      scale_base: the routing message row the scales are written from.
      num_shards: the mesh width, a power of two in [1, MAX_SHARDS]. One
        shard quantizes its rows and exchanges nothing.
      mesh_axes: the mesh's axis names; MESH_AXIS is the expert-parallel
        one, the others hold replicas.
      token_dtype: the rows' element type on the wire, fp8 (quantized here,
        one scale per row or per activation block) or bf16 (as they are,
        unit scales).
      scale_dtype: the row scales' rounding.
      activation_block: values per fp8 scale of a row (0: one scale per
        row). Plane s of a tile's scales is message row
        scale_base + s * tiles + tile.
      elementwise: the element type of the scaling product
        (Config.elementwise_dtype).

    Returns:
      (gathered rows [num_shards * tokens, lane blocks, LANES] token_dtype with the
      copies still landing, its send semaphore, its receive semaphores,
      routing messages [num_shards, message_rows, MESSAGE_LANES] int32).

    Each tile's rows go to this shard's slot of the gathered rows and
    to the mate's, and its scale column becomes one routing message row.
    The routing message goes out to every peer ahead of round 1 in every
    link's queue (issued behind it, it would land behind round 1's
    megabytes), then round 1, then the routing message waits.
    """
    check_width(num_shards)
    tokens, hidden = rows.shape
    n_message_rows, message_lanes = routing_message.shape
    lane_blocks = hidden // MESSAGE_LANES
    tile = quant_tile_rows(tokens)
    n_tiles = quant_tiles(tokens)
    if tokens % ROW_BLOCK:
        raise ValueError(
            f"{tokens} tokens per shard is not a whole number of "
            f"{ROW_BLOCK}-row blocks, the unit the exchange moves")

    def rows_of(i):
        """Rows of tile i: a whole tile, or what is left on the last one."""
        return min(tile, tokens - i * tile)

    n_scales = scale_planes(hidden, activation_block)
    if scale_base + n_scales * n_tiles > n_message_rows:
        raise ValueError(f"the routing message has {n_message_rows} rows and "
                         f"the scales need rows {scale_base} to "
                         f"{scale_base + n_scales * n_tiles}")
    # Streamed from HBM a tile at a time. Without the constraint XLA stages
    # the whole block in its alternate memory first.
    rows = pltpu.with_memory_space_constraint(rows, pltpu.MemorySpace.HBM)

    def kernel(*refs):
        # shard_index: this shard's index as an SMEM operand (layout.
        # SHARD_INDEX_OPERAND), else read from the mesh axis.
        if shard_index is not None:
            shard_ref, *refs = refs
            shard = shard_ref[0]
        else:
            shard = lax.axis_index(MESH_AXIS)
        (rows_hbm, message_in, gathered_rows, send_sem, receive_sems,
         routing_messages, rows_vmem, quantized_vmem, row_fetch_sems,
         gathered_rows_write_sems, mate_send_sems, message_send_sem,
         message_receive_sem, message_local_sem) = refs
        exchange = Exchange(gathered_rows,
                            send_sem,
                            receive_sems,
                            shard,
                            tokens,
                            num_shards,
                            mesh_axes=mesh_axes)
        all_shards_barrier(num_shards, mesh_axes)

        def fetch_rows(i):
            return pltpu.make_async_copy(rows_hbm.at[pl.ds(
                i * tile, rows_of(i))],
                                         rows_vmem.at[i % 2,
                                                      pl.ds(0, rows_of(i))],
                                         row_fetch_sems.at[i % 2],
                                         strict_ordering=False)

        def write_gathered_rows(i):
            return pltpu.make_async_copy(
                quantized_vmem.at[i % 2, pl.ds(0, rows_of(i))],
                gathered_rows.at[pl.ds(shard * tokens + i * tile, rows_of(i))],
                gathered_rows_write_sems.at[i % 2],
                strict_ordering=False)

        def share_with_mate(i):
            return pltpu.make_async_remote_copy(
                src_ref=quantized_vmem.at[i % 2, pl.ds(0, rows_of(i))],
                dst_ref=gathered_rows.at[pl.ds(shard * tokens + i * tile,
                                               rows_of(i))],
                send_sem=mate_send_sems.at[i % 2],
                recv_sem=receive_sems.at[MATE_TILE_SEMAPHORE],
                device_id=device_coords(exchange.mate, mesh_axes),
                device_id_type=pl.DeviceIdType.MESH,
                strict_ordering=False)

        def send_message(peer):
            return pltpu.make_async_remote_copy(
                src_ref=message_in,
                dst_ref=routing_messages.at[shard],
                send_sem=message_send_sem,
                recv_sem=message_receive_sem,
                device_id=device_coords(peer, mesh_axes),
                device_id_type=pl.DeviceIdType.MESH,
                strict_ordering=False)

        fetch_rows(0).start()
        for i in range(n_tiles):
            if i + 1 < n_tiles:
                fetch_rows(i + 1).start()
            fetch_rows(i).wait()
            if jnp.dtype(token_dtype) == jnp.dtype(jnp.bfloat16):
                # bf16 token rows: as they are, unit scales.
                quantized = rows_vmem[i % 2].astype(jnp.bfloat16)
                scale = jnp.ones((rows_of(i), n_scales), jnp.float32)
            else:
                quantized, scale = quantize_rows(rows_vmem[i % 2],
                                                 scale_dtype,
                                                 block=activation_block,
                                                 dtype=elementwise,
                                                 mask_non_finite=False)
            if i >= 2:
                write_gathered_rows(i - 2).wait()
                if num_shards > 1:
                    share_with_mate(i - 2).wait_send()
            quantized_vmem[i % 2] = quantized.reshape(tile, lane_blocks,
                                                      MESSAGE_LANES)
            write_gathered_rows(i).start()
            if num_shards > 1:
                share_with_mate(i).start()
            for plane in range(n_scales):
                column = jnp.broadcast_to(scale[:, plane:plane + 1],
                                          (tile, MESSAGE_LANES))
                if tile < QUANT_TILE_ROWS:
                    column = jnp.concatenate([
                        column,
                        jnp.zeros((QUANT_TILE_ROWS - tile, MESSAGE_LANES),
                                  column.dtype)
                    ],
                                             axis=0)
                row = scale_base + plane * n_tiles + i
                message_in[row:row + 1, :] = pltpu.bitcast(
                    column.T[0:1, :], jnp.int32)
        local_message = pltpu.make_async_copy(message_in,
                                              routing_messages.at[shard],
                                              message_local_sem,
                                              strict_ordering=False)
        local_message.start()
        for i in range(1, num_shards):
            send_message((shard + i) & (num_shards - 1)).start()
        for i in range(max(0, n_tiles - 2), n_tiles):
            write_gathered_rows(i).wait()
            if num_shards > 1:
                share_with_mate(i).wait_send()
        for i in range(n_tiles if num_shards > 1 else 0):
            if layout.SHARD_INDEX_OPERAND:
                pltpu.make_async_copy(
                    gathered_rows.at[pl.ds(shard * tokens + i * tile,
                                           rows_of(i))],
                    gathered_rows.at[pl.ds(shard * tokens + i * tile,
                                           rows_of(i))],
                    receive_sems.at[MATE_TILE_SEMAPHORE],
                    strict_ordering=False).wait()
            else:
                share_with_mate(i).wait_recv()
        issue_first_round(exchange)
        for _ in range(1, num_shards):
            if layout.SHARD_INDEX_OPERAND:
                pltpu.make_async_copy(routing_messages.at[shard],
                                      routing_messages.at[shard],
                                      message_receive_sem,
                                      strict_ordering=False).wait()
            else:
                send_message(shard ^ 1).wait_recv()
        for _ in range(1, num_shards):
            send_message(shard ^ 1).wait_send()
        local_message.wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(
            (num_shards * tokens, lane_blocks, MESSAGE_LANES),
            token_dtype), pltpu.SemaphoreType.DMA(
                ()), pltpu.SemaphoreType.DMA(
                    (receive_semaphores(num_shards), )),
                   jax.ShapeDtypeStruct(
                       (num_shards, n_message_rows, message_lanes),
                       jnp.int32)),
        in_specs=([pl.BlockSpec(
            memory_space=pltpu.SMEM)] if shard_index is not None else []) +
        [hbm, vmem],
        out_specs=(hbm, sem, sem, hbm),
        scratch_shapes=[
            pltpu.VMEM((2, tile, hidden), rows.dtype),
            pltpu.VMEM((2, tile, lane_blocks, MESSAGE_LANES), token_dtype),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID_TRANSPORT,
            has_side_effects=layout.SIDE_EFFECTS),
        name="fused_ep_transport_start",
    )(*(() if shard_index is None else (shard_index, )), rows, routing_message)

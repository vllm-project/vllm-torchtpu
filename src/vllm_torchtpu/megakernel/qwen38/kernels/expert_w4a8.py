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
"""Unqualified Group32 follow-up; Group16/8 fallbacks and every Ref boundary retained.

Two gate/up DMA slots overlap successive experts. A single down slot is reused
only after all MXU reads complete. ETP collectives carry FP32, and quantization
happens after the complete intermediate absmax / down sum, including ETP8.
Residual reduction instead quantizes each rank's intermediate slice locally and
adds its FP32 down partial unquantized; the layer's residual sum combines them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

F32 = jnp.float32
BF16 = jnp.bfloat16
FP8 = jnp.float8_e4m3fn


def rowquant_fp8(value, amax=None):
    """Match the pinned BF16 reciprocal multiplication and FP8 cast exactly."""
    if amax is None:
        amax = jnp.max(jnp.abs(value), axis=-1, keepdims=True)
    scale = amax.astype(F32) / 448.0
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    inverse = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    quantized = jnp.where(usable, value * inverse.astype(value.dtype), jnp.zeros_like(value))
    return quantized.astype(FP8), scale


def stacks_blocks(layout):
    """Whether run_experts contracts 32 // rows K64 blocks per 32-row FP8 LHS tile."""
    return (
        not layout.expert_output_tile
        and layout.config.dim % 512 == 0
        and layout.expert_width % 512 == 0
    )


def row_tiles(layout, rows):
    """Eight-row tiles run_experts contracts separately: 1, or rows // 8 for verification.

    Each expert then contracts only the tiles holding one of its routed rows. Sixteen rows
    gain nothing: one padded contraction costs less than a tile's plus its bookkeeping.
    """
    tiled = rows > 16 and layout.expert_reduction == "residual" and stacks_blocks(layout)
    return rows // 8 if tiled else 1


def tiles_per_pass(tiles):
    """Row tiles a verification pass contracts against one weight unpack.

    An unpack costs several times a tile's scale epilogue, so a pass takes up to two
    tiles, or four once a visit commonly holds three or more (eight tiles).
    """
    return 4 if tiles >= 8 else 2


def weight_slots(rows):
    """Routed-expert gate/up and DOWN weight slots: the current expert's and later ones'.

    Transfers start two experts ahead, so each one's issue latency hides behind the
    expert before; sixty-four padded rows, with four-tile passes, leave VMEM for DOWN
    transfers one ahead.
    """
    return (3, 2) if rows > 32 else (3, 3)


def scratch_shapes(layout, batch_size, rows=8):
    """Scoped VMEM and semaphores of run_experts, for callers opening the scope early."""
    width, hidden = layout.expert_width, layout.config.dim
    peers = 1 if layout.expert_reduction == "residual" else layout.expert_tp
    compact = batch_size == 1 and hidden % (rows * 128) == 0
    packet_columns = hidden // rows if compact else hidden
    tiles = row_tiles(layout, rows)
    lhs_rows = rows // tiles

    def vm(shape, dtype=F32):
        return tpu.VMEM(shape, dtype)

    def dm(count):
        return tpu.SemaphoreType.DMA((max(1, count),))

    passes = (tiles_per_pass(tiles),) if tiles > 1 else ()
    scratch = dict(
        gu=vm((weight_slots(rows)[0], hidden // 8, 2 * width), jnp.uint32),
        gus=vm((weight_slots(rows)[0], hidden // 64, 2 * width)),
        down=vm((weight_slots(rows)[1], width // 8, hidden), jnp.uint32),
        downs=vm((weight_slots(rows)[1], width // 64, hidden)),
        dma=dm(2 * sum(weight_slots(rows))),
        xq=vm((lhs_rows, hidden), FP8),
        middle=vm((lhs_rows, width), BF16),
        midq=vm((lhs_rows, width), FP8),
        gateup=vm((*passes, lhs_rows, 2 * width)),
        projection=vm((*passes, lhs_rows, hidden)),
        send=dm(peers - 1),
        receive=dm(peers - 1),
        barrier=tpu.SemaphoreType.REGULAR,
    )
    if peers > 1:
        # Expert-TP exchanges; residual reduction leaves partials to the layer's sum.
        scratch.update(
            amax_packet=vm((peers, rows, 128)),
            down_packet=vm((peers, rows, packet_columns)),
        )
    if stacks_blocks(layout):
        # An FP8 LHS tile holds 32 rows; each contraction stacks 32 // rows K64 blocks.
        stacked_k = 64 * (32 // lhs_rows)
        stacked_shape = (hidden // stacked_k, 32, stacked_k)
        scratch.update(
            stacked_x=vm((tiles, *stacked_shape) if tiles > 1 else stacked_shape, FP8),
            stacked_mid=vm((*passes, width // stacked_k, 32, stacked_k), FP8),
        )
    if tiles > 1:
        # Per-tile input scales and one expert's routing probabilities, by lane 0.
        scratch.update(
            row_scale=vm((tiles, lhs_rows, 128)),
            row_prob=vm((tiles, lhs_rows, 128)),
        )
    return scratch


def _transfers(weights_at_layer, scratch, local, slot, down_slot=None):
    """One local expert's GU transfers into a slot and DOWN transfers into a DOWN slot."""
    dma = scratch["dma"]
    down_slot = slot if down_slot is None else down_slot
    first_down = 2 * scratch["gu"].shape[0]
    gate_up = (
        tpu.make_async_copy(
            weights_at_layer["egu"].at[local], scratch["gu"].at[slot], dma.at[slot * 2]
        ),
        tpu.make_async_copy(
            weights_at_layer["egu_scale"].at[local], scratch["gus"].at[slot], dma.at[slot * 2 + 1]
        ),
    )
    down = (
        tpu.make_async_copy(
            weights_at_layer["edw"].at[local],
            scratch["down"].at[down_slot],
            dma.at[first_down + down_slot * 2],
        ),
        tpu.make_async_copy(
            weights_at_layer["edw_scale"].at[local],
            scratch["downs"].at[down_slot],
            dma.at[first_down + down_slot * 2 + 1],
        ),
    )
    return gate_up, down


class EarlyStart:
    """Start the first local expert's transfers at the top-k step that selects it.

    Call once per top-k step with that step's selected expert for every padded
    row. ``first`` is then the started expert, or the expert count if no active
    row selected a local expert; pass it to run_experts as ``started``. An optional
    [rows, 1] ``live`` mask replaces ``row < batch_size`` as the rows that may choose.
    With expert overlap only experts the successor group cannot take are started.
    """

    def __init__(self, weights_at_layer, scratch, rank, layout, batch_size, live=None):
        self.weights, self.scratch, self.batch_size = weights_at_layer, scratch, batch_size
        self.live = live
        self.experts = layout.config.experts
        self.start = rank // layout.expert_tp * layout.local_experts
        self.end = self.start + layout.local_experts
        self.kept = self.start + layout.expert_overlap
        self.first = jnp.int32(self.experts)

    def __call__(self, selected):
        selected = selected[:, None]
        live = self.live
        if live is None:
            live = jnp.arange(selected.shape[0])[:, None] < self.batch_size
        local = live & (selected >= self.kept) & (selected < self.end)
        candidate = jnp.min(jnp.where(local, selected, self.experts))
        pending = self.first == self.experts

        @pl.when(pending & (candidate < self.experts))
        def start_first():
            gate_up, down = _transfers(self.weights, self.scratch, candidate - self.start, 0)
            for transfer in (*gate_up, *down):
                transfer.start()

        self.first = jnp.where(pending, candidate, self.first)


def overlap_cuts(route_ids, active, rank, layout):
    """Expert IDs at which this rank's group gives and takes overlapping experts.

    Every rank derives the same assignment from all rows' routes. Group g may hand
    its routed experts among IDs [start_g, start_g + overlap) to group g + 1
    (cyclically); the moves minimize the largest number of distinct experts a group
    visits, then the moves out of the last group, when that maximum is at most fifteen
    above the even share (otherwise nothing moves). Group g gives its first ``give``
    routed overlapping experts in ID order and takes its predecessor's first
    ``take``. Returns (give_cut, take_cut): own experts below give_cut and the
    predecessor's below take_cut change group.
    """
    c, groups, width = layout.config, layout.ep, layout.local_experts
    overlap = layout.expert_overlap
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, c.experts), 1)
    seen = route_ids[:, :1] == lanes
    for k in range(1, route_ids.shape[1]):
        seen |= route_ids[:, k : k + 1] == lanes
    hit = jnp.max(jnp.where(seen & active, 1, 0), axis=0, keepdims=True)
    # Row g counts group g's routed experts, row groups + g its overlapping ones.
    row = jax.lax.broadcasted_iota(jnp.int32, (2 * groups, c.experts), 0)
    column = jax.lax.broadcasted_iota(jnp.int32, (2 * groups, c.experts), 1)
    counted = (column // width == row % groups) & ((row < groups) | (column % width < overlap))
    sums = jnp.sum(jnp.where(counted, hit, 0), axis=1, keepdims=True)
    counts = [sums[g, 0] for g in range(groups)]
    movable = [sums[groups + g, 0] for g in range(groups)]
    # Rows: candidate maxima from the even share up; lanes: moves out of the last group
    # into group 0. Without a feasible candidate nothing moves.
    floor = (sum(counts) + groups - 1) // groups
    maximum = floor + jax.lax.broadcasted_iota(jnp.int32, (16, 128), 0)
    closing = jax.lax.broadcasted_iota(jnp.int32, (16, 128), 1)
    feasible = closing <= movable[-1]
    moved = closing
    for g in range(groups - 1):
        moved = jnp.maximum(0, counts[g] + moved - maximum)
        feasible &= moved <= movable[g]
    feasible &= counts[-1] + moved - closing <= maximum
    limit = 1 << 20
    best = jnp.min(jnp.where(feasible, maximum * 128 + closing, limit))
    target, last = best // 128, jnp.where(best < limit, best % 128, 0)
    moves, moved = [], last
    for g in range(groups - 1):
        moved = jnp.where(best < limit, jnp.maximum(0, counts[g] + moved - target), 0)
        moves.append(moved)
    moves.append(last)
    group = rank // layout.expert_tp
    predecessor = (group + groups - 1) % groups
    give, take = jnp.int32(0), jnp.int32(0)
    for g in range(groups):
        give = jnp.where(group == g, moves[g], give)
        take = jnp.where(predecessor == g, moves[g], take)

    def cut(first, count):
        # One past the count-th routed expert in [first, first + overlap).
        def step(_, carry):
            remaining, _ = carry
            least = jnp.min(jnp.where(remaining > 0, lanes, c.experts))
            return jnp.where(lanes == least, 0, remaining), least + 1

        window = (lanes >= first) & (lanes < first + overlap)
        return jax.lax.fori_loop(0, count, step, (jnp.where(window, hit, 0), first))[1]

    start = group * width
    return cut(start, give), cut(predecessor * width, take)


def run_experts(
    weights_at_layer,
    normed,
    route_ids,
    route_probs,
    output_moe,
    rank,
    layout,
    batch_size,
    *,
    id_type=pl.DeviceIdType.MESH,
    shared_contribution=None,
    scratch=None,
    started=None,
    live=None,
):
    """Accumulate local routed experts into an existing FP32 shared contribution.

    Each aligned ETP group follows the same ascending expert order. All group
    members participate in both collectives; only its leader adds the complete
    rounded expert contribution. Completion barriers protect inbox reuse. With
    residual reduction there are no collectives and every member adds its partial.

    If supplied, shared_contribution() initializes output_moe exactly once after
    the first expert's GU/down transfers start, including on empty-route ranks.
    It may use scoped scratch but must preserve normed, route_ids and route_probs.
    Its DMA operations must complete before it returns. With no callback, the
    caller supplies the existing shared contribution as before.

    A caller that opened scratch_shapes() itself passes ``scratch``. With
    ``started`` from EarlyStart, the first expert's transfers are already in
    flight and experts follow top-k selection order instead of ascending IDs.
    An optional [rows, 1] ``live`` mask replaces ``row < batch_size`` as the rows
    that reach their routed experts.
    """
    if not layout.fast or layout.expert_tp not in (1, 4, 8):
        raise ValueError("W4A8 expert execution supports expert_tp=1, 4 or 8")
    c, etp, width = layout.config, layout.expert_tp, layout.expert_width
    residual = layout.expert_reduction == "residual"
    peers = 1 if residual else etp
    rows, hidden = normed.shape
    if (rows not in (8, 16, 32, 64) or batch_size not in (1, 2, 4, 8, 16, 32, 64)
            or batch_size > rows):
        raise ValueError("Expert scratch requires 8, 16, 32 or 64 padded rows and batch 1-64")
    if hidden != c.dim:
        raise ValueError("Expert rows must hold the model width")
    if started is not None and (scratch is None or not residual):
        raise ValueError("Early-started experts require caller scratch and residual reduction")
    compact = batch_size == 1 and hidden % (rows * 128) == 0
    packet_columns = hidden // rows if compact else hidden
    top_k = route_ids.shape[1]

    def peer_id(value):
        return value if id_type == pl.DeviceIdType.LOGICAL else (value,)

    def compute(
        gu,
        gus,
        down,
        downs,
        dma,
        xq,
        middle,
        midq,
        gateup,
        projection,
        send,
        receive,
        barrier,
        amax_packet=None,
        down_packet=None,
        stacked_x=None,
        stacked_mid=None,
        row_scale=None,
        row_prob=None,
    ):
        first_expert = rank // etp * layout.local_experts
        active = jnp.arange(rows)[:, None] < batch_size if live is None else live
        own = (route_ids[...] >= first_expert) & (
            route_ids[...] < first_expert + layout.local_experts
        )
        if layout.expert_overlap:
            give_cut, take_cut = overlap_cuts(route_ids[...], active, rank, layout)
            previous = (rank // etp + layout.ep - 1) % layout.ep * layout.local_experts
            taken = (route_ids[...] >= previous) & (route_ids[...] < take_cut)
            valid = active & ((own & (route_ids[...] >= give_cut)) | taken)
        else:
            previous = None
            valid = active & own
        route_ids[...] = jnp.where(valid, route_ids[...], c.experts)

        def held(expert):
            # Taken experts occupy the slots past the group's own.
            if previous is None:
                return expert - first_expert
            return jnp.where(
                (expert >= first_expert) & (expert < first_expert + layout.local_experts),
                expert - first_expert,
                layout.local_experts + expert - previous,
            )

        lhs_rows = middle.shape[0]
        tiles = rows // lhs_rows
        if tiles == 1:
            quantized_input, input_scale = rowquant_fp8(normed[...].astype(BF16))
            xq[...] = quantized_input
        groups = 32 // lhs_rows
        stacked = stacked_x is not None

        def stack_blocks(source, destination):
            # Row group g of slice i holds source's K64 block i * groups + g on its own
            # columns and zeros elsewhere: one contraction yields every block's product.
            width_k = destination.shape[-1]
            if groups == 1:
                # Thirty-two rows fill the LHS tile: slice i is K64 block i alone.
                @pl.loop(0, destination.shape[0] // 2)
                def split(index):
                    pair = source[:, pl.ds(pl.multiple_of(index * 128, 128), 128)]
                    destination[2 * index] = pair[:, :64]
                    destination[2 * index + 1] = pair[:, 64:]

                return
            owner = jax.lax.broadcasted_iota(jnp.int32, (lhs_rows, width_k), 1) // 64

            # Unrolled slices pipeline; one slice per iteration is a serial chain.
            @pl.loop(0, destination.shape[0], unroll=min(8, destination.shape[0]))
            def build(index):
                block = source[:, pl.ds(pl.multiple_of(index * width_k, 128), width_k)]
                block = block.astype(F32)
                destination[index] = jnp.concatenate(
                    [jnp.where(owner == g, block, 0.0) for g in range(groups)], axis=0
                ).astype(FP8)

        def stacked_dot(lhs, weight, scales, destination):
            # Products are scaled and added per block in ascending block order, as in
            # block_dot; DOWN columns are contracted 1024 at a time. Tuples of LHS
            # tiles and destinations contract their tiles' rows against one unpack.
            lhs = lhs if isinstance(lhs, tuple) else (lhs,)
            destination = destination if isinstance(destination, tuple) else (destination,)
            for target in destination:
                target[...] = jnp.zeros(target.shape, F32)
            blocks, columns = weight.shape[0] // 8, weight.shape[-1]
            chunk = min(columns, 1024 // len(lhs))
            # Eight- and sixteen-row tiles contract every block in one iteration: each
            # iteration ends by draining the MXU, which idles while its last results pop.
            unroll = blocks if groups >= 2 else 32 if blocks % 32 == 0 else 8

            @pl.loop(0, blocks // unroll)
            def group(index):
                base = pl.multiple_of(index * unroll, 8)
                scale_rows = {}
                for quad in range(unroll // groups):
                    first = index * unroll + quad * groups
                    left = [part[index * (unroll // groups) + quad] for part in lhs]
                    left = left[0] if len(left) == 1 else jnp.concatenate(left, axis=0)
                    for start in range(0, columns, chunk):
                        if quad * groups % 8 == 0:
                            scale_rows[start] = scales[
                                pl.ds(pl.multiple_of(base + quad * groups, 8), 8),
                                start : start + chunk,
                            ]
                        unpacked = tpu.bitcast(
                            weight[pl.ds(pl.multiple_of(first * 8, 8), 8 * groups),
                                   start : start + chunk],
                            jnp.float4_e2m1fn,
                        ).astype(FP8)
                        product = jnp.dot(left, unpacked, preferred_element_type=F32)
                        for part, target in enumerate(destination):
                            for g in range(groups):
                                scale = scale_rows[start][(quad * groups + g) % 8]
                                row = part * 32 + g * lhs_rows
                                target[:, start : start + chunk] += (
                                    product[row : row + lhs_rows] * scale[None, :]
                                )

        if stacked and tiles == 1:
            stack_blocks(xq, stacked_x)
        for tile in range(tiles if tiles > 1 else 0):
            # Each row's quantization is its own, as in the whole padded tile.
            tile_input, tile_scale = rowquant_fp8(
                normed[tile * lhs_rows : (tile + 1) * lhs_rows, :].astype(BF16)
            )
            xq[...] = tile_input
            row_scale[tile] = jnp.broadcast_to(tile_scale, (lhs_rows, 128))
            stack_blocks(xq, stacked_x.at[tile])

        def group_barrier():
            if peers > 1:
                for peer in range(1, etp):
                    pl.semaphore_signal(
                        barrier, 1, device_id=peer_id(rank ^ peer), device_id_type=id_type
                    )
                pl.semaphore_wait(barrier, etp - 1)

        def exchange(packet):
            for peer in range(1, etp):
                tpu.make_async_remote_copy(
                    packet.at[0],
                    packet.at[peer],
                    send.at[peer - 1],
                    receive.at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).start()
            for peer in range(1, etp):
                tpu.make_async_remote_copy(
                    packet.at[0],
                    packet.at[peer],
                    send.at[peer - 1],
                    receive.at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).wait()

        slots = dict(gu=gu, gus=gus, down=down, downs=downs, dma=dma)

        def gu_copies(expert, slot):
            return _transfers(weights_at_layer, slots, held(expert), slot, 0)[0]

        def down_copies(expert, slot):
            return _transfers(weights_at_layer, slots, held(expert), 0, slot)[1]

        def next_expert(skipped=()):
            # The next unvisited expert after those ``skipped``, already started.
            ids = route_ids[...]
            for expert in skipped:
                ids = jnp.where(ids == expert, c.experts, ids)
            if started is None:
                return jnp.min(ids)
            # Earliest top-k step first, ascending ID within a step: the least key
            # step * experts + ID names the expert itself.
            limit = top_k * c.experts
            keys = jnp.where(
                ids < c.experts, jnp.arange(top_k)[None, :] * c.experts + ids, limit
            )
            least = jnp.min(keys)
            return jnp.where(least < limit, least % c.experts, c.experts)

        def block_dot(activation, weight, scales, destination):
            def output_tile(weight, scales, destination):
                # U32 packs eight adjacent K nibbles. TPU bitcast widens the
                # contraction dimension, matching the pinned MXU inner body.
                destination[...] = jnp.zeros(destination.shape, F32)

                blocks = weight.shape[0] // 8
                if blocks >= 32 and blocks % 32 == 0:

                    @pl.loop(0, blocks // 32)
                    def scale_group32(index):
                        # Thirty-two complete block64 scale rows. Keep the
                        # original K128 activation tiles and Ref add boundaries.
                        aligned = pl.multiple_of(index * 32, 8)
                        scale_rows = scales[pl.ds(aligned, 32), :]
                        for pair_index in range(16):
                            activation_pair = activation[
                                :, pl.ds(index * 2048 + pair_index * 128, 128)
                            ]
                            for half in range(2):
                                offset = pair_index * 2 + half
                                block = index * 32 + offset
                                unpacked = tpu.bitcast(
                                    weight[pl.ds(block * 8, 8), :], jnp.float4_e2m1fn
                                ).astype(FP8)
                                scale = scale_rows[offset, :]
                                product = jnp.dot(
                                    activation_pair[:, half * 64 : (half + 1) * 64],
                                    unpacked,
                                    preferred_element_type=F32,
                                )
                                destination[...] += product * scale[None, :]
                elif blocks >= 16 and blocks % 16 == 0:

                    @pl.loop(0, blocks // 16)
                    def scale_group16(index):
                        # Sixteen complete block64 scale rows. Keep the
                        # original K128 activation tiles and Ref add boundaries.
                        aligned = pl.multiple_of(index * 16, 8)
                        scale_rows = scales[pl.ds(aligned, 16), :]
                        for pair_index in range(8):
                            activation_pair = activation[
                                :, pl.ds(index * 1024 + pair_index * 128, 128)
                            ]
                            for half in range(2):
                                offset = pair_index * 2 + half
                                block = index * 16 + offset
                                unpacked = tpu.bitcast(
                                    weight[pl.ds(block * 8, 8), :], jnp.float4_e2m1fn
                                ).astype(FP8)
                                scale = scale_rows[offset, :]
                                product = jnp.dot(
                                    activation_pair[:, half * 64 : (half + 1) * 64],
                                    unpacked,
                                    preferred_element_type=F32,
                                )
                                destination[...] += product * scale[None, :]
                elif blocks >= 8 and blocks % 8 == 0:

                    @pl.loop(0, blocks // 8)
                    def scale_group(index):
                        # Exactly eight complete block64 scale rows. Keep the
                        # original K128 activation tiles and Ref add boundaries.
                        aligned = pl.multiple_of(index * 8, 8)
                        scale_rows = scales[pl.ds(aligned, 8), :]
                        for pair_index in range(4):
                            activation_pair = activation[
                                :, pl.ds(index * 512 + pair_index * 128, 128)
                            ]
                            for half in range(2):
                                offset = pair_index * 2 + half
                                block = index * 8 + offset
                                unpacked = tpu.bitcast(
                                    weight[pl.ds(block * 8, 8), :], jnp.float4_e2m1fn
                                ).astype(FP8)
                                scale = scale_rows[offset, :]
                                product = jnp.dot(
                                    activation_pair[:, half * 64 : (half + 1) * 64],
                                    unpacked,
                                    preferred_element_type=F32,
                                )
                                destination[...] += product * scale[None, :]
                else:

                    @pl.loop(0, weight.shape[0] // 16)
                    def pair(index):
                        # Dynamically slice on the physical 128-column boundary, then
                        # take the two static 64-K subblocks. A dynamic 64-column VMEM
                        # slice would violate the TPU vector lane alignment.
                        activation_pair = activation[:, pl.ds(index * 128, 128)]
                        for half in range(2):
                            block = index * 2 + half
                            unpacked = tpu.bitcast(
                                weight[pl.ds(block * 8, 8), :], jnp.float4_e2m1fn
                            ).astype(FP8)
                            if scales.shape[0] < 8:
                                scale_rows = scales[...]
                                selected = jnp.arange(scales.shape[0])[:, None] == block
                            else:
                                aligned = pl.multiple_of((block // 8) * 8, 8)
                                scale_rows = scales[pl.ds(aligned, 8), :]
                                selected = jnp.arange(8)[:, None] == block % 8
                            scale = jnp.sum(jnp.where(selected, scale_rows, 0), axis=0)
                            product = jnp.dot(
                                activation_pair[:, half * 64 : (half + 1) * 64],
                                unpacked,
                                preferred_element_type=F32,
                            )
                            destination[...] += product * scale[None, :]

            tile = min(layout.expert_output_tile or weight.shape[-1], weight.shape[-1])
            if not layout.expert_output_tile or tile == weight.shape[-1]:
                output_tile(weight, scales, destination)
            else:
                if weight.shape[-1] % tile:
                    raise ValueError("Expert output columns must divide the configured tile")

                @pl.loop(0, weight.shape[-1] // tile)
                def columns(index):
                    start = pl.multiple_of(index * tile, 128)
                    output_tile(
                        weight.at[:, pl.ds(start, tile)],
                        scales.at[:, pl.ds(start, tile)],
                        destination.at[:, pl.ds(start, tile)],
                    )

        # All ETP members enter this scope, including empty route groups.
        # Align their scoped communication storage before the first transfer.
        group_barrier()
        if started is None:
            first = next_expert()

            @pl.when(first < c.experts)
            def prefetch_first():
                for transfer in (*gu_copies(first, 0), *down_copies(first, 0)):
                    transfer.start()
        elif layout.expert_overlap:
            # The early start skips experts that may change group; start one of those
            # here when no other local expert is routed.
            first = jnp.where(started < c.experts, started, next_expert())

            @pl.when((started == c.experts) & (first < c.experts))
            def prefetch_first():
                for transfer in (*gu_copies(first, 0), *down_copies(first, 0)):
                    transfer.start()
        else:
            first = started
        depth, down_depth = gu.shape[0], down.shape[0]
        pending = []
        for ahead in range(1, depth - 1):
            expert_ahead = next_expert((first, *pending))

            @pl.when(expert_ahead < c.experts)
            def prefetch_ahead(expert_ahead=expert_ahead, ahead=ahead):
                for transfer in gu_copies(expert_ahead, ahead):
                    transfer.start()
                if ahead < down_depth - 1:
                    for transfer in down_copies(expert_ahead, ahead):
                        transfer.start()

            pending.append(expert_ahead)
        # Each visit starts the transfers of an expert chosen one visit earlier, so the
        # vector-to-scalar reduction naming it overlaps that earlier visit's contractions.
        queued = next_expert((first, *pending))

        def tile_hits(expert):
            # Whether each eight-row tile holds a row routed to the expert.
            routed = jnp.max(jnp.where(route_ids[...] == expert, 1, 0), axis=-1, keepdims=True)
            return tuple(
                jnp.max(routed[tile * lhs_rows : (tile + 1) * lhs_rows]) for tile in range(tiles)
            )

        if shared_contribution is not None:
            shared_contribution()

        def expert(carry):
            current, *following, queued, visit = carry[: depth + 1]
            hits = carry[depth + 1 :]
            slot, down_slot = visit % depth, visit % down_depth
            with jax.named_scope("expert_gu_dma_wait"):
                for transfer in gu_copies(current, slot):
                    transfer.wait()
            # The previous expert's slots are free once its contractions are done. Issuing
            # after the wait keeps two experts' transfers, not three, in flight.
            ahead = (visit + depth - 1) % depth
            down_next = (*following, queued)[down_depth - 2]

            @pl.when(queued < c.experts)
            def prefetch_queued():
                for transfer in gu_copies(queued, ahead):
                    transfer.start()
                if down_depth == depth:
                    for transfer in down_copies(queued, ahead):
                        transfer.start()

            if down_depth < depth:

                @pl.when(down_next < c.experts)
                def prefetch_down():
                    for transfer in down_copies(down_next, (visit + down_depth - 1) % down_depth):
                        transfer.start()

            selected = route_ids[...] == current
            probability = jnp.sum(jnp.where(selected, route_probs[...], 0), axis=-1)
            route_ids[...] = jnp.where(selected, c.experts, route_ids[...])
            later = next_expert((*following, queued))
            following_hits = tile_hits(following[0]) if tiles > 1 else ()

            if tiles > 1:
                tiled_expert(current, slot, down_slot, selected, hits)
            else:
                padded_expert(current, slot, down_slot, probability)

            return *following, queued, later, visit + 1, *following_hits

        def padded_expert(current, slot, down_slot, probability):
            with jax.named_scope("expert_gate_up_mxu"):
                if stacked:
                    stacked_dot(stacked_x, gu.at[slot], gus.at[slot], gateup)
                else:
                    block_dot(xq, gu.at[slot], gus.at[slot], gateup)
            gate = gateup[:, :width] * input_scale
            up = gateup[:, width:] * input_scale
            middle[...] = (jax.nn.silu(gate) * up).astype(BF16)
            local_amax = jnp.max(jnp.abs(middle[...]), axis=-1, keepdims=True).astype(F32)
            if peers > 1:
                amax_packet[0, ...] = jnp.broadcast_to(local_amax, (rows, 128))
                exchange(amax_packet)
                maximum = amax_packet[0, ...]
                for peer in range(1, etp):
                    maximum = jnp.maximum(maximum, amax_packet[peer, ...])
                # Commit the result before acknowledging inbox consumption.
                amax_packet[0, ...] = maximum
                group_barrier()
                local_amax = amax_packet[0, :, :1]
            quantized_mid, mid_scale = rowquant_fp8(middle[...], local_amax)
            midq[...] = quantized_mid
            if stacked:
                stack_blocks(midq, stacked_mid)
            with jax.named_scope("expert_down_dma_wait"):
                for transfer in down_copies(current, down_slot):
                    transfer.wait()
            with jax.named_scope("expert_down_mxu"):
                if stacked:
                    stacked_dot(stacked_mid, down.at[down_slot], downs.at[down_slot], projection)
                else:
                    block_dot(midq, down.at[down_slot], downs.at[down_slot], projection)
            if peers > 1:
                packet = (
                    projection[:1, :].reshape(rows, packet_columns) if compact else projection[...]
                )
                down_packet[0, ...] = packet
                exchange(down_packet)
                total = down_packet[rank % etp, ...]
                for absolute in range(1, etp):
                    total += down_packet[(rank % etp) ^ absolute, ...]
                projection[...] = (
                    jnp.pad(total.reshape(1, hidden), ((0, rows - 1), (0, 0))) if compact else total
                )
                group_barrier()
            if residual:
                output_moe[...] += projection[...] * mid_scale * probability[:, None]
            else:
                projection[...] *= mid_scale
                wire, wire_scale = rowquant_fp8(projection[...].astype(BF16))
                coefficient = wire_scale * probability[:, None]
                contribution = wire.astype(F32) * coefficient
                output_moe[...] += jnp.where(
                    rank % etp == 0, contribution, jnp.zeros_like(contribution)
                )

        def tiled_expert(current, slot, down_slot, selected, hits):
            # Only tiles holding one of the expert's routed rows are contracted. A
            # row's products, scales and sums match the padded tile's; unrouted rows
            # of a contracted tile add their zero-probability product, as there.
            probability = jnp.sum(jnp.where(selected, route_probs[...], 0), axis=-1, keepdims=True)
            for tile in range(tiles):
                row_prob[tile] = jnp.broadcast_to(
                    probability[tile * lhs_rows : (tile + 1) * lhs_rows], (lhs_rows, 128)
                )
            # Hit tiles in ascending order, then `tiles` past the last hit.
            order, previous = [], jnp.int32(-1)
            for _ in range(tiles):
                following = jnp.int32(tiles)
                for tile in reversed(range(tiles)):
                    following = jnp.where((hits[tile] > 0) & (tile > previous), tile, following)
                order.append(following)
                previous = following
            count = sum(jnp.where(hit > 0, 1, 0) for hit in hits)

            def wait_down():
                with jax.named_scope("expert_down_dma_wait"):
                    for transfer in down_copies(current, down_slot):
                        transfer.wait()

            def contract(parts, first_pass, last_added=True):
                # Each tile's rows keep their own scales; the tiles share one unpack.
                # ``last_added`` is false when the last part repeats an earlier tile.
                with jax.named_scope("expert_gate_up_mxu"):
                    stacked_dot(
                        tuple(stacked_x.at[tile] for tile in parts),
                        gu.at[slot],
                        gus.at[slot],
                        tuple(gateup.at[part] for part in range(len(parts))),
                    )
                # DOWN weights are waited by the first pass, or after the passes when
                # no row remains routed to an early-started expert.
                pl.when(first_pass)(wait_down)
                mid_scales = []
                for part, tile in enumerate(parts):
                    input_scale = row_scale[tile][:, :1]
                    gate = gateup[part, :, :width] * input_scale
                    up = gateup[part, :, width:] * input_scale
                    middle[...] = (jax.nn.silu(gate) * up).astype(BF16)
                    local_amax = jnp.max(jnp.abs(middle[...]), axis=-1, keepdims=True).astype(F32)
                    quantized_mid, mid_scale = rowquant_fp8(middle[...], local_amax)
                    midq[...] = quantized_mid
                    stack_blocks(midq, stacked_mid.at[part])
                    mid_scales.append(mid_scale)
                with jax.named_scope("expert_down_mxu"):
                    stacked_dot(
                        tuple(stacked_mid.at[part] for part in range(len(parts))),
                        down.at[down_slot],
                        downs.at[down_slot],
                        tuple(projection.at[part] for part in range(len(parts))),
                    )
                for part, tile in enumerate(parts):

                    def add(part=part, tile=tile):
                        base = pl.multiple_of(tile * lhs_rows, 8)
                        output_moe[pl.ds(base, lhs_rows), :] += (
                            projection[part] * mid_scales[part] * row_prob[tile][:, :1]
                        )

                    if part == len(parts) - 1 and last_added is not True:
                        pl.when(last_added)(add)
                    else:
                        add()

            per_pass = tiles_per_pass(tiles)

            @pl.loop(0, (count + per_pass - 1) // per_pass)
            def tile_pass(index):
                picked = order[:per_pass]
                for later in range(1, tiles // per_pass):
                    picked = [
                        jnp.where(index == later, order[later * per_pass + k], tile)
                        for k, tile in enumerate(picked)
                    ]
                remaining = count - index * per_pass
                if per_pass == 4:
                    # Three remaining tiles repeat the third rather than run a third variant.
                    full = remaining >= 4
                    fourth = jnp.where(full, picked[3], picked[2])
                    pl.when(remaining >= 3)(
                        lambda: contract((*picked[:3], fourth), index == 0, full)
                    )
                    pl.when(remaining == 2)(lambda: contract(tuple(picked[:2]), index == 0))
                else:
                    pl.when(remaining >= 2)(lambda: contract(tuple(picked[:2]), index == 0))
                pl.when(remaining == 1)(lambda: contract((picked[0],), index == 0))

            pl.when(count == 0)(wait_down)

        jax.lax.while_loop(
            lambda carry: carry[0] < c.experts,
            expert,
            (first, *pending, queued, jnp.int32(0), *(tile_hits(first) if tiles > 1 else ())),
        )

    if scratch is None:
        pl.run_scoped(compute, **scratch_shapes(layout, batch_size, rows))
    else:
        compute(**scratch)

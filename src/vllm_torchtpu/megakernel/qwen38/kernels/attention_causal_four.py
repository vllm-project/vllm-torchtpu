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
"""UNQUALIFIED four causal positions, derived from pinned paired attention.

One, two or four requests, shared bounded historical KV tiles, four separate
causal masks per request. Four consecutive rows span at most two128-token pages
on distinct CP8 owners; each rank can therefore append at most one local page
per request. All append DMA completes before tile reuse. No production runtime
selects this prototype.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from .attention_append import fold_scores, packet_rows, tile_scores

F32 = jnp.float32
BF16 = jnp.bfloat16


def attend_paged_causal_four(
    query,
    key,
    value,
    key_cache,
    value_cache,
    position,
    rank,
    layout,
    scales,
    id_type,
    inbox,
    active=None,
    buffers=2,
):
    """Attend four consecutive positions of each request, reading each KV tile once.

    Query is [requests * 4, local_heads, D], request-major, for one, two or four
    requests; caches are [requests, pages, 128, D]. Each request's positions
    must be p..p+3, constructed by the block-verifier wrapper. Each row retains
    its own causal mask, 1024-token softmax arithmetic and ascending CP merge.
    A request's four rows' gathered heads share one contraction per tile; a
    row's statistics change only on tiles its own position reaches. Optional
    per-row ``active`` flags mask a request's unverified tail: inactive rows
    neither append nor extend the attended prefix. ``inbox`` holds the refs of
    ``attention_append.exchange_scratch(layout, requests * 4)``. One query
    gather and one statistics exchange serve every request. Requests' KV blocks
    rotate through ``buffers`` banks. With two banks each request's fold is
    serial and reads one block ahead. With more, the next block's QK is issued
    before this block's softmax and PV so that they overlap; a bank then stays
    live until its PV, and blocks are read ``buffers - 2`` ahead.
    """
    cp, page, hd = layout.context_parallel, layout.kv_page_size, layout.config.head_dim
    rows, local_heads = query.shape[:2]
    requests = rows // 4
    if rows != 4 * requests or requests not in (1, 2, 4, 8, 16) or cp != 8:
        raise ValueError("Causal-four attention requires 1, 2, 4, 8 or 16 requests and CP8")
    # A request's joined rows are [CP rank, causal row, local head]: each rank's
    # heads for its four rows form one aligned group. A peer's packet holds
    # every request's group in turn.
    width = 4 * local_heads
    group = packet_rows(width)
    packet = packet_rows(rows * local_heads)
    joined_rows = cp * group
    dtype = key_cache.dtype
    # Transfer neighboring *local* pages together. Eight pages bound scratch
    # at 1024 tokens; the physical pages are contiguous even though global
    # positions are striped. Choose a divisor of capacity so the last tile is
    # always in bounds, then mask its future tokens.
    capacity_pages = key_cache.shape[1]
    block_pages = 1
    while block_pages < 8 and capacity_pages % (2 * block_pages) == 0:
        block_pages *= 2
    block_tokens = block_pages * page
    block_shape = (block_tokens, hd) if cp == 1 else (block_pages, page, hd)
    cp_rank = rank % cp
    pipelined = buffers > 2
    key_scale, value_scale = scales if layout.fast else (1.0, 1.0)

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    def body(
        keys,
        values,
        queries,
        stats_out,
        maximum,
        mass,
        numerator,
        append_sem,
        page_sem,
        cp_queries,
        cp_stats,
        cp_sends,
        cp_recvs,
        carried=None,
    ):
        # Each request's current rows are merged into their owned page before
        # any of its queries reads the tile. Consecutive positions can share a
        # page or lie on adjacent CP owners; each lane therefore writes at most
        # one local page per request.
        if active is None:
            owners = tuple(position[b] // page % cp for b in range(rows))
            last_positions = tuple(position[4 * r + 3] for r in range(requests))
        else:
            owners = tuple(
                jnp.where(active[b], position[b] // page % cp, -1) for b in range(rows)
            )
            last_positions = tuple(
                jnp.max(jnp.stack(tuple(
                    jnp.where(active[b], position[b], -1) for b in range(4 * r, 4 * r + 4)
                )))
                for r in range(requests)
            )
        pages = tuple(position[b] // (page * cp) for b in range(rows))
        owned_blocks = tuple(
            (jnp.maximum(0, (last // page + cp - cp_rank) // cp) + block_pages - 1) // block_pages
            for last in last_positions
        )
        write_pages, has_writes = [], []
        for r in range(requests):
            write_page, has_write = pages[4 * r], jnp.bool_(False)
            for b in range(4 * r, 4 * r + 4):
                write_page = jnp.where(owners[b] == cp_rank, pages[b], write_page)
                has_write = has_write | (owners[b] == cp_rank)
            write_pages.append(write_page)
            has_writes.append(has_write)

        def transfers(r, index, slot):
            start = pl.ds(index * block_pages, block_pages)
            return (
                tpu.make_async_copy(
                    key_cache.at[r, start, :, :], keys.at[slot], page_sem.at[2 * slot]
                ),
                tpu.make_async_copy(
                    value_cache.at[r, start, :, :], values.at[slot], page_sem.at[2 * slot + 1]
                ),
            )

        # Requests' owned blocks form one sequence; block g uses bank g % buffers.
        bases = [sum(owned_blocks[:r]) for r in range(requests)]

        def start_block(sequence):
            # The last request starting at or before the block owns it: a request
            # without blocks shares its base with its successor.
            request, first = jnp.int32(0), jnp.int32(0)
            for r in range(1, requests):
                later = sequence >= bases[r]
                request = jnp.where(later, r, request)
                first = jnp.where(later, bases[r], first)

            @pl.when(sequence < bases[-1] + owned_blocks[-1])
            def start():
                for dma in transfers(request, sequence - first, sequence % buffers):
                    dma.start()

        # Local page reads need no peer; they overlap the query gather.
        for sequence in range(buffers - 1):
            start_block(jnp.int32(sequence))

        def exchange(kind, source, destination):
            # source(peer) is what rank ^ peer needs from this rank.
            copies = [
                tpu.make_async_remote_copy(
                    source(peer),
                    destination.at[peer],
                    cp_sends.at[kind, peer - 1],
                    cp_recvs.at[kind, peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                )
                for peer in range(1, cp)
            ]
            for copy in copies:
                copy.start()
            for copy in copies:
                copy.wait()

        cp_queries[0, ...] = jnp.zeros(cp_queries.shape[1:], BF16)
        for b in range(rows):
            cp_queries[0, pl.ds(b * local_heads, local_heads), :] = query[b]
        with jax.named_scope("cp_query_gather"):
            exchange(0, lambda peer: cp_queries.at[0], cp_queries)
            if width < group:
                queries[...] = jnp.zeros(queries.shape, BF16)
            for r in range(requests):
                for absolute_rank in range(cp):
                    queries[r, pl.ds(absolute_rank * group, width), :] = cp_queries[
                        cp_rank ^ absolute_rank, pl.ds(r * width, width), :
                    ]

        maximum[...] = jnp.full(maximum.shape, -jnp.inf, F32)
        mass[...] = jnp.zeros(mass.shape, F32)
        numerator[...] = jnp.zeros(numerator.shape, F32)

        def process_request(r):
            # One traced request index keeps a single copy of the page fold in
            # the instruction stream, whatever the request count.
            def select(values):
                chosen = values[0]
                for other in range(1, requests):
                    chosen = jnp.where(r == other, values[other], chosen)
                return chosen

            def per_row(values):
                return [select(values[slot_row::4]) for slot_row in range(4)]

            base, write_page, has_write, owned = (
                select(values) for values in (bases, write_pages, has_writes, owned_blocks)
            )
            row_positions, row_owners, row_pages = (
                per_row(values) for values in (position, owners, pages)
            )
            row_keys = per_row(tuple(key[b] for b in range(rows)))
            row_values = per_row(tuple(value[b] for b in range(rows)))

            def append_transfers(slot):
                local_page = write_page % block_pages
                return (
                    tpu.make_async_copy(
                        keys.at[slot, local_page, :, :],
                        key_cache.at[r, write_page, :, :],
                        append_sem.at[0],
                    ),
                    tpu.make_async_copy(
                        values.at[slot, local_page, :, :],
                        value_cache.at[r, write_page, :, :],
                        append_sem.at[1],
                    ),
                )

            def append_due(index):
                return has_write & (index == write_page // block_pages)

            def receive(index):
                slot = (base + index) % buffers
                for dma in transfers(r, index, slot):
                    dma.wait()

                def append_position(j):
                    @pl.when((row_owners[j] == cp_rank) & (index == row_pages[j] // block_pages))
                    def append_row():
                        # Only the row's own page of the bank changes.
                        local_page = row_pages[j] % block_pages
                        mask = jnp.arange(page)[:, None] == row_positions[j] % page
                        for bank, row, scale in (
                            (keys, row_keys[j], scales[0]),
                            (values, row_values[j], scales[1]),
                        ):
                            encoded = row
                            if layout.fast:
                                encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(dtype)
                            tile = bank[slot, local_page]
                            bank[slot, local_page] = jnp.where(mask, encoded[None, :], tile)

                for slot_row in range(4):
                    append_position(slot_row)

                @pl.when(append_due(index))
                def start_append():
                    for dma in append_transfers(slot):
                        dma.start()

            # These immutable tiles serve all four causal rows.
            def operand(bank, index):
                return bank[(base + index) % buffers, ...].reshape(block_tokens, hd)

            # Joining query rows changes no row's K/V contraction. Each row
            # keeps its own causal mask, and rows whose position ends before
            # a tile keep their statistics bit for bit.
            def by_row(values):
                causal_row = (
                    jax.lax.broadcasted_iota(jnp.int32, (joined_rows, 1), 0) % group
                ) // local_heads
                joined = jnp.zeros((joined_rows, 1), jnp.int32)
                for slot_row, value in enumerate(values):
                    joined = jnp.where(causal_row == slot_row, value, joined)
                return joined

            def valid(index):
                global_pages = (index * block_pages + jnp.arange(block_pages)) * cp + cp_rank
                token_positions = (
                    global_pages[:, None] * page + jnp.arange(page)[None, :]
                ).reshape(block_tokens)
                row_position = by_row(row_positions)
                return token_positions[None, :] <= row_position

            def block_scores(index):
                return tile_scores(
                    queries[r, ...], operand(keys, index), valid(index), hd**-0.5 * key_scale
                )

            def fold(index, scores):
                m, updated_mass, n = fold_scores(
                    maximum[r, :, :1],
                    mass[r, :, :1],
                    numerator[r, ...],
                    scores,
                    operand(values, index),
                    value_scale,
                )
                # A row whose position ends before this block sees only -inf scores,
                # which leave its statistics bit for bit.
                maximum[r, ...] = jnp.broadcast_to(m, maximum.shape[1:])
                mass[r, ...] = jnp.broadcast_to(updated_mass, mass.shape[1:])
                numerator[r, ...] = n

                @pl.when(append_due(index))
                def complete_append():
                    # Source banks cannot be reused until both writes finish.
                    for dma in append_transfers((base + index) % buffers):
                        dma.wait()

            if pipelined:

                @pl.when(owned > 0)
                def stream():
                    receive(jnp.int32(0))
                    carried[0] = block_scores(jnp.int32(0))

                    @pl.loop(0, owned - 1)
                    def page_loop(index):
                        # The preceding block's bank was folded in the previous
                        # iteration and its append, if any, has completed.
                        start_block(base + index + buffers - 1)
                        receive(index + 1)
                        following = block_scores(index + 1)
                        fold(index, carried[index % 2])
                        carried[(index + 1) % 2] = following

                    # The final fold also starts its block's successor in the
                    # request sequence, which may belong to the next request.
                    last = owned - 1
                    start_block(base + last + buffers - 1)
                    fold(last, carried[last % 2])

            else:

                @pl.loop(0, owned)
                def page_loop(index):
                    # The bank of the preceding block is free: it was attended and
                    # its append, if any, has completed.
                    start_block(base + index + buffers - 1)
                    receive(index)
                    fold(index, block_scores(index))

        with jax.named_scope("causal_pair_streaming_attention_pages"):
            pl.loop(0, requests)(process_request)

        # Each peer receives only its own heads' statistics for every row.
        with jax.named_scope("cp_softmax_merge"):
            for absolute_rank in range(cp):
                own = slice(absolute_rank * group, absolute_rank * group + width)
                for r in range(requests):
                    sent = pl.ds(r * width, width)
                    stats_out[absolute_rank, sent, :] = jnp.concatenate(
                        (maximum[r, own, :], numerator[r, own, :]), axis=1
                    )
                    stats_out[absolute_rank, sent, 1] = mass[r, own, 0]
            cp_stats[0, ...] = stats_out[cp_rank, ...]
            exchange(1, lambda peer: stats_out.at[cp_rank ^ peer], cp_stats)
            global_max = jnp.full((packet, 1), -jnp.inf, F32)
            for absolute_rank in range(cp):
                received = cp_stats[cp_rank ^ absolute_rank, ...]
                global_max = jnp.maximum(global_max, received[:, :1])
            safe_max = jnp.where(jnp.isfinite(global_max), global_max, 0)
            total_mass = jnp.zeros((packet, 1), F32)
            total_value = jnp.zeros((packet, hd), F32)
            for absolute_rank in range(cp):
                received = cp_stats[cp_rank ^ absolute_rank, ...]
                correction = jnp.where(
                    jnp.isfinite(received[:, :1]), jnp.exp(received[:, :1] - safe_max), 0
                )
                total_mass += received[:, 1:2] * correction
                total_value += received[:, 128:] * correction
            result = jnp.where(total_mass > 0, total_value / jnp.maximum(total_mass, 1e-30), 0)
            # Selecting own heads by summing zeros elsewhere made zeros positive.
            result = jnp.where(result == 0, 0.0, result)
            return jnp.stack(
                [result[b * local_heads : (b + 1) * local_heads] for b in range(rows)]
            ).astype(BF16)

    carried = {"carried": tpu.VMEM((2, joined_rows, block_tokens), F32)} if pipelined else {}
    with jax.named_scope("paged_attention"):
        return pl.run_scoped(
            lambda **refs: body(**refs, **inbox),
            keys=tpu.VMEM((buffers, *block_shape), dtype),
            values=tpu.VMEM((buffers, *block_shape), dtype),
            queries=tpu.VMEM((requests, joined_rows, hd), BF16),
            stats_out=tpu.VMEM((cp, packet, hd + 128), F32),
            maximum=tpu.VMEM((requests, joined_rows, 128), F32),
            mass=tpu.VMEM((requests, joined_rows, 128), F32),
            numerator=tpu.VMEM((requests, joined_rows, hd), F32),
            append_sem=tpu.SemaphoreType.DMA((2,)),
            page_sem=tpu.SemaphoreType.DMA((2 * buffers,)),
            **carried,
        )

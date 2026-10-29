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
"""Bounded paged decode attention, with host-local context-parallel exchange.

KV pages are striped over adjacent CP ranks. Every DMA is waited before its
buffer is reused or the scope returns; empty ranks participate in both exchanges
with maximum -inf and zero mass. Scratch capacity never depends on context.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from .attention_append import fold_scores, packet_rows, tile_scores
from .paged_kv import PoolPages, bank_shape, set_column, transposed

F32 = jnp.float32
BF16 = jnp.bfloat16


def validate_kv_scales(layout, kv_scales):
    """Freeze the actual cache calibration into the compiled decode contract."""
    count = len(layout.config.full_layers)
    if kv_scales is None:
        if layout.fast and count:
            raise ValueError("FP8 KV requires explicit per-layer (key, value) calibration scales")
        return ((1.0, 1.0),) * count
    if len(kv_scales) != count:
        raise ValueError("kv_scales must contain one (key, value) pair per full-attention layer")
    pairs = tuple(tuple(float(x) for x in pair) for pair in kv_scales)
    if any(len(pair) != 2 or any(not math.isfinite(x) or x <= 0 for x in pair) for pair in pairs):
        raise ValueError("KV calibration scales must be positive finite pairs")
    return pairs


def attend_paged_batched_append(
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
    buffers=2,
):
    """Return this rank's GQA results and update its owned KV page in-place.

    ``query`` has [requests, local query heads, D] for two or four requests.
    Cache references select the layer and KV head, retaining every request:
    one [requests,pages,128,D] reference or a tuple of [pages,128,D] references;
    or ``key_cache`` is a ``paged_kv.PoolPages`` (any row; FP8, no
    ``value_cache``), whose row selects each request.
    A request at a negative position neither attends nor appends.
    ``scales`` are the source cache's key/value scales, not projection scales.
    ``inbox`` holds the refs of ``attention_append.exchange_scratch(layout, requests)``.
    Requests' KV blocks rotate through ``buffers`` banks. With two banks each
    request's fold is serial and reads one block ahead. With more, the next
    block's QK is issued before this block's softmax and PV so that they
    overlap; a bank then stays live until its PV, and blocks are read
    ``buffers - 2`` ahead.
    """
    cp, page, hd = layout.context_parallel, layout.kv_page_size, layout.config.head_dim
    batch, local_heads = query.shape[:2]
    if batch not in (2, 4, 8, 16) or cp != 8:
        raise ValueError("Batched attention supports C2/C4/C8/C16 CP8 only")
    total_heads = local_heads * cp
    # All requests' heads for one rank form the only rows that rank receives.
    group = packet_rows(batch * local_heads)
    all_rows = max(8, (total_heads + 7) // 8 * 8)
    paged = isinstance(key_cache, PoolPages)
    if paged:
        pages = key_cache
        # A bank is one vLLM block of pool pages; a traced request selects its row.
        stacked = None
        dtype = jnp.float8_e4m3fn
        block_pages = pages.pages_per_block
        block_shape = bank_shape(block_pages, hd, page)
    else:
        # One stacked cache per kind lets a traced request index select a block's source.
        stacked = None if isinstance(key_cache, tuple) else (key_cache, value_cache)
        if stacked is not None:
            key_cache = tuple(key_cache.at[b] for b in range(batch))
            value_cache = tuple(value_cache.at[b] for b in range(batch))
        dtype = key_cache[0].dtype
        # Transfer neighboring *local* pages together. Eight pages bound scratch
        # at 1024 tokens; the physical pages are contiguous even though global
        # positions are striped. Choose a divisor of capacity so the last tile is
        # always in bounds, then mask its future tokens.
        capacity_pages = key_cache[0].shape[0] // page if cp == 1 else key_cache[0].shape[0]
        block_pages = 1
        while block_pages < 8 and capacity_pages % (2 * block_pages) == 0:
            block_pages *= 2
        block_shape = (block_pages * page, hd) if cp == 1 else (block_pages, page, hd)
    block_tokens = block_pages * page
    # Whether a traced request index can select a block's source.
    traced_rows = paged or stacked is not None
    cp_rank = rank % cp
    pipelined = buffers > 2
    # The KV is FP8 in the fast profile and in vLLM's pools (any profile).
    fp8 = layout.fast or paged
    key_scale, value_scale = scales if fp8 else (1.0, 1.0)

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    def live(index, j, position):
        # Tokens of local page index * block_pages + j at or before ``position``.
        start = ((index * block_pages + j) * cp + cp_rank) * page
        return jnp.clip(position - start + 1, 0, page)

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
        owned_blocks = tuple(
            (jnp.maximum(0, (position[b] // page + cp - cp_rank) // cp) + block_pages - 1)
            // block_pages
            for b in range(batch)
        )

        def cache_block(cache, index):
            if cp == 1:
                return cache.at[pl.ds(index * block_tokens, block_tokens), :]
            return cache.at[pl.ds(index * block_pages, block_pages), :, :]

        def transfers(b, index, slot):
            if paged:
                request = dataclasses.replace(pages, row=b)
                return (
                    request.block_copy(index, 0, keys.at[slot], page_sem.at[2 * slot]),
                    request.block_copy(index, 1, values.at[slot], page_sem.at[2 * slot + 1]),
                )
            sources = (
                (key_cache[b], value_cache[b])
                if isinstance(b, int)
                else (stacked[0].at[b], stacked[1].at[b])
            )
            return (
                tpu.make_async_copy(
                    cache_block(sources[0], index), keys.at[slot], page_sem.at[2 * slot]
                ),
                tpu.make_async_copy(
                    cache_block(sources[1], index),
                    values.at[slot],
                    page_sem.at[2 * slot + 1],
                ),
            )

        # Requests' owned blocks form one sequence; block g uses bank g % buffers.
        bases = [sum(owned_blocks[:b]) for b in range(batch)]

        def start_block(sequence):
            if not traced_rows:
                for b in range(batch):

                    @pl.when((sequence >= bases[b]) & (sequence < bases[b] + owned_blocks[b]))
                    def start_own(b=b):
                        for dma in transfers(b, sequence - bases[b], sequence % buffers):
                            dma.start()

                return
            # The last request starting at or before the block owns it: a request
            # without blocks shares its base with its successor.
            request, first = jnp.int32(0), jnp.int32(0)
            for b in range(1, batch):
                later = sequence >= bases[b]
                request = jnp.where(later, b, request)
                first = jnp.where(later, bases[b], first)

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
        for b in range(batch):
            cp_queries[0, pl.ds(b * local_heads, local_heads), :] = query[b]
        with jax.named_scope("cp_query_gather"):
            exchange(0, lambda peer: cp_queries.at[0], cp_queries)
            queries[...] = jnp.zeros(queries.shape, BF16)
            for b in range(batch):
                for absolute_rank in range(cp):
                    queries[b, pl.ds(absolute_rank * local_heads, local_heads), :] = cp_queries[
                        cp_rank ^ absolute_rank, pl.ds(b * local_heads, local_heads), :
                    ]

        maximum[...] = jnp.full(maximum.shape, -jnp.inf, F32)
        mass[...] = jnp.zeros(mass.shape, F32)
        numerator[...] = jnp.zeros(numerator.shape, F32)

        def process_request(b):
            pos = position[b]
            owner = jnp.where(pos >= 0, pos // page % cp, -1)
            current_page = pos // (cp * page)
            base = bases[b]

            def append_transfers(slot):
                local_page = current_page % block_pages
                if paged:
                    # The pools are read-only here: the caller commits the new row.
                    return ()
                return (
                    tpu.make_async_copy(
                        keys.at[slot, local_page, :, :],
                        key_cache[b].at[current_page, :, :],
                        append_sem.at[0],
                    ),
                    tpu.make_async_copy(
                        values.at[slot, local_page, :, :],
                        value_cache[b].at[current_page, :, :],
                        append_sem.at[1],
                    ),
                )

            def receive(index):
                slot = (base + index) % buffers
                for dma in transfers(b, index, slot):
                    dma.wait()

                @pl.when((owner == cp_rank) & (index == owned_blocks[b] - 1))
                def append_current_row():
                    # Only the row's own page of the bank changes.
                    local_page = current_page % block_pages
                    mask = jnp.arange(page)[:, None] == pos % page
                    for bank, row, scale in (
                        (keys, key[b], scales[0]),
                        (values, value[b], scales[1]),
                    ):
                        encoded = row
                        if fp8:
                            encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(dtype)
                        if paged:
                            set_column(bank.at[slot, local_page], pos % page, encoded)
                        else:
                            tile = bank[slot, local_page]
                            bank[slot, local_page] = jnp.where(mask, encoded[None, :], tile)
                    for dma in append_transfers(slot):
                        dma.start()

            def operand(bank, index, mask=False):
                slot = (base + index) % buffers
                if paged:
                    return tuple(
                        transposed(bank.at[slot, j], live(index, j, pos) if mask else None)
                        for j in range(block_pages)
                    )
                return bank[slot, ...].reshape(block_tokens, hd)

            def valid(index):
                global_pages = (index * block_pages + jnp.arange(block_pages)) * cp + cp_rank
                positions = (global_pages[:, None] * page + jnp.arange(page)[None, :]).reshape(
                    block_tokens
                )
                return positions <= pos

            def block_scores(index):
                return tile_scores(
                    queries[b, ...], operand(keys, index), valid(index), hd**-0.5 * key_scale
                )

            def fold(index, scores):
                m, updated_mass, n = fold_scores(
                    maximum[b, :, :1],
                    mass[b, :, :1],
                    numerator[b, ...],
                    scores,
                    operand(values, index, mask=True),
                    value_scale,
                )
                maximum[b, ...] = jnp.broadcast_to(m, maximum.shape[1:])
                mass[b, ...] = jnp.broadcast_to(updated_mass, mass.shape[1:])
                numerator[b, ...] = n

            with jax.named_scope("streaming_attention_pages"):
                if pipelined:

                    @pl.when(owned_blocks[b] > 0)
                    def stream():
                        receive(jnp.int32(0))
                        carried[0] = block_scores(jnp.int32(0))

                        @pl.loop(0, owned_blocks[b] - 1)
                        def page_loop(index):
                            # The preceding block's bank was folded in the previous
                            # iteration and any earlier request's append has completed.
                            start_block(base + index + buffers - 1)
                            receive(index + 1)
                            following = block_scores(index + 1)
                            fold(index, carried[index % 2])
                            carried[(index + 1) % 2] = following

                        # The final fold also starts its block's successor in the
                        # request sequence, which may belong to the next request.
                        last = owned_blocks[b] - 1
                        start_block(base + last + buffers - 1)
                        fold(last, carried[last % 2])

                else:

                    @pl.loop(0, owned_blocks[b])
                    def page_loop(index):
                        # The bank of the preceding block is free: it was attended
                        # and any earlier request's append has completed.
                        start_block(base + index + buffers - 1)
                        receive(index)
                        fold(index, block_scores(index))

            @pl.when(owner == cp_rank)
            def complete_append():
                # Both owned-page writes must finish before the next request
                # reuses these banks; all streaming reads are already waited.
                for dma in append_transfers((base + owned_blocks[b] - 1) % buffers):
                    dma.wait()

        for b in range(batch):
            process_request(b)

        # One stats exchange for all independent requests; each peer receives
        # only its own heads.
        with jax.named_scope("cp_softmax_merge"):
            lane = jax.lax.broadcasted_iota(jnp.int32, (local_heads, 128), 1)
            for absolute_rank in range(cp):
                rows = slice(absolute_rank * local_heads, (absolute_rank + 1) * local_heads)
                packet = jnp.concatenate(
                    [
                        jnp.concatenate(
                            (
                                jnp.where(lane == 1, mass[b, rows, :], maximum[b, rows, :]),
                                numerator[b, rows, :],
                            ),
                            axis=1,
                        )
                        for b in range(batch)
                    ],
                    axis=0,
                )
                stats_out[absolute_rank, ...] = jnp.pad(
                    packet, ((0, group - batch * local_heads), (0, 0))
                )
            cp_stats[0, ...] = stats_out[cp_rank, ...]
            exchange(1, lambda peer: stats_out.at[cp_rank ^ peer], cp_stats)
            global_max = jnp.full((group, 1), -jnp.inf, F32)
            for absolute_rank in range(cp):
                packet = cp_stats[cp_rank ^ absolute_rank, ...]
                global_max = jnp.maximum(global_max, packet[:, :1])
            safe_max = jnp.where(jnp.isfinite(global_max), global_max, 0)
            total_mass = jnp.zeros((group, 1), F32)
            total_value = jnp.zeros((group, hd), F32)
            for absolute_rank in range(cp):
                packet = cp_stats[cp_rank ^ absolute_rank, ...]
                correction = jnp.where(
                    jnp.isfinite(packet[:, :1]), jnp.exp(packet[:, :1] - safe_max), 0
                )
                total_mass += packet[:, 1:2] * correction
                total_value += packet[:, 128:] * correction
            result = jnp.where(total_mass > 0, total_value / jnp.maximum(total_mass, 1e-30), 0)
            # Selecting own heads by summing zeros elsewhere made zeros positive.
            result = jnp.where(result == 0, 0.0, result)
            return jnp.stack(
                [result[b * local_heads : (b + 1) * local_heads] for b in range(batch)]
            ).astype(BF16)

    carried = {"carried": tpu.VMEM((2, all_rows, block_tokens), F32)} if pipelined else {}
    with jax.named_scope("paged_attention"):
        return pl.run_scoped(
            lambda **refs: body(**refs, **inbox),
            keys=tpu.VMEM((buffers, *block_shape), dtype),
            values=tpu.VMEM((buffers, *block_shape), dtype),
            queries=tpu.VMEM((batch, all_rows, hd), BF16),
            stats_out=tpu.VMEM((cp, group, hd + 128), F32),
            maximum=tpu.VMEM((batch, all_rows, 128), F32),
            mass=tpu.VMEM((batch, all_rows, 128), F32),
            numerator=tpu.VMEM((batch, all_rows, hd), F32),
            append_sem=tpu.SemaphoreType.DMA((2,)),
            page_sem=tpu.SemaphoreType.DMA((2 * buffers,)),
            **carried,
        )

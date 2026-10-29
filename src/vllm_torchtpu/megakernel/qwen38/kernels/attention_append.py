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

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

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


def _mask(valid):
    # A [tokens] mask serves every query row; a [rows, tokens] mask is per row.
    return valid if valid.ndim == 2 else valid[None, :]


def tile_scores(query, keys, valid, scale):
    """Masked FP32 scores of one bounded KV tile.

    FP8 keys enter the MXU as stored: a BF16 x FP8 dot pushes them without a
    vector-unit conversion and keeps exact FP32 products and sums. ``keys`` is
    ``[tokens, D]``, or a tuple of pool pages' ``[D, page]`` transposes.
    """
    if isinstance(keys, tuple):
        product = jnp.concatenate(
            [
                jax.lax.dot_general(query, page, (((1,), (0,)), ((), ())), preferred_element_type=F32)
                for page in keys
            ],
            axis=1,
        )
    else:
        product = jax.lax.dot_general(
            query, keys, (((1,), (1,)), ((), ())), preferred_element_type=F32
        )
    return jnp.where(_mask(valid), product * scale, -jnp.inf)


def _weighted_values(probability, values):
    """``probability @ values`` in FP32 (``values`` as :func:`tile_scores` takes
    keys)."""
    if isinstance(values, tuple):
        width = values[0].shape[1]
        parts = [
            jax.lax.dot_general(
                probability[:, j * width : (j + 1) * width].astype(BF16),
                page,
                (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )
            for j, page in enumerate(values)
        ]
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total
    return jnp.dot(probability.astype(BF16), values, preferred_element_type=F32)


def fold_scores(maximum, mass, numerator, scores, values, value_scale=1.0):
    """Fold one tile's masked scores into the FlashAttention recurrence in FP32.

    The probability operand narrows to BF16 only at the MXU boundary, where
    FP8 values enter as stored and ``value_scale`` multiplies their FP32
    product. The denominator retains FP32 exponentials. Masked scores are
    -inf, so their probabilities are exactly zero; a finite reference maximum
    keeps an empty tile from producing -inf - -inf, and a row without a
    visible token keeps its statistics.
    """
    updated_max = jnp.maximum(maximum, jnp.max(scores, axis=1, keepdims=True))
    safe_max = jnp.where(jnp.isfinite(updated_max), updated_max, 0)
    correction = jnp.where(jnp.isfinite(maximum), jnp.exp(maximum - safe_max), 0)
    probability = jnp.exp(scores - safe_max)
    updated_mass = mass * correction + jnp.sum(probability, axis=1, keepdims=True)
    updated_numerator = (
        numerator * correction
        + _weighted_values(probability, values) * value_scale
    )
    return updated_max, updated_mass, updated_numerator


def packet_rows(rows):
    """Rows of one peer's packet: ``rows`` heads in one aligned group."""
    return max(8, (rows + 7) // 8 * 8)


def exchange_scratch(layout, batch=1):
    """Kernel-lifetime CP8 inboxes for barrier-free calls with ``batch`` requests.

    Peers write an inbox whenever they reach the call, so it must be kernel
    scratch that no scoped allocation reuses. Consecutive calls need no
    barrier: a peer cannot send the next call's queries before it has merged
    this rank's statistics, which this rank sends only after reading its query
    inbox, and it cannot send the next statistics before receiving this rank's
    next queries, which follow this rank's merge.
    """
    rows = packet_rows(batch * layout.local_q_heads // layout.local_kv_heads)
    return _inboxes(layout.context_parallel, rows, layout.config.head_dim)


def _inboxes(cp, rows, hd):
    # Slot peer of each inbox receives from rank ^ peer; slot 0 is this rank's packet.
    return dict(
        cp_queries=tpu.VMEM((cp, rows, hd), BF16),
        cp_stats=tpu.VMEM((cp, rows, hd + 128), F32),
        cp_sends=tpu.SemaphoreType.DMA((2, max(1, cp - 1))),
        cp_recvs=tpu.SemaphoreType.DMA((2, max(1, cp - 1))),
    )


def attend_paged_append(
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
    barriers=None,
    inbox=None,
    buffers=2,
):
    """Return this rank's GQA results and update its owned KV page in-place.

    ``query`` has [local query heads for this KV head, D]. Cache references
    already select layer, request and KV head: CP1 [context,D], CP8 [pages,128,D];
    or ``key_cache`` is the request's ``paged_kv.PoolPages`` (CP8, FP8; no
    ``value_cache``).
    ``scales`` are the source cache's key/value scales, not projection scales.
    ``inbox`` holds the refs of :func:`exchange_scratch`; without it the
    exchanges use scoped inboxes between entry/exit ``barriers``. KV blocks
    rotate through ``buffers`` banks. With two banks the fold is serial and
    reads one block ahead. With more, the next block's QK is issued before this
    block's softmax and PV so that they overlap; a bank then stays live until
    its PV, one iteration after its QK, and blocks are read ``buffers - 2`` ahead.
    """
    paged = isinstance(key_cache, PoolPages)
    if paged and layout.context_parallel != 8:
        raise ValueError("Pool pages serve CP8 decode")
    if layout.context_parallel != 8:
        if inbox is not None:
            raise ValueError("Persistent CP inboxes require CP8")
        from .attention import attend_paged

        return attend_paged(
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
            barriers=barriers,
        )
    cp, page, hd = layout.context_parallel, layout.kv_page_size, layout.config.head_dim
    scoped = inbox is None
    if cp > 1 and scoped and barriers is None:
        raise ValueError("CP attention requires persistent inboxes or entry/exit barriers")
    local_heads = query.shape[0]
    total_heads = local_heads * cp
    group = packet_rows(local_heads)
    all_rows = max(8, (total_heads + 7) // 8 * 8)
    if paged:
        # A bank is one vLLM block of pool pages.
        dtype = jnp.float8_e4m3fn
        block_pages = key_cache.pages_per_block
        block_shape = bank_shape(block_pages, hd, page)
    else:
        dtype = key_cache.dtype
        # Transfer neighboring *local* pages together. Eight pages bound scratch
        # at 1024 tokens; the physical pages are contiguous even though global
        # positions are striped. Choose a divisor of capacity so the last tile is
        # always in bounds, then mask its future tokens.
        capacity_pages = key_cache.shape[0] // page if cp == 1 else key_cache.shape[0]
        block_pages = 1
        while block_pages < 8 and capacity_pages % (2 * block_pages) == 0:
            block_pages *= 2
        block_shape = (block_pages * page, hd) if cp == 1 else (block_pages, page, hd)
    block_tokens = block_pages * page
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

    def synchronize(index):
        if cp > 1:
            for peer in range(1, cp):
                pl.semaphore_signal(
                    barriers.at[index],
                    1,
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                )
            pl.semaphore_wait(barriers.at[index], cp - 1)

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
        owner = position // page % cp
        current_page = position // (cp * page)
        owned_pages = jnp.maximum(0, (position // page + cp - cp_rank) // cp)
        owned_blocks = (owned_pages + block_pages - 1) // block_pages

        def cache_block(cache, index):
            if cp == 1:
                return cache.at[pl.ds(index * block_tokens, block_tokens), :]
            return cache.at[pl.ds(index * block_pages, block_pages), :, :]

        def transfers(index, slot):
            if paged:
                return (
                    key_cache.block_copy(index, 0, keys.at[slot], page_sem.at[2 * slot]),
                    key_cache.block_copy(index, 1, values.at[slot], page_sem.at[2 * slot + 1]),
                )
            return (
                tpu.make_async_copy(
                    cache_block(key_cache, index), keys.at[slot], page_sem.at[2 * slot]
                ),
                tpu.make_async_copy(
                    cache_block(value_cache, index), values.at[slot], page_sem.at[2 * slot + 1]
                ),
            )

        def start_block(index):
            @pl.when(index < owned_blocks)
            def start():
                for dma in transfers(index, index % buffers):
                    dma.start()

        # Local page reads need no peer; they overlap the query gather.
        for index in range(buffers - 1):
            start_block(jnp.int32(index))

        if scoped:
            # Remote inbox addresses can alias a peer's preceding dense scope.
            # Persistent semaphores establish readiness before any remote write.
            synchronize(0)

        def append_transfers(slot):
            # The final streaming bank already holds the complete current page.
            # Publish that page after encoding its new row; no separate RMW DMA.
            local_page = current_page % block_pages
            if paged:
                # The pools are read-only here: the caller commits the new row.
                return ()
            return (
                tpu.make_async_copy(
                    keys.at[slot, local_page, :, :],
                    key_cache.at[current_page, :, :],
                    append_sem.at[0],
                ),
                tpu.make_async_copy(
                    values.at[slot, local_page, :, :],
                    value_cache.at[current_page, :, :],
                    append_sem.at[1],
                ),
            )

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

        cp_queries[0, ...] = jnp.pad(query, ((0, group - local_heads), (0, 0)))
        with jax.named_scope("cp_query_gather"):
            exchange(0, lambda peer: cp_queries.at[0], cp_queries)
            queries[...] = jnp.zeros(queries.shape, BF16)
            for absolute_rank in range(cp):
                queries[pl.ds(absolute_rank * local_heads, local_heads), :] = cp_queries[
                    cp_rank ^ absolute_rank, :local_heads, :
                ]

        maximum[...] = jnp.full(maximum.shape, -jnp.inf, F32)
        mass[...] = jnp.zeros(mass.shape, F32)
        numerator[...] = jnp.zeros(numerator.shape, F32)

        def receive(index):
            slot = index % buffers
            for dma in transfers(index, slot):
                dma.wait()

            @pl.when((owner == cp_rank) & (index == owned_blocks - 1))
            def append_current_row():
                # Only the row's own page of the bank changes.
                local_page = current_page % block_pages
                mask = jnp.arange(page)[:, None] == position % page
                for bank, row, scale in ((keys, key, scales[0]), (values, value, scales[1])):
                    encoded = row
                    if fp8:
                        encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(dtype)
                    if paged:
                        set_column(bank.at[slot, local_page], position % page, encoded)
                    else:
                        tile = bank[slot, local_page]
                        bank[slot, local_page] = jnp.where(mask, encoded[None, :], tile)
                for dma in append_transfers(slot):
                    dma.start()

        def operand(bank, index, mask=False):
            if paged:
                return tuple(
                    transposed(bank.at[index % buffers, j], live(index, j, position) if mask else None)
                    for j in range(block_pages)
                )
            return bank[index % buffers, ...].reshape(block_tokens, hd)

        def valid(index):
            global_pages = (index * block_pages + jnp.arange(block_pages)) * cp + cp_rank
            positions = (global_pages[:, None] * page + jnp.arange(page)[None, :]).reshape(
                block_tokens
            )
            return positions <= position

        def block_scores(index):
            return tile_scores(
                queries[...], operand(keys, index), valid(index), hd**-0.5 * key_scale
            )

        def fold(index, scores):
            m, updated_mass, n = fold_scores(
                maximum[:, :1],
                mass[:, :1],
                numerator[...],
                scores,
                operand(values, index, mask=True),
                value_scale,
            )
            maximum[...] = jnp.broadcast_to(m, maximum.shape)
            mass[...] = jnp.broadcast_to(updated_mass, mass.shape)
            numerator[...] = n

        with jax.named_scope("streaming_attention_pages"):
            if pipelined:

                @pl.when(owned_blocks > 0)
                def stream():
                    receive(jnp.int32(0))
                    carried[0] = block_scores(jnp.int32(0))

                    @pl.loop(0, owned_blocks - 1)
                    def page_loop(index):
                        # The preceding block's bank was folded in the previous iteration.
                        start_block(index + buffers - 1)
                        receive(index + 1)
                        following = block_scores(index + 1)
                        fold(index, carried[index % 2])
                        carried[(index + 1) % 2] = following

                    last = owned_blocks - 1
                    start_block(last + buffers - 1)
                    fold(last, carried[last % 2])

            else:

                @pl.loop(0, owned_blocks)
                def page_loop(index):
                    # The preceding block's bank was attended in the previous iteration.
                    start_block(index + buffers - 1)
                    receive(index)
                    fold(index, block_scores(index))

        @pl.when(owner == cp_rank)
        def complete_append():
            # Last-bank storage cannot be reused or leave scope while either
            # cache write remains in flight. All page reads have also completed.
            for dma in append_transfers((owned_blocks - 1) % buffers):
                dma.wait()

        # Each peer receives only its own heads' statistics.
        with jax.named_scope("cp_softmax_merge"):
            lane = jax.lax.broadcasted_iota(jnp.int32, (local_heads, 128), 1)
            for absolute_rank in range(cp):
                rows = slice(absolute_rank * local_heads, (absolute_rank + 1) * local_heads)
                packet = jnp.concatenate(
                    (jnp.where(lane == 1, mass[rows, :], maximum[rows, :]), numerator[rows, :]),
                    axis=1,
                )
                stats_out[absolute_rank, ...] = jnp.pad(packet, ((0, group - local_heads), (0, 0)))
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
            result = jnp.where(result == 0, 0.0, result)[:local_heads].astype(BF16)
            if scoped:
                # Commit every inbox read before acknowledging consumption.
                queries[:local_heads, :] = result
                synchronize(1)
                return queries[:local_heads, :]
            return result

    persistent, scoped_inbox = ({}, _inboxes(cp, group, hd)) if scoped else (inbox, {})
    carried = {"carried": tpu.VMEM((2, all_rows, block_tokens), F32)} if pipelined else {}
    with jax.named_scope("paged_attention"):
        return pl.run_scoped(
            lambda **refs: body(**refs, **persistent),
            keys=tpu.VMEM((buffers, *block_shape), dtype),
            values=tpu.VMEM((buffers, *block_shape), dtype),
            queries=tpu.VMEM((all_rows, hd), BF16),
            stats_out=tpu.VMEM((cp, group, hd + 128), F32),
            maximum=tpu.VMEM((all_rows, 128), F32),
            mass=tpu.VMEM((all_rows, 128), F32),
            numerator=tpu.VMEM((all_rows, hd), F32),
            append_sem=tpu.SemaphoreType.DMA((2,)),
            page_sem=tpu.SemaphoreType.DMA((2 * buffers,)),
            **scoped_inbox,
            **carried,
        )

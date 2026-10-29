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


def online_update(maximum, mass, numerator, query, keys, values, valid, scale):
    """One bounded tile of the FlashAttention recurrence, accumulating in FP32.

    The probability operand narrows to BF16 only at the MXU boundary. The
    denominator retains FP32 exponentials. Mask before exponentiation so empty
    pages cannot introduce NaNs through -inf - -inf.
    """
    # A [tokens] mask serves every query row; a [rows, tokens] mask is per row.
    mask = valid if valid.ndim == 2 else valid[None, :]
    scores = jnp.dot(query, keys.T, preferred_element_type=F32) * scale
    scores = jnp.where(mask, scores, -jnp.inf)
    updated_max = jnp.maximum(maximum, jnp.max(scores, axis=1, keepdims=True))
    safe_max = jnp.where(jnp.isfinite(updated_max), updated_max, 0)
    correction = jnp.where(jnp.isfinite(maximum), jnp.exp(maximum - safe_max), 0)
    probability = jnp.where(mask, jnp.exp(scores - safe_max), 0)
    updated_mass = mass * correction + jnp.sum(probability, axis=1, keepdims=True)
    updated_numerator = numerator * correction + jnp.dot(
        probability.astype(BF16), values, preferred_element_type=F32
    )
    return updated_max, updated_mass, updated_numerator


def attend_paged(
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
):
    """Return this rank's GQA results and update its owned KV page in-place.

    ``query`` has [local query heads for this KV head, D]. Cache references
    already select layer, request and KV head: CP1 [context,D], CP8 [pages,128,D].
    ``scales`` are the source cache's key/value scales, not projection scales.
    """
    cp, page, hd = layout.context_parallel, layout.kv_page_size, layout.config.head_dim
    if cp > 1 and barriers is None:
        raise ValueError("CP attention requires persistent entry/exit barrier semaphores")
    local_heads = query.shape[0]
    total_heads = local_heads * cp
    query_rows = max(8, (local_heads + 7) // 8 * 8)
    all_rows = max(8, (total_heads + 7) // 8 * 8)
    dtype = key_cache.dtype
    # Transfer neighboring *local* pages together. Eight pages bound scratch
    # at 1024 tokens; the physical pages are contiguous even though global
    # positions are striped. Choose a divisor of capacity so the last tile is
    # always in bounds, then mask its future tokens.
    capacity_pages = key_cache.shape[0] // page if cp == 1 else key_cache.shape[0]
    block_pages = 1
    while block_pages < 8 and capacity_pages % (2 * block_pages) == 0:
        block_pages *= 2
    legacy_short = cp == 1 and not layout.fast and capacity_pages <= 8
    if legacy_short:
        block_pages = capacity_pages
    block_tokens = block_pages * page
    block_shape = (block_tokens, hd) if cp == 1 else (block_pages, page, hd)
    cp_rank = rank % cp

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

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
        write_page,
        query_parts,
        queries,
        stats,
        maximum,
        mass,
        numerator,
        copy_sem,
        page_sem,
        sends,
        recvs,
    ):
        # Remote inbox addresses can alias a peer's preceding dense scope.
        # Persistent semaphores establish readiness before any remote write.
        synchronize(0)

        def copy(source, destination):
            dma = tpu.make_async_copy(source, destination, copy_sem)
            dma.start()
            dma.wait()

        def cache_page(cache, index):
            if cp == 1:
                return cache.at[pl.ds(index * page, page), :]
            return cache.at[index, :, :]

        owner = position // page % cp
        current_page = position // (cp * page)

        @pl.when(owner == cp_rank)
        def update_cache():
            # Whole-page RMW also satisfies FP8's packed-row DMA alignment.
            mask = jnp.arange(page)[:, None] == position % page
            for cache, row, scale in ((key_cache, key, scales[0]), (value_cache, value, scales[1])):
                target = cache_page(cache, current_page)
                copy(target, write_page)
                encoded = row
                if layout.fast:
                    encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(dtype)
                write_page[...] = jnp.where(mask, encoded[None, :], write_page[...])
                copy(write_page, target)

        query_parts[0, ...] = jnp.pad(query, ((0, query_rows - local_heads), (0, 0)))

        def exchange(buffer):
            for peer in range(1, cp):
                tpu.make_async_remote_copy(
                    buffer.at[0],
                    buffer.at[peer],
                    sends.at[peer - 1],
                    recvs.at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).start()
            for peer in range(1, cp):
                tpu.make_async_remote_copy(
                    buffer.at[0],
                    buffer.at[peer],
                    sends.at[peer - 1],
                    recvs.at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).wait()

        with jax.named_scope("cp_query_gather"):
            exchange(query_parts)
            queries[...] = jnp.zeros(queries.shape, BF16)
            for absolute_rank in range(cp):
                queries[pl.ds(absolute_rank * local_heads, local_heads), :] = query_parts[
                    cp_rank ^ absolute_rank, :local_heads, :
                ]

        maximum[...] = jnp.full(maximum.shape, -jnp.inf, F32)
        mass[...] = jnp.zeros(mass.shape, F32)
        numerator[...] = jnp.zeros(numerator.shape, F32)
        owned_pages = jnp.maximum(0, (position // page + cp - cp_rank) // cp)

        def cache_block(cache, index):
            if cp == 1:
                return cache.at[pl.ds(index * block_tokens, block_tokens), :]
            return cache.at[pl.ds(index * block_pages, block_pages), :, :]

        def transfers(index, slot):
            return (
                tpu.make_async_copy(
                    cache_block(key_cache, index), keys.at[slot], page_sem.at[2 * slot]
                ),
                tpu.make_async_copy(
                    cache_block(value_cache, index), values.at[slot], page_sem.at[2 * slot + 1]
                ),
            )

        owned_blocks = (owned_pages + block_pages - 1) // block_pages

        @pl.when(owned_blocks > 0)
        def prefetch_first():
            for dma in transfers(jnp.int32(0), jnp.int32(0)):
                dma.start()

        with jax.named_scope("streaming_attention_pages"):

            @pl.loop(0, owned_blocks)
            def page_loop(index):
                slot = index % 2
                for dma in transfers(index, slot):
                    dma.wait()

                @pl.when(index + 1 < owned_blocks)
                def prefetch_next():
                    for dma in transfers(index + 1, 1 - slot):
                        dma.start()

                k = keys[slot, ...].reshape(block_tokens, hd).astype(BF16)
                v = values[slot, ...].reshape(block_tokens, hd).astype(BF16)
                if layout.fast:
                    k = (k.astype(F32) * scales[0]).astype(BF16)
                    v = (v.astype(F32) * scales[1]).astype(BF16)
                global_pages = (index * block_pages + jnp.arange(block_pages)) * cp + cp_rank
                positions = (global_pages[:, None] * page + jnp.arange(page)[None, :]).reshape(
                    block_tokens
                )
                if legacy_short:
                    # Preserve the validated native CP1 cast order whenever
                    # the complete cache fits a bounded <=1024-token tile.
                    # The long/CP path rounds unnormalized exponentials at
                    # its streaming PV MXU; this path rounds normalized P.
                    scores = jnp.dot(queries[...], k.T, preferred_element_type=F32) * hd**-0.5
                    scores = jnp.where(positions[None, :] <= position, scores, -jnp.inf)
                    probability = jax.nn.softmax(scores, axis=1).astype(BF16)
                    maximum[...] = jnp.zeros(maximum.shape, F32)
                    mass[...] = jnp.ones(mass.shape, F32)
                    numerator[...] = jnp.dot(probability, v, preferred_element_type=F32)
                else:
                    m, updated_mass, n = online_update(
                        maximum[:, :1],
                        mass[:, :1],
                        numerator[...],
                        queries[...],
                        k,
                        v,
                        positions <= position,
                        hd**-0.5,
                    )
                    maximum[...] = jnp.broadcast_to(m, maximum.shape)
                    mass[...] = jnp.broadcast_to(updated_mass, mass.shape)
                    numerator[...] = n

        # Query and statistics exchanges alternate. The following query
        # exchange cannot complete until every peer has consumed these stats,
        # which protects inbox reuse across requests and layers.
        with jax.named_scope("cp_softmax_merge"):
            stats[0, ...] = jnp.concatenate((maximum[...], numerator[...]), axis=1)
            stats[0, :, 1] = mass[:, 0]
            exchange(stats)
            global_max = jnp.full((all_rows, 1), -jnp.inf, F32)
            for absolute_rank in range(cp):
                packet = stats[cp_rank ^ absolute_rank, ...]
                global_max = jnp.maximum(global_max, packet[:, :1])
            safe_max = jnp.where(jnp.isfinite(global_max), global_max, 0)
            total_mass = jnp.zeros((all_rows, 1), F32)
            total_value = jnp.zeros((all_rows, hd), F32)
            for absolute_rank in range(cp):
                packet = stats[cp_rank ^ absolute_rank, ...]
                correction = jnp.where(
                    jnp.isfinite(packet[:, :1]), jnp.exp(packet[:, :1] - safe_max), 0
                )
                total_mass += packet[:, 1:2] * correction
                total_value += packet[:, 128:] * correction
            result = jnp.where(total_mass > 0, total_value / jnp.maximum(total_mass, 1e-30), 0)
            heads = result[:total_heads].reshape(cp, local_heads, hd)
            selected = jnp.sum(
                jnp.where(jnp.arange(cp)[:, None, None] == cp_rank, heads, 0), axis=0
            ).astype(BF16)
            # Commit every inbox read before acknowledging consumption. The
            # output buffer is local and is not a target of remote transfers.
            queries[:local_heads, :] = selected
            synchronize(1)
            return queries[:local_heads, :]

    with jax.named_scope("paged_attention"):
        return pl.run_scoped(
            body,
            keys=tpu.VMEM((2, *block_shape), dtype),
            values=tpu.VMEM((2, *block_shape), dtype),
            write_page=tpu.VMEM((page, hd), dtype),
            query_parts=tpu.VMEM((cp, query_rows, hd), BF16),
            queries=tpu.VMEM((all_rows, hd), BF16),
            stats=tpu.VMEM((cp, all_rows, hd + 128), F32),
            maximum=tpu.VMEM((all_rows, 128), F32),
            mass=tpu.VMEM((all_rows, 128), F32),
            numerator=tpu.VMEM((all_rows, hd), F32),
            copy_sem=tpu.SemaphoreType.DMA,
            page_sem=tpu.SemaphoreType.DMA((4,)),
            sends=tpu.SemaphoreType.DMA((max(1, cp - 1),)),
            recvs=tpu.SemaphoreType.DMA((max(1, cp - 1),)),
        )

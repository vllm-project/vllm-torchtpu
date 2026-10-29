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
"""Two-position causal full attention with shared bounded KV DMA tiles."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from .attention import online_update

F32 = jnp.float32
BF16 = jnp.bfloat16


def attend_paged_causal_pair(
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
    *,
    _row_update=None,
):
    """Attend two consecutive positions of one request, reading each KV tile once.

    Query is [2, local_heads, D]; caches are ONE request [pages,128,D].
    Positions must be p and p+1, constructed by the block-verifier wrapper.
    Each row retains its own causal mask, 1024-token softmax arithmetic and
    ascending CP merge. This is not the independent-request C2 helper.

    The internal optional update serves only the qualified arithmetic geometry
    [two rows,16 gathered heads,256 dimensions,1024 tile tokens]. Other shapes
    retain the original per-row update, including shorter context capacities.
    """
    if _row_update is not None and not callable(_row_update):
        raise ValueError("Internal causal row update must be callable or None")
    cp, page, hd = layout.context_parallel, layout.kv_page_size, layout.config.head_dim
    if cp > 1 and barriers is None:
        raise ValueError("CP attention requires persistent entry/exit barrier semaphores")
    batch, local_heads = query.shape[:2]
    if batch != 2 or cp != 8:
        raise ValueError("Causal pair attention requires two positions and CP8")
    total_heads = local_heads * cp
    query_rows = max(8, (local_heads + 7) // 8 * 8)
    all_rows = max(8, (total_heads + 7) // 8 * 8)
    dtype = key_cache.dtype
    # Transfer neighboring *local* pages together. Eight pages bound scratch
    # at 1024 tokens; the physical pages are contiguous even though global
    # positions are striped. Choose a divisor of capacity so the last tile is
    # always in bounds, then mask its future tokens.
    capacity_pages = key_cache.shape[0]
    block_pages = 1
    while block_pages < 8 and capacity_pages % (2 * block_pages) == 0:
        block_pages *= 2
    block_tokens = block_pages * page
    block_shape = (block_tokens, hd) if cp == 1 else (block_pages, page, hd)
    cp_rank = rank % cp
    combined_rows = _row_update is not None and (all_rows, hd, block_tokens) == (16, 256, 1024)

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
        query_parts,
        queries,
        stats,
        maximum,
        mass,
        numerator,
        append_sem,
        page_sem,
        sends,
        recvs,
    ):
        # Remote inbox addresses can alias a peer's preceding dense scope.
        # Persistent semaphores establish readiness before any remote write.
        synchronize(0)

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

        for b in range(batch):
            query_parts[0, b, ...] = jnp.pad(query[b], ((0, query_rows - local_heads), (0, 0)))
        with jax.named_scope("cp_query_gather"):
            exchange(query_parts)
            queries[...] = jnp.zeros(queries.shape, BF16)
            for b in range(batch):
                for absolute_rank in range(cp):
                    queries[b, pl.ds(absolute_rank * local_heads, local_heads), :] = query_parts[
                        cp_rank ^ absolute_rank, b, :local_heads, :
                    ]

        maximum[...] = jnp.full(maximum.shape, -jnp.inf, F32)
        mass[...] = jnp.zeros(mass.shape, F32)
        numerator[...] = jnp.zeros(numerator.shape, F32)

        # Both current rows are merged into their owned page before either
        # query reads the tile. The first query masks the second row explicitly.
        # Consecutive positions can share a page or lie on adjacent CP owners;
        # each lane therefore writes at most one local page per iteration.
        first_pos, last_pos = position[0], position[1]
        owners = (first_pos // page % cp, last_pos // page % cp)
        pages = (first_pos // (page * cp), last_pos // (page * cp))
        owned_pages = jnp.maximum(0, (last_pos // page + cp - cp_rank) // cp)
        owned_blocks = (owned_pages + block_pages - 1) // block_pages
        write_page = jnp.where(owners[0] == cp_rank, pages[0], pages[1])
        has_write = (owners[0] == cp_rank) | (owners[1] == cp_rank)

        def transfers(index, slot):
            start = pl.ds(index * block_pages, block_pages)
            return (
                tpu.make_async_copy(
                    key_cache.at[start, :, :], keys.at[slot], page_sem.at[2 * slot]
                ),
                tpu.make_async_copy(
                    value_cache.at[start, :, :], values.at[slot], page_sem.at[2 * slot + 1]
                ),
            )

        def append_transfers(slot):
            local_page = write_page % block_pages
            return (
                tpu.make_async_copy(
                    keys.at[slot, local_page, :, :],
                    key_cache.at[write_page, :, :],
                    append_sem.at[0],
                ),
                tpu.make_async_copy(
                    values.at[slot, local_page, :, :],
                    value_cache.at[write_page, :, :],
                    append_sem.at[1],
                ),
            )

        @pl.when(owned_blocks > 0)
        def prefetch_first():
            for dma in transfers(jnp.int32(0), jnp.int32(0)):
                dma.start()

        with jax.named_scope("causal_pair_streaming_attention_pages"):

            @pl.loop(0, owned_blocks)
            def page_loop(index):
                slot = index % 2
                for dma in transfers(index, slot):
                    dma.wait()

                @pl.when(index + 1 < owned_blocks)
                def prefetch_next():
                    for dma in transfers(index + 1, 1 - slot):
                        dma.start()

                def append_position(b):
                    @pl.when((owners[b] == cp_rank) & (index == pages[b] // block_pages))
                    def append_row():
                        offset = (pages[b] % block_pages) * page + position[b] % page
                        mask = jnp.arange(block_tokens)[:, None] == offset
                        for bank, row, scale in (
                            (keys, key[b], scales[0]),
                            (values, value[b], scales[1]),
                        ):
                            encoded = row
                            if layout.fast:
                                encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(dtype)
                            tile = bank[slot, ...].reshape(block_tokens, hd)
                            bank[slot, ...] = jnp.where(mask, encoded[None, :], tile).reshape(
                                block_shape
                            )

                for b in range(batch):
                    append_position(b)

                @pl.when(has_write & (index == write_page // block_pages))
                def start_append():
                    for dma in append_transfers(slot):
                        dma.start()

                # These immutable dequantized operands serve both causal rows.
                # The default keeps each dot at the incumbent head-row shape.
                # Optional combined arithmetic is restricted to its qualified geometry.
                k = keys[slot, ...].reshape(block_tokens, hd).astype(BF16)
                v = values[slot, ...].reshape(block_tokens, hd).astype(BF16)
                if layout.fast:
                    k = (k.astype(F32) * scales[0]).astype(BF16)
                    v = (v.astype(F32) * scales[1]).astype(BF16)
                global_pages = (index * block_pages + jnp.arange(block_pages)) * cp + cp_rank
                token_positions = (
                    global_pages[:, None] * page + jnp.arange(page)[None, :]
                ).reshape(block_tokens)

                if combined_rows:
                    row_positions = jnp.stack((first_pos, last_pos))
                    row_pages = jnp.maximum(0, (row_positions // page + cp - cp_rank) // cp)
                    row_blocks = (row_pages + block_pages - 1) // block_pages
                    m, updated_mass, n = _row_update(
                        maximum[..., :1], mass[..., :1], numerator[...], queries[...],
                        k, v, token_positions, row_positions, index < row_blocks, hd**-0.5,
                    )
                    maximum[...] = jnp.broadcast_to(m, maximum.shape)
                    mass[...] = jnp.broadcast_to(updated_mass, mass.shape)
                    numerator[...] = n
                else:
                    def attend_position(b):
                        row_pages = jnp.maximum(0, (position[b] // page + cp - cp_rank) // cp)
                        row_blocks = (row_pages + block_pages - 1) // block_pages

                        @pl.when(index < row_blocks)
                        def attend_row():
                            m, updated_mass, n = online_update(
                                maximum[b, :, :1],
                                mass[b, :, :1],
                                numerator[b, ...],
                                queries[b, ...],
                                k,
                                v,
                                token_positions <= position[b],
                                hd**-0.5,
                            )
                            maximum[b, ...] = jnp.broadcast_to(m, maximum.shape[1:])
                            mass[b, ...] = jnp.broadcast_to(updated_mass, mass.shape[1:])
                            numerator[b, ...] = n

                    for b in range(batch):
                        attend_position(b)

                @pl.when(has_write & (index == write_page // block_pages))
                def complete_append():
                    # Source banks cannot be reused until both writes finish.
                    for dma in append_transfers(slot):
                        dma.wait()

        # One stats exchange for both causal positions. Entry/exit barriers
        # still protect scoped inbox allocation and consumption on every rank.
        with jax.named_scope("cp_softmax_merge"):
            for b in range(batch):
                stats[0, b, ...] = jnp.concatenate((maximum[b, ...], numerator[b, ...]), axis=1)
                stats[0, b, :, 1] = mass[b, :, 0]
            exchange(stats)
            for b in range(batch):
                global_max = jnp.full((all_rows, 1), -jnp.inf, F32)
                for absolute_rank in range(cp):
                    packet = stats[cp_rank ^ absolute_rank, b, ...]
                    global_max = jnp.maximum(global_max, packet[:, :1])
                safe_max = jnp.where(jnp.isfinite(global_max), global_max, 0)
                total_mass = jnp.zeros((all_rows, 1), F32)
                total_value = jnp.zeros((all_rows, hd), F32)
                for absolute_rank in range(cp):
                    packet = stats[cp_rank ^ absolute_rank, b, ...]
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
                queries[b, :local_heads, :] = selected
            synchronize(1)
            return queries[:, :local_heads, :]

    with jax.named_scope("paged_attention"):
        return pl.run_scoped(
            body,
            keys=tpu.VMEM((2, *block_shape), dtype),
            values=tpu.VMEM((2, *block_shape), dtype),
            query_parts=tpu.VMEM((cp, batch, query_rows, hd), BF16),
            queries=tpu.VMEM((batch, all_rows, hd), BF16),
            stats=tpu.VMEM((cp, batch, all_rows, hd + 128), F32),
            maximum=tpu.VMEM((batch, all_rows, 128), F32),
            mass=tpu.VMEM((batch, all_rows, 128), F32),
            numerator=tpu.VMEM((batch, all_rows, hd), F32),
            append_sem=tpu.SemaphoreType.DMA((2,)),
            page_sem=tpu.SemaphoreType.DMA((4,)),
            sends=tpu.SemaphoreType.DMA((max(1, cp - 1),)),
            recvs=tpu.SemaphoreType.DMA((max(1, cp - 1),)),
        )

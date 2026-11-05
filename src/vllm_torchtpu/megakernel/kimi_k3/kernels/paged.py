# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The serving step of the Kimi K3 decode megakernel over vLLM's unified pools.

vLLM keeps each MLA layer's latent pages in its unified pools (``[blocks, 128,
2, 640]`` bf16, token ``t`` of a block at ``[t // 2, t % 2]``; FP8 packs 512
tokens), each DCP rank holding its share of every sequence's pages, and the
KDA state in per-request slots. Two pieces serve it:

* :func:`decode_paged` -- one decode step for ``batch_size`` rows: gathers the
  rows' KDA slots, runs the fused decoder with the pools as read-only
  operands (``decode_local``), scatters the KDA rows back and commits the new
  latent rows with :func:`commit_latent_rows` (and, for Kimi-K3.1, the rows'
  index keys into its block-major index cache with
  :func:`commit_index_keys`). The fused decoder communicates across devices,
  and XLA copies every operand such a kernel updates in place (a full copy of
  the pools per step), so the only in-place pool update is the separate,
  non-communicating commit.
* :func:`commit_latent_rows` -- writes each row's new latent on the rank that
  owns its position; :func:`commit_index_keys` does the same for Kimi-K3.1's
  index keys, in their own cache.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_decode, mla_dcp
from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
    MLA_LAYERS,
    decode_local,
)


def latent_rows_from_handoff(new_keys):
    """``[batch, M, 640]`` latent rows from a DCP decoder's handoff (packed in
    new-key rows 0..2, see ``decode_megakernel.mla``)."""
    return jnp.concatenate((new_keys[:, :, 0], new_keys[:, :, 1], new_keys[:, :, 2, :128]), axis=-1)


def _commit_latent_rows_fused(pools, rows, positions, blocks, owned, pool_map, index_keys=None):
    """Alias all unified pools in one non-communicating cache commit.

    Overlap the layers' native-tile reads. Retire all writes before
    advancing to the next token so rows sharing a tile preserve each other.
    ``index_keys``: Kimi-K3.1's ``(index_cache, key_bytes, exponents)``
    (:func:`dsa_cache.commit_index_rows`), committed in the same row loop,
    each row's key tile and its page's scale tile updated and retired like
    the latent tiles; returns ``(pools, index_cache)`` then.
    """
    batch, layers, _ = rows.shape
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = 8 * packing
    metadata = jnp.stack((positions[:batch], blocks, owned.astype(jnp.int32)))
    count = len(pools)
    scale_rows = dsa_cache.SCALE_ROWS
    index_cache = key_bytes = None
    if index_keys is not None:
        index_cache, key_bytes, exponents = index_keys
        if index_cache.shape[2] - scale_rows != page_tokens or index_cache.shape[1] != layers:
            raise ValueError("the index cache's pages must match the latent pages")
        # Row 3 + m holds each row's MLA layer m exponent byte.
        metadata = jnp.concatenate((metadata, exponents[:batch].astype(jnp.int32).T))

    def kernel(meta, values, *refs):
        if index_keys is not None:
            keys, refs = refs[0], refs[1:]
            outputs, cache = refs[count + 1:2 * count + 1], refs[2 * count + 1]
            key_tiles, scale_tiles, index_semaphores = refs[-5:-2]
        else:
            outputs = refs[count:2 * count]
        tiles, semaphores = refs[-2:]

        def commit_index_row(row, position, block):
            size = cache.shape[0]
            target = jnp.where(block < 0, block + size, block)
            slot = position % page_tokens
            first = pl.multiple_of(slot // scale_rows * scale_rows, scale_rows)

            def key_copy(layer, reading):
                tile, rows_ = key_tiles.at[layer], cache.at[target, layer, pl.ds(first, scale_rows)]
                return tpu.make_async_copy(*((rows_, tile) if reading else (tile, rows_)),
                                           index_semaphores.at[0, layer])

            def scale_copy(layer, reading):
                tile, rows_ = scale_tiles.at[layer], cache.at[target, layer, pl.ds(page_tokens, scale_rows)]
                return tpu.make_async_copy(*((rows_, tile) if reading else (tile, rows_)),
                                           index_semaphores.at[1, layer])

            @pl.when((target >= 0) & (target < size))
            def commit():
                for layer in range(layers):
                    key_copy(layer, True).start()
                    scale_copy(layer, True).start()
                shape = (scale_rows, 128)
                tile_row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
                lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
                for layer in range(layers):
                    key_copy(layer, True).wait()
                    scale_copy(layer, True).wait()
                    key_tiles[layer] = jnp.where(tile_row == slot % scale_rows, keys[pl.ds(row, 1), layer, :],
                                                 key_tiles[layer].astype(jnp.int32)).astype(jnp.uint8)
                    scale_tiles[layer] = jnp.where((tile_row == slot // 128) & (lane == slot % 128),
                                                   meta[3 + layer, row],
                                                   scale_tiles[layer].astype(jnp.int32)).astype(jnp.uint8)
                    key_copy(layer, False).start()
                    scale_copy(layer, False).start()
                for layer in range(layers):
                    key_copy(layer, False).wait()
                    scale_copy(layer, False).wait()

        @pl.loop(0, batch)
        def commit_row(row):
            @pl.when(meta[2, row] != 0)
            def owned_row():
                position, block = meta[0, row], meta[1, row]
                first = pl.multiple_of((position % page_tokens // tile_tokens) * 8, 8)

                def normalized_block(index):
                    size = outputs[index].shape[0]
                    target = jnp.where(block < 0, block + size, block)
                    return target, (target >= 0) & (target < size)

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def read(layer=layer, index=index, target=target):
                        tpu.make_async_copy(outputs[index].at[target, pl.ds(first, 8)],
                                            tiles.at[layer], semaphores.at[layer]).start()

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def update(layer=layer, index=index, target=target):
                        copy = tpu.make_async_copy(outputs[index].at[target, pl.ds(first, 8)],
                                                   tiles.at[layer], semaphores.at[layer])
                        copy.wait()
                        token = (packing * jax.lax.broadcasted_iota(jnp.int32, (8, packing, 640), 0)
                                 + jax.lax.broadcasted_iota(jnp.int32, (8, packing, 640), 1))
                        tiles[layer] = jnp.where(token == position % tile_tokens,
                                                values[row, layer][None, None, :], tiles[layer])
                        tpu.make_async_copy(tiles.at[layer],
                                            outputs[index].at[target, pl.ds(first, 8)],
                                            semaphores.at[layer]).start()

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def retire(layer=layer, index=index, target=target):
                        tpu.make_async_copy(tiles.at[layer],
                                            outputs[index].at[target, pl.ds(first, 8)],
                                            semaphores.at[layer]).wait()

                if index_keys is not None:
                    commit_index_row(row, position, block)

    scratch = (tpu.VMEM((layers, 8, packing, 640), dtype), tpu.SemaphoreType.DMA((layers,)))
    if index_cache is None or key_bytes is None:
        return tuple(pl.pallas_call(
            kernel, out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),)
                         + (pl.BlockSpec(memory_space=tpu.HBM),) * count,
                out_specs=(pl.BlockSpec(memory_space=tpu.HBM),) * count,
                scratch_shapes=scratch),
            input_output_aliases={2 + i: i for i in range(count)},
            name='kimi_commit_unified_pools')(metadata, rows.astype(dtype), *pools))
    index_scratch = (tpu.VMEM((layers, scale_rows, 128), jnp.uint8),
                     tpu.VMEM((layers, scale_rows, 128), jnp.uint8),
                     tpu.SemaphoreType.DMA((2, layers)))
    outputs = pl.pallas_call(
        kernel, out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in (*pools, index_cache)),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),) * 2
                     + (pl.BlockSpec(memory_space=tpu.HBM),) * (count + 1),
            out_specs=(pl.BlockSpec(memory_space=tpu.HBM),) * (count + 1),
            scratch_shapes=index_scratch + scratch),
        input_output_aliases={3 + i: i for i in range(count + 1)},
        name='kimi_commit_unified_pools_index')(
            metadata, rows.astype(dtype), key_bytes[:batch].astype(jnp.int32), *pools, index_cache)
    return tuple(outputs[:count]), outputs[count]


def commit_latent_rows(pools, rows, block_tables, positions, pool_map, *, group=mla_dcp.GROUP,
                       owner_table=None, global_pages=None, index_keys=None):
    """Write each row's new latent into vLLM's unified pools on the rank that
    owns its position (XLA level, inside ``shard_map`` over ``tp``): MLA layer
    ``m``'s row at position ``p`` lands in ``pools[pool_map[m]]`` at token
    ``p % page_tokens`` of block ``block_tables[b, e]``, ``e = p // page_tokens``
    (global tables) or ``p // (page_tokens group)`` (local tables).
    ``owner_table``: as the decoder's ``dcp_owner_table`` (default the mesh
    position). ``global_pages``: default ``mla_dcp.GLOBAL_PAGES``.
    ``index_keys``: also commit Kimi-K3.1's ``(index_cache, key_bytes,
    exponents)`` (:func:`dsa_cache.commit_index_rows`) and return ``(pools,
    index_cache)``."""
    if global_pages is None:
        global_pages = mla_dcp.GLOBAL_PAGES
    member = mla_dcp.owner_index(owner_table) % group
    pools = list(pools)
    packing = pools[0].shape[-2]
    page_tokens = pools[0].shape[-3] * packing
    pages = positions[:rows.shape[0]] // page_tokens
    owned = pages % group == member
    # Local tables hold one entry per ``group`` global pages.
    blocks = block_tables[jnp.arange(rows.shape[0]), pages if global_pages else pages // group]
    # One fused commit for all MLA layers; the generic scatter below handles
    # other layouts and repeated pool maps (ordered by layer, not token).
    if (_config().fused_cache_commit
            and rows.shape[0] in (1, 2, 4, 8, 16, 24, 32, 48, 64)
            and rows.shape[1:] == (len(pool_map), 640)
            and 0 < len(pool_map) <= MLA_LAYERS and len(set(pool_map)) == len(pool_map)
            and all(0 <= index < len(pools) for index in pool_map)
            and all(pool.shape[0] > 0 and pool.shape[1:] == (128, packing, 640)
                    and pool.dtype == pools[0].dtype for pool in pools)):
        return _commit_latent_rows_fused(pools, rows, positions, blocks, owned, pool_map, index_keys)
    pairs = positions[:rows.shape[0]] % page_tokens // packing
    halves = positions[:rows.shape[0]] % packing
    # One scatter per pool for all rows; rows this rank does not own point
    # past the pool and are dropped.
    for layer, index in enumerate(pool_map):
        pool = pools[index]
        targets = jnp.where(owned, blocks, pool.shape[0])
        pools[index] = pool.at[targets, pairs, halves].set(rows[:, layer].astype(pool.dtype), mode="drop")
    if index_keys is not None:
        index_cache, key_bytes, exponents = index_keys
        return tuple(pools), dsa_cache.commit_index_rows(index_cache, key_bytes, exponents, block_tables,
                                                          positions, member=member, group=group,
                                                          global_pages=global_pages)
    return tuple(pools)


def _commit_kda_states(conv_slots, recurrent_slots, conv, recurrent, slots, source_rows=None):
    """Copy complete KDA rows back without XLA's per-row scatter loop.

    Row writes finish in input order, including repeated null-slot indices.
    The banks alias their outputs so untouched slots retain their bytes.
    With ``source_rows``, DMA only the selected verifier post-states directly
    from their HBM buffer; do not gather a second intermediate with XLA.
    """
    batch = slots.shape[0]
    selected = source_rows is not None
    prefetch = 2 if selected else 1

    def body(indices, *args):
        sources, args = (args[0], args[1:]) if selected else (None, args)
        (old_conv, old_state, new_conv, new_state, out_conv, out_state,
         conv_buffer, state_buffer, semaphores) = args
        @pl.loop(0, batch)
        def row(i):
            source = sources[i] if sources is not None else i
            reads = (
                tpu.make_async_copy(new_conv.at[pl.ds(source, 1)], conv_buffer, semaphores.at[0]),
                tpu.make_async_copy(new_state.at[pl.ds(source, 1)], state_buffer, semaphores.at[1]),
            )
            for item in reads:
                item.start()
            for item in reads:
                item.wait()
            target = indices[i]
            writes = (
                tpu.make_async_copy(conv_buffer, out_conv.at[pl.ds(target, 1)], semaphores.at[0]),
                tpu.make_async_copy(state_buffer, out_state.at[pl.ds(target, 1)], semaphores.at[1]),
            )
            for item in writes:
                item.start()
            for item in writes:
                item.wait()

    return pl.pallas_call(
        body,
        name="kda_selected_state_commit" if selected else "kda_state_commit",
        out_shape=(jax.ShapeDtypeStruct(conv_slots.shape, conv_slots.dtype),
                   jax.ShapeDtypeStruct(recurrent_slots.shape, recurrent_slots.dtype)),
        input_output_aliases={prefetch: 0, prefetch + 1: 1},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=prefetch,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 4,
            out_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 2,
            scratch_shapes=(
                tpu.VMEM((1, *conv.shape[1:]), conv.dtype),
                tpu.VMEM((1, *recurrent.shape[1:]), recurrent.dtype),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
    )(slots, *((source_rows,) if selected else ()), conv_slots, recurrent_slots, conv, recurrent)

def decode_paged(weights, conv_slots, recurrent_slots, pool, tokens, positions, block_tables,
                 slots, *, batch_size, dcp_pool_map, dcp_owner_table=None, state_row=None,
                 defer_kda_commit=False, index_cache=None, **decode_options):
    """One decode step on this rank (inside a ``tp`` shard_map).

    ``conv_slots [S, 69, 3, 3, 3, 128]`` / ``recurrent_slots [S, 69, 3, 128,
    128]`` hold every request's KDA state; row ``b`` uses slot ``slots[b]``
    (padding rows the null slot, pages the null page). ``pool`` is the tuple
    of vLLM's unified pools (MLA layer ``m`` in ``pool[dcp_pool_map[m]]``),
    read in place; ``block_tables`` are the DCP tables. Returns ``(logits
    [batch, V/32], conv_slots, recurrent_slots, pool)`` with the slots and new
    latent rows committed (callers donate them); ``aux_hidden_layers`` adds
    ``aux [L, batch, hidden]`` after the logits.

    Speculative verify: the rows are one sequence's anchor and drafts;
    ``state_row`` (traced) picks the row whose input KDA state the sequence
    continues from. Every row's post-state and latent are committed; rejected
    drafts lie past the next step's pool limit. ``defer_kda_commit``
    (verifiers): return per-row post-states (under attention DP this group's
    rows, shard-major); the caller commits accepted rows.

    ``index_cache``: Kimi-K3.1's index-key cache ``[blocks, 24, tokens + 32,
    128]`` uint8 (``dsa_cache``); the rows' new index keys are committed into
    it with :func:`commit_index_keys` (a verify's every row, as its latents)
    and it is returned after ``pool``
    (callers donate it). It is also the stack's read-only operand (the scorer
    reads the cached keys), and with ``index_debug=True`` the indexer's check
    outputs follow it.
    """
    if tokens.shape != (batch_size,) or positions.shape != (batch_size,) or slots.shape != (batch_size,):
        raise ValueError("tokens, positions and slots must have one entry per row")
    dp = decode_options.get("attention_dp", 1)
    if index_cache is not None and (batch_size % dp or batch_size // dp not in dsa_decode.BATCH_CLASSES):
        raise NotImplementedError(f"K3.1 index keys: {dsa_decode.BATCH_CLASSES} rows a DP group")
    group_rows = None
    state_index = slots
    if dp > 1:
        # This rank's DP group attends its rows of each local row owner's rows
        # (see decode_megakernel.stack); KDA state row k * G + p holds shard
        # k of group row p, at bank entry ``s + k * n`` for the row's slot s
        # of the group's n slots (G = batch / dp; pools.group_slots).
        owners = decode_options.get("sequence_parallel_size", 4)
        per_pair = batch_size // dp // owners
        data_group = jax.lax.axis_index("tp") // (32 // dp)
        row = jnp.arange(batch_size // dp)
        group_rows = row // per_pair * (batch_size // owners) + data_group * per_pair + row % per_pair
        group_slots = conv_slots.shape[0] // dp
        local = slots[group_rows]
        local = jnp.where(local == 0, 0, local - data_group * (group_slots - 1))
        state_index = jnp.concatenate([local + k * group_slots for k in range(dp)])
    # The stack reads the input KDA states from their slots itself where it
    # can (DP1 above four rows or a selected verify row; DP2 at B2..B128,
    # DP4 at B32..B128). DP indices already include each held head shard's
    # bank offset.
    in_place = ((state_row is not None or batch_size > 4) and dp == 1
                or dp == 2 and batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
                or dp == 4 and batch_size in (32, 64, 96, 128))
    if in_place:
        conv, recurrent = conv_slots, recurrent_slots
    else:
        conv, recurrent = conv_slots[state_index], recurrent_slots[state_index]
    pools = tuple(pool)
    result = decode_local(
        weights, (conv, recurrent, *pools), tokens, positions, batch_size=batch_size,
        block_tables=block_tables, dcp_pool_map=dcp_pool_map, dcp_owner_table=dcp_owner_table,
        state_row=state_row, state_slots=state_index if in_place else None,
        live_rows=slots != 0, index_cache=index_cache, **decode_options)
    # (logits, [aux,] states): the states are the four, then the check outputs.
    logits, states = result[0], result[-1]
    aux = result[1] if decode_options.get("aux_hidden_layers") else None
    conv, recurrent, new_keys, new_values = states[:4]
    index_debug = states[4:]
    # Attention DP returns shard-major rows; commit each to its shard-bank
    # index, preserving untouched slots and ordered null-slot writes.
    if defer_kda_commit:
        # Keep the input banks alive until acceptance. Return the native
        # post-states without touching persistent slots or gathering rows.
        conv_slots, recurrent_slots = conv, recurrent
    elif batch_size > 16 and dp in (1, 2, 4):
        conv_slots, recurrent_slots = _commit_kda_states(
            conv_slots, recurrent_slots, conv, recurrent, state_index)
    else:
        conv_slots = conv_slots.at[state_index].set(conv)
        recurrent_slots = recurrent_slots.at[state_index].set(recurrent)
    latents = latent_rows_from_handoff(new_keys)
    if group_rows is not None:
        # The handoff holds the group's rows first; only their owners commit.
        latents, block_tables, positions = (
            latents[: group_rows.shape[0]], block_tables[group_rows], positions[group_rows])
    if index_cache is not None and _config().fused_index_commit:
        # The MLA layers hand their encoded index keys out in new-value rows 0 and 1.
        key_bytes, exponents = dsa_cache.index_key_bytes(new_values[:, :, 0], new_values[:, :, 1])
        if group_rows is not None:  # the group's rows, as the latents above
            key_bytes, exponents = key_bytes[: group_rows.shape[0]], exponents[: group_rows.shape[0]]
        pool, index_cache = commit_latent_rows(pools, latents, block_tables, positions, dcp_pool_map,
                                               owner_table=dcp_owner_table,
                                               index_keys=(index_cache, key_bytes, exponents))
        tail = (pool, index_cache, *index_debug)
    else:
        pool = commit_latent_rows(pools, latents, block_tables, positions,
                                  dcp_pool_map, owner_table=dcp_owner_table)
        tail = (pool,)
    if index_cache is not None and not _config().fused_index_commit:
        # The MLA layers hand their encoded index keys out in new-value rows 0 and 1.
        key_bytes, exponents = dsa_cache.index_key_bytes(new_values[:, :, 0], new_values[:, :, 1])
        if group_rows is not None:  # the group's rows, as the latents above
            key_bytes, exponents = key_bytes[: group_rows.shape[0]], exponents[: group_rows.shape[0]]
        tail += (commit_index_keys(index_cache, key_bytes, exponents, block_tables, positions,
                                   owner_table=dcp_owner_table),)
        tail += tuple(index_debug)
    if aux is not None:
        return logits[0], aux, conv_slots, recurrent_slots, *tail
    return logits[0], conv_slots, recurrent_slots, *tail


def commit_index_keys(index_cache, key_bytes, exponents, block_tables, positions, *, owner_table=None):
    """Write each row's encoded index keys (``key_bytes [batch, 24, 128]``,
    ``exponents [batch, 24]`` uint8) into Kimi-K3.1's index-key cache on the
    rank that owns its position, at the token its latent row takes
    (:func:`commit_latent_rows`; inside ``shard_map`` over ``tp``)."""
    return dsa_cache.commit_index_rows(
        index_cache, key_bytes, exponents, block_tables, positions,
        member=mla_dcp.owner_index(owner_table) % mla_dcp.GROUP, group=mla_dcp.GROUP,
        global_pages=mla_dcp.GLOBAL_PAGES)

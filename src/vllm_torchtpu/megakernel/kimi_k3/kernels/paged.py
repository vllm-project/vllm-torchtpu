# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The serving step of the Kimi K3 decode megakernel over vLLM's unified pools.

vLLM keeps each MLA layer's latent pages in its unified pools (``[blocks, 128,
2, 640]`` bf16, token ``t`` of a block at ``[t // 2, t % 2]``; FP8 packs 512
tokens), each DCP rank holding its share of every sequence's pages, and the
KDA state in per-request slots. Two pieces serve it:

* :func:`decode_paged` -- one decode step for ``batch_size`` rows: gathers the
  rows' KDA slots, runs the fused decoder with the pools as read-only
  operands (``decode_local``), scatters the KDA rows back and commits the new
  latent rows with :func:`commit_latent_rows`. The fused decoder
  communicates across devices, and XLA copies every operand such a kernel
  updates in place (a full copy of the pools per step), so the only in-place
  pool update is the separate, non-communicating commit.
* :func:`commit_latent_rows` -- writes each row's new latent on the rank that
  owns its position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp
from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
    MLA_LAYERS,
    decode_local,
)


def latent_rows_from_handoff(new_keys):
    """``[batch, M, 640]`` latent rows from a DCP decoder's handoff (packed in
    new-key rows 0..2, see ``decode_megakernel.mla``)."""
    return jnp.concatenate((new_keys[:, :, 0], new_keys[:, :, 1], new_keys[:, :, 2, :128]), axis=-1)


def _commit_latent_rows_fused(pools, rows, positions, blocks, owned, pool_map):
    """Alias all unified pools in one non-communicating cache commit.

    Overlap the layers' native-tile reads. Retire all writes before
    advancing to the next token so rows sharing a tile preserve each other.
    """
    batch, layers, _ = rows.shape
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = 8 * packing
    metadata = jnp.stack((positions[:batch], blocks, owned.astype(jnp.int32)))
    count = len(pools)

    def kernel(meta, values, *refs):
        outputs = refs[count:2 * count]
        tiles, semaphores = refs[-2:]

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

    return tuple(pl.pallas_call(
        kernel, out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),)
                     + (pl.BlockSpec(memory_space=tpu.HBM),) * count,
            out_specs=(pl.BlockSpec(memory_space=tpu.HBM),) * count,
            scratch_shapes=(tpu.VMEM((layers, 8, packing, 640), dtype),
                            tpu.SemaphoreType.DMA((layers,)))),
        input_output_aliases={2 + i: i for i in range(count)},
        name='kimi_commit_unified_pools')(metadata, rows.astype(dtype), *pools))


def commit_latent_rows(pools, rows, block_tables, positions, pool_map, *, group=mla_dcp.GROUP,
                       owner_table=None, global_pages=None):
    """Write each row's new latent into vLLM's unified pools on the rank that
    owns its position (XLA level, inside ``shard_map`` over ``tp``): MLA layer
    ``m``'s row at position ``p`` lands in ``pools[pool_map[m]]`` at token
    ``p % page_tokens`` of block ``block_tables[b, e]``, ``e = p // page_tokens``
    (global tables) or ``p // (page_tokens group)`` (local tables).
    ``owner_table``: as the decoder's ``dcp_owner_table`` (default the mesh
    position). ``global_pages``: default ``mla_dcp.GLOBAL_PAGES``."""
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
            and rows.shape[0] in (1, 2, 4, 8, 16, 32, 48, 64)
            and rows.shape[1:] == (len(pool_map), 640)
            and 0 < len(pool_map) <= MLA_LAYERS and len(set(pool_map)) == len(pool_map)
            and all(0 <= index < len(pools) for index in pool_map)
            and all(pool.shape[0] > 0 and pool.shape[1:] == (128, packing, 640)
                    and pool.dtype == pools[0].dtype for pool in pools)):
        return _commit_latent_rows_fused(pools, rows, positions, blocks, owned, pool_map)
    pairs = positions[:rows.shape[0]] % page_tokens // packing
    halves = positions[:rows.shape[0]] % packing
    # One scatter per pool for all rows; rows this rank does not own point
    # past the pool and are dropped.
    for layer, index in enumerate(pool_map):
        pool = pools[index]
        targets = jnp.where(owned, blocks, pool.shape[0])
        pools[index] = pool.at[targets, pairs, halves].set(rows[:, layer].astype(pool.dtype), mode="drop")
    return tuple(pools)


def _commit_kda_states(conv_slots, recurrent_slots, conv, recurrent, slots, source_rows=None,
                       null_period=None):
    """Copy complete KDA rows back without XLA's per-row scatter loop.

    Row writes finish in input order. With ``null_period``, rows whose
    target is a null slot (index % null_period == 0: padding rows) are skipped.
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
        def copy_row(i):
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

        @pl.loop(0, batch)
        def row(i):
            if null_period is None:
                copy_row(i)
            else:
                pl.when(indices[i] % null_period != 0)(lambda: copy_row(i))

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
                 defer_kda_commit=False, **decode_options):
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
    """
    if tokens.shape != (batch_size,) or positions.shape != (batch_size,) or slots.shape != (batch_size,):
        raise ValueError("tokens, positions and slots must have one entry per row")
    dp = decode_options.get("attention_dp", 1)
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
    # can (DP1 above four rows or a selected verify row; DP2 at B2..B128).
    # DP2 indices already include each held head shard's bank offset.
    in_place = ((state_row is not None or batch_size > 4) and dp == 1
                or dp == 2 and batch_size in (2, 4, 8, 16, 32, 64, 96, 128))
    if in_place:
        conv, recurrent = conv_slots, recurrent_slots
    else:
        conv, recurrent = conv_slots[state_index], recurrent_slots[state_index]
    pools = tuple(pool)
    result = decode_local(
        weights, (conv, recurrent, *pools), tokens, positions, batch_size=batch_size,
        block_tables=block_tables, dcp_pool_map=dcp_pool_map, dcp_owner_table=dcp_owner_table,
        state_row=state_row, state_slots=state_index if in_place else None,
        live_rows=slots != 0, **decode_options)
    aux = None
    if decode_options.get("aux_hidden_layers"):
        logits, aux, (conv, recurrent, new_keys, new_values) = result
    else:
        logits, (conv, recurrent, new_keys, new_values) = result
    # DP2 returns shard-major rows; commit each to its shard-bank index,
    # preserving untouched slots and ordered null-slot writes.
    if defer_kda_commit:
        # Keep the input banks alive until acceptance. Return the native
        # post-states without touching persistent slots or gathering rows.
        conv_slots, recurrent_slots = conv, recurrent
    elif batch_size > 16 and dp in (1, 2):
        conv_slots, recurrent_slots = _commit_kda_states(
            conv_slots, recurrent_slots, conv, recurrent, state_index,
            null_period=conv_slots.shape[0] // dp)
    else:
        conv_slots = conv_slots.at[state_index].set(conv)
        recurrent_slots = recurrent_slots.at[state_index].set(recurrent)
    latents = latent_rows_from_handoff(new_keys)
    if group_rows is not None:
        # The handoff holds the group's rows first; only their owners commit.
        latents, block_tables, positions = (
            latents[: group_rows.shape[0]], block_tables[group_rows], positions[group_rows])
    pool = commit_latent_rows(pools, latents, block_tables, positions,
                              dcp_pool_map, owner_table=dcp_owner_table)
    if aux is not None:
        return logits[0], aux, conv_slots, recurrent_slots, pool
    return logits[0], conv_slots, recurrent_slots, pool

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
"""QSA over donated main and BF16 index pools with shared logical block IDs."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    TopKBackend,
)
from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import select_index_blocks
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _indexer_updates,
    _token_metadata,
    produce,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import (
    pack_rows,
    qsa_attention,
    selected_tokens,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
    read_ring_and_tags,
    write_ring_and_tags,
)
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import (
    get_main_kv_dma_batch,
)


def _write_main_cache_universal(pool, rows, page, logical, valid):
    """Write contiguous prefill tiles without copying the full cache pool.

    Complete page-local tiles use one large DMA. Tiles that cross a request or
    page boundary fall back to row DMAs inside the same aliased custom call.
    An outer ``lax.cond`` cannot be used here: XLA then copies the multi-GiB
    cache to reconcile the two aliased branch outputs.
    """
    tile = 128

    if len(rows) % tile or pool.shape[1] % tile:
        safe_page = jnp.where(valid, page, pool.shape[0])
        return pool.at[safe_page, logical % pool.shape[1]].set(rows, mode="drop")

    offsets = logical % pool.shape[1]
    pages_by_tile = page.reshape(-1, tile)
    offsets_by_tile = offsets.reshape(-1, tile)
    valid_by_tile = valid.reshape(-1, tile)
    bulk = (
        jnp.all(valid_by_tile, axis=1)
        & jnp.all(pages_by_tile == pages_by_tile[:, :1], axis=1)
        & jnp.all(
            offsets_by_tile
            == offsets_by_tile[:, :1] + jnp.arange(tile, dtype=offsets.dtype),
            axis=1,
        )
    )

    def kernel(
        pages_ref,
        offsets_ref,
        valid_ref,
        bulk_ref,
        source_ref,
        _old_ref,
        target_ref,
        semaphore,
    ):
        program = pl.program_id(0)
        first = program * tile

        @pl.when(bulk_ref[program] != 0)
        def store_tile():
            pltpu.async_copy(
                source_ref,
                target_ref.at[pages_ref[first], pl.ds(offsets_ref[first], tile)],
                semaphore,
            ).wait()

        @pl.when(bulk_ref[program] == 0)
        def store_edges():
            for lane in range(tile):
                row = first + lane

                @pl.when(valid_ref[row] != 0)
                def store_row(*, lane=lane, row=row):
                    pltpu.async_copy(
                        source_ref.at[pl.ds(lane, 1)],
                        target_ref.at[pages_ref[row], pl.ds(offsets_ref[row], 1)],
                        semaphore,
                    ).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(len(rows) // tile,),
            in_specs=[
                pl.BlockSpec(
                    (tile, *rows.shape[1:]),
                    lambda program, *_: (program,) + (0,) * (rows.ndim - 1),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={5: 0},
        name="qwen4_exp_main_kv_universal_tiled_scatter",
    )(
        page.astype(jnp.int32),
        offsets.astype(jnp.int32),
        valid.astype(jnp.int32),
        bulk.astype(jnp.int32),
        rows,
        pool,
    )


def write_main_cache(
    pool: jax.Array,
    rows: jax.Array,
    page: jax.Array,
    logical: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Batch independent row DMAs while preserving duplicate write order."""
    tile = 128
    batch = get_main_kv_dma_batch(pool, rows)
    if batch == 1:
        return _write_main_cache_universal(pool, rows, page, logical, valid)
    if len(rows) % tile or pool.shape[1] % tile:
        return _write_main_cache_universal(pool, rows, page, logical, valid)
    offsets = logical % pool.shape[1]
    pp, oo, vv = (
        page.reshape(-1, tile),
        offsets.reshape(-1, tile),
        valid.reshape(-1, tile),
    )
    bulk = (
        jnp.all(vv, axis=1)
        & jnp.all(pp == pp[:, :1], axis=1)
        & jnp.all(oo == oo[:, :1] + jnp.arange(tile, dtype=offsets.dtype), axis=1)
    )
    pg, off, val = (
        page.reshape(-1, batch),
        offsets.reshape(-1, batch),
        valid.reshape(-1, batch).astype(jnp.bool_),
    )
    conflict = (
        (pg[:, :, None] == pg[:, None, :])
        & (off[:, :, None] == off[:, None, :])
        & val[:, :, None]
        & val[:, None, :]
        & ~jnp.eye(batch, dtype=jnp.bool_)[None]
    )
    unique = ~jnp.any(conflict, axis=(1, 2))

    def kernel(
        pages_ref,
        offsets_ref,
        valid_ref,
        bulk_ref,
        unique_ref,
        source_ref,
        _old_ref,
        target_ref,
        semaphores,
    ):
        program = pl.program_id(0)
        first = program * tile

        @pl.when(bulk_ref[program] != 0)
        def store_tile():
            pltpu.async_copy(
                source_ref,
                target_ref.at[pages_ref[first], pl.ds(offsets_ref[first], tile)],
                semaphores.at[0],
            ).wait()

        @pl.when(bulk_ref[program] == 0)
        def store_edges():
            for packet in range(tile // batch):
                start = packet * batch
                group = program * (tile // batch) + packet

                @pl.when(unique_ref[group] != 0)
                def store_unique(*, start=start):
                    for lane in range(batch):
                        row = first + start + lane

                        @pl.when(valid_ref[row] != 0)
                        def begin(*, lane=lane, row=row, start=start):
                            pltpu.make_async_copy(
                                source_ref.at[pl.ds(start + lane, 1)],
                                target_ref.at[
                                    pages_ref[row], pl.ds(offsets_ref[row], 1)
                                ],
                                semaphores.at[lane],
                            ).start()

                    for lane in range(batch):
                        row = first + start + lane

                        @pl.when(valid_ref[row] != 0)
                        def finish(*, lane=lane, row=row, start=start):
                            pltpu.make_async_copy(
                                source_ref.at[pl.ds(start + lane, 1)],
                                target_ref.at[
                                    pages_ref[row], pl.ds(offsets_ref[row], 1)
                                ],
                                semaphores.at[lane],
                            ).wait()

                @pl.when(unique_ref[group] == 0)
                def store_ordered(*, start=start):
                    for lane in range(batch):
                        row = first + start + lane

                        @pl.when(valid_ref[row] != 0)
                        def store(*, lane=lane, row=row, start=start):
                            pltpu.async_copy(
                                source_ref.at[pl.ds(start + lane, 1)],
                                target_ref.at[
                                    pages_ref[row], pl.ds(offsets_ref[row], 1)
                                ],
                                semaphores.at[0],
                            ).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(len(rows) // tile,),
            in_specs=[
                pl.BlockSpec(
                    (tile, *rows.shape[1:]),
                    lambda program, *_: (program,) + (0,) * (rows.ndim - 1),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA((batch,))],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={6: 0},
        name="qwen4_exp_main_kv_batched_ordered_scatter",
    )(
        page.astype(jnp.int32),
        offsets.astype(jnp.int32),
        valid.astype(jnp.int32),
        bulk.astype(jnp.int32),
        unique.astype(jnp.int32),
        rows,
        pool,
    )


def _compressed_rows(ratio):
    # Each128-original-token logical block must contain complete BF16 pairs.
    if ratio < 1 or ratio & (ratio - 1) or 128 % ratio or (128 // ratio) % 2:
        raise ValueError("QSA ratio must divide128 into an even number of BF16 groups")
    return 128 // ratio


def write_compressed(
    pool,
    keys,
    complete,
    req,
    logical,
    starts,
    context,
    table,
    ratio,
    *,
    logical_block_size=None,
):
    """One writer per BF16 pair, preserving the other half of partial pairs.

    An FP8 carrier packs two BF16 rows in one 32-bit sublane. Writing a lone
    BF16 row needs a read/modify/write; the last completion in each pair owns
    that write and includes both new rows when both completed this step.
    """
    _compressed_rows(ratio)
    if len(keys) == 0:
        return pool
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    width = pool.shape[-1]
    pairs_per_page = math.prod(pool.shape[1:-1]) // packing
    view = pool.reshape(-1, packing, width)
    group = logical // ratio
    end = context[req] + starts[req + 1] - starts[req]
    write = complete & ((group % 2 == 1) | (logical + ratio >= end))
    block_size = pool.shape[1] if logical_block_size is None else logical_block_size
    pair = table[req, logical // block_size] * pairs_per_page + (
        logical % block_size
    ) // (2 * ratio)
    first_end = (group // 2 * 2 + 1) * ratio - 1
    local = (
        starts[req, None]
        + first_end[:, None]
        + jnp.arange(2)[None] * ratio
        - context[req, None]
    )
    mask = (local >= starts[req, None]) & (local < starts[req + 1, None])
    values = keys[jnp.clip(local, 0, len(keys) - 1)]
    values = jnp.pad(values, ((0, 0), (0, 0), (0, width - keys.shape[-1])))

    tile = 128
    if len(keys) >= tile and len(keys) % tile == 0:
        # A complete ratio-4 prefill page contains sixteen adjacent carrier
        # rows. Pack them in VMEM and issue one 16 KiB DMA instead of sixteen
        # 1 KiB writes. Only compact routing scalars live in SMEM; the full
        # fallback metadata is block-loaded into VMEM so T8192 fits v7 SMEM.
        pair_lanes = tuple(range(2 * ratio - 1, tile, 2 * ratio))
        expected = jnp.zeros((tile,), jnp.bool_).at[jnp.asarray(pair_lanes)].set(True)
        tiled_write = write.reshape(-1, tile)
        tiled_pair = pair.reshape(-1, tile)
        tiled_mask = mask.reshape(-1, tile, 2)
        first_pair = tiled_pair[:, pair_lanes[0]]
        bulk = (
            jnp.all(tiled_write == expected[None], axis=1)
            & jnp.all(tiled_mask[:, pair_lanes], axis=(1, 2))
            & jnp.all(
                tiled_pair[:, pair_lanes]
                == first_pair[:, None]
                + jnp.arange(len(pair_lanes), dtype=tiled_pair.dtype)[None],
                axis=1,
            )
        )

        def tiled_kernel(
            first_ref,
            bulk_ref,
            indices_ref,
            active_ref,
            masks_ref,
            values_ref,
            _old,
            updated,
        ):
            program = pl.program_id(0)

            @pl.when(bulk_ref[program] != 0)
            def store_tile():
                def copy(rows, semaphore):
                    bf = (
                        rows.bitcast(jnp.bfloat16)
                        if pool.dtype != jnp.bfloat16
                        else rows
                    )
                    for output, lane in enumerate(pair_lanes):
                        bf[output] = values_ref[lane]
                    target = updated.at[pl.ds(first_ref[program], len(pair_lanes))]
                    pltpu.async_copy(rows, target, semaphore).wait()

                pl.run_scoped(
                    copy,
                    pltpu.VMEM((len(pair_lanes), packing, width), pool.dtype),
                    pltpu.SemaphoreType.DMA,
                )

            @pl.when(bulk_ref[program] == 0)
            def store_edges():
                for lane in range(tile):

                    @pl.when(active_ref[lane] != 0)
                    def store_row(*, lane=lane):
                        def copy(row, semaphore, *, lane=lane):
                            target = updated.at[indices_ref[lane]]
                            bf = (
                                row.bitcast(jnp.bfloat16)
                                if pool.dtype != jnp.bfloat16
                                else row
                            )
                            pltpu.async_copy(target, row, semaphore).wait()
                            local_row = jnp.broadcast_to(
                                jnp.arange(2)[:, None], (2, width)
                            )
                            row_mask = (
                                jnp.where(
                                    local_row == 0,
                                    masks_ref[lane, 0],
                                    masks_ref[lane, 1],
                                )
                                != 0
                            )
                            bf[...] = jnp.where(row_mask, values_ref[lane], bf[...])
                            pltpu.async_copy(row, target, semaphore).wait()

                        pl.run_scoped(
                            copy,
                            pltpu.VMEM((packing, width), pool.dtype),
                            pltpu.SemaphoreType.DMA,
                        )

        view = pl.pallas_call(
            tiled_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(len(keys) // tile,),
                in_specs=[
                    pl.BlockSpec((tile,), lambda program, *_: (program,)),
                    pl.BlockSpec((tile,), lambda program, *_: (program,)),
                    pl.BlockSpec((tile, 2), lambda program, *_: (program, 0)),
                    pl.BlockSpec((tile, 2, width), lambda program, *_: (program, 0, 0)),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
            input_output_aliases={6: 0},
            name="qwen4_exp_compressed_kv_page_tiled_scatter",
        )(
            first_pair.astype(jnp.int32),
            bulk.astype(jnp.int32),
            pair.astype(jnp.int32),
            write.astype(jnp.int32),
            mask.astype(jnp.int32),
            values,
            view,
        )
        return view.reshape(pool.shape)

    def per_pair_kernel(indices, active, row_mask, values_ref, old, updated):
        t = pl.program_id(0)

        @pl.when(active[t] != 0)
        def store():
            def copy(tile, semaphore):
                target = updated.at[indices[t]]
                bf = tile.bitcast(jnp.bfloat16) if pool.dtype != jnp.bfloat16 else tile
                full_pair = (row_mask[t, 0] != 0) & (row_mask[t, 1] != 0)

                @pl.when(full_pair)
                def initialize_full_pair():
                    # Both BF16 rows cover every byte in the packed carrier;
                    # no old value is semantically observable.
                    bf[...] = values_ref[0]

                @pl.when(~full_pair)
                def merge_partial_pair():
                    pltpu.async_copy(target, tile, semaphore).wait()
                    row = jnp.broadcast_to(jnp.arange(2)[:, None], (2, width))
                    mask = jnp.where(row == 0, row_mask[t, 0], row_mask[t, 1]) != 0
                    bf[...] = jnp.where(mask, values_ref[0], bf[...])

                pltpu.async_copy(tile, target, semaphore).wait()

            pl.run_scoped(
                copy, pltpu.VMEM((packing, width), pool.dtype), pltpu.SemaphoreType.DMA
            )

    def update(view, indices, active, row_mask, rows):
        return pl.pallas_call(
            per_pair_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(len(indices),),
                in_specs=[
                    pl.BlockSpec((1, 2, width), lambda t, *args: (t, 0, 0)),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
            input_output_aliases={4: 0},
        )(indices, active, row_mask, rows, view)

    pair = pair.astype(jnp.int32)
    write = write.astype(jnp.int32)
    mask = mask.astype(jnp.int32)
    # Scalar prefetch contains index, active and two mask words per query;
    # values use the VMEM BlockSpec above. Match the shared metadata reserve,
    # retaining half of SMEM for pipeline/compiler state.512 bounds scan work,
    # not a chip capacity or the size of the BF16 values operand.
    scalar_budget = (pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024) // 2
    max_rows = scalar_budget // (4 * jnp.dtype(jnp.int32).itemsize)
    if max_rows < 1:
        raise ValueError("QSA pair metadata exceeds the scalar-prefetch budget")
    chunk_size = min(len(keys), 512, 1 << (max_rows.bit_length() - 1))
    if len(keys) <= chunk_size:
        view = update(view, pair, write, mask, values)
    else:
        padding = (-len(keys)) % chunk_size

        def split(x):
            x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
            return x.reshape(-1, chunk_size, *x.shape[1:])

        chunks = tuple(map(split, (pair, write, mask, values)))

        def step(owner, inputs):
            return update(owner, *inputs), None

        view, _ = jax.lax.scan(step, view, chunks)
    return view.reshape(pool.shape)


def select_blocks(
    query,
    pool,
    lengths,
    table,
    req,
    starts,
    context,
    *,
    ratio,
    block_topk,
    topk_backend: TopKBackend | None = None,
    logical_block_size=None,
    vmem_limit_bytes=None,
):
    """Read BF16 key tiles directly from the carrier, without a cache conversion."""
    width = pool.shape[-1]
    rows = _compressed_rows(ratio)
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    physical_pages = (
        math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * width * 2)
    )
    logical_pages = (
        pool.shape[1] if logical_block_size is None else logical_block_size
    ) // 128
    cache = pool.reshape(-1, rows // 2, packing, width)
    resource_options = (
        {} if vmem_limit_bytes is None else {"vmem_limit_bytes": vmem_limit_bytes}
    )

    def expand_pages(pages):
        return (pages[..., None] * physical_pages + jnp.arange(logical_pages)).reshape(
            -1
        )

    return select_index_blocks(
        query,
        cache,
        lengths,
        expand_pages(table).reshape(len(table), -1),
        req,
        starts=starts,
        context=context,
        ratio=ratio,
        block_topk=block_topk,
        topk_backend=topk_backend,
        **resource_options,
    )


def compact_pooled_qsa_step(
    qgate: jax.Array,
    key: jax.Array,
    value: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    nik: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    pool: jax.Array,
    index_pool: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    *,
    topk_backend: TopKBackend | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Keep the persisted custom-op parameter name while naming its main owner.
    main_pool = pool
    if len(qgate) == 0:
        return (
            main_pool,
            index_pool,
            jnp.empty((0, qgate.shape[1] // 2), jnp.bfloat16),
            jnp.empty((0, token_topk + compress_ratio - 1), jnp.int32),
        )
    # Keep the Runner's flat group tables at the Torch boundary. Static JAX
    # shapes reconstruct their per-request rows inside the operator.
    main_table = main_table.reshape(len(context), -1)
    compressed_table = compressed_table.reshape(len(context), -1)
    # Native BlockTable pads this one-block ring row to token alignment.
    # Only column zero owns storage; the remaining entries are padding.
    ring_ids = ring_ids.reshape(len(context), -1)[:, 0]
    q, k, v, gate, iq, raw = produce(
        qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim
    )
    # Native allocation supplies an owner for every nonempty query row;
    # padded/idle rows have empty query ranges. Keep lengths independent of
    # layer-specific page IDs so XLA can share RPA schedules across layers.
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(len(ring_ids), dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(len(q), starts, context, slots)
    page = jnp.where(
        valid, main_table[req, logical // main_pool.shape[1]], len(main_pool)
    )
    fp8 = main_pool.dtype == jnp.float8_e4m3fn
    main_pool = write_main_cache(
        main_pool,
        pack_rows(k, v, fp8=fp8, k_scale=k_scale, v_scale=v_scale),
        page,
        logical,
        valid,
    )
    # Retain the write dependency across the two owners.
    main_pool, index_pool = jax.lax.optimization_barrier((main_pool, index_pool))
    # The persistent ring/tag ABI pads request rows to eight logical lanes;
    # VMEM padding of those typed views is independently handled by lowering.
    ring_rows = (compress_ratio + 7) // 8 * 8
    ring_elements = ring_rows * raw.shape[-1]
    ring, tags = read_ring_and_tags(index_pool, ring_ids, ring_elements, ring_rows * 2)
    ring = ring.reshape(len(ring_ids), ring_rows, -1)
    tags = tags.reshape(len(ring_ids), ring_rows, 2)
    completed, complete, req, logical, ring, tags = _indexer_updates(
        raw,
        positions,
        ring,
        tags,
        nik,
        trig,
        starts,
        context,
        slots,
        compress_ratio,
        norm_eps,
        rotary_dim,
    )
    index_pool = write_compressed(
        index_pool,
        completed,
        complete,
        req,
        logical,
        starts,
        context,
        compressed_table,
        compress_ratio,
        logical_block_size=main_pool.shape[1],
    )
    ids = jnp.where(active, ring_ids, 0)
    index_pool = write_ring_and_tags(
        index_pool, ring.reshape(len(ids), -1), tags.reshape(len(ids), -1), ids
    )
    lengths = jnp.where(valid, logical + 1, 0).astype(jnp.int32)
    blocks = select_blocks(
        iq,
        index_pool,
        lengths,
        compressed_table,
        req,
        starts,
        context,
        ratio=compress_ratio,
        block_topk=token_topk // compress_ratio,
        topk_backend=topk_backend,
        logical_block_size=main_pool.shape[1],
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
    )
    output, main_pool = qsa_attention(
        q,
        gate,
        blocks,
        lengths,
        main_table[req],
        main_pool,
        starts,
        kv_heads=k.shape[1],
        ratio=compress_ratio,
        k_scale=k_scale if fp8 else 1.0,
        v_scale=v_scale if fp8 else 1.0,
    )
    result = (
        main_pool,
        output.reshape(len(q), -1),
        selected_tokens(blocks, lengths, token_topk, compress_ratio),
    )

    return main_pool, index_pool, *result[1:]


# Keep the protocol object outside the POD-only Torch custom-op schema.
compact_pooled_qsa_step = functools.partial(compact_pooled_qsa_step, topk_backend=None)

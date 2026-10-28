# SPDX-License-Identifier: Apache-2.0
"""QSA over one donated pool, with independent group-native block tables."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import BITS_NEG_INF
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    cp_local_to_global,
    streamindex_topk,
)
from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import decode_query_chunk_size
from vllm_torchtpu.kernels.qwen4_exp.fused_kv_write import (
    fused_layer_write,
    fused_write_supported,
    read_ring_and_tags_concurrent,
)
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _indexer_updates,
    _token_metadata,
    produce,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import (
    pack_rows,
    qsa_attention,
    selected_tokens,
    virtual_page_metadata,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
    read_ring_and_tags,
    write_ring_and_tags,
)
from vllm_torchtpu.kernels.qwen4_exp.request_types import decode_prefix
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
        return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")

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


def main_slot_rows(rows: jax.Array, slot: int) -> jax.Array:
    """Move ``pack_rows`` output (K, V, 0, 0) to head slot ``slot``'s pair."""
    return jnp.roll(rows, 2 * slot, axis=-2) if slot else rows


def _main_slot_keep_mask(slot: int) -> int:
    # The kernels unpack an FP8 32-bit word low byte first: slot s of a
    # paired page owns bytes 2s (K) and 2s + 1 (V); see
    # ``batched_rpa.utils.convert_to_target_bitwidth``.
    return ~(0xFFFF << (16 * slot)) & 0xFFFFFFFF


def _write_main_cache_paired(pool, rows, page, logical, valid, slot):
    """Write one layer's K/V head pair, preserving the partner layer's pair.

    Paired main KV pages (``QWEN4_EXP_PACK_MAIN_KV``) hold two layers' single
    KV heads: layer slot ``s`` stores K and V at head slots ``2s`` and
    ``2s + 1`` of every token row. FP8 packs a lane's four head slots into
    one 32-bit word, below DMA granularity, so each written row is read,
    merged word-wise and written back. Reads of one 128-row tile complete
    before any of its writes, and tiles run in order, so a token written
    twice in one step keeps its last value while the partner bytes never
    change.
    """
    packing = pool.shape[-2]
    if (
        pool.dtype != jnp.float8_e4m3fn
        or pool.shape[2] != 1
        or rows.shape[1:] != pool.shape[2:]
        or not 0 <= slot < packing // 2
    ):
        raise ValueError("Paired main KV needs one FP8 head-slot row per token")
    rows = main_slot_rows(rows, slot)
    offsets = logical % pool.shape[1]
    tile = 128
    if len(rows) % tile or pool.shape[1] % tile:
        # Small steps: a gather of the touched rows feeds the in-place scatter.
        mine = (jnp.arange(packing) // 2 == slot)[:, None]
        old = pool[jnp.minimum(page, len(pool) - 1), offsets]
        merged = jnp.where(mine, rows, old)
        return pool.at[page, offsets].set(merged, mode="drop")

    keep = _main_slot_keep_mask(slot)
    width = pool.shape[-1]
    pp, oo, vv = (
        page.reshape(-1, tile),
        offsets.reshape(-1, tile),
        valid.reshape(-1, tile).astype(jnp.bool_),
    )
    bulk = (
        jnp.all(vv, axis=1)
        & jnp.all(pp == pp[:, :1], axis=1)
        & jnp.all(oo == oo[:, :1] + jnp.arange(tile, dtype=offsets.dtype), axis=1)
    )
    conflict = (
        (pp[:, :, None] == pp[:, None, :])
        & (oo[:, :, None] == oo[:, None, :])
        & vv[:, :, None]
        & vv[:, None, :]
        & ~jnp.eye(tile, dtype=jnp.bool_)[None]
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
        buffer_ref,
        semaphore,
    ):
        program = pl.program_id(0)
        first = program * tile

        def merge():
            old = buffer_ref.bitcast(jnp.uint32).reshape(-1, width)
            new = source_ref.bitcast(jnp.uint32).reshape(-1, width)
            old[...] = (old[...] & jnp.uint32(keep)) | new[...]

        def row_copy(lane, read):
            row = first + lane
            stored = target_ref.at[pages_ref[row], pl.ds(offsets_ref[row], 1)]
            staged = buffer_ref.at[pl.ds(lane, 1)]
            source, target = (stored, staged) if read else (staged, stored)
            return pltpu.make_async_copy(source, target, semaphore)

        def each_valid(action):
            for lane in range(tile):

                @pl.when(valid_ref[first + lane] != 0)
                def run(*, lane=lane):
                    action(lane)

        @pl.when(bulk_ref[program] != 0)
        def store_tile():
            stored = target_ref.at[pages_ref[first], pl.ds(offsets_ref[first], tile)]
            pltpu.async_copy(stored, buffer_ref, semaphore).wait()
            merge()
            pltpu.async_copy(buffer_ref, stored, semaphore).wait()

        @pl.when(bulk_ref[program] == 0)
        def store_edges():
            each_valid(lambda lane: row_copy(lane, True).start())
            each_valid(lambda lane: row_copy(lane, True).wait())
            merge()

            @pl.when(unique_ref[program] != 0)
            def store_unique():
                each_valid(lambda lane: row_copy(lane, False).start())
                each_valid(lambda lane: row_copy(lane, False).wait())

            @pl.when(unique_ref[program] == 0)
            def store_ordered():
                def store(lane):
                    copy = row_copy(lane, False)
                    copy.start()
                    copy.wait()

                each_valid(store)

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
            scratch_shapes=[
                pltpu.VMEM((tile, *rows.shape[1:]), pool.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={6: 0},
        name="qwen4_exp_main_kv_paired_slot_scatter",
    )(
        page.astype(jnp.int32),
        offsets.astype(jnp.int32),
        valid.astype(jnp.int32),
        bulk.astype(jnp.int32),
        unique.astype(jnp.int32),
        rows,
        pool,
    )


def write_main_cache(
    pool: jax.Array,
    rows: jax.Array,
    page: jax.Array,
    logical: jax.Array,
    valid: jax.Array,
    slot: int | None = None,
) -> jax.Array:
    """Batch independent row DMAs while preserving duplicate write order.

    ``slot`` (paired main KV pages only) writes this layer's head pair of
    each row and keeps the other's; ``None`` writes whole rows.
    """
    if slot is not None:
        return _write_main_cache_paired(pool, rows, page, logical, valid, slot)
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


def compressed_pairs(
    pool, keys, complete, req, logical, starts, context, table, ratio, slot=0
):
    """Pair destinations, writers, half masks and values of ``write_compressed``.

    Returns ``(pair, write, mask, values)``: ``pair`` indexes
    ``pool.reshape(-1, packing, width)``, ``write`` marks the one writer of
    each touched pair, ``mask[:, h]`` marks the BF16 halves it fills from
    ``values[:, h]`` (padded to the carrier width).
    """
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    width = pool.shape[-1]
    pairs_per_page = math.prod(pool.shape[1:-1]) // packing
    group = logical // ratio
    end = context[req] + starts[req + 1] - starts[req]
    write = complete & ((group % 2 == 1) | (logical + ratio >= end))
    pair = table[req, logical // pool.shape[1]] * pairs_per_page + (
        logical % pool.shape[1]
    ) // (2 * ratio)
    if slot:
        layer_pairs = pool.shape[1] // (2 * ratio)
        assert (slot + 1) * layer_pairs <= pairs_per_page, (
            "packed compressed-key slot exceeds its pool page"
        )
        pair = pair + slot * layer_pairs
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
    return pair, write, mask, values


def write_compressed(
    pool, keys, complete, req, logical, starts, context, table, ratio, slot=0
):
    """One writer per BF16 pair, preserving the other half of partial pairs.

    An FP8 carrier packs two BF16 rows in one 32-bit sublane. Writing a lone
    BF16 row needs a read/modify/write; the last completion in each pair owns
    that write and includes both new rows when both completed this step.

    ``slot`` selects this layer's region of a page shared by packed layers:
    each layer's keys for one manager page occupy ``pool.shape[1] // (2 *
    ratio)`` consecutive pairs (see ``pool_layout.pooled_cache_specs``).
    """
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    width = pool.shape[-1]
    view = pool.reshape(-1, packing, width)
    pair, write, mask, values = compressed_pairs(
        pool, keys, complete, req, logical, starts, context, table, ratio, slot
    )
    if (
        envs.VLLM_TPU_QSA_COMPRESSED_RUN_WRITE
        and len(keys) >= envs.VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS
    ):
        from vllm_torchtpu.kernels.qwen4_exp.compressed_write import write_runs

        return write_runs(
            view, pair, write, mask, values, len(context), ratio
        ).reshape(pool.shape)
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
    # The compiler prefetches the rows operand into SMEM. A 4K-token step
    # makes it 2 MiB, above v7's 1 MiB limit. Bound each write to 512 KiB.
    bytes_per_token = 2 * width * jnp.dtype(jnp.bfloat16).itemsize
    max_rows = max(1, (512 * 1024) // bytes_per_token)
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


def _request_window(t0, size, starts, context):
    """The requests owning flat tokens ``[t0, t0 + size)``, as a ragged batch.

    Returns ``(ids, seq_lens, cu_q_lens, distribution)`` over ``size`` request
    rows. Nonempty requests overlapping the window come first in token order;
    each owns at least one window token, so ``size`` rows always suffice. A
    request cut by the window keeps only its window tokens, and its sequence
    length drops by the tokens cut from its end, so every kept token keeps
    its absolute position. Filler rows are empty and never scheduled.
    """
    num_requests = len(context)
    begins, ends = starts[:-1], starts[1:]
    overlap = (ends > begins) & (begins < t0 + size) & (ends > t0)
    (ids,) = jnp.nonzero(overlap, size=size, fill_value=num_requests)
    real = ids < num_requests
    ids = jnp.minimum(ids, num_requests - 1)
    window_end = jnp.minimum(ends[ids], t0 + size)
    hi = jnp.where(real, window_end - t0, 0)
    hi = jnp.where(real, hi, jnp.max(hi))
    cu_q_lens = jnp.concatenate([jnp.zeros((1,), jnp.int32), hi.astype(jnp.int32)])
    seq_lens = jnp.where(real, context[ids] + window_end - begins[ids], 0)
    num_decodes = decode_prefix(cu_q_lens)
    num_seqs = jnp.sum(real.astype(jnp.int32))
    distribution = jnp.stack([num_decodes, num_decodes, num_seqs]).astype(jnp.int32)
    return ids, seq_lens.astype(jnp.int32), cu_q_lens, distribution


def _select_shared_stream(query, starts, context, table, *, chunk_size, score):
    """Decode/verify selection with one KV stream per request.

    The per-token decode path scores every query token as its own one-token
    sequence, so the K+1 verify tokens of a request each re-stream its whole
    compressed-key history. Here every ``chunk_size``-token window is scored as
    a ragged batch of its requests: one-token requests keep the batched decode
    pass, longer ones share one stream per query block. ``score`` returns
    ``[window, block_topk]`` selections for one window.
    """
    if len(query) <= chunk_size:
        return score(
            query, *_shared_stream_operands(0, len(query), starts, context, table)
        )
    padding = (-len(query)) % chunk_size
    q = jnp.pad(query, ((0, padding),) + ((0, 0),) * (query.ndim - 1))
    windows = q.reshape(-1, chunk_size, *q.shape[1:])
    t0s = jnp.arange(len(windows), dtype=jnp.int32) * chunk_size

    def window(carry, inputs):
        q, t0 = inputs
        return carry, score(
            q, *_shared_stream_operands(t0, chunk_size, starts, context, table)
        )

    _, selected = jax.lax.scan(window, None, (windows, t0s))
    return selected.reshape(-1, *selected.shape[2:])[: len(query)]


def _shared_stream_operands(t0, size, starts, context, table):
    ids, seq_lens, cu_q_lens, distribution = _request_window(t0, size, starts, context)
    return seq_lens, table[ids], cu_q_lens, distribution


def _shard_pages(pages, shard, align):
    """This rank's kernel pages of each row: columns ``rank, rank + size, ...``.

    ``pages`` is ``[rows, pages_per_row]`` of 128-token kernel pages. Rank
    ``r`` of ``size`` owns kernel page ``j`` of every request when
    ``j % size == r``, which is StreamIndex's context-parallel layout with a
    one-page interleave: local compressed index ``l`` is global
    ``cp_local_to_global(l, r, size, page_rows)``. The local width is padded
    to ``align`` pages (the widest KV block) with page 0; lengths mask it.
    """
    rank, size = shard
    width = pages.shape[1]
    local = -(-width // size)
    local += (-local) % align
    pages = jnp.pad(pages, ((0, 0), (0, local * size - width)))
    return pages.reshape(len(pages), local, size)[:, :, rank]


def _sharded_topk(
    q, weights, cache, seq_lens, pages, cu, distribution, *, shard, rows, **kw
):
    """``streamindex_topk`` over this rank's key slice, as merge candidates.

    Returns int32 ``[tokens, k, 2]``: f32 score bits and the *global*
    compressed position of each local winner; unfilled slots are ``-1`` with
    score ``-inf``. ``pages`` is the full ``[seqs, pages_per_seq]`` table.
    """
    align = max(
        kw["num_kv_pages_per_block"]
        if isinstance(kw["num_kv_pages_per_block"], tuple)
        else (kw["num_kv_pages_per_block"],)
    )
    rank, size = shard
    kw.pop("enable_early_exit", None)
    idx, bits = streamindex_topk(
        q,
        weights,
        cache,
        seq_lens,
        _shard_pages(pages, shard, align).reshape(-1),
        cu,
        distribution,
        enable_early_exit=False,
        cp_size=size,
        cp_rank=rank,
        interleave_size=rows * kw["compression_ratio"],
        return_scores=True,
        **kw,
    )
    filled = idx >= 0
    position = jnp.where(filled, cp_local_to_global(idx, rank, size, rows), -1)
    bits = jnp.where(filled, bits.astype(jnp.int32), BITS_NEG_INF)
    return jnp.stack([bits, position.astype(jnp.int32)], axis=-1)


def merge_candidates(candidates, k, *, topk=None):
    """Exact top-``k`` of all ranks' candidates, lower position first on ties.

    ``candidates`` is int32 ``[tokens, n, 2]`` (f32 score bits, global
    position; position ``-1`` is empty). Every global winner is in its owner
    rank's local top-``k``: that rank ranks it with the same (score, lower
    position) order, so at most ``k - 1`` of its own keys precede it. A
    SparseCore top-``k`` over the gathered scores finds the threshold score;
    only rows with more threshold ties than free slots are re-selected, taking
    the lowest-position ties as the replicated indexer does. Returns int32
    ``[tokens, k]`` positions with a ``-1`` suffix.
    """
    if topk is None:
        from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
            sparsecore_topk,
        )

        def topk(bits, k):
            return sparsecore_topk(bits, k, write_empty_rows=True)

    bits, position = candidates[..., 0], candidates[..., 1]
    filled = position >= 0
    bits = jnp.where(filled, bits, BITS_NEG_INF)
    slots = topk(bits, k)
    taken = slots >= 0
    safe = jnp.maximum(slots, 0)
    blocks = jnp.where(taken, jnp.take_along_axis(position, safe, axis=1), -1)

    scores = jax.lax.bitcast_convert_type(bits, jnp.float32)
    threshold = jnp.min(
        jnp.where(taken, jnp.take_along_axis(scores, safe, axis=1), jnp.inf),
        axis=1,
        keepdims=True,
    )
    strict = filled & (scores > threshold)
    tied = filled & (scores == threshold)
    quota = jnp.sum(taken, axis=1) - jnp.sum(strict, axis=1)
    ambiguous = jnp.sum(tied, axis=1) > quota

    def reselect(blocks):
        # The quota-th smallest tied position, one bit at a time.
        def bit(i, cut):
            trial = cut + jnp.left_shift(1, 30 - i)
            below = jnp.sum(tied & (position < trial[:, None]), axis=1)
            return jnp.where(below < quota, trial, cut)

        cut = jax.lax.fori_loop(0, 31, bit, jnp.zeros_like(quota))
        won = strict | (tied & (position <= cut[:, None]))
        slot = jnp.where(won, jnp.cumsum(won, axis=1) - 1, k)
        rows = jnp.broadcast_to(jnp.arange(len(won))[:, None], won.shape)
        exact = (
            jnp.full(blocks.shape, -1, jnp.int32)
            .at[rows, slot]
            .set(position, mode="drop")
        )
        return jnp.where(ambiguous[:, None], exact, blocks)

    return jax.lax.cond(jnp.any(ambiguous), reselect, lambda b: b, blocks)


def token_shard_step(starts, context, shard_index, rows):
    """Request layout of token rows ``[s * rows, (s + 1) * rows)`` of a step.

    ``shard_index``: int32 ``[1]`` (an operand, so every rank traces the same
    program). Returns ``(base, sub_starts, sub_context)``: the shard's first
    row, the request starts re-based to the shard and each request's context
    advanced by its rows before the shard. A request's rows in the shard see
    the same keys as in the whole step, so the indexer selects the same
    blocks for them; rows past ``starts[-1]`` stay outside every request.
    """
    base = shard_index.reshape(()).astype(jnp.int32) * rows
    sub_starts = jnp.clip(starts - base, 0, rows).astype(starts.dtype)
    before = jnp.clip(base - starts[:-1], 0, starts[1:] - starts[:-1])
    return base, sub_starts, (context + before).astype(context.dtype)


def compressed_kernel_pages(pages: jax.Array, pool: jax.Array, slot: int = 0, ratio: int = 4) -> jax.Array:
    """Flat kernel-page ids of compressed block-table columns `pages`.

    Each column (a pool block of `pool.shape[1]` tokens) holds `pool.shape[1] // 128` kernel
    pages of `128 // ratio` compressed keys for this layer, at `slot` times that within the
    block's physical pages: column `c` -> `c * physical_pages + slot * logical_pages + j`. The
    one addressing of compressed keys, shared by the stock selection and the whole-step
    megakernel's page tables.
    """
    rows = 128 // ratio
    physical_pages = math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * pool.shape[-1] * 2)
    logical_pages = pool.shape[1] // 128
    scaled = pages[..., None] * physical_pages
    first = jnp.arange(logical_pages)
    if slot:
        first = first + slot * logical_pages
    return (scaled + first).reshape(-1)


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
    slot=0,
    max_query_len=1,
    shard=None,
):
    """Read BF16 key tiles directly from the carrier, without a cache conversion.

    ``max_query_len`` routes as in ``qsa_attention``: steps with at most that
    many tokens per request row use the per-token decode indexer.

    ``slot`` selects this layer's kernel pages in a packed page, as in
    ``write_compressed``.

    ``shard=(rank, size)`` scores only this rank's interleaved 1/size of every
    request's compressed keys (``_shard_pages``) and returns merge candidates
    ``[tokens, block_topk, 2]`` for ``merge_candidates`` instead of blocks.
    """
    width = pool.shape[-1]
    rows = 128 // ratio
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    physical_pages = (
        math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * width * 2)
    )
    logical_pages = pool.shape[1] // 128
    cache = pool.reshape(-1, rows // 2, packing, width)
    assert not slot or (slot + 1) * logical_pages <= physical_pages, (
        "packed compressed-key slot exceeds its pool page"
    )

    def expand_pages(pages):
        return compressed_kernel_pages(pages, pool, slot, ratio)

    def topk(q, weights, cache, seq_lens, pages, cu, distribution, **kw):
        """``pages``: [seqs, kernel pages per seq]."""
        if shard is None:
            return streamindex_topk(
                q, weights, cache, seq_lens, pages.reshape(-1), cu, distribution, **kw
            )
        return _sharded_topk(
            q,
            weights,
            cache,
            seq_lens,
            pages,
            cu,
            distribution,
            shard=shard,
            rows=rows,
            **kw,
        )

    def select_chunk(q, lens, pages):
        batch = len(q)
        expanded = expand_pages(pages).reshape(batch, -1)
        index_query = (
            jnp.pad(q, ((0, 0), (0, 0), (0, width - q.shape[-1])))
            if pool.dtype == jnp.bfloat16
            else q
        )
        return topk(
            index_query,
            jnp.full(q.shape[:2], q.shape[-1] ** -0.5, jnp.float32),
            cache,
            lens,
            expanded,
            jnp.arange(batch + 1, dtype=jnp.int32),
            jnp.full((3,), batch, jnp.int32),
            k=block_topk,
            compression_ratio=ratio,
            num_kv_pages_per_block=256 // rows,
            num_queries_per_block=1,
            decode_req_batch_size=8 if batch >= 8 else 4,
            enable_early_exit=True,
            components=qsa_indexing(),
        )

    if len(query) > len(context) * max_query_len:
        # Present prefill as its native ragged request batch. The previous path
        # treated every query as a one-token decode sequence, so every row
        # reloaded the same KV history and ran an M=1 score matmul. BQ128 reuses
        # each KV tile across neighboring queries; chunking overlaps chunk g's
        # SparseCore top-k with chunk g+1's TensorCore scoring.
        query_lens = starts[1:] - starts[:-1]
        num_seqs = jnp.max(
            jnp.where(query_lens > 0, jnp.arange(len(context), dtype=jnp.int32) + 1, 0)
        )
        num_decodes = decode_prefix(starts)
        seq_lens = context + query_lens
        chunk_tokens = 512 if len(query) % 512 == 0 else None
        return topk(
            jnp.pad(query, ((0, 0), (0, 0), (0, width - query.shape[-1]))),
            jnp.full(query.shape[:2], query.shape[-1] ** -0.5, jnp.float32),
            cache,
            seq_lens,
            expand_pages(table).reshape(len(table), -1),
            starts,
            jnp.asarray([num_decodes, num_decodes, num_seqs], dtype=jnp.int32),
            k=block_topk,
            compression_ratio=ratio,
            num_kv_pages_per_block=(256 // rows,) * 3,
            num_queries_per_block=(1, 128, 128),
            decode_req_batch_size=8,
            enable_early_exit=True,
            chunk_tokens=chunk_tokens,
            components=qsa_indexing(ratio - 1),
        )

    pages_per_query = table.shape[1] * logical_pages
    if shard is not None:
        pages_per_query = -(-pages_per_query // shard[1])
    chunk_size = min(len(query), decode_query_chunk_size(pages_per_query))
    query_block = envs.VLLM_TPU_QSA_DECODE_QUERY_BLOCK
    if query_block:
        kv_block = envs.VLLM_TPU_QSA_DECODE_KV_BLOCK
        # The larger single-token block only pays on steps whose requests
        # carry one token (MTP draft rows, plain decode). On verify steps the
        # one-token decode passes are empty and a larger block makes them
        # slower, so they keep the 256-key block.
        decode_kv_block = (
            envs.VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK
            if len(query) <= len(context)
            else 256
        )
        for name, block in (
            ("VLLM_TPU_QSA_DECODE_KV_BLOCK", kv_block),
            ("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", decode_kv_block),
        ):
            if block < rows or block % 128 or block % rows:
                raise ValueError(
                    f"{name} must be a positive multiple of 128 "
                    f"and of {rows} compressed keys per page, got {block}"
                )
        mixed_only = envs.VLLM_TPU_QSA_INDEXER_MIXED_ONLY
        if shard is not None:
            # Room for the local table padded to the widest KV block.
            pages_per_query += max(256, kv_block, decode_kv_block) // rows
            chunk_size = min(len(query), decode_query_chunk_size(pages_per_query))
        # VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: hand StreamIndex the pool-page
        # table and let its KV reader expand each entry into the page's
        # kernel pages, so the scalar-prefetched table (and with it the query
        # window) is not bounded by logical_pages x the table width.
        pool_table = envs.VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE and shard is None
        expansion = (1, 1, 0)
        if pool_table:
            # Same headroom as the sharded table, in pool pages.
            pool_pages = table.shape[1] + -(
                -max(256, kv_block, decode_kv_block) // (rows * logical_pages)
            )
            chunk_size = min(len(query), decode_query_chunk_size(pool_pages))
            expansion = (logical_pages, physical_pages, slot * logical_pages)
        return _select_shared_stream(
            query,
            starts,
            context,
            table,
            chunk_size=chunk_size,
            score=lambda q, seq_lens, pages, cu, distribution: topk(
                jnp.pad(q, ((0, 0), (0, 0), (0, width - q.shape[-1])))
                if pool.dtype == jnp.bfloat16
                else q,
                jnp.full(q.shape[:2], q.shape[-1] ** -0.5, jnp.float32),
                cache,
                seq_lens,
                pages if pool_table else expand_pages(pages).reshape(len(pages), -1),
                cu,
                # VLLM_TPU_QSA_INDEXER_MIXED_ONLY: every sequence through the
                # mixed pass. With per-token scores it scores a 1-token request
                # exactly as the decode passes do, and the two decode launches
                # are empty on verify steps.
                distribution.at[:2].set(0) if mixed_only else distribution,
                k=block_topk,
                compression_ratio=ratio,
                num_kv_pages_per_block=(decode_kv_block // rows,)
                + (kv_block // rows,) * 2,
                num_queries_per_block=(1, query_block, query_block),
                decode_req_batch_size=8 if len(q) >= 8 else 4,
                enable_early_exit=True,
                components=qsa_indexing(
                    ratio - 1, per_token_scores=True, table_expansion=expansion
                ),
                decode_passes=not mixed_only,
                hoist_step_metadata=envs.VLLM_TPU_QSA_INDEXER_HOIST_META,
                # VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY: the score rows are
                # sized for max_model_len; select from each row's live prefix
                # only instead of copying the whole row to the SparseCore.
                **(
                    {"topk_backend": SparseCoreTopK(bounded_copy=True)}
                    if envs.VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY and shard is None
                    else {}
                ),
            ),
        )
    if shard is not None:
        pages_per_query += 256 // rows
        chunk_size = min(len(query), decode_query_chunk_size(pages_per_query))
    if len(query) <= chunk_size:
        return select_chunk(query, lengths, table[req])

    padding = (-len(query)) % chunk_size
    q = jnp.pad(query, ((0, padding),) + ((0, 0),) * (query.ndim - 1))
    lens = jnp.pad(lengths, (0, padding))
    request_ids = jnp.pad(req, (0, padding))
    chunks = (
        q.reshape(-1, chunk_size, *q.shape[1:]),
        lens.reshape(-1, chunk_size),
        request_ids.reshape(-1, chunk_size),
    )
    _, selected = jax.lax.scan(
        lambda carry, inputs: (
            carry,
            select_chunk(inputs[0], inputs[1], table[inputs[2]]),
        ),
        None,
        chunks,
    )
    return selected.reshape(-1, *selected.shape[2:])[: len(query)]


def qsa_cache_writes(
    pool: jax.Array,
    k: jax.Array,
    v: jax.Array,
    raw: jax.Array,
    positions: jax.Array,
    nik: jax.Array,
    trig: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    *,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One QSA layer's cache writes for a step's tokens: main K/V, the completed groups'
    compressed keys, ring rows and tags. Returns ``(pool, req, logical, valid)``.

    ``k`` / ``v`` are ``produce``'s normed+roped keys and values, ``raw`` its raw indexer
    keys; tables are the per-request rows (``ring_ids`` already column 0). Every token
    writes, verify drafts included (later steps overwrite rejected positions). Shared by
    ``_pooled_qsa`` and the Qwen3.8 whole-step megakernel's post-kernel scatter, so both
    leave the same cache bytes for the same inputs.
    """
    # Native allocation supplies an owner for every nonempty query row;
    # padded/idle rows have empty query ranges. Keep lengths independent of
    # layer-specific page IDs so XLA can share RPA schedules across layers.
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(len(ring_ids), dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(len(k), starts, context, slots)
    page = jnp.where(valid, main_table[req, logical // pool.shape[1]], len(pool))
    fp8 = pool.dtype == jnp.float8_e4m3fn
    rows_packed = pack_rows(k, v, fp8=fp8, k_scale=k_scale, v_scale=v_scale)
    slot_kw = {"slot": main_slot} if main_slot >= 0 else {}
    # Decode/verify-sized steps only; prefill keeps the bulk tile writers.
    fused = (
        envs.VLLM_TPU_QSA_FUSED_KV_WRITE
        and len(k) <= len(context) * max_query_len
        and fused_write_supported(
            pool, len(k), len(ring_ids), main_slot if main_slot >= 0 else None
        )
    )
    if not fused:
        pool = write_main_cache(pool, rows_packed, page, logical, valid, **slot_kw)
    ring_rows = math.ceil(compress_ratio / 8) * 8
    ring_elements = ring_rows * raw.shape[-1]
    # A RING_DMA_BATCH override keeps the ring reads on the batched reader.
    read_state = (
        read_ring_and_tags_concurrent
        if fused and not envs.VLLM_TPU_QSA_RING_DMA_BATCH
        else read_ring_and_tags
    )
    ring, tags = read_state(pool, ring_ids, ring_elements, ring_rows * 2)
    ring = ring.reshape(len(ring_ids), ring_rows, -1)
    tags = tags.reshape(len(ring_ids), ring_rows, 2)
    main_logical = logical
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
        # Request row r reads and writes ring row r (slots is r or -1).
        rows_are_slots=fused or envs.VLLM_TPU_QSA_RING_SELECT,
    )
    ids = jnp.where(active, ring_ids, 0)
    if fused:
        pair, write, mask, values = compressed_pairs(
            pool,
            completed,
            complete,
            req,
            logical,
            starts,
            context,
            compressed_table,
            compress_ratio,
            compressed_slot,
        )
        pool = fused_layer_write(
            pool,
            main_rows=main_slot_rows(rows_packed, main_slot)
            if main_slot > 0
            else rows_packed,
            main_page=page,
            main_offset=main_logical,
            main_valid=valid,
            main_slot=main_slot if main_slot >= 0 else None,
            comp_pair=pair,
            comp_write=write,
            comp_mask=mask,
            comp_values=values,
            ring=ring.reshape(len(ids), -1),
            tags=tags.reshape(len(ids), -1),
            # Stopgap: the served model hangs on its first warmup forward
            # when this kernel also writes the rings (it serves with any one
            # region on the separate writers; the kernel alone does not
            # hang). Keep the rings on write_ring_and_tags until the
            # interaction is understood.
            ring_ids=jnp.zeros_like(ids),
            run_starts=starts[:-1],
            run_lengths=starts[1:] - starts[:-1],
            run_length=max_query_len,
        )
        pool = write_ring_and_tags(
            pool, ring.reshape(len(ids), -1), tags.reshape(len(ids), -1), ids
        )
    else:
        pool = write_compressed(
            pool,
            completed,
            complete,
            req,
            logical,
            starts,
            context,
            compressed_table,
            compress_ratio,
            compressed_slot,
        )
        pool = write_ring_and_tags(
            pool, ring.reshape(len(ids), -1), tags.reshape(len(ids), -1), ids
        )
    return pool, req, logical, valid


def _pooled_qsa(
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
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
    rows: jax.Array | None = None,
    select_only: bool = False,
    shard: tuple[int, int] | None = None,
    token_shard: tuple[jax.Array, int] | None = None,
    shared: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """One QSA layer over the unified pool.

    ``select_only``: stop after the indexer and return ``(pool, q, gate, pages,
    kv_lens)`` for an attention computed elsewhere (the Qwen3.8 segment
    megakernel): ``q`` / ``gate`` ``[T, heads, D]`` BF16 and the virtual
    micro-page ids / visible lengths of ``virtual_page_metadata``.

    ``main_slot`` >= 0 selects this layer's KV head pair in main KV pages
    shared by two layers (``pool_layout.pooled_cache_specs``); the default -1
    keeps the layer's whole-row pages.

    ``shard=(rank, size)`` stops after this rank's slice of the indexer and
    returns ``(pool, q, gate, lengths, req, starts, candidates)`` for
    ``pooled_qsa_shard_attend``; ``req`` then indexes the per-request main
    table rows and ``starts`` is the attended token layout.

    ``token_shard=(index, size)`` (``index`` int32 ``[1]``) runs the indexer
    on token rows ``[index * T / size, (index + 1) * T / size)`` only and
    returns the same tuple with that shard's ``[T / size, k]`` blocks in place
    of ``candidates``; all-gather them for ``pooled_qsa_shard_attend`` with
    ``merged``.
    """
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
    pool, req, logical, valid = qsa_cache_writes(
        pool,
        k,
        v,
        raw,
        positions,
        nik,
        trig,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        rotary_dim=rotary_dim,
        k_scale=k_scale,
        v_scale=v_scale,
        compressed_slot=compressed_slot,
        main_slot=main_slot,
        max_query_len=max_query_len,
    )
    fp8 = pool.dtype == jnp.float8_e4m3fn
    lengths = jnp.where(valid, logical + 1, 0).astype(jnp.int32)
    # Attend only the requested rows; every token above still wrote its KV,
    # compressed key and ring state. Up to one row per request slot, the
    # rows take the row-wise decode indexer and RPA.
    keep = None if rows is None else rows >= 0
    picked = None if rows is None else jnp.clip(rows, 0, len(q) - 1)
    if rows is not None and len(rows) <= len(context):
        q, gate, iq, req = q[picked], gate[picked], iq[picked], req[picked]
        lengths = jnp.where(keep, lengths[picked], 0)
        # Present the rows as their own decode batch: row i is a one-token
        # request with its own tables and causal length. The shared-stream
        # indexer (VLLM_TPU_QSA_DECODE_QUERY_BLOCK) maps query rows to
        # requests through starts/context, not req/lengths; the step's token
        # layout would hand row i the i-th token of the step's first requests.
        main_table, compressed_table = main_table[req], compressed_table[req]
        req_rows = req
        req = jnp.arange(len(rows), dtype=jnp.int32)
        starts = jnp.arange(len(rows) + 1, dtype=jnp.int32)
        context = jnp.maximum(lengths - 1, 0)
        keep = picked = None
    if token_shard is not None:
        assert rows is None, "the token shard covers the all-token prefill path"
        index, size = token_shard
        width = len(iq) // size
        base, sub_starts, sub_context = token_shard_step(starts, context, index, width)

        def own(x):
            return jax.lax.dynamic_slice_in_dim(x, base, width)

        blocks = select_blocks(
            own(iq),
            pool,
            own(lengths),
            compressed_table,
            own(req),
            sub_starts,
            sub_context,
            ratio=compress_ratio,
            block_topk=token_topk // compress_ratio,
            slot=compressed_slot,
            # The whole step takes the ragged prefill indexer; keep the
            # shard on it too (it serves verify-sized requests as well).
            max_query_len=1,
        )
        return pool, q, gate, lengths, req, starts, blocks
    if shard is not None:
        block_topk = token_topk // compress_ratio

        def everything_visible(_):
            # Every visible block wins; the merge never runs on these rows.
            return jnp.full((len(iq), block_topk, 2), -1, jnp.int32)

        def score(_):
            return select_blocks(
                iq,
                pool,
                lengths,
                compressed_table,
                req,
                starts,
                context,
                ratio=compress_ratio,
                block_topk=block_topk,
                slot=compressed_slot,
                max_query_len=max_query_len,
                shard=shard,
            )

        candidates = jax.lax.cond(
            jnp.max(lengths) // compress_ratio <= block_topk,
            everything_visible,
            score,
            None,
        )
        if picked is None and rows is not None:
            # The rows' own decode batch: row i owns main table row req[i].
            req = req_rows
        return pool, q, gate, lengths, req, starts, candidates
    if shared is not None:
        blocks = shared_blocks(shared, lengths, compress_ratio, token_topk)
    else:
        blocks = select_blocks(
            iq,
            pool,
            lengths,
            compressed_table,
            req,
            starts,
            context,
            ratio=compress_ratio,
            block_topk=token_topk // compress_ratio,
            slot=compressed_slot,
            max_query_len=max_query_len,
        )
    if envs.VLLM_TPU_QSA_TOPK_STATS:
        from vllm_torchtpu.kernels.qwen4_exp import topk_stats

        topk_stats.record(blocks, lengths, starts, envs.VLLM_TPU_QSA_TOPK_STATS)
    if select_only:
        assert rows is None, "select_only covers the all-token decode/verify path"
        pages, kv_lens, _, _ = virtual_page_metadata(
            blocks, lengths, main_table[req], ratio=compress_ratio, page_size=pool.shape[1]
        )
        heads = q.shape[1]
        return (
            pool,
            q.astype(jnp.bfloat16),
            gate.reshape(len(q), heads, -1).astype(jnp.bfloat16),
            pages.astype(jnp.int32),
            kv_lens.astype(jnp.int32),
        )
    return _attend(
        pool,
        q,
        gate,
        blocks,
        lengths,
        main_table[req],
        starts,
        keep,
        picked,
        kv_heads=k.shape[1],
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        k_scale=k_scale if fp8 else 1.0,
        v_scale=v_scale if fp8 else 1.0,
        main_slot=main_slot,
        max_query_len=max_query_len,
    )


def _attend(
    pool,
    q,
    gate,
    blocks,
    lengths,
    table,
    starts,
    keep,
    picked,
    *,
    kv_heads,
    compress_ratio,
    token_topk,
    k_scale,
    v_scale,
    main_slot,
    max_query_len,
):
    output, pool = qsa_attention(
        q,
        gate,
        blocks,
        lengths,
        table,
        pool,
        starts,
        kv_heads=kv_heads,
        ratio=compress_ratio,
        k_scale=k_scale,
        v_scale=v_scale,
        **({"kv_head_offset": main_slot} if main_slot > 0 else {}),
        max_query_len=max_query_len,
    )
    output = output.reshape(len(q), -1)
    selected = selected_tokens(blocks, lengths, token_topk, compress_ratio)
    if picked is not None:
        # More rows than the decode path holds: attend every token, then pick.
        output = jnp.where(keep[:, None], output[picked], 0)
        selected = jnp.where(keep[:, None], selected[picked], -1)
    return pool, output, selected


def shared_blocks(shared, lengths, compress_ratio, token_topk):
    """Blocks taken from another layer's selection instead of this indexer.

    ``shared`` is that layer's ``selected`` (``selected_tokens`` layout,
    ``[rows, token_topk + ratio - 1]``) for the rows attended here: block j
    sits at columns ``[j * ratio, (j + 1) * ratio)``. Rows whose completed
    groups all fit keep every visible block, as the indexer would; a group
    the other layer did not cover falls back to group 0.
    """
    block_topk = token_topk // compress_ratio
    groups = shared[:, :token_topk:compress_ratio]
    groups = jnp.where(groups >= 0, groups // compress_ratio, 0).astype(jnp.int32)
    visible = visible_blocks(lengths, compress_ratio, block_topk)
    fits = (lengths // compress_ratio <= block_topk)[:, None]
    return jnp.where(fits, visible, groups)


def visible_blocks(lengths, compress_ratio, block_topk):
    """Every completed group of each row: the selection when all fit."""
    count = lengths // compress_ratio
    ids = jnp.arange(block_topk, dtype=jnp.int32)
    return jnp.where(ids < count[:, None], ids, -1)


def pooled_qsa_shard_select_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    shard_rank: int,
    shard_size: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Cache writes and this TP rank's indexer slice of ``pooled_qsa_step``.

    Returns ``(pool, q, gate, lengths, req, starts, candidates)``; gather
    ``candidates`` over the TP group and finish with
    ``pooled_qsa_shard_attend_step``.
    """
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        rows=None,
        shard=(shard_rank, shard_size),
    )


def pooled_qsa_token_select_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    shard_index: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    shard_size: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Cache writes and the indexer on this TP rank's token shard.

    ``shard_index``: int32 ``[1]``, the rank. Returns ``(pool, q, gate,
    lengths, req, starts, blocks)``; all-gather ``blocks`` over the TP group
    and finish with ``pooled_qsa_shard_attend_step(merged=True)``.
    """
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        token_shard=(shard_index, shard_size),
    )


def pooled_qsa_shard_select_rows_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    rows: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    shard_rank: int,
    shard_size: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """``pooled_qsa_shard_select_step`` for ``pooled_qsa_rows_step``'s rows."""
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        rows=rows,
        shard=(shard_rank, shard_size),
    )


def pooled_qsa_shard_merge(
    received: jax.Array, compress_ratio: int, token_topk: int
) -> jax.Array:
    """Merge ``[ranks, tokens, k, 2]`` candidates into ``[tokens, k]`` blocks."""
    ranks, tokens, block_topk, _ = received.shape
    candidates = jnp.moveaxis(received, 0, 1).reshape(tokens, ranks * block_topk, 2)
    return merge_candidates(candidates, token_topk // compress_ratio)


def _shard_attend(
    q,
    gate,
    lengths,
    req,
    starts,
    main_table,
    context,
    pool,
    gathered,
    rows,
    *,
    kv_heads,
    compress_ratio,
    token_topk,
    k_scale,
    v_scale,
    main_slot,
    max_query_len,
    merged,
):
    """Merge the gathered candidates, attend, and pick ``rows`` if given.

    ``gathered`` is ``[ranks, tokens, k, 2]`` candidates, or ``[tokens, k]``
    already merged blocks when ``merged``. ``k_scale``/``v_scale`` are the
    FP8 scales (1.0 for BF16 KV).
    """
    block_topk = token_topk // compress_ratio
    main_table = main_table.reshape(len(context), -1)

    def merge(_):
        if merged:
            return gathered
        return pooled_qsa_shard_merge(gathered, compress_ratio, token_topk)

    blocks = jax.lax.cond(
        jnp.max(lengths) // compress_ratio <= block_topk,
        lambda _: visible_blocks(lengths, compress_ratio, block_topk),
        merge,
        None,
    )
    keep = picked = None
    if rows is not None and len(rows) > len(context):
        keep = rows >= 0
        picked = jnp.clip(rows, 0, len(q) - 1)
    return _attend(
        pool,
        q,
        gate,
        blocks,
        lengths,
        main_table[req],
        starts,
        keep,
        picked,
        kv_heads=kv_heads,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        k_scale=k_scale,
        v_scale=v_scale,
        main_slot=main_slot,
        max_query_len=max_query_len,
    )


def pooled_qsa_shard_attend_step(
    q: jax.Array,
    gate: jax.Array,
    lengths: jax.Array,
    req: jax.Array,
    starts: jax.Array,
    main_table: jax.Array,
    context: jax.Array,
    pool: jax.Array,
    gathered: jax.Array,
    kv_heads: int,
    compress_ratio: int,
    token_topk: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    main_slot: int = -1,
    max_query_len: int = 1,
    merged: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Finish ``pooled_qsa_shard_select_step``: merge, attend, select."""
    return _shard_attend(
        q,
        gate,
        lengths,
        req,
        starts,
        main_table,
        context,
        pool,
        gathered,
        None,
        kv_heads=kv_heads,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        k_scale=k_scale,
        v_scale=v_scale,
        main_slot=main_slot,
        max_query_len=max_query_len,
        merged=merged,
    )


def pooled_qsa_shard_attend_rows_step(
    q: jax.Array,
    gate: jax.Array,
    lengths: jax.Array,
    req: jax.Array,
    starts: jax.Array,
    main_table: jax.Array,
    context: jax.Array,
    pool: jax.Array,
    gathered: jax.Array,
    rows: jax.Array,
    kv_heads: int,
    compress_ratio: int,
    token_topk: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    main_slot: int = -1,
    max_query_len: int = 1,
    merged: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``pooled_qsa_shard_attend_step`` picking ``rows`` as the rows step."""
    return _shard_attend(
        q,
        gate,
        lengths,
        req,
        starts,
        main_table,
        context,
        pool,
        gathered,
        rows,
        kv_heads=kv_heads,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        k_scale=k_scale,
        v_scale=v_scale,
        main_slot=main_slot,
        max_query_len=max_query_len,
        merged=merged,
    )


def pooled_qsa_shard_pages_step(
    q: jax.Array,
    gate: jax.Array,
    lengths: jax.Array,
    req: jax.Array,
    main_table: jax.Array,
    context: jax.Array,
    gathered: jax.Array,
    compress_ratio: int,
    token_topk: int,
    page_size: int,
    merged: bool = False,
) -> tuple[jax.Array, ...]:
    """Finish ``pooled_qsa_shard_select_step`` for the segment megakernel.

    Merges the gathered candidates as ``pooled_qsa_shard_attend_step`` does and
    returns ``pooled_qsa_select_step``'s attention inputs ``(q, gate, pages,
    kv_lens)`` instead of attending.
    """
    block_topk = token_topk // compress_ratio

    def merge(_):
        if merged:
            return gathered
        return pooled_qsa_shard_merge(gathered, compress_ratio, token_topk)

    blocks = jax.lax.cond(
        jnp.max(lengths) // compress_ratio <= block_topk,
        lambda _: visible_blocks(lengths, compress_ratio, block_topk),
        merge,
        None,
    )
    table = main_table.reshape(len(context), -1)[req]
    pages, kv_lens, _, _ = virtual_page_metadata(
        blocks, lengths, table, ratio=compress_ratio, page_size=page_size
    )
    heads = q.shape[1]
    return (
        q.astype(jnp.bfloat16),
        gate.reshape(len(q), heads, -1).astype(jnp.bfloat16),
        pages.astype(jnp.int32),
        kv_lens.astype(jnp.int32),
    )


def pooled_qsa_step(
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
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One QSA layer over the unified pool.

    ``main_slot`` >= 0 selects this layer's KV head pair in main KV pages
    shared by two layers (``pool_layout.pooled_cache_specs``); the default -1
    keeps the layer's whole-row pages.
    """
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
    )


def pooled_qsa_shared_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    shared: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``pooled_qsa_step`` attending ``shared``'s selection (``shared_blocks``)
    instead of scoring its own: caches are written as usual, the indexer's
    scoring and top-k are skipped."""
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        shared=shared,
    )


def pooled_qsa_shared_rows_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    rows: jax.Array,
    shared: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``pooled_qsa_rows_step`` with ``pooled_qsa_shared_step``'s selection;
    ``shared`` holds one row per entry of ``rows``."""
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        rows=rows,
        shared=shared,
    )


def pooled_qsa_select_step(
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
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, ...]:
    """``pooled_qsa_step`` up to the indexer: ``(pool, q, gate, pages, kv_lens)``.

    One QSA layer over the unified pool.

    ``main_slot`` >= 0 selects this layer's KV head pair in main KV pages
    shared by two layers (``pool_layout.pooled_cache_specs``); the default -1
    keeps the layer's whole-row pages.
    """
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        select_only=True,
    )


def pooled_qsa_rows_step(
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
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    rows: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compressed_slot: int = 0,
    main_slot: int = -1,
    max_query_len: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``pooled_qsa_step`` that attends only token ``rows`` (-1: none).

    Every token's KV, compressed key and ring state is written as in
    ``pooled_qsa_step``; the output and selected tokens cover ``rows`` only.
    An MTP draft's first step needs attention only at each request's last
    token, while later draft steps read the whole step's cache entries.
    """
    return _pooled_qsa(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        compressed_slot,
        main_slot,
        max_query_len=max_query_len,
        rows=rows,
    )

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
"""Asynchronous lookup of encoded 160-byte PLE rows in one HOST table."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

ROW_BYTES = 160
# The pinned Pallas HOST lowering requires a 4 KiB DMA start address.
_PAGE_BYTES = 4096
_WORD_BYTES = jnp.dtype(jnp.int32).itemsize
_PAGE_WORDS = _PAGE_BYTES // _WORD_BYTES
_ROW_WORDS = ROW_BYTES // _WORD_BYTES
CHUNK_WORDS = math.gcd(_PAGE_WORDS, _ROW_WORDS)
ROW_CHUNKS = _ROW_WORDS // CHUNK_WORDS
_PAGE_CHUNKS = _PAGE_WORDS // CHUNK_WORDS
_ROW_PAGES = math.ceil((_ROW_WORDS + _PAGE_WORDS - CHUNK_WORDS) / _PAGE_WORDS)
_DMA_AXIS = "ple_host_dma"


def host_table_num_words(num_rows: int) -> int:
    """Physical int32 storage, including the final partial DMA page."""
    words = num_rows * _ROW_WORDS
    return (words + _PAGE_WORDS - 1) // _PAGE_WORDS * _PAGE_WORDS


def _compact_owned_queries(
    ids: jax.Array, num_rows: int
) -> tuple[jax.Array, jax.Array]:
    """Pack (local row, original destination) pairs, padding with invalid IDs."""
    flat = ids.reshape(-1)
    valid = (flat >= 0) & (flat < num_rows)
    original = jnp.arange(flat.size, dtype=jnp.int32)
    destinations = jnp.cumsum(valid, dtype=jnp.int32) - 1
    # Invalid queries use distinct out-of-bounds positions, so all scatter
    # indices are unique and no repeated gather is needed for the padded tail.
    destinations = jnp.where(valid, destinations, flat.size + original)
    metadata = jnp.full((flat.size, 2), -1, dtype=jnp.int32)
    metadata = metadata.at[destinations].set(
        jnp.stack((flat, original), axis=-1),
        unique_indices=True,
        mode="drop",
    )
    return metadata.reshape(-1), valid.sum(dtype=jnp.int32)


def start_host_lookup(
    table: jax.Array,
    ids: jax.Array,
    num_rows: int,
    *,
    num_shards: int = 1,
) -> jax.Array:
    """Return pending raw words without a dependency on decoder hidden states.

    Args:
        table: Flat int32 view of a page-aligned HOST allocation, 160 bytes/row.
        ids: Int32 local row IDs. Negative or out-of-range IDs produce zero rows.
        num_rows: Number of encoded rows in the table.
        num_shards: Table owners sharing the gathered IDs. Used to estimate
            owned query work when choosing between the small and large kernels.
    """
    if table.ndim != 1 or table.dtype != jnp.int32:
        raise ValueError("HOST PLE table must be a flat int32 view")
    if ids.ndim != 2 or ids.dtype != jnp.int32:
        raise ValueError("HOST PLE IDs must be int32 [tokens, heads]")
    if not 0 < num_rows <= jnp.iinfo(jnp.int32).max:
        raise ValueError("HOST PLE row count must fit a positive int32")
    if table.size != host_table_num_words(num_rows):
        raise ValueError("HOST PLE requires page-aligned storage of 160-byte rows")
    if num_shards < 1:
        raise ValueError("HOST PLE requires at least one table shard")
    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise NotImplementedError("HOST PLE lookup requires a TPU with SparseCore")

    # The multidimensional view permits public Ref.reshape inside the kernel.
    # XLA must preserve this HOST reshape as a bitcast, not copy the table.
    host = jax.new_ref(
        pltpu.with_memory_space_constraint(
            table.reshape(-1, CHUNK_WORDS, _PAGE_CHUNKS), pl.HOST
        ),
        memory_space=pl.HOST,
    )
    # Bound SMEM, page staging and outstanding DMA even for an 8192-token
    # prefill. Small queries need neither compaction nor multiple scalar cores.
    tile_rows = math.gcd(ids.size, 256)
    tiled = ids.size > tile_rows
    # Use all scalar subcores once the expected owned queries fill one tile
    # per core. Small decode batches retain the single-core kernel; sharded
    # prefill compacts away other owners before distributing tiles round-robin.
    num_cores = (
        sparse_core.num_cores
        if ids.size >= num_shards * tile_rows * sparse_core.num_cores
        else 1
    )
    compact_queries = num_cores > 1 and num_shards > 1
    metadata_width = 2 if compact_queries else 1
    if compact_queries:
        metadata, count = _compact_owned_queries(ids, num_rows)
        # The SMEM input transfer requires a full 32-byte DMA chunk.
        tile_count = jax.new_ref(
            jnp.full((CHUNK_WORDS,), (count + tile_rows - 1) // tile_rows),
            memory_space=pltpu.SMEM,
        )
    else:
        metadata = ids.reshape(-1)
    indices = jax.new_ref(metadata, memory_space=pltpu.HBM if tiled else pltpu.SMEM)
    pages = pl.empty_ref_like(
        pltpu.HBM(
            (num_cores, tile_rows, _ROW_PAGES, CHUNK_WORDS, _PAGE_CHUNKS), jnp.int32
        )
    )

    # A flat carrier avoids TensorCore padding of the tiny (5, 8) row axes
    # when the PLE consumer masks and reshapes the DMA result.
    @pl.kernel(
        mesh=plsc.ScalarSubcoreMesh(axis_name=_DMA_AXIS, num_cores=num_cores),
        out_type=pltpu.HBM((ids.size * _ROW_WORDS,), jnp.int32),
        scratch_types=(pltpu.SemaphoreType.DMA,)
        + (
            (
                pltpu.SMEM((2, tile_rows * metadata_width), jnp.int32),
                pltpu.SemaphoreType.DMA,
            )
            if tiled
            else ()
        ),
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=False),
        name=(
            f"ple_host_compact_gather_sc{num_cores}_rows{ids.size}"
            if compact_queries
            else f"ple_host_prefix_gather_rows{ids.size}"
        ),
    )
    def lookup(out, sem, *scratch):
        core = jax.lax.axis_index(_DMA_AXIS)

        def gather_tile(row_ids, row_base):
            for wait in (False, True):
                for i in range(tile_rows):
                    row = row_ids[i * metadata_width]

                    @pl.when((row >= 0) & (row < num_rows))
                    def fetch_row(*, i=i, wait=wait, row=row):
                        # Group first so real table addresses never overflow int32.
                        local = (row % _PAGE_CHUNKS) * ROW_CHUNKS
                        page = (row // _PAGE_CHUNKS) * ROW_CHUNKS
                        page += local // _PAGE_CHUNKS
                        end = local % _PAGE_CHUNKS + ROW_CHUNKS
                        for edge in range(_ROW_PAGES):

                            @pl.when((edge == 0) | (end > _PAGE_CHUNKS))
                            def fetch_page(*, edge=edge):
                                size = (
                                    jnp.minimum(end, _PAGE_CHUNKS)
                                    if edge == 0
                                    else end - _PAGE_CHUNKS
                                )
                                src = (
                                    host.at[page + edge]
                                    .reshape(_PAGE_CHUNKS, CHUNK_WORDS)
                                    .at[pl.ds(0, size), :]
                                )
                                dst = (
                                    pages.at[core, i, edge]
                                    .reshape(_PAGE_CHUNKS, CHUNK_WORDS)
                                    .at[pl.ds(0, size), :]
                                )
                                transfer = pltpu.make_async_copy(src, dst, sem)
                                if wait:
                                    transfer.wait()
                                else:
                                    transfer.start()

            for wait in (False, True):
                for i in range(tile_rows):
                    row = row_ids[i * metadata_width]

                    @pl.when((row >= 0) & (row < num_rows))
                    def pack_row(*, i=i, wait=wait, row=row):
                        offset = (row % _PAGE_CHUNKS) * ROW_CHUNKS
                        offset %= _PAGE_CHUNKS
                        src = (
                            pages.at[core, i]
                            .reshape(_ROW_PAGES * _PAGE_CHUNKS, CHUNK_WORDS)
                            .at[pl.ds(offset, ROW_CHUNKS), :]
                        )
                        destination = (
                            row_ids[i * metadata_width + 1]
                            if compact_queries
                            else row_base + i
                        )
                        dst = out.at[pl.ds(destination * _ROW_WORDS, _ROW_WORDS)]
                        transfer = pltpu.make_async_copy(
                            src.reshape(_ROW_WORDS), dst, sem
                        )
                        if wait:
                            transfer.wait()
                        else:
                            transfer.start()

        if tiled:
            row_ids, ids_sem = scratch
            num_tiles = tile_count[0] if compact_queries else ids.size // tile_rows
            num_tiles = (num_tiles + num_cores - 1 - core) // num_cores
            tile_words = tile_rows * metadata_width

            @pl.when(num_tiles > 0)
            def start_ids():
                pltpu.make_async_copy(
                    indices.at[pl.ds(core * tile_words, tile_words)],
                    row_ids.at[0],
                    ids_sem,
                ).start()

            @pl.loop(0, num_tiles)
            def tile_loop(tile):
                global_tile = tile * num_cores + core
                row_base = global_tile * tile_rows
                metadata_base = global_tile * tile_words
                slot = tile % 2
                pltpu.make_async_copy(
                    indices.at[pl.ds(metadata_base, tile_words)],
                    row_ids.at[slot],
                    ids_sem,
                ).wait()

                @pl.when(tile + 1 < num_tiles)
                def prefetch_ids():
                    pltpu.make_async_copy(
                        indices.at[
                            pl.ds(metadata_base + num_cores * tile_words, tile_words)
                        ],
                        row_ids.at[1 - slot],
                        ids_sem,
                    ).start()

                gather_tile(row_ids.at[slot], row_base)
        else:
            gather_tile(indices, 0)

    return jax.device_put(lookup(), jax.memory.Space.Device)


def finish_host_lookup(
    words: jax.Array, ids: jax.Array, hidden: jax.Array, num_rows: int
) -> tuple[jax.Array, jax.Array]:
    """Consume pending words at the PLE layer, preserving the hidden dependency."""
    valid = (ids.reshape(-1) >= 0) & (ids.reshape(-1) < num_rows)
    words = jnp.where(valid[:, None], words.reshape(ids.size, _ROW_WORDS), 0)
    # Keep words packed through owner collectives; unpack only local rows.
    words = words.reshape(*ids.shape, _ROW_WORDS)
    return jax.lax.optimization_barrier((words, hidden))

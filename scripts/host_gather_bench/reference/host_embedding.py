# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split device-initiated gathers from immutable, paged HOST tables."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.extend import core
from jax.interpreters import mlir

# DP8 probes on the pinned runtime validate aligned 4 KiB HOST reads.
HOST_PAGE_BYTES = 4096
HOST_PAGE_WORDS = HOST_PAGE_BYTES // jnp.dtype(jnp.int32).itemsize
_CORE_AXIS = "host_dma_core"

# Torch tensors cannot express Pallas's extended semaphore dtype. These views
# preserve its physical int32 SSA value; they emit no instruction or copy.
# The handle must stay inside one compiled graph, from start through wait.
_semaphore_view = core.Primitive("host_embedding_semaphore_view")
_semaphore_view.def_abstract_eval(lambda aval, *, dtype: aval.update(dtype=dtype))
mlir.register_lowering(_semaphore_view, lambda ctx, value, *, dtype: [value])


def start_host_rows(
    table: jax.Array, rows: jax.Array, *, row_words: int
) -> tuple[jax.Array, jax.Array]:
    """Return an opaque completion handle and unfinished HBM pages.

    Keep ``table`` immutable and pass it to ``finish_host_rows`` in the same
    compiled graph. Do not read, copy, or return the pages before that wait.
    """
    if table.ndim != 1 or table.dtype != jnp.int32:
        raise ValueError("Host embedding storage must be flat int32 pages")
    if table.size % HOST_PAGE_WORDS or not 0 < row_words <= HOST_PAGE_WORDS:
        raise ValueError("Host embedding rows must fit in complete 4096-byte pages")
    if table.size > 2**32:
        raise ValueError("HOST DMA supports at most 16 GiB per owner table")
    if rows.ndim != 1 or rows.dtype != jnp.int32:
        raise ValueError("Host embedding indices must be a flat int32 array")
    if not rows.size:
        return jnp.zeros((1,), jnp.int32), jnp.empty((0,), table.dtype)

    info = pltpu.get_tpu_info().sparse_core
    if info is None:
        raise NotImplementedError("Host embedding DMA requires native SparseCore")
    mesh = plsc.ScalarSubcoreMesh(axis_name=_CORE_AXIS)
    batch = info.num_cores * info.num_subcores
    count = -(-rows.size // batch) * batch
    page_ids = jnp.pad(
        rows // (HOST_PAGE_WORDS // row_words), (0, count - rows.size)
    ).astype(jnp.uint32)
    ids_hbm = jax.new_ref(page_ids)
    source = jax.new_ref(table, memory_space=pl.HOST)

    @pl.kernel(
        out_type=(
            (pltpu.SEMAPHORE @ mesh)((1,), pltpu.SemaphoreType.DMA.dtype),
            pltpu.HBM((count * HOST_PAGE_WORDS,), table.dtype),
        ),
        mesh=mesh,
        scratch_types={
            "ids": pltpu.SMEM((batch,), page_ids.dtype),
            "ids_sem": pltpu.SemaphoreType.DMA(()),
        },
        name="host_embedding_dma_start",
    )
    def start(sem, target, ids, ids_sem):
        core_id = lax.axis_index(_CORE_AXIS)

        def wave(w, _):
            pltpu.async_copy(ids_hbm.at[pl.ds(w * batch, batch)], ids, ids_sem).wait()

            def issue(slot, _):
                local = slot * info.num_cores + core_id
                query = w * batch + local
                pltpu.make_async_copy(
                    source.at[pl.ds(ids[local] * HOST_PAGE_WORDS, HOST_PAGE_WORDS)],
                    target.at[pl.ds(query * HOST_PAGE_WORDS, HOST_PAGE_WORDS)],
                    sem.at[0],
                ).start()

            lax.fori_loop(0, info.num_subcores, issue, None)

        lax.fori_loop(0, count // batch, wave, None)

    semaphore, pages = start()
    return _semaphore_view.bind(semaphore, dtype=jnp.dtype(jnp.int32)), pages


def finish_host_rows(
    table: jax.Array,
    rows: jax.Array,
    semaphore: jax.Array,
    pages: jax.Array,
    *,
    row_words: int,
) -> jax.Array:
    """Wait for all pages, then extract the requested packed rows on device."""
    if not rows.size:
        return jnp.empty((0, row_words), table.dtype)
    mesh = plsc.ScalarSubcoreMesh(axis_name=_CORE_AXIS)
    source = jax.new_ref(table, memory_space=pl.HOST)
    target = jax.new_ref(pages)
    semaphore = _semaphore_view.bind(semaphore, dtype=pltpu.SemaphoreType.DMA.dtype)
    sem = jax.new_ref(semaphore, memory_space=pltpu.SEMAPHORE @ mesh)
    count = pages.size // HOST_PAGE_WORDS

    @pl.kernel(mesh=mesh, name="host_embedding_dma_wait")
    def wait():
        core_id = lax.axis_index(_CORE_AXIS)

        def finish(slot, _):
            query = slot * mesh.num_cores + core_id
            # The receive wait consumes one page's byte count; its source
            # address is not read again. Keep the original HOST allocation live.
            pltpu.make_async_copy(
                source.at[pl.ds(0, HOST_PAGE_WORDS)],
                target.at[pl.ds(query * HOST_PAGE_WORDS, HOST_PAGE_WORDS)],
                sem.at[0],
            ).wait()

        lax.fori_loop(0, count // mesh.num_cores, finish, None)

    wait()
    ready = jax.freeze(target).reshape(count, HOST_PAGE_WORDS)[: rows.size]
    rows_per_page = HOST_PAGE_WORDS // row_words
    columns = (rows % rows_per_page)[:, None] * row_words + jnp.arange(row_words)
    return jnp.take_along_axis(ready, columns, axis=1)


def gather_host_rows(table: jax.Array, rows: jax.Array, *, row_words: int) -> jax.Array:
    """Complete a gather within one call when no compute window is available."""
    semaphore, pages = start_host_rows(table, rows, row_words=row_words)
    return finish_host_rows(table, rows, semaphore, pages, row_words=row_words)

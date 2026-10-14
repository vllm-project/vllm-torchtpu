# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct writes to records in a shared, TensorCore-tiled word pool."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _scatter_word_rows_after_read(cache, records, words, after):
    # Keep read-before-write ring updates on TensorCore. Offloading both the
    # old-pool gather and its writer to SparseCore makes XLA copy the owner
    # to overlap them, even with an optimization barrier on the cache.
    cache, words, _ = jax.lax.optimization_barrier((cache, words, after))
    row_words = cache.shape[1]
    chunks = words.shape[1] // row_words
    if words.shape[1] % row_words or not chunks:
        raise ValueError("Read-before-write updates require whole word rows")
    # Reuse the batched TensorCore DMA from the native cache writer: keep
    # 64 records in flight instead of waiting after each 512-byte row.
    block = 64
    records = jnp.pad(records, (0, -records.shape[0] % block), constant_values=-1)

    def kernel(record_ref, values_ref, _input_ref, output_ref, sem):
        def copies(index):
            token = pl.program_id(0) * block + index
            record = record_ref[token]
            valid = (record >= 0) & (record < cache.shape[0] // chunks)
            return valid, [
                pltpu.make_async_copy(
                    values_ref.at[
                        pl.ds((token * chunks + chunk) * row_words, row_words)
                    ],
                    output_ref.at[record * chunks + chunk, :],
                    sem,
                )
                for chunk in range(chunks)
            ]

        def run(method):
            def body(index, carry):
                valid, row_copies = copies(index)

                @pl.when(valid)
                def _():
                    for copy in row_copies:
                        getattr(copy, method)()

                return carry

            jax.lax.fori_loop(0, block, body, 0)

        run("start")
        run("wait")

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.shape[0] // block,),
            in_specs=(hbm, hbm),
            out_specs=hbm,
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        name="scatter_word_rows_after_read",
    )(records, words.reshape(-1), cache)


@jax.jit
def scatter_word_records(
    cache: jax.Array,
    records: jax.Array,
    words: jax.Array,
    after: jax.Array | None = None,
) -> jax.Array:
    """Write unique records without reading their old contents or padding.

    Negative and out-of-range records are ignored. Records can occupy a
    contiguous part of a word row or several whole rows. The owner retains
    its layout; only the small update batch is flattened for DMA addressing.
    With ``after``, whole-row writes wait for that read result on TensorCore.
    """
    if cache.ndim != 2 or cache.shape[1] != 128:
        raise ValueError("Expected a word cache with shape [rows, 128]")
    row_words = cache.shape[1]
    count, width = words.shape
    if cache.dtype != jnp.int32 or words.dtype != cache.dtype:
        raise ValueError("Cache and updates must contain int32 words")
    if width % row_words and row_words % width:
        raise ValueError("Records must divide a cache row or span whole rows")
    if width * words.dtype.itemsize % 16:
        raise ValueError("Record writes require 16-byte aligned lengths")
    if records.shape != (count,):
        raise ValueError("Supply one destination record per update")
    if not count:
        return cache
    if after is not None:
        return _scatter_word_rows_after_read(cache, records, words, after)
    from vllm_torchtpu.kernels.mla.scatter_utils import scatter_rows

    # Keep the owner's dtype and row boundaries intact so donation can alias
    # it through the SparseCore write without copying the entire shared pool.
    return scatter_rows(
        cache,
        words,
        records,
        name="scatter_word_records",
    )

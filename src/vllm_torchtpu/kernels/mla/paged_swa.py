# SPDX-License-Identifier: Apache-2.0
"""Native BF16 page production for direct-DMA sliding-window attention."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _write_page(ids, values, _owner, output, page_buf, values_buf, sem):
    page_size = output.shape[1]
    i = pl.program_id(0)
    record = ids[i]
    page = record // page_size
    first = (i == 0) | (ids[jnp.maximum(i - 1, 0)] // page_size != page)

    @pl.when(first & (record >= 0) & (page < output.shape[0]))
    def write():
        # At most page_size unique rows target a page. Two aligned input
        # blocks cover its entire sorted run, including an unaligned start.
        start = i // page_size * page_size
        copies = [
            pltpu.make_async_copy(source, dest, sem.at[k])
            for k, (source, dest) in enumerate(
                (
                    (output.at[page], page_buf),
                    (values.at[pl.ds(start, 2 * page_size), :], values_buf),
                )
            )
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()
        rows = jnp.arange(page_size)[:, None]
        result = page_buf[...]
        updates = values_buf[...]
        for source in range(2 * page_size):
            source_record = ids[start + source]
            present = (source_record // page_size == page) & (
                rows == source_record % page_size
            )
            result = jnp.where(present, updates[source : source + 1], result)
        page_buf[...] = result
        copy = pltpu.make_async_copy(page_buf, output.at[page], sem.at[0])
        copy.start()
        copy.wait()


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("interpret",))
def write_swa_pages(
    pages: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Write physical token records, preserving adjacent BF16 rows and null page.

    Records address concatenated banks. Negative and out-of-range records are
    dropped; page zero is reserved. Valid records must be unique within a call.
    Sort the update batch so each physical page has one DMA owner. The aliased
    output preserves untouched BF16 rows without copying or reshaping the pool.
    """
    banks = pages if isinstance(pages, tuple) else (pages,)
    if (isinstance(pages, tuple) and len(banks) != 2) or any(
        bank.ndim != 3
        or bank.shape[0] < 1
        or bank.shape[-1] != 512
        or bank.dtype != jnp.bfloat16
        for bank in banks
    ):
        raise ValueError("SWA owners must be BF16 [pages,page_size,512]")
    page_size = banks[0].shape[1]
    if (
        page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128)
    ) or any(bank.shape[1] != page_size for bank in banks):
        raise ValueError("SWA banks need matching page sizes: 8,16,32,64 or 128*n")
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return pages
    result = []
    base = 0
    for bank in banks:
        capacity = bank.shape[0] * page_size
        valid = (records >= max(base, page_size)) & (records < base + capacity)
        local = jnp.where(valid, records - base, capacity)
        order = jnp.argsort(local, stable=True)
        padding = -records.size % page_size + 2 * page_size
        ids = jnp.pad(local[order], (0, padding), constant_values=capacity)
        updates = jnp.pad(values[order], ((0, padding), (0, 0)))
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        result.append(
            pl.pallas_call(
                _write_page,
                out_shape=jax.ShapeDtypeStruct(bank.shape, bank.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    grid=(records.size,),
                    in_specs=(hbm, hbm),
                    out_specs=hbm,
                    scratch_shapes=(
                        pltpu.VMEM((page_size, 512), jnp.bfloat16),
                        pltpu.VMEM((2 * page_size, 512), jnp.bfloat16),
                        pltpu.SemaphoreType.DMA((2,)),
                    ),
                ),
                input_output_aliases={2: 0},
                interpret=interpret,
                name="write_swa_owned_pages",
            )(ids, updates, bank)
        )
        base += capacity
    return (result[0], result[1]) if isinstance(pages, tuple) else result[0]

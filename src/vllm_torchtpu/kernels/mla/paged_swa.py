# SPDX-License-Identifier: Apache-2.0
"""Native BF16 page production for direct-DMA sliding-window attention."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, donate_argnums=(0,))
def write_swa_pages(
    pages: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Write physical token records, preserving adjacent BF16 rows and null page.

    Records address concatenated banks. Negative and out-of-range records are
    dropped; page zero is reserved. Valid records must be unique within a call.
    Scatter owns the subword update so two tokens sharing a physical BF16 tile
    cannot overwrite one another through independent DMA read/modify/writes.
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
        flat = bank.reshape(-1, 512)

        def scatter(owner, *, local=local):
            return owner.at[local].set(values, mode="drop", unique_indices=True)

        if records.size <= capacity:
            # A contiguous request can publish the whole range in one update.
            # Keep padded tokens unchanged, including their BF16 neighbours.
            def contiguous(owner, *, local=local, valid=valid):
                previous = jax.lax.dynamic_slice(owner, (local[0], 0), values.shape)
                update = jnp.where(valid[:, None], values, previous)
                return jax.lax.dynamic_update_slice(owner, update, (local[0], 0))

            consecutive = (local[0] + records.size <= capacity) & jnp.all(
                (~valid) | (local == local[0] + jnp.arange(records.size))
            )
            updated = jax.lax.cond(consecutive, contiguous, scatter, flat)
        else:
            updated = scatter(flat)
        result.append(updated.reshape(bank.shape))
        base += capacity
    return (result[0], result[1]) if isinstance(pages, tuple) else result[0]

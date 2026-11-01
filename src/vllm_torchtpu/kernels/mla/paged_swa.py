# SPDX-License-Identifier: Apache-2.0
"""Native BF16 page production for direct-DMA sliding-window attention."""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# One bank's plan: sort order [N], word-row destinations [R], row masks [R]
# and the same masks broadcast to [R, 512], R = swa_plan_rows(N).
SwaBankPlan = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _check_page_size(page_size: int) -> None:
    if page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128):
        raise ValueError("SWA banks need matching page sizes: 8,16,32,64 or 128*n")


def _check_owners(
    pages: jax.Array | tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, ...]:
    banks = pages if isinstance(pages, tuple) else (pages,)
    if (isinstance(pages, tuple) and len(banks) != 2) or any(
        bank.ndim != 3
        or bank.shape[0] < 1
        or bank.shape[-1] != 512
        or bank.dtype != jnp.bfloat16
        for bank in banks
    ):
        raise ValueError("SWA owners must be BF16 [pages,page_size,512]")
    _check_page_size(banks[0].shape[1])
    if any(bank.shape[1] != banks[0].shape[1] for bank in banks):
        raise ValueError("SWA banks need matching page sizes: 8,16,32,64 or 128*n")
    return banks


def swa_plan_rows(count: int) -> int:
    """Rows of a publication plan: ``count`` padded to whole SparseCore rounds."""
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "SWA publication requires SparseCore"
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    return count + -count % (lanes * cores)


@functools.partial(jax.jit, static_argnames=("page_size", "bank_pages"))
def plan_swa_pages(
    records: jax.Array, *, page_size: int, bank_pages: tuple[int, ...]
) -> tuple[SwaBankPlan, ...]:
    """Destinations and masks of `write_swa_pages` for owners of this geometry.

    Depends only on the records, so every layer of a stage that publishes the
    same records shares one plan: sorting, pairing and mask construction run
    once, and each layer only permutes and packs its own values.
    """
    _check_page_size(page_size)
    if len(bank_pages) not in (1, 2) or min(bank_pages) < 1:
        raise ValueError("SWA publication requires one or two nonempty banks")
    if records.ndim != 1 or records.dtype != jnp.int32 or not records.size:
        raise ValueError("SWA records must be nonempty INT32 [N]")
    rows = swa_plan_rows(records.size)
    plans, base = [], 0
    for pages in bank_pages:
        capacity = pages * page_size
        valid = (records >= max(base, page_size)) & (records < base + capacity)
        local = jnp.where(valid, records - base, capacity)
        order = jnp.argsort(local, stable=True).astype(jnp.int32)
        ids = local[order]
        next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
        # TC BF16 tiles pack adjacent token rows into one physical word row.
        pairs = ids // 2
        boundary = (ids < capacity) & jnp.concatenate(
            (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
        )
        even = ids % 2 == 0
        has_odd = (~even) | (next_ids // 2 == pairs)
        masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
            has_odd, jnp.int32(-65536), jnp.int32(0)
        )
        padding = rows - records.size
        destinations = jnp.pad(
            jnp.where(boundary, pairs, -1), (0, padding), constant_values=-1
        )
        masks = jnp.pad(masks, (0, padding))
        # SC cannot reshape a lane vector for a column broadcast in the kernel.
        plans.append(
            (order, destinations, masks, jnp.broadcast_to(masks[:, None], (rows, 512)))
        )
        base += capacity
    return tuple(plans)


@functools.partial(jax.jit, donate_argnums=(0,))
def write_planned_swa_pages(
    pages: jax.Array | tuple[jax.Array, jax.Array],
    plan: tuple[SwaBankPlan, ...],
    values: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Write ``values`` at the records ``plan`` was built from, for these owners."""
    banks = _check_owners(pages)
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    count = values.shape[0]
    if not count:
        return pages
    rows = swa_plan_rows(count)
    if len(plan) != len(banks) or any(
        order.shape != (count,)
        or destinations.shape != (rows,)
        or row_masks.shape != (rows,)
        or masks.shape != (rows, 512)
        for order, destinations, row_masks, masks in plan
    ):
        raise ValueError("SWA plan does not match these owners and values")
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "SWA publication requires SparseCore"
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    result = []
    for bank, (order, destinations, row_masks, masks) in zip(banks, plan, strict=True):
        flags = row_masks[:count]
        even = (flags & jnp.int32(65535)) != 0
        has_odd = (flags & jnp.int32(-65536)) != 0
        bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(
            jnp.uint32
        )
        next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
        low = jnp.where(even[:, None], bits, jnp.uint32(0))
        high = jnp.where(even[:, None], next_bits, bits)
        high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
        words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32)
        words = jnp.pad(words, ((0, rows - count), (0, 0)))
        # Keep the page-shaped owner: reshaping the whole pool can break aliasing.
        owner = jax.new_ref(bank)

        def kernel(dst_ref, mask_ref, words_ref, owner_ref, old_ref, sem):
            core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
            cache = owner_ref.bitcast(jnp.int32)
            old_ref[...] = jnp.zeros(old_ref.shape, jnp.int32)

            def body(destinations_ref, masks_ref, updates_ref):
                destinations = destinations_ref[...]
                masks = masks_ref[...]
                reads, writes = [], []
                for i in range(lanes):
                    pair = jnp.maximum(destinations[i], 0)
                    cache_row = cache.at[
                        pair // cache.shape[1], pl.ds(pair % cache.shape[1], 1), :
                    ]
                    partial = (destinations[i] >= 0) & (masks[i, 0] != -1)
                    read = pltpu.make_async_copy(
                        cache_row, old_ref.at[pl.ds(i, 1), :], sem.at[i]
                    )
                    reads.append((partial, read))
                    writes.append(
                        (
                            destinations[i] >= 0,
                            pltpu.make_async_copy(
                                old_ref.at[pl.ds(i, 1), :], cache_row, sem.at[i]
                            ),
                        )
                    )

                    @pl.when(partial)
                    def start_read(read=read):
                        read.start()

                for partial, read in reads:

                    @pl.when(partial)
                    def wait_read(read=read):
                        read.wait()

                old_ref[...] = (old_ref[...] & ~masks) | updates_ref[...]
                for valid, write in writes:

                    @pl.when(valid)
                    def start_write(write=write):
                        write.start()

                for valid, write in writes:

                    @pl.when(valid)
                    def wait_write(write=write):
                        write.wait()

            pltpu.emit_pipeline(
                body,
                grid=(rows // (lanes * cores),),
                tiling=pltpu.Tiling.SPARSE_CORE,
                in_specs=(
                    pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                    pl.BlockSpec((lanes, 512), lambda b: (b * cores + core, 0)),
                    pl.BlockSpec((lanes, 512), lambda b: (b * cores + core, 0)),
                ),
            )(dst_ref, mask_ref, words_ref)

        cast(
            Callable[..., tuple[()]],
            pl.kernel(
                kernel,
                mesh=mesh,
                out_type=(),
                scratch_types=(
                    pltpu.VMEM((lanes, 512), jnp.int32),
                    pltpu.SemaphoreType.DMA((lanes,)),
                ),
                compiler_params=pltpu.CompilerParams(
                    use_tc_tiling_on_sc=True,
                    needs_layout_passes=True,
                    disable_bounds_checks=True,
                ),
                name="write_swa_owned_pages",
            ),
        )(destinations, masks, words, owner)
        result.append(jax.freeze(owner))
    return (result[0], result[1]) if isinstance(pages, tuple) else result[0]


@functools.partial(jax.jit, donate_argnums=(0,))
def write_swa_pages(
    pages: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Write physical token records, preserving adjacent BF16 rows and null page.

    Records address concatenated banks. Negative and out-of-range records are
    dropped; page zero is reserved. Valid records must be unique within a call.
    Each physical word row has one writer, preserving both halves of packed
    words. The aliased output preserves untouched rows without copying the pool.
    Callers that publish the same records repeatedly can share
    `plan_swa_pages` and call `write_planned_swa_pages` instead.
    """
    banks = _check_owners(pages)
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return pages
    plan = plan_swa_pages(
        records,
        page_size=banks[0].shape[1],
        bank_pages=tuple(bank.shape[0] for bank in banks),
    )
    return write_planned_swa_pages(pages, plan, values)

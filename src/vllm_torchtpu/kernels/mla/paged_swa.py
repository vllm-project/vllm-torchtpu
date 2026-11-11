# SPDX-License-Identifier: Apache-2.0
"""Publish sliding-window attention into the shared INT32 word pool."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def swa_word_pages(
    raw: jax.Array | tuple[jax.Array, jax.Array], page_size: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """View shared INT32 [rows,128] owners as 128-word SWA pages.

    The allocator's physical page stride/offset must map to whole 2*S-row
    pages. Banks split on page boundaries. This does not change ownership or
    allocate a second cache. Each token pair occupies four 128-word rows.
    """
    if type(page_size) is not int or (
        page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128)
    ):
        raise ValueError(
            "SWA page size must be 8,16,32,64 or a positive multiple of 128"
        )
    if isinstance(raw, tuple):
        return tuple(swa_word_pages(bank, page_size) for bank in raw)
    if (
        raw.ndim != 2
        or raw.shape[1] != 128
        or raw.dtype != jnp.int32
        or raw.shape[0] % (2 * page_size)
    ):
        raise ValueError("SWA owner must be INT32 [rows,128] with whole physical pages")
    return raw.reshape(-1, 2 * page_size, 128)


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("page_size",))
def write_swa_word_pool(
    raw: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    *,
    page_size: int,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Publish SWA while preserving the shared flat word-row owner interface."""
    swa_word_pages(raw, page_size)
    return _publish_swa_sc(raw, records, values, page_size)


def _publish_swa_sc(raw, records, values, page_size):
    """Publish each token pair once, preserving any unmodified BF16 half."""
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return raw
    banks = raw if isinstance(raw, tuple) else (raw,)
    capacity = sum(bank.shape[0] for bank in banks) // 2
    valid = (records >= page_size) & (records < capacity)
    local = jnp.where(valid, records, capacity)
    order = jnp.argsort(local, stable=True)
    ids = local[order]
    next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
    pairs = ids // 2
    boundary = (ids < capacity) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
    )
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
    low = jnp.where(even[:, None], bits, jnp.uint32(0))
    high = jnp.where(even[:, None], next_bits, bits)
    high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
    words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32).reshape(-1, 128)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        has_odd, jnp.int32(-65536), jnp.int32(0)
    )
    masks = jnp.broadcast_to(masks[:, None, None], (records.size, 4, 128)).reshape(
        -1, 128
    )
    physical = pairs[:, None] * 4 + jnp.arange(4, dtype=jnp.int32)[None, :]
    physical = jnp.where(boundary[:, None], physical, -1).reshape(-1)
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    padding = -physical.size % (lanes * cores)
    ids = jnp.pad(physical, (0, padding), constant_values=-1)
    words = jnp.pad(words, ((0, padding), (0, 0)))
    masks = jnp.pad(masks, ((0, padding), (0, 0)))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    owners = tuple(jax.new_ref(bank) for bank in banks)

    def kernel(ids_ref, values_ref, masks_ref, out_refs, old_ref, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word, mask):
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                read = pltpu.make_async_copy(
                    out_refs[bank].at[indices], old_ref, sem.at[bank]
                )
                read.start()
                read.wait()
                old_ref[...] = (old_ref[...] & ~mask[...]) | word[...]
                write = pltpu.make_async_copy(
                    old_ref, out_refs[bank].at[indices], sem.at[bank]
                )
                write.start()
                write.wait()
                base += banks[bank].shape[0]

        pltpu.emit_pipeline(
            store,
            grid=(ids.size // (lanes * cores),),
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
            ),
        )(ids_ref, values_ref, masks_ref)

    pl.kernel(
        kernel,
        mesh=mesh,
        out_type=(),
        scratch_types=(
            pltpu.VMEM((lanes, 128), jnp.int32),
            pltpu.SemaphoreType.DMA((len(banks),)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="publish_swa_word_pool",
    )(ids, words, masks, owners)
    result = tuple(jax.freeze(owner) for owner in owners)
    return result if isinstance(raw, tuple) else result[0]

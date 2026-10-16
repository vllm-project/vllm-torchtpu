# SPDX-License-Identifier: Apache-2.0
"""BF16 page production for direct-DMA sliding-window attention."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def swa_word_pages(
    raw: jax.Array | tuple[jax.Array, jax.Array], page_size: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """View shared INT32 [rows,128] owners as plane-major SWA pages.

    The allocator's physical page stride/offset must map to whole 2*S-row
    pages. Banks split on page boundaries. This does not change ownership or
    allocate a second cache; verify bitcast/alias lowering on the target TPU.
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
    return raw.reshape(-1, 2, page_size, 128)


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
    """Pack each current token once, then scatter both planes across all banks."""
    from jax.experimental.pallas import tpu_sc as plsc

    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return raw
    banks = raw if isinstance(raw, tuple) else (raw,)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = bits.reshape(-1, 2, 2, 128)
    words = (bits[:, :, 0] | (bits[:, :, 1] << 16)).reshape(-1, 128)
    words = jax.lax.bitcast_convert_type(words, jnp.int32)
    physical = (records // page_size * (2 * page_size) + records % page_size)[
        :, None
    ] + jnp.arange(2, dtype=jnp.int32)[None, :] * page_size
    physical = jnp.where(records[:, None] >= page_size, physical, -1).reshape(-1)
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    padding = -physical.size % (lanes * cores)
    ids = jnp.pad(physical, (0, padding), constant_values=-1)
    words = jnp.pad(words, ((0, padding), (0, 0)))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    owners = tuple(jax.new_ref(bank) for bank in banks)

    def kernel(ids_ref, values_ref, out_refs, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word):
            copies = []
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                copy = pltpu.make_async_copy(
                    word, out_refs[bank].at[indices], sem.at[bank]
                )
                copy.start()
                copies.append(copy)
                base += banks[bank].shape[0]
            for copy in copies:
                copy.wait()

        pltpu.emit_pipeline(
            store,
            grid=(ids.size // (lanes * cores),),
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
            ),
        )(ids_ref, values_ref)

    pl.kernel(
        kernel,
        mesh=mesh,
        scratch_types=(pltpu.SemaphoreType.DMA((len(banks),)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="publish_swa_word_pool",
    )(ids, words, owners)
    result = tuple(owner[...] for owner in owners)
    return result if isinstance(raw, tuple) else result[0]

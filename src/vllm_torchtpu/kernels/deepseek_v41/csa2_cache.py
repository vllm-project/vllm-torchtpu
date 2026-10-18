# SPDX-License-Identifier: Apache-2.0
"""Native word-row producers for CSA2's full-row FP8 representation.

This is a distinct quantization policy from the epic NoPE/FP32-scale-periodic
plus BF16 RoPE cache. Callers must opt into that policy, not reinterpret an
existing native cache as these words.
"""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp


@jax.jit
def encode_csa2_kv(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Post-RoPE BF16 [N,512] -> INT32 [N,128], FP32 [N] scales.

    Byte g of word lane encodes feature g*128+lane. Quantize the unique
    compressed records once, before publishing them, not after query gather.
    """
    if latent.ndim != 2 or latent.shape[1] != 512 or latent.dtype != jnp.bfloat16:
        raise ValueError("CSA2 latents must be BF16 [N,512]")
    values = latent.astype(jnp.float32)
    scales = jnp.maximum(jnp.abs(values).max(axis=1), 1e-4) / 448
    quantized = (values / scales[:, None]).astype(jnp.float8_e4m3fn)
    planes = jax.lax.bitcast_convert_type(quantized, jnp.uint8).reshape(-1, 4, 128)
    planes = planes.astype(jnp.uint32)
    words = (
        planes[:, 0] | (planes[:, 1] << 8) | (planes[:, 2] << 16) | (planes[:, 3] << 24)
    )
    return jax.lax.bitcast_convert_type(words, jnp.int32), scales


@jax.jit
def gather_csa2_kv_sc(
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """SparseCore producer; invalid rows are zeroed once before any reuse."""
    banks = words if isinstance(words, tuple) else (words,)
    scale_banks = scales if isinstance(scales, tuple) else (scales,)
    if len(banks) not in (1, 2) or len(banks) != len(scale_banks):
        raise ValueError("CSA2 requires matching one or two word/scale banks")
    for bank, scale in zip(banks, scale_banks, strict=True):
        if bank.ndim != 2 or bank.shape[1] != 128 or bank.dtype != jnp.int32:
            raise ValueError("CSA2 word banks must be INT32 [records,128]")
        if (
            not bank.shape[0]
            or scale.shape != bank.shape[:1]
            or scale.dtype != jnp.float32
        ):
            raise ValueError("CSA2 scale banks must be nonempty FP32 [records]")
    if records.ndim != 2 or records.shape[1] != 512 or records.dtype != jnp.int32:
        raise ValueError("CSA2 selection IDs must be INT32 [T,512]")
    if not records.size:
        return jnp.zeros((*records.shape, 128), jnp.int32), jnp.zeros(
            records.shape, jnp.float32
        )
    return _gather_sc_banked(banks, scale_banks, records)


def _gather_sc_banked(words, scales, records):
    """Masked DMA into shared VMEM; no concatenated HBM cache.

    Two streams follow the latest epic banked gather's v7x DMA concurrency
    bound. Complementary bank masks fill disjoint rows of each staging slot.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes = info.num_lanes
    cores = info.num_cores * info.num_subcores
    streams, subchunks = (4 if len(words) == 1 else 2), 512 // lanes
    block = cores * lanes * subchunks
    count = records.size
    padding = -count % block
    indices = jnp.pad(records.reshape(-1), (0, padding), constant_values=-1)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    def kernel(
        word_refs,
        scale_refs,
        ids_ref,
        word_out,
        scale_out,
        word_buf,
        scale_buf,
        read_sem,
        write_sem,
    ):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        split = word_refs[0].shape[0]
        steps = subchunks // streams

        def outer(ids):
            base = (pl.program_id(0) * cores + core) * subchunks * lanes

            def gather(step, wait=False):
                slot = step % 2
                for stream in range(streams):
                    selected = ids[pl.ds((step * streams + stream) * lanes, lanes)]
                    if not wait:
                        word_buf[slot, stream] = jnp.zeros((lanes, 128), jnp.int32)
                        scale_buf[slot, stream] = jnp.zeros((lanes,), jnp.float32)
                    for bank in range(len(words)):
                        own = (selected >= 0) & (
                            selected >= split if bank else selected < split
                        )
                        local = selected - split if bank else selected
                        indexed = plsc.Indices(
                            jnp.where(own, local, -1), ignored_value=-1
                        )
                        for kind, (src, dst) in enumerate(
                            ((word_refs[bank], word_buf), (scale_refs[bank], scale_buf))
                        ):
                            copy = pltpu.make_async_copy(
                                src.at[indexed],
                                dst.at[slot, stream],
                                read_sem.at[slot, bank, stream, kind],
                            )
                            if wait:
                                copy.wait()
                            else:
                                copy.start()

            def step(index, _):
                slot = index % 2
                gather(index, wait=True)

                @pl.when(index + 1 < steps)
                def prefetch():
                    gather(index + 1)

                copies = []
                for stream in range(streams):
                    offset = base + (index * streams + stream) * lanes
                    for kind, (src, dst) in enumerate(
                        ((word_buf, word_out), (scale_buf, scale_out))
                    ):
                        copy = pltpu.make_async_copy(
                            src.at[slot, stream],
                            dst.at[pl.ds(offset, lanes)],
                            write_sem.at[stream, kind],
                        )
                        copy.start()
                        copies.append(copy)
                for copy in copies:
                    copy.wait()

            gather(jnp.int32(0))
            jax.lax.fori_loop(0, steps, step, None)

        pltpu.emit_pipeline(
            outer,
            grid=((count + padding) // block,),
            in_specs=(
                pl.BlockSpec((lanes * subchunks,), lambda b: (b * cores + core,)),
            ),
        )(ids_ref)

    out, scale_out = cast(
        Callable[..., tuple[jax.Array, jax.Array]],
        pl.kernel(
            kernel,
            out_type=(
                jax.ShapeDtypeStruct((count + padding, 128), jnp.int32),
                jax.ShapeDtypeStruct((count + padding,), jnp.float32),
            ),
            scratch_types=(
                pltpu.VMEM((2, streams, lanes, 128), jnp.int32),
                pltpu.VMEM((2, streams, lanes), jnp.float32),
                pltpu.SemaphoreType.DMA((2, len(words), streams, 2)),
                pltpu.SemaphoreType.DMA((streams, 2)),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=mesh,
            name="csa2_banked_word_scale_gather",
        ),
    )(words, scales, indices)
    return out[:count].reshape(*records.shape, 128), scale_out[:count].reshape(
        records.shape
    )


@functools.partial(jax.jit, donate_argnums=(0, 1))
def publish_csa2_kv(
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    latent: jax.Array,
) -> tuple[
    jax.Array | tuple[jax.Array, jax.Array], jax.Array | tuple[jax.Array, jax.Array]
]:
    """Quantize unique current records once and publish words plus scalar scales.

    A pair of banks has concatenated record addressing. Negative records skip
    both stores. Valid record IDs must be unique and in range. Arrays retain
    their shared owner's physical row geometry; no full-cache concatenation
    or intermediate expanded KV is produced.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    banked = isinstance(words, tuple)
    banks = words if banked else (words,)
    scale_banks = scales if isinstance(scales, tuple) else (scales,)
    if len(banks) not in (1, 2) or len(banks) != len(scale_banks):
        raise ValueError("Publication requires matching one or two banks")
    for bank, scale_bank in zip(banks, scale_banks, strict=True):
        if bank.ndim != 2 or bank.shape[1] != 128 or bank.dtype != jnp.int32:
            raise ValueError("Publication words must be INT32 [records,128]")
        if scale_bank.shape != bank.shape[:1] or scale_bank.dtype != jnp.float32:
            raise ValueError("Publication scales must be FP32 [records]")
    if records.dtype != jnp.int32 or records.shape != latent.shape[:1]:
        raise ValueError("Publication records must be INT32 [N]")
    encoded, encoded_scales = encode_csa2_kv(latent)
    if not records.size:
        return words, scales
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    padding = -records.size % (lanes * cores)
    ids = jnp.pad(records, (0, padding), constant_values=-1)
    values = jnp.pad(encoded, ((0, padding), (0, 0)))
    value_scales = jnp.pad(encoded_scales, (0, padding))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    word_refs = tuple(jax.new_ref(b) for b in banks)
    scale_refs = tuple(jax.new_ref(b) for b in scale_banks)

    def kernel(ids_ref, values_ref, scales_ref, out_words, out_scales, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word, scale):
            copies = []
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                for kind, (src, dst) in enumerate(
                    ((word, out_words[bank]), (scale, out_scales[bank]))
                ):
                    copy = pltpu.make_async_copy(
                        src, dst.at[indices], sem.at[bank, kind]
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
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
            ),
        )(ids_ref, values_ref, scales_ref)

    cast(
        Callable[..., tuple[()]],
        pl.kernel(
            kernel,
            mesh=mesh,
            scratch_types=(pltpu.SemaphoreType.DMA((len(banks), 2)),),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            name="csa2_publish_words_scales",
        ),
    )(ids, values, value_scales, word_refs, scale_refs)
    out_words, out_scales = (
        tuple(r[...] for r in word_refs),
        tuple(r[...] for r in scale_refs),
    )
    return (out_words, out_scales) if banked else (out_words[0], out_scales[0])

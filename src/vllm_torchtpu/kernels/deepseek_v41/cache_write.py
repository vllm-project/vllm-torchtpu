# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize and publish disjoint 128-key native index-cache slabs."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SLAB_KEYS = 128
INPUT_KEYS = 2 * SLAB_KEYS
HEAD_DIM = 128


def _quantize(values):
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scale = pltpu.bitcast(
        (scale_bytes.astype(jnp.uint32) << 23)[None, :], jnp.float32
    ).reshape(-1)
    quantized = (values / scale[:, None]).astype(jnp.float8_e4m3fn)
    return pltpu.bitcast(quantized, jnp.uint8), scale_bytes.astype(jnp.int32)


def _write_kernel(
    records_smem,
    records_hbm,
    keys_hbm,
    _pool,
    _scales,
    pool_out,
    scales_out,
    keys_buf,
    records_buf,
    words_buf,
    scales_buf,
    sem,
    *,
    page_stride,
    count,
):
    index = pl.program_id(0)
    record = records_smem[index]
    base = record // SLAB_KEYS * SLAB_KEYS
    first = (index == 0) | (
        records_smem[jnp.maximum(index - 1, 0)] // SLAB_KEYS != base // SLAB_KEYS
    )

    @pl.when(first & (record >= 0) & (record < pool_out.shape[0] * 4))
    def publish():
        input_start = index // SLAB_KEYS * SLAB_KEYS
        copies = (
            (
                keys_hbm.at[pl.ds(input_start * HEAD_DIM, INPUT_KEYS * HEAD_DIM)],
                keys_buf,
            ),
            (records_hbm.at[pl.ds(input_start, INPUT_KEYS)], records_buf),
            (pool_out.at[pl.ds(base // 4, SLAB_KEYS // 4), :], words_buf),
            (
                scales_out.at[
                    base // page_stride, (base % page_stride) // SLAB_KEYS, :
                ],
                scales_buf,
            ),
        )
        for source, destination in copies:
            pltpu.make_async_copy(source, destination, sem).start()
            pltpu.make_async_copy(source, destination, sem).wait()
        key_bytes, scale_words = _quantize(keys_buf[...].reshape(INPUT_KEYS, HEAD_DIM))
        mapping = records_buf[...][:, None] == (base + jnp.arange(SLAB_KEYS))[None, :]
        # A unique destination has exactly one source. BF16 represents every
        # byte exactly; the MXU permutation therefore preserves the FP8 bits.
        mapped = jnp.einsum(
            "ik,id->kd",
            mapping.astype(jnp.bfloat16),
            key_bytes.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)
        mapped_scale_matrix = jnp.einsum(
            "ik,id->kd",
            mapping.astype(jnp.bfloat16),
            jnp.broadcast_to(scale_words[:, None], (INPUT_KEYS, HEAD_DIM)).astype(
                jnp.bfloat16
            ),
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)
        mapped_scales = mapped_scale_matrix[:, 0]
        mapped = mapped.reshape(SLAB_KEYS // 4, 4, HEAD_DIM)
        lane_present = mapped_scale_matrix.reshape(SLAB_KEYS // 4, 4, HEAD_DIM) > 0
        words = words_buf[...]
        for lane in range(4):
            mask = jnp.where(lane_present[:, lane], jnp.int32(255) << (8 * lane), 0)
            words = (words & ~mask) | ((mapped[:, lane] << (8 * lane)) & mask)
        words_buf[...] = words
        scales_buf[...] = jnp.where(mapped_scales > 0, mapped_scales, scales_buf[...])
        copies = (
            (words_buf, pool_out.at[pl.ds(base // 4, SLAB_KEYS // 4), :]),
            (
                scales_buf,
                scales_out.at[
                    base // page_stride, (base % page_stride) // SLAB_KEYS, :
                ],
            ),
        )
        for source, destination in copies:
            pltpu.make_async_copy(source, destination, sem).start()
            pltpu.make_async_copy(source, destination, sem).wait()


def write_index_cache(pool, scales, records, keys, page_stride, *, interpret=False):
    """Preserve untouched bytes while writing each affected slab exactly once.

    Valid records must be unique, as required by the serving cache writer.
    Sorting only the update batch gives each physical slab one DMA owner.
    """
    count = keys.shape[0]
    if not count:
        return pool, scales
    if keys.shape != (count, HEAD_DIM) or keys.dtype != jnp.bfloat16:
        raise ValueError("Index writes require BF16 [tokens,128] keys")
    if page_stride % SLAB_KEYS:
        raise ValueError("Index pages must be 128-key aligned")
    sentinel = pool.shape[0] * 4
    order = jnp.argsort(jnp.where(records >= 0, records, sentinel), stable=True)
    sorted_records = jnp.where(records[order] >= 0, records[order], sentinel)
    sorted_records = jnp.pad(sorted_records, (0, INPUT_KEYS), constant_values=sentinel)
    sorted_keys = jnp.pad(keys[order], ((0, INPUT_KEYS), (0, 0))).reshape(-1)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_write_kernel, page_stride=page_stride, count=count),
        out_shape=(
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            jax.ShapeDtypeStruct(scales.shape, scales.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count,),
            in_specs=(hbm,) * 4,
            out_specs=(hbm,) * 2,
            scratch_shapes=(
                pltpu.VMEM((INPUT_KEYS * HEAD_DIM,), jnp.bfloat16),
                pltpu.VMEM((INPUT_KEYS,), jnp.int32),
                pltpu.VMEM((SLAB_KEYS // 4, HEAD_DIM), jnp.int32),
                pltpu.VMEM((SLAB_KEYS,), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0, 4: 1},
        interpret=interpret,
        name="dsv41_quantize_write_index_slabs",
    )(sorted_records, sorted_records, sorted_keys, pool, scales)


def _word_write_kernel(
    records,
    values,
    _cache,
    output,
    values_buf,
    row_buf,
    sem,
    *,
    width,
):
    index = pl.program_id(0)
    per_row = 128 // width
    record = records[index]
    row = record // per_row
    first = (index == 0) | (records[jnp.maximum(index - 1, 0)] // per_row != row)

    @pl.when(first & (record >= 0) & (row < output.shape[0]))
    def publish():
        start = index // per_row * per_row
        for source, destination in (
            (values.at[pl.ds(start, 2 * per_row), :], values_buf),
            (output.at[pl.ds(row, 1), :], row_buf),
        ):
            pltpu.make_async_copy(source, destination, sem).start()
            pltpu.make_async_copy(source, destination, sem).wait()
        columns = jnp.arange(128)[None, :]
        result = row_buf[...]
        values_matrix = values_buf[...]
        for source in range(2 * per_row):
            source_record = records[start + source]
            updates = values_matrix[source : source + 1]
            present = (source_record // per_row == row) & (
                columns // width == source_record % per_row
            )
            result = jnp.where(present, updates, result)
        row_buf[...] = result
        pltpu.make_async_copy(row_buf, output.at[pl.ds(row, 1), :], sem).start()
        pltpu.make_async_copy(row_buf, output.at[pl.ds(row, 1), :], sem).wait()


def write_cache_words(
    cache: jax.Array,
    records: jax.Array,
    words: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Publish packed sub-row records through one DMA owner per word row.

    Valid destinations must be unique. Untouched records keep every bit.
    The writer also serves compressed RoPE records in the companion pool.
    """
    count, width = words.shape
    if not count:
        return cache
    if (
        cache.ndim != 2
        or cache.shape[1] != 128
        or cache.dtype != jnp.int32
        or words.dtype != jnp.int32
        or not 0 < width < 128
        or 128 % width
    ):
        raise ValueError("Sub-row writes require int32 records that divide 128 words")
    per_row = 128 // width
    sentinel = cache.shape[0] * per_row
    order = jnp.argsort(jnp.where(records >= 0, records, sentinel), stable=True)
    padding = -count % per_row + 2 * per_row
    sorted_records = jnp.pad(
        jnp.where(records[order] >= 0, records[order], sentinel),
        (0, padding),
        constant_values=sentinel,
    )
    # Keep sublane permutation outside Mosaic. Each source row is replicated
    # within 128 words, so the kernel needs only aligned loads and masked merges.
    values = jnp.tile(jnp.pad(words[order], ((0, padding), (0, 0))), (1, per_row))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_word_write_kernel, width=width),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count,),
            in_specs=(hbm, hbm),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2 * per_row, 128), jnp.int32),
                pltpu.VMEM((1, 128), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={2: 0},
        interpret=interpret,
        name="dsv41_write_packed_cache_rows",
    )(sorted_records, values, cache)

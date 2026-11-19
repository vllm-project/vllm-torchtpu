# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish exact candidate block IDs from resident FP32 block scores.

Cutoff refinement follows deepseek_v4/sparsecore_topk.py. This specialization
reads flat score rows directly and fuses the last-visible-block publication rule.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    FP32_BITS,
    ORDERED_NEG_INF,
    RADIX_BITS,
    RADIX_BUCKETS,
    RADIX_MASK,
    _ordered_bits,
)


class Radix:
    """Radix selection over ordered FP32 key bits staged in VMEM.

    Byte histograms refine the quota-th largest key, then one pass emits the
    winners in candidate order, so IDs come out ascending without a sort.
    """

    def __init__(self, histogram, lanes):
        self.histogram, self.lanes = histogram, lanes
        self.lane_ids = jnp.arange(lanes, dtype=jnp.int32)

    def positions(self, i):
        return i * self.lanes + self.lane_ids

    def clear(self):
        @plsc.parallel_loop(0, RADIX_BUCKETS, self.lanes, unroll=4)
        def clear(i):
            self.histogram[pl.ds(i, self.lanes)] = jnp.zeros(self.lanes, jnp.int32)

    def add(self, bits):
        """Count the top byte of the keys above -inf."""
        digits = (bits >> (FP32_BITS - RADIX_BITS)).astype(jnp.int32)
        counts, last = plsc.scan_count(digits, mask=bits > ORDERED_NEG_INF)
        plsc.addupdate_scatter(self.histogram, [digits], counts, mask=last)

    def pivot(self, quota):
        """The bucket holding the quota-th largest counted key."""
        histogram, lanes = self.histogram, self.lanes
        zero = jnp.asarray(0, jnp.int32)

        @plsc.parallel_loop(0, RADIX_BUCKETS, lanes, carry=zero, unroll=2)
        def total(i, running):
            cumulative = plsc.cumsum(histogram[pl.ds(i, lanes)]) + running
            histogram[pl.ds(i, lanes)] = cumulative
            return cumulative[-1]

        @plsc.parallel_loop(0, RADIX_BUCKETS, lanes, carry=zero, unroll=2)
        def find(i, bucket):
            below = histogram[pl.ds(i, lanes)] <= total - quota
            return bucket + plsc.all_reduce_population_count(below)[0]

        bucket = jnp.minimum(find, RADIX_MASK)
        index = jnp.maximum(bucket - self.lane_ids, 0)
        gathered = plsc.load_gather(histogram, [index])
        cumulative, previous = gathered[0], gathered[1]
        count = cumulative - jnp.where(bucket > 0, previous, 0)
        return bucket, quota - (total - cumulative), total, count

    def threshold(self, values, chunks, quota):
        """The quota-th largest key; the top byte's histogram is filled."""
        lanes = self.lanes
        prefix, remaining, total, count = self.pivot(jnp.int32(quota))
        keep_all = total <= quota
        done = keep_all | (remaining == count)
        remaining_bits = jnp.int32(FP32_BITS - RADIX_BITS)
        for shift in reversed(range(0, FP32_BITS - RADIX_BITS, RADIX_BITS)):
            self.clear()

            @plsc.parallel_loop(0, jnp.where(done, 0, chunks), unroll=8)
            def refine(i, shift=shift, prefix=prefix):
                bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
                match = (bits >> (shift + RADIX_BITS)) == prefix.astype(jnp.uint32)
                digits = ((bits >> shift) & RADIX_MASK).astype(jnp.int32)
                counts, last = plsc.scan_count(digits, mask=match)
                plsc.addupdate_scatter(self.histogram, [digits], counts, mask=last)

            bucket, next_remaining, _, count = self.pivot(remaining)
            prefix = jnp.where(done, prefix, (prefix << RADIX_BITS) | bucket)
            remaining = jnp.where(done, remaining, next_remaining)
            remaining_bits = jnp.where(done, remaining_bits, shift)
            done |= remaining == count
        lower = prefix.astype(jnp.uint32) << remaining_bits.astype(jnp.uint32)
        floor = jnp.uint32(ORDERED_NEG_INF + 1)
        gate = jnp.where(keep_all, floor, jnp.maximum(lower, floor))
        return total, lower, gate, remaining, done

    def emit(self, values, chunks, quota, selection, destination, ids=None):
        """Write `ids(i)` (default: positions) of the selected keys, -1 after."""
        lanes, ids = self.lanes, ids or self.positions
        total, lower, gate, remaining, done = selection

        @plsc.parallel_loop(
            jnp.minimum(total // lanes * lanes, quota), quota, lanes, unroll=4
        )
        def initialize(i):
            destination[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

        def store(i, count, selected):
            ranks = plsc.cumsum(selected.astype(jnp.int32))
            plsc.store_scatter(destination, [count + ranks - 1], ids(i), mask=selected)
            return count + plsc.all_reduce_population_count(selected)

        @plsc.parallel_loop(
            0, jnp.where(done, chunks, 0), carry=jnp.zeros(lanes, jnp.int32), unroll=8
        )
        def emit_whole(i, count):
            bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
            return store(i, count, bits >= gate)

        @plsc.parallel_loop(
            0,
            jnp.where(done, 0, chunks),
            carry=(jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32)),
            unroll=4,
        )
        def emit_ranked(i, state):
            count, ties = state
            bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
            equal = bits == lower
            ranks = plsc.cumsum(equal.astype(jnp.int32)) + ties
            selected = (bits > lower) | (equal & (ranks <= remaining))
            return store(i, count, selected), ties + plsc.all_reduce_population_count(
                equal
            )


@functools.partial(jax.jit, static_argnames=("block_size",))
def candidate_block_scores(
    scores: jax.Array, lengths: jax.Array, *, block_size: int
) -> jax.Array:
    """Reduce visible token scores by block and force the newest block to win."""
    rows, width = scores.shape
    blocks = pl.cdiv(width, block_size)
    lengths = jnp.clip(lengths, 0, width)
    config = None
    if (
        scores.dtype == jnp.float32
        and lengths.dtype == jnp.int32
        and any(
            key[1:] == (rows, width, block_size)
            for key in tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG
        )
        and jax.default_backend() == "tpu"
    ):
        config = tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG.get(
            (pltpu.get_tpu_info().generation, rows, width, block_size)
        )
    if config is not None:
        return _candidate_block_scores(scores, lengths, block_size, config)
    masked = jnp.where(jnp.arange(width)[None, :] < lengths[:, None], scores, -jnp.inf)
    reduced = (
        jnp.pad(
            masked,
            ((0, 0), (0, blocks * block_size - width)),
            constant_values=-jnp.inf,
        )
        .reshape(rows, blocks, block_size)
        .max(-1)
    )
    last = (lengths - 1) // block_size
    return jnp.where(jnp.arange(blocks)[None, :] == last[:, None], jnp.inf, reduced)


def _candidate_block_scores(scores, lengths, block_size, config):
    rows, width = scores.shape
    row_tile, block_tile = config
    lanes = pltpu.get_tpu_info().num_lanes
    blocks = pl.cdiv(width, block_size)
    padded_blocks = pl.cdiv(blocks, block_tile) * block_tile
    token_tile = block_tile * block_size

    def kernel(length_ref, score_ref, out_ref):
        start = pl.program_id(1) * block_tile
        limits = jnp.stack(
            [length_ref[pl.program_id(0) * row_tile + i] for i in range(row_tile)]
        )
        columns = jnp.arange(token_tile) + start * block_size
        values = jnp.where(columns[None, :] < limits[:, None], score_ref[...], -jnp.inf)
        vectors = values.reshape(row_tile, token_tile // lanes, lanes)
        # Each lane ending a block receives its maximum; gather those lanes
        # without expanding the block axis into padded vectors.
        for bit in range(block_size.bit_length() - 1):
            vectors = jnp.maximum(vectors, pltpu.roll(vectors, 1 << bit, axis=2))
        indices = jnp.arange(lanes // block_size) * block_size + block_size - 1
        indices = jnp.broadcast_to(indices, (*vectors.shape[:-1], indices.size))
        reduced = jnp.take_along_axis(vectors, indices, axis=2).reshape(
            row_tile, block_tile
        )
        last = (limits - 1) // block_size
        out_ref[...] = jnp.where(
            jnp.arange(block_tile)[None, :] + start == last[:, None], jnp.inf, reduced
        )

    # The final input window can be partial. Its invisible columns are masked
    # in the kernel, avoiding a full score-buffer copy to pad the DMA tile.
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, padded_blocks), scores.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(rows // row_tile, padded_blocks // block_tile),
            in_specs=(pl.BlockSpec((row_tile, token_tile), lambda i, j, _: (i, j)),),
            out_specs=pl.BlockSpec((row_tile, block_tile), lambda i, j, _: (i, j)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="indexer_candidate_block_max",
    )(lengths, scores)
    return result[:, :blocks]


def is_tuned_for_publish_candidates(scores: jax.Array, k: int, block_size: int) -> bool:
    """Measured dispatch shape, narrower than the kernel's valid input shapes."""
    return (
        scores.ndim == 2
        and scores.shape[0] > 0
        and scores.dtype == jnp.float32
        and jax.default_backend() == "tpu"
        and (pltpu.get_tpu_info().generation, scores.shape[1], k, block_size)
        in tuned_block_sizes.PUBLISH_CANDIDATE_SHAPES
    )


@functools.partial(jax.jit, static_argnames=("k", "block_size"))
def publish_candidates(
    scores: jax.Array,
    lengths: jax.Array,
    *,
    k: int = 2048,
    block_size: int = 8,
) -> jax.Array:
    """Return ascending block IDs, forcing the last visible block into Top-K.

    Scores are FP32 [request * query, block]. Lengths count visible tokens.
    Exact FP32 radix order prefers +0 over -0 and lower IDs for equal bits.
    Negative infinity is excluded and short selections end with -1 padding.
    NaNs are unsupported. Lengths are clipped to the score capacity.
    """
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    hbm_alignment = pltpu.get_tpu_info().num_lanes
    rows, width = scores.shape
    if (
        scores.dtype != jnp.float32
        or lengths.dtype != jnp.int32
        or lengths.shape != (rows,)
        or block_size < 1
        or not 0 < k <= width
        or width % hbm_alignment
        or k % hbm_alignment
        or rows < 1
    ):
        raise ValueError("Publication requires FP32 rows and HBM-aligned width/Top-K")
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, math.ceil(rows / info.num_subcores)),
    )
    token_lengths = jnp.broadcast_to(lengths[:, None], (rows, lanes))

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((rows, k), jnp.int32),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((2 * width,), jnp.float32),
            pltpu.VMEM((RADIX_BUCKETS,), jnp.int32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((rows, lanes), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="publish_candidate_blocks",
    )
    def kernel(
        score_ref,
        length_ref,
        out_ref,
        key_buffers,
        histogram,
        output,
        limits,
        input_sems,
        output_sem,
        length_sem,
    ):
        radix = Radix(histogram, lanes)
        first = jax.lax.axis_index("core") * mesh.num_subcores
        first += jax.lax.axis_index("subcore")
        stride = mesh.num_cores * mesh.num_subcores

        def copy_scores(row_id, slot):
            return pltpu.make_async_copy(
                score_ref.at[row_id],
                key_buffers.at[pl.ds(slot * width, width)],
                input_sems.at[slot],
            )

        length_copy = pltpu.async_copy(length_ref, limits, length_sem)

        @pl.when(first < rows)
        def prefetch():
            copy_scores(first, 0).start()

        length_copy.wait()

        @pl.loop(first, rows, step=stride)
        def row(row_id):
            slot = ((row_id - first) // stride) % 2
            copy_scores(row_id, slot).wait()
            keys = key_buffers.at[pl.ds(slot * width, width)]

            @pl.when(row_id + stride < rows)
            def next_row():
                copy_scores(row_id + stride, 1 - slot).start()

            tokens = plsc.load_gather(
                limits,
                [
                    jnp.full(lanes, row_id, jnp.int32),
                    jnp.arange(lanes, dtype=jnp.int32),
                ],
            )[0]
            length = jnp.clip(pl.cdiv(tokens, block_size), 0, width)
            last = (tokens - 1) // block_size
            chunks = pl.cdiv(length, lanes)
            radix.clear()

            @plsc.parallel_loop(0, chunks, unroll=8)
            def prepare(i):
                positions = radix.positions(i)
                values = jnp.where(
                    positions == last, jnp.inf, keys[pl.ds(i * lanes, lanes)]
                )
                bits = _ordered_bits(jnp.where(positions < length, values, -jnp.inf))
                keys[pl.ds(i * lanes, lanes)] = bits.view(jnp.float32)
                radix.add(bits)

            selection = radix.threshold(keys, chunks, k)

            @pl.when(row_id > first)
            def wait_previous_output():
                pltpu.make_async_copy(
                    output, out_ref.at[row_id - stride], output_sem
                ).wait()

            radix.emit(keys, chunks, k, selection, output)
            pltpu.async_copy(output, out_ref.at[row_id], output_sem)

        @pl.when(first < rows)
        def drain():
            last_row = first + ((rows - 1 - first) // stride) * stride
            pltpu.make_async_copy(output, out_ref.at[last_row], output_sem).wait()

    return kernel(scores, token_lengths)

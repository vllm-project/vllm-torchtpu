# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024-2025 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore DeepSelect-style Top-K over logical key indices.

Adapted from deepseek-ai/DeepSelect's scan/filter/compact algorithm:
https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.zh.md
Uses shuffled block visits, pipelined DMA and exact score/index radix selection.
"""

from __future__ import annotations

import functools
import math
import random
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# Byte-wise radix passes over IEEE FP32 keys. These describe the representation,
# independently of TensorCore/SparseCore lane counts.
RADIX_BITS = jnp.iinfo(jnp.uint8).bits
RADIX_BUCKETS = 1 << RADIX_BITS
RADIX_MASK = RADIX_BUCKETS - 1
FP32_BITS = jnp.finfo(jnp.float32).bits
FP32_SIGN_BIT = 1 << (FP32_BITS - 1)
UINT32_MAX = jnp.iinfo(jnp.uint32).max
FP32_NEG_INF_BITS = 0xFF800000
ORDERED_NEG_INF = UINT32_MAX ^ FP32_NEG_INF_BITS
BF16_BITS = jnp.finfo(jnp.bfloat16).bits
BF16_HIGH_MASK = (UINT32_MAX << BF16_BITS) & UINT32_MAX


class _Selection(NamedTuple):
    count: jax.Array
    threshold: jax.Array
    threshold_id: jax.Array
    have_threshold: jax.Array
    boundary_ties: jax.Array


def _ordered_bits(keys: jax.Array) -> jax.Array:
    """Preserve FP32 radix ordering, including signed zero."""
    bits = keys.view(jnp.uint32)
    sign = (bits.view(jnp.int32) >> (FP32_BITS - 1)).view(jnp.uint32)
    return bits ^ (sign | jnp.uint32(FP32_SIGN_BIT))


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "block_size",
        "reconstruct_threshold",
        "exclude_neg_inf",
        "scheduling_group_id",
    ),
)
def streaming_topk(
    scores: jax.Array,
    *,
    valid_lengths: jax.Array | None = None,
    k: int = 512,
    block_size: int = 8192,
    reconstruct_threshold: int = 8192,
    exclude_neg_inf: bool = False,
    logical_blocks: jax.Array | None = None,
    start_request: jax.Array | None = None,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Return Top-K IDs for 2-D score rows, ascending then -1 padded.

    Scores follow exact FP32 radix order (+0 above -0); equal bit patterns
    prefer smaller logical indices regardless of block visitation order.
    The reproducible block shuffle is input-independent, not a fresh random
    permutation per invocation. NaNs are unsupported. Negative infinity is
    eligible unless exclude_neg_inf is set. Lengths are clipped to [0, N].
    Scores have shape [rows, N] in request-major order.

    Optional int32 logical_blocks [all_requests, blocks] maps compact slots
    to token IDs. It requires request/query valid_lengths and a block count dividing N.
    Valid blocks must be increasing, padded blocks must have ineligible scores,
    and mapped IDs must fit int32. start_request addresses a valid request
    window in that metadata (default zero). Only final winners are mapped.
    """
    if scores.ndim != 2 or scores.dtype != jnp.float32:
        raise ValueError("scores must be a rank-2 float32 array")
    batch_size, width = scores.shape
    if not 1 <= k <= width:
        raise ValueError("k must be in [1, N]")
    if block_size <= 0 or reconstruct_threshold <= 0:
        raise ValueError("block_size and reconstruct_threshold must be positive")
    if valid_lengths is not None and (
        valid_lengths.size != batch_size or valid_lengths.dtype != jnp.int32
    ):
        raise ValueError("valid_lengths must contain one int32 entry per score row")
    valid_width = width
    # HBM arrays retain TensorCore tiling when SparseCore consumes them.
    hbm_alignment = pltpu.get_tpu_info().num_lanes
    mapped = logical_blocks is not None
    mapping_width = 0
    if logical_blocks is not None:
        if (
            valid_lengths is None
            or valid_lengths.ndim != 2
            or logical_blocks.ndim != 2
            or logical_blocks.dtype != jnp.int32
            or logical_blocks.shape[0] < valid_lengths.shape[0]
            or not logical_blocks.shape[1]
            or valid_width % logical_blocks.shape[1]
        ):
            raise ValueError(
                "logical_blocks must be int32 [requests, blocks] covering "
                "the request/query length window, with blocks dividing N"
            )
        assert valid_lengths is not None
        queries = valid_lengths.shape[1]
        records_per_block = valid_width // logical_blocks.shape[1]
        mapping_width = pl.cdiv(logical_blocks.shape[1], hbm_alignment) * hbm_alignment
        logical_blocks = jnp.pad(
            logical_blocks,
            ((0, 0), (0, mapping_width - logical_blocks.shape[1])),
            constant_values=-1,
        )
    input_alignment = hbm_alignment if block_size % hbm_alignment == 0 else block_size
    padding = -width % input_alignment
    if padding:
        scores = jnp.pad(
            scores,
            ((0, 0), (0, padding)),
            constant_values=-jnp.inf,
        )
        width += padding
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if any(n % lanes for n in (k, block_size, reconstruct_threshold)):
        raise ValueError("k, block_size and reconstruct_threshold must be lane-aligned")
    lengths = (
        jnp.full((batch_size,), valid_width, jnp.int32)
        if valid_lengths is None
        else jnp.clip(valid_lengths.reshape(-1), 0, valid_width)
    )
    lengths = jnp.broadcast_to(lengths[:, None], (batch_size, lanes))
    num_blocks = pl.cdiv(width, block_size)
    order_width = pl.cdiv(num_blocks, hbm_alignment) * hbm_alignment
    order = list(range(1, num_blocks))
    random.Random(0).shuffle(order)
    order.insert(0, 0)
    order.extend([0] * (order_width - num_blocks))
    block_order = jnp.asarray(order, jnp.int32)
    output_width = pl.cdiv(k, hbm_alignment) * hbm_alignment
    # Retaining a quarter of the bootstrap establishes an exact cutoff without
    # radix-selecting a full large DMA block. This is a tuned scan policy, not
    # a sampled estimate: bootstrap winners remain resident throughout the pass.
    bootstrap_width = min(block_size, 4 * k)
    chunk_size = math.gcd(block_size, bootstrap_width)
    capacity = max(
        k + reconstruct_threshold + chunk_size, bootstrap_width, output_width
    )
    histogram_size = RADIX_BUCKETS
    id_digits = max(1, pl.cdiv((valid_width - 1).bit_length(), RADIX_BITS))
    id_capacity = max(capacity, output_width + mapping_width)
    score_bytes, id_bytes = scores.dtype.itemsize, jnp.dtype(jnp.int32).itemsize
    resident_bytes = score_bytes * (capacity + k) + id_bytes * (id_capacity + k)
    resident_bytes += id_bytes * (histogram_size + lanes + order_width)
    candidate_bytes = score_bytes * capacity + id_bytes * id_capacity
    # A second candidate bank removes compaction's read/write alias. Keep the
    # ordered in-place path when the caller's capacity exhausts VMEM.
    candidate_buffers = (
        2
        if resident_bytes + candidate_bytes + score_bytes * block_size
        <= info.vmem_capacity_bytes
        else 1
    )
    resident_bytes += (candidate_buffers - 1) * candidate_bytes
    buffers = (
        2
        if resident_bytes + 2 * score_bytes * block_size <= info.vmem_capacity_bytes
        else 1
    )
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(batch_size, info.num_subcores))
        if batch_size
        else 1,
    )
    out_type = jax.ShapeDtypeStruct((batch_size, output_width), jnp.int32)

    @pl.kernel(
        out_type=out_type,
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((buffers, block_size), jnp.float32),
            pltpu.VMEM((candidate_buffers * capacity,), jnp.float32),
            pltpu.VMEM((candidate_buffers * id_capacity,), jnp.int32),
            pltpu.VMEM((k,), jnp.float32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((histogram_size,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((order_width,), jnp.int32),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="streaming_topk",
    )
    def kernel(scores_ref, lengths_ref, order_ref, *refs):
        out_indices_ref, *scratch = refs[2:] if mapped else refs
        (
            blocks,
            candidate_banks,
            id_banks,
            selected,
            selected_ids,
            histogram,
            row_length,
            visit_order,
            block_sems,
            sem,
        ) = scratch
        candidates = candidate_banks.at[pl.ds(0, capacity)]
        candidate_ids = id_banks.at[pl.ds(0, id_capacity)]
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        pltpu.async_copy(order_ref, visit_order, sem).wait()
        if mapped:
            logical_blocks_ref, start_ref = refs[:2]
            pltpu.async_copy(start_ref, row_length, sem).wait()
            request_start = row_length[...][0]

        def store_candidates(count, keys, ids, mask):
            plsc.store_compressed(candidates.at[pl.ds(count, lanes)], keys, mask=mask)
            plsc.store_compressed(candidate_ids.at[pl.ds(count, lanes)], ids, mask=mask)

        def clear_histogram():
            @plsc.parallel_loop(0, histogram_size, lanes)
            def clear(i):
                histogram[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

        def sort_selected():
            # Reuse selection scratch. Stable byte passes keep padding behind
            # valid IDs even when their low bytes coincide with the sentinel.

            def sort_digit(digit):
                clear_histogram()

                @plsc.parallel_loop(0, k, lanes)
                def count_keys(i):
                    bits = candidate_ids[pl.ds(i, lanes)].astype(jnp.uint32)
                    digits = (
                        bits >> digit * RADIX_BITS & jnp.uint32(RADIX_MASK)
                    ).astype(jnp.int32)
                    counts, last = plsc.scan_count(digits)
                    plsc.addupdate_scatter(histogram, [digits], counts, mask=last)

                @plsc.parallel_loop(
                    0, histogram_size, lanes, carry=jnp.asarray(0, jnp.int32)
                )
                def prefix(i, total):
                    counts = histogram[pl.ds(i, lanes)]
                    cumulative = jnp.cumsum(counts)
                    histogram[pl.ds(i, lanes)] = total + cumulative - counts
                    return total + cumulative[-1]

                def scatter(vector, carry):
                    i = vector * lanes
                    ids = candidate_ids[pl.ds(i, lanes)]
                    digits = (
                        ids.astype(jnp.uint32) >> digit * RADIX_BITS
                        & jnp.uint32(RADIX_MASK)
                    ).astype(jnp.int32)
                    counts, last = plsc.scan_count(digits)
                    positions = plsc.load_gather(histogram, [digits]) + counts - 1
                    plsc.store_scatter(selected_ids, [positions], ids)
                    plsc.addupdate_scatter(histogram, [digits], counts, mask=last)
                    return carry

                jax.lax.fori_loop(0, k // lanes, scatter, ())

                @plsc.parallel_loop(0, k, lanes)
                def copy_sorted(i):
                    candidate_ids[pl.ds(i, lanes)] = selected_ids[pl.ds(i, lanes)]

            for digit in range(id_digits):
                sort_digit(digit)

        def compact(num_candidates):
            """Radix-select the lexicographic (score, -logical ID) winners."""

            @plsc.parallel_loop(0, k, lanes)
            def clear_selected(i):
                selected[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                selected_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            def cond(state):
                digit, remaining, found, bank = state
                return (digit >= 0) & (remaining > k - found)

            def radix_pass(state, *, by_id):
                digit, remaining, found, bank = state
                limit = pl.cdiv(remaining, lanes) * lanes

                next_bank = 1 - bank if candidate_buffers == 2 else bank
                source_keys = candidate_banks.at[pl.ds(bank * capacity, capacity)]
                source_ids = id_banks.at[pl.ds(bank * id_capacity, id_capacity)]
                destination_keys = candidate_banks.at[
                    pl.ds(next_bank * capacity, capacity)
                ]
                destination_ids = id_banks.at[
                    pl.ds(next_bank * id_capacity, id_capacity)
                ]

                clear_histogram()

                def digits_for(bits):
                    return bits >> (digit * RADIX_BITS).astype(jnp.uint32) & jnp.uint32(
                        RADIX_MASK
                    )

                @plsc.parallel_loop(0, limit, lanes)
                def fill_histogram(i):
                    bits = (
                        ~source_ids[pl.ds(i, lanes)].astype(jnp.uint32)
                        if by_id
                        else _ordered_bits(source_keys[pl.ds(i, lanes)])
                    )
                    digits = digits_for(bits)
                    valid = i + lane_ids < remaining
                    counts, mask = plsc.scan_count(digits, mask=valid)
                    plsc.addupdate_scatter(histogram, [digits], counts, mask=mask)

                below = remaining - (k - found)
                init = (jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32))

                @plsc.parallel_loop(0, histogram_size, lanes, carry=init)
                def find_pivot(i, carry):
                    pivot, prefix = carry
                    cumulative = jnp.cumsum(histogram[pl.ds(i, lanes)])
                    offset = plsc.all_reduce_ffs(prefix + cumulative > below)
                    return (pivot + offset, prefix + cumulative[-1])

                pivot, _ = find_pivot

                def filter_candidates(i, carry: tuple[jax.Array, jax.Array]):
                    survivors, outputs = carry
                    keys = source_keys[pl.ds(i, lanes)]
                    ids = source_ids[pl.ds(i, lanes)]
                    bits = ~ids.astype(jnp.uint32) if by_id else _ordered_bits(keys)
                    digits = digits_for(bits)
                    valid = i + lane_ids < remaining
                    accept = valid & (digits > pivot)
                    survive = valid & (digits == pivot)
                    plsc.store_compressed(
                        selected.at[pl.ds(outputs, lanes)], keys, mask=accept
                    )
                    plsc.store_compressed(
                        selected_ids.at[pl.ds(outputs, lanes)], ids, mask=accept
                    )
                    plsc.store_compressed(
                        destination_keys.at[pl.ds(survivors, lanes)], keys, mask=survive
                    )
                    plsc.store_compressed(
                        destination_ids.at[pl.ds(survivors, lanes)], ids, mask=survive
                    )
                    return (
                        survivors + plsc.all_reduce_population_count(survive)[0],
                        outputs + plsc.all_reduce_population_count(accept)[0],
                    )

                initial = (jnp.asarray(0, jnp.int32), found)
                if candidate_buffers == 2:
                    survivors, outputs = plsc.parallel_loop(
                        0, limit, lanes, carry=initial
                    )(filter_candidates)
                else:
                    survivors, outputs = jax.lax.fori_loop(
                        0,
                        pl.cdiv(remaining, lanes),
                        lambda i, carry: filter_candidates(i * lanes, carry),
                        initial,
                    )
                return (digit - 1, survivors, outputs, next_bank)

            _, remaining, found, bank = jax.lax.while_loop(
                cond,
                functools.partial(radix_pass, by_id=False),
                (
                    jnp.int32(FP32_BITS // RADIX_BITS - 1),
                    num_candidates,
                    jnp.int32(0),
                    jnp.int32(0),
                ),
            )
            score_ties = remaining > k - found
            _, remaining, found, bank = jax.lax.while_loop(
                cond,
                functools.partial(radix_pass, by_id=True),
                (jnp.int32(id_digits - 1), remaining, found, bank),
            )
            needed = jnp.minimum(remaining, k - found)

            @plsc.parallel_loop(0, pl.cdiv(needed, lanes) * lanes, lanes)
            def postfill(i):
                mask = i + lane_ids < needed
                plsc.store_compressed(
                    selected.at[pl.ds(found + i, lanes)],
                    candidate_banks[pl.ds(bank * capacity + i, lanes)],
                    mask=mask,
                )
                plsc.store_compressed(
                    selected_ids.at[pl.ds(found + i, lanes)],
                    id_banks[pl.ds(bank * id_capacity + i, lanes)],
                    mask=mask,
                )

            initial = (
                jnp.full(lanes, jnp.uint32(UINT32_MAX)),
                jnp.full(lanes, -1, jnp.int32),
                jnp.zeros(lanes, jnp.int32),
            )

            @plsc.parallel_loop(0, k, lanes, carry=initial)
            def retain_topk(i, carry):
                minimum, last_id, multiplicity = carry
                keys = selected[pl.ds(i, lanes)]
                ids = selected_ids[pl.ds(i, lanes)]
                candidates[pl.ds(i, lanes)] = keys
                candidate_ids[pl.ds(i, lanes)] = ids
                bits = jnp.where(ids >= 0, _ordered_bits(keys), jnp.uint32(UINT32_MAX))
                last_id = jnp.where(
                    bits < minimum,
                    ids,
                    jnp.where(bits == minimum, jnp.maximum(last_id, ids), last_id),
                )
                multiplicity = jnp.where(
                    bits < minimum,
                    1,
                    multiplicity + (bits == minimum).astype(jnp.int32),
                )
                return (jnp.minimum(minimum, bits), last_id, multiplicity)

            minimum, last_id, multiplicity = retain_topk
            threshold = jnp.min(minimum)
            threshold_id = jnp.max(jnp.where(minimum == threshold, last_id, -1))
            boundary_ties = score_ties | (
                jnp.sum(jnp.where(minimum == threshold, multiplicity, 0)) > 1
            )
            return _Selection(
                jnp.minimum(num_candidates, k),
                threshold,
                threshold_id,
                num_candidates >= k,
                boundary_ties,
            )

        def lookup(position):
            indices = jnp.full((lanes,), jnp.minimum(position, num_blocks - 1))
            return plsc.load_gather(visit_order, [indices])[0]

        def next_valid(position, length):
            def cond(state):
                step, block_id = state
                return (step < num_blocks) & (block_id * block_size >= length)

            def advance(state):
                step = state[0] + 1
                return (step, lookup(step))

            return jax.lax.while_loop(cond, advance, (position, lookup(position)))

        def transfer(row, block_id, slot, *, wait):
            def move(source, destination):
                copy = pltpu.make_async_copy(source, destination, block_sems.at[slot])
                if wait:
                    copy.wait()
                else:
                    copy.start()

            def full():
                move(
                    scores_ref.at[row, pl.ds(block_id * block_size, block_size)],
                    blocks.at[slot],
                )

            tail_size = width % block_size
            if tail_size:

                def tail():
                    move(
                        scores_ref.at[row, pl.ds(width - tail_size, tail_size)],
                        blocks.at[slot, pl.ds(0, tail_size)],
                    )

                if width < block_size:
                    tail()
                else:
                    jax.lax.cond(block_id == num_blocks - 1, tail, full)
            else:
                full()

        subcore = jax.lax.axis_index("subcore")
        core = jax.lax.axis_index("core")
        first_row = core * mesh.num_subcores + subcore
        row_stride = mesh.num_cores * mesh.num_subcores

        @pl.loop(first_row, batch_size, step=row_stride)
        def process_row(row):
            pltpu.async_copy(lengths_ref.at[row], row_length, sem).wait()
            length = jnp.min(row_length[:])
            row_blocks = pl.cdiv(length, block_size)

            def prime():
                position, block_id = next_valid(jnp.int32(0), length)
                transfer(row, block_id, jnp.int32(0), wait=False)
                return (position, block_id)

            position, block_id = jax.lax.cond(
                row_blocks > 0, prime, lambda: (jnp.int32(0), jnp.int32(0))
            )

            @plsc.parallel_loop(0, output_width, lanes)
            def clear_outputs(i):
                candidates[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                candidate_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            def scan_block(iteration, state):
                selection, position, block_id = state
                slot = iteration % buffers

                def prefetch():
                    next_position, next_id = next_valid(position + 1, length)
                    transfer(row, next_id, (iteration + 1) % buffers, wait=False)
                    return (next_position, next_id)

                def advance():
                    return jax.lax.cond(
                        iteration + 1 < row_blocks,
                        prefetch,
                        lambda: (position, block_id),
                    )

                next_position, next_id = (
                    advance() if buffers == 2 else (position, block_id)
                )
                jax.lax.cond(
                    iteration > 0,
                    lambda: transfer(row, block_id, slot, wait=True),
                    lambda: None,
                )

                def scan_chunk(visit, selection: _Selection):
                    chunk = block_size // chunk_size - 1 - visit
                    count, threshold, threshold_id, have_threshold, boundary_ties = (
                        selection
                    )
                    offset = block_id * block_size + chunk * chunk_size
                    cutoff_raw = jnp.where(
                        threshold >= jnp.uint32(FP32_SIGN_BIT),
                        threshold ^ jnp.uint32(FP32_SIGN_BIT),
                        ~threshold,
                    )
                    cutoff_raw = jnp.where(
                        have_threshold, cutoff_raw, jnp.uint32(FP32_NEG_INF_BITS)
                    )
                    cutoff_hi = cutoff_raw >> jnp.uint32(BF16_BITS)
                    cutoff_pair = cutoff_hi | cutoff_hi << jnp.uint32(BF16_BITS)
                    cutoff_bf16 = plsc.bitcast(
                        jnp.full((lanes,), cutoff_pair, jnp.uint32), jnp.bfloat16
                    )
                    paired_size = chunk_size // (2 * lanes) * (2 * lanes)

                    # BF16 high words only screen; retain original FP32 values
                    # and use inclusive comparisons so the screen cannot lose
                    # a winner. The final radix selection remains exact.
                    def append_packed(inclusive, full_block, count: jax.Array):
                        @plsc.parallel_loop(0, paired_size, 2 * lanes, carry=count)
                        def append_pairs(i, count):
                            keys0 = blocks[slot, pl.ds(chunk * chunk_size + i, lanes)]
                            keys1 = blocks[
                                slot, pl.ds(chunk * chunk_size + (i + lanes), lanes)
                            ]
                            raw0 = keys0.view(jnp.uint32)
                            raw1 = keys1.view(jnp.uint32)
                            packed = raw0 >> jnp.uint32(BF16_BITS) | raw1 & jnp.uint32(
                                BF16_HIGH_MASK
                            )
                            projected = plsc.bitcast(packed, jnp.bfloat16)
                            screen = (
                                projected >= cutoff_bf16
                                if inclusive
                                else projected > cutoff_bf16
                            )
                            mask_words = plsc.bitcast(
                                screen.astype(jnp.uint16), jnp.uint32
                            )
                            mask0 = mask_words & jnp.uint32(1) != 0
                            mask1 = mask_words & jnp.uint32(1 << BF16_BITS) != 0
                            ids0 = offset + i + lane_ids
                            ids1 = ids0 + lanes
                            accept0 = mask0
                            accept1 = mask1
                            if not full_block:
                                accept0 = accept0 & (ids0 < length)
                                accept1 = accept1 & (ids1 < length)
                            added0 = plsc.all_reduce_population_count(accept0)[0]
                            added1 = plsc.all_reduce_population_count(accept1)[0]
                            store_candidates(count, keys0, ids0, accept0)
                            store_candidates(count + added0, keys1, ids1, accept1)
                            return count + added0 + added1

                        count = append_pairs
                        if paired_size != chunk_size:
                            keys = blocks[
                                slot, pl.ds(chunk * chunk_size + paired_size, lanes)
                            ]
                            ids = offset + paired_size + lane_ids
                            projected = (
                                keys.view(jnp.uint32) & jnp.uint32(BF16_HIGH_MASK)
                            ).view(jnp.float32)
                            cutoff = jnp.full(
                                lanes, cutoff_hi << jnp.uint32(BF16_BITS), jnp.uint32
                            ).view(jnp.float32)
                            accept = (
                                projected >= cutoff if inclusive else projected > cutoff
                            )
                            if not full_block:
                                accept = accept & (ids < length)
                            store_candidates(count, keys, ids, accept)
                            count = count + plsc.all_reduce_population_count(accept)[0]
                        return count

                    def packed_mode(inclusive):
                        return jax.lax.cond(
                            offset + chunk_size <= length,
                            lambda: append_packed(inclusive, True, count),
                            lambda: append_packed(inclusive, False, count),
                        )

                    def append_exact():
                        def append(full_block):
                            @plsc.parallel_loop(0, chunk_size, lanes, carry=count)
                            def append_candidates(i, count):
                                keys = blocks[
                                    slot, pl.ds(chunk * chunk_size + i, lanes)
                                ]
                                ids = offset + i + lane_ids
                                bits = _ordered_bits(keys)
                                accept = (bits > threshold) | (bits == threshold) & (
                                    ids < threshold_id
                                )
                                if not full_block:
                                    accept = accept & (ids < length)
                                store_candidates(count, keys, ids, accept)
                                return (
                                    count + plsc.all_reduce_population_count(accept)[0]
                                )

                            return append_candidates

                        return jax.lax.cond(
                            offset + chunk_size <= length,
                            lambda: append(True),
                            lambda: append(False),
                        )

                    count = jax.lax.cond(
                        boundary_ties,
                        append_exact,
                        lambda: jax.lax.cond(
                            have_threshold,
                            lambda: packed_mode(True),
                            lambda: packed_mode(not exclude_neg_inf),
                        ),
                    )
                    should_compact = (
                        count >= k + reconstruct_threshold
                    ) | ~have_threshold & (count >= k)
                    return jax.lax.cond(
                        should_compact,
                        lambda: compact(count),
                        lambda: selection._replace(count=count),
                    )

                # Bootstrap consumes a single-chunk block on its first visit.
                # Express the remaining zero-or-one scan directly, without a loop.
                if block_size == chunk_size:
                    selection = jax.lax.cond(
                        iteration > 0,
                        lambda: scan_chunk(0, selection),
                        lambda: selection,
                    )
                else:
                    chunks_to_scan = block_size // chunk_size - jnp.where(
                        iteration == 0, bootstrap_width // chunk_size, 0
                    )
                    selection = jax.lax.fori_loop(
                        jnp.int32(0), chunks_to_scan, scan_chunk, selection
                    )
                selection = jax.lax.cond(
                    (iteration + 1 == row_blocks) & (selection.count > k),
                    lambda: compact(selection.count),
                    lambda: selection,
                )
                if buffers == 1:
                    next_position, next_id = advance()
                return (selection, next_position, next_id)

            initial_threshold = jnp.uint32(ORDERED_NEG_INF if exclude_neg_inf else 0)

            def establish_threshold():
                transfer(row, block_id, jnp.int32(0), wait=True)

                @plsc.parallel_loop(
                    0, bootstrap_width, lanes, carry=jnp.asarray(0, jnp.int32)
                )
                def bootstrap(i, count):
                    keys = blocks[0, pl.ds(i, lanes)]
                    ids = block_id * block_size + i + lane_ids
                    accept = ids < length
                    if exclude_neg_inf:
                        accept = accept & (keys > -jnp.inf)
                    store_candidates(count, keys, ids, accept)
                    return count + plsc.all_reduce_population_count(accept)[0]

                selection = compact(bootstrap)
                have = selection.have_threshold
                return selection._replace(
                    threshold=jnp.where(have, selection.threshold, initial_threshold),
                    boundary_ties=have & selection.boundary_ties,
                )

            seed = jax.lax.cond(
                row_blocks > 0,
                establish_threshold,
                lambda: _Selection(
                    jnp.int32(0),
                    initial_threshold,
                    jnp.int32(-1),
                    jnp.bool_(False),
                    jnp.bool_(False),
                ),
            )
            if bootstrap_width < valid_width:
                jax.lax.fori_loop(0, row_blocks, scan_block, (seed, position, block_id))
            if mapped:
                # Selection is finished: reuse the ID tail for metadata.
                # Sorting touches only the first k entries and overlaps this DMA.
                mapping_copy = pltpu.make_async_copy(
                    logical_blocks_ref.at[request_start + row // queries],
                    candidate_ids.at[pl.ds(output_width, mapping_width)],
                    sem,
                )
                mapping_copy.start()
                sort_selected()
                mapping_copy.wait()

                @plsc.parallel_loop(0, k, lanes)
                def map_selected(i):
                    slots = candidate_ids[pl.ds(i, lanes)]
                    blocks = plsc.load_gather(
                        candidate_ids,
                        [output_width + jnp.maximum(slots, 0) // records_per_block],
                    )
                    logical = blocks * records_per_block + slots % records_per_block
                    candidate_ids[pl.ds(i, lanes)] = jnp.where(slots >= 0, logical, -1)
            else:
                sort_selected()
            pltpu.async_copy(
                candidate_ids.at[:output_width], out_indices_ref.at[row], sem
            ).wait()

    if logical_blocks is not None:
        start = jnp.broadcast_to(
            jnp.asarray(0 if start_request is None else start_request, jnp.int32),
            (lanes,),
        )
        result = kernel(scores, lengths, block_order, logical_blocks, start)
    else:
        result = kernel(scores, lengths, block_order)
    if output_width != k:
        result = result[:, :k]
    if scheduling_group_id is not None:
        result = xla_metadata.set_xla_metadata(
            result, _scheduling_group_id=scheduling_group_id
        )
    return result


__all__ = ["streaming_topk"]

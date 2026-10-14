# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024-2025 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill full-row SparseCore radix selection.

Adapted from sgl-jax radix_topk/v2/kernel.py, revision
15081a48747885a12c99d63b70935b529fc41219 (Apache-2.0).
Changes: logical indices only, forward scanning only, prefill defaults and
repository interface. No page remapping or decode routing.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _ordered_bits(keys: jax.Array) -> jax.Array:
    """Match v1's FP32 radix ordering, including signed zero."""
    bits = keys.view(jnp.uint32)
    sign = (bits.view(jnp.int32) >> 31).view(jnp.uint32)
    return bits ^ (sign | jnp.uint32(1 << 31))


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "block_size",
        "reconstruct_threshold",
    ),
)
def prefill_radix_topk(
    scores: jax.Array,
    *,
    valid_lengths: jax.Array | None = None,
    k: int = 512,
    block_size: int = 8192,
    reconstruct_threshold: int = 8192,
) -> jax.Array:
    """Return unordered logical TopK indices, padded with -1.

    Each SparseCore subcore scans complete FP32 rows in increasing index order.
    B controls scan DMA size; B2 controls candidate compaction, not grouped
    local TopK. Equal bit-pattern scores retain the lowest logical indices:
    stable compaction preserves their order, and subsequent blocks only append
    scores strictly above the retained threshold. Signed zeros use FP32 radix
    order (+0 above -0), as in the existing SparseCore selector.

    Callers exclude selected -inf values and sort logical indices. Keeping this
    adapter outside the kernel matches the existing benchmark's full cost.
    Inputs must contain ordinary FP32 scores or -inf masking, not NaNs.
    """
    if scores.ndim != 2 or scores.dtype != jnp.float32:
        raise ValueError("scores must be a rank-2 float32 array")
    batch_size, width = scores.shape
    if not 1 <= k <= width:
        raise ValueError("k must be in [1, N]")
    if block_size <= 0 or reconstruct_threshold <= 0:
        raise ValueError("block_size and reconstruct_threshold must be positive")
    if valid_lengths is not None and (
        valid_lengths.shape != (batch_size,) or valid_lengths.dtype != jnp.int32
    ):
        raise ValueError("valid_lengths must be an int32 vector with one entry per row")
    valid_width = width
    # HBM rows require 128-element alignment, but need not be B-aligned.
    # Padding to B here copies the entire score matrix on every pipeline step.
    # Preserve the existing path for unusual sub-128/non-tile-aligned blocks.
    input_alignment = 128 if block_size % 128 == 0 else block_size
    padding = (-width) % input_alignment
    if padding:
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        width += padding

    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if any(n % lanes for n in (k, block_size, reconstruct_threshold)):
        raise ValueError("k, block_size and reconstruct_threshold must be lane-aligned")
    lengths = (
        jnp.full((batch_size,), valid_width, jnp.int32)
        if valid_lengths is None
        else jnp.clip(valid_lengths, 0, valid_width)
    )
    lengths = jnp.broadcast_to(lengths[:, None], (batch_size, lanes))
    # HBM output rows need a full 128-element tile, including for k=16.
    output_width = pl.cdiv(k, 128) * 128
    capacity = max(k + reconstruct_threshold + block_size, output_width)
    histogram_size = 256  # Four exact 8-bit radix passes for FP32.
    # Keys + indices for candidates and selected K, plus one input DMA buffer.
    scratch_bytes = 8 * (capacity + k) + 4 * (block_size + histogram_size + lanes)
    if scratch_bytes > info.vmem_capacity_bytes:
        raise ValueError(
            f"RadixTopKV2 scratch needs {scratch_bytes} bytes per subcore, "
            f"but only {info.vmem_capacity_bytes} bytes are available"
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
            pltpu.VMEM((block_size,), jnp.float32),
            pltpu.VMEM((capacity,), jnp.float32),
            pltpu.VMEM((capacity,), jnp.int32),
            pltpu.VMEM((k,), jnp.float32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((histogram_size,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="prefill_radix_topk",
    )
    def kernel(scores_ref, lengths_ref, out_indices_ref, *scratch):
        (
            block,
            candidates,
            candidate_ids,
            selected,
            selected_ids,
            histogram,
            row_length,
            sem,
        ) = scratch
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)

        def compact(num_candidates):
            """In-place exact radix selection; retain K pairs in candidates."""

            @plsc.parallel_loop(0, k, lanes)
            def clear_selected(i):
                selected[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                selected_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            # State: current digit, active candidates, already accepted outputs.
            def cond(state):
                digit, remaining, found = state
                # If all remaining candidates are needed, no more digits are
                # necessary. After the final digit, survivors are tied.
                return (digit >= 0) & (remaining > k - found)

            def radix_pass(state):
                digit, remaining, found = state
                limit = pl.cdiv(remaining, lanes) * lanes

                @plsc.parallel_loop(0, histogram_size, lanes)
                def clear_histogram(i):
                    histogram[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

                def digits_for(keys):
                    return (
                        _ordered_bits(keys) >> (digit * 8).astype(jnp.uint32)
                    ) & jnp.uint32(255)

                @plsc.parallel_loop(0, limit, lanes)
                def fill_histogram(i):
                    digits = digits_for(candidates[pl.ds(i, lanes)])
                    valid = i + lane_ids < remaining
                    counts, mask = plsc.scan_count(digits, mask=valid)
                    plsc.addupdate_scatter(histogram, [digits], counts, mask=mask)

                # Locate the bucket containing the (remaining - needed + 1)th
                # smallest key. All higher buckets are guaranteed winners.
                below = remaining - (k - found)
                init = (jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32))

                @plsc.parallel_loop(0, histogram_size, lanes, carry=init)
                def find_pivot(i, carry):
                    pivot, prefix = carry
                    cumulative = jnp.cumsum(histogram[pl.ds(i, lanes)])
                    offset = plsc.all_reduce_ffs(prefix + cumulative > below)
                    return pivot + offset, prefix + cumulative[-1]

                pivot, _ = find_pivot

                @plsc.parallel_loop(0, limit, lanes, carry=(jnp.int32(0), found))
                def filter_candidates(i, carry):
                    survivors, outputs = carry
                    keys = candidates[pl.ds(i, lanes)]
                    ids = candidate_ids[pl.ds(i, lanes)]
                    digits = digits_for(keys)
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
                        candidates.at[pl.ds(survivors, lanes)], keys, mask=survive
                    )
                    plsc.store_compressed(
                        candidate_ids.at[pl.ds(survivors, lanes)], ids, mask=survive
                    )
                    return (
                        survivors + plsc.all_reduce_population_count(survive)[0],
                        outputs + plsc.all_reduce_population_count(accept)[0],
                    )

                survivors, outputs = filter_candidates
                return digit - 1, survivors, outputs

            _, remaining, found = jax.lax.while_loop(
                cond, radix_pass, (jnp.int32(3), num_candidates, jnp.int32(0))
            )
            needed = jnp.minimum(remaining, k - found)

            @plsc.parallel_loop(0, pl.cdiv(needed, lanes) * lanes, lanes)
            def postfill(i):
                mask = i + lane_ids < needed
                plsc.store_compressed(
                    selected.at[pl.ds(found + i, lanes)],
                    candidates[pl.ds(i, lanes)],
                    mask=mask,
                )
                plsc.store_compressed(
                    selected_ids.at[pl.ds(found + i, lanes)],
                    candidate_ids[pl.ds(i, lanes)],
                    mask=mask,
                )

            @plsc.parallel_loop(
                0, k, lanes, carry=jnp.full(lanes, jnp.uint32(0xFFFFFFFF))
            )
            def retain_topk(i, minimum):
                keys = selected[pl.ds(i, lanes)]
                candidates[pl.ds(i, lanes)] = keys
                candidate_ids[pl.ds(i, lanes)] = selected_ids[pl.ds(i, lanes)]
                return jnp.minimum(minimum, _ordered_bits(keys))

            return jnp.min(retain_topk)

        subcore = jax.lax.axis_index("subcore")
        core = jax.lax.axis_index("core")
        first_row = core * mesh.num_subcores + subcore
        row_stride = mesh.num_cores * mesh.num_subcores

        @pl.loop(first_row, batch_size, step=row_stride)
        def process_row(row):
            pltpu.async_copy(lengths_ref.at[row], row_length, sem).wait()
            length = jnp.min(row_length[:])
            num_blocks = pl.cdiv(length, block_size)

            @plsc.parallel_loop(0, output_width, lanes)
            def clear_outputs(i):
                candidates[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                candidate_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            # Carry: candidate count, last Kth ordered key, threshold initialized.
            def scan_block(block_id, state):
                count, threshold, have_threshold = state
                offset = block_id * block_size

                def load_full_block():
                    copy = pltpu.async_copy(
                        scores_ref.at[row, pl.ds(offset, block_size)], block, sem
                    )
                    copy.wait()

                tail_size = width % block_size
                if tail_size:

                    def load_tail_block():
                        # Only the final physical block can be short. Keep
                        # DMA sizes static and mask its scratch padding via
                        # the original valid length when appending candidates.
                        @plsc.parallel_loop(0, block_size, lanes)
                        def clear_input(i):
                            block[pl.ds(i, lanes)] = jnp.full(
                                lanes, -jnp.inf, jnp.float32
                            )

                        pltpu.async_copy(
                            scores_ref.at[row, pl.ds(width - tail_size, tail_size)],
                            block.at[pl.ds(0, tail_size)],
                            sem,
                        ).wait()

                    if width < block_size:
                        load_tail_block()
                    else:
                        jax.lax.cond(
                            offset == width - tail_size,
                            load_tail_block,
                            load_full_block,
                        )
                else:
                    load_full_block()

                @plsc.parallel_loop(0, block_size, lanes, carry=count)
                def append_candidates(i, count):
                    keys = block[pl.ds(i, lanes)]
                    ids = offset + i + lane_ids
                    # Exclude scratch padding before compaction. The caller
                    # removes real -inf entries from the returned selection.
                    accept = (ids < length) & (
                        ~have_threshold | (_ordered_bits(keys) > threshold)
                    )
                    plsc.store_compressed(
                        candidates.at[pl.ds(count, lanes)], keys, mask=accept
                    )
                    plsc.store_compressed(
                        candidate_ids.at[pl.ds(count, lanes)], ids, mask=accept
                    )
                    return count + plsc.all_reduce_population_count(accept)[0]

                count = append_candidates
                should_compact = (count >= k + reconstruct_threshold) | (
                    block_id == num_blocks - 1
                )

                def reconstruct():
                    return jnp.int32(k), compact(count), jnp.bool_(True)

                return jax.lax.cond(
                    should_compact,
                    reconstruct,
                    lambda: (count, threshold, have_threshold),
                )

            jax.lax.fori_loop(
                0,
                num_blocks,
                scan_block,
                (jnp.int32(0), jnp.uint32(0), jnp.bool_(False)),
            )
            pltpu.async_copy(
                candidate_ids.at[:output_width], out_indices_ref.at[row], sem
            ).wait()

    result = kernel(scores, lengths)
    if output_width != k:
        result = jax.tree.map(lambda x: x[:, :k], result)
    return result


__all__ = ["prefill_radix_topk"]

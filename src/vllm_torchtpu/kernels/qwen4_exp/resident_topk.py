# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen-local resident-prefix policy using the shared exact SC selection body."""

import functools
import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    LANES,
    MAX_COOPERATIVE_PARTITIONS,
    MAX_SLICE_WORDS,
    NUM_BUCKETS,
    _align_to,
    _cdiv,
    _pick_partition,
    _sc_topk_direct,
    _topk_body,
    _topk_scratch_types,
    _validate_topk_inputs,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)


def _resident_prefix_words(
    b: int,
    n: int,
    k: int,
    *,
    return_scores: bool,
    capacity_bytes: int,
    num_lanes: int,
    dma_granule_bytes: int,
    pipeline_lanes: int,
) -> int | None:
    """One resident-row budget, independent of device lengths or context buckets.

    MAX_SLICE_WORDS reserves half a 512KiB tile for double-buffered scores.
    Account for the named P1 histogram/row/length/output buffers as well; the
    existing cap preserves room for compiler state. Cooperative slices shrink
    with P and retain the original scratch/partition and compiler resource gates.
    """
    alignment = math.lcm(LANES, num_lanes, _cdiv(dma_granule_bytes, 4))
    word_bytes = np.dtype(np.int32).itemsize
    outputs = 2 if return_scores else 1
    scratch_words = NUM_BUCKETS + LANES + outputs * (k + LANES)
    pipeline_words = 2 * (LANES + outputs * k)
    overhead = word_bytes * (scratch_words + pipeline_words)
    words = min(MAX_SLICE_WORDS, (capacity_bytes - overhead) // (2 * word_bytes))
    # Default pipeline buffers use compact tiling, even on SparseCore.
    # Their slice size must align; HBM pitch only needs the SC DMA alignment.
    slice_alignment = math.lcm(alignment, pipeline_lanes)
    # Every possible power-of-two partition divides this quantum, so each
    # cooperative slice retains the pipeline alignment.
    quantum = MAX_COOPERATIVE_PARTITIONS * slice_alignment
    resident = words // quantum * quantum
    if resident < k or resident >= n or n % alignment:
        return None
    p = _pick_partition(b, resident, k)
    slice_len = _align_to(resident // p, LANES)
    if p * slice_len > n or slice_len % slice_alignment:
        return None
    return resident


@functools.cache
def _resident_body_for_shape(
    b,
    n,
    k,
    p,
    slice_len,
    num_waves,
    write_empty,
    return_scores,
    resident_n,
    prefix_p,
    prefix_waves,
):
    """Reuse static prefix/full bodies without capturing a scheduling ID."""
    body_fn = functools.partial(
        _topk_body,
        b=b,
        n=n,
        k=k,
        p=p,
        slice_len=slice_len,
        num_waves=num_waves,
        write_empty=write_empty,
        return_scores=return_scores,
    )
    full_body = body_fn
    prefix_body = functools.partial(
        _topk_body,
        b=b,
        n=resident_n,
        k=k,
        p=prefix_p,
        slice_len=_align_to(resident_n // prefix_p, LANES),
        num_waves=prefix_waves,
        write_empty=write_empty,
        return_scores=return_scores,
        physical_pitch=p * slice_len,
    )

    def body_fn(scores, lengths, prefix_lengths, choice, out, keys, *scratch):
        choice_vmem = scratch[-1]
        common_scratch = scratch[:-1]
        pltpu.sync_copy(choice, choice_vmem)
        enabled = choice_vmem[pl.ds(0, LANES)][0] != 0

        @pl.when(enabled)
        def prefix():
            prefix_body(scores, prefix_lengths, out, keys, *common_scratch)

        @pl.when(~enabled)
        def full():
            full_body(scores, lengths, out, keys, *common_scratch)

    return body_fn


def _sc_topk_resident(
    scores: jax.Array,  # f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    p: int = 1,
    write_empty: bool = True,
    return_scores: bool = True,
    scheduling_group_id: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Assemble Qwen resident/full pipelines inside one SC custom call."""
    b, n = scores.shape
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    groups_per_chip = 32 // p
    slice_len = _align_to(n // p, LANES)
    num_waves = _cdiv(b, groups_per_chip)
    padded_b = num_waves * groups_per_chip
    pad_b = padded_b - b
    pad_n = slice_len * p - n
    if pad_b > 0 or pad_n > 0:
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))

    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b))
    if p > 1:
        lengths = jnp.repeat(lengths, p)
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1)))

    resident_n = None
    prefix_p = prefix_waves = prefix_padded_b = 0
    candidate_n = _resident_prefix_words(
        b,
        n,
        k,
        return_scores=return_scores,
        capacity_bytes=sc.vmem_capacity_bytes,
        num_lanes=sc.num_lanes,
        dma_granule_bytes=sc.dma_granule_size_bytes,
        pipeline_lanes=info.num_lanes,
    )
    if candidate_n is not None:
        prefix_p = _pick_partition(b, candidate_n, k)
        prefix_waves = _cdiv(b, 32 // prefix_p)
        prefix_padded_b = prefix_waves * (32 // prefix_p)

        def pipeline_words(extent, partitions, waves):
            size = _align_to(extent // partitions, LANES)
            buffers = 1 if waves == 1 or size > MAX_SLICE_WORDS else 2
            outputs = (1 + int(return_scores)) * k if partitions == 1 else 0
            return buffers * (size + LANES + outputs)

        scratch_words = NUM_BUCKETS + LANES + (1 + int(return_scores)) * (k + LANES)
        if p > 1:
            scratch_words += p * NUM_BUCKETS + LANES
            scratch_words += (1 + int(return_scores)) * (k + LANES)
        # Conservative sum: no reliance on branch-local VMEM reuse.
        total_words = scratch_words + LANES
        total_words += pipeline_words(n, p, num_waves)
        total_words += pipeline_words(candidate_n, prefix_p, prefix_waves)
        if prefix_p <= p and 4 * total_words <= sc.vmem_capacity_bytes:
            resident_n = candidate_n

    if resident_n is None:
        return _sc_topk_direct(
            scores,
            k,
            row_lengths,
            p=p,
            write_empty=write_empty,
            return_scores=return_scores,
            scheduling_group_id=scheduling_group_id,
        )

    prefix_lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, prefix_padded_b - b))
    if prefix_p > 1:
        prefix_lengths = jnp.repeat(prefix_lengths, prefix_p)
    prefix_lengths = jnp.pad(prefix_lengths[:, None], ((0, 0), (0, LANES - 1)))
    use_prefix = jnp.all(jnp.clip(row_lengths, 0, n) <= resident_n)
    choice = jnp.full((LANES,), use_prefix, jnp.int32)
    extra_inputs = (prefix_lengths.reshape(-1), choice)
    output_b = max(padded_b, prefix_padded_b)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((output_b * k,), jnp.int32)
    out_type = (slot_type, slot_type if return_scores else None)

    scratch_types = _topk_scratch_types(k, p, return_scores, sc.num_subcores)

    scratch_types = (*scratch_types, pltpu.VMEM((LANES,), jnp.int32))
    body_fn = _resident_body_for_shape(
        b,
        n,
        k,
        p,
        slice_len,
        num_waves,
        write_empty,
        return_scores,
        resident_n,
        prefix_p,
        prefix_waves,
    )

    out = pl.kernel(
        body_fn,
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p{p}_resident{resident_n}",
    )(words.reshape(-1), lengths_2d.reshape(-1), *extra_inputs)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    # pl.kernel and set_xla_metadata are untyped upstream; both return the
    # out_type pytree here.
    slots, keys = cast(tuple[jax.Array, jax.Array | None], out)
    slots_2d = slots.reshape(output_b, k)[:b]
    if return_scores:
        assert keys is not None
        keys_2d = keys.reshape(output_b, k)[:b]
        return slots_2d, keys_2d
    return slots_2d


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "write_empty_rows",
        "scheduling_group_id",
        "return_scores",
    ),
)
def qwen_sparsecore_topk(
    scores: jax.Array,  # f32[b, n] or i32[b, n] of f32 bits
    k: int,
    row_lengths: jax.Array | None = None,  # i32[b], defaults to n
    *,
    write_empty_rows: bool = True,
    scheduling_group_id: int | None = None,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:  # i32[b, k] or (i32[b, k], i32[b, k])
    """Exact selection with Qwen's resource-bounded resident prefix policy.

    Resource admission is static; the prefix/full predicate stays on device
    inside one SC call. Omitted lengths retain the shared full-extent behavior.
    """
    b, n = _validate_topk_inputs(scores, k)

    if row_lengths is None:
        return sparsecore_topk(
            scores,
            k,
            write_empty_rows=write_empty_rows,
            scheduling_group_id=scheduling_group_id,
            return_scores=return_scores,
        )
    row_lengths = row_lengths.astype(jnp.int32)

    p = _pick_partition(b, n, k)
    return _sc_topk_resident(
        scores,
        k,
        row_lengths,
        p=p,
        write_empty=write_empty_rows,
        return_scores=return_scores,
        scheduling_group_id=scheduling_group_id,
    )


class QwenSparseCoreTopK(SparseCoreTopK):
    """Qwen's resource-bounded selection policy over shared SparseCore math."""

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        return qwen_sparsecore_topk(
            score_bits,
            k,
            row_lengths=row_lengths,
            write_empty_rows=True,
            scheduling_group_id=scheduling_group_id,
            return_scores=return_scores,
        )

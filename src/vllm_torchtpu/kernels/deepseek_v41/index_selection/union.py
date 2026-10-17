# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build candidate memberships and ascending request unions in SparseCore VMEM."""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


@functools.partial(jax.jit, static_argnames=("logical_blocks", "capacity"))
def sparse_union_metadata(
    candidates: jax.Array, logical_blocks: int, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build sorted IDs and query bits; retain the full count on overflow.

    Each subcore accumulates query-local histograms in VMEM, then streams
    their membership masks in ID order without an HBM scatter or sort.
    """
    requests, queries, candidates_per_query = candidates.shape
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    histogram_width = pl.cdiv(logical_blocks, 128) * 128
    output_width = pl.cdiv(capacity, 128) * 128
    if (
        candidates.dtype != jnp.int32
        or candidates_per_query % 128
        or not 1 <= queries <= 8
        or min(requests, candidates_per_query, logical_blocks, capacity) < 1
    ):
        raise ValueError(
            "Expected nonempty aligned int32 candidate rows and 1..8 queries"
        )
    scratch = 4 * (
        queries * histogram_width
        + candidates_per_query
        + 2 * (output_width + lanes)
        + 128
    )
    if scratch > info.vmem_capacity_bytes:
        raise ValueError(
            f"Candidate union needs {scratch} bytes per subcore, "
            f"available {info.vmem_capacity_bytes}"
        )
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(requests, info.num_subcores)),
    )
    shape = jax.ShapeDtypeStruct((requests, output_width), jnp.int32)

    @pl.kernel(
        out_type=(shape, shape, jax.ShapeDtypeStruct((requests, 128), jnp.int32)),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((queries * histogram_width,), jnp.int32),
            pltpu.VMEM((candidates_per_query,), jnp.int32),
            pltpu.VMEM((output_width + lanes,), jnp.int32),
            pltpu.VMEM((output_width + lanes,), jnp.int32),
            pltpu.VMEM((128,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="candidate_union_metadata_sc",
    )
    def kernel(src, ids_out, bits_out, count_out, hist, block, ids, bits, counts, sem):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index("core") * mesh.num_subcores + jax.lax.axis_index(
            "subcore"
        )

        @pl.loop(first, requests, step=mesh.num_cores * mesh.num_subcores)
        def request(row):
            @plsc.parallel_loop(0, queries * histogram_width, lanes)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            @pl.loop(0, queries)
            def query(query_index):
                pltpu.async_copy(src.at[row, query_index], block, sem).wait()

                @plsc.parallel_loop(0, candidates_per_query, lanes)
                def scatter(i):
                    keys = block[pl.ds(i, lanes)]
                    valid = (keys >= 0) & (keys < logical_blocks)
                    inc, mask = plsc.scan_count(keys, mask=valid)
                    addresses = query_index * histogram_width + jnp.clip(
                        keys, 0, histogram_width - 1
                    )
                    plsc.addupdate_scatter(hist, [addresses], inc, mask=mask)

            @plsc.parallel_loop(0, output_width, lanes)
            def clear_out(i):
                ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)
                bits[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            def compact(i: jax.Array, n: jax.Array) -> jax.Array:
                membership = jnp.zeros(lanes, jnp.int32)
                for query_index in range(queries):
                    membership = membership | (
                        (
                            hist[pl.ds(query_index * histogram_width + i, lanes)] > 0
                        ).astype(jnp.int32)
                        << query_index
                    )
                valid = (membership != 0) & (i + lane_ids < logical_blocks)
                keep = valid & (n + jnp.cumsum(valid.astype(jnp.int32)) <= capacity)
                off = jnp.minimum(n, output_width)
                plsc.store_compressed(
                    ids.at[pl.ds(off, lanes)], i + lane_ids, mask=keep
                )
                plsc.store_compressed(bits.at[pl.ds(off, lanes)], membership, mask=keep)
                return n + plsc.all_reduce_population_count(valid)[0]

            # The public annotation omits parallel_loop's carry overload.
            compact_loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(0, histogram_width, lanes, carry=jnp.int32(0)),
            )
            count = compact_loop(compact)

            @plsc.parallel_loop(0, 128, lanes)
            def save_count(i):
                counts[pl.ds(i, lanes)] = jnp.full(lanes, count, jnp.int32)

            pltpu.async_copy(
                ids.at[pl.ds(0, output_width)], ids_out.at[row], sem
            ).wait()
            pltpu.async_copy(
                bits.at[pl.ds(0, output_width)], bits_out.at[row], sem
            ).wait()
            pltpu.async_copy(counts, count_out.at[row], sem).wait()

    ids, bits, count = kernel(candidates)
    return ids[:, :capacity], bits[:, :capacity], count[:, 0], count[:, 0] > capacity

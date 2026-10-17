# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore stable counting positions without a dense rank tensor."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as sc

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k


def counting_positions(keys, buckets, *, finish_sc=True, prefix_sc=False):
    info = pltpu.get_tpu_info().sparse_core
    workers = info.num_cores * info.num_subcores
    size = keys.shape[0]
    chunk = size // workers
    width = pl.cdiv(buckets, 16) * 16
    if size % workers or chunk % 16:
        raise ValueError("Expected a multiple of 16 keys per SC worker")
    mesh = sc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="count_core",
        subcore_axis_name="count_subcore",
    )

    def kernel(kh, rh, hh, kv, rv, hv, sem):
        worker = lax.axis_index(("count_core", "count_subcore"))
        copy = pltpu.make_async_copy(kh.at[pl.ds(worker * chunk, chunk)], kv, sem)
        copy.start()
        copy.wait()

        @sc.parallel_loop(0, width, step=16)
        def zero(i):
            hv[pl.ds(i, 16)] = jnp.zeros((16,), jnp.int32)

        @pl.loop(0, chunk // 16)
        def each(i):
            key = kv[pl.ds(i * 16, 16)]
            duplicate, last = sc.scan_count(key)
            previous = sc.load_gather(hv, (key,))
            # scan_count is inclusive: first occurrence has count one.
            rank = previous + duplicate
            rv[pl.ds(i * 16, 16)] = rank - 1
            sc.store_scatter(hv, (key,), rank, mask=last)

        copies = (
            pltpu.make_async_copy(rv, rh.at[pl.ds(worker * chunk, chunk)], sem),
            pltpu.make_async_copy(hv, hh.at[worker, :], sem),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    ranks, hist = k.core_map_kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((size,), jnp.int32),
            jax.ShapeDtypeStruct((workers, width), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((chunk,), jnp.int32),
            pltpu.VMEM((chunk,), jnp.int32),
            pltpu.VMEM((width,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        cost_estimate=pl.CostEstimate(
            flops=0, transcendentals=0, bytes_accessed=size * 8 + workers * width * 4
        ),
        name="sc_count_local_ranks",
    )(keys)
    counts = hist.sum(axis=0)
    if prefix_sc:
        positions = _finish_positions(
            keys, ranks, hist, mesh, chunk, width, prefix_sc=True
        )
    else:
        offsets = jnp.cumsum(hist, axis=0) - hist + jnp.cumsum(counts) - counts
        if finish_sc:
            positions = _finish_positions(keys, ranks, offsets, mesh, chunk, width)
        else:
            # Alternative XLA lookup (not used by default).
            positions = ranks.reshape(workers, chunk) + jnp.take_along_axis(
                offsets, keys.reshape(workers, chunk), axis=1
            )
    return positions.reshape(-1), counts[:buckets]


def _finish_positions(keys, ranks, offsets, mesh, chunk, width, *, prefix_sc=False):
    workers = offsets.shape[0]

    def kernel(kh, rh, bh, oh, kv, rv, bv, hv, sem):
        worker = lax.axis_index(("count_core", "count_subcore"))
        copies = (
            pltpu.make_async_copy(kh.at[pl.ds(worker * chunk, chunk)], kv, sem),
            pltpu.make_async_copy(rh.at[pl.ds(worker * chunk, chunk)], rv, sem),
            pltpu.make_async_copy(bh, hv, sem)
            if prefix_sc
            else pltpu.make_async_copy(bh.at[worker, :], bv, sem),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

        if prefix_sc:
            carry = jnp.int32(0)
            for block in range(width // 16):

                def accumulate(w, state, block=block):
                    total, before = state
                    values = hv[pl.ds(w * width + block * 16, 16)]
                    return total + values, before + jnp.where(w < worker, values, 0)

                total, before = lax.fori_loop(
                    0,
                    workers,
                    accumulate,
                    (jnp.zeros((16,), jnp.int32), jnp.zeros((16,), jnp.int32)),
                )
                bv[pl.ds(block * 16, 16)] = before + sc.cumsum(total) - total + carry
                carry += jnp.sum(total)

        @sc.parallel_loop(0, chunk, step=16)
        def each(i):
            key = kv[pl.ds(i, 16)]
            rv[pl.ds(i, 16)] += sc.load_gather(bv, (key,))

        copy = pltpu.make_async_copy(rv, oh.at[pl.ds(worker * chunk, chunk)], sem)
        copy.start()
        copy.wait()

    return k.core_map_kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(keys.shape, jnp.int32),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((chunk,), jnp.int32),
            pltpu.VMEM((chunk,), jnp.int32),
            pltpu.VMEM((width,), jnp.int32),
            pltpu.VMEM((workers * width if prefix_sc else 16,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        name="sc_count_prefix_positions" if prefix_sc else "sc_count_finish_positions",
    )(keys, ranks, offsets.reshape(-1) if prefix_sc else offsets)

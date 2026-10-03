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
"""Reduce-scatter of a small bf16 block as one Pallas kernel, for the
data-parallel MoE combine.

Input per core: [cores * rows, hidden], block r holding its partial sums for
core r's rows. Output per core: [rows, hidden], the sum of its own block over
the cores, accumulated in float32 and rounded once. The whole input and one
[cores, rows, hidden] scratch sit in VMEM, so the caller bounds the rows.
"""

import math

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    SUBLANES,
    barrier,
    call,
    pad_cols,
    pad_rows,
    push,
    resolve_axis_and_cores,
    split_rank,
    sum_slots,
    wait_bytes_of,
)

# A barrier group distinct from the all-reduce kernel's.
COLLECTIVE_ID = 7


def reference(blocks, core):
    """[cores, cores * rows, hidden] -> core ``core``'s output."""
    rows = blocks.shape[1] // blocks.shape[0]
    mine = blocks[:, core * rows : (core + 1) * rows, :]
    return jnp.sum(mine.astype(jnp.float32), axis=0).astype(blocks.dtype)


def small_reduce_scatter(
    x,
    *,
    num_cores: int | None = None,
    axis: str | None = None,
    axis_name: str | None = None,
    rank=None,
    impl: str | None = None,
    form: str | None = None,
    allow_col_padding: bool = True,
    collective_id: int = COLLECTIVE_ID,
):
    """Block r of ``x`` summed onto core r of mesh axis ``axis`` /
    ``axis_name``, inside a shard_map over that axis; ``rank`` as in
    ``small_all_reduce``."""
    if x.ndim == 0:
        raise ValueError("small_reduce_scatter requires at least a 1D array")
    mesh_axis, n = resolve_axis_and_cores(axis, axis_name, num_cores)

    orig_shape = x.shape
    if x.ndim == 1:
        if orig_shape[0] % n:
            raise ValueError(f"{orig_shape[0]} elements do not split over {n} cores")
        x_2d = x.reshape(n, orig_shape[0] // n)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(orig_shape[0], math.prod(orig_shape[1:]))

    total, hidden = x_2d.shape
    if total <= 0 or hidden <= 0:
        raise ValueError(f"Expected positive dimensions, got {orig_shape}")
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    if hidden % LANES != 0 and not allow_col_padding:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")

    mode = impl or form or "oneshot"
    build = {
        "oneshot": _reduce_scatter,
        "hierarchical": _reduce_scatter_hierarchical,
    }.get(mode)
    if build is None:
        raise ValueError(
            f"Unknown reduce_scatter impl {mode!r}; expected 'oneshot' or 'hierarchical'"
        )

    rows = total // n
    x_3d = pad_cols(pad_rows(x_2d.reshape(n, rows, hidden), SUBLANES), LANES)
    _, padded_rows, padded_cols = x_3d.shape
    out_padded = build(
        x_3d.reshape(n * padded_rows, padded_cols), n, mesh_axis, rank, collective_id
    )
    out_2d = out_padded[:rows, :hidden]

    if x.ndim == 1:
        return out_2d.reshape(orig_shape[0] // n)
    if x.ndim == 2:
        return out_2d
    return out_2d.reshape(rows, *orig_shape[1:])


def _reduce_scatter(x, n, axis, rank, collective_id):
    total, hidden = x.shape
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    rows = total // n
    if rows % SUBLANES:
        raise ValueError(f"rows {rows} is not a multiple of {SUBLANES} sublanes")
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, slices, send_sem, recv_sem) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        # block r into slot `me` of core r's slices
        for r in range(n):

            @pl.when(jnp.int32(r) != me)
            def _(r=r):
                push(
                    x_ref.at[pl.ds(r * rows, rows)],
                    slices.at[me],
                    send_sem,
                    recv_sem,
                    r,
                    axis,
                    dict_ids,
                )

        # my own block: a static row slice picked by a scalar compare
        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                slices[pl.ds(me, 1)] = x_ref[pl.ds(r * rows, rows), :][None]

        if n > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(slices, n).astype(o_ref.dtype)
        if n > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel, x, rows, rank, scratch, collective_id, f"small_reduce_scatter_r{rows}"
    )


def _reduce_scatter_hierarchical(x, n, axis, rank, collective_id):
    if n < 2 or n % 2 != 0:
        return _reduce_scatter(x, n, axis, rank, collective_id)
    total, hidden = x.shape
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    rows = total // n
    if rows % SUBLANES:
        raise ValueError(f"rows {rows} is not a multiple of {SUBLANES} sublanes")
    num_pkgs = n // 2
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, stage, slices, ocn_send, ocn_recv, ici_send, ici_recv) = (
            split_rank(refs, dict_ids, axis)
        )
        barrier(n, axis, dict_ids)
        sib = jnp.bitwise_xor(me, jnp.int32(1))
        pkg = jnp.right_shift(me, jnp.int32(1))
        lane_id = jnp.bitwise_and(me, jnp.int32(1))

        # Phase 1: intra-package OCN exchange of cross-lane target blocks.
        for q in range(num_pkgs):
            for l in range(2):

                @pl.when(lane_id == jnp.int32(l))
                def _(q=q, l=l):
                    stage[pl.ds(2 * q + l, 1)] = x_ref[
                        pl.ds((2 * q + l) * rows, rows), :
                    ][None]
                    push(
                        x_ref.at[pl.ds((2 * q + (1 - l)) * rows, rows)],
                        stage.at[2 * q + l],
                        ocn_send,
                        ocn_recv,
                        sib,
                        axis,
                        dict_ids,
                    )

        wait_bytes_of(stage.at[pl.ds(0, num_pkgs)], ocn_recv)
        wait_bytes_of(stage.at[pl.ds(0, num_pkgs)], ocn_send)

        # Phase 2: inter-package coalesced 2-slot ICI push to target = 2*q + lane_id.
        for q in range(num_pkgs):

            @pl.when(jnp.int32(q) == pkg)
            def _(q=q):
                slices[pl.ds(2 * q, 2)] = stage[pl.ds(2 * q, 2)]

            @pl.when(jnp.int32(q) != pkg)
            def _(q=q):
                target = jnp.int32(2 * q) + lane_id
                push(
                    stage.at[pl.ds(2 * q, 2)],
                    slices.at[pl.ds(2 * pkg, 2)],
                    ici_send,
                    ici_recv,
                    target,
                    axis,
                    dict_ids,
                )

        if num_pkgs > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 2)], ici_recv)
        o_ref[...] = sum_slots(slices, n).astype(o_ref.dtype)
        if num_pkgs > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 2)], ici_send)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.VMEM((n, rows, hidden), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_reduce_scatter_hierarchical_r{rows}",
    )

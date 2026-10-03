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
"""All-gather of a small tensor as one Pallas kernel.

    oneshot        Every core pushes its local [rows, hidden] slice directly
                   into ``o_ref.at[pl.ds(me * rows, rows)]`` on every peer
                   ``r != me`` and copies locally into slot ``me``, using a
                   double-Wait DMA barrier with zero VMEM/HBM scratch buffer.
    hierarchical   Two-phase chiplet-coalesced OCN + ICI push for Ghostfish
                   (Cloud TPU v7x) dual-core packages: Phase 1 exchanges local
                   slices between sibling cores ``me`` and ``me ^ 1`` over the
                   600 GB/s intra-package OCN into contiguous rows
                   ``[2 * pkg * rows : (2 * pkg + 2) * rows]`` of ``o_ref``;
                   Phase 2 pushes the coalesced ``2 * rows`` package block
                   across ICI to the matching core lane on every remote package.
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
    wait_bytes_of,
)

# Barrier semaphore group distinct from all-reduce (5) and reduce-scatter (7).
COLLECTIVE_ID = 8


def reference(blocks, *, gather_dimension: int = 0, tiled: bool = True):
    """Reference all-gather output from per-core blocks [cores, ...]."""
    n = blocks.shape[0]
    ndim_item = blocks.ndim - 1
    if ndim_item <= 0:
        raise ValueError(f"blocks must have at least 2 dimensions, got {blocks.shape}")
    g_dim = gather_dimension % ndim_item
    moved = jnp.moveaxis(blocks, 0, g_dim)
    if not tiled:
        return moved
    new_shape = (
        *moved.shape[:g_dim],
        n * moved.shape[g_dim + 1],
        *moved.shape[g_dim + 2 :],
    )
    return moved.reshape(new_shape)


def _oneshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    if rows % SUBLANES:
        raise ValueError(f"rows {rows} is not a multiple of {SUBLANES} sublanes")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, send_sem, recv_sem) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)
        for r in range(n):

            @pl.when(jnp.int32(r) != me)
            def _(r=r):
                push(
                    x_ref,
                    o_ref.at[pl.ds(me * rows, rows)],
                    send_sem,
                    recv_sem,
                    r,
                    axis,
                    dict_ids,
                )

        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                o_ref[pl.ds(r * rows, rows), :] = x_ref[...]

        if n > 1:
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows)], recv_sem)
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows)], send_sem)

    scratch = [
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        n * rows,
        rank,
        scratch,
        collective_id,
        f"small_all_gather_oneshot_r{rows}",
    )


def _hierarchical(x, n, axis, rank, collective_id):
    if n < 2 or n % 2 != 0:
        return _oneshot(x, n, axis, rank, collective_id)
    rows, hidden = x.shape
    if rows % SUBLANES:
        raise ValueError(f"rows {rows} is not a multiple of {SUBLANES} sublanes")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    dict_ids = rank is not None
    num_pkgs = n // 2
    pkg_rows = 2 * rows

    def kernel(*refs):
        me, (x_ref, o_ref, ocn_send, ocn_recv, ici_send, ici_recv) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        sib = jnp.bitwise_xor(me, jnp.int32(1))
        pkg = jnp.right_shift(me, jnp.int32(1))
        lane_id = jnp.bitwise_and(me, jnp.int32(1))

        # Phase 1: intra-package OCN exchange between sibling cores (me and me ^ 1).
        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                o_ref[pl.ds(r * rows, rows), :] = x_ref[...]

        push(
            x_ref,
            o_ref.at[pl.ds(me * rows, rows)],
            ocn_send,
            ocn_recv,
            sib,
            axis,
            dict_ids,
        )
        wait_bytes_of(o_ref.at[pl.ds(0, rows)], ocn_recv)
        wait_bytes_of(o_ref.at[pl.ds(0, rows)], ocn_send)

        # Phase 2: inter-package coalesced 2*rows push over ICI to matching lane_id.
        for q in range(num_pkgs):

            @pl.when(jnp.int32(q) != pkg)
            def _(q=q):
                target = jnp.int32(q * 2) + lane_id
                push(
                    o_ref.at[pl.ds(pkg * pkg_rows, pkg_rows)],
                    o_ref.at[pl.ds(pkg * pkg_rows, pkg_rows)],
                    ici_send,
                    ici_recv,
                    target,
                    axis,
                    dict_ids,
                )

        if num_pkgs > 1:
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 2) * rows)], ici_recv)
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 2) * rows)], ici_send)

    scratch = [pltpu.SemaphoreType.DMA] * 4
    return call(
        kernel,
        x,
        n * rows,
        rank,
        scratch,
        collective_id,
        f"small_all_gather_hierarchical_r{rows}",
    )


def _all_gather_2d(
    x_2d,
    n: int,
    axis: str,
    rank,
    mode: str,
    allow_col_padding: bool,
    collective_id: int,
):
    rows, cols = x_2d.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Expected positive 2D shape, got {x_2d.shape}")
    if not allow_col_padding and cols % LANES != 0:
        raise ValueError(f"hidden {cols} is not a multiple of {LANES} lanes")

    x_padded = pad_cols(pad_rows(x_2d, SUBLANES), LANES)
    padded_rows, padded_cols = x_padded.shape

    build = {"oneshot": _oneshot, "hierarchical": _hierarchical}.get(mode)
    if build is None:
        raise ValueError(
            f"Unknown all_gather impl {mode!r}; expected 'oneshot' or 'hierarchical'"
        )
    out_padded = build(x_padded, n, axis, rank, collective_id)
    if padded_rows == rows and padded_cols == cols:
        return out_padded
    return out_padded.reshape(n, padded_rows, padded_cols)[:, :rows, :cols].reshape(
        n * rows, cols
    )


def small_all_gather(
    x,
    *,
    num_cores: int | None = None,
    axis: str | None = None,
    axis_name: str | None = None,
    gather_dimension: int = 0,
    gather_axis: int | None = None,
    rank=None,
    impl: str | None = None,
    form: str | None = None,
    tiled: bool = True,
    allow_col_padding: bool = True,
    collective_id: int = COLLECTIVE_ID,
):
    """Gather ``x`` across the ``num_cores`` cores of mesh axis ``axis`` /
    ``axis_name`` along ``gather_dimension`` as one Pallas kernel.

    Supports 1D, 2D, and ND tensors with automatic row padding to
    ``SUBLANES = 8`` and column padding to ``LANES = 128``.
    """
    if x.ndim == 0:
        raise ValueError("small_all_gather requires at least a 1D array")
    mesh_axis, n = resolve_axis_and_cores(axis, axis_name, num_cores)
    mode = impl or form or "oneshot"
    g_dim = (gather_axis if gather_axis is not None else gather_dimension) % x.ndim

    if x.ndim == 1:
        length = x.shape[0]
        out_2d = _all_gather_2d(
            x.reshape(1, length),
            n,
            mesh_axis,
            rank,
            mode,
            allow_col_padding,
            collective_id,
        )
        return out_2d.reshape(n * length) if tiled else out_2d.reshape(n, length)

    if x.ndim == 2 and g_dim == 0 and tiled:
        return _all_gather_2d(
            x, n, mesh_axis, rank, mode, allow_col_padding, collective_id
        )

    x_moved = jnp.moveaxis(x, g_dim, 0)
    rows = x_moved.shape[0]
    rest_dims = x_moved.shape[1:]
    cols = math.prod(rest_dims)
    out_2d = _all_gather_2d(
        x_moved.reshape(rows, cols),
        n,
        mesh_axis,
        rank,
        mode,
        allow_col_padding,
        collective_id,
    )
    if tiled:
        out_moved = out_2d.reshape(n * rows, *rest_dims)
        return jnp.moveaxis(out_moved, 0, g_dim)
    out_moved = out_2d.reshape(n, rows, *rest_dims)
    return jnp.moveaxis(out_moved, 0, g_dim)

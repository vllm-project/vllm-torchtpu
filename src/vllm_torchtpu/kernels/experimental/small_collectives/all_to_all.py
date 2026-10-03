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
"""All-to-all of a small tensor as one Pallas kernel.

    oneshot        Every core ``me`` pushes slice ``r`` of its local input
                   ``x_ref.at[pl.ds(r * rows_per_peer, rows_per_peer)]``
                   directly into slot ``me`` on peer ``r``
                   (``o_ref.at[pl.ds(me * rows_per_peer, rows_per_peer)]``)
                   and copies slice ``me`` locally, using a double-Wait DMA
                   barrier with zero VMEM/HBM scratch buffer.
    hierarchical   Two-phase OCN + ICI coalesced all-to-all for Ghostfish
                   dual-core packages: Phase 1 exchanges cross-lane chunks
                   between sibling cores ``me`` and ``me ^ 1`` over the
                   600 GB/s intra-package OCN; Phase 2 pushes a single
                   coalesced ``2 * rows_per_peer`` block per remote package
                   over ICI.
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

# Barrier semaphore group distinct from all-reduce (5), reduce-scatter (7), and all-gather (8).
COLLECTIVE_ID = 6


def reference(blocks, core: int | None = None, *, split_axis: int = 0, concat_axis: int = 0):
    """Reference all-to-all output from per-core blocks [cores, ...]."""
    n = blocks.shape[0]
    ndim_item = blocks.ndim - 1
    if ndim_item <= 0:
        raise ValueError(f"blocks must have at least 2 dimensions, got {blocks.shape}")
    s_ax = split_axis % ndim_item
    c_ax = concat_axis % ndim_item
    if blocks.shape[1 + s_ax] % n != 0:
        raise ValueError(
            f"split_axis {s_ax} of size {blocks.shape[1 + s_ax]} is not divisible by {n} cores"
        )
    chunk_s = blocks.shape[1 + s_ax] // n
    # Split axis (1 + s_ax) into (n_dst, chunk_s), swap (n_src=0, n_dst=1 + s_ax),
    # then merge n_src into concat_axis.
    split_shape = (
        n,
        *blocks.shape[1 : 1 + s_ax],
        n,
        chunk_s,
        *blocks.shape[2 + s_ax :],
    )
    reshaped = blocks.reshape(split_shape)
    # Move n_dst (at axis 1 + s_ax) to axis 0, and n_src (originally axis 0) to 1 + c_ax
    transposed = jnp.swapaxes(reshaped, 0, 1 + s_ax)
    # Now axis 0 is dst core, and axis (1 + s_ax) is src core (size n).
    # Move src core axis (1 + s_ax) right before (1 + c_ax) and flatten into c_ax.
    moved = jnp.moveaxis(transposed, 1 + s_ax, 1 + c_ax)
    out_shape = (
        n,
        *moved.shape[1 : 1 + c_ax],
        n * moved.shape[2 + c_ax],
        *moved.shape[3 + c_ax :],
    )
    out = moved.reshape(out_shape)
    return out if core is None else out[core]


def _oneshot(x, n, axis, rank, collective_id):
    total_rows, hidden = x.shape
    if total_rows % n:
        raise ValueError(f"{total_rows} rows do not split over {n} cores")
    rows_per_peer = total_rows // n
    if rows_per_peer % SUBLANES:
        raise ValueError(
            f"rows_per_peer {rows_per_peer} is not a multiple of {SUBLANES} sublanes"
        )
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
                    x_ref.at[pl.ds(r * rows_per_peer, rows_per_peer)],
                    o_ref.at[pl.ds(me * rows_per_peer, rows_per_peer)],
                    send_sem,
                    recv_sem,
                    r,
                    axis,
                    dict_ids,
                )

        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                o_ref[pl.ds(r * rows_per_peer, rows_per_peer), :] = x_ref[
                    pl.ds(r * rows_per_peer, rows_per_peer), :
                ]

        if n > 1:
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows_per_peer)], recv_sem)
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows_per_peer)], send_sem)

    scratch = [
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        total_rows,
        rank,
        scratch,
        collective_id,
        f"small_all_to_all_oneshot_r{rows_per_peer}",
    )


def _hierarchical(x, n, axis, rank, collective_id):
    if n < 2 or n % 2 != 0:
        return _oneshot(x, n, axis, rank, collective_id)
    total_rows, hidden = x.shape
    if total_rows % n:
        raise ValueError(f"{total_rows} rows do not split over {n} cores")
    rows = total_rows // n
    if rows % SUBLANES:
        raise ValueError(f"rows_per_peer {rows} is not a multiple of {SUBLANES} sublanes")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    num_pkgs = n // 2
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, stage, ocn_send, ocn_recv, ici_send, ici_recv) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        sib = jnp.bitwise_xor(me, jnp.int32(1))
        pkg = jnp.right_shift(me, jnp.int32(1))
        lane_id = jnp.bitwise_and(me, jnp.int32(1))

        # Phase 1: intra-package OCN exchange of cross-lane chunks between me and me ^ 1.
        for q in range(num_pkgs):
            for l in range(2):

                @pl.when(lane_id == jnp.int32(l))
                def _(q=q, l=l):
                    stage[pl.ds((2 * q + l) * rows, rows), :] = x_ref[
                        pl.ds((2 * q + l) * rows, rows), :
                    ]
                    push(
                        x_ref.at[pl.ds((2 * q + (1 - l)) * rows, rows)],
                        stage.at[pl.ds((2 * q + l) * rows, rows)],
                        ocn_send,
                        ocn_recv,
                        sib,
                        axis,
                        dict_ids,
                    )

        wait_bytes_of(stage.at[pl.ds(0, num_pkgs * rows)], ocn_recv)
        wait_bytes_of(stage.at[pl.ds(0, num_pkgs * rows)], ocn_send)

        # Phase 2: inter-package coalesced 2*rows push over ICI to target = 2*q + lane_id.
        for q in range(num_pkgs):

            @pl.when(jnp.int32(q) == pkg)
            def _(q=q):
                o_ref[pl.ds((2 * q) * rows, 2 * rows), :] = stage[
                    pl.ds((2 * q) * rows, 2 * rows), :
                ]

            @pl.when(jnp.int32(q) != pkg)
            def _(q=q):
                target = jnp.int32(2 * q) + lane_id
                push(
                    stage.at[pl.ds(2 * q * rows, 2 * rows)],
                    o_ref.at[pl.ds(pkg * 2 * rows, 2 * rows)],
                    ici_send,
                    ici_recv,
                    target,
                    axis,
                    dict_ids,
                )

        if num_pkgs > 1:
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 2) * rows)], ici_recv)
            wait_bytes_of(o_ref.at[pl.ds(0, (n - 2) * rows)], ici_send)

    scratch = [
        pltpu.VMEM((total_rows, hidden), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        total_rows,
        rank,
        scratch,
        collective_id,
        f"small_all_to_all_hierarchical_r{rows}",
    )


def _all_to_all_chunks(
    x_3d,
    n: int,
    axis: str,
    rank,
    mode: str,
    allow_col_padding: bool,
    collective_id: int,
):
    """Run all-to-all on [n, rows_per_peer, cols] and return [n, rows_per_peer, cols]."""
    _, rows_per_peer, cols = x_3d.shape
    if rows_per_peer <= 0 or cols <= 0:
        raise ValueError(f"Expected positive chunk dimensions, got {x_3d.shape}")
    if not allow_col_padding and cols % LANES != 0:
        raise ValueError(f"hidden {cols} is not a multiple of {LANES} lanes")

    x_padded_3d = pad_cols(pad_rows(x_3d, SUBLANES), LANES)
    _, padded_rows, padded_cols = x_padded_3d.shape
    x_padded_2d = x_padded_3d.reshape(n * padded_rows, padded_cols)

    build = {"oneshot": _oneshot, "hierarchical": _hierarchical}.get(mode)
    if build is None:
        raise ValueError(
            f"Unknown all_to_all impl {mode!r}; expected 'oneshot' or 'hierarchical'"
        )
    out_padded_2d = build(x_padded_2d, n, axis, rank, collective_id)
    out_padded_3d = out_padded_2d.reshape(n, padded_rows, padded_cols)
    if padded_rows == rows_per_peer and padded_cols == cols:
        return out_padded_3d
    return out_padded_3d[:, :rows_per_peer, :cols]


def small_all_to_all(
    x,
    *,
    axis_name: str | None = None,
    axis: str | None = None,
    num_cores: int | None = None,
    split_axis: int = 0,
    concat_axis: int = 0,
    rank=None,
    impl: str = "oneshot",
    form: str | None = None,
    allow_col_padding: bool = True,
    collective_id: int = COLLECTIVE_ID,
):
    """All-to-all of ``x`` across ``num_cores`` cores of mesh axis ``axis_name`` /
    ``axis``, splitting ``x`` along ``split_axis`` and concatenating received
    chunks along ``concat_axis``.

    Supports 1D, 2D, and ND tensors with automatic row padding to
    ``SUBLANES = 8`` and column padding to ``LANES = 128``.
    """
    if x.ndim == 0:
        raise ValueError("small_all_to_all requires at least a 1D array")
    mesh_axis, n = resolve_axis_and_cores(axis, axis_name, num_cores)
    mode = form if form is not None else impl
    s_ax = split_axis % x.ndim
    c_ax = concat_axis % x.ndim

    if mode == "xla" or n <= 2:
        import jax.lax

        return jax.lax.all_to_all(
            x,
            mesh_axis,
            split_axis=s_ax,
            concat_axis=c_ax,
            tiled=True,
        )

    if x.shape[s_ax] % n != 0:
        raise ValueError(
            f"Dimension {x.shape[s_ax]} along split_axis={s_ax} does not split over {n} cores"
        )
    rows_per_peer = x.shape[s_ax] // n

    if x.ndim == 1:
        x_3d = x.reshape(n, 1, rows_per_peer)
        out_3d = _all_to_all_chunks(
            x_3d, n, mesh_axis, rank, mode, allow_col_padding, collective_id
        )
        return out_3d.reshape(n * rows_per_peer)

    if x.ndim == 2 and s_ax == 0 and c_ax == 0:
        cols = x.shape[1]
        out_3d = _all_to_all_chunks(
            x.reshape(n, rows_per_peer, cols),
            n,
            mesh_axis,
            rank,
            mode,
            allow_col_padding,
            collective_id,
        )
        return out_3d.reshape(n * rows_per_peer, cols)

    x_moved = jnp.moveaxis(x, s_ax, 0)
    rest_dims = x_moved.shape[1:]
    cols = math.prod(rest_dims)
    out_3d = _all_to_all_chunks(
        x_moved.reshape(n, rows_per_peer, cols),
        n,
        mesh_axis,
        rank,
        mode,
        allow_col_padding,
        collective_id,
    )
    # Restore rest_dims: [n_src, rows_per_peer, *rest_dims]
    restored = out_3d.reshape(n, rows_per_peer, *rest_dims)
    # Move rows_per_peer (axis 1) back to its original position (1 + s_ax)
    unsplit = jnp.moveaxis(restored, 1, 1 + s_ax)
    # Move n_src (axis 0) right before concat_axis (c_ax) and merge them
    concat_ready = jnp.moveaxis(unsplit, 0, c_ax)
    final_shape = (
        *concat_ready.shape[:c_ax],
        n * concat_ready.shape[c_ax + 1],
        *concat_ready.shape[c_ax + 2 :],
    )
    return concat_ready.reshape(final_shape)

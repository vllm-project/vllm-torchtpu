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
"""All-reduce of a small bf16 [rows, hidden] block as one Pallas kernel.

    oneshot        Every core pushes its whole block to every other core and sums
                   the arrivals locally. Only requires padded width to be a
                   multiple of ``LANES = 128`` (automatically padded/unpadded),
                   unlocking ``H = 6144`` on 32 cores (2x2x4) and 1 KiB..64 KiB
                   small payloads.
    twoshot        Core j owns columns [j * own, (j + 1) * own). Every core
                   pushes each owner its columns; the owner sums them and pushes
                   the sum back to everyone. Automatically pads columns to a
                   multiple of ``num_cores * LANES`` and slices back to ``hidden``.
    hierarchical   Two-phase OCN intra-package + ICI inter-package all-reduce
                   for Ghostfish dual-core packages: Phase 1 exchanges blocks
                   between sibling cores ``me`` and ``me ^ 1`` over the 600 GB/s
                   intra-package OCN; Phase 2 pushes coalesced 2-core package
                   blocks across ICI to matching lanes on remote packages, then
                   sums all ``num_cores`` slots in float32 and rounds once.

All forms start with a barrier so remote scratch is ready, sum in float32 and
round once to the input dtype.
"""

import math

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    call,
    pad_cols,
    push,
    resolve_axis_and_cores,
    split_rank,
    sum_slots,
    wait_bytes_of,
)

# One-shot up to this many bytes per core, two-shot above: one-shot sends more
# bytes in one round, two-shot fewer in two; they cross at ~5.9 rows of H=4096
# bf16 on 8 cores (j-de6349e8).
ONE_SHOT_MAX_BYTES = 32 * 1024
# Barrier semaphore group. Every kernel compiled with the same id in one
# program shares it; the fused EP MoE kernels use 0.
COLLECTIVE_ID = 5


def form_for(rows, hidden, itemsize):
    """The form for a block of rows x hidden elements of itemsize bytes."""
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def reference(blocks):
    """The float32 sum over the cores' blocks [cores, rows, hidden], rounded
    once to the blocks' dtype."""
    return jnp.sum(blocks.astype(jnp.float32), axis=0).astype(blocks.dtype)


def _oneshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem, recv_sem) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        for j in range(n):  # my block into slot `me` of every other core

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(x_ref, arrivals.at[me], send_sem, recv_sem, j, axis, dict_ids)

        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        if n > 1:
            wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(arrivals, n).astype(o_ref.dtype)
        if n > 1:
            wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_oneshot_r{rows}",
    )


def _twoshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    if hidden % (n * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {n} cores x {LANES} lanes"
        )
    own = hidden // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, refs = split_rank(refs, dict_ids, axis)
        x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2 = refs
        barrier(n, axis, dict_ids)
        for j in range(n):  # slot j: the columns core j owns
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        # round 1: owner j's columns into slot `me` of its slices
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(mine.at[j], slices.at[me], send1, recv1, j, axis, dict_ids)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        if n > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        gathered[pl.ds(me, 1)] = sum_slots(slices, n).astype(gathered.dtype)[None]
        # round 2: my summed columns into slot `me` of every other core
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(gathered.at[me], gathered.at[me], send2, recv2, j, axis, dict_ids)

        if n > 1:
            wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        if n > 1:
            wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
            wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_twoshot_r{rows}",
    )


def _hierarchical(x, n, axis, rank, collective_id):
    if n < 2 or n % 2 != 0:
        return _oneshot(x, n, axis, rank, collective_id)
    rows, hidden = x.shape
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    dict_ids = rank is not None
    num_pkgs = n // 2

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, ocn_send, ocn_recv, ici_send, ici_recv) = (
            split_rank(refs, dict_ids, axis)
        )
        barrier(n, axis, dict_ids)
        sib = jnp.bitwise_xor(me, jnp.int32(1))
        pkg = jnp.right_shift(me, jnp.int32(1))
        lane_id = jnp.bitwise_and(me, jnp.int32(1))

        # Phase 1: intra-package OCN exchange between sibling cores (me and me ^ 1).
        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        push(x_ref, arrivals.at[me], ocn_send, ocn_recv, sib, axis, dict_ids)
        wait_bytes_of(arrivals.at[pl.ds(0, 1)], ocn_recv)
        wait_bytes_of(arrivals.at[pl.ds(0, 1)], ocn_send)

        # Phase 2: inter-package coalesced 2-slot ICI push to matching lane_id.
        for q in range(num_pkgs):

            @pl.when(jnp.int32(q) != pkg)
            def _(q=q):
                target = jnp.int32(q * 2) + lane_id
                push(
                    arrivals.at[pl.ds(pkg * 2, 2)],
                    arrivals.at[pl.ds(pkg * 2, 2)],
                    ici_send,
                    ici_recv,
                    target,
                    axis,
                    dict_ids,
                )

        if num_pkgs > 1:
            wait_bytes_of(arrivals.at[pl.ds(0, n - 2)], ici_recv)
        o_ref[...] = sum_slots(arrivals, n).astype(o_ref.dtype)
        if num_pkgs > 1:
            wait_bytes_of(arrivals.at[pl.ds(0, n - 2)], ici_send)

    scratch = [
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
        f"small_all_reduce_hierarchical_r{rows}",
    )


def small_all_reduce(
    x,
    *,
    num_cores: int | None = None,
    axis: str | None = None,
    axis_name: str | None = None,
    rank=None,
    form: str | None = None,
    impl: str | None = None,
    allow_col_padding: bool = True,
    collective_id: int = COLLECTIVE_ID,
):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis`` /
    ``axis_name``, inside a shard_map over that axis.

    Args:
      x: [rows, hidden] (or arbitrary 1D/ND tensor), this core's block.
      rank: int32 [1], this core's index on the axis as data (the torch
        bridge), or None to read it from the axis.
      form / impl: "oneshot", "twoshot", or "hierarchical"; by default
        ``form_for`` the block.
      allow_col_padding: When True (default), automatically pads columns to a
        multiple of ``LANES`` (for ``oneshot`` / ``hierarchical``) or
        ``num_cores * LANES`` (for ``twoshot``) and slices ``[:hidden]`` on
        return.
    """
    if x.ndim == 0:
        raise ValueError("small_all_reduce requires at least a 1D array")
    mesh_axis, n = resolve_axis_and_cores(axis, axis_name, num_cores)

    orig_shape = x.shape
    if x.ndim == 1:
        x_2d = x.reshape(1, x.shape[0])
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(x.shape[0], math.prod(x.shape[1:]))

    rows, hidden = x_2d.shape
    if rows <= 0 or hidden <= 0:
        raise ValueError(f"Expected positive dimensions, got {orig_shape}")

    mode = impl or form or form_for(rows, hidden, jnp.dtype(x.dtype).itemsize)
    build = {
        "oneshot": _oneshot,
        "twoshot": _twoshot,
        "hierarchical": _hierarchical,
    }.get(mode)
    if build is None:
        raise ValueError(
            f"Unknown all_reduce form/impl {mode!r}; expected 'oneshot', 'twoshot', or 'hierarchical'"
        )

    col_multiple = (n * LANES) if mode == "twoshot" else LANES
    if hidden % col_multiple != 0:
        if not allow_col_padding:
            raise ValueError(
                f"hidden {hidden} is not a multiple of {col_multiple} lanes"
            )
        x_padded = pad_cols(x_2d, col_multiple)
        out_2d = build(x_padded, n, mesh_axis, rank, collective_id)[:, :hidden]
    else:
        out_2d = build(x_2d, n, mesh_axis, rank, collective_id)

    return out_2d if x.ndim == 2 else out_2d.reshape(orig_shape)

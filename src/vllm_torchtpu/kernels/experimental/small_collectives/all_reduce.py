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
"""One-shot and two-shot Pallas all-reduce for small `[rows, hidden]` blocks."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    call,
    push,
    split_rank,
    sum_slots,
    wait_bytes_of,
)

ONE_SHOT_MAX_BYTES = 32 * 1024
COLLECTIVE_ID = 5


def form_for(rows, hidden, itemsize):
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def reference(blocks):
    return jnp.sum(blocks.astype(jnp.float32), axis=0).astype(blocks.dtype)


def _push_peers(n, me, src_fn, dst_fn, send_sem, recv_sem, axis, dict_ids):
    for j in range(n):
        pl.when(jnp.int32(j) != me)(
            lambda j=j: push(
                src_fn(j), dst_fn(j), send_sem, recv_sem, j, axis, dict_ids
            )
        )


def _twoshot_scratch(n, rows, own, dtype):
    return [*[pltpu.VMEM((n, rows, own), dtype)] * 3, *[pltpu.SemaphoreType.DMA] * 4]


def _oneshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem, recv_sem) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        _push_peers(
            n,
            me,
            lambda _j: x_ref,
            lambda _j: arrivals.at[me],
            send_sem,
            recv_sem,
            axis,
            dict_ids,
        )
        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        peers = arrivals.at[pl.ds(0, n - 1)]
        wait_bytes_of(peers, recv_sem)
        o_ref[...] = sum_slots(arrivals, n).astype(o_ref.dtype)
        wait_bytes_of(peers, send_sem)

    scratch = [pltpu.VMEM((n, rows, hidden), x.dtype), *[pltpu.SemaphoreType.DMA] * 2]
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
    own = hidden // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, refs = split_rank(refs, dict_ids, axis)
        x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2 = refs
        barrier(n, axis, dict_ids)
        for j in range(n):
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        _push_peers(
            n,
            me,
            lambda j: mine.at[j],
            lambda _j: slices.at[me],
            send1,
            recv1,
            axis,
            dict_ids,
        )
        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        gathered[pl.ds(me, 1)] = sum_slots(slices, n).astype(gathered.dtype)[None]
        _push_peers(
            n,
            me,
            lambda _j: gathered.at[me],
            lambda _j: gathered.at[me],
            send2,
            recv2,
            axis,
            dict_ids,
        )
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = _twoshot_scratch(n, rows, own, x.dtype)
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_twoshot_r{rows}",
    )


def fold_geometry(rows, hidden, num_cores, tile_rows):
    own = hidden // num_cores
    if hidden % (num_cores * LANES) or not rows > tile_rows // 2:
        return None
    k = own // LANES
    fold_rows = tile_rows // k
    return (k, fold_rows) if k >= 2 and fold_rows >= 1 else None


def _twoshot_valid_rows(x, n, axis, tile_rows):
    rows, hidden = x.shape
    own = hidden // n
    geometry = fold_geometry(rows, hidden, n, tile_rows)
    assert geometry is not None, "callers check fold_geometry first"
    k, fr = geometry
    hk = hidden // k

    def exchange(rows_x, width, me, mine, slices, gathered, sems, write_out):
        send1, recv1, send2, recv2 = sems

        def slot(buf, j):
            return buf.at[j, pl.ds(0, rows_x), pl.ds(0, width)]

        def slots(buf):
            return buf.at[pl.ds(0, n - 1), pl.ds(0, rows_x), pl.ds(0, width)]

        _push_peers(
            n,
            me,
            lambda j: slot(mine, j),
            lambda _j: slot(slices, me),
            send1,
            recv1,
            axis,
            True,
        )
        slices[pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)] = mine[
            pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)
        ]
        wait_bytes_of(slots(slices), recv1)
        acc = slices[0, pl.ds(0, rows_x), pl.ds(0, width)].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j, pl.ds(0, rows_x), pl.ds(0, width)].astype(jnp.float32)
        gathered[pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)] = acc.astype(
            gathered.dtype
        )[None]
        _push_peers(
            n,
            me,
            lambda _j: slot(gathered, me),
            lambda _j: slot(gathered, me),
            send2,
            recv2,
            axis,
            True,
        )
        wait_bytes_of(slots(gathered), recv2)
        write_out()
        wait_bytes_of(slots(slices), send1)
        wait_bytes_of(slots(gathered), send2)

    def kernel(n_valid_ref, rank_ref, x_ref, o_ref, mine, slices, gathered, *sems):
        n_valid = n_valid_ref[0]
        me = rank_ref[0]
        barrier(n, axis, True)

        @pl.when(n_valid <= fr)
        def _fold():
            row = lax.broadcasted_iota(jnp.int32, (tile_rows, hk), 0)
            p = x_ref[pl.ds(0, tile_rows), :].astype(jnp.float32)
            valid_mask = lax.broadcasted_iota(jnp.int32, p.shape, 0) < n_valid
            p = jnp.where(valid_mask, p, 0.0)
            folded = jnp.zeros((tile_rows, hk), jnp.float32)
            for j in range(k):
                blk = p[:, j * hk : (j + 1) * hk]
                if j:
                    blk = pltpu.roll(blk, j * fr, 0)
                in_band = (row >= j * fr) & (row < (j + 1) * fr)
                folded = jnp.where(in_band, blk, folded)
            folded = folded.astype(x.dtype)
            for j in range(n):
                mine[j, pl.ds(0, tile_rows), pl.ds(0, LANES)] = folded[
                    :, j * LANES : (j + 1) * LANES
                ]

            def write_out():
                summed = jnp.concatenate(
                    [
                        gathered[j, pl.ds(0, tile_rows), pl.ds(0, LANES)]
                        for j in range(n)
                    ],
                    axis=1,
                ).astype(jnp.float32)
                for j in range(k):
                    blk = (
                        summed if j == 0 else pltpu.roll(summed, tile_rows - j * fr, 0)
                    )
                    o_ref[pl.ds(0, tile_rows), pl.ds(j * hk, hk)] = jnp.where(
                        row < n_valid, blk, 0.0
                    ).astype(o_ref.dtype)
                if rows > tile_rows:
                    o_ref[pl.ds(tile_rows, rows - tile_rows), :] = jnp.zeros(
                        (rows - tile_rows, hidden), o_ref.dtype
                    )

            exchange(tile_rows, LANES, me, mine, slices, gathered, sems, write_out)

        @pl.when(n_valid > fr)
        def _full():
            for j in range(n):
                mine[j] = x_ref[:, pl.ds(j * own, own)]

            def write_out():
                for j in range(n):
                    o_ref[:, pl.ds(j * own, own)] = gathered[j]

            exchange(rows, own, me, mine, slices, gathered, sems, write_out)

    scratch = _twoshot_scratch(n, rows, own, x.dtype)
    return scratch, kernel, f"small_all_reduce_twoshot_valid_r{rows}"


def small_all_reduce_valid_rows(
    x, n_valid, rank, *, num_cores, axis, tile_rows, collective_id=COLLECTIVE_ID
):
    rows, hidden = x.shape
    if rows < tile_rows or fold_geometry(rows, hidden, num_cores, tile_rows) is None:
        raise ValueError(
            f"[{rows}, {hidden}] over {num_cores} cores has no fold "
            f"(tile_rows={tile_rows})"
        )
    scratch, kernel, name = _twoshot_valid_rows(x, num_cores, axis, tile_rows)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[smem, smem, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(n_valid.astype(jnp.int32), rank.astype(jnp.int32), x)


def small_all_reduce(
    x, *, num_cores, axis, rank=None, form=None, collective_id=COLLECTIVE_ID
):
    """Sum `x` (`[rows, hidden]`) over `num_cores` cores of mesh axis `axis`."""
    rows, hidden = x.shape
    if hidden % (num_cores * LANES):
        raise ValueError(
            f"hidden {hidden} not a multiple of {num_cores} x {LANES} lanes"
        )
    form = form or form_for(rows, hidden, jnp.dtype(x.dtype).itemsize)
    build = {"oneshot": _oneshot, "twoshot": _twoshot}[form]
    return build(x, num_cores, axis, rank, collective_id)

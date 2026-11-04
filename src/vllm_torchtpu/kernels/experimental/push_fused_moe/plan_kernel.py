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
"""The dispatch and shard tables of the push layer as one TensorCore kernel.

Every table below is a function of the exchanged [ep, e_total] pair counts,
this shard's expert-sorted segment starts and its rank. In XLA they are
some fifty small fusions (host.push_dispatch_tables, the plan's run tables
and v2's shard_* tables) whose fixed cost per op dominates; here they are
one kernel over [ep, 128]-lane tiles (one sublane row per destination or
source, local experts on lanes). Same integers as the XLA tables.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.push_fused_moe import tc_exchange as tx
from vllm_torchtpu.kernels.fused_moe.v2.host import ROWBLK, ROWBLK_SHIFT, pow2_shift

LANES = 128


def _lane_excl_cumsum(x):
    """Exclusive prefix sum along the lanes of an int32 [rows, 128] tile."""
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    inc = x
    k = 1
    while k < x.shape[1]:
        inc = inc + jnp.where(lane >= k, pltpu.roll(inc, k, 1), 0)
        k *= 2
    return inc - x


def _pad_rows(x, rows):
    return jnp.concatenate([x, jnp.zeros((rows - x.shape[0], x.shape[1]), x.dtype)])


def _transpose(x):
    """[r, 128] -> [128, 128] transposed (rows past r are zero)."""
    return _pad_rows(x, LANES).T


def _sel(cond, x):
    return jnp.where(cond, x, 0)


def _tables(r3, seg3, me, *, ep, g, n_pre, tile_m):
    """All tables from r3 [ep (s), ep (d), 128] (rows_by_dest[s, d*g+e] on
    lane e, zero past g), seg3 [ep (d), 128] (segment starts, same lanes)
    and the scalar rank `me`; see build_push_tables for the outputs."""
    i32 = jnp.int32
    lane = lax.broadcasted_iota(i32, (ep, LANES), 1)
    row = lax.broadcasted_iota(i32, (ep, LANES), 0)
    real = lane < g
    ar3 = (r3 + (ROWBLK - 1)) & i32(-ROWBLK)
    ea = functools.reduce(lambda a, b: a + b, [ar3[s] for s in range(ep)])  # [d, e]
    zero = jnp.zeros((ep, LANES), i32)
    mine = zero  # [d, e] my pairs per expert: r3[me]
    run_me = zero  # [d, e] sum_{s < me} ar3[s]
    ar_src_me = zero  # [d, e] ar3[me]
    for s in range(ep):
        mine = mine + _sel(me == s, r3[s])
        run_me = run_me + _sel(s < me, ar3[s])
        ar_src_me = ar_src_me + _sel(me == s, ar3[s])
    # [s, e] views of the column d == me.
    r_dme = zero
    ar_dme = zero
    for d in range(ep):
        r_dme = r_dme + _sel(
            row == d, jnp.sum(_sel(row == me, r3[d]), 0, keepdims=True)
        )
        ar_dme = ar_dme + _sel(
            row == d, jnp.sum(_sel(row == me, ar3[d]), 0, keepdims=True)
        )
    local_base = _lane_excl_cumsum(ea)

    # Visit lists: experts with rows, most rows first, ties by index
    # (v2's expert_visit_list key, ranked by compare).
    key = jnp.where(real, lane - ea * g, i32(2**30))
    rank = zero
    for e in range(g):
        rank = rank + (key[:, e : e + 1] < key).astype(i32)
    n_visit = jnp.sum((real & (ea > 0)).astype(i32), 1, keepdims=True)  # [d, 1]
    visit = zero
    for v in range(g):
        col = jnp.sum(_sel(real & (rank == v), lane), 1, keepdims=True)
        visit = jnp.where(lane == v, jnp.minimum(col, g - 1), visit)
    live = real & (lane < n_visit)

    def by_visit(tbl):
        out = zero
        for v in range(g):
            hit = (lane == visit[:, v : v + 1]) & real
            col = jnp.sum(_sel(hit, tbl), 1, keepdims=True)
            out = jnp.where(lane == v, col, out)
        return _sel(live, out)  # [d, v]

    cnt = by_visit(mine)
    src_off = by_visit(seg3)
    dst_row = by_visit(local_base + run_me)
    seg_a = (row << 24) | (visit << 16) | cnt
    seg_a = jnp.where(real, seg_a, 0)
    seg_c = _sel(cnt > 0, dst_row)
    recv = jnp.sum(r_dme, 0, keepdims=True)  # [1, e]
    remote = _sel(row != me, cnt)
    my_visit = jnp.sum(_sel(row == me, visit), 0, keepdims=True)  # [1, v]
    me_live = jnp.sum(_sel(row == me, live.astype(i32)), 0, keepdims=True) > 0
    lane1 = lane[:1]
    rows_at_v = jnp.zeros((1, LANES), i32)
    for v in range(g):
        hit = lane1 == my_visit[:, v : v + 1]
        val = jnp.sum(_sel(hit, recv), 1, keepdims=True)
        rows_at_v = jnp.where(lane1 == v, val, rows_at_v)
    rows_at_v = _sel(me_live, rows_at_v)
    remote_v = jnp.sum(remote, 0, keepdims=True)  # [1, v]
    pre = lane1 < n_pre
    odd = (lane1 & 1) == 1
    real1 = lane1 < g
    sc = [
        jnp.sum(_sel(pre & ~odd & real1, rows_at_v)),
        jnp.sum(_sel(pre & odd & real1, rows_at_v)),
        jnp.sum(_sel(pre & ~odd & real1, remote_v)),
        jnp.sum(_sel(pre & odd & real1, remote_v)),
    ]
    n_remote = jnp.sum(_sel((lane1 >= n_pre) & real1, remote_v))
    n_sent = jnp.sum(cnt)
    vec = jnp.zeros((1, LANES), i32)
    for j, val in enumerate(sc):
        vec = jnp.where(lane1 == j, val, vec)
    counts_sc = vec
    # meta = [0 x (g + 1), recv (g), n_remote, n_sent], laid out as in
    # host.push_dispatch_segments.
    recv_shift = pltpu.roll(recv, g + 1, 1)
    meta = jnp.where((lane1 > g) & (lane1 <= 2 * g), recv_shift, 0)
    meta = jnp.where(lane1 == 2 * g + 1, n_remote, meta)
    meta = jnp.where(lane1 == 2 * g + 2, n_sent, meta)

    # Shard tables, [e, d] after the transposes.
    true_rows_t = _transpose(r_dme)  # [e, s] = r3[s, me, e]
    aligned_t = _transpose(ar_dme)
    run_start_t = _lane_excl_cumsum(aligned_t)
    out_total = jnp.sum(aligned_t, 0, keepdims=True)  # [1, d]
    contrib_base = _lane_excl_cumsum(out_total)
    contrib_off_t = _transpose(_lane_excl_cumsum(ar_dme)) + contrib_base
    s1 = jnp.zeros((ep, 1), i32)  # [x, 1] rows source x sends to shards < me
    for d in range(ep):
        s1 = s1 + _sel(d < me, jnp.sum(ar3[:, d, :], 1, keepdims=True))
    push_dst_t = _transpose(s1 + _lane_excl_cumsum(ar_dme))
    expert_rows = jnp.sum(_sel(row == me, ea), 0, keepdims=True)
    slab_base = jnp.sum(_sel(row == me, local_base), 0, keepdims=True)

    shift = pow2_shift(tile_m, "tile_m")

    def tiles_of(start, rows):
        last = start + jnp.maximum(rows, 1) - 1
        return _sel(rows > 0, (last >> shift) - (start >> shift) + 1)

    other = row[:, :1] != me  # [d, 1]
    is_me = row[:, :1] == me
    # [x, e]: the mirror tiles of run (me's expert e, source x), whose start
    # is the sum of the rows above x in ar_dme.
    tiles_d = zero
    acc = jnp.zeros((1, LANES), i32)
    for x in range(ep):
        rows_x = ar_dme[x : x + 1]
        tiles_d = jnp.where(row == x, tiles_of(acc, rows_x), tiles_d)
        acc = acc + rows_x
    tot_r3_me = jnp.sum(mine)  # rows this shard sends in total
    tot_ar3_me = jnp.sum(ar_src_me)
    tiles_all_me = jnp.sum(_sel(real, tiles_of(run_me, ar_src_me)))
    sum_e = lambda t: jnp.sum(_sel(real, t), 1, keepdims=True)  # noqa: E731
    cols = [
        _sel(other, sum_e(r_dme)),
        _sel(is_me, tot_r3_me - sum_e(r_dme)),
        _sel(other, sum_e(ar_dme)),
        _sel(is_me, tot_ar3_me - sum_e(ar_dme)),
        _sel(is_me, jnp.sum((real1 & (expert_rows > 0)).astype(i32))),
        _sel(other, sum_e(tiles_d)),
        _sel(is_me, tiles_all_me - sum_e(tiles_d)),
        _sel(is_me, sum_e(tiles_d)),
    ]
    counts = zero
    for j, col in enumerate(cols):
        counts = jnp.where(lane == j, col, counts)
    counts_t = _transpose(counts)  # [count, d]

    return dict(
        seg=(_transpose(seg_a), _transpose(src_off), _transpose(seg_c)),
        small=(meta, my_visit, counts_sc, expert_rows, slab_base),
        kernel_tables=(
            run_start_t >> ROWBLK_SHIFT,
            aligned_t >> ROWBLK_SHIFT,
            contrib_off_t >> ROWBLK_SHIFT,
            push_dst_t >> ROWBLK_SHIFT,
            true_rows_t,
        ),
        counts=counts_t,
    )


@functools.cache
def _build(ep, g, n_pre, tile_m, interpret):
    def kernel(me_sm, r3_ref, seg3_ref, *outs):
        t = _tables(r3_ref[...], seg3_ref[...], me_sm[0], ep=ep, g=g, n_pre=n_pre,
                    tile_m=tile_m)  # fmt: skip
        _store_tables(t, outs, ep=ep, g=g)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem],
            out_specs=[vmem] * 8,
            grid=(),
        ),
        out_shape=_table_out_shapes(ep, g),
        interpret=interpret,
        name=f"moe_push_tables_ep{ep}_g{g}",
    )


def build_push_tables(rows_by_dest, seg_start_local, me, *, ep, n_pre, tile_m,
                      interpret=False):  # fmt: skip
    """The push layer's dispatch and shard tables in one kernel.

    Returns (seg, dmeta, visit, counts_sc, kernel_tables, expert_rows,
    slab_base, counts) equal to host.push_dispatch_tables(..., n_pre)'s
    (seg, meta, my_visit, counts_sc) and, for the same routing, v2's
    shard_transport_tables_in_blocks [:3] + [5], shard_push_tables_in_rows
    [0], shard_expert_slabs and shard_count_vector.
    """
    e_total = rows_by_dest.shape[1]
    g = e_total // ep
    if g > LANES or ep > LANES or 2 * g + 3 > LANES:
        raise ValueError(f"the table kernel takes g_local, ep <= {LANES}")
    r3 = jnp.pad(
        rows_by_dest.astype(jnp.int32).reshape(ep, ep, g),
        ((0, 0), (0, 0), (0, LANES - g)),
    )
    me_arr = jnp.asarray(me, jnp.int32).reshape(1)
    seg, small, *kt, counts = _build(ep, g, int(n_pre), int(tile_m), bool(interpret))(
        me_arr, r3, _seg3(seg_start_local, ep, g)
    )
    return _unpack_tables(seg, small, kt, counts, g)


def _store_tables(t, outs, *, ep, g):
    seg_o, small_o, kt_o, counts_o = outs[0], outs[1], outs[2:7], outs[7]
    for k, part in enumerate(t["seg"]):
        seg_o[k] = part[:g, :ep]
    for k, part in enumerate(t["small"]):
        small_o[k : k + 1, :] = part
    for o, part in zip(kt_o, t["kernel_tables"]):
        o[...] = part[:g, :ep]
    counts_o[...] = t["counts"][:8, :ep]


def _table_out_shapes(ep, g):
    i32 = jnp.int32
    return (
        jax.ShapeDtypeStruct((3, g, ep), i32),
        jax.ShapeDtypeStruct((8, LANES), i32),
        *(jax.ShapeDtypeStruct((g, ep), i32) for _ in range(5)),
        jax.ShapeDtypeStruct((8, ep), i32),
    )


def _unpack_tables(seg, small, kt, counts, g):
    return (
        seg.reshape(-1),
        small[0, : 2 * g + 3],
        small[1, :g],
        small[2, :8],
        tuple(kt),
        small[3, :g],
        small[4, :g],
        counts,
    )


def _seg3(seg_start_local, ep, g):
    return jnp.pad(
        seg_start_local.astype(jnp.int32).reshape(ep, g), ((0, 0), (0, LANES - g))
    )


@functools.cache
def _build_exchange_rows_tables(ep, width, n_rows, count_at, g, n_pre, tile_m):
    slab_rows = tx._slab_rows(width)
    r_lo = count_at // LANES
    r_hi = (count_at + ep * g - 1) // LANES
    n_cnt = r_hi - r_lo + 1

    def kernel(rank_sm, idx_sm, x_hbm, src_hbm, seg3_vm, out_hbm, dst_hbm, *rest):
        outs, (send_sem, recv_sem, lsem, gsem, csem, cnt_vm, r3_vm) = rest[:8], rest[8:]
        me = rank_sm[0]

        def copy_rows():
            def one(j, c):
                pltpu.make_async_copy(
                    src_hbm.at[pl.ds(idx_sm[j], 1)], dst_hbm.at[pl.ds(j, 1)], gsem
                ).start()
                return c

            lax.fori_loop(0, n_rows, one, jnp.int32(0), unroll=8)

        tx._exchange(ep, me, x_hbm, out_hbm, send_sem, recv_sem, lsem, copy_rows)
        # Every peer's row has landed; the row copies may still be in flight,
        # and the tables are built meanwhile from the exchanged counts.
        cp = pltpu.make_async_copy(out_hbm.at[:, pl.ds(r_lo, n_cnt), :], cnt_vm, csem)
        cp.start()
        cp.wait()
        cnt = cnt_vm[...]  # [s, n_cnt, 128]
        lane = lax.broadcasted_iota(jnp.int32, (ep, LANES), 1)
        for d in range(ep):
            start = count_at + d * g - r_lo * LANES
            r0, off = divmod(start, LANES)
            lo = cnt[:, r0, :]
            if off:
                lo = pltpu.roll(lo, LANES - off, 1)
                if off + g > LANES:
                    hi = pltpu.roll(cnt[:, r0 + 1, :], LANES - off, 1)
                    lo = jnp.where(lane < LANES - off, lo, hi)
            r3_vm[:, d, :] = jnp.where(lane < g, lo, 0)
        t = _tables(
            r3_vm[...], seg3_vm[...], me, ep=ep, g=g, n_pre=n_pre, tile_m=tile_m
        )
        _store_tables(t, outs, ep=ep, g=g)
        pltpu.make_async_copy(dst_hbm, dst_hbm, gsem).wait()

    def fn(packed, rank, rows, idx, seg_start_local):
        idx = idx.astype(jnp.int32).reshape(-1)
        r, slab = tx._payload(packed, rank, width)
        vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
        out, gathered, seg, small, *kt, counts = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[tx._HBM, tx._HBM, vmem],
                out_specs=(tx._HBM, tx._HBM) + (vmem,) * 8,
                scratch_shapes=(pltpu.SemaphoreType.DMA,) * 5
                + (
                    pltpu.VMEM((ep, n_cnt, LANES), jnp.int32),
                    pltpu.VMEM((ep, ep, LANES), jnp.int32),
                ),
                grid=(),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((ep, slab_rows, LANES), jnp.int32),
                jax.ShapeDtypeStruct((n_rows, *rows.shape[1:]), rows.dtype),
            )
            + _table_out_shapes(ep, g),
            compiler_params=tx._compiler_params(),
            name=f"moe_tc_routing_exchange_ep{ep}_w{width}_rows{n_rows}_tables",
        )(r, idx, slab, rows, _seg3(seg_start_local, ep, g))
        return (tx._unpad(out, ep, width), gathered) + _unpack_tables(
            seg, small, kt, counts, g
        )

    return fn


def build_exchange_rows_tables(*, ep, width, n_rows, count_at, e_total, n_pre, tile_m):
    """tc_exchange.build_tc_exchange_all_gather_rows' kernel that also builds
    build_push_tables' tables from the exchanged counts (words
    [count_at, count_at + e_total) of every rank's payload), while its row
    copies finish. fn(packed, rank, rows, idx, seg_start_local) ->
    (exchange, rows[idx], *build_push_tables(...))."""
    g = e_total // ep
    if g > LANES or ep > LANES or 2 * g + 3 > LANES:
        raise ValueError(f"the table kernel takes g_local, ep <= {LANES}")
    return _build_exchange_rows_tables(
        ep, width, n_rows, count_at, g, int(n_pre), int(tile_m)
    )

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


def _flatten_rows(x, g, n):
    """[rows, 128] with g live lanes per row -> the n = rows * g words laid
    end to end, as ceil(n / 128) [1, 128] rows."""
    lane = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    flat = [jnp.zeros((1, LANES), x.dtype) for _ in range(-(-n // LANES))]
    for d in range(x.shape[0]):
        r, off = divmod(d * g, LANES)
        if r >= len(flat):
            break
        piece = pltpu.roll(x[d : d + 1], off, 1) if off else x[d : d + 1]
        flat[r] = flat[r] + _sel((lane >= off) & (lane < off + g), piece)
        if off + g > LANES:
            flat[r + 1] = flat[r + 1] + _sel(lane < off + g - LANES, piece)
    return flat


def _tables(r3, seg3, me, *, ep, g, n_pre, tile_m, light_prepush=0):
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
    if light_prepush:
        # The SparseCore pre-push carries the n_pre LIGHTEST experts with
        # rows (lightest at 0), so it moves the fewest bytes. The rest follow
        # most rows first (1), or (2) rising to the heaviest mid-kernel and
        # falling back: the heavy experts' rows, now dispatched by the
        # kernel, get visits to arrive in. Both keep a light expert last,
        # whose result push is the kernel's exposed tail. Same experts, same
        # tiles: only the visit order changes.
        n_lp = jnp.minimum(n_visit, n_pre)
        m = n_visit - n_lp
        light = (rank >= m) & (rank < n_visit)
        rest = rank
        if light_prepush == 2:
            # Most-rows rank r: odd r rise on the left, even r fall on the
            # right from the peak r = 0.
            half = m >> 1
            rest = jnp.where((rank & 1) == 1, half - 1 - (rank >> 1), half + (rank >> 1))
        rank = jnp.where(light, n_visit - 1 - rank, jnp.where(rank < m, rest + n_lp, rank))
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

    # Inputs of the plan's per-pair tables (host.routing_pairs_from_tables):
    # this shard's arrival base per global expert, the mirror base, and the
    # slab row each (source, local expert) run starts at.
    rs_src = jnp.sum(_sel(real, ar_src_me), 1, keepdims=True)  # [d, 1]
    above = jnp.zeros((ep, 1), i32)
    acc_d = jnp.zeros((1, 1), i32)
    for d in range(ep):
        above = jnp.where(row[:, :1] == d, acc_d, above)
        acc_d = acc_d + rs_src[d : d + 1]
    my_recv_base = _sel(real, above + _lane_excl_cumsum(ar_src_me))  # [d, e]
    mir_base = _sel(
        real,
        ((my_recv_base >> shift) + 2 * (row * g + lane)) * tile_m
        + (run_me & (tile_m - 1)),
    )
    run_start_se = zero  # [s, e] = sum over sources above s of ar_dme
    acc = jnp.zeros((1, LANES), i32)
    for x in range(ep):
        run_start_se = jnp.where(row == x, acc, run_start_se)
        acc = acc + ar_dme[x : x + 1]
    run_base = run_start_se + slab_base

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
        plan=(
            _flatten_rows(my_recv_base, g, ep * g),
            _flatten_rows(mir_base, g, ep * g),
            run_base,
        ),
    )


@functools.cache
def _build(ep, g, n_pre, tile_m, interpret, light_prepush=0):
    def kernel(me_sm, r3_ref, seg3_ref, *outs):
        t = _tables(r3_ref[...], seg3_ref[...], me_sm[0], ep=ep, g=g, n_pre=n_pre,
                    tile_m=tile_m, light_prepush=light_prepush)  # fmt: skip
        _store_tables(t, outs, ep=ep, g=g)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem],
            out_specs=[vmem] * N_TABLE_OUTS,
            grid=(),
        ),
        out_shape=_table_out_shapes(ep, g),
        interpret=interpret,
        name=f"moe_push_tables_ep{ep}_g{g}" + (f"_lp{light_prepush}" if light_prepush else ""),
    )


def build_push_tables(rows_by_dest, seg_start_local, me, *, ep, n_pre, tile_m,
                      interpret=False, light_prepush=0):  # fmt: skip
    """The push layer's dispatch and shard tables in one kernel.

    Returns (seg, dmeta, visit, counts_sc, kernel_tables, expert_rows,
    slab_base, counts, plan_tables) equal to host.push_dispatch_tables(...,
    n_pre)'s (seg, meta, my_visit, counts_sc) and, for the same routing,
    v2's shard_transport_tables_in_blocks [:3] + [5],
    shard_push_tables_in_rows [0], shard_expert_slabs and
    shard_count_vector; plan_tables = (arrival base [e_total], mirror base
    [e_total], run slab base [ep, g_local]) for
    host.routing_pairs_from_tables. `light_prepush` (1 or 2) reorders the
    visits (see _tables) and then has no host equivalent.
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
    outs = _build(ep, g, int(n_pre), int(tile_m), bool(interpret), int(light_prepush))(
        me_arr, r3, _seg3(seg_start_local, ep, g)
    )
    return _unpack_tables(outs)


def _small_sizes(g):
    """Lengths of (meta, visit, pre-push counts, expert rows, slab base)."""
    return (2 * g + 3, g, 8, g, g)


N_TABLE_OUTS = 1 + 5 + 5 + 1 + 3


def _store_tables(t, outs, *, ep, g):
    seg_o, small_o, kt_o, counts_o = outs[0], outs[1:6], outs[6:11], outs[11]
    for k, part in enumerate(t["seg"]):
        seg_o[k] = part[:g, :ep]
    for o, part, n in zip(small_o, t["small"], _small_sizes(g)):
        o[...] = part[0, :n]
    for o, part in zip(kt_o, t["kernel_tables"]):
        o[...] = part[:g, :ep]
    counts_o[...] = t["counts"][:8, :ep]
    pos_o, mir_o, run_o = outs[12:15]
    n = ep * g
    for o, flat in ((pos_o, t["plan"][0]), (mir_o, t["plan"][1])):
        for r, part in enumerate(flat):
            w = min(LANES, n - r * LANES)
            o[pl.ds(r * LANES, w)] = part[0, :w]
    run_o[...] = t["plan"][2][:ep, :g]


def _table_out_shapes(ep, g):
    i32 = jnp.int32
    return (
        jax.ShapeDtypeStruct((3, g, ep), i32),
        *(jax.ShapeDtypeStruct((n,), i32) for n in _small_sizes(g)),
        *(jax.ShapeDtypeStruct((g, ep), i32) for _ in range(5)),
        jax.ShapeDtypeStruct((8, ep), i32),
        jax.ShapeDtypeStruct((ep * g,), i32),
        jax.ShapeDtypeStruct((ep * g,), i32),
        jax.ShapeDtypeStruct((ep, g), i32),
    )


def _unpack_tables(outs):
    seg, meta, visit, counts_sc, expert_rows, slab_base = outs[:6]
    return (
        seg.reshape(-1),
        meta,
        visit,
        counts_sc,
        tuple(outs[6:11]),
        expert_rows,
        slab_base,
        outs[11],
        tuple(outs[12:15]),
    )


def _seg3(seg_start_local, ep, g):
    return jnp.pad(
        seg_start_local.astype(jnp.int32).reshape(ep, g), ((0, 0), (0, LANES - g))
    )


def _relane_counts(cnt, r3_ref, *, count_at, g):
    """Exchanged rows' count words -> r3_ref [s, d, 128] (local expert e of
    destination d on lane e, zero past g). cnt [s, rows, 128] holds every
    source's flat count section starting at word count_at."""
    ep = cnt.shape[0]
    lane = lax.broadcasted_iota(jnp.int32, (ep, LANES), 1)
    for d in range(ep):
        r0, off = divmod(count_at + d * g, LANES)
        lo = cnt[:, r0, :]
        if off:
            lo = pltpu.roll(lo, LANES - off, 1)
            if off + g > LANES:
                hi = pltpu.roll(cnt[:, r0 + 1, :], LANES - off, 1)
                lo = jnp.where(lane < LANES - off, lo, hi)
        r3_ref[:, d, :] = jnp.where(lane < g, lo, 0)


@functools.cache
def _build_exchange_rows_tables(
    ep,
    width,
    n_rows,
    count_at,
    g,
    n_pre,
    tile_m,
    light_prepush=0,
    single_barrier=False,
):
    slab_rows = tx._slab_rows(width)
    r_lo = count_at // LANES
    r_hi = (count_at + ep * g - 1) // LANES
    n_cnt = r_hi - r_lo + 1

    def kernel(rank_sm, idx_sm, x_hbm, src_hbm, seg3_vm, out_hbm, dst_hbm, *rest):
        outs = rest[:N_TABLE_OUTS]
        send_sem, recv_sem, lsem, gsem, csem, cnt_vm, r3_vm = rest[N_TABLE_OUTS:]
        me = rank_sm[0]

        def copy_rows():
            def one(j, c):
                pltpu.make_async_copy(
                    src_hbm.at[pl.ds(idx_sm[j], 1)], dst_hbm.at[pl.ds(j, 1)], gsem
                ).start()
                return c

            lax.fori_loop(0, n_rows, one, jnp.int32(0), unroll=8)

        tx._exchange(
            ep,
            me,
            x_hbm,
            out_hbm,
            send_sem,
            recv_sem,
            lsem,
            copy_rows,
            single_barrier=single_barrier,
        )
        # Every peer's row has landed; the row copies may still be in flight,
        # and the tables are built meanwhile from the exchanged counts.
        cp = pltpu.make_async_copy(out_hbm.at[:, pl.ds(r_lo, n_cnt), :], cnt_vm, csem)
        cp.start()
        cp.wait()
        _relane_counts(cnt_vm[...], r3_vm, count_at=count_at - r_lo * LANES, g=g)
        t = _tables(
            r3_vm[...],
            seg3_vm[...],
            me,
            ep=ep,
            g=g,
            n_pre=n_pre,
            tile_m=tile_m,
            light_prepush=light_prepush,
        )
        _store_tables(t, outs, ep=ep, g=g)
        pltpu.make_async_copy(dst_hbm, dst_hbm, gsem).wait()

    def fn(packed, rank, rows, idx, seg_start_local):
        idx = idx.astype(jnp.int32).reshape(-1)
        r, slab = tx._payload(packed, rank, width)
        vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
        out, gathered, *tables = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[tx._HBM, tx._HBM, vmem],
                out_specs=(tx._HBM, tx._HBM) + (vmem,) * N_TABLE_OUTS,
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
            name=f"moe_tc_routing_exchange_ep{ep}_w{width}_rows{n_rows}_tables"
            + (f"_lp{light_prepush}" if light_prepush else "")
            + ("_1b" if single_barrier else ""),
        )(r, idx, slab, rows, _seg3(seg_start_local, ep, g))
        return (tx._unpad(out, ep, width), gathered) + _unpack_tables(tables)

    return fn


def build_exchange_rows_tables(
    *,
    ep,
    width,
    n_rows,
    count_at,
    e_total,
    n_pre,
    tile_m,
    light_prepush=0,
    single_barrier=False,
):
    """tc_exchange.build_tc_exchange_all_gather_rows' kernel that also builds
    build_push_tables' tables from the exchanged counts (words
    [count_at, count_at + e_total) of every rank's payload), while its row
    copies finish. fn(packed, rank, rows, idx, seg_start_local) ->
    (exchange, rows[idx], *build_push_tables(...))."""
    g = e_total // ep
    if g > LANES or ep > LANES or 2 * g + 3 > LANES:
        raise ValueError(f"the table kernel takes g_local, ep <= {LANES}")
    return _build_exchange_rows_tables(
        ep,
        width,
        n_rows,
        count_at,
        g,
        int(n_pre),
        int(tile_m),
        int(light_prepush),
        bool(single_barrier),
    )

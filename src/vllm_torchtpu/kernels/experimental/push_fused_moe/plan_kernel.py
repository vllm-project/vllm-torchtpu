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
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    ROWBLK,
    ROWBLK_SHIFT,
    off_shard_slab_row,
    pow2_shift,
)

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
    return _flatten_row_list([x[d : d + 1] for d in range(x.shape[0])], g, n)


def _flatten_row_list(rows, g, n):
    """`_flatten_rows` of a list of [1, 128] rows."""
    lane = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    flat = [jnp.zeros((1, LANES), rows[0].dtype) for _ in range(-(-n // LANES))]
    for d, x_d in enumerate(rows):
        r, off = divmod(d * g, LANES)
        if r >= len(flat):
            break
        piece = pltpu.roll(x_d, off, 1) if off else x_d
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
    # (v2's expert_visit_list key, ranked by compare). From here on the
    # visit tables are worked in the [e or v, d] layout they are stored in
    # (experts, then visits, on sublanes): rank is a permutation of [0, g)
    # over each destination's experts, so expert e of destination d lands on
    # sublane rank[d, e] by selects alone (gathering per visit instead costs
    # a cross-lane reduction per visit and table).
    gr = -(-g // 8) * 8
    sub = lax.broadcasted_iota(i32, (gr, LANES), 0)
    col = lax.broadcasted_iota(i32, (gr, LANES), 1)  # destination d

    def to_t(x):
        return _transpose(x)[:gr]

    def row_of(x, e):
        return jnp.broadcast_to(x[e : e + 1], (gr, LANES))

    ea_t = to_t(ea)
    real_t = (sub < g) & (col < ep)
    key = jnp.where(real_t, sub - ea_t * g, i32(2**30))
    rank = jnp.zeros((gr, LANES), i32)
    for e in range(g):
        rank = rank + (row_of(key, e) < key).astype(i32)
    n_visit = jnp.sum((real_t & (ea_t > 0)).astype(i32), 0, keepdims=True)  # [1, d]
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
            rest = jnp.where(
                (rank & 1) == 1, half - 1 - (rank >> 1), half + (rank >> 1)
            )
        rank = jnp.where(
            light, n_visit - 1 - rank, jnp.where(rank < m, rest + n_lp, rank)
        )
    tables = [to_t(x) for x in (mine, seg3, local_base + run_me)]
    visit_t = jnp.zeros((gr, LANES), i32)
    got = [visit_t] * len(tables)
    for e in range(g):
        hit = sub == row_of(rank, e)
        visit_t = jnp.where(hit, e, visit_t)
        got = [jnp.where(hit, row_of(x, e), o) for x, o in zip(tables, got)]
    live_t = (sub < n_visit) & (col < ep)  # [v, d]
    cnt_t, src_off_t, dst_t = (_sel(live_t, x) for x in got)
    seg_a_t = jnp.where(sub < g, (col << 24) | (visit_t << 16) | cnt_t, 0)
    seg_c_t = _sel(cnt_t > 0, dst_t)
    recv = jnp.sum(r_dme, 0, keepdims=True)  # [1, e]
    my_visit = jnp.sum(_sel(row == me, _transpose(visit_t)[:ep]), 0, keepdims=True)
    lane1 = lane[:1]
    real1 = lane1 < g
    # rows_at_v (rows this shard receives at its visit v), summed over the
    # pre-push positions straight from the expert layout: column me of rank
    # is my visit of each expert.
    recv_t = jnp.sum(to_t(r_dme), 1, keepdims=True)  # [e, 1]
    at_me = (col == me) & (sub < g) & (rank < n_visit)
    pre_e = at_me & (rank < n_pre)
    odd_e = (rank & 1) == 1
    pre_v = (col != me) & (sub < n_pre)
    odd_v = (sub & 1) == 1
    sc = [
        jnp.sum(_sel(pre_e & ~odd_e, recv_t)),
        jnp.sum(_sel(pre_e & odd_e, recv_t)),
        jnp.sum(_sel(pre_v & ~odd_v, cnt_t)),
        jnp.sum(_sel(pre_v & odd_v, cnt_t)),
    ]
    n_remote = jnp.sum(_sel((col != me) & (sub >= n_pre), cnt_t))
    n_sent = jnp.sum(cnt_t)
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
        seg=(seg_a_t, src_off_t, seg_c_t),
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
        name=f"moe_push_tables_ep{ep}_g{g}"
        + (f"_lp{light_prepush}" if light_prepush else ""),
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
    # Flat, as its scalar-prefetch readers take it: a [3, g, ep] output costs
    # XLA a synchronous relayout to 1-D in HBM before the first reader.
    rows = [part[e : e + 1] for part in t["seg"] for e in range(g)]
    _store_flat(seg_o, _flatten_row_list(rows, ep, 3 * g * ep), 3 * g * ep)
    for o, part, n in zip(small_o, t["small"], _small_sizes(g)):
        o[...] = part[0, :n]
    for o, part in zip(kt_o, t["kernel_tables"]):
        o[...] = part[:g, :ep]
    counts_o[...] = t["counts"][:8, :ep]
    pos_o, mir_o, run_o = outs[12:15]
    for o, flat in ((pos_o, t["plan"][0]), (mir_o, t["plan"][1])):
        _store_flat(o, flat, ep * g)
    run_o[...] = t["plan"][2][:ep, :g]


def _store_flat(o, flat, n):
    """Store `_flatten_rows`' [1, 128] rows into the 1-D ref o of n words."""
    for r, part in enumerate(flat):
        w = min(LANES, n - r * LANES)
        o[pl.ds(r * LANES, w)] = part[0, :w]


def _table_out_shapes(ep, g):
    i32 = jnp.int32
    return (
        jax.ShapeDtypeStruct((3 * g * ep,), i32),
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
        seg,
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


def _count_rows(count_at, ep, g):
    """(first row, rows) of the payload slab holding the e_total count words."""
    r_lo = count_at // LANES
    return r_lo, (count_at + ep * g - 1) // LANES - r_lo + 1


def _exchange_name(
    ep, width, n_rows, light_prepush, single_barrier, part="", scatter_rows=False
):
    return (
        f"moe_tc_routing_exchange_ep{ep}_w{width}_rows{n_rows}_tables"
        + (f"_lp{light_prepush}" if light_prepush else "")
        + part
        + ("_1b" if single_barrier else "")
        + ("_rowmap" if scatter_rows else "")
    )


def _issue_row_copies(src_hbm, dst_hbm, idx_sm, gsem, n_rows, scatter=False):
    """dst[j] = src[idx[j]], one local DMA per row, left in flight on gsem;
    `scatter`: dst[idx[j] & 0xFFFF] = src[idx[j] >> 16] (build_local_ranks'
    row map), the same copies issued in pair order."""

    def one(j, c):
        if scatter:
            src, dst = idx_sm[j] >> 16, idx_sm[j] & 0xFFFF
        else:
            src, dst = idx_sm[j], j
        pltpu.make_async_copy(
            src_hbm.at[pl.ds(src, 1)], dst_hbm.at[pl.ds(dst, 1)], gsem
        ).start()
        return c

    lax.fori_loop(0, n_rows, one, jnp.int32(0), unroll=8)


def _tables_after_exchange(
    me, out_hbm, dst_hbm, seg3_vm, outs, csem, gsem, cnt_vm, r3_vm, *, g, **kw
):
    """Every peer's row has landed in out_hbm; the row copies may still be in
    flight, and the tables are built meanwhile from the exchanged counts.
    `kw`: count_at and _tables' ep, n_pre, tile_m, light_prepush."""
    count_at = kw.pop("count_at")
    r_lo, n_cnt = _count_rows(count_at, kw["ep"], g)
    cp = pltpu.make_async_copy(out_hbm.at[:, pl.ds(r_lo, n_cnt), :], cnt_vm, csem)
    cp.start()
    cp.wait()
    _relane_counts(cnt_vm[...], r3_vm, count_at=count_at - r_lo * LANES, g=g)
    _store_tables(
        _tables(r3_vm[...], seg3_vm[...], me, g=g, **kw), outs, ep=kw["ep"], g=g
    )
    pltpu.make_async_copy(dst_hbm, dst_hbm, gsem).wait()


def _table_scratch(ep, n_cnt):
    """The count words and their re-laned [s, d, 128] form."""
    return (
        pltpu.VMEM((ep, n_cnt, LANES), jnp.int32),
        pltpu.VMEM((ep, ep, LANES), jnp.int32),
    )


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
    scatter_rows=False,
):
    slab_rows = tx._slab_rows(width)
    n_cnt = _count_rows(count_at, ep, g)[1]
    table_kw = dict(
        g=g,
        count_at=count_at,
        ep=ep,
        n_pre=n_pre,
        tile_m=tile_m,
        light_prepush=light_prepush,
    )

    def kernel(rank_sm, idx_sm, x_hbm, src_hbm, seg3_vm, out_hbm, dst_hbm, *rest):
        outs = rest[:N_TABLE_OUTS]
        send_sem, recv_sem, lsem, gsem, csem, cnt_vm, r3_vm = rest[N_TABLE_OUTS:]
        me = rank_sm[0]
        tx._exchange(
            ep,
            me,
            x_hbm,
            out_hbm,
            send_sem,
            recv_sem,
            lsem,
            lambda: _issue_row_copies(
                src_hbm, dst_hbm, idx_sm, gsem, n_rows, scatter_rows
            ),
            single_barrier=single_barrier,
        )
        _tables_after_exchange(
            me, out_hbm, dst_hbm, seg3_vm, outs, csem, gsem, cnt_vm, r3_vm, **table_kw
        )

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
                + _table_scratch(ep, n_cnt),
                grid=(),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((ep, slab_rows, LANES), jnp.int32),
                jax.ShapeDtypeStruct((n_rows, *rows.shape[1:]), rows.dtype),
            )
            + _table_out_shapes(ep, g),
            compiler_params=tx._compiler_params(),
            name=_exchange_name(
                ep,
                width,
                n_rows,
                light_prepush,
                single_barrier,
                scatter_rows=scatter_rows,
            ),
        )(r, idx, slab, rows, _seg3(seg_start_local, ep, g))
        # The raw [ep, rows, 128] payload last, for build_pair_tables.
        return (tx._unpad(out, ep, width), gathered) + _unpack_tables(tables) + (out,)

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
    scatter_rows=False,
):
    """tc_exchange.build_tc_exchange_all_gather_rows' kernel that also builds
    build_push_tables' tables from the exchanged counts (words
    [count_at, count_at + e_total) of every rank's payload), while its row
    copies finish. fn(packed, rank, rows, idx, seg_start_local) ->
    (exchange, rows[idx], *build_push_tables(...)); with `scatter_rows`, idx
    is build_local_ranks' row map and the rows are the ones it maps."""
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
        bool(scatter_rows),
    )


_SEM = pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)
# The split kernels' DMA semaphores: send, recv, local row, row copies.
_N_SPLIT_SEMS = 4


@functools.cache
def _build_exchange_rows_tables_split(
    ep,
    width,
    n_rows,
    count_at,
    g,
    n_pre,
    tile_m,
    light_prepush=0,
    single_barrier=False,
    scatter_rows=False,
):
    slab_rows = tx._slab_rows(width)
    n_cnt = _count_rows(count_at, ep, g)[1]
    table_kw = dict(
        g=g,
        count_at=count_at,
        ep=ep,
        n_pre=n_pre,
        tile_m=tile_m,
        light_prepush=light_prepush,
    )
    pair_rows = not scatter_rows

    def start_kernel(rank_sm, idx_sm, x_hbm, src_hbm, out_hbm, dst_hbm, sems):
        send_sem, recv_sem, lsem, gsem = (sems.at[i] for i in range(_N_SPLIT_SEMS))
        tx._exchange_start(
            ep,
            rank_sm[0],
            x_hbm,
            out_hbm,
            send_sem,
            recv_sem,
            lsem,
            lambda: _issue_row_copies(
                src_hbm, dst_hbm, idx_sm, gsem, n_rows, scatter_rows
            ),
            single_barrier=single_barrier,
        )

    def done_kernel(rank_sm, x_hbm, src_hbm, out_in, dst_in, sems, seg3_vm, *rest):
        # src_hbm is only kept live for the row copies still reading it;
        # out_in and dst_in are out_hbm and dst_hbm (aliased).
        del src_hbm, out_in, dst_in
        out_hbm, dst_hbm = rest[:2]
        outs = rest[2 : 2 + N_TABLE_OUTS]
        rest = rest[2 + N_TABLE_OUTS :]
        if pair_rows:
            pay_o, psem, *rest = rest
        csem, cnt_vm, r3_vm = rest
        send_sem, recv_sem, lsem, gsem = (sems.at[i] for i in range(_N_SPLIT_SEMS))
        me = rank_sm[0]
        tx._exchange_wait(ep, me, x_hbm, out_hbm, send_sem, recv_sem, lsem)
        if pair_rows:
            pay = pltpu.make_async_copy(out_hbm.at[:, pl.ds(0, 2), :], pay_o, psem)
            pay.start()
        _tables_after_exchange(
            me, out_hbm, dst_hbm, seg3_vm, outs, csem, gsem, cnt_vm, r3_vm, **table_kw
        )
        if pair_rows:
            pay.wait()

    def start(packed, rank, rows, idx):
        idx = idx.astype(jnp.int32).reshape(-1)
        r, slab = tx._payload(packed, rank, width)
        out, gathered, sems = pl.pallas_call(
            start_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[tx._HBM, tx._HBM],
                out_specs=(tx._HBM, tx._HBM, _SEM),
                grid=(),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((ep, slab_rows, LANES), jnp.int32),
                jax.ShapeDtypeStruct((n_rows, *rows.shape[1:]), rows.dtype),
                pltpu.SemaphoreType.DMA((_N_SPLIT_SEMS,)),
            ),
            compiler_params=tx._compiler_params(),
            name=_exchange_name(
                ep,
                width,
                n_rows,
                light_prepush,
                single_barrier,
                "_start",
                scatter_rows,
            ),
        )(r, idx, slab, rows)
        return r, slab, out, gathered, sems

    def done(handle, rows, seg_start_local):
        r, slab, out, gathered, sems = handle
        vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
        out, gathered, *tables = pl.pallas_call(
            done_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[tx._HBM, tx._HBM, tx._HBM, tx._HBM, _SEM, vmem],
                out_specs=(tx._HBM, tx._HBM) + (vmem,) * (N_TABLE_OUTS + pair_rows),
                scratch_shapes=(pltpu.SemaphoreType.DMA,) * (1 + pair_rows)
                + _table_scratch(ep, n_cnt),
                grid=(),
            ),
            out_shape=(
                jax.ShapeDtypeStruct(out.shape, out.dtype),
                jax.ShapeDtypeStruct(gathered.shape, gathered.dtype),
            )
            + _table_out_shapes(ep, g)
            + ((jax.ShapeDtypeStruct((ep, 2, LANES), jnp.int32),) if pair_rows else ()),
            input_output_aliases={3: 0, 4: 1},
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
            name=_exchange_name(
                ep,
                width,
                n_rows,
                light_prepush,
                single_barrier,
                "_done",
                scatter_rows,
            ),
        )(r, slab, rows, out, gathered, sems, _seg3(seg_start_local, ep, g))
        # The payload last: for build_pair_tables, its pair and scale rows,
        # copied into VMEM beside the table build (XLA would otherwise stage
        # the whole HBM payload right before that kernel); for the wide path's
        # build_pair_rows, which reads every row, the raw [ep, rows, 128].
        pay = tables.pop() if pair_rows else out
        return (tx._unpad(out, ep, width), gathered) + _unpack_tables(tables) + (pay,)

    return start, done


def build_exchange_rows_tables_split(
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
    scatter_rows=False,
):
    """build_exchange_rows_tables' kernel as two calls, so independent work
    can run while its rows are in flight: start(packed, rank, rows, idx) ->
    handle (barrier, every push and row copy issued, none waited);
    done(handle, rows, seg_start_local) -> build_exchange_rows_tables'
    outputs, the last one (the raw payload) cut to the [ep, 2, 128] rows
    build_pair_tables reads unless `scatter_rows`. `rows` goes to both,
    keeping it live for the copies reading it.
    The handle holds buffers peers are still writing: it is opaque, and goes
    to done once and to nothing else (a copy of it would read rows mid-flight)."""
    g = e_total // ep
    if g > LANES or ep > LANES or 2 * g + 3 > LANES:
        raise ValueError(f"the table kernel takes g_local, ep <= {LANES}")
    return _build_exchange_rows_tables_split(
        ep,
        width,
        n_rows,
        count_at,
        g,
        int(n_pre),
        int(tile_m),
        int(light_prepush),
        bool(single_barrier),
        bool(scatter_rows),
    )


def pair_tables_apply(t_local, topk, scale_at, scale_rows):
    """Whether build_pair_tables serves this step: every shard's routed pairs
    in one payload row, its row scales in the next, and an f32 scale slab."""
    n = t_local * topk
    return n <= LANES and scale_at == LANES and t_local <= LANES and scale_rows > 0


@functools.cache
def _build_pair_tables(ep, g, t_local, topk, block, e_total, scale_rows, interpret):
    n = t_local * topk
    a_rows = -(-4 * scale_rows // 16) * 16

    def kernel(me_sm, out_hbm, rb_ref, mb_ref, run_ref, pos_o, mir_o, slab_o,
               pay_vm, slab_vm, sem):  # fmt: skip
        i32, bf16, f32 = jnp.int32, jnp.bfloat16, jnp.float32
        me = me_sm[0]
        cp = pltpu.make_async_copy(out_hbm.at[:, pl.ds(0, 2), :], pay_vm, sem)
        cp.start()
        cp.wait()
        pairs = pay_vm[:, 0, :]  # [s, q]: source s's q-th routed pair's expert
        sbits = pay_vm[:, 1, :]  # [s, t]: source s's token t's row-scale bits
        pairs_t = _transpose(pairs)  # [q, s]
        shape = (LANES, LANES)
        sub = lax.broadcasted_iota(i32, shape, 0)  # the pair q (rows)
        lane = lax.broadcasted_iota(i32, shape, 1)
        earlier = lane < sub  # [q, q']: q' before q
        row8 = lax.broadcasted_iota(i32, (ep, LANES), 0)

        def col(x, s):
            """Column s of a [q, s] tile, as [q, 1]."""
            return jnp.sum(_sel(lane == s, x), 1, keepdims=True)

        def lookup(table_row, idx):
            """table_row[0, idx[q]] for an [1, 128] table, [q, 1] -> [q, 1]."""
            return jnp.sum(_sel(idx == lane, table_row), 1, keepdims=True)

        # This shard's own pairs: arrival row and mirror position. A pair
        # routed nowhere keeps the plan's rank within its routing block.
        e_me = col(pairs_t, me)
        me_pairs = jnp.sum(_sel(row8 == me, pairs), 0, keepdims=True)
        same_me = (e_me == me_pairs) & earlier
        before = jnp.sum(same_me.astype(i32), 1, keepdims=True)
        blk = pow2_shift(block, "the routing block")
        same_block = (sub >> blk) == (lane >> blk)
        in_block = jnp.sum((same_me & same_block).astype(i32), 1, keepdims=True)
        rb = rb_ref[...]
        mb = mb_ref[...]
        base = sum(lookup(rb[:, c * LANES : (c + 1) * LANES], e_me - c * LANES)
                   for c in range(rb.shape[1] // LANES))  # fmt: skip
        mbase = sum(lookup(mb[:, c * LANES : (c + 1) * LANES], e_me - c * LANES)
                    for c in range(mb.shape[1] // LANES))  # fmt: skip
        routed = (e_me >= 0) & (e_me < e_total)
        pos = jnp.where(routed, base + before, in_block)
        mir = jnp.where(routed, mbase + before, in_block)
        both = jnp.where(lane == 0, pos, jnp.where(lane == 1, mir, 0)).T
        pos_o[...] = both[0:1, :]
        mir_o[...] = both[1:2, :]

        # The scale slab: source s's pair q on one of this shard's experts
        # lands on run_base[s, e - first] plus its rank among the source's
        # pairs on that expert, and carries its token's row-scale bits. Built
        # as layer._scale_slab_onehot builds it (same exactness argument),
        # with the operand rows ordered byte-major per source.
        first = me * g
        token_of = (sub >= lane * topk) & (sub < (lane + 1) * topk)  # [q, t]
        r_a = lax.broadcasted_iota(i32, (a_rows, LANES), 0)
        # Operand row r holds byte r // scale_rows of slab row r % scale_rows.
        byte_of = sum((r_a >= b * scale_rows).astype(i32) for b in range(1, 4))
        hi_of = r_a - byte_of * scale_rows
        acc = jnp.zeros((a_rows, LANES), f32)
        run = run_ref[...]
        for s in range(ep):
            e_s = col(pairs_t, s)
            loc = e_s - first
            mine = (loc >= 0) & (loc < g) & (sub[:, :1] < n)
            rank = jnp.sum(
                ((e_s == pairs[s : s + 1]) & earlier).astype(i32), 1, keepdims=True
            )
            row = lookup(run[s : s + 1], loc) + rank
            row = jnp.where(mine, row, scale_rows * LANES)  # off the slab: dropped
            bits = jnp.sum(_sel(token_of, sbits[s : s + 1]), 1, keepdims=True)
            hi = row >> 7
            lo = row & (LANES - 1)
            cols = jnp.where(lane == 0, hi, 0)
            for b in range(4):
                cols = jnp.where(lane == 1 + b, (bits >> (8 * b)) & 255, cols)
            rows = cols.T  # row 0: hi, rows 1-4: the bytes
            byte = jnp.zeros((a_rows, LANES), i32)
            for b in range(4):
                byte = jnp.where(byte_of == b, rows[1 + b : 2 + b], byte)
            a = _sel((hi_of == rows[0:1]) & (r_a < 4 * scale_rows), byte)
            onehot = (lo == lane).astype(bf16)  # [q, lane]
            acc = acc + jnp.dot(a.astype(bf16), onehot, preferred_element_type=f32)
        words = acc.astype(i32)
        w = words[0:scale_rows]
        for b in range(1, 4):
            w = w | (words[b * scale_rows : (b + 1) * scale_rows] << (8 * b))
        slab_vm[...] = lax.bitcast_convert_type(w, f32)
        out = pltpu.make_async_copy(slab_vm, slab_o, sem)
        out.start()
        out.wait()

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm, vmem, vmem, vmem],
            out_specs=(vmem, vmem, hbm),
            scratch_shapes=(
                pltpu.VMEM((ep, 2, LANES), jnp.int32),
                pltpu.VMEM((scale_rows, LANES), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((1, LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, LANES), jnp.int32),
            jax.ShapeDtypeStruct((scale_rows, LANES), jnp.float32),
        ),
        interpret=interpret,
        name=f"moe_push_pair_tables_ep{ep}_t{t_local}_k{topk}",
    )


def build_pair_tables(
    exchange_out, recv_base, mirror_base, run_base, me, *, ep, t_local, topk,
    block, e_total, scale_rows, interpret=False,
):  # fmt: skip
    """routing_pairs_from_tables' arrival rows and mirror positions for this
    shard's pairs, and the activation scale slab its routed pairs fill, in one
    kernel straight from the exchanged payload (`exchange_out`, the exchange
    kernel's [ep, rows, 128] output or its first two rows: pairs in row 0,
    row-scale bits in row 1). `recv_base`/`mirror_base` [e_total] and
    `run_base` [ep, g] are the table kernel's plan tables. Same integers as
    host.routing_pairs_from_tables and the scale slab's drop-mode scatter
    (every routed pair has a slab row of its own). Returns (pos [t_local,
    topk], mirror_pos, scale slab f32 [scale_rows, 128])."""
    g = e_total // ep
    pad = (-e_total) % LANES
    call = _build_pair_tables(
        ep, g, t_local, topk, block, e_total, scale_rows, interpret
    )
    # No with_memory_space_constraint on the payload or the slab (unlike the
    # decode combine): pinning them made the kernel wait on the payload's HBM
    # read (layer bench: 3.3 vs 1.95 us).
    pos, mir, slab = call(
        jnp.asarray(me, jnp.int32).reshape(1),
        exchange_out,
        jnp.pad(recv_base.astype(jnp.int32), (0, pad)).reshape(1, -1),
        jnp.pad(mirror_base.astype(jnp.int32), (0, pad)).reshape(1, -1),
        jnp.pad(run_base.astype(jnp.int32), ((0, 0), (0, LANES - g))),
    )
    n = t_local * topk
    return (
        pos[0, :n].reshape(t_local, topk),
        mir[0, :n].reshape(t_local, topk),
        slab,
    )


# Largest step build_local_ranks / build_pair_rows take: their words pack
# a pair's sorted row or rank in 16 bits (below 2**15, so the sign is clear),
# and they unroll one loop body per 1024 pairs.
WIDE_PAIRS_MAX = 1 << 14


def wide_pair_tables_apply(t_local, topk, block):
    """Whether build_local_ranks / build_pair_rows serve this step: more
    routed pairs than build_pair_tables takes, at most WIDE_PAIRS_MAX, in
    whole-lane routing blocks and whole [8, 128] tiles of pairs."""
    n = t_local * topk
    return (
        LANES < n <= WIDE_PAIRS_MAX
        and block in (LANES, 2 * LANES)
        and n % (8 * LANES) == 0
    )


def _align(v, m):
    return -(-v // m) * m


def _lookup_rows(table, idx):
    """table[r, idx[r, l]] per element of an int32 [8, 128] tile, for [8, k]
    tables (k a multiple of 128) and 0 <= idx: in-register lane gathers,
    one per 128 entries; an index past the table reads 0."""
    out = jnp.zeros(idx.shape, table.dtype)
    for t in range(table.shape[1] // LANES):
        got = jnp.take_along_axis(
            table[:, t * LANES : (t + 1) * LANES],
            idx & (LANES - 1),
            axis=1,
            mode="promise_in_bounds",
        )
        out = jnp.where((idx >> 7) == t, got, out)
    return out


def _lookup(table, idx):
    """_lookup_rows with one [1, k] table for every row."""
    return _lookup_rows(jnp.broadcast_to(table, (idx.shape[0], table.shape[1])), idx)


@functools.cache
def _build_local_ranks(n, e_total, topk, block, interpret):
    i32, bf16, f32 = jnp.int32, jnp.bfloat16, jnp.float32
    nc = n // LANES  # 128-pair chunks, 8 to a tile
    ncp = _align(nc, LANES)
    kp = _align(e_total + 1, LANES)  # expert bins, nowhere (e_total) included
    cpb = block // LANES
    t_local = n // topk

    def kernel(pairs_ref, counts_o, seg_o, rowmap_o, words_o, hist_vm, carry_vm):
        sub = lax.broadcasted_iota(i32, (8, LANES), 0)
        lane = lax.broadcasted_iota(i32, (8, LANES), 1)

        def experts(i):
            rows = pl.ds(pl.multiple_of(8 * i, 8), 8)
            return rows, jnp.minimum(pairs_ref[rows, :], e_total)

        # Pass 1: a tile's 8 chunk histograms in one matmul over its pairs:
        # [(high bits h, chunk j), (chunk j', pair)] one-hot (j == j') times
        # the low 7 bits' one-hot [128, (chunk j', pair)] transposed counts
        # bin 128 * h + l of chunk j in row 8 * h + j, lane l.
        n_high = kp // LANES
        hj = lax.broadcasted_iota(i32, (8 * n_high, LANES), 0)
        sub_sq = lax.broadcasted_iota(i32, (LANES, LANES), 0)

        def count(i, carry):
            rows, e = experts(i)
            high = jnp.concatenate(
                [
                    ((hj & 7) == j) & ((hj >> 3) == (e[j : j + 1] >> 7))
                    for j in range(8)
                ],
                1,
            ).astype(bf16)
            low = jnp.concatenate(
                [sub_sq == (e[j : j + 1] & (LANES - 1)) for j in range(8)], 1
            ).astype(bf16)
            h = lax.dot_general(
                high, low, (((1,), (1,)), ((), ())), preferred_element_type=f32
            )
            hist_vm[rows, :] = jnp.concatenate(
                [h[8 * t : 8 * (t + 1)] for t in range(n_high)], 1
            )
            return carry

        hist_vm[...] = jnp.zeros((ncp, kp), f32)
        lax.fori_loop(0, nc // 8, count, 0, unroll=True)
        # Each expert's pairs in earlier chunks: a strictly-lower-triangular
        # matmul over counts <= 128 (exact in bf16), sums <= n exact in f32.
        r = lax.broadcasted_iota(i32, (ncp, ncp), 0)
        before = (lax.broadcasted_iota(i32, (ncp, ncp), 1) < r).astype(bf16)
        hist = hist_vm[...]
        carry_vm[...] = jnp.dot(
            before, hist.astype(bf16), preferred_element_type=f32
        ).astype(i32)
        total = jnp.sum(hist, 0, keepdims=True).astype(i32)  # [1, kp]
        # Segment starts; nowhere's run follows every routed pair.
        seg, start = [], jnp.zeros((1, 1), i32)
        for t in range(kp // LANES):
            cnt = total[:, t * LANES : (t + 1) * LANES]
            seg.append(_lane_excl_cumsum(cnt) + start)
            start = start + jnp.sum(cnt, 1, keepdims=True)
            w = min(LANES, e_total - t * LANES)
            if w > 0:
                counts_o[pl.ds(t * LANES, w)] = cnt[0, :w]
                seg_o[pl.ds(t * LANES, w)] = seg[-1][0, :w]
        seg = jnp.concatenate(seg, 1)

        # Pass 2: a pair's rank is its chunk's carry for its expert (a lane
        # gather per sublane) plus the same-expert pairs before it in the
        # chunk (a pairwise compare).
        earlier = lax.broadcasted_iota(i32, (LANES, LANES), 0) < lax.broadcasted_iota(
            i32, (LANES, LANES), 1
        )

        def place(i, carry):
            rows, e = experts(i)
            carry_rows = carry_vm[rows, :]
            e_t = _transpose(e)  # [pair, chunk]
            w = jnp.zeros((8, LANES), i32)
            for j in range(8):
                other = jnp.broadcast_to(e_t[:, j : j + 1], (LANES, LANES))
                same = other == jnp.broadcast_to(e[j : j + 1], (LANES, LANES))
                w_j = jnp.sum((same & earlier).astype(i32), 0, keepdims=True)
                w = jnp.where(sub == j, w_j, w)
            rank = _lookup_rows(carry_rows, e) + w
            # A pair routed nowhere keeps the plan's rank within its routing
            # block: less the nowhere pairs before the block.
            at_nowhere = _lookup_rows(carry_rows, jnp.full(e.shape, e_total, i32))
            blk0 = jnp.take_along_axis(
                at_nowhere, sub - sub % cpb, axis=0, mode="promise_in_bounds"
            )
            nowhere = e == e_total
            # p // topk in f32: |error| < n * 2**-23, far inside the 0.5 / topk
            # margin to the next integer.
            p = ((8 * i + sub) * LANES + lane).astype(f32) + 0.5
            src = jnp.where(nowhere, t_local - 1, (p * (1.0 / topk)).astype(i32))
            rowmap_o[rows, :] = (_lookup(seg, e) + rank) | (src << 16)
            words_o[rows, :] = e | (jnp.where(nowhere, rank - blk0, rank) << 16)
            return carry

        lax.fori_loop(0, nc // 8, place, 0, unroll=True)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    return pl.pallas_call(
        kernel,
        in_specs=[vmem],
        out_specs=(vmem,) * 4,
        out_shape=(
            jax.ShapeDtypeStruct((e_total,), i32),
            jax.ShapeDtypeStruct((e_total,), i32),
            jax.ShapeDtypeStruct((nc, LANES), i32),
            jax.ShapeDtypeStruct((nc, LANES), i32),
        ),
        scratch_shapes=(
            pltpu.VMEM((ncp, kp), f32),
            pltpu.VMEM((ncp, kp), i32),
        ),
        interpret=interpret,
        name=f"moe_push_local_ranks_n{n}_e{e_total}",
    )


def build_local_ranks(local_pairs, *, e_total, topk, block, interpret=False):
    """This shard's routed pairs ranked in one kernel, for a step
    wide_pair_tables_apply admits. `local_pairs` [t_local * topk] are expert
    ids in [0, e_total], e_total routing nowhere. Returns (counts [e_total],
    seg_start [e_total] (host.local_sort_tokens'), row_map [n], words [n]):
    row_map[p] = the expert-sorted row of pair p | its source token << 16,
    the token host.local_sort_tokens puts on that row (the exchange kernel's
    `scatter_rows` map), and words[p] = its expert | its rank << 16, the
    payload word build_pair_rows reads: the rank among this shard's earlier
    pairs on its expert or, routed nowhere, within its routing block."""
    n = local_pairs.shape[0]
    counts, seg, row_map, words = _build_local_ranks(
        n, e_total, topk, block, interpret
    )(local_pairs.astype(jnp.int32).reshape(n // LANES, LANES))
    return counts, seg, row_map.reshape(n), words.reshape(n)


@functools.cache
def _build_pair_rows(ep, g, n, e_total, off_shard, shard_stride, interpret):
    i32 = jnp.int32
    nc = n // LANES
    tiles = nc // 8

    def kernel(me_sm, pay_vm, rb_ref, mb_ref, run_ref, pos_o, mir_o, slab_o):
        me = me_sm[0]
        rb, mb = rb_ref[...], mb_ref[...]

        def local(i, carry):
            rows = pl.ds(pl.multiple_of(8 * i, 8), 8)
            w = pay_vm[me, rows, :]
            e, r = w & 0xFFFF, w >> 16
            pos_o[rows, :] = r + _lookup(rb, e)
            mir_o[rows, :] = r + _lookup(mb, e)
            return carry

        lax.fori_loop(0, tiles, local, 0, unroll=True)

        # A pair on one of this shard's experts lands on its (source, expert)
        # run's slab row plus its rank among the source's pairs on that expert.
        first = me * g
        off = off_shard - me * shard_stride
        run = jnp.concatenate([run_ref[...], jnp.zeros((ep, LANES - g), i32)], 1)
        for s in range(ep):

            def slab(i, carry, s=s):
                w = pay_vm[s, pl.ds(pl.multiple_of(8 * i, 8), 8), :]
                loc = (w & 0xFFFF) - first
                mine = (loc >= 0) & (loc < g)
                got = _lookup(run[s : s + 1], jnp.where(mine, loc, 0)) + (w >> 16)
                at = pl.ds(pl.multiple_of(s * nc + 8 * i, 8), 8)
                slab_o[at, :] = jnp.where(mine, got, off)
                return carry

            lax.fori_loop(0, tiles, slab, 0, unroll=True)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem] * 4,
            out_specs=(vmem, vmem, vmem),
            grid=(),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((nc, LANES), i32),
            jax.ShapeDtypeStruct((nc, LANES), i32),
            jax.ShapeDtypeStruct((ep * nc, LANES), i32),
        ),
        interpret=interpret,
        name=f"moe_push_pair_rows_ep{ep}_n{n}",
    )


def build_pair_rows(
    exchange_out, recv_base, mirror_base, run_base, me, *, ep, t_local, topk,
    e_total, shard_stride, interpret=False,
):  # fmt: skip
    """host.routing_pairs_from_tables' (pos, mirror_pos, slab_row) for a step
    wide_pair_tables_apply admits, in one kernel: from the exchanged payload
    (`exchange_out`, the exchange kernel's raw [ep, rows, 128] output, every
    source's build_local_ranks words in its leading rows) and the table
    kernel's plan tables (`recv_base`/`mirror_base` [e_total], `run_base`
    [ep, g]). Same integers."""
    g = e_total // ep
    n = t_local * topk
    ke = _align(e_total, LANES)
    call = _build_pair_rows(
        ep, g, n, e_total, off_shard_slab_row(shard_stride, ep), shard_stride,
        interpret,
    )  # fmt: skip
    pad = ((0, ke - e_total),)
    pos, mir, slab_row = call(
        jnp.asarray(me, jnp.int32).reshape(1),
        exchange_out,
        jnp.pad(recv_base.astype(jnp.int32), pad).reshape(1, ke),
        jnp.pad(mirror_base.astype(jnp.int32), pad).reshape(1, ke),
        run_base.astype(jnp.int32),
    )
    return pos.reshape(t_local, topk), mir.reshape(t_local, topk), slab_row.reshape(-1)

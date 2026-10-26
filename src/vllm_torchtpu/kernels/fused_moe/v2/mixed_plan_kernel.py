# SPDX-License-Identifier: Apache-2.0
"""The MIXED-step routing plan as one Pallas call.

At a mixed bucket (more than SMALL_BATCH_LOCAL_TOKENS local tokens) the
sharded routing plan is ~100 XLA ops over a few thousand integers plus two
SparseCore scatters (the token-gather table and the activation-scale slab)
that the TensorCore waits on, and which the layer orders behind the
activation all-gather because they share its SparseCore. Here every kernel
operand comes out of one call over the gathered routing exchange, with no
scatter at all, so the plan can run on the TensorCore while the activation
all-gather is in flight.

The decode plan (plan_kernel.py) does not stretch to these shapes: its pair
loop is unrolled O(P) and its scatter is O(P) more. This kernel replaces both
with matrix forms:

* counts come from the exchange's count section (no histogram);
* the small per-shard tables are the decode kernel's matmul prefix sums;
* the combine tables (pos, mirror_pos) rank each pair inside its routing
  block with one triangular matmul per block plus running block offsets, and
  look all three tables up in ONE byte-planed dot;
* the slab tables are built as a GATHER: slab row r lies in run k(r) (runs
  in (local expert, source) order), at offset j(r) inside it, and its token
  is the position of the (j+1)-th pair of run k(r)'s source that chose run
  k(r)'s expert. With C[k, t] = that run's pairs before token t (a matmul
  prefix), token(r) = #{t : C[k(r), t] <= j(r)} - 1. The lookup of row k(r)
  of C is TELESCOPED: A[k, r] = [start_k <= r] is a prefix of ones over k
  for nondecreasing run starts, so (D^T @ A)[:, r] = V[k(r)] when D holds the
  first differences of V over k. Zero-length runs drop out of the sum. The
  scale of row r is telescoped the same way over tokens: sum_t [t <= tok] *
  (sb[t] - sb[t-1]) = sb[tok], in wrapping int32 byte planes.

Every product is a bf16 product of an integer below 2**8 in magnitude and a
0/1 (or -1/0/1) value, and every f32 partial stays below 2**24, so the
integer results are exact and order independent.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    ROWBLK,
    act_scale_slab_rows,
    gather_clamp_bounds,
    pow2_shift,
    token_gather_window_rows,
)
from vllm_torchtpu.kernels.fused_moe.v2.plan_kernel import (
    DecodePlan,
    _bf16,
    _iota,
    _mm,
    _mm_nt,
)

_MAX_TABLE_VALUE = 1 << 16
# Slab rows per S3 block: 8 lane-dense rows of 128 = one aligned [8, 128] tile.
_BLOCK_ROWS = 8 * HIDDEN_LANE_BLOCK


def _lane_rows(x, n, lanes=HIDDEN_LANE_BLOCK):
    """[1, n * lanes] -> [n, lanes] by masked selects (no lane->sublane reshape)."""
    sub = _iota((n, lanes), 0)
    out = jnp.zeros((n, lanes), x.dtype)
    for i in range(n):
        out = jnp.where(sub == i, x[:, i * lanes : (i + 1) * lanes], out)
    return out


def _mixed_plan_kernel(
    me_ref,
    ex_ref,
    *refs,
    e_total,
    ep,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    row_stride,
    scale_at,
    count_at,
    gather_hi,
    gather_units,
    scale_rows,
    alloc_units,
    rank_mode,
    dc_hi,
    fault,
):
    (rs_h, al_h, co_h, pd_h, tr_h, tg_h, ss_h, er_h, sb_h, vi_h, cn_h, pos_h, mir_h) = refs[:13]
    (rs_s, al_s, co_s, pd_s, tr_s, tg_s, ss_s, er_s, sb_s, vi_s, cn_s, pos_s, mir_s) = refs[13:26]
    sem = refs[26]
    E, G, P = e_total, e_total // ep, t_local * topk
    epp = -(-ep // 8) * 8
    me = me_ref[0]
    first = me * G
    copies = []

    def ship(src, dst):
        c = pltpu.make_async_copy(src, dst, sem)
        c.start()
        copies.append(c)

    skip = set(fault[5:].split(",")) if fault and fault.startswith("skip:") else set()
    # ---------------------------------------------------------------- S1
    counts = ex_ref[:, count_at : count_at + E]  # [ep, E]
    aligned = (counts + (ROWBLK - 1)) & jnp.int32(-ROWBLK)

    def stage1():
        lower_d = _bf16(_iota((ep, ep), 1) < _iota((ep, ep), 0))
        upper_e = _bf16(_iota((E, E), 0) < _iota((E, E), 1))
        same_shard = _iota((E, E), 0) // G == _iota((E, E), 1) // G
        upper_seg = _bf16((_iota((E, E), 0) < _iota((E, E), 1)) & same_shard)
        run_start = _mm(lower_d, aligned)  # [d, e] exclusive over sources
        recv_base = _mm(aligned, upper_e)  # [d, e] exclusive over experts
        expert_rows_all = jnp.broadcast_to(jnp.sum(aligned, axis=0, keepdims=True), (8, E))
        local_base = _mm(expert_rows_all, upper_seg)
        shift = pow2_shift(tile_m, "tile_m")
        last = run_start + jnp.maximum(aligned, 1) - 1
        tiles = jnp.where(
            aligned > 0,
            jnp.right_shift(last, shift) - jnp.right_shift(run_start, shift) + 1,
            0,
        )
        sel = _bf16(_iota((G, E), 1) == first + _iota((G, E), 0))
        sel_t = _bf16(_iota((E, G), 0) == first + _iota((E, G), 1))
        run_start_g = _mm_nt(sel, run_start)
        aligned_g = _mm_nt(sel, aligned)
        counts_g = _mm_nt(sel, counts)
        recv_base_g = _mm_nt(sel, recv_base)
        tiles_g = _mm_nt(sel, tiles)
        lower_g = _bf16(_iota((G, G), 1) < _iota((G, G), 0))
        upper_d = _bf16(_iota((ep, ep), 0) < _iota((ep, ep), 1))
        out_total = jnp.broadcast_to(jnp.sum(aligned_g, axis=0, keepdims=True), (8, ep))
        contrib_off = _mm(out_total, upper_d)[0:1] + _mm(lower_g, aligned_g)
        rb = ROWBLK.bit_length() - 1
        if fault == "s1_contrib":  # positive control: contribution offsets skip the source term
            contrib_off = _mm(out_total, upper_d)[0:1] + 0 * contrib_off
        rs_s[...] = jnp.right_shift(run_start_g, rb)
        al_s[...] = jnp.right_shift(aligned_g, rb)
        co_s[...] = jnp.right_shift(contrib_off, rb)
        pd_s[...] = jnp.right_shift(recv_base_g, rb)
        tr_s[...] = counts_g
        rows_row = _mm(expert_rows_all, sel_t)[0:1]
        er_s[...] = rows_row
        sb_s[...] = _mm(local_base, sel_t)[0:1]
        rows_col = _mm_nt(sel, expert_rows_all)[:, 0:1]
        key_row = _iota((1, G), 1) - rows_row * G
        key_col = _iota((G, 1), 0) - rows_col * G
        rank_col = jnp.sum((key_row < key_col).astype(jnp.int32), axis=1, keepdims=True)
        perm = jnp.sum(
            jnp.where(rank_col == _iota((G, G), 1), _iota((G, G), 0), 0),
            axis=0,
            keepdims=True,
        )
        vi_s[...] = jnp.minimum(perm, G - 1)
        ones = _bf16(jnp.ones((8, E), jnp.int32))
        lane_d = _iota((1, ep), 1)
        mine = (lane_d == me).astype(jnp.int32)
        other = 1 - mine

        def over_g(x):
            return jnp.sum(x, axis=0, keepdims=True)

        recv_rows_all = _mm_nt(ones, counts)[0:1]
        recv_total = _mm_nt(ones, aligned)[0:1]
        recv_tiles_all = _mm_nt(ones, tiles)[0:1]
        n_visit = jnp.sum((rows_row > 0).astype(jnp.int32), axis=1, keepdims=True)
        for row, value in enumerate(
            (
                other * over_g(counts_g),
                mine * (recv_rows_all - over_g(counts_g)),
                other * over_g(aligned_g),
                mine * (recv_total - over_g(aligned_g)),
                mine * n_visit,
                other * over_g(tiles_g),
                mine * (recv_tiles_all - over_g(tiles_g)),
                mine * over_g(tiles_g),
            )
        ):
            cn_s[row : row + 1, :] = value

        return run_start, recv_base

    if "s1" in skip:
        run_start, recv_base = aligned, aligned
    else:
        run_start, recv_base = stage1()
    for s, h in ((rs_s, rs_h), (al_s, al_h), (co_s, co_h), (pd_s, pd_h), (tr_s, tr_h),
                 (er_s, er_h), (sb_s, sb_h), (vi_s, vi_h), (cn_s, cn_h)):
        ship(s, h)

    # ---------------------------------------------------------------- S2
    shift = pow2_shift(tile_m, "tile_m")
    def stage2():
        mine_d = _iota((ep, E), 0) == me
        recv_base_me = jnp.sum(jnp.where(mine_d, recv_base, 0), axis=0, keepdims=True)
        run_start_me = jnp.sum(jnp.where(mine_d, run_start, 0), axis=0, keepdims=True)
        my_pairs = ex_ref[pl.ds(me, 1), :][:, 0:P]  # [1, P]
        nbins = E + 8
        onehot = (_iota((nbins, P), 0) == my_pairs).astype(jnp.int32)  # [bin, p]
        onehot_b = _bf16(onehot)
        # All three lookup tables, two byte planes each, in one [8, E] LHS.
        tabs = (
            recv_base_me,
            jnp.right_shift(recv_base_me, shift) + 2 * _iota((1, E), 1),
            run_start_me & (tile_m - 1),
        )
        sub8 = _iota((8, E), 0)
        lhs = jnp.zeros((8, E), jnp.int32)
        for i, tab in enumerate(tabs):
            lhs = jnp.where(sub8 == 2 * i, jnp.right_shift(tab, 8) & 255, lhs)
            lhs = jnp.where(sub8 == 2 * i + 1, tab & 255, lhs)
        look = lax.dot_general(
            _bf16(lhs), onehot_b[:E], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)  # [8, P]
        base_pos = look[0:1] * 256 + look[1:2]
        base_mir = (look[2:3] * 256 + look[3:4]) * tile_m + look[4:5] * 256 + look[5:6]
        valid_bin = _iota((nbins, 1), 0) < E
        if rank_mode == "block":
            # Rank inside each routing block (one triangular dot per block) plus
            # the running count of earlier blocks for routed ids; an id outside
            # [0, E) keeps its block-local rank, as in the sharded builder.
            lstrict = _bf16(_iota((block, block), 0) < _iota((block, block), 1))
            ones_b = _bf16(jnp.ones((block, HIDDEN_LANE_BLOCK), jnp.int32))
            off = jnp.zeros((nbins, 1), jnp.int32)
            for b in range(P // block):
                oh = onehot_b[:, b * block : (b + 1) * block]
                pre = lax.dot_general(
                    oh, lstrict, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
                ).astype(jnp.int32) + jnp.where(valid_bin, off, 0)
                rk = jnp.sum(onehot[:, b * block : (b + 1) * block] * pre, axis=0, keepdims=True)
                if fault == "s2_rank":  # positive control: block offsets dropped from the rank
                    rk = jnp.sum(onehot[:, b * block : (b + 1) * block] * (pre - jnp.where(valid_bin, off, 0)), axis=0, keepdims=True)
                lanes = slice(b * block, (b + 1) * block)
                pos_s[:, lanes] = base_pos[:, lanes] + rk
                mir_s[:, lanes] = base_mir[:, lanes] + rk
                hist = lax.dot_general(
                    oh, ones_b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                off = off + hist[:, 0:1]
        else:  # "pp": the decode kernel's [P, P] form (small P, block < 128)
            before = _iota((P, P), 0) < _iota((P, P), 1)
            same_block = _iota((P, P), 0) // block == _iota((P, P), 1) // block

            def rank_among(earlier):
                prior = _mm(onehot, _bf16(earlier))
                return jnp.sum(onehot * prior, axis=0, keepdims=True)

            valid = (my_pairs >= 0) & (my_pairs < E)
            rk = jnp.where(valid, rank_among(before), rank_among(before & same_block))
            pos_s[...] = base_pos + rk
            mir_s[...] = base_mir + rk


    if "s2" not in skip:
        stage2()
    ship(pos_s, pos_h)
    ship(mir_s, mir_h)

    # ---------------------------------------------------------------- S3
    def stage3():
        # Runs k = g * ep + d in slab order; their counts / aligned rows as rows.
        lane_k = _iota((1, E), 1)
        sel_k = _bf16(_iota((E, E), 0) == first + _iota((E, E), 1) // ep)  # [e, k]
        by_k = _mm(counts, sel_k)  # [d, k] = counts[d, first + k // ep]
        pick_d = _iota((ep, E), 0) == _iota((ep, E), 1) % ep
        cnt_row = jnp.sum(jnp.where(pick_d, by_k, 0), axis=0, keepdims=True)  # [1, E]
        al_row = (cnt_row + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
        ustrict = _bf16(_iota((E, E), 0) < _iota((E, E), 1))  # [k', k] k' < k
        al8 = jnp.broadcast_to(al_row, (8, E))
        start_row = _mm(al8, ustrict)[0:1]  # [1, E]
        start_col = _mm_nt(_bf16(_iota((E, E), 1) < _iota((E, E), 0)), al8)[:, 0:1]  # [E, 1]
        total_rows = jnp.sum(al_row)

        # C^T[t, k]: run k's pairs in tokens before t.
        pairs_all = ex_ref[:, 0:P]  # [ep, P]
        g_col = _iota((G, 1, 1), 0) + first
        if ep % 16 == 0:
            hit_gd = pairs_all[None, :, :] == g_col  # [G, ep, P]  # whole bf16 tiles: merge the leading dim in bf16
            m = hit_gd.astype(jnp.bfloat16).reshape(E, P)
        else:  # rows k = source k % ep, replicated by a 0/1 dot (no sub-tile reshape)
            rep = _bf16(_iota((E, ep), 1) == _iota((E, ep), 0) % ep)
            pk = _mm(rep, pairs_all)  # [E, P] = pairs[k % ep, p]
            m = _bf16(pk == first + _iota((E, 1), 0) // ep)
        q = _bf16(_iota((t_local, P), 1) // topk == _iota((t_local, P), 0))
        s_t = lax.dot_general(q, m, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        lower_t = _bf16(_iota((t_local, t_local), 1) < _iota((t_local, t_local), 0))
        c_t = lax.dot_general(
            lower_t, s_t.astype(jnp.bfloat16), (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)  # [t, E], exact (S <= topk is bf16-exact)
        # V's scalar rows: start, count, source.
        sub8e = _iota((8, E), 0)
        v_s = jnp.where(sub8e == 0, start_row, 0)
        v_s = jnp.where(sub8e == 1, cnt_row, v_s)
        v_s = jnp.where(sub8e == 2, lane_k % ep, v_s)
        shift_k = _bf16(_iota((E, E), 0) == _iota((E, E), 1) - 1)  # [k', k] k' = k - 1
        d_c = c_t - _mm(c_t, shift_k)
        d_s = v_s - _mm(v_s, shift_k)
        if fault == "tele_off":  # positive control: differences taken without V[k-1]
            d_c = c_t
        # |d_c| <= 255 when no token selects an expert twice: one signed plane.
        hi_c = jnp.right_shift(d_c, 8)
        if fault == "sign":  # positive control: high byte of a negative difference unsigned
            hi_c = lax.shift_right_logical(d_c, 8) & 255
        planes = [d_c & 255, hi_c] if dc_hi else [d_c]
        planes += [d_s & 255, jnp.right_shift(d_s, 8)]
        lhs_y = _bf16(jnp.concatenate(planes, axis=0))  # [(1|2) t + 16, E]
        nc = len(planes) - 2

        # Scale: first differences over tokens of each source's bits, byte planes.
        sb = ex_ref[:, scale_at : scale_at + t_local]  # [ep, t] i32
        shift_t = _bf16(_iota((t_local, t_local), 0) == _iota((t_local, t_local), 1) - 1)
        prev = jnp.zeros_like(sb)
        for qb in range(4):
            byte = _bf16(jnp.right_shift(sb, 8 * qb) & 255)
            r = lax.dot_general(
                byte, shift_t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            ).astype(jnp.int32)
            prev = prev + jnp.left_shift(r, 8 * qb)
        dsb = sb - prev
        # [4 epp, t]: row qb * epp + d = byte qb of dsb[d] (epp = ep rounded up to a tile).
        lhs_s = jnp.zeros((4 * epp, t_local), jnp.float32)
        for qb in range(4):
            pick = _bf16(_iota((4 * epp, ep), 0) == qb * epp + _iota((4 * epp, ep), 1))
            lhs_s = lhs_s + lax.dot_general(
                pick, _bf16(jnp.right_shift(dsb, 8 * qb) & 255), (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        lhs_s = lhs_s.astype(jnp.bfloat16)

        tg_s[...] = jnp.zeros_like(tg_s)
        ss_s[...] = jnp.zeros_like(ss_s)
        lane_r = _iota((1, _BLOCK_ROWS), 1)
        sub_d = _iota((epp, _BLOCK_ROWS), 0)
        k_sub = start_col  # [E, 1]

        def blk(b, carry):
            r = b * _BLOCK_ROWS + lane_r  # [1, BR]
            a_t = _bf16(k_sub <= r)  # [E, BR] prefix ones over k
            if fault == "rm_ydot":  # timing-only removal arm: the telescoped dot
                y = jnp.broadcast_to(r + jnp.sum(a_t[0:8].astype(jnp.int32)), (lhs_y.shape[0], _BLOCK_ROWS))
            else:
                y = lax.dot_general(
                    lhs_y, a_t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
                ).astype(jnp.int32)
            o = nc * t_local
            if dc_hi:
                yc = y[t_local : 2 * t_local] * 256 + y[0:t_local]
            else:
                yc = y[0:t_local]
            ys = y[o + 8 : o + 16] * 256 + y[o : o + 8]
            start, cnt, src = ys[0:1], ys[1:2], ys[2:3]
            j = r - start
            mask = yc <= j  # [t, BR], a prefix of ones over t
            n_le = jnp.sum(mask.astype(jnp.int32), axis=0, keepdims=True)
            valid = j < cnt
            token = jnp.clip(src * row_stride + n_le - 1, 0, gather_hi)
            token = jnp.where(valid, token, 0)
            sc = lax.dot_general(
                lhs_s, _bf16(mask), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            ).astype(jnp.int32)  # [4 ep, BR]
            hit = sub_d == src
            bits = jnp.zeros_like(token)
            for qb in range(4):
                part = jnp.sum(jnp.where(hit, sc[qb * epp : (qb + 1) * epp], 0), axis=0, keepdims=True)
                bits = bits + jnp.left_shift(part, 8 * qb)
            bits = jnp.where(valid, bits, 0)
            u0 = pl.multiple_of(b * 8, 8)
            tg_s[pl.ds(u0, 8), :] = _lane_rows(token, 8)
            ss_s[pl.ds(u0, 8), :] = lax.bitcast_convert_type(_lane_rows(bits, 8), jnp.float32)
            return carry

        n_blocks = (total_rows + (_BLOCK_ROWS - 1)) // _BLOCK_ROWS
        if fault == "stale_block":  # positive control: last live block left zero
            n_blocks = n_blocks - 1
        if fault == "rm_s3":  # timing-only removal arm: the whole slab loop
            n_blocks = n_blocks * 0
        lax.fori_loop(0, n_blocks, blk, 0)

    if "s3" not in skip:
        stage3()
    else:
        tg_s[...] = jnp.zeros_like(tg_s)
        ss_s[...] = jnp.zeros_like(ss_s)
    ship(tg_s.at[pl.ds(0, gather_units), :], tg_h)
    ship(ss_s.at[pl.ds(0, scale_rows), :], ss_h)
    for c in copies:
        c.wait()


def mixed_plan_supported(*, e_total, ep, t_local, topk, shard_stride, tile_m, block):
    """Whether build_mixed_plan covers this shape exactly."""
    G, P = e_total // ep, t_local * topk
    gather_rows = shard_stride + token_gather_window_rows(tile_m)
    return (
        e_total % ep == 0
        and G % 8 == 0
        and e_total % 8 == 0
        and t_local % 16 == 0
        and t_local <= 256
        and (block % HIDDEN_LANE_BLOCK == 0 or P <= 512)
        and gather_rows % HIDDEN_LANE_BLOCK == 0
        and 2 * (shard_stride + 2 * e_total) < _MAX_TABLE_VALUE
    )


def build_mixed_plan(
    exchange,
    rank,
    *,
    layout,
    e_total,
    ep,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    row_stride=None,
    dc_hi=True,
    fault=None,
    interpret=False,
):
    """Shard ``rank``'s routing-plan operands from the gathered exchange.

    exchange [ep, width] i32 is the gathered routing payload of
    layer._pack_routing_exchange with ``layout`` = (scale offset, count
    offset, width), quantized (it must carry row scales). Returns a
    DecodePlan holding the values of build_routing_tables_sharded and the
    shard_* cutters bit for bit (token_gather already padded by its window).

    dc_hi: carry the high byte plane of the run-prefix differences. Required
    for exactness when a token can select the same expert twice (prefix
    counts above 255); off only for a timing arm.
    fault: positive-control fault injection for the accuracy test only.
    """
    scale_at, count_at, width = layout
    if scale_at == count_at:
        raise ValueError("the mixed plan needs quantized activations (row scales)")
    if not mixed_plan_supported(
        e_total=e_total, ep=ep, t_local=t_local, topk=topk,
        shard_stride=shard_stride, tile_m=tile_m, block=block,
    ):
        raise ValueError(
            f"the mixed plan does not cover e_total={e_total} ep={ep} t_local={t_local} "
            f"topk={topk} shard_stride={shard_stride} block={block}"
        )
    G, P = e_total // ep, t_local * topk
    row_stride = t_local if row_stride is None else row_stride
    gather_rows = shard_stride + token_gather_window_rows(tile_m)
    gather_units = gather_rows // HIDDEN_LANE_BLOCK
    scale_rows = act_scale_slab_rows(shard_stride)
    T = ep * t_local
    _, gather_hi = gather_clamp_bounds(
        (T - 1) // t_local * row_stride + (T - 1) % t_local + 1
    )
    max_blocks = -(-shard_stride // _BLOCK_ROWS)
    alloc_units = max(8 * max_blocks, -(-max(gather_units, scale_rows) // 8) * 8)
    rank_mode = "block" if block % HIDDEN_LANE_BLOCK == 0 else "pp"
    kernel = functools.partial(
        _mixed_plan_kernel,
        e_total=e_total, ep=ep, t_local=t_local, topk=topk, block=block, tile_m=tile_m,
        shard_stride=shard_stride, row_stride=row_stride, scale_at=scale_at,
        count_at=count_at, gather_hi=gather_hi, gather_units=gather_units,
        scale_rows=scale_rows, alloc_units=alloc_units, rank_mode=rank_mode, dc_hi=dc_hi,
        fault=fault,
    )
    i32, f32 = jnp.int32, jnp.float32
    hbm_shapes = [(G, ep)] * 5 + [(gather_units, HIDDEN_LANE_BLOCK), (scale_rows, HIDDEN_LANE_BLOCK)]
    hbm_shapes += [(1, G)] * 3 + [(8, ep), (1, P), (1, P)]
    dtypes = [i32] * 6 + [f32] + [i32] * 6
    scratch_shapes = [(G, ep)] * 5 + [(alloc_units, HIDDEN_LANE_BLOCK)] * 2
    scratch_shapes += [(1, G)] * 3 + [(8, ep), (1, P), (1, P)]
    vmem = pltpu.MemorySpace.VMEM
    space = vmem if interpret else pltpu.MemorySpace.HBM
    outs = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=vmem)],
            out_specs=[pl.BlockSpec(memory_space=space) for _ in hbm_shapes],
            scratch_shapes=[pltpu.VMEM(s, d) for s, d in zip(scratch_shapes, dtypes)]
            + [pltpu.SemaphoreType.DMA],
        ),
        out_shape=[jax.ShapeDtypeStruct(s, d) for s, d in zip(hbm_shapes, dtypes)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        interpret=interpret,
        name=f"fused_ep_moe_v2_mixed_plan_t{t_local}",
    )(rank.reshape(1).astype(i32), exchange.astype(i32).reshape(ep, width))
    rs, al, co, pd, tr, tg, ss, er, sb, vi, cn, pos, mir = outs
    return DecodePlan(
        block_tables=(rs, al, co, pd),
        true_rows=tr,
        token_gather=tg.reshape(gather_rows),
        scale_slab=ss,
        expert_rows=er.reshape(G),
        slab_base=sb.reshape(G),
        visit=vi.reshape(G),
        counts=cn,
        pos=pos.reshape(t_local, topk),
        mirror_pos=mir.reshape(t_local, topk),
    )

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
"""Fused EP MoE v2 routing exchange and routing plan as one TensorCore kernel.

Replaces the XLA stage between top-k and the fused kernel (the all-gather of
each shard's routing payload, the sharded routing plan, and the token-gather
and scale-slab scatters) and returns the same RoutePlan as
layer.xla_route_plan's default path, bit for bit.

1. Exchange: after an all-pairs barrier, each shard DMAs its payload (expert
   ids, row scales, per-expert counts) into row `me` of every peer's VMEM
   buffer.
2. Per-expert tables: vector passes over the count table transposed to
   [experts, ep]. Prefix sums and the pick of this shard's G experts (a one-hot
   matmul, so G needs no 8-alignment) run on the MXU. Every bf16 operand is an
   integer bf16 holds exactly (<= 256, or a power of two), and f32 accumulates
   them exactly.
3. Own selections: the arrival row and mirror position of each (token, k).
4. Slab tables: a vector pass bitmasks the gathered pairs routed to this
   shard's experts (about 1/ep of them). The scalar core walks the set bits,
   writes each pair's token number and row scale into SMEM, and copies out
   only the live rows. The aliased, zero-filled outputs keep zeros past them,
   as the scatter does.
"""

import threading
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    ROWBLK,
    ROWBLK_SHIFT,
    act_scale_slab_rows,
    align_up,
    pow2_shift,
    token_gather_window_rows,
)

# The fused kernel holds 0; DeepSeek-V4's attention kernels 3 and 4. A barrier
# semaphore is per collective id, so this kernel's barrier never counts another
# kernel's signals.
ROUTE_PLAN_COLLECTIVE_ID = 5

LANES = HIDDEN_LANE_BLOCK
# bf16 holds every integer in [0, 256] exactly; every MXU operand below is one.
BF16_EXACT_MAX = 256
# f32 accumulates integers exactly below 2**24.
F32_EXACT_MAX = 1 << 24
WORD_BITS = 32
# The kernel's SMEM scratch (the gathered pairs and the two slab tables) grows
# with ep * t_local * topk; a shape over the chip's SMEM fails to compile, so
# route_plan_supported refuses it and the layer keeps XLA. The headroom covers
# what route_plan_smem_bytes leaves out (register spill slots, small-array
# padding, the rank prefetch): at EP32, t_local=256 on v7x (1 MiB SMEM) the
# compiler reported 5.2 KiB over against 1.6 KiB estimated.
SMEM_HEADROOM_BYTES = 32 << 10

# DMA semaphore slots.
_SEM_SEND, _SEM_RECV, _SEM_OWN, _SEM_ZERO, _SEM_PAIRS, _SEM_STAGE, _SEM_OUT = range(7)
_N_SEMS = 7


class RoutePlan(NamedTuple):
    """What the fused kernel and the combine take, as the XLA path builds it."""

    # (run_start, aligned_rows, contrib_off, push_dst) in ROWBLK blocks and the
    # true run rows, each [G, ep] i32: the kernel's transport/push tables.
    kernel_tables: tuple
    # [shard_stride + token_gather_window_rows(tile_m)] i32, padded.
    token_gather: jax.Array
    # [G] i32 each: rows and slab base of this shard's experts.
    expert_rows: jax.Array
    slab_base: jax.Array
    # [act_scale_slab_rows(shard_stride), 128] f32 or None (unquantized rows).
    scale_slab: jax.Array | None
    # [G] i32: the expert visit list.
    visit: jax.Array
    # [N_COUNTS, ep] i32.
    counts: jax.Array
    # [t_local, topk] i32 each: arrival row and mirror position per selection.
    pos: jax.Array
    mirror_pos: jax.Array


def route_plan_smem_bytes(*, ep, e_total, t_local, topk, shard_stride):
    """SMEM the kernel's scratch and scalar prefetches take, in bytes.

    Mirrors _build_route_plan's SMEM scratch_shapes plus the tokp prefetch.
    The pairs row width is the quantized one (the lane-aligned pair count plus
    the lane-aligned per-token scales), which is at least the unquantized TKp.
    """
    G = e_total // ep
    TKp = align_up(t_local * topk, LANES)
    psw = TKp + align_up(t_local, LANES)
    g8 = align_up(G, 8)
    st_rows = align_up(ep, 8) + g8 + 8
    tg_rows = shard_stride // LANES
    sc_rows = act_scale_slab_rows(shard_stride)
    words = ep * psw + (st_rows + tg_rows + sc_rows) * LANES + g8 + TKp
    return words * np.dtype(np.int32).itemsize


def route_plan_supported(
    *, ep, e_total, t_local, topk, block, tile_m, shard_stride, info=None
):
    """Whether the kernel covers this shape; the layer keeps XLA otherwise.

    The SMEM budget is this chip's (pltpu.get_tpu_info()) unless `info` is
    given, which is read only once every other condition holds.
    """
    T = ep * t_local

    def smem_budget():
        chip = info if info is not None else pltpu.get_tpu_info()
        return chip.smem_capacity_bytes - SMEM_HEADROOM_BYTES

    return bool(
        1 < ep <= LANES
        and e_total % ep == 0
        # A count is the rows one shard sends one expert. A token selects an
        # expert at most once, so that is at most t_local; aligned up to
        # ROWBLK it is still a bf16-exact MXU operand.
        and align_up(t_local, ROWBLK) <= BF16_EXACT_MAX
        # The live-pair bitmask is one row of LANES words per source, so it
        # holds at most LANES * WORD_BITS of this shard's pairs.
        and t_local * topk <= LANES * WORD_BITS
        and T * topk + ROWBLK * e_total < F32_EXACT_MAX
        and block & (block - 1) == 0
        and tile_m & (tile_m - 1) == 0
        and tile_m <= BF16_EXACT_MAX
        and shard_stride % LANES == 0
        and token_gather_window_rows(tile_m) % LANES == 0
        and route_plan_smem_bytes(
            ep=ep,
            e_total=e_total,
            t_local=t_local,
            topk=topk,
            shard_stride=shard_stride,
        )
        <= smem_budget()
    )


def _bit_index(lsb):
    """log2 of a one-hot int32 word (bit 31 included), branch-free."""
    b = jnp.where((lsb & jnp.int32(-65536)) != 0, 16, 0)  # 0xFFFF0000
    b = b + jnp.where((lsb & jnp.int32(-16711936)) != 0, 8, 0)  # 0xFF00FF00
    b = b + jnp.where((lsb & jnp.int32(-252645136)) != 0, 4, 0)  # 0xF0F0F0F0
    b = b + jnp.where((lsb & jnp.int32(-858993460)) != 0, 2, 0)  # 0xCCCCCCCC
    b = b + jnp.where((lsb & jnp.int32(-1431655766)) != 0, 1, 0)  # 0xAAAAAAAA
    return b


def _as_bf16(a):
    """int32 -> bf16 through f32; exact for the small integers used here."""
    return a.astype(jnp.float32).astype(jnp.bfloat16)


def _exact_dot(lhs, rhs):
    """int32 lhs @ rhs through bf16, both holding small exact integers."""
    return jnp.dot(
        _as_bf16(lhs), _as_bf16(rhs), preferred_element_type=jnp.float32
    ).astype(jnp.int32)


def _exact_dot_nt(lhs, rhs):
    """int32 lhs @ rhs.T through bf16 (contract both minor dims)."""
    return lax.dot_general(
        _as_bf16(lhs),
        _as_bf16(rhs),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)


def _build_route_plan(
    *, ep, e_total, t_local, topk, block, tile_m, shard_stride, layout, quantized
):
    G = e_total // ep
    TK = t_local * topk
    TKp = align_up(TK, LANES)
    scale_at, count_at, width = layout
    width_p = align_up(width, LANES)
    Ep = align_up(e_total, LANES)
    width_g = max(width_p, count_at + Ep)
    # Columns of the gathered payload the scalar loop reads: the pairs and,
    # where the rows are quantized, the row scales after them: one per token,
    # so t_local of them, which is more than one lane block past 128 tokens.
    psw = scale_at + align_up(t_local, LANES) if quantized else TKp
    n_words = -(-TK // WORD_BITS)
    ep8 = align_up(ep, 8)
    g8 = align_up(G, 8)
    st_t2 = ep8
    st_misc = ep8 + g8
    st_rows = st_misc + 8
    tg_rows = shard_stride // LANES
    tg_out_rows = (shard_stride + token_gather_window_rows(tile_m)) // LANES
    sc_rows = act_scale_slab_rows(shard_stride)
    tile_shift = pow2_shift(tile_m, "tile_m")
    block_shift = pow2_shift(block, "block")
    mesh_id = pl.DeviceIdType.MESH

    def lane(shape):
        return lax.broadcasted_iota(jnp.int32, shape, len(shape) - 1)

    def sub(shape):
        return lax.broadcasted_iota(jnp.int32, shape, 0)

    def align_rows(x):
        return (x + (ROWBLK - 1)) & jnp.int32(-ROWBLK)

    def kernel(
        rank_ref,
        tokp_ref,
        payload_ref,
        tg_zero_hbm,
        sc_zero_hbm,
        rs_ref,
        al_ref,
        co_ref,
        pd_ref,
        tr_ref,
        er_ref,
        sb_ref,
        visit_ref,
        counts_ref,
        pos_ref,
        mir_ref,
        tg_hbm,
        sc_hbm,
        gbuf,
        stage,
        zbuf,
        pairs_sm,
        stage_sm,
        tg_sm,
        sc_sm,
        cnt_sm,
        sems,
    ):
        del tg_zero_hbm, sc_zero_hbm  # aliased by tg_hbm, sc_hbm
        me = rank_ref[0]
        first = me * G

        # -- 0. Zero the SMEM slab windows, under the exchange. -------------
        zbuf[...] = jnp.zeros(zbuf.shape, jnp.int32)
        zeroing = [
            pltpu.make_async_copy(zbuf.at[pl.ds(0, tg_rows)], tg_sm, sems.at[_SEM_ZERO])
        ]
        if quantized:
            zeroing.append(pltpu.make_async_copy(zbuf, sc_sm, sems.at[_SEM_ZERO]))
        for c in zeroing:
            c.start()

        # -- 1. The exchange: row `me` of every shard's buffer. ---------------
        own_row = gbuf.at[pl.ds(me, 1), pl.ds(0, width_p)]
        own = pltpu.make_async_copy(payload_ref, own_row, sems.at[_SEM_OWN])
        own.start()
        barrier = pltpu.get_barrier_semaphore()
        # Peers in rotated order, so no shard is everyone's first target.
        peers = [lax.rem(me + i, jnp.int32(ep)) for i in range(1, ep)]
        for d in peers:
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=mesh_id)
        pl.semaphore_wait(barrier, ep - 1)
        sends = []
        for d in peers:
            c = pltpu.make_async_remote_copy(
                payload_ref,
                own_row,
                sems.at[_SEM_SEND],
                sems.at[_SEM_RECV],
                device_id=(d,),
                device_id_type=mesh_id,
            )
            c.start()
            sends.append(c)
        for c in sends:
            c.wait_recv()
        for c in sends:
            c.wait_send()
        own.wait()

        # The pairs (and row scales) go to SMEM for the scalar placement while
        # the vector unit builds the tables.
        pairs_copy = pltpu.make_async_copy(
            gbuf.at[:, pl.ds(0, psw)], pairs_sm, sems.at[_SEM_PAIRS]
        )
        pairs_copy.start()

        # -- 2. Per-expert tables, transposed: expert on sublanes. ------------
        cnt = gbuf[:, pl.ds(count_at, Ep)]  # [ep, Ep]: rows_by_dest
        cnt = jnp.where(lane((ep, Ep)) < e_total, cnt, 0)
        if ep < LANES:
            cnt = jnp.concatenate([cnt, jnp.zeros((LANES - ep, Ep), jnp.int32)], 0)
        ct = cnt.T  # [Ep, 128]: run_rows[e, d]
        at = align_rows(ct)  # run_rows_aligned
        u = (sub((LANES, LANES)) < lane((LANES, LANES))).astype(jnp.int32)
        rsa = _exact_dot(at, u)  # run_start_aligned: prefix over d

        def run_tiles(a, rs):
            """mirror_run_tiles: the tiles a padded run of a rows at rs spans."""
            last = rs + jnp.maximum(a, 1) - 1
            return jnp.where(
                a > 0,
                jnp.right_shift(last, tile_shift) - jnp.right_shift(rs, tile_shift) + 1,
                0,
            )

        recv_rows = jnp.sum(at, axis=0, keepdims=True)  # [1, 128]
        recv_rows_all = jnp.sum(ct, axis=0, keepdims=True)
        recv_tiles_all = jnp.sum(run_tiles(at, rsa), axis=0, keepdims=True)
        # recv_base[d, me, 0]: every row of the experts before this shard's.
        pre = jnp.sum(jnp.where(sub((Ep, LANES)) < first, at, 0), axis=0, keepdims=True)
        # This shard's G experts, rows first .. first + G, by a one-hot pick:
        # a dynamic sublane window would need first, so G, 8-aligned. Only the
        # counts are picked; they are <= t_local, so bf16-exact. The run starts
        # are not (they reach the total row count), but a row of rsa = at @ u is
        # that row of at times u, so the picked counts give them exactly.
        pick = (lane((G, Ep)) == first + sub((G, Ep))).astype(jnp.int32)
        my_c = _exact_dot(pick, ct)  # [G, 128]
        my_a = _exact_dot(pick, at)
        my_rsa = _exact_dot(my_a, u)
        my_tiles = run_tiles(my_a, my_rsa)

        ltg = (lane((G, G)) < sub((G, G))).astype(jnp.int32)
        grp_off = _exact_dot(ltg, my_a)  # [G, 128]: prefix over local experts
        er_col = jnp.sum(my_a, axis=1, keepdims=True)  # expert_rows [G, 1]
        sb_col = jnp.sum(grp_off, axis=1, keepdims=True)  # slab base [G, 1]
        out_total = jnp.sum(my_a, axis=0, keepdims=True)  # [1, 128]
        contrib_base = jnp.sum(my_rsa, axis=0, keepdims=True)  # prefix of out_total

        rs_ref[...] = jnp.right_shift(my_rsa, ROWBLK_SHIFT)[:, :ep]
        al_ref[...] = jnp.right_shift(my_a, ROWBLK_SHIFT)[:, :ep]
        co_ref[...] = jnp.right_shift(contrib_base + grp_off, ROWBLK_SHIFT)[:, :ep]
        pd_ref[...] = jnp.right_shift(pre + grp_off, ROWBLK_SHIFT)[:, :ep]
        tr_ref[...] = my_c[:, :ep]
        er_ref[...] = er_col
        sb_ref[...] = sb_col

        # The visit list: experts by (rows descending, index ascending).
        ones = jnp.ones((8, LANES), jnp.int32)
        er_row = _exact_dot_nt(ones, my_a)[0:1, :]  # [1, G]
        key_col = sub((G, 1)) - er_col * G
        key_row = lane((1, G)) - er_row * G
        rank_col = jnp.sum((key_row < key_col).astype(jnp.int32), axis=1, keepdims=True)
        perm_row = jnp.sum(
            jnp.where(rank_col == lane((G, G)), sub((G, G)), 0), axis=0, keepdims=True
        )
        visit_ref[...] = jnp.minimum(perm_row, G - 1)

        # The count table, row by row as shard_count_vector lays it out.
        mine = (lane((1, LANES)) == me).astype(jnp.int32)
        other = 1 - mine
        my_true = jnp.sum(my_c, axis=0, keepdims=True)
        my_tile_sum = jnp.sum(my_tiles, axis=0, keepdims=True)
        n_active = jnp.sum((er_col > 0).astype(jnp.int32), axis=0, keepdims=True)
        counts = jnp.concatenate(
            [
                my_true * other,
                (recv_rows_all - my_true) * mine,
                out_total * other,
                (recv_rows - out_total) * mine,
                n_active * mine,
                my_tile_sum * other,
                (recv_tiles_all - my_tile_sum) * mine,
                my_tile_sum * mine,
            ],
            axis=0,
        )
        counts_ref[...] = counts[:, :ep]

        # -- 3. This shard's own selections: arrival row and mirror slot. -----
        e_row = payload_ref[:, pl.ds(0, TKp)]  # [1, TKp]
        e_row = jnp.where(lane((1, TKp)) < TK, e_row, e_total)
        e_col = jnp.broadcast_to(e_row, (8, TKp)).T[:, 0:1]  # [TKp, 1]
        valid = (e_col >= 0) & (e_col < e_total)
        e_lane = lane((TKp, Ep))
        # recv_base[me, e] at e = the pair's expert: rows before it here.
        before = (e_lane < e_col).astype(jnp.int32)
        pos_base_all = _exact_dot(before, at)  # [TKp, 128]
        pos_base = jnp.sum(
            jnp.where(lane((TKp, LANES)) == me, pos_base_all, 0), axis=1, keepdims=True
        )
        onehot = (e_lane == e_col).astype(jnp.int32)
        rs_low_all = _exact_dot(onehot, rsa & (tile_m - 1))
        rs_low = jnp.sum(
            jnp.where(lane((TKp, LANES)) == me, rs_low_all, 0), axis=1, keepdims=True
        )
        p_sub, j_lane = sub((TKp, TKp)), lane((TKp, TKp))
        same = (e_col == e_row) & (j_lane < p_sub) & (j_lane < TK)
        rank_all = jnp.sum(same.astype(jnp.int32), axis=1, keepdims=True)
        same_blk = same & (
            jnp.right_shift(j_lane, block_shift) == jnp.right_shift(p_sub, block_shift)
        )
        rank_blk = jnp.sum(same_blk.astype(jnp.int32), axis=1, keepdims=True)
        pos = jnp.where(valid, pos_base + rank_all, rank_blk)
        mirror = jnp.where(
            valid,
            rank_all
            + (jnp.right_shift(pos_base, tile_shift) + 2 * e_col) * tile_m
            + rs_low,
            rank_blk,
        )
        pos_ref[...] = pos[:TK]
        mir_ref[...] = mirror[:TK]

        # -- 4. Which gathered pairs land here, as a bitmask per source. -----
        pr = gbuf[:, pl.ds(0, TKp)]  # [ep, TKp]
        live = (pr >= first) & (pr < first + G) & (lane((ep, TKp)) < TK)
        live = live.astype(jnp.int32)
        p_i, w_i = sub((TKp, LANES)), lane((TKp, LANES))
        in_word = jnp.right_shift(p_i, 5) == w_i
        bit = p_i & (WORD_BITS - 1)
        pow16 = jnp.left_shift(1, bit & 15)
        lo = _exact_dot(live, jnp.where(in_word & (bit < 16), pow16, 0))
        hi = _exact_dot(live, jnp.where(in_word & (bit >= 16), pow16, 0))
        words = lo | jnp.left_shift(hi, 16)  # [ep, 128]: n_words live
        # Row of the run (g, s) starts at: slab base + run start.
        t2 = sb_col + my_rsa  # [G, 128], s on lanes
        live_total = jnp.sum(er_col, axis=0, keepdims=True)  # [1, 1]
        stage[pl.ds(0, ep), :] = words
        stage[pl.ds(st_t2, G), :] = t2
        stage[pl.ds(st_misc, 8), :] = jnp.broadcast_to(
            jnp.where(lane((1, LANES)) == 0, live_total, 0), (8, LANES)
        )
        stage_copy = pltpu.make_async_copy(stage, stage_sm, sems.at[_SEM_STAGE])
        stage_copy.start()
        stage_copy.wait()
        pairs_copy.wait()
        for c in zeroing:
            c.wait()

        # -- 5. Scalar placement of the live pairs. --------------------------
        for g in range(G):
            cnt_sm[g] = jnp.int32(0)

        def per_source(s, carry):
            for w in range(n_words):

                def place(m, *, w=w):
                    lsb = m & (-m)
                    p = w * WORD_BITS + _bit_index(lsb)
                    g = pairs_sm[s, p] - first
                    c = cnt_sm[g]
                    c = jnp.where(jnp.right_shift(c, 16) == s, c & 0xFFFF, 0)
                    row = stage_sm[st_t2 + g, s] + c
                    cnt_sm[g] = jnp.left_shift(s, 16) | (c + 1)
                    tl = tokp_ref[p]
                    r_hi = jnp.right_shift(row, 7)
                    r_lo = row & (LANES - 1)
                    tg_sm[r_hi, r_lo] = s * t_local + tl
                    if quantized:
                        sc_sm[r_hi, r_lo] = pairs_sm[s, scale_at + tl]
                    return m ^ lsb

                lax.while_loop(lambda m: m != 0, place, stage_sm[s, w])
            return carry

        lax.fori_loop(0, ep, per_source, jnp.int32(0))

        # -- 6. The live row range out; the rest stays the aliased zero. ------
        n_chunks = jnp.right_shift(stage_sm[st_misc, 0] + (LANES - 1), 7)

        def chunk_copies(j):
            rows = pl.ds(j, 1)
            cps = [
                pltpu.make_async_copy(
                    tg_sm.at[rows], tg_hbm.at[rows], sems.at[_SEM_OUT]
                )
            ]
            if quantized:
                cps.append(
                    pltpu.make_async_copy(
                        sc_sm.at[rows], sc_hbm.at[rows], sems.at[_SEM_OUT]
                    )
                )
            return cps

        def start(j, carry):
            for c in chunk_copies(j):
                c.start()
            return carry

        def wait(j, carry):
            for c in chunk_copies(j):
                c.wait()
            return carry

        lax.fori_loop(0, n_chunks, start, jnp.int32(0))
        lax.fori_loop(0, n_chunks, wait, jnp.int32(0))

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    i32 = jnp.int32
    out_shape = (
        *(jax.ShapeDtypeStruct((G, ep), i32) for _ in range(5)),
        jax.ShapeDtypeStruct((G, 1), i32),
        jax.ShapeDtypeStruct((G, 1), i32),
        jax.ShapeDtypeStruct((1, G), i32),
        jax.ShapeDtypeStruct((8, ep), i32),
        jax.ShapeDtypeStruct((TK, 1), i32),
        jax.ShapeDtypeStruct((TK, 1), i32),
        jax.ShapeDtypeStruct((tg_out_rows, LANES), i32),
        jax.ShapeDtypeStruct((sc_rows, LANES), i32),
    )
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, hbm, hbm],
            out_specs=[vmem] * 11 + [hbm, hbm],
            scratch_shapes=(
                pltpu.VMEM((ep, width_g), i32),
                pltpu.VMEM((st_rows, LANES), i32),
                pltpu.VMEM((max(sc_rows, tg_rows), LANES), i32),
                pltpu.SMEM((ep, psw), i32),
                pltpu.SMEM((st_rows, LANES), i32),
                pltpu.SMEM((tg_rows, LANES), i32),
                pltpu.SMEM((sc_rows, LANES), i32),
                pltpu.SMEM((g8,), i32),
                pltpu.SemaphoreType.DMA((_N_SEMS,)),
            ),
            grid=(),
        ),
        out_shape=out_shape,
        # Operands count the two scalar prefetches: (rank, tokp, payload,
        # tg_zero, sc_zero).
        input_output_aliases={3: 11, 4: 12},
        compiler_params=pltpu.CompilerParams(
            # No has_side_effects: every output is consumed, and torch_tpu's
            # SPMD re-partitioning refuses a side-effecting custom call
            # ("Side-effect HLO must have sharding").
            collective_id=ROUTE_PLAN_COLLECTIVE_ID,
        ),
        name=f"fused_ep_moe_v2_route_plan_ep{ep}_g{G}_t{t_local}",
    )
    tokp = np.zeros((TKp,), np.int32)
    tokp[:TK] = np.arange(TK) // topk

    def fn(payload, rank):
        """payload [width] i32 (this shard's), rank [1, 1] i32 -> RoutePlan."""
        if payload.dtype != jnp.int32 or tuple(payload.shape) != (width,):
            raise ValueError(
                f"payload must be int32[{width}], "
                f"got {payload.dtype}{list(payload.shape)}"
            )
        payload = jnp.pad(payload, (0, width_p - width)).reshape(1, width_p)
        outs = call(
            rank.astype(i32).reshape(1),
            jnp.asarray(tokp),
            payload,
            jnp.zeros((tg_out_rows, LANES), i32),
            jnp.zeros((sc_rows, LANES), i32),
        )
        rs, al, co, pd, tr, er, sb, visit, counts, pos, mir, tg, sc = outs
        return RoutePlan(
            kernel_tables=(rs, al, co, pd, tr),
            token_gather=tg.reshape(-1),
            expert_rows=er.reshape(G),
            slab_base=sb.reshape(G),
            scale_slab=lax.bitcast_convert_type(sc, jnp.float32) if quantized else None,
            visit=visit.reshape(G),
            counts=counts,
            pos=pos.reshape(t_local, topk),
            mirror_pos=mir.reshape(t_local, topk),
        )

    return fn


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def build_route_plan(**kwargs):
    """Memoizing front door (pallas keys its trace cache on the kernel object)."""
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    fn = _CACHE.get(key)
    if fn is None:
        with _CACHE_LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                fn = _build_route_plan(**kwargs)
                _CACHE[key] = fn
    return fn

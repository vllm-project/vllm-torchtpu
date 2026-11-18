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
# Not a copy of a kernel on main: a layer of its own that only the Qwen3.5
# DP4 x TP2 decode layout runs (fused_moe_ep_v3.QWEN3_5_CONFIG_BY_LAYOUT turns it
# on). It calls the fused EP v3 package's select, row quantization, FFN and
# layout helpers (kernels/fused_moe/v3); its in-program weighted sum repeats the
# v3 combine's per-token sum, so a change to the combine does not reach it.
"""Fused EP MoE with landing rows fixed by the routing alone ("dense slots";
the Qwen3.5 DP4xTP2 decode layout, default off).

The v3 layer moves data the same way (an all-gather of every shard's fp8
token rows, every local expert streamed once, result rows pushed back to the
token owners), but it builds packed per-(expert, destination) tables on the
TensorCore and the sparse cores and sorts the routed rows into expert order
before the expert kernel. This layer needs none of that. One Pallas program
per shard:

  * a barrier and the first weight refills; the shard's top-k indices go to
    every peer by remote DMA, and the shard's rows, quantized to fp8, are
    all-gathered the same way into a VMEM-resident copy of every token row;
  * from every token's top-k each shard computes, with vector ops under the
    weight refills, where every one of its result rows lands: the
    destination's arrival buffer is ordered (source shard, local expert,
    token), with no padding;
  * each local expert with at least one routed row once (the experts no row
    is routed to are skipped and their weights never streamed, apart from
    the first WEIGHT_SLOTS - 1, whose refills start before the tables
    exist), its rows picked out of the resident rows by a one-hot matmul
    (exact), the FFN math of v3 (ffn.expert_ffn_fp8), and each tile's
    result rows pushed per (expert, destination) run straight into the
    destination's arrival buffer in VMEM;
  * the destination sums each token's top_k arrivals in k order, as v3's
    weighted_sum does, in the same program.

Result rows travel as bf16 (Config.result_rows "bf16" in v3 terms): no row
scale has to travel with them. The same per-row math as v3 with bf16 result
rows, so the output can be compared bit for bit against v3 run with
result_rows="bf16" (and elementwise float32, where every compilation agrees).
"""

import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import device, ffn, layout
from vllm_torchtpu.kernels.fused_moe.v3.config import Config
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, MESH_AXIS, device_coords
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8, quantize_rows

# A barrier semaphore is shared by kernels with the same collective id.
COLLECTIVE_ID_DENSE = 5
# fp8 rows are tiled 32 sublanes deep: each shard's block of the resident rows
# starts on a tile.
ROW_ALIGN = 32
# Rows per one-hot gather pass (the tile's rows are gathered this many at a
# time, only as many passes as the tile has live rows).
GATHER_ROWS = 32
TAB_LANES = 32
# VMEM weight slots (whole experts) of the program: it keeps
# WEIGHT_SLOTS - 1 expert refills in flight.
WEIGHT_SLOTS = 3


def _align(n, m):
    return -(-n // m) * m


def _tables_math(me, allv, own, *, D, Tp, G, K, E):
    """The routing tables of shard `me`, from every shard's lane-padded top-k
    (allv [D * Tp, LANES] int32 global expert ids, E or more for none) and
    this shard's own (own [Tp, LANES]), as vector ops inside the program; the
    running sums are matmuls of 0/1 matrices (exact: every count is at most
    Tp * K, every bf16 operand at most Tp). Returns:
      mp [G, N]: row of token t among local expert e's rows (tokens in
        order), -1 where e is not among t's choices;
      tab [G, TAB_LANES]: per (expert, destination) run, where it starts
        among the expert's rows (lanes [0, D)), its length ([D, 2D)) and the
        destination's arrival row it lands on ([2D, 3D)); the expert's rows
        at lane 3D;
      pos [Tp, K]: arrival row of each of this shard's pairs, 0 for a pair
        routed nowhere;
      misc [8, LANES]: row 0 holds (me, rows arriving from other shards, rows
        sent to self, rows sent to other shards).
    """
    N = D * Tp

    def dot(a, b):
        return lax.dot_general(
            a.astype(jnp.bfloat16),
            b.astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    lane = lax.broadcasted_iota(jnp.int32, (N, LANES), 1)
    valid_all = (allv >= 0) & (allv < E) & (lane < K)
    loc_t = jnp.transpose(jnp.where(valid_all, allv - me * G, -1))
    e_iota = lax.broadcasted_iota(jnp.int32, (G, N), 0)
    m = loc_t[0:1, :] == e_iota
    for k in range(1, K):
        m = m | (loc_t[k : k + 1, :] == e_iota)
    upper = lax.broadcasted_iota(jnp.int32, (N, N), 0) <= lax.broadcasted_iota(
        jnp.int32, (N, N), 1
    )
    incl = dot(m.astype(jnp.float32), upper.astype(jnp.float32)).astype(
        jnp.int32
    )  # [G, N]
    mp = jnp.where(m, incl - 1, -1)
    ends = [incl[:, (d + 1) * Tp - 1 : (d + 1) * Tp] for d in range(D)]
    zero_col = jnp.zeros((G, 1), jnp.int32)
    run_start = jnp.concatenate([zero_col] + ends[:-1], axis=1)  # [G, D]
    run_len = jnp.concatenate(ends, axis=1) - run_start
    shard_of = jnp.where(valid_all, allv // G, D)
    base = jnp.concatenate(
        [
            jnp.sum(
                (shard_of[d * Tp : (d + 1) * Tp, :] < me).astype(jnp.int32),
                keepdims=True,
            ).reshape(1, 1)
            for d in range(D)
        ],
        axis=1,
    )  # [1, D]
    lower = lax.broadcasted_iota(jnp.int32, (G, G), 1) < lax.broadcasted_iota(
        jnp.int32, (G, G), 0
    )
    dst_row = base + dot(lower.astype(jnp.float32), run_len.astype(jnp.float32)).astype(
        jnp.int32
    )
    cnt = ends[-1]
    tab = jnp.concatenate(
        [
            run_start,
            run_len,
            dst_row,
            cnt,
            jnp.zeros((G, TAB_LANES - 3 * D - 1), jnp.int32),
        ],
        axis=1,
    )
    # Arrival row of each own pair: own pairs with a smaller expert, plus
    # earlier own tokens with the same expert.
    own_lane = lax.broadcasted_iota(jnp.int32, (Tp, LANES), 1)
    own_valid = (own >= 0) & (own < E) & (own_lane < K)
    x_iota = lax.broadcasted_iota(jnp.int32, (Tp, E), 1)
    ohs = []
    member = jnp.zeros((Tp, E), jnp.float32)
    for k in range(K):
        ohk = ((own[:, k : k + 1] == x_iota) & own_valid[:, k : k + 1]).astype(
            jnp.float32
        )
        ohs.append(ohk)
        member = member + ohk
    hist = jnp.broadcast_to(jnp.sum(member, axis=0, keepdims=True), (8, E))
    strict_x = lax.broadcasted_iota(jnp.int32, (E, E), 0) < lax.broadcasted_iota(
        jnp.int32, (E, E), 1
    )
    smaller = dot(hist, strict_x.astype(jnp.float32))[0:1, :]  # [1, E]
    strict_t = lax.broadcasted_iota(jnp.int32, (Tp, Tp), 1) < lax.broadcasted_iota(
        jnp.int32, (Tp, Tp), 0
    )
    earlier = dot(strict_t.astype(jnp.float32), member)  # [Tp, E]
    rank = smaller + earlier
    pos = jnp.concatenate(
        [jnp.sum(ohk * rank, axis=1, keepdims=True) for ohk in ohs], axis=1
    )
    pos = pos.astype(jnp.int32)
    n_valid = jnp.sum(own_valid.astype(jnp.int32), keepdims=True).reshape(1, 1)
    d_iota = lax.broadcasted_iota(jnp.int32, (G, D), 1)
    own_local = jnp.sum(jnp.where(d_iota == me, run_len, 0), keepdims=True).reshape(
        1, 1
    )
    n_rows = jnp.sum(cnt, keepdims=True).reshape(1, 1)
    ml = lax.broadcasted_iota(jnp.int32, (8, LANES), 1)
    ms = lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
    misc = jnp.where(ml == 0, me, 0)
    misc = jnp.where(ml == 1, n_valid - own_local, misc)
    misc = jnp.where(ml == 2, own_local, misc)
    misc = jnp.where(ml == 3, n_rows - own_local, misc)
    misc = jnp.where(ms == 0, misc, 0)
    return mp, tab, pos, misc


def _wait_rows(sem, ref, rows):
    """Waits until `rows` rows' worth of copies on `sem` have landed."""
    pltpu.make_async_copy(ref.at[pl.ds(0, rows)], ref.at[pl.ds(0, rows)], sem).wait()


def _barrier(num_shards, mesh_axes):
    barrier = pltpu.get_barrier_semaphore()
    for i in range(num_shards):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=device_coords(jnp.int32(i), mesh_axes),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, num_shards)


@functools.cache
def build_kernel(
    *,
    num_shards,
    tokens_per_shard,
    real_rows,
    hidden,
    inter,
    experts_per_shard,
    top_k,
    tile_rows,
    mesh_axes,
    elementwise,
    columns_per_pass,
    result_slots=3,
    out_dtype=jnp.bfloat16,
    num_experts=None,
):
    """The per-shard Pallas program: a callable of the shard index, the
    valid-row count, the shard's rows, top-k indices and weights, and the
    local experts' weights and scales.

    The program takes the shard's real_rows rows, top-k and weights
    unpadded, plus the valid-row count, and does the padding to
    tokens_per_shard rows and the valid-row masking itself (no XLA pads or
    masks around it). The index exchange and the tables run inside the
    program, after the first weight refills are issued (so they stream under
    the table math), and reach the scalar core by a VMEM -> SMEM copy. The
    arrivals and the weighted sum stay in VMEM."""
    R = real_rows
    if R % 8 or tokens_per_shard < R:
        raise ValueError(
            f"real_rows {real_rows}: a multiple of 8, at most {tokens_per_shard}"
        )
    D, Tp, H, INTER, G, K, T = (
        num_shards,
        tokens_per_shard,
        hidden,
        inter,
        experts_per_shard,
        top_k,
        tile_rows,
    )
    N = D * Tp
    LB = H // LANES
    if Tp % ROW_ALIGN or T % GATHER_ROWS or H % LANES:
        raise ValueError(
            f"tokens_per_shard {Tp} (x{ROW_ALIGN}), tile {T} "
            f"(x{GATHER_ROWS}), hidden {H} (x{LANES})"
        )
    weight_slots = WEIGHT_SLOTS
    weight_prefetch = WEIGHT_SLOTS - 1
    n_arrival = Tp * K
    passes = T // GATHER_ROWS

    E = num_experts if num_experts is not None else G * D

    def kernel(*refs):
        it = iter(refs)
        me_ref, nv_ref = next(it), next(it)
        x_ref, idx_ref, wts_ref = next(it), next(it), next(it)
        w1_ref, w2_ref, w1s_ref, w2s_ref = (next(it) for _ in range(4))
        out_ref = next(it)
        (
            _xq_vmem,
            _xs_vmem,
            rows_all,
            scales_all,
            tile_vmem,
            w1_vmem,
            w2_vmem,
            result_vmem,
            xs_tile,
        ) = (next(it) for _ in range(9))
        arrivals_ref, coef_vmem, sum_vmem = (next(it) for _ in range(3))
        (idx_all, mp_v, tab_v, pos_v, misc_v, tab_s, pos_s, misc_s) = (
            next(it) for _ in range(8)
        )
        order_s = next(it)
        (
            sem_w1,
            sem_w2,
            sem_ag_send,
            sem_ag_recv,
            _sem_ag_local,
            _sem_load,
            sem_push_send,
            sem_push_local,
            sem_push_recv,
        ) = (next(it) for _ in range(9))
        sem_idx_send, sem_idx_recv, sem_smem = (next(it) for _ in range(3))
        me = me_ref[0]

        def w_copy(e, slot):
            return (
                pltpu.make_async_copy(w1_ref.at[e], w1_vmem.at[slot], sem_w1.at[slot]),
                pltpu.make_async_copy(w2_ref.at[e], w2_vmem.at[slot], sem_w2.at[slot]),
            )

        # Weight refills depend on nothing but the expert: they start first
        # and stream while the rows are exchanged.
        for i in range(min(weight_prefetch, G)):
            for cp in w_copy(i, i):
                cp.start()

        # The shard's rows quantized as v3's transport does (quantize_rows,
        # per-row scale), then all-gathered: one remote copy per peer into this
        # shard's block of the peer's resident rows.
        keep = lax.broadcasted_iota(jnp.int32, (R, 1), 0) < nv_ref[0]
        x_in = jnp.where(keep, x_ref[...], jnp.zeros_like(x_ref[...]))
        xq, xs = quantize_rows(x_in, jnp.float32, dtype=elementwise)
        block = pl.ds(pl.multiple_of(me * Tp, ROW_ALIGN), Tp)
        # Straight into every peer's resident copy (VMEM): no landing buffer,
        # no reload.
        real_block = pl.ds(pl.multiple_of(me * Tp, ROW_ALIGN), R)
        rows_all[block, :] = jnp.zeros((Tp, H), FP8)
        scales_all[block, :] = jnp.zeros((Tp, LANES), jnp.float32)
        rows_all[real_block, :] = xq
        scales_all[real_block, :] = jnp.broadcast_to(xs, (R, LANES))
        w_in = jnp.where(keep, wts_ref[...], 0.0)
        idx_in = jnp.where(keep & (w_in != 0), idx_ref[...], E)
        idx_all[block, :] = jnp.full((Tp, LANES), E, jnp.int32)
        idx_all[real_block, 0:K] = idx_in
        coef_vmem[...] = w_in
        ag_src = (rows_all.at[block], scales_all.at[block])
        ag_dst = ag_src
        _barrier(D, mesh_axes)
        for i in range(1, D):
            peer = lax.rem(me + i, jnp.int32(D))
            # The indices first: the tables wait on them alone.
            pltpu.make_async_remote_copy(
                src_ref=idx_all.at[block],
                dst_ref=idx_all.at[block],
                send_sem=sem_idx_send,
                recv_sem=sem_idx_recv,
                device_id=device_coords(peer, mesh_axes),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
            for src, dst in zip(ag_src, ag_dst):
                pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=sem_ag_send,
                    recv_sem=sem_ag_recv,
                    device_id=device_coords(peer, mesh_axes),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
        if D > 1:
            _wait_rows(sem_idx_recv, idx_all, (D - 1) * Tp)
        own_idx = idx_all[block, :]
        mp, tab, pos, misc = _tables_math(
            me, idx_all[...], own_idx, D=D, Tp=Tp, G=G, K=K, E=E
        )
        mp_v[...] = mp
        tab_v[...] = tab
        pos_v[...] = pos
        misc_v[...] = misc
        to_smem = [
            pltpu.make_async_copy(v, s_, sem_smem)
            for v, s_ in ((tab_v, tab_s), (pos_v, pos_s), (misc_v, misc_s))
        ]
        for cp in to_smem:
            cp.start()
        for cp in to_smem:
            cp.wait()
        tab_ref, pos_ref, misc_ref, mp_ref = tab_s, pos_s, misc_s, mp_v
        recv_remote = misc_ref[0, 1]
        own_local = misc_ref[0, 2]
        if D > 1:
            _wait_rows(sem_ag_recv, rows_all, (D - 1) * Tp)
            _wait_rows(sem_ag_recv, scales_all, (D - 1) * Tp)
        # The scales as one lane-major row: column t is token t's scale.
        scale_row = jnp.transpose(scales_all[...])[0:1, :]  # [1, N]
        tile_vmem[...] = jnp.zeros_like(tile_vmem)
        xs_tile[...] = jnp.zeros_like(xs_tile)

        def push_tile(e, j, live, rs):
            """Pushes the tile's rows of each (expert, destination) run."""
            n_remote = jnp.int32(0)
            n_local = jnp.int32(0)
            for d in range(D):
                start = tab_ref[e, d]
                length = tab_ref[e, D + d]
                lo = jnp.maximum(start, j * T)
                hi = jnp.minimum(start + length, j * T + live)
                n = jnp.maximum(hi - lo, 0)
                src = result_vmem.at[rs, pl.ds(lo - j * T, n)]
                dst = arrivals_ref.at[pl.ds(tab_ref[e, 2 * D + d] + lo - start, n)]
                is_me = jnp.int32(d) == me

                @pl.when(jnp.logical_and(n > 0, jnp.logical_not(is_me)))
                def _(src=src, dst=dst, d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=sem_push_send.at[rs],
                        recv_sem=sem_push_recv,
                        device_id=device_coords(jnp.int32(d), mesh_axes),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                @pl.when(jnp.logical_and(n > 0, is_me))
                def _(src=src, dst=dst):
                    pltpu.make_async_copy(src, dst, sem_push_local.at[rs]).start()

                n_remote = n_remote + jnp.where(is_me, 0, n)
                n_local = n_local + jnp.where(is_me, n, 0)
            return n_remote, n_local

        def drain_slot(rs, n_remote, n_local):
            @pl.when(n_remote > 0)
            def _():
                _wait_rows(sem_push_send.at[rs], result_vmem.at[rs], n_remote)

            @pl.when(n_local > 0)
            def _():
                _wait_rows(sem_push_local.at[rs], result_vmem.at[rs], n_local)

        def tile_step(e, wslot, n_rows, j, carry):
            tile_count, pend_r, pend_l = carry
            rs = lax.rem(tile_count, jnp.int32(result_slots))
            pr = pend_r[0]
            plc = pend_l[0]
            for i in range(1, result_slots):
                pr = jnp.where(rs == i, pend_r[i], pr)
                plc = jnp.where(rs == i, pend_l[i], plc)
            drain_slot(rs, pr, plc)
            live = jnp.minimum(n_rows - j * T, T)
            mp_row = mp_ref[pl.ds(e, 1), :]  # [1, N]
            for p in range(passes):

                @pl.when(j * T + p * GATHER_ROWS < n_rows)
                def _(p=p):
                    want = (
                        j * T
                        + p * GATHER_ROWS
                        + lax.broadcasted_iota(jnp.int32, (GATHER_ROWS, N), 0)
                    )
                    hit = mp_row == want  # [GATHER_ROWS, N], one hit a live row
                    onehot = jnp.where(hit, 1.0, 0.0).astype(FP8)
                    picked = lax.dot_general(
                        onehot,
                        rows_all[...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    tile_vmem[p * GATHER_ROWS : (p + 1) * GATHER_ROWS, :] = (
                        picked.astype(FP8)
                    )
                    s = jnp.sum(jnp.where(hit, scale_row, 0.0), axis=1, keepdims=True)
                    tile_s = jnp.broadcast_to(s, (GATHER_ROWS, LANES))
                    xs_tile[p * GATHER_ROWS : (p + 1) * GATHER_ROWS, :] = tile_s

            row_scales = xs_tile[:, 0:1]
            acc2, mid_scales = ffn.expert_ffn_fp8(
                tile_vmem[...],
                row_scales,
                w1_vmem[wslot],
                w2_vmem[wslot],
                w1s_ref[pl.ds(e, 1), :],
                activation="silu",
                w1_bias=None,
                no_gate=False,
                elementwise=elementwise,
                columns_per_pass=columns_per_pass,
            )
            result, _ = ffn.result_rows(
                acc2,
                mid_scales,
                w2_scales=w2s_ref[pl.ds(e, 1), :],
                w2_bias=None,
                dtype=jnp.bfloat16,
                elementwise=elementwise,
            )
            result = result.reshape(T, LB, LANES)
            for i in range(result_slots):

                @pl.when(rs == i)
                def _(i=i):
                    result_vmem[i] = result

            n_r, n_l = push_tile(e, j, live, rs)
            pend_r = tuple(
                jnp.where(rs == i, n_r, pend_r[i]) for i in range(result_slots)
            )
            pend_l = tuple(
                jnp.where(rs == i, n_l, pend_l[i]) for i in range(result_slots)
            )
            return tile_count + 1, pend_r, pend_l

        zeros = tuple(jnp.int32(0) for _ in range(result_slots))
        # The live experts in order. The first P are in the list whatever
        # their rows: their refills started before the tables existed, and
        # list position i uses weight slot i % weight_slots, which for
        # i < P is the slot the prologue filled. A store at every position
        # and a conditional advance keep the scan branch-free.
        P = min(weight_prefetch, G)
        for i in range(P):
            order_s[i] = jnp.int32(i)

        def collect(e, n):
            live = tab_ref[e, 3 * D] > 0
            order_s[n] = e
            return n + jnp.where(live, 1, 0)

        n_list = lax.fori_loop(P, G, collect, jnp.int32(P))

        def listed_step(i, carry):
            e = order_s[i]
            wslot = lax.rem(i, jnp.int32(weight_slots))
            for cp in w_copy(e, wslot):
                cp.wait()

            @pl.when(i + weight_prefetch < n_list)
            def _():
                ahead = i + weight_prefetch
                for cp in w_copy(
                    order_s[ahead], lax.rem(ahead, jnp.int32(weight_slots))
                ):
                    cp.start()

            n_rows = tab_ref[e, 3 * D]
            n_tiles = (n_rows + T - 1) // T
            return lax.fori_loop(
                0, n_tiles, lambda j, c: tile_step(e, wslot, n_rows, j, c), carry
            )

        _, pend_r, pend_l = lax.fori_loop(
            0, n_list, listed_step, (jnp.int32(0), zeros, zeros)
        )
        for i in range(result_slots):
            drain_slot(i, pend_r[i], pend_l[i])
        if D > 1:
            _wait_rows(sem_ag_send, rows_all, (D - 1) * Tp)
            _wait_rows(sem_ag_send, scales_all, (D - 1) * Tp)
            _wait_rows(sem_idx_send, idx_all, (D - 1) * Tp)

        @pl.when(recv_remote > 0)
        def _():
            _wait_rows(sem_push_recv, arrivals_ref, recv_remote)

        del own_local  # local pushes are drained per slot above

        # v3's weighted_sum body on the arrivals in place: per token the top_k
        # rows (looked up by pair), k in order, each times its coefficient, a
        # zero coefficient's term dropped.
        def token(t, carry):
            c = coef_vmem[pl.ds(t, 1), :]
            acc = None
            for k in range(K):
                coef = c[:, k : k + 1]
                term = arrivals_ref[pos_ref[t, k]].astype(jnp.float32) * coef
                term = jnp.where(coef != 0, term, jnp.zeros_like(term))
                acc = term if k == 0 else cast("jax.Array", acc) + term
            row0 = pl.multiple_of(t * LB, 8)
            sum_vmem[pl.ds(row0, LB), :] = acc
            return carry

        lax.fori_loop(0, R, token, 0, unroll=8)
        out_ref[...] = sum_vmem[...].reshape(R, H).astype(out_ref.dtype)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    scratch = [
        # xq_vmem, xs_vmem and the "ag local" and "load" semaphores are not
        # used by this program; they stay in its plan, the one its performance
        # was measured with.
        pltpu.VMEM((Tp, H), FP8),  # xq_vmem
        pltpu.VMEM((Tp, LANES), jnp.float32),  # xs_vmem
        pltpu.VMEM((N, H), FP8),  # rows_all
        pltpu.VMEM((N, LANES), jnp.float32),  # scales_all
        pltpu.VMEM((T, H), FP8),  # tile_vmem
        pltpu.VMEM((weight_slots, H, 2 * INTER), FP8),  # w1_vmem
        pltpu.VMEM((weight_slots, INTER, H), FP8),  # w2_vmem
        pltpu.VMEM((result_slots, T, LB, LANES), jnp.bfloat16),  # result
        pltpu.VMEM((T, LANES), jnp.float32),  # xs_tile
        pltpu.VMEM((n_arrival, LB, LANES), jnp.bfloat16),  # arrivals
        pltpu.VMEM((R, K), jnp.float32),  # coefficients
        pltpu.VMEM((R * LB, LANES), jnp.float32),  # weighted sums
        pltpu.VMEM((N, LANES), jnp.int32),  # idx_all
        pltpu.VMEM((G, N), jnp.int32),  # mp
        pltpu.VMEM((G, TAB_LANES), jnp.int32),  # tab (vector side)
        pltpu.VMEM((Tp, K), jnp.int32),  # pos
        pltpu.VMEM((8, LANES), jnp.int32),  # misc
        pltpu.SMEM((G, TAB_LANES), jnp.int32),  # tab (scalar side)
        pltpu.SMEM((Tp, K), jnp.int32),  # pos
        pltpu.SMEM((8, LANES), jnp.int32),  # misc
        pltpu.SMEM((G,), jnp.int32),  # the live-expert list
        pltpu.SemaphoreType.DMA((weight_slots,)),
        pltpu.SemaphoreType.DMA((weight_slots,)),
        pltpu.SemaphoreType.DMA,  # ag send
        pltpu.SemaphoreType.DMA,  # ag recv
        pltpu.SemaphoreType.DMA,  # ag local
        pltpu.SemaphoreType.DMA,  # load
        pltpu.SemaphoreType.DMA((result_slots,)),  # push send
        pltpu.SemaphoreType.DMA((result_slots,)),  # push local
        pltpu.SemaphoreType.DMA,  # push recv
        pltpu.SemaphoreType.DMA,  # idx send
        pltpu.SemaphoreType.DMA,  # idx recv
        pltpu.SemaphoreType.DMA,  # smem
    ]
    out_shape = [jax.ShapeDtypeStruct((R, H), out_dtype)]
    out_specs = [vmem]
    # x, top-k, weights, w1, w2 (streamed from HBM), w1 and w2 scales.
    in_specs = [vmem, vmem, vmem, hbm, hbm, vmem, vmem]
    params = dict(
        collective_id=COLLECTIVE_ID_DENSE, vmem_limit_bytes=device.vmem_limit()
    )
    if layout.SIDE_EFFECTS:
        params["has_side_effects"] = True
    program = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=tuple(scratch),
            grid=(),
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(**params),
        name=f"fused_ep_dense_slots_fmus_g{G}_t{T}_n{N}",
    )
    return program


def dense_slots_moe(
    x,
    w1,
    w2,
    w1_scales,
    w2_scales,
    router_logits,
    *,
    top_k,
    renormalize,
    mesh,
    tile_rows=128,
    config=None,
    experts_per_shard=None,
    shard_index=None,
    valid_tokens=None,
    drop_nan_rows=False,
):
    """The dense-slots layer: same operands and result as v3's fused_ep_moe
    with fp8 weights, per-channel scales, silu and bf16 result rows (the
    DP4xTP2 decode binding). x [tokens, hidden] bf16 sharded over the mesh
    axis; router logits [tokens, num_experts]. Each shard's rows, a multiple
    of 8, go to the program unpadded; it pads them to ROW_ALIGN rows
    itself."""
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer  # select kernel

    if config is None:
        config = Config()
    tokens, hidden = x.shape
    num_experts_w, _, cols = w1.shape
    inter = cols // 2
    num_shards = mesh.shape[MESH_AXIS]
    if tuple(mesh.axis_names) != (MESH_AXIS,):
        raise ValueError(f"a mesh of the one axis {MESH_AXIS!r}")
    G = experts_per_shard or num_experts_w // num_shards
    num_experts = G * num_shards
    real = tokens // num_shards
    if real % 8:
        raise ValueError(
            f"{real} rows per shard: the dense-slots layer takes a multiple of 8 "
            "(the fused EP adapter keeps other counts on the v3 layer)"
        )
    Tp = _align(real, ROW_ALIGN)
    K = top_k
    program = build_kernel(
        out_dtype=x.dtype,
        num_experts=num_experts,
        real_rows=real,
        num_shards=num_shards,
        tokens_per_shard=Tp,
        hidden=hidden,
        inter=inter,
        experts_per_shard=G,
        top_k=K,
        tile_rows=tile_rows,
        mesh_axes=tuple(mesh.axis_names),
        elementwise=config.elementwise,
        columns_per_pass=config.activation_columns_per_pass,
    )
    with_valid = valid_tokens is not None

    def body(x_l, logits_l, w1_l, w2_l, s1_l, s2_l, shard_l, *rest):
        me = shard_l.reshape(-1)[0]
        n_valid = rest[0].reshape(-1)[0] if with_valid else None
        weights, indices = v3_layer._select(
            logits_l,
            top_k=K,
            tokens_per_shard=real,
            renormalize=renormalize,
            drop_nan_rows=drop_nan_rows,
            message_rows=None,
            valid_tokens=None if n_valid is None else n_valid.reshape(1),
        )
        nv = (
            jnp.full((1,), real, jnp.int32)
            if n_valid is None
            else n_valid.astype(jnp.int32).reshape(1)
        )
        (out,) = program(
            me.reshape(1),
            nv,
            x_l.astype(jnp.bfloat16),
            indices.astype(jnp.int32),
            weights.astype(jnp.float32),
            w1_l[:G],
            w2_l[:G],
            s1_l[:G].astype(jnp.float32),
            s2_l[:G].astype(jnp.float32),
        )
        return out

    if shard_index is None:
        shard_index = jnp.arange(num_shards, dtype=jnp.int32).reshape(num_shards, 1)
    rows_spec = jax.sharding.PartitionSpec(MESH_AXIS)
    args = [
        x,
        router_logits,
        w1,
        w2,
        w1_scales,
        w2_scales,
        shard_index.astype(jnp.int32).reshape(num_shards, 1),
    ]
    specs = [rows_spec] * 7
    if with_valid:
        args.append(valid_tokens.astype(jnp.int32).reshape(num_shards, 1))
        specs.append(rows_spec)
    sm = jax.shard_map(
        body, mesh=mesh, in_specs=tuple(specs), out_specs=rows_spec, check_vma=False
    )
    return sm(*args)

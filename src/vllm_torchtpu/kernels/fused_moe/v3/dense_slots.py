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
"""Fused EP MoE with landing rows fixed by the routing alone ("dense slots";
DP4xTP2 decode, experimental, default off).

The v3 layer moves data the same way (an all-gather of every shard's fp8
token rows, every local expert streamed once, result rows pushed back to the
token owners), but it builds packed per-(expert, destination) tables on the
TensorCore and the sparse cores and sorts the routed rows into expert order
before the expert kernel. This layer needs none of that:

  * every shard knows every token's top-k (a small all-gather of the
    indices), so each shard computes, with a few dense XLA ops, where every
    one of its result rows lands: the destination's arrival buffer is ordered
    (source shard, local expert, token), with no padding;
  * one Pallas program per shard: a barrier, the first weight refills, the
    shard's rows quantized to fp8 and all-gathered by remote DMA into a
    VMEM-resident copy of every token row; then each local expert once,
    its rows picked out of the resident rows by a one-hot matmul (exact),
    the FFN math of v3 (ffn.expert_ffn_fp8), and each tile's result rows
    pushed per (expert, destination) run straight into the destination's
    arrival buffer;
  * the destination gathers its arrivals into pair order (token-major,
    pair t * top_k + k) and sums them with v3's weighted_sum, which adds the
    k terms in order exactly as v3's combine does.

Result rows travel as bf16 (Config.result_rows "bf16" in v3 terms): no row
scale has to travel with them. The same per-row math as v3 with bf16 result
rows, so the output can be compared bit for bit against v3 run with
result_rows="bf16" (and elementwise float32, where every compilation agrees).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import device, ffn, layout
from .combine import weighted_sum
from .config import Config
from .layout import LANES, MESH_AXIS, device_coords
from .rowquant import FP8, quantize_rows

# A barrier semaphore is shared by kernels with the same collective id.
COLLECTIVE_ID_DENSE = 5
# fp8 rows are tiled 32 sublanes deep: each shard's block of the resident rows
# starts on a tile.
ROW_ALIGN = 32
# Rows per one-hot gather pass (the tile's rows are gathered this many at a
# time, only as many passes as the tile has live rows).
GATHER_ROWS = 32
INTERPRET = None  # tests set a pltpu.InterpretParams()
VMEM_LIMIT = None  # tests (no TPU to ask) set a byte count
# The combine inside the program (arrivals and the exchange in VMEM).
FUSED = False
# The index exchange and the tables as one Pallas program instead of XLA ops.
PALLAS_TABLES = False
COLLECTIVE_ID_TABLES = 6
# Tests only: a function applied to the tables dict (mutation checks).
TABLE_HOOK = None
# The tables inside the fused program (no separate tables program).
MERGED = False
# Tests only: the merged program's arrival rows shifted by this much (mutation check).
MERGED_POS_SHIFT = 0
# Merged program on the unpadded rows (padding and valid-row masks inside).
UNPADDED = False
TAB_LANES = 32


def _align(n, m):
    return -(-n // m) * m


def shard_tables(idx_all, me, *, experts_per_shard, num_shards, tokens_per_shard,
                 num_experts):
    """The routing tables of shard `me`, from every token's top-k.

    idx_all: [num_shards * tokens_per_shard, top_k] int32 global expert ids
    (num_experts or more for none). Returns a dict of int32 arrays:
      mp [G, N]: row of token t among local expert e's rows (tokens in
        order), -1 where e is not among t's choices;
      cnt [G]: rows of each local expert;
      run_start, run_len, dst_row [G * D]: per (expert, destination) run,
        where it starts among the expert's rows, its length, and the
        destination's arrival row it lands on;
      pos [tokens_per_shard * top_k]: arrival row of each of this shard's
        pairs (pair order), 0 for a pair routed nowhere;
      misc [4]: (me, rows arriving from other shards, rows sent to self,
        rows sent to other shards).
    Packed as the programs read them: tab [G, TAB_LANES] (run_start in
    lanes [0, D), run_len [D, 2D), dst_row [2D, 3D), cnt at 3D), pos
    [tokens_per_shard, top_k], misc [8, LANES] (row 0).
    """
    G, D, Tp = experts_per_shard, num_shards, tokens_per_shard
    N, K = idx_all.shape
    idx_all = idx_all.astype(jnp.int32)
    valid_all = (idx_all >= 0) & (idx_all < num_experts)
    loc = jnp.where(valid_all, idx_all - me * G, -1)
    e_iota = jnp.arange(G, dtype=jnp.int32)[:, None]
    m = jnp.zeros((G, N), jnp.bool_)
    for k in range(K):
        m = m | (loc[:, k][None, :] == e_iota)
    mi = m.astype(jnp.int32)
    incl = jnp.cumsum(mi, axis=1)
    mp = jnp.where(m, incl - mi, -1)
    cnt = incl[:, -1]
    run_len = mi.reshape(G, D, Tp).sum(-1)
    run_start = jnp.cumsum(run_len, axis=1) - run_len
    shard_of = jnp.where(valid_all, idx_all // G, D)  # [N, K]
    by_dest = shard_of.reshape(D, Tp * K)
    s_iota = jnp.arange(D, dtype=jnp.int32)
    # c[s, d]: pairs of destination d's tokens whose expert is on shard s
    c = jnp.sum((by_dest[None, :, :] == s_iota[:, None, None]).astype(jnp.int32),
                axis=2)
    base = jnp.sum(jnp.where((s_iota < me)[:, None], c, 0), axis=0)  # [D]
    dst_row = base[None, :] + jnp.cumsum(run_len, axis=0) - run_len
    own = lax.dynamic_slice_in_dim(idx_all, me * Tp, Tp, 0).reshape(-1)
    valid = (own >= 0) & (own < num_experts)
    key = jnp.where(valid, own, num_experts)
    tok = jnp.arange(Tp * K, dtype=jnp.int32) // K
    less = (key[None, :] < key[:, None]) | ((key[None, :] == key[:, None]) &
                                           (tok[None, :] < tok[:, None]))
    pos = jnp.sum((less & valid[None, :]).astype(jnp.int32), axis=1)
    pos = jnp.where(valid, pos, 0)
    n_valid = jnp.sum(valid.astype(jnp.int32))
    own_local = jnp.sum(jnp.where(s_iota == me, run_len, 0))
    misc = jnp.stack([
        jnp.asarray(me, jnp.int32), n_valid - own_local, own_local,
        jnp.sum(cnt) - own_local
    ]).astype(jnp.int32)
    tab = jnp.zeros((G, TAB_LANES), jnp.int32)
    tab = tab.at[:, 0:D].set(run_start).at[:, D:2 * D].set(run_len)
    tab = tab.at[:, 2 * D:3 * D].set(dst_row).at[:, 3 * D].set(cnt)
    misc_t = jnp.zeros((8, LANES), jnp.int32).at[0, 0:4].set(misc)
    return dict(mp=mp, tab=tab, pos=pos.reshape(Tp, K), misc=misc_t)


def _tables_math(me, allv, own, *, D, Tp, G, K, E):
    """shard_tables' tables from every shard's lane-padded top-k (allv
    [D * Tp, LANES]) and this shard's own (own [Tp, LANES]), as vector ops
    inside a program; the running sums are matmuls of 0/1 matrices (exact).
    Returns (mp [G, N], tab [G, TAB_LANES], pos [Tp, K], misc [8, LANES])."""
    N = D * Tp

    def dot(a, b):
        return lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16),
                               (((1, ), (0, )), ((), ())),
                               preferred_element_type=jnp.float32)

    lane = lax.broadcasted_iota(jnp.int32, (N, LANES), 1)
    valid_all = (allv >= 0) & (allv < E) & (lane < K)
    loc_t = jnp.transpose(jnp.where(valid_all, allv - me * G, -1))
    e_iota = lax.broadcasted_iota(jnp.int32, (G, N), 0)
    m = loc_t[0:1, :] == e_iota
    for k in range(1, K):
        m = m | (loc_t[k:k + 1, :] == e_iota)
    upper = (lax.broadcasted_iota(jnp.int32, (N, N), 0) <=
             lax.broadcasted_iota(jnp.int32, (N, N), 1))
    incl = dot(m.astype(jnp.float32), upper.astype(jnp.float32)).astype(
        jnp.int32)  # [G, N]
    mp = jnp.where(m, incl - 1, -1)
    ends = [incl[:, (d + 1) * Tp - 1:(d + 1) * Tp] for d in range(D)]
    zero_col = jnp.zeros((G, 1), jnp.int32)
    run_start = jnp.concatenate([zero_col] + ends[:-1], axis=1)  # [G, D]
    run_len = jnp.concatenate(ends, axis=1) - run_start
    shard_of = jnp.where(valid_all, allv // G, D)
    base = jnp.concatenate([
        jnp.sum((shard_of[d * Tp:(d + 1) * Tp, :] < me).astype(jnp.int32),
                keepdims=True).reshape(1, 1) for d in range(D)
    ], axis=1)  # [1, D]
    lower = (lax.broadcasted_iota(jnp.int32, (G, G), 1) <
             lax.broadcasted_iota(jnp.int32, (G, G), 0))
    dst_row = base + dot(lower.astype(jnp.float32),
                         run_len.astype(jnp.float32)).astype(jnp.int32)
    cnt = ends[-1]
    tab = jnp.concatenate(
        [run_start, run_len, dst_row, cnt,
         jnp.zeros((G, TAB_LANES - 3 * D - 1), jnp.int32)], axis=1)
    # Arrival row of each own pair: own pairs with a smaller expert, plus
    # earlier own tokens with the same expert.
    own_lane = lax.broadcasted_iota(jnp.int32, (Tp, LANES), 1)
    own_valid = (own >= 0) & (own < E) & (own_lane < K)
    x_iota = lax.broadcasted_iota(jnp.int32, (Tp, E), 1)
    ohs = []
    member = jnp.zeros((Tp, E), jnp.float32)
    for k in range(K):
        ohk = ((own[:, k:k + 1] == x_iota) &
               own_valid[:, k:k + 1]).astype(jnp.float32)
        ohs.append(ohk)
        member = member + ohk
    hist = jnp.broadcast_to(jnp.sum(member, axis=0, keepdims=True), (8, E))
    strict_x = (lax.broadcasted_iota(jnp.int32, (E, E), 0) <
                lax.broadcasted_iota(jnp.int32, (E, E), 1))
    smaller = dot(hist, strict_x.astype(jnp.float32))[0:1, :]  # [1, E]
    strict_t = (lax.broadcasted_iota(jnp.int32, (Tp, Tp), 1) <
                lax.broadcasted_iota(jnp.int32, (Tp, Tp), 0))
    earlier = dot(strict_t.astype(jnp.float32), member)  # [Tp, E]
    rank = smaller + earlier
    pos = jnp.concatenate(
        [jnp.sum(ohk * rank, axis=1, keepdims=True) for ohk in ohs], axis=1)
    pos = pos.astype(jnp.int32)
    n_valid = jnp.sum(own_valid.astype(jnp.int32), keepdims=True).reshape(1, 1)
    d_iota = lax.broadcasted_iota(jnp.int32, (G, D), 1)
    own_local = jnp.sum(jnp.where(d_iota == me, run_len, 0),
                        keepdims=True).reshape(1, 1)
    n_rows = jnp.sum(cnt, keepdims=True).reshape(1, 1)
    ml = lax.broadcasted_iota(jnp.int32, (8, LANES), 1)
    ms = lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
    misc = jnp.where(ml == 0, me, 0)
    misc = jnp.where(ml == 1, n_valid - own_local, misc)
    misc = jnp.where(ml == 2, own_local, misc)
    misc = jnp.where(ml == 3, n_rows - own_local, misc)
    misc = jnp.where(ms == 0, misc, 0)
    return mp, tab, pos, misc


@functools.lru_cache(maxsize=None)
def build_tables_kernel(*, num_shards, tokens_per_shard, experts_per_shard,
                        top_k, num_experts, mesh_axes):
    """The index exchange and shard_tables as one Pallas program: every
    shard's top-k (lane-padded [Tp, LANES] int32, num_experts for none) to
    every peer by remote DMA, then the same tables as shard_tables, with the
    running sums as matmuls of 0/1 matrices (exact: every count is at most
    tokens_per_shard * top_k, every bf16 operand at most tokens_per_shard).
    Returns a callable (me [1] int32, idx [Tp, LANES]) -> (mp, tab, pos, misc).
    """
    D, Tp, G, K, E = (num_shards, tokens_per_shard, experts_per_shard, top_k,
                      num_experts)
    N = D * Tp
    if K > LANES or 4 * D > TAB_LANES or Tp % 8:
        raise ValueError(f"top_k {K}, shards {D}, tokens {Tp}")

    def dot(a, b):
        return lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16),
                               (((1, ), (0, )), ((), ())),
                               preferred_element_type=jnp.float32)

    def kernel(me_ref, idx_ref, mp_ref, tab_ref, pos_ref, misc_ref, idx_all,
               sem_send, sem_recv):
        me = me_ref[0]
        block = pl.ds(pl.multiple_of(me * Tp, 8), Tp)
        own = idx_ref[...]
        idx_all[block, :] = own
        _barrier(D, mesh_axes)
        for i in range(1, D):
            peer = lax.rem(me + i, jnp.int32(D))
            pltpu.make_async_remote_copy(
                src_ref=idx_all.at[block], dst_ref=idx_all.at[block],
                send_sem=sem_send, recv_sem=sem_recv,
                device_id=device_coords(peer, mesh_axes),
                device_id_type=pl.DeviceIdType.MESH).start()
        if D > 1:
            _wait_rows(sem_recv, idx_all, (D - 1) * Tp)
        mp, tab, pos, misc = _tables_math(me, idx_all[...], own, D=D, Tp=Tp,
                                          G=G, K=K, E=E)
        mp_ref[...] = mp
        tab_ref[...] = tab
        pos_ref[...] = pos
        misc_ref[...] = misc
        if D > 1:
            _wait_rows(sem_send, idx_all, (D - 1) * Tp)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    params = dict(collective_id=COLLECTIVE_ID_TABLES,
                  vmem_limit_bytes=VMEM_LIMIT or device.vmem_limit())
    if layout.SIDE_EFFECTS:
        params["has_side_effects"] = True
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, in_specs=[vmem], out_specs=[vmem] * 4,
            scratch_shapes=(pltpu.VMEM((N, LANES), jnp.int32),
                            pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            grid=()),
        out_shape=[
            jax.ShapeDtypeStruct((G, N), jnp.int32),
            jax.ShapeDtypeStruct((G, TAB_LANES), jnp.int32),
            jax.ShapeDtypeStruct((Tp, K), jnp.int32),
            jax.ShapeDtypeStruct((8, LANES), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(**params),
        interpret=INTERPRET if INTERPRET is not None else False,
        name=f"fused_ep_dense_slots_tables_g{G}_n{N}")


def _wait_rows(sem, ref, rows):
    """Waits until `rows` rows' worth of copies on `sem` have landed."""
    pltpu.make_async_copy(ref.at[pl.ds(0, rows)], ref.at[pl.ds(0, rows)],
                          sem).wait()


def _barrier(num_shards, mesh_axes):
    barrier = pltpu.get_barrier_semaphore()
    for i in range(num_shards):
        pl.semaphore_signal(barrier, inc=1,
                            device_id=device_coords(jnp.int32(i), mesh_axes),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, num_shards)


@functools.lru_cache(maxsize=None)
def build_kernel(*, num_shards, tokens_per_shard, hidden, inter,
                 experts_per_shard, top_k, tile_rows, mesh_axes, elementwise,
                 columns_per_pass, weight_slots=3, weight_prefetch=2,
                 result_slots=3, vmem_limit=None, fused=False,
                 out_dtype=jnp.bfloat16, merged=False, num_experts=None,
                 real_rows=None):
    """The per-shard Pallas program (a callable of the tables and operands).

    merged (fused only): the index exchange and the tables run inside this
    program, after the first weight refills are issued (so they stream
    under the table math), and reach the scalar core by a VMEM -> SMEM
    copy; the program then takes the shard index and its own top-k
    instead of the tables."""
    if merged and not fused:
        raise ValueError("merged tables need the fused program")
    # real_rows (merged only): the program takes the shard's real rows, top-k
    # and weights unpadded, plus the valid-row count, and does the padding and
    # the valid-row masking itself (no XLA pads or masks around it).
    R = tokens_per_shard if real_rows is None else real_rows
    if real_rows is not None and (not merged or R % 8 or R > tokens_per_shard):
        raise ValueError(f"real_rows {real_rows}: merged only, a multiple of 8, "
                         f"at most {tokens_per_shard}")
    unpadded = real_rows is not None
    D, Tp, H, I, G, K, T = (num_shards, tokens_per_shard, hidden, inter,
                            experts_per_shard, top_k, tile_rows)
    N = D * Tp
    LB = H // LANES
    if Tp % ROW_ALIGN or T % GATHER_ROWS or H % LANES:
        raise ValueError(f"tokens_per_shard {Tp} (x{ROW_ALIGN}), tile {T} "
                         f"(x{GATHER_ROWS}), hidden {H} (x{LANES})")
    if not 1 <= weight_prefetch < weight_slots:
        raise ValueError("weight_prefetch must be in [1, weight_slots)")
    n_arrival = Tp * K
    passes = T // GATHER_ROWS

    E = num_experts if num_experts is not None else G * D

    def kernel(*refs):
        it = iter(refs)
        if merged:
            me_ref = next(it)
            nv_ref = next(it) if unpadded else None
            tab_ref = misc_ref = pos_ref = mp_ref = None
            x_ref, idx_ref = next(it), next(it)
        else:
            tab_ref, misc_ref = next(it), next(it)
            pos_ref = next(it) if fused else None
            mp_ref, x_ref = next(it), next(it)
        wts_ref = next(it) if fused else None
        w1_ref, w2_ref, w1s_ref, w2s_ref = (next(it) for _ in range(4))
        if fused:
            out_ref = next(it)
            landing_ref = landing_s_ref = arrivals_ref = None
        else:
            landing_ref, landing_s_ref, arrivals_ref = (next(it)
                                                        for _ in range(3))
        (xq_vmem, xs_vmem, rows_all, scales_all, tile_vmem, w1_vmem, w2_vmem,
         result_vmem, xs_tile) = (next(it) for _ in range(9))
        if fused:
            arrivals_ref, coef_vmem, sum_vmem = (next(it) for _ in range(3))
        if merged:
            (idx_all, mp_v, tab_v, pos_v, misc_v, tab_s, pos_s,
             misc_s) = (next(it) for _ in range(8))
        (sem_w1, sem_w2, sem_ag_send, sem_ag_recv, sem_ag_local, sem_load,
         sem_push_send, sem_push_local, sem_push_recv) = (next(it)
                                                          for _ in range(9))
        if merged:
            sem_idx_send, sem_idx_recv, sem_smem = (next(it) for _ in range(3))
            me = me_ref[0]
        else:
            me = misc_ref[0, 0]

        def w_copy(e, slot):
            return (pltpu.make_async_copy(w1_ref.at[e], w1_vmem.at[slot],
                                          sem_w1.at[slot]),
                    pltpu.make_async_copy(w2_ref.at[e], w2_vmem.at[slot],
                                          sem_w2.at[slot]))

        # Weight refills depend on nothing but the expert: they start first
        # and stream while the rows are exchanged.
        for i in range(min(weight_prefetch, G)):
            for cp in w_copy(i, i):
                cp.start()

        # The shard's rows quantized as v3's transport does (quantize_rows,
        # per-row scale), then all-gathered: one remote copy per peer into its
        # landing block for this shard.
        if unpadded:
            keep = lax.broadcasted_iota(jnp.int32, (R, 1), 0) < nv_ref[0]
            x_in = jnp.where(keep, x_ref[...], jnp.zeros_like(x_ref[...]))
        else:
            x_in = x_ref[...]
        xq, xs = quantize_rows(x_in, jnp.float32, block=0, dtype=elementwise)
        block = pl.ds(pl.multiple_of(me * Tp, ROW_ALIGN), Tp)
        if fused:
            # Straight into every peer's resident copy (VMEM): no landing
            # buffer, no reload.
            if unpadded:
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
            else:
                rows_all[block, :] = xq
                scales_all[block, :] = jnp.broadcast_to(xs, (Tp, LANES))
                if merged:
                    idx_all[block, :] = idx_ref[...]
            ag_src = (rows_all.at[block], scales_all.at[block])
            ag_dst = ag_src
        else:
            xq_vmem[...] = xq
            xs_vmem[...] = jnp.broadcast_to(xs, (Tp, LANES))
            ag_src = (xq_vmem, xs_vmem)
            ag_dst = (landing_ref.at[block], landing_s_ref.at[block])
        _barrier(D, mesh_axes)
        if not fused:
            local_rows = pltpu.make_async_copy(ag_src[0], ag_dst[0],
                                               sem_ag_local)
            local_scales = pltpu.make_async_copy(ag_src[1], ag_dst[1],
                                                 sem_ag_local)
            local_rows.start()
            local_scales.start()
        for i in range(1, D):
            peer = lax.rem(me + i, jnp.int32(D))
            if merged:  # the indices first: the tables wait on them alone
                pltpu.make_async_remote_copy(
                    src_ref=idx_all.at[block], dst_ref=idx_all.at[block],
                    send_sem=sem_idx_send, recv_sem=sem_idx_recv,
                    device_id=device_coords(peer, mesh_axes),
                    device_id_type=pl.DeviceIdType.MESH).start()
            for src, dst in zip(ag_src, ag_dst):
                pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst, send_sem=sem_ag_send,
                    recv_sem=sem_ag_recv,
                    device_id=device_coords(peer, mesh_axes),
                    device_id_type=pl.DeviceIdType.MESH).start()
        if merged:
            if D > 1:
                _wait_rows(sem_idx_recv, idx_all, (D - 1) * Tp)
            own_idx = idx_all[block, :] if unpadded else idx_ref[...]
            mp, tab, pos, misc = _tables_math(me, idx_all[...], own_idx,
                                              D=D, Tp=Tp, G=G, K=K, E=E)
            if MERGED_POS_SHIFT:
                pos = jnp.minimum(pos + MERGED_POS_SHIFT, n_arrival - 1)
            mp_v[...] = mp
            tab_v[...] = tab
            pos_v[...] = pos
            misc_v[...] = misc
            to_smem = [pltpu.make_async_copy(v, s_, sem_smem)
                       for v, s_ in ((tab_v, tab_s), (pos_v, pos_s),
                                     (misc_v, misc_s))]
            for cp in to_smem:
                cp.start()
            for cp in to_smem:
                cp.wait()
            tab_ref, pos_ref, misc_ref, mp_ref = tab_s, pos_s, misc_s, mp_v
        recv_remote = misc_ref[0, 1]
        own_local = misc_ref[0, 2]
        wait_ref = rows_all if fused else landing_ref
        wait_s_ref = scales_all if fused else landing_s_ref
        if not fused:
            local_rows.wait()
            local_scales.wait()
        if D > 1:
            _wait_rows(sem_ag_recv, wait_ref, (D - 1) * Tp)
            _wait_rows(sem_ag_recv, wait_s_ref, (D - 1) * Tp)
        if not fused:
            load_rows = pltpu.make_async_copy(landing_ref, rows_all, sem_load)
            load_scales = pltpu.make_async_copy(landing_s_ref, scales_all,
                                                sem_load)
            load_rows.start()
            load_scales.start()
            load_rows.wait()
            load_scales.wait()
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
                dst = arrivals_ref.at[pl.ds(tab_ref[e, 2 * D + d] + lo - start,
                                            n)]
                is_me = jnp.int32(d) == me

                @pl.when(jnp.logical_and(n > 0, jnp.logical_not(is_me)))
                def _(src=src, dst=dst, d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=src, dst_ref=dst,
                        send_sem=sem_push_send.at[rs], recv_sem=sem_push_recv,
                        device_id=device_coords(jnp.int32(d), mesh_axes),
                        device_id_type=pl.DeviceIdType.MESH).start()

                @pl.when(jnp.logical_and(n > 0, is_me))
                def _(src=src, dst=dst):
                    pltpu.make_async_copy(src, dst,
                                          sem_push_local.at[rs]).start()

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
                    want = (j * T + p * GATHER_ROWS + lax.broadcasted_iota(
                        jnp.int32, (GATHER_ROWS, N), 0))
                    hit = mp_row == want  # [GATHER_ROWS, N], one hit a live row
                    onehot = jnp.where(hit, 1.0, 0.0).astype(FP8)
                    picked = lax.dot_general(
                        onehot, rows_all[...], (((1, ), (0, )), ((), ())),
                        preferred_element_type=jnp.float32)
                    tile_vmem[p * GATHER_ROWS:(p + 1) * GATHER_ROWS, :] = (
                        picked.astype(FP8))
                    s = jnp.sum(jnp.where(hit, scale_row, 0.0), axis=1,
                                keepdims=True)
                    tile_s = jnp.broadcast_to(s, (GATHER_ROWS, LANES))
                    xs_tile[p * GATHER_ROWS:(p + 1) * GATHER_ROWS, :] = tile_s

            row_scales = xs_tile[:, 0:1]
            acc2, mid_scales = ffn.expert_ffn_fp8(
                tile_vmem[...], row_scales, w1_vmem[wslot], w2_vmem[wslot],
                w1s_ref[pl.ds(e, 1), :], activation="silu", w1_bias=None,
                no_gate=False, block=0, elementwise=elementwise,
                columns_per_pass=columns_per_pass, token_block=0)
            result, _ = ffn.result_rows(acc2, mid_scales,
                                        w2_scales=w2s_ref[pl.ds(e, 1), :],
                                        w2_bias=None, dtype=jnp.bfloat16,
                                        elementwise=elementwise)
            result = result.reshape(T, LB, LANES)
            for i in range(result_slots):

                @pl.when(rs == i)
                def _(i=i):
                    result_vmem[i] = result

            n_r, n_l = push_tile(e, j, live, rs)
            pend_r = tuple(jnp.where(rs == i, n_r, pend_r[i])
                           for i in range(result_slots))
            pend_l = tuple(jnp.where(rs == i, n_l, pend_l[i])
                           for i in range(result_slots))
            return tile_count + 1, pend_r, pend_l

        def expert_step(e, carry):
            wslot = lax.rem(e, jnp.int32(weight_slots))
            for cp in w_copy(e, wslot):
                cp.wait()

            @pl.when(e + weight_prefetch < G)
            def _():
                ahead = e + weight_prefetch
                for cp in w_copy(ahead, lax.rem(ahead,
                                                jnp.int32(weight_slots))):
                    cp.start()

            n_rows = tab_ref[e, 3 * D]
            n_tiles = (n_rows + T - 1) // T
            return lax.fori_loop(
                0, n_tiles,
                lambda j, c: tile_step(e, wslot, n_rows, j, c), carry)

        zeros = tuple(jnp.int32(0) for _ in range(result_slots))
        _, pend_r, pend_l = lax.fori_loop(0, G, expert_step,
                                          (jnp.int32(0), zeros, zeros))
        for i in range(result_slots):
            drain_slot(i, pend_r[i], pend_l[i])
        if D > 1:
            _wait_rows(sem_ag_send, wait_ref, (D - 1) * Tp)
            _wait_rows(sem_ag_send, wait_s_ref, (D - 1) * Tp)
            if merged:
                _wait_rows(sem_idx_send, idx_all, (D - 1) * Tp)

        @pl.when(recv_remote > 0)
        def _():
            _wait_rows(sem_push_recv, arrivals_ref, recv_remote)

        del own_local  # local pushes are drained per slot above
        if fused:
            # v3's weighted_sum body on the arrivals in place: per token the
            # top_k rows (looked up by pair), k in order, each times its
            # coefficient, a zero coefficient's term dropped.
            if not unpadded:
                coef_vmem[...] = wts_ref[...]

            def token(t, carry):
                c = coef_vmem[pl.ds(t, 1), :]
                acc = None
                for k in range(K):
                    coef = c[:, k:k + 1]
                    term = arrivals_ref[pos_ref[t, k]].astype(
                        jnp.float32) * coef
                    term = jnp.where(coef != 0, term, jnp.zeros_like(term))
                    acc = term if k == 0 else acc + term
                row0 = pl.multiple_of(t * LB, 8)
                sum_vmem[pl.ds(row0, LB), :] = acc
                return carry

            lax.fori_loop(0, R, token, 0, unroll=8)
            out_ref[...] = sum_vmem[...].reshape(R, H).astype(out_ref.dtype)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    scratch = [
        pltpu.VMEM((Tp, H), FP8),  # xq_vmem
        pltpu.VMEM((Tp, LANES), jnp.float32),  # xs_vmem
        pltpu.VMEM((N, H), FP8),  # rows_all
        pltpu.VMEM((N, LANES), jnp.float32),  # scales_all
        pltpu.VMEM((T, H), FP8),  # tile_vmem
        pltpu.VMEM((weight_slots, H, 2 * I), FP8),  # w1_vmem
        pltpu.VMEM((weight_slots, I, H), FP8),  # w2_vmem
        pltpu.VMEM((result_slots, T, LB, LANES), jnp.bfloat16),  # result
        pltpu.VMEM((T, LANES), jnp.float32),  # xs_tile
    ]
    if fused:
        scratch += [
            pltpu.VMEM((n_arrival, LB, LANES), jnp.bfloat16),  # arrivals
            pltpu.VMEM((R, K), jnp.float32),  # coefficients
            pltpu.VMEM((R * LB, LANES), jnp.float32),  # weighted sums
        ]
    if merged:
        scratch += [
            pltpu.VMEM((N, LANES), jnp.int32),  # idx_all
            pltpu.VMEM((G, N), jnp.int32),  # mp
            pltpu.VMEM((G, TAB_LANES), jnp.int32),  # tab (vector side)
            pltpu.VMEM((Tp, K), jnp.int32),  # pos
            pltpu.VMEM((8, LANES), jnp.int32),  # misc
            pltpu.SMEM((G, TAB_LANES), jnp.int32),  # tab (scalar side)
            pltpu.SMEM((Tp, K), jnp.int32),  # pos
            pltpu.SMEM((8, LANES), jnp.int32),  # misc
        ]
    scratch += [
        pltpu.SemaphoreType.DMA((weight_slots, )),
        pltpu.SemaphoreType.DMA((weight_slots, )),
        pltpu.SemaphoreType.DMA,  # ag send
        pltpu.SemaphoreType.DMA,  # ag recv
        pltpu.SemaphoreType.DMA,  # ag local
        pltpu.SemaphoreType.DMA,  # load
        pltpu.SemaphoreType.DMA((result_slots, )),  # push send
        pltpu.SemaphoreType.DMA((result_slots, )),  # push local
        pltpu.SemaphoreType.DMA,  # push recv
    ]
    if merged:
        scratch += [pltpu.SemaphoreType.DMA] * 3  # idx send / recv, smem
    if fused:
        out_shape = [jax.ShapeDtypeStruct((R, H), out_dtype)]
        out_specs = [vmem]
        in_specs = ([vmem, vmem, vmem, hbm, hbm, vmem, vmem] if merged else
                    [vmem, vmem, vmem, hbm, hbm, vmem, vmem])
    else:
        out_shape = [
            jax.ShapeDtypeStruct((N, H), FP8),
            jax.ShapeDtypeStruct((N, LANES), jnp.float32),
            jax.ShapeDtypeStruct((n_arrival, LB, LANES), jnp.bfloat16),
        ]
        out_specs = [hbm, hbm, hbm]
        in_specs = [vmem, vmem, hbm, hbm, vmem, vmem]
    limit = vmem_limit or VMEM_LIMIT or device.vmem_limit()
    params = dict(collective_id=COLLECTIVE_ID_DENSE, vmem_limit_bytes=limit)
    if layout.SIDE_EFFECTS:
        params["has_side_effects"] = True
    program = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=(2 if unpadded else 1) if merged else (3 if fused else 2),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=tuple(scratch),
            grid=()),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(**params),
        interpret=INTERPRET if INTERPRET is not None else False,
        name=(f"fused_ep_dense_slots{'_f' if fused else ''}"
              f"{'m' if merged else ''}{'u' if unpadded else ''}_g{G}_t{T}_n{N}"))
    return program


def dense_slots_moe(x, w1, w2, w1_scales, w2_scales, router_logits, *, top_k,
                    renormalize, mesh, tile_rows=128, config=None,
                    experts_per_shard=None, shard_index=None,
                    valid_tokens=None, drop_nan_rows=False, fused=None,
                    pallas_tables=None, merged=None, unpadded=None):
    """The dense-slots layer: same operands and result as v3's fused_ep_moe
    with fp8 weights, per-channel scales and silu (the DP4xTP2 decode
    binding). x [tokens, hidden] bf16 sharded over the mesh axis; router
    logits [tokens, num_experts]."""
    from . import layer as v3_layer  # the select kernel
    if config is None:
        config = Config()
    tokens, hidden = x.shape
    num_experts_w, _, cols = w1.shape
    inter = cols // 2
    num_shards = mesh.shape[MESH_AXIS]
    if tuple(mesh.axis_names) != (MESH_AXIS, ):
        raise ValueError(f"a mesh of the one axis {MESH_AXIS!r}")
    G = experts_per_shard or num_experts_w // num_shards
    num_experts = G * num_shards
    real = tokens // num_shards
    Tp = _align(real, ROW_ALIGN)
    K = top_k
    elementwise = config.elementwise
    if fused is None:
        fused = FUSED
    if pallas_tables is None:
        pallas_tables = PALLAS_TABLES
    if merged is None:
        merged = MERGED
    merged = merged and fused
    if merged:
        pallas_tables = False
    if unpadded is None:
        unpadded = UNPADDED
    unpadded = unpadded and merged and real % 8 == 0
    tables_program = (build_tables_kernel(
        num_shards=num_shards, tokens_per_shard=Tp, experts_per_shard=G,
        top_k=K, num_experts=num_experts, mesh_axes=tuple(mesh.axis_names))
                      if pallas_tables else None)
    program = build_kernel(fused=fused, out_dtype=x.dtype, merged=merged,
                           num_experts=num_experts,
                           real_rows=real if unpadded else None,
                           num_shards=num_shards, tokens_per_shard=Tp,
                           hidden=hidden, inter=inter, experts_per_shard=G,
                           top_k=K, tile_rows=tile_rows,
                           mesh_axes=tuple(mesh.axis_names),
                           elementwise=elementwise,
                           columns_per_pass=config.activation_columns_per_pass)
    with_valid = valid_tokens is not None

    def body(x_l, logits_l, w1_l, w2_l, s1_l, s2_l, shard_l, *rest):
        me = shard_l.reshape(-1)[0]
        n_valid = rest[0].reshape(-1)[0] if with_valid else None
        if unpadded:
            weights, indices = v3_layer._select(
                logits_l, top_k=K, tokens_per_shard=real,
                renormalize=renormalize, drop_nan_rows=drop_nan_rows,
                message_rows=None,
                valid_tokens=None if n_valid is None else n_valid.reshape(1))
            nv = (jnp.full((1, ), real, jnp.int32) if n_valid is None else
                  n_valid.astype(jnp.int32).reshape(1))
            out, = program(me.reshape(1), nv, x_l.astype(jnp.bfloat16),
                           indices.astype(jnp.int32), weights.astype(jnp.float32),
                           w1_l[:G], w2_l[:G], s1_l[:G].astype(jnp.float32),
                           s2_l[:G].astype(jnp.float32))
            return out
        if Tp > real:
            x_l = jnp.pad(x_l, ((0, Tp - real), (0, 0)))
            logits_l = jnp.pad(logits_l, ((0, Tp - real), (0, 0)),
                               constant_values=jnp.nan)
        if n_valid is not None:
            keep = jnp.arange(Tp, dtype=jnp.int32) < n_valid
            x_l = jnp.where(keep[:, None], x_l, 0)
        weights, indices = v3_layer._select(
            logits_l, top_k=K, tokens_per_shard=Tp, renormalize=renormalize,
            drop_nan_rows=drop_nan_rows, message_rows=None,
            valid_tokens=None if n_valid is None else n_valid.reshape(1))
        # A row with no finite logit gets weight zero from the select.
        indices = jnp.where(weights != 0, indices, num_experts)
        if n_valid is not None:
            keep = (jnp.arange(Tp, dtype=jnp.int32) < n_valid)[:, None]
            indices = jnp.where(keep, indices, num_experts)
            weights = jnp.where(keep, weights, 0.0)
        if merged:
            idx_pad = jnp.pad(indices.astype(jnp.int32), ((0, 0), (0, LANES - K)),
                              constant_values=num_experts)
            out, = program(me.reshape(1), x_l.astype(jnp.bfloat16), idx_pad,
                           weights.astype(jnp.float32), w1_l[:G], w2_l[:G],
                           s1_l[:G].astype(jnp.float32),
                           s2_l[:G].astype(jnp.float32))
            return out[:real]
        if pallas_tables:
            idx_pad = jnp.pad(indices.astype(jnp.int32), ((0, 0), (0, LANES - K)),
                              constant_values=num_experts)
            mp, tab, pos, misc_t = tables_program(me.reshape(1), idx_pad)
            t = dict(mp=mp, tab=tab, pos=pos, misc=misc_t)
        else:
            idx_all = lax.all_gather(indices.astype(jnp.int32), MESH_AXIS,
                                     tiled=True)
            t = shard_tables(idx_all, me, experts_per_shard=G,
                             num_shards=num_shards, tokens_per_shard=Tp,
                             num_experts=num_experts)
        if TABLE_HOOK is not None:
            t = TABLE_HOOK(t)
        operands = (x_l.astype(jnp.bfloat16), w1_l[:G], w2_l[:G],
                    s1_l[:G].astype(jnp.float32), s2_l[:G].astype(jnp.float32))
        if fused:
            out, = program(t["tab"], t["misc"], t["pos"], t["mp"], operands[0],
                           weights.astype(jnp.float32), *operands[1:])
            return out[:real]
        _, _, arrivals = program(t["tab"], t["misc"], t["mp"], *operands)
        rows = jnp.take(arrivals, t["pos"].reshape(-1), axis=0)
        out = weighted_sum(rows, None, weights.astype(jnp.float32),
                           tokens_per_shard=Tp, hidden=hidden,
                           out_dtype=x.dtype,
                           tokens_per_tile=min(Tp, 128),
                           dtype=config.combine)
        return out[:real]

    if shard_index is None:
        shard_index = jnp.arange(num_shards, dtype=jnp.int32).reshape(
            num_shards, 1)
    rows_spec = jax.sharding.PartitionSpec(MESH_AXIS)
    args = [x, router_logits, w1, w2, w1_scales, w2_scales,
            shard_index.astype(jnp.int32).reshape(num_shards, 1)]
    specs = [rows_spec] * 7
    if with_valid:
        args.append(valid_tokens.astype(jnp.int32).reshape(num_shards, 1))
        specs.append(rows_spec)
    sm = jax.shard_map(body, mesh=mesh, in_specs=tuple(specs),
                       out_specs=rows_spec, check_vma=False)
    return sm(*args)

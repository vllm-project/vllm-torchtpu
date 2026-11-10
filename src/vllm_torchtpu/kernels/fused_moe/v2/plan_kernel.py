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
"""The routing plan's per-pair tables as one Pallas TPU kernel (K2, Option B).

`build_routing_tables_sharded` gives every routed (token, expert) pair three
numbers, all of the form base[e] + c, where c counts the EARLIER pairs of the
same expert from the same source shard:

  pos[p]        = recv_base[me, e] + c          (this shard's own pairs)
  mirror_pos[p] = mirror_base[e] + c            (this shard's own pairs)
  slab_row[p]   = expert_base[e] + run_start_aligned[e, src(p)] + c
                  for e on this shard, an off-shard sentinel otherwise.

XLA builds c with one-hot passes over pairs x experts (about 103 M element
operations per call at 8,704 tokens). This kernel holds the gathered pairs in
VMEM as 128-pair lane rows and builds c with the expert id factored as
e = 32 * hi + lo:

  1. per row, independently: the row's histogram over (hi, lo) as one NT
     matmul of the two small one-hots ([HI, 128] x [32, 128]^T), and every
     pair's rank among the row's earlier pairs of its expert (a [128, 128]
     compare summed over sublanes);
  2. per source, serially: the exclusive prefix of the row histograms, i.e.
     the count of each expert in the source's earlier rows (vector adds only);
  3. per row, independently: the lookups. A [M, 32] table times the row's
     [32, 128] lo one-hot picks every pair's lo column for every hi; the hi
     one-hot then keeps the pair's own hi and a sublane sum reduces it. Tables
     travel as exact bf16 byte planes (counts: two, bases: three), so the
     matmul is one bf16 pass and exact.

The per-expert bases come from the exchanged count table and are built by the
caller (`plan_bases`), exactly as the XLA builder builds them. Pairs whose
expert id is e_total (routed nowhere) fall outside the hi one-hot, so they
count nowhere; their pos / mirror_pos are unspecified (the combine masks
them) and their slab_row is the sentinel, as in the XLA builder.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2 import host

LANES = 128
LO = 32  # experts per hi group: e = LO * hi + lo
LO_SHIFT = 5
# Count planes (a source's count of one expert is below 2**16) and base planes
# (every base is below 2**24; at the served shapes below 2**19).
COUNT_PLANES = 2
BASE_PLANES = 3
BYTE = 8
_DN_NT = (((1,), (1,)), ((), ()))
_MUTANTS = ("pt_rank_incl", "pt_prefix_incl", "pt_src_reset", "pt_split", "pt_sentinel",
            "pt_unrouted", "pt_precision")


def plan_kernel_supported(*, e_total, ep, t_local, topk):
    """Shapes the kernel takes: whole 128-pair rows per source and an expert
    count that is a whole number of 32-expert groups."""
    return (
        e_total % LO == 0
        and e_total >= LO
        and e_total % ep == 0
        and (t_local * topk) % LANES == 0
        and ep >= 1
    )


def plan_small_tables(rows_by_dest, me, *, e_total, ep, tile_m, shard_stride):
    """The plan's small tables from the exchanged count table, exactly as
    `host.build_routing_tables_sharded` builds them (that builder is left
    untouched so its own program stays byte-identical), and the per-expert
    bases [2 + ep, e_total] i32 the kernel adds its counts to:
    recv_base[me] (pos), the mirror base (mirror_pos) and, per source s,
    expert_base + run_start_aligned[:, s] (slab_row)."""
    g_local = e_total // ep
    run_rows = rows_by_dest.T
    run_rows_aligned = (run_rows + (host.ROWBLK - 1)) & jnp.int32(-host.ROWBLK)
    run_start_aligned = _xcumsum(run_rows_aligned, 1)
    expert_rows_aligned = run_rows_aligned.sum(axis=1)
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    local_base = _xcumsum(rows_by_shard, 1)
    expert_base = (local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] * shard_stride)).reshape(e_total)
    region_rows = run_rows_aligned.reshape(ep, g_local, ep)
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
    recv_base = _xcumsum(rows_per_dest, 1).reshape(ep, ep, g_local)
    recv_rows = rows_per_dest.sum(axis=1)
    my_recv_base = lax.dynamic_slice(recv_base, (me, 0, 0), (1, ep, g_local)).reshape(e_total)
    shift = host.pow2_shift(tile_m, "tile_m")
    my_run_start = lax.dynamic_slice(run_start_aligned, (0, me), (e_total, 1)).reshape(e_total)
    mirror_base = (
        (jnp.right_shift(my_recv_base, shift) + 2 * jnp.arange(e_total, dtype=jnp.int32)) * tile_m
        + (my_run_start & (tile_m - 1))
    )
    slab_base = _slab_base(expert_base, run_start_aligned, rows_by_dest)  # [ep, e_total]
    bases = jnp.concatenate([my_recv_base[None], mirror_base[None], slab_base], axis=0).astype(jnp.int32)
    small = dict(
        run_tiles=host.mirror_run_tiles(run_start_aligned, run_rows_aligned, tile_m),
        run_rows=run_rows, run_rows_aligned=run_rows_aligned, run_start_aligned=run_start_aligned,
        region_rows=region_rows, recv_base=recv_base, recv_rows=recv_rows,
        expert_rows_aligned=expert_rows_aligned, expert_base=expert_base)
    return small, bases


def plan_bases(rows_by_dest, me, *, e_total, ep, tile_m, shard_stride):
    """The per-expert bases [2 + ep, e_total] i32 (see plan_small_tables)."""
    return plan_small_tables(rows_by_dest, me, e_total=e_total, ep=ep, tile_m=tile_m,
                             shard_stride=shard_stride)[1]


def _planes(x, n):
    """x (non-negative int32 below 2**(8n)) as n exact bf16 byte planes stacked on axis 0."""
    return jnp.concatenate(
        [((x >> (BYTE * d)) & 0xFF).astype(jnp.float32).astype(jnp.bfloat16) for d in range(n)], axis=0)


def _tree_sum_tiles(x):
    """x [n * 8, 128] summed over its 8-sublane tiles as a balanced tree: [8, 128]."""
    parts = [x[k * 8:(k + 1) * 8] for k in range(x.shape[0] // 8)]
    while len(parts) > 1:
        parts = [parts[k] + parts[k + 1] if k + 1 < len(parts) else parts[k] for k in range(0, len(parts), 2)]
    return parts[0]


def _plan_body(me_ref, pairs_ref, bases_ref, pos_ref, mir_ref, slab_ref, cnt_scr, rank_scr, *rest,
               rows, ep, hi_n, e_total, g_local, sentinel, mutant, unroll, fuse, lookup, exchange_rows):
    me = me_ref[0]
    rows_pad = -(-rows // 8) * 8
    if exchange_rows:
        # The gathered exchange [ep, W, 128] stays in HBM; each source's pair rows (the first
        # `rows` of its W, rounded up to a whole (8, 128) tile) come into VMEM by DMA.
        pairs_scr, sem = rest
        for src in range(ep):
            cp = pltpu.make_async_copy(pairs_ref.at[src, pl.ds(0, rows_pad)],
                                       pairs_scr.at[pl.ds(src * rows_pad, rows_pad)], sem)
            cp.start()
            cp.wait()
    lo_shift = 4 if mutant == "pt_split" else LO_SHIFT
    iota_hi = lax.broadcasted_iota(jnp.int32, (hi_n, LANES), 0)
    iota_lo = lax.broadcasted_iota(jnp.int32, (LO, LANES), 0)
    jj = lax.broadcasted_iota(jnp.int32, (LANES, LANES), 0)
    ii = lax.broadcasted_iota(jnp.int32, (LANES, LANES), 1)
    earlier = jnp.where((jj <= ii) if mutant == "pt_rank_incl" else (jj < ii), 1.0, 0.0)

    def load(src, i):
        if exchange_rows:
            e = pairs_scr[pl.ds(src * rows_pad + i, 1), :]
        else:
            e = pairs_ref[pl.ds(src * rows + i, 1), :]
        if mutant == "pt_unrouted":
            e = jnp.where(e == e_total, e_total - 1, e)
        hi = jnp.right_shift(e, lo_shift)
        lo = e & (LO - 1)
        oh_hi = jnp.where(iota_hi == hi, 1.0, 0.0)
        oh_lo = jnp.where(iota_lo == lo, 1.0, 0.0)
        return e, oh_hi, oh_lo

    # 1. Row histograms and in-row ranks; rows are independent.
    def phase1(src, i, carry):
        r = src * rows + i
        e, oh_hi, oh_lo = load(src, i)
        cnt_scr[r] = lax.dot_general(oh_hi.astype(jnp.bfloat16), oh_lo.astype(jnp.bfloat16), _DN_NT,
                                     preferred_element_type=jnp.float32)
        e1 = e[0]
        # One select per vreg (the mask is a constant f32), then a depth-4 tree over the
        # 16 sublane tiles instead of a 16-long serial add chain.
        hits = jnp.where(e1[:, None] == e1[None, :], earlier, 0.0)
        rank_scr[pl.ds(r, 1), :] = jnp.sum(_tree_sum_tiles(hits), axis=0, keepdims=True)
        return carry

    # 2. Exclusive prefix of the row histograms within each source.
    zero = jnp.zeros((hi_n, LO), jnp.float32)

    def phase2(r, run):
        h = cnt_scr[r]
        cnt_scr[r] = run + h if mutant == "pt_prefix_incl" else run
        return run + h

    def prefix(s, start):
        return lax.fori_loop(s * rows, (s + 1) * rows, phase2, start, unroll=unroll[1])

    # 3. Lookups. The base planes are per expert and per source: built once.
    base_planes = ([_planes(bases_ref[t], BASE_PLANES) for t in range(2 + ep)]  # each [3 * hi_n, 32]
                   if lookup == "planes" else None)
    first = me * g_local

    def planes_at(v, k, n):
        """Planes k .. k + n - 1 of the stacked lookup v recombined into values: [hi_n, 128] f32,
        exact (every value is below 2**24). Recombining before the hi pick keeps one pick per table."""
        out = v[k * hi_n:(k + 1) * hi_n]
        for d in range(1, n):
            if mutant == "pt_precision" and d == BASE_PLANES - 1 and n == BASE_PLANES:
                continue
            out = out + v[(k + d) * hi_n:(k + d + 1) * hi_n] * float(1 << (BYTE * d))
        return out

    def pick(vals, oh_hi):
        """vals at each pair's own hi: [1, 128] f32."""
        return jnp.sum(vals * oh_hi, axis=0, keepdims=True)

    bases_f = [bases_ref[t].astype(jnp.float32) for t in range(2 + ep)]  # each [hi_n, 32]

    def lookup_row(s, local, i):
        r = s * rows + i
        e, oh_hi, oh_lo = load(s, i)
        rank = rank_scr[pl.ds(r, 1), :]
        on_me = jnp.logical_and(e >= first, e < first + g_local)
        off = 0 if mutant == "pt_sentinel" else sentinel
        if lookup == "highest":
            # count + base per table (all below 2**20), one f32 matmul at HIGHEST precision:
            # exact, because the one-hot's lower bf16 parts are zero.
            cnt = cnt_scr[r]
            tabs = [cnt + bases_f[2 + s]]
            if local:
                tabs += [cnt + bases_f[0], cnt + bases_f[1]]
            if mutant == "pt_precision":
                tabs = [t - jnp.floor(t * (1.0 / 65536.0)) * 65536.0 for t in tabs]  # drops the 3rd byte
            v = jnp.dot(jnp.concatenate(tabs, axis=0), oh_lo, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
            vals = [v[k * hi_n:(k + 1) * hi_n] for k in range(len(tabs))]
        else:
            cnt = cnt_scr[r].astype(jnp.int32)
            tabs = [_planes(cnt, COUNT_PLANES), base_planes[2 + s]]
            if local:
                tabs += [base_planes[0], base_planes[1]]
            v = jnp.dot(jnp.concatenate(tabs, axis=0), oh_lo.astype(jnp.bfloat16),
                        preferred_element_type=jnp.float32)
            cnt_v = planes_at(v, 0, COUNT_PLANES)
            k = COUNT_PLANES
            vals = [cnt_v + planes_at(v, k + t * BASE_PLANES, BASE_PLANES) for t in range(3 if local else 1)]
        slab = (pick(vals[0], oh_hi) + rank).astype(jnp.int32)
        slab_ref[pl.ds(r, 1), :] = jnp.where(on_me, slab, jnp.int32(off))
        if local:
            pos_ref[pl.ds(i, 1), :] = (pick(vals[1], oh_hi) + rank).astype(jnp.int32)
            mir_ref[pl.ds(i, 1), :] = (pick(vals[2], oh_hi) + rank).astype(jnp.int32)

    def sweep(s, local, with_next):
        """Source s's lookups; with `with_next`, source s + 1's histograms and ranks in the
        same iterations (independent work the scheduler can interleave)."""
        def body(i, carry):
            if with_next:
                phase1(s + 1, i, carry)
            lookup_row(s, local, i)
            return carry

        lax.fori_loop(0, rows, body, jnp.int32(0), unroll=unroll[2])

    def carry_in(run):
        return run if mutant == "pt_src_reset" else zero

    if fuse:
        lax.fori_loop(0, rows, functools.partial(phase1, 0), jnp.int32(0), unroll=unroll[0])
        run = prefix(0, zero)
        for s in range(ep):
            nxt = s + 1 < ep
            pl.when(me == s)(functools.partial(sweep, s, True, nxt))
            pl.when(me != s)(functools.partial(sweep, s, False, nxt))
            if nxt:
                run = prefix(s + 1, carry_in(run))
    else:
        for src in range(ep):
            lax.fori_loop(0, rows, functools.partial(phase1, src), jnp.int32(0), unroll=unroll[0])
        run = zero
        for s in range(ep):
            run = prefix(s, carry_in(run))
        for s in range(ep):
            pl.when(me == s)(functools.partial(sweep, s, True, False))
            pl.when(me != s)(functools.partial(sweep, s, False, False))


# MO4 (R1 push, MOE): the EP activation all-gather as DMAs issued at the plan kernel's head and waited at
# its tail, so the 17.8 MB per core moves under the kernel's own ~39 us of routing compute. On this stack
# XLA emits every all-gather synchronously, so the exposed collective (28.4 us per call on base13) cannot
# be overlapped by scheduling. Same collective id as the FFN kernel: each kernel's barrier completes
# inside that kernel, and the two never run concurrently.
AG_COLLECTIVE_ID = 0
_AG_MUTANTS = ("ag_slot",)


def _ag_barrier(ep):
    """kernel._all_pairs_barrier: every shard is inside this call before any remote write lands."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(ep):
        pl.semaphore_signal(barrier_sem, inc=1, device_id=(jnp.int32(i),), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, ep)

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        for i in range(ep):
            pl.semaphore_signal(second, inc=1, device_id=(jnp.int32(i),), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(second, ep)


def _plan_body_ag(me_ref, pairs_ref, bases_ref, ql_ref, pos_ref, mir_ref, slab_ref, qg_ref, cnt_scr, rank_scr,
                  *rest, ep, t_local, ag_mutant, **kw):
    """_plan_body with the activation all-gather riding on it: q_g[s * t_local + r] = shard s's q_l[r].

    Head: a barrier (the peers' q_g exists and is this call's), then this shard's rows to its own slot
    locally and to the same slot of every peer. Tail: the local copy, the sends and the ep - 1 arrivals.
    Pure data movement, so q_g is the all-gather's result bit for bit."""
    *rest, local_sem, send_sem, recv_sem = rest
    me = me_ref[0]
    # Test mutant "ag_slot": the rows land one slot over (same bytes, so nothing hangs).
    slot = (me + 1) % ep if ag_mutant == "ag_slot" else me
    _ag_barrier(ep)
    local = pltpu.make_async_copy(ql_ref, qg_ref.at[pl.ds(slot * t_local, t_local)], local_sem)
    local.start()
    sends = []
    for d in range(ep):
        cp = pltpu.make_async_remote_copy(
            src_ref=ql_ref, dst_ref=qg_ref.at[pl.ds(slot * t_local, t_local)], send_sem=send_sem,
            recv_sem=recv_sem, device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH)
        sends.append((d, cp))

        @pl.when(me != d)
        def _(cp=cp):
            cp.start()

    _plan_body(me_ref, pairs_ref, bases_ref, pos_ref, mir_ref, slab_ref, cnt_scr, rank_scr, *rest, ep=ep, **kw)
    local.wait()
    for d, cp in sends:
        @pl.when(me != d)
        def _(cp=cp):
            cp.wait_send()
            cp.wait_recv()


# Rows per loop iteration in the three phases: independent rows (1 and 3) let
# the scheduler overlap one row's MXU and XLU latency with the next row's VALU
# work; the prefix (2) is one add per row on a serial chain.
UNROLL = (16, 8, 8)  # l1rt3b j-7f656e51: kernel 39.6 us at 8,704 vs 57.4 at (4, 8, 4), tables bitwise
# Source s's lookups sharing their loop with source s + 1's histograms (phase 1): off by default,
# the static screen found no gain (l1 K2 notes).
FUSE = False
# The lookup matmul: "highest" (f32 tables, Precision.HIGHEST) or "planes" (exact bf16 byte planes).
LOOKUP = "highest"


def plan_tables_kernel(pairs, bases, me, *, e_total, ep, t_local, topk, shard_stride,
                       interpret=None, mutant=None, unroll=None, fuse=None, lookup=None,
                       exchange_rows=None, gather_rows=None):
    """pos [t_local, topk], mirror_pos [t_local, topk], slab_row [ep * t_local * topk].

    pairs: [ep, t_local * topk] int32, each source's pair expert ids in pair
    order (the exchange's pair section); or, with `exchange_rows` = W, the
    gathered exchange itself as [ep, W, 128] int32 lane rows (each source's
    W rows start with its pair section), read in place. bases:
    plan_bases(...). me: int32 scalar shard index.
    """
    if not plan_kernel_supported(e_total=e_total, ep=ep, t_local=t_local, topk=topk):
        raise ValueError(f"plan kernel: unsupported shape e_total={e_total} ep={ep} "
                         f"t_local={t_local} topk={topk}")
    mutant = envs.MOE_FUSED_EP_V2_TEST_MUTANT if mutant is None else mutant
    if mutant not in _MUTANTS:
        mutant = ""
    if interpret is None:
        interpret = envs.MOE_FUSED_EP_V2_INTERPRET
    rows = t_local * topk // LANES
    hi_n = e_total // LO
    g_local = e_total // ep
    sentinel = host.off_shard_slab_row(shard_stride, ep)
    vm = pltpu.MemorySpace.VMEM
    unroll = tuple(UNROLL if unroll is None else unroll)
    rows_pad = -(-rows // 8) * 8
    if exchange_rows:
        if exchange_rows < rows_pad or pairs.shape != (ep, exchange_rows, LANES):
            raise ValueError(f"plan kernel: exchange rows {pairs.shape} do not hold {ep} sources of "
                             f"{exchange_rows} rows with {rows_pad} pair rows each")
        pairs_in = pairs
        pairs_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
        extra_scratch = [pltpu.VMEM((ep * rows_pad, LANES), jnp.int32), pltpu.SemaphoreType.DMA(())]
    else:
        pairs_in = pairs.reshape(ep * rows, LANES)
        pairs_spec = pl.BlockSpec(memory_space=vm)
        extra_scratch = []
    kw = dict(rows=rows, ep=ep, hi_n=hi_n, e_total=e_total, g_local=g_local, sentinel=int(sentinel),
              mutant=mutant, unroll=unroll, fuse=FUSE if fuse is None else bool(fuse), lookup=lookup or LOOKUP,
              exchange_rows=int(exchange_rows or 0))
    m_all = envs.MOE_FUSED_EP_V2_TEST_MUTANT
    ag_mutant = m_all if m_all in _AG_MUTANTS else ""
    ag = gather_rows is not None
    if ag:
        if gather_rows.shape[0] != t_local:
            raise ValueError(f"plan kernel: gather rows {gather_rows.shape} are not this shard's {t_local}")
        body = functools.partial(_plan_body_ag, t_local=t_local, ag_mutant=ag_mutant, **kw)
    else:
        body = functools.partial(_plan_body, **kw)
    name = ("fused_ep_plan_tables" + ("_x" if exchange_rows else "") + ("_ag" if ag else "")
            + (f"_mut{mutant}" if mutant else "") + (f"_mut{ag_mutant}" if ag and ag_mutant else ""))
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = [pl.BlockSpec(memory_space=pltpu.SMEM), pairs_spec, pl.BlockSpec(memory_space=vm)]
    out_specs = [pl.BlockSpec(memory_space=vm)] * 3
    out_shape = [jax.ShapeDtypeStruct((rows, LANES), jnp.int32),
                 jax.ShapeDtypeStruct((rows, LANES), jnp.int32),
                 jax.ShapeDtypeStruct((ep * rows, LANES), jnp.int32)]
    scratch = [pltpu.VMEM((ep * rows, hi_n, LO), jnp.float32),
               pltpu.VMEM((ep * rows, LANES), jnp.float32)] + extra_scratch
    args = [jnp.asarray(me, jnp.int32).reshape(1), pairs_in, bases.reshape(2 + ep, hi_n, LO)]
    params = {}
    if ag:
        in_specs.append(hbm)
        out_specs.append(hbm)
        out_shape.append(jax.ShapeDtypeStruct((ep * t_local,) + tuple(gather_rows.shape[1:]), gather_rows.dtype))
        scratch += [pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(())]
        args.append(gather_rows)
        params["compiler_params"] = pltpu.CompilerParams(collective_id=AG_COLLECTIVE_ID)
    res = pl.pallas_call(
        body,
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        out_shape=tuple(out_shape),
        scratch_shapes=scratch,
        interpret=(pltpu.InterpretParams() if interpret else False),
        name=name,
        **params,
    )(*args)
    pos, mir, slab = res[:3]
    tables = (pos.reshape(t_local, topk), mir.reshape(t_local, topk), slab.reshape(-1))
    return tables + ((res[3],) if ag else ())


def plan_shard_tables(pairs, rows_by_dest, me, *, e_total, ep, t_local, topk, tile_m, shard_stride,
                      exchange_rows=None, interpret=None, gather_rows=None):
    """host.ShardRoutingTables for shard `me` with the per-pair tables from the plan kernel
    and the small tables from plan_small_tables: the same values as
    host.build_routing_tables_sharded (every small table, slab_row, and pos / mirror_pos on
    live pairs)."""
    small, bases = plan_small_tables(rows_by_dest, me, e_total=e_total, ep=ep, tile_m=tile_m,
                                     shard_stride=shard_stride)
    out = plan_tables_kernel(pairs, bases, me, e_total=e_total, ep=ep, t_local=t_local, topk=topk,
                             shard_stride=shard_stride, interpret=interpret,
                             exchange_rows=exchange_rows, gather_rows=gather_rows)
    pos, mir, slab = out[:3]
    tables = host.ShardRoutingTables(pos=pos, mirror_pos=mir, slab_row=slab, rows_alloc=ep * shard_stride,
                                     n_tokens=ep * t_local, topk=topk, **small)
    # With gather_rows the kernel also returns the all-gathered rows (MO4).
    return (tables, out[3]) if gather_rows is not None else tables


def _xcumsum(x, axis):
    """Exclusive cumsum of a non-negative int32 table along `axis`. With
    MOE_FUSED_EP_V2_PLAN_CUMSUM_DOT it is a HIGHEST-precision matmul with a strictly upper-triangular
    ones matrix, exact while every partial sum stays below 2**24 (the plan's row counts are below
    2**20), instead of XLA's reduce-window cumsum. Test mutant "cs_incl": inclusive."""
    if not envs.MOE_FUSED_EP_V2_PLAN_CUMSUM_DOT:
        return jnp.cumsum(x, axis=axis) - x
    n = x.shape[axis]
    i = jnp.arange(n, dtype=jnp.int32)
    incl = envs.MOE_FUSED_EP_V2_TEST_MUTANT == "cs_incl"
    tri = ((i[:, None] <= i[None, :]) if incl else (i[:, None] < i[None, :])).astype(jnp.float32)
    xm = jnp.moveaxis(x, axis, -1).astype(jnp.float32)
    y = jnp.dot(xm, tri, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return jnp.moveaxis(y.astype(jnp.int32), -1, axis)


def _slab_base(expert_base, run_start_aligned, rows_by_dest):
    """expert_base + run_start_aligned[:, s] per source s, as [ep, e_total] i32. With
    MOE_FUSED_EP_V2_SLAB_BASE_T the run starts come from the exchanged count table in its own
    [ep, e_total] orientation (rows aligned, exclusive cumsum over the sources), so no [e_total, ep]
    transpose is built; integer-exact, the same bits. Kept at the file end, as _xcumsum, so the
    Mosaic bodies above keep their line numbers. Test mutant "sb_incl": inclusive cumsum."""
    if not envs.MOE_FUSED_EP_V2_SLAB_BASE_T:
        return expert_base[None, :] + run_start_aligned.T
    rows_aligned = (rows_by_dest + (host.ROWBLK - 1)) & jnp.int32(-host.ROWBLK)
    start = _xcumsum(rows_aligned, 0)
    if envs.MOE_FUSED_EP_V2_TEST_MUTANT == "sb_incl":
        start = start + rows_aligned
    return expert_base[None, :] + start

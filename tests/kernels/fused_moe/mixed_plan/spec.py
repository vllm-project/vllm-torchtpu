# SPDX-License-Identifier: Apache-2.0
"""One spec for the mixed-step routing-plan instruments (accuracy test + bench).

The block is the XLA glue fused_ep_moe_v2.compute_local runs around the
fused-EP kernel at MIXED buckets (t_local > SMALL_BATCH_LOCAL_TOKENS, where the
decode Pallas plan does not apply): gathered routing exchange -> every kernel
operand (transport tables, token gather, activation scale slab, slab bases,
visit order, count vector) + the combine's arrival tables (pos, mirror_pos),
and the combine itself. `incumbent_plan` is that code path verbatim
(sharded_plan=True); `oracle_plan` re-derives every output from first
principles in numpy int64; `FAULTS` are the failure classes a candidate can
have, used as the positive control.

Cells are per-device shapes of the served deployments (DeepSeek-V4-Pro, EP16,
384 experts, top-6, fp8 activations, capacity 128):
  mix128 -- c16 T512 (DP4xTP4, SP: each rank routes T/4) and c4/c8 T1024 (TP8)
  mix256 -- c4/c8 T2048
  mix32/mix64 -- c4/c8 T256/T512 (rare buckets), the edge of the mixed range
Inputs follow the served routing distribution: per-layer expert shares from
the c16 expert-count taps (j-af3b5b8a) under the stack's placement table.
"""

import dataclasses
import functools
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v2 import host
from vllm_torchtpu.kernels.fused_moe.v2.layer import (
    _combine_arrivals,
    _exchange_layout,
    _pack_routing_exchange,
    _unpack_routing_exchange,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SERVED = os.path.join(HERE, "served_routing_c16.npz")


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    t_local: int
    ep: int = 16
    e_total: int = 384
    topk: int = 6
    tile_m: int = 128
    hidden: int = 7168

    @property
    def g_local(self):
        return self.e_total // self.ep

    @property
    def T(self):
        return self.ep * self.t_local

    @property
    def P(self):
        return self.t_local * self.topk

    @property
    def block(self):
        return host.routing_block(self.t_local, self.topk)

    @property
    def stride(self):
        return host.ragged_stride_bound(self.T, self.topk, self.e_total, self.tile_m)

    @property
    def layout(self):
        return _exchange_layout(self.t_local, self.topk, self.e_total, True)

    @property
    def recv_rows(self):
        return host.align_up(
            self.P + (host.ROWBLK - 1) * self.e_total, host.ROWBLK
        )

    @property
    def scale_rows(self):
        return host.act_scale_slab_rows(self.stride)

    @property
    def gather_rows(self):
        return self.stride + host.token_gather_window_rows(self.tile_m)


CELLS = {c.name: c for c in (
    Cell("mix32", 32), Cell("mix64", 64), Cell("mix128", 128), Cell("mix256", 256),
    # the /workload-microbenchmark twins: c16 TP4 twin (EP4, 96 experts), c4/c8 TP8 twin (EP8, 192)
    Cell("tw16", 128, ep=4, e_total=96), Cell("tw4", 256, ep=8, e_total=192),
)}
# Production cells (the operating points); mix32/64 are edge cells.
PROD_CELLS = ("mix128", "mix256")
KINDS = ("served", "served_pad", "uniform", "hot", "one_shard", "unrouted", "dup")


@functools.lru_cache(None)
def _served():
    z = np.load(SERVED)
    return {k: z[k] for k in z.files}


def _select(rng, logp, T, K):
    """Top-K distinct experts per token by Gumbel-top-k on log shares."""
    g = rng.gumbel(size=(T, logp.shape[0])) + logp[None, :]
    return np.argsort(-g, axis=1)[:, :K]


def make_pairs(cell, kind, seed, layer=20, placement="ep16", pad_frac=0.25):
    """[ep, t_local * topk] i32 selections in SLOT ids (e_total = routes nowhere)."""
    rng = np.random.default_rng(seed)
    E, T, K, G = cell.e_total, cell.T, cell.topk, cell.g_local
    if kind in ("served", "served_pad"):
        s = _served()
        p = s["p"][layer].astype(np.float64)
        p = np.maximum(p, 1e-9)
        sel = _select(rng, np.log(p / p.sum()), T, K)
        sel = s["slots_" + placement][layer][sel]
    elif kind in ("hot", "dup"):
        hot = rng.choice(E, max(K, 8), replace=False)
        sel = np.stack([rng.choice(hot, K, replace=False) for _ in range(T)])
    elif kind == "one_shard":
        pool = 3 * G + np.arange(G)
        sel = np.stack([rng.choice(pool, K, replace=False) for _ in range(T)])
    else:  # uniform, unrouted
        sel = _select(rng, np.zeros(E), T, K)
    sel = sel.astype(np.int32)
    if kind == "dup":
        # A token selecting one expert twice (never produced by top-k) over
        # 8 hot experts: the candidate's exactness edge, run prefix counts
        # above 255 at t_local 256.
        d = rng.random(T) < 0.5
        sel[d, :] = sel[0, 0]  # half the tokens put all K picks on one expert
    if kind == "unrouted":
        sel[rng.random(T) < 0.4] = E
    if kind == "served_pad":
        # Bucket padding sits at the tail of the batch; under the token split
        # each source owns a contiguous slice, so the tail sources carry it.
        n_pad = int(T * pad_frac)
        if n_pad:
            sel[T - n_pad :] = E
    return sel.reshape(cell.ep, cell.P)


def make_scale_bits(cell, seed):
    rng = np.random.default_rng(seed + 7)
    s = rng.uniform(1e-3, 0.5, cell.T).astype(np.float32)
    return s.view(np.int32).reshape(cell.ep, cell.t_local)


def counts_of(cell, pairs):
    c = np.zeros((cell.ep, cell.e_total), np.int32)
    for d in range(cell.ep):
        v = pairs[d][(pairs[d] >= 0) & (pairs[d] < cell.e_total)]
        np.add.at(c[d], v, 1)
    return c


def busiest_shard(cell, pairs):
    c = counts_of(cell, pairs)
    al = (c + host.ROWBLK - 1) // host.ROWBLK * host.ROWBLK
    return int(np.argmax(al.sum(0).reshape(cell.ep, cell.g_local).sum(1)))


def make_exchange(cell, pairs, sbits):
    """The gathered [ep, width] routing payload, exactly as select_local packs it."""
    c = counts_of(cell, pairs)
    rows = [
        _pack_routing_exchange(
            jnp.asarray(pairs[d]), jnp.asarray(sbits[d]), jnp.asarray(c[d]), cell.layout
        )
        for d in range(cell.ep)
    ]
    return jnp.stack(rows).astype(jnp.int32)


def local_inputs(cell, pairs, me):
    local_blocks = jnp.asarray(pairs[me]).reshape(-1, cell.block)
    return local_blocks, host.pair_block_hist(local_blocks, cell.e_total)


OUT_NAMES = (
    "tables", "token_gather", "scale_slab", "expert_rows", "slab_base",
    "visit", "counts", "pos", "mirror_pos",
)


def incumbent_plan(cell, exchange, local_blocks, block_hist, me):
    """layer.compute_local's glue (sharded_plan=True), stack d1ba4b9c5, verbatim.

    me is a traced int32 scalar, as rank_l[0, 0] is in the layer. The AGQ
    optimization barrier (slab_row with q_g) is omitted: there is no q_g here.
    scale_slab is returned as its int32 bits.
    """
    E, ep, tl, K = cell.e_total, cell.ep, cell.t_local, cell.topk
    pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
        exchange, cell.layout, tl, K, E
    )
    gathered = host.GatheredPairs(
        expert_blocks=pairs_g.reshape(cell.T * K // cell.block, cell.block),
        local_blocks=local_blocks,
        block_hist=block_hist,
        rows_by_dest=rows_by_dest,
        n_tokens=cell.T,
        topk=K,
    )
    kw = dict(e_total=E, ep=ep, t_local=tl, block=cell.block, tile_m=cell.tile_m,
              shard_stride=cell.stride)
    routing = host.build_routing_tables_sharded(None, me, gathered=gathered, **kw)
    block_tables = host.shard_transport_tables_in_blocks(routing, me, e_total=E, ep=ep)
    row_tables = host.shard_push_tables_in_rows(routing, me, e_total=E, ep=ep)
    slab_row = host.local_slab_rows(routing, me, shard_stride=cell.stride)
    token_gather = host.shard_token_gather(routing, me, shard_stride=cell.stride, rows=slab_row)
    token_gather = jnp.pad(token_gather, (0, host.token_gather_window_rows(cell.tile_m)))
    scale_bits = jnp.repeat(scale_bits_g, K)
    scale_slab = (
        jnp.zeros((cell.scale_rows * host.HIDDEN_LANE_BLOCK,), jnp.int32)
        .at[slab_row]
        .add(scale_bits, mode="drop")
        .reshape(cell.scale_rows, host.HIDDEN_LANE_BLOCK)
    )
    expert_rows, slab_base = host.shard_expert_slabs(routing, me, e_total=E, ep=ep)
    visit, _ = host.expert_visit_list(expert_rows, cell.g_local)
    counts = host.shard_count_vector(routing, expert_rows, me, e_total=E, ep=ep)
    kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
    return dict(
        tables=jnp.stack(kernel_tables),
        token_gather=token_gather,
        scale_slab=scale_slab,
        expert_rows=expert_rows,
        slab_base=slab_base,
        visit=visit,
        counts=counts,
        pos=routing.pos,
        mirror_pos=routing.mirror_pos,
    )


# ---------------------------------------------------------------- oracle


def oracle_plan(cell, pairs, sbits, me, fault=None):
    """Every output from first principles, numpy int64. `fault` injects one
    failure class (see FAULTS) so the comparison's discriminating power can be
    measured -- the positive control."""
    E, ep, tl, K, G, RB = cell.e_total, cell.ep, cell.t_local, cell.topk, cell.g_local, host.ROWBLK
    tm, P, B = cell.tile_m, cell.P, cell.block
    sh = tm.bit_length() - 1
    pairs = np.asarray(pairs, np.int64)
    valid = (pairs >= 0) & (pairs < E)
    counts = np.zeros((ep, E), np.int64)
    for d in range(ep):
        np.add.at(counts[d], pairs[d][valid[d]], 1)
    al = counts if fault == "align" else (counts + RB - 1) // RB * RB
    rs = np.cumsum(al, axis=0) - al  # [d, e] exclusive over sources
    er = al.sum(0)  # [E]
    lb = (np.cumsum(er.reshape(ep, G), 1) - er.reshape(ep, G)).reshape(E)
    recv_base = np.cumsum(al, axis=1) - al  # [d, e] exclusive over experts
    first = me * G

    def rank_src(d):
        """Rank of each pair among earlier same-id pairs of source d; for an
        id outside [0, E) only within its routing block."""
        r = np.zeros(P, np.int64)
        seen = {}
        for p in range(P):
            e = pairs[d, p]
            key = e if valid[d, p] else (e, p // B)
            if fault == "invalid_rank" and not valid[d, p]:
                key = e
            r[p] = seen.get(key, 0)
            seen[key] = r[p] + 1
        return r

    # The combine's tables over this shard's own pairs.
    rk = rank_src(me)
    pos = np.zeros(P, np.int64)
    mir = np.zeros(P, np.int64)
    for p in range(P):
        e = pairs[me, p]
        if valid[me, p]:
            r = rk[p] + (1 if fault == "rank_off" and e == np.argmax(counts[me]) else 0)
            pos[p] = recv_base[me, e] + r
            mir[p] = ((recv_base[me, e] >> sh) + 2 * e) * tm + (rs[me, e] & (tm - 1)) + r
        else:
            pos[p] = mir[p] = rk[p]

    # Slab rows of every pair computing here: token gather and scale slab.
    tg = np.zeros(cell.stride, np.int64)
    sc = np.zeros(cell.scale_rows * host.HIDDEN_LANE_BLOCK, np.int64)
    row_stride = tl + 1 if fault == "src_stride" else tl
    for d in range(ep):
        r = rank_src(d)
        mine = valid[d] & (pairs[d] >= first) & (pairs[d] < first + G)
        idx = np.nonzero(mine)[0]
        if fault == "tail_drop" and len(idx):
            idx = idx[:-1]
        for p in idx:
            e = pairs[d, p]
            row = lb[e] + rs[d, e] + r[p]
            t = p // K
            if fault == "run_order" and r[p] in (0, 1) and counts[d, e] >= 2:
                row = lb[e] + rs[d, e] + (1 - r[p])
            tg[row] = min(max(d * row_stride + t, 0), cell.T - 1)
            st = t - 1 if (fault == "scale_stale" and t > 0) else t
            sc[row] = sbits[d, st]
        if fault == "pad_row":
            for g in range(G):
                e = first + g
                if counts[d, e] % RB:
                    tg[lb[e] + rs[d, e] + counts[d, e]] = tg[lb[e] + rs[d, e] + counts[d, e] - 1]
    tg = np.concatenate([tg, np.zeros(host.token_gather_window_rows(tm), np.int64)])

    sel = slice(first, first + G)
    rs_g = rs[:, sel].T
    al_g = al[:, sel].T
    out_total = al_g.sum(0)
    contrib = (np.cumsum(out_total) - out_total)[None, :] + (np.cumsum(al_g, 0) - al_g)
    push = recv_base[:, sel].T
    true_g = counts[:, sel].T
    tables = np.stack([rs_g >> 1, al_g >> 1, contrib >> 1, push >> 1, true_g]) if RB == 2 else None
    er_g = er[sel]
    order = sorted(range(G), key=lambda g: (-er_g[g], g if fault != "visit_tie" else -g))
    visit = np.minimum(np.array(order), G - 1)

    def tiles_of(start, rows):
        last = start + np.maximum(rows, 1) - 1
        return np.where(rows > 0, (last >> sh) - (start >> sh) + 1, 0)

    tiles = tiles_of(rs, al)  # [d, e]
    cme = (me + 1) % ep if fault == "me_off" else me
    mine = (np.arange(ep) == cme).astype(np.int64)
    other = 1 - mine
    tiles_g = tiles[:, sel].T
    cv = np.stack([
        other * true_g.sum(0),
        mine * (counts.sum(1) - true_g.sum(0)),
        other * al_g.sum(0),
        mine * (al.sum(1) - al_g.sum(0)),
        mine * (er_g > 0).sum(),
        other * tiles_g.sum(0),
        mine * (tiles.sum(1) - tiles_g.sum(0)),
        mine * tiles_g.sum(0),
    ])
    return dict(
        tables=tables,
        token_gather=tg,
        scale_slab=sc.reshape(cell.scale_rows, host.HIDDEN_LANE_BLOCK),
        expert_rows=er_g,
        slab_base=lb[sel],
        visit=visit,
        counts=cv,
        pos=pos.reshape(tl, K),
        mirror_pos=mir.reshape(tl, K),
    )


FAULTS = (
    "align",         # ROWBLK alignment of runs skipped
    "rank_off",      # rank +1 for one expert's pairs (stale accumulator / off-by-one)
    "tail_drop",     # last pair of each source not written (dropped tail)
    "src_stride",    # wrong per-source token stride
    "scale_stale",   # scale from the previous token
    "invalid_rank",  # unrouted pair ranked source-wide, not block-local
    "run_order",     # first two rows of a run swapped (unstable order)
    "visit_tie",     # visit ties broken the wrong way
    "pad_row",       # ROWBLK pad row given a token instead of 0
    "me_off",        # count vector built for the wrong shard
)


def compare(want, got, names=OUT_NAMES):
    """Names of outputs that differ (exact integer comparison)."""
    bad = []
    for n in names:
        a, b = want[n], got[n]
        if a is None or b is None:
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.array_equal(a.astype(np.int64), b.astype(np.int64)):
            bad.append(n)
    return bad


# ---------------------------------------------------------------- combine


def make_combine_inputs(cell, pairs, me, seed):
    rng = np.random.default_rng(seed + 11)
    lane_blocks = cell.hidden // host.HIDDEN_LANE_BLOCK
    arr = rng.standard_normal((cell.recv_rows, lane_blocks, host.HIDDEN_LANE_BLOCK)).astype(np.float32)
    arrivals = jnp.asarray(arr).astype(jnp.float8_e4m3fn)
    mirror = host.arrival_mirror_rows(cell.recv_rows, cell.e_total, cell.tile_m)
    arrival_scales = jnp.asarray(rng.uniform(1e-3, 1.0, (mirror, cell.tile_m)).astype(np.float32))
    my = np.asarray(pairs[me]).reshape(cell.t_local, cell.topk)
    w = rng.uniform(0.01, 1.0, my.shape).astype(np.float32)
    routes = (my < cell.e_total).any(1, keepdims=True)
    w = np.where(routes, w / w.sum(1, keepdims=True), 0.0).astype(np.float32)
    return arrivals, arrival_scales, jnp.asarray(w), jnp.asarray(routes)


def incumbent_combine(arrivals, arrival_scales, pos, mirror_pos, weights, row_routes):
    """layer.compute_local's combine (stack d1ba4b9c5), verbatim."""
    out = _combine_arrivals(arrivals, arrival_scales, pos, mirror_pos, weights, jnp.bfloat16)
    return jnp.where(row_routes, out, jnp.zeros_like(out))


def oracle_combine(arrivals, arrival_scales, pos, mirror_pos, weights, row_routes):
    """float64: out[t] = sum_k scale(mirror[t,k]) * w[t,k] * arrivals[pos[t,k]]."""
    a = np.asarray(arrivals.astype(jnp.float32), np.float64).reshape(arrivals.shape[0], -1)
    s = np.asarray(arrival_scales, np.float64).reshape(-1)
    pos = np.minimum(np.asarray(pos), a.shape[0] - 1)
    mir = np.minimum(np.asarray(mirror_pos), s.size - 1)
    coef = s[mir] * np.asarray(weights, np.float64)
    out = np.einsum("tk,tkh->th", coef, a[pos])
    return np.where(np.asarray(row_routes), out, 0.0)


# ---------------------------------------------------------------- candidates


def candidate_plan(cell, exchange, local_blocks, block_hist, me, interpret=False, **kw):
    """Stage A1 (mixed_plan_kernel.build_mixed_plan) in incumbent_plan's output form."""
    del local_blocks, block_hist  # the kernel reads everything from the exchange
    from vllm_torchtpu.kernels.fused_moe.v2.mixed_plan_kernel import build_mixed_plan

    p = build_mixed_plan(
        exchange, jnp.asarray(me, jnp.int32), layout=cell.layout, e_total=cell.e_total,
        ep=cell.ep, t_local=cell.t_local, topk=cell.topk, block=cell.block,
        tile_m=cell.tile_m, shard_stride=cell.stride, interpret=interpret, **kw,
    )
    return dict(
        tables=jnp.stack([*p.block_tables, p.true_rows]),
        token_gather=p.token_gather,
        scale_slab=lax.bitcast_convert_type(p.scale_slab, jnp.int32),
        expert_rows=p.expert_rows,
        slab_base=p.slab_base,
        visit=p.visit,
        counts=p.counts,
        pos=p.pos,
        mirror_pos=p.mirror_pos,
    )


# Candidate faults injected into the KERNEL (positive control, WHITEBOARD v1 5.4).
CANDIDATE_FAULTS = {
    "tele_off": dict(fault="tele_off"),        # telescoping without the k-1 term
    "sign": dict(fault="sign"),                # signed-byte recombine sign error
    "stale_block": dict(fault="stale_block"),  # last live S3 block left zero
    "dup_nohi": dict(dc_hi=False),             # duplicate ids: high plane dropped
    "s1_contrib": dict(fault="s1_contrib"),    # S1: contribution offsets without the per-source term
    "s2_rank": dict(fault="s2_rank"),          # S2: rank without the earlier-block offsets
}


# ---------------------------------------------------------------- combine candidates (stage A2)

# name -> build_mixed_combine kwargs. Timing-only removal arms carry opts from
# mixed_combine_kernel.SKIPS (wrong outputs allowed).
COMBINE_VARIANTS = {
    "a2": dict(gather="bulk", pick="pre", out="reshape"),
    "a2_inline": dict(gather="bulk", pick="inline", out="reshape"),
    "a2_ring": dict(gather="ring", out="reshape"),
    "a2m": dict(gather="bulk", pick="mxu", out="reshape"),
    "a2s": dict(gather="bulk", pick="inline", out="strided"),
    "a2s_ring": dict(gather="ring", out="strided"),
}


def candidate_combine(arrivals, arrival_scales, pos, mirror_pos, weights, row_routes,
                      variant="a2", opts=(), interpret=False):
    """Stage A2 (mixed_combine_kernel.build_mixed_combine) in incumbent_combine's form."""
    from vllm_torchtpu.kernels.fused_moe.v2.mixed_combine_kernel import build_mixed_combine

    return build_mixed_combine(
        arrivals, arrival_scales, pos, mirror_pos, weights, row_routes, jnp.bfloat16,
        opts=opts, interpret=interpret, **COMBINE_VARIANTS[variant],
    )


def forced_pallas_combine(arrivals, arrival_scales, pos, mirror_pos, weights, row_routes):
    """Arm pk128: the incumbent's moe_v2_combine ring forced at t_local 128 (tb 32),
    coef still built in XLA (SC scale gather + fusion). Timing reference only."""
    from vllm_torchtpu.kernels.fused_moe.v2 import kernel as kmod

    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    m, tile_m = arrival_scales.shape
    rows = jnp.minimum(pos, recv_rows - 1)
    scl = jnp.minimum(mirror_pos, m * tile_m - 1)
    coef = arrival_scales.reshape(-1)[scl] * weights.astype(jnp.float32)
    saved = kmod.COMBINE_MIN_TOKENS
    kmod.COMBINE_MIN_TOKENS = min(saved, t_local)
    try:
        fn = kmod._build_combine_kernel(
            t_local=t_local, topk=topk, hidden=lane_blocks * lanes, lane_blocks=lane_blocks,
            wire_dtype=arrivals.dtype, out_dtype=jnp.bfloat16)
    finally:
        kmod.COMBINE_MIN_TOKENS = saved
    out = fn(arrivals, coef, rows)
    return jnp.where(row_routes, out, jnp.zeros_like(out))


def nan_poison(arrivals, pos, row_routes):
    """Arrival rows read only by unrouted tokens -> NaN (they may be unwritten in
    the layer). Rows a routed token reads stay finite."""
    a = np.array(arrivals.astype(jnp.float32))
    p = np.minimum(np.asarray(pos), a.shape[0] - 1)
    r = np.asarray(row_routes).reshape(-1)
    used = np.zeros(a.shape[0], bool)
    used[p[r].reshape(-1)] = True
    a[~used] = np.nan
    return jnp.asarray(a).astype(arrivals.dtype)

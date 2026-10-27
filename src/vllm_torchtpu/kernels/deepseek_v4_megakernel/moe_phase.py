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
"""The MoE phase of the DSv4 whole-step kernel (step_phases.moe_phase).

Per layer, on every rank: route all T global rows (noaux_tc top-k on the
skeleton's gate logits, or the hash layers' tid2eid ids), run this rank's
local experts (EP, served fp4 operands read in place, moe_experts.MoEBlock in
layout "stock") onto the shared-expert partial the skeleton seeded, then the
hierarchical all-reduce into ``out``. All-reduce buffers alternate by layer
parity; every send is retired before the phase returns.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import moe_router as rt
from . import vmem_alias
from .allreduce import AllReduce
from .moe_experts import F32, FP4, I32, U32, MoEBlock, MoEConfig

RING = (
    "w1_vm",
    "w2_vm",
    "es1",
    "es2",
)  # the buffers a PoolLease can hold (stock layout)


# Work-item size: slices per local expert. 8 (384 columns, two scale-block
# pieces) is ~6% faster isolated; 12 (256 columns, one piece) issues half the
# DMAs per item, which shrinks the per-layer dispatch code (580 vs 780 KB at
# L=61, T=16). Read at trace time.
EP_SLICES = 8


def phase_config(geom, nbuf=4):
    """The MoEConfig the phase runs for a StepGeometry."""
    return MoEConfig(
        R=geom.R,
        mode="ep",
        layout="stock",
        rows=-(-geom.T // 32) * 32,
        ar_rows=geom.T,
        hidden=geom.hidden,
        inter=geom.moe_inter,
        n_experts=geom.n_experts,
        topk=geom.topk,
        nbuf=nbuf,
        ar_mode="hrsag",
        ar_slots=2,
        ep_slices=EP_SLICES,
        w1_cols=math.gcd(256, 2 * geom.moe_inter // EP_SLICES),
        item_groups=1,  # unrolled: rolled iterations expose MXU drains (+1 us/item)
        w1_chunks=1,  # unused by the stock layout (its own per-piece DMAs)
    )


# Time-attribution knobs, read at trace time (numerics are wrong when set).
ABLATE: frozenset[str] = frozenset()


def _ring_nbuf(geom, leased):
    """Ring depth: 4 slots, or as many as the lease holds (>= 2)."""
    if not leased:
        return 4
    for nbuf in (4, 3, 2):
        if _ring_bytes(phase_config(geom, nbuf)) <= geom.lease_bytes:
            return nbuf
    raise ValueError(f"MoE ring needs {_ring_bytes(phase_config(geom, 2))} lease bytes")


def _ring_bytes(cfg):
    return sum(
        vmem_alias.storage_size(v.shape, v.dtype)
        for k, v in MoEBlock.scratch(cfg).items()
        if k in RING
    )


def _route_groups(geom, cfg):
    """Number of intra-host ranks that shard non-hash router top-k selection."""
    if "unsharded_route" in ABLATE or geom.R < 2:
        return 1
    return math.gcd(min(geom.R, 8), cfg.rows // 8)


def _scratch(geom, leased, own_ar=True):
    cfg = phase_config(geom, _ring_nbuf(geom, leased))
    rg = _route_groups(geom, cfg)
    blk = {k: v for k, v in MoEBlock.scratch(cfg).items() if not (leased and k in RING)}
    extra = {
        "bias_vm": pltpu.VMEM((1, geom.n_experts), F32),
        "sel_ids": pltpu.VMEM((cfg.rows, 128), I32),
        "sel_w": pltpu.VMEM((cfg.rows, 128), F32),
        "psem": pltpu.SemaphoreType.DMA((1,)),
        **(
            {
                "sel_src": pltpu.VMEM((cfg.rows // rg, 128), F32),
                "rsem": pltpu.SemaphoreType.DMA((2, rg)),
            }
            if rg > 1
            else {}
        ),
    }
    return cfg, {**blk, **extra, **(cfg.ar.scratch() if own_ar else {})}


def moe_scratch_shapes(geom, leased, own_ar=True):
    """Phase scratch; with own_ar=False the caller passes its AllReduce (an
    AllReduceSpec(R, T, hidden, "hrsag", slots=2) site it shares)."""
    return list(_scratch(geom, leased, own_ar)[1].values())


class _PhaseBlock(MoEBlock):
    """MoEBlock whose weights are picked by a traced layer: either stacked
    per-rank arrays [L, EL, ...] (indexed in the DMA: no branch) or per-layer
    arrays (LayerRefs: each item's DMAs issued from one leaf of a binary
    dispatch over the layer). Waits use layer 0's views (a wait only needs the
    transfer shape)."""

    def __init__(
        self, cfg, s, me, layer_refs, layer, expert_block=None, placement=None
    ):
        self.layer_refs, self.layer = layer_refs, layer
        self.is_stacked = tuple(not hasattr(r, "refs") for r in layer_refs)
        self.stacked = all(self.is_stacked)
        first = tuple(
            r.at[0] if st else r.refs[0] for r, st in zip(layer_refs, self.is_stacked)
        )
        super().__init__(cfg, s, me, first, expert_block, placement)

    def start_item(self, i, slot):
        pre = self.item_dma_args(i)
        if self.stacked:
            self.issue_item(
                i, slot, tuple(r.at[self.layer] for r in self.layer_refs), pre=pre
            )
            return
        first_lr = next(
            lr for lr, st in zip(self.layer_refs, self.is_stacked) if not st
        )
        cases = list(enumerate(first_lr.layers))
        _dispatch(
            self.layer,
            cases,
            lambda k: self.issue_item(
                i,
                slot,
                tuple(
                    lr.at[self.layer] if st else lr.refs[k]
                    for lr, st in zip(self.layer_refs, self.is_stacked)
                ),
                pre=pre,
            ),
        )


def _dispatch(layer, cases, fn):
    """fn(k) for the case whose layer id equals the traced `layer`, by binary
    search over the sorted ids: log2(L) branches per issue rather than L
    (a linear chain of 61 branches cost ~0.45 us per item)."""
    cases = sorted(cases, key=lambda c: c[1])
    if len(cases) == 1:
        k, lid = cases[0]
        pl.when(layer == lid)(lambda: fn(k))
        return
    mid = len(cases) // 2
    pivot = cases[mid][1]
    pl.when(layer < pivot)(lambda: _dispatch(layer, cases[:mid], fn))
    pl.when(layer >= pivot)(lambda: _dispatch(layer, cases[mid:], fn))


def _as_packed(lr):
    """fp4 served arrays (stacked ref or LayerRefs) viewed as packed u32
    along K (see MoEConfig.layout)."""
    from .step_phases import LayerRefs

    if not hasattr(lr, "refs"):
        return lr.bitcast(U32) if lr.dtype == FP4 else lr
    if lr.refs[0].dtype != FP4:
        return lr
    return LayerRefs([r.bitcast(U32) for r in lr.refs], lr.layers)


def _smem_columns(read, n_cols, n, rows):
    """n_cols [rows, 1] int32 vectors; entry t < n of column k is read(t, k)
    (SMEM scalars). A rolled loop: code does not grow with n."""
    iota = lax.broadcasted_iota(I32, (rows, 1), 0)

    def body(t, cols):
        return tuple(jnp.where(iota == t, read(t, k), c) for k, c in enumerate(cols))

    init = tuple(jnp.zeros((rows, 1), I32) for _ in range(n_cols))
    return list(lax.fori_loop(0, n, body, init))


CHUNK = 1024  # max columns per iteration of the rolled row-wide loops


def _chunk(width):
    c = math.gcd(width, CHUNK)
    if c % 128:
        raise ValueError(f"row width {width} is not a multiple of 128")
    return c


def _over_cols(width, fn):
    """fn(column slice, chunk width) over all of `width` in a rolled loop."""
    c = _chunk(width)
    lax.fori_loop(
        0, width // c, lambda i, k: (fn(pl.ds(pl.multiple_of(i * c, c), c), c), k)[1], 0
    )


def _quantize_rows(s, x, T, M, amax=None):
    """moe_router.rowquant_fp8 of x [T, H] bf16 into s["xq"] (rows >= T zero)
    and s["xs"], in two rolled passes (row max, then quantize)."""
    H = x.shape[1]
    if amax is None:
        ch = _chunk(H)

        def amax_chunk(c, m):
            v = x[:, pl.ds(pl.multiple_of(c * ch, ch), ch)]
            return jnp.maximum(m, jnp.max(jnp.abs(v), axis=-1, keepdims=True))

        amax = lax.fori_loop(0, H // ch, amax_chunk, jnp.zeros((T, 1), x.dtype))
    scale = amax.astype(F32) / rt.FP8_MAX
    usable = (scale > 0) & (scale < jnp.inf)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0).astype(x.dtype)

    def quant(cs, w):
        v = x[:, cs]
        s["xq"][0:T, cs] = jnp.where(usable, v * sinv, jnp.zeros_like(v)).astype(rt.FP8)
        if M > T:
            s["xq"][T:M, cs] = jnp.zeros((M - T, w), rt.FP8)

    _over_cols(H, quant)
    full = jnp.concatenate([scale, jnp.zeros((M - T, 1), F32)]) if M > T else scale
    s["xs"][...] = jnp.broadcast_to(full, s["xs"].shape)


def moe_phase(
    geom,
    layer,
    is_hash,
    me,
    inputs,
    weights,
    out,
    lease,
    scratch,
    ar=None,
    ar_slot=None,
    expert_block=None,
    before_ar=None,
    expert_held=None,
    expert_copy=None,
    shared_expert=None,
):
    """expert_block: which EP block of the global expert ids this rank's
    expert arrays hold (default `me`; they differ when the mesh is ordered by
    device id but the experts were sharded by EP rank).
    expert_held / expert_copy: a static expert placement instead (this
    rank's rows of placement_tables): SMEM int32 [L, slots] global expert in
    each local slot of each layer (the served arrays are in slot order), and
    None (no replicas) or [L, slots] copy_idx << 8 | n_copies."""
    leased = lease is not None and lease.pool is not None
    cfg, spec = _scratch(geom, leased, own_ar=ar is None)
    if ar is not None:
        sp = ar.spec
        if (sp.R, sp.rows, sp.hidden) != (
            geom.R,
            geom.T,
            geom.hidden,
        ) or ar_slot is None:
            raise ValueError(
                f"shared all-reduce {sp} / slot {ar_slot} does not fit {geom}"
            )
    s = dict(zip(spec, scratch))
    if leased:
        off = 0
        for k in RING:
            v = MoEBlock.scratch(cfg)[k]
            n = vmem_alias.storage_size(v.shape, v.dtype) // 128
            s[k] = vmem_alias.view(
                lease.pool.at[pl.ds(lease.row0 + off, n)], v.shape, v.dtype
            )
            off += n
    T, M, E = geom.T, cfg.rows, geom.n_experts
    rg = _route_groups(geom, cfg)
    rpo = M // rg
    bcp = pltpu.make_async_copy(
        weights.bias.at[pl.ds(layer, 1)], s["bias_vm"], s["psem"].at[0]
    )
    bcp.start()
    lrefs = tuple(
        _as_packed(lr)
        for lr in (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale)
    )
    placement = None if expert_held is None else (expert_held, expert_copy, layer)
    blk = _PhaseBlock(cfg, s, me, lrefs, layer, expert_block, placement)

    def pad(v, fill=0.0):
        return (
            v
            if M == T
            else jnp.concatenate([v, jnp.full((M - T,) + v.shape[1:], fill, v.dtype)])
        )

    with jax.named_scope("moe_route"):
        _quantize_rows(s, inputs.x, T, M, amax=getattr(inputs, "amax", None))
        live = _smem_columns(lambda t, k: inputs.row_valid[t], 1, T, M)[0] > 0
        logits = pad(inputs.logits[...])
        bcp.wait()
        lane = lax.broadcasted_iota(I32, (M, 128), 1)

        def select(hashed):
            if hashed:
                hids = _smem_columns(
                    lambda t, k: inputs.hash_ids[layer, t * 8 + k], cfg.topk, T, M
                )
                ids, ws = rt.route(
                    logits,
                    hash_ids=hids,
                    live_rows=live,
                    topk=cfg.topk,
                    n_experts=E,
                    routed_scaling_factor=cfg.routed_scaling_factor,
                )
            elif rg > 1:
                owner = lax.rem(me, rg)
                my_logits = logits[:rpo]
                live_i32 = live.astype(I32)
                my_live_i32 = live_i32[:rpo]
                for g in range(1, rg):
                    is_g = owner == g
                    my_logits = jnp.where(
                        is_g, logits[g * rpo : (g + 1) * rpo], my_logits
                    )
                    my_live_i32 = jnp.where(
                        is_g, live_i32[g * rpo : (g + 1) * rpo], my_live_i32
                    )
                my_live = my_live_i32 > 0
                ids_own, ws_own = rt.route(
                    my_logits,
                    bias=s["bias_vm"][...],
                    live_rows=my_live,
                    topk=cfg.topk,
                    n_experts=E,
                    routed_scaling_factor=cfg.routed_scaling_factor,
                )
                lane_own = lax.broadcasted_iota(I32, (rpo, 128), 1)
                packed = jnp.zeros((rpo, 128), F32)
                for k in range(cfg.topk):
                    packed = jnp.where(lane_own == k, ids_own[k].astype(F32), packed)
                    packed = jnp.where(lane_own == cfg.topk + k, ws_own[k], packed)
                s["sel_src"][...] = packed
                dst = s["sel_w"].at[pl.ds(pl.multiple_of(owner * rpo, rpo), rpo)]
                dst[...] = packed
                copies = [
                    pltpu.make_async_remote_copy(
                        src_ref=s["sel_src"],
                        dst_ref=dst,
                        send_sem=s["rsem"].at[0, off],
                        recv_sem=s["rsem"].at[1, off],
                        device_id=(me ^ off,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    for off in range(1, rg)
                ]
                for cp in copies:
                    cp.start()
                for cp in copies:
                    cp.wait()
                gw = s["sel_w"][...]
                ids = [gw[:, k : k + 1].astype(I32) for k in range(cfg.topk)]
                ws = [gw[:, cfg.topk + k : cfg.topk + k + 1] for k in range(cfg.topk)]
            else:
                ids, ws = rt.route(
                    logits,
                    bias=s["bias_vm"][...],
                    live_rows=live,
                    topk=cfg.topk,
                    n_experts=E,
                    routed_scaling_factor=cfg.routed_scaling_factor,
                )
            sid, sw = jnp.zeros((M, 128), I32), jnp.zeros((M, 128), F32)
            for k in range(cfg.topk):
                sid = jnp.where(lane == k, ids[k], sid)
                sw = jnp.where(lane == k, ws[k], sw)
            s["sel_ids"][...] = sid
            s["sel_w"][...] = sw

        if isinstance(is_hash, bool):
            select(is_hash)
        else:
            pl.when(is_hash)(lambda: select(True))
            pl.when(jnp.logical_not(is_hash))(lambda: select(False))
        ids = [s["sel_ids"][:, k : k + 1] for k in range(cfg.topk)]
        ws = [s["sel_w"][:, k : k + 1] for k in range(cfg.topk)]
        dense, hit = rt.combine_matrix(ids, ws, E)
        s["dense"][...] = dense
        n_items = blk.build_worklist(hit)
    for a in ABLATE:  # attribution only: "cap<k>" runs at most k items
        if a.startswith("cap"):
            n_items = jnp.minimum(n_items, int(a[3:]))

    def seed(cs, w):
        s["acc"][0:T, cs] = inputs.partial[:, cs]
        if M > T:
            s["acc"][T:M, cs] = jnp.zeros((M - T, w), F32)

    if shared_expert is not None and not leased:
        prefetched = blk.prefetch(n_items)
        shared_expert()
    else:
        if shared_expert is not None:
            shared_expert()
        prefetched = blk.prefetch(n_items)

    with jax.named_scope("moe_items"):
        _over_cols(geom.hidden, seed)
        blk.run(n_items, acc_init=False, prefetched=prefetched)
    if before_ar is not None:  # every item DMA has landed: the lease is free
        before_ar()
    if ar is None:
        ar, slot = AllReduce(cfg.ar, s, me), lax.rem(layer, 2)
    else:  # the caller alternates slots between its all-reduces
        slot = ar_slot
    with jax.named_scope("moe_ar"):
        ar.reduce_into(s["acc"], out, slot)
        ar.wait_sends(slot)


def placement_tables(placement, n_layers, n_ranks, n_experts):
    """expert_placement.Placement -> (held int32 [R, L, slots], copy int32
    [R, L, slots] or None) for moe_phase: per rank and layer the global expert
    of each local slot (-1 past a layer's slot count; unlisted layers keep the
    contiguous blocks), and where any expert has copies, copy_idx << 8 |
    n_copies per slot (copies numbered by ascending slot id r * slots + j, as
    expert_placement.slot_table). Copy k of n runs the slices s = k mod n.
    At most 2 copies: the 2-deep per-expert scale buffer needs every entry
    to span >= nbuf - 1 items."""
    import numpy as np

    el = n_experts // n_ranks
    per_layer = {
        li: placement.ranks(li)
        or tuple(tuple(range(r * el, (r + 1) * el)) for r in range(n_ranks))
        for li in range(n_layers)
    }
    slots = max(len(ranks[0]) for ranks in per_layer.values())
    held = np.full((n_ranks, n_layers, slots), -1, np.int32)
    copy = np.zeros((n_ranks, n_layers, slots), np.int32)
    replicas = False
    for li, ranks in per_layer.items():
        seen: dict[int, list[tuple[int, int]]] = {}
        for r, row in enumerate(ranks):
            for j, e in enumerate(row):
                held[r, li, j] = e
                seen.setdefault(e, []).append((r, j))
        for e, where in seen.items():
            replicas |= len(where) > 1
            if len(where) > 2:
                raise ValueError(
                    f"layer {li}: expert {e} has {len(where)} copies (max 2)"
                )
            for k, (r, j) in enumerate(where):
                copy[r, li, j] = (k << 8) | len(where)
    return held, (copy if replicas else None)

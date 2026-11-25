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
"""q38 HIX700 PREGATE: predicted routed experts prefetched into the idle expert ring before routing (bitwise).

At 16 rows the routed-expert loop streams its weights at the HBM floor and waits on them (ewait ~25 % of the loop),
while HBM sits idle during the mixer's all-reduce, the pre-MoE HC mix and the route chain. PREGATE uses that window:

  issue    before the mixer sublayer's all-reduce (`pregate_at="ar"`, or at layer start, `"start"`), each rank takes
           `PG` predicted experts of its own (slot j = combined hit-list position j) and issues their `TE` weight units
           into expert-ring banks `j*TE .. j*TE+TE-1` and their scales into `es` slot j (the ring's own `fetch_e`).
           The expert ring (`epool` / `ering_*`) is persistent VMEM that no other qwen38 block uses outside the MoE.
  route    after dedup the true hit list is re-laid as the COMBINED list `[slot 0 .. slot PG-1, the true hits that
           were not prefetched (true order)]`; `st` covers every combined unit, so the ring's release / refill chain is
           unchanged; route_fetch issues only the units after the prefetched ones.
  phase A  slot j (static): a hit runs the expert body with its output `y` stored in a BF16 side buffer (`y` is
           BF16-exact: `rnd`); a miss waits its units and scales (semaphore balance) and releases them; an empty slot
           only releases. Every release refills its bank, so the stream continues while phase A computes.
  phase B  the not-prefetched hits run the expert body in true order; before each one adds into `part`, every
           pending side buffer whose true position is earlier is added (`part += wcol * y`, the same f32 op), and the
           rest after the loop. Every row therefore receives its routed outputs in true hit-list order: bit-identical
           to PREGATE off for any prediction (hit, miss or empty).

The prediction source in this build is the `MoEDebug.pregate_list` input `pg_pred` (`[L, 128]` int32 per rank, global
expert ids, -1 = empty), used by the CPU proof and the MB-PGPF microbench (oracle / degraded arms). A prediction that
is not one of the rank's own experts, or repeats an earlier slot, is an empty slot.

Engagement output `PG_ENG` (`[L, 8]` int32 per rank, written every layer, a runtime value, not a trace-time line):
  [0] slots issued  [1] slots hit (prefetched experts routing used)  [2] true hits n_hit  [3] combined length
  [4] the layer the issue was tagged with (== the layer when wired; -1 = never issued)  [5..7] slots 0..2 as predicted.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...errors import MegakernelConfigError
from ...ops import BF16, F32, I32, rnd, rup
from ...spec import ExtraInput, StateDecl
from .context import MoEEmit
from .ring import egive, egive_group, es_wait, etake, fetch_e

PG_PRED = "pg_pred"
PG_ENG = "moe_pg_eng"
PART_TRACE = "moe_trace_part"
PGX_TRACE = "moe_trace_pgx"
_BIG = 1 << 20
# `pg_ids` SMEM layout (int32)
_ID, _ST, _POS, _ORD = 0, 8, 16, 24  # predicted id / issued flag / true position / slots ordered by position (8 each)
_NHIT, _PTR, _TAG = 32, 33, 35  # prefetched hits / merge pointer / layer tag of the last issue


# ---------------------------------------------------------------- declarations
def pg_extra_inputs(dbg: Any) -> list[ExtraInput]:
    return [ExtraInput(PG_PRED, "hbm")] if dbg.pregate_list else []


def pg_outputs(tt: Any, g: Any, ex: Any, dbg: Any = None) -> list[StateDecl]:
    out = [StateDecl(PG_ENG, (8,), I32, "w")] if int(tt.pregate) > 0 else []
    if dbg is not None and dbg.part_trace:
        out.append(StateDecl(PART_TRACE, (ex.MB, g.H), F32, "w"))
    if dbg is not None and dbg.pregate_xtrace and int(tt.pregate) > 0:
        out.append(StateDecl(PGX_TRACE, (ex.MB, g.H), BF16, "w"))
    return out


def part_trace(m: MoEEmit, l_idx: Any) -> None:
    """`MoEDebug.part_trace` (tests only): copy this layer's f32 `part` after the routed experts (synchronous)."""
    o = m.sp.block_ord(l_idx, "moe")
    c = pltpu.make_async_copy(m.scr["part"], m.ctx.O[PART_TRACE].at[o], m.scr["misc_sem"])
    c.start()
    c.wait()


def pg_scratch(mp: Any, ex: Any) -> dict[str, Any]:
    if not mp.PG:
        return {}
    nb_picks = rup(ex.B * mp.TOPK, 128)
    return {
        "pg_y": pltpu.VMEM((mp.PG, ex.MB, mp.H), BF16),
        "pg_ids": pltpu.SMEM((128,), I32),
        "pg_pred_s": pltpu.SMEM((128,), I32),
        "pg_true": pltpu.SMEM((nb_picks,), I32),
        "pg_tpos": pltpu.SMEM((nb_picks,), I32),
        "pg_eng": pltpu.SMEM((8,), I32),
        "pg_sem": pltpu.SemaphoreType.DMA(()),
        **({"pg_wr": pltpu.VMEM((mp.PG_RNT * mp.PG_RTK, 128), BF16), "pg_wsem": pltpu.SemaphoreType.DMA(()),
            "pg_idv": pltpu.VMEM((8, 128), I32), "pg_ids2": pltpu.SMEM((8, 128), I32),
            "pg_isem": pltpu.SemaphoreType.DMA(())} if mp.PG_SRC == "pred" else {}),
        **({"pg_xt": pltpu.VMEM((ex.MB, mp.H), BF16)} if mp.PG_SRC == "pred" and getattr(ex.debug.blocks.get("moe"),
                                                                                      "pregate_xtrace", False) else {}),
    }


def pg_init(ctx: Any, mp: Any) -> None:
    """Prologue: no slot issued yet (the layer tag can never match before the first issue)."""
    if not mp.PG:
        return
    ids = ctx.scr["pg_ids"]
    ids[_TAG] = jnp.int32(-1)
    for j in range(8):
        ids[_ST + j] = jnp.int32(0)


# ---------------------------------------------------------------- issue (mixer phase)
def pregate_issue(block: Any, ctx: Any, l_idx: Any) -> None:
    """Issue layer `l_idx`'s predicted experts into ring banks `0 .. PG*TE-1` (`fetch_e`, scales into `es` slot j)."""
    mp = ctx.block_params(block.name)
    dbg = block.debug_flags(ctx.ex)
    m = MoEEmit.build(ctx, mp, dbg)
    scr, me = ctx.scr, ctx.me
    PG, TE, ER = mp.PG, mp.TE, mp.ER
    ids = scr["pg_ids"]
    with ctx.sc("pg_issue", 2):
        if mp.PG_SRC == "list":
            cp = pltpu.make_async_copy(ctx.extra[PG_PRED].at[l_idx], scr["pg_pred_s"], scr["pg_sem"])
            cp.start()
            cp.wait()
            pred = [scr["pg_pred_s"][j] for j in range(PG)]
        else:  # the predictor's ids (global), VMEM -> SMEM copy started by `pregate_predict`
            pltpu.make_async_copy(scr["pg_idv"], scr["pg_ids2"], scr["pg_isem"]).wait()
            pred = [scr["pg_ids2"][0, j] for j in range(PG)]
        n_iss = jnp.int32(0)
        for j in range(PG):
            e = pred[j]
            ok = jnp.logical_and(e >= 0, lax.div(jnp.maximum(e, 0), jnp.int32(ER)) == me)
            for k in range(j):
                ok = jnp.logical_and(ok, e != pred[k])
            ids[_ID + j] = jnp.where(ok, e, jnp.int32(-1))
            if j < 3:
                scr["pg_eng"][5 + j] = e  # the prediction as given (engagement row lanes 5..7)
            ids[_ST + j] = ok.astype(I32)
            scr["elist"][j] = jnp.where(ok, e, me * ER)  # fetch_e reads the slot's expert from elist[j]
            n_iss = n_iss + ok.astype(I32)

            @pl.when(ok)
            def _(j=j):
                for t in range(TE):
                    fetch_e(m, j * TE + t, l_idx, t)

        ids[_TAG] = l_idx
        scr["pg_eng"][0] = n_iss


def pregate_wiring(model_spec: Any, ex: Any) -> tuple[Any, Any, tuple]:
    """(issue fn, predict fn, owner blocks): the blocks of the slot right before the MoE slot fire the `"ar"` issue
    before their all-reduce (`fire`) and, with the in-kernel predictor, the prediction right after their HC mix
    (`fire_predict`). `(None, None, ())` when no MoE block has `pregate_at="ar"` with `pregate > 0`."""
    from ..hc import HCSublayer
    from ...spec import SwitchSlot
    from .block import MoEBlock

    slots = list(model_spec.layer_type.slots)
    for k, s in enumerate(slots):
        inner = s.inner if isinstance(s, HCSublayer) else s
        if not isinstance(inner, MoEBlock):
            continue
        tt = inner.tunables(ex)
        if int(tt.pregate) <= 0:
            return None, None, ()
        for b in model_spec.all_blocks():  # the expert ring must be idle outside the MoE: no epool borrower
            bi = b.inner if isinstance(b, HCSublayer) else b
            if type(bi).__name__.startswith("GQA"):
                raise MegakernelConfigError("pregate refuses a GQA block (it may borrow the expert ring, kv_ering)")
        if str(tt.pregate_at) != "ar":
            return None, None, ()
        if k == 0:
            raise MegakernelConfigError("pregate_at='ar' needs a sublayer before the MoE slot")
        prev = slots[k - 1]
        owners = tuple(prev.branches) if isinstance(prev, SwitchSlot) else (prev,)
        pred = None
        if not inner.debug_flags(ex).pregate_list:  # in-kernel predictor: fired after the mixer's HC mix
            if not all(isinstance(o, HCSublayer) for o in owners):
                raise MegakernelConfigError("pregate's in-kernel predictor needs HC mixer sublayers")
            pred = functools.partial(pregate_predict, inner)
        return functools.partial(_issue_from_owner, inner), pred, owners
    return None, None, ()


def _issue_from_owner(moe_block: Any, ctx: Any, l_idx: Any) -> None:
    pregate_issue(moe_block, ctx, l_idx)


def fire(ctx: Any, blk: Any, l_idx: Any, part: Any, where: str) -> Any:
    """Called with a sublayer's partial output right before its all-reduce; issues the prefetch when `blk` owns it.
    `where="kernel"` fires only for core all-reduce blocks, `"hc"` for HC sublayers (their reduction is custom)."""
    fn = getattr(ctx, "pregate", None)
    if fn is not None and any(blk is o for o in getattr(ctx, "pregate_owners", ())):
        if where == "hc" or getattr(blk, "reduction", None) == "allreduce":
            fn(ctx, l_idx)
    return part


# ---------------------------------------------------------------- route (after dedup)
def pregate_plan(m: MoEEmit, l_idx: Any, n_hit: Any) -> Any:
    """Re-lay the true hit list as the combined list; returns the combined length `PG + n_not_prefetched`."""
    mp, scr = m.mp, m.scr
    PG = mp.PG
    ids = scr["pg_ids"]
    with m.sc("pg_plan", 2):
        valid = ids[_TAG] == l_idx  # an issue for this layer happened (else every slot counts as empty)
        for j in range(PG):
            ids[_ST + j] = jnp.where(valid, ids[_ST + j], jnp.int32(0))
            ids[_POS + j] = jnp.int32(_BIG)

        def copy(q, c):
            scr["pg_true"][q] = scr["elist"][q]
            return c

        lax.fori_loop(0, n_hit, copy, 0)

        def scan(q, c):
            e = scr["pg_true"][q]
            hit = jnp.bool_(False)
            for j in range(PG):
                mj = jnp.logical_and(ids[_ST + j] == 1, ids[_ID + j] == e)
                ids[_POS + j] = jnp.where(mj, q, ids[_POS + j])
                hit = jnp.logical_or(hit, mj)

            @pl.when(jnp.logical_not(hit))
            def _():
                scr["elist"][PG + c] = e
                scr["pg_tpos"][PG + c] = q

            return c + jnp.where(hit, 0, 1).astype(I32)

        n_np = lax.fori_loop(0, n_hit, scan, jnp.int32(0))
        for j in range(PG):  # combined positions 0..PG-1 = the slots (an empty slot keeps a harmless own id)
            scr["elist"][j] = jnp.where(ids[_ST + j] == 1, ids[_ID + j], m.me * mp.ER)
        # slots ordered by true position (misses / empties last): rank_j = #k with (pos_k, k) < (pos_j, j)
        n_pf = jnp.int32(0)
        for j in range(PG):
            pj = ids[_POS + j]
            rank = jnp.int32(0)
            for k in range(PG):
                if k == j:
                    continue
                pk = ids[_POS + k]
                before = jnp.logical_or(pk < pj, jnp.logical_and(pk == pj, k < j))
                rank = rank + before.astype(I32)
            ids[_ORD + rank] = jnp.int32(j)
            n_pf = n_pf + (pj < _BIG).astype(I32)
        ids[_NHIT] = n_pf
        ids[_PTR] = jnp.int32(0)
        eng = scr["pg_eng"]
        eng[0] = jnp.where(valid, eng[0], jnp.int32(0))
        eng[1] = n_pf
        eng[2] = n_hit
        eng[3] = PG + n_np
        eng[4] = ids[_TAG]
    return PG + n_np


def pregate_route_fetch(m: MoEEmit, l_idx: Any, n_comb: Any) -> None:
    """route_fetch for the combined list: the first `PG*TE` units are the prefetched slots (issued, or empty)."""
    mp = m.mp
    lax.fori_loop(
        mp.PG * mp.TE,
        jnp.minimum(mp.NBE, n_comb * mp.TE),
        lambda k, _: (fetch_e(m, k, l_idx), 0)[1],
        0,
    )


# ---------------------------------------------------------------- experts
def _wcol(m: MoEEmit, e: Any) -> Any:
    return jnp.sum(jnp.where(m.lane_e8 == e, m.scr["wn"][...], 0.0), axis=1, keepdims=True)


def _merge_upto(m: MoEEmit, q: Any) -> None:
    """Add every pending side buffer whose true position is below `q`, in true order (`part += wcol * y`)."""
    mp, scr = m.mp, m.scr
    MR, H, CH, PG = m.ex.MR, mp.H, mp.CH, mp.PG
    ids, part = scr["pg_ids"], scr["part"]
    for _ in range(PG):
        ptr = ids[_PTR]
        j = ids[_ORD + jnp.minimum(ptr, PG - 1)]
        go = jnp.logical_and(ptr < ids[_NHIT], ids[_POS + j] < q)

        @pl.when(go)
        def _():
            jj = ids[_ORD + jnp.minimum(ids[_PTR], PG - 1)]
            wcol = _wcol(m, ids[_ID + jj])
            for c in range(H // CH):
                lo = c * CH
                y = scr["pg_y"][jj, 0:MR, lo : lo + CH].astype(F32)
                part[0:MR, lo : lo + CH] += wcol * y
            ids[_PTR] = ids[_PTR] + 1


def _body(m: MoEEmit, l_idx: Any, i: Any, sink: Any, pre_down: Any) -> None:
    """`experts.expert` with its three `part += wcol * y` sites routed through `sink(lo, hi, wcol, y)` and a
    `pre_down(e)` hook before the first of them. Arithmetic, tile order, waits and releases are expert()'s."""
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    CH, H, IR, NSS, T13, T2, TE = mp.CH, mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE
    e = scr["elist"][i]
    s = i % NSS
    base = i * TE
    acc, xb, xb2 = scr["acc"], scr["xb"], scr["xb2"]
    DIRECT0 = mp.r2_direct_store
    if T13 == 1:
        h = stream.mm(xb[...], etake(m, "e13", base)[...])
        egive(m, base, l_idx)
    elif not mp.unroll:
        acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)

        def b13(t, cc):
            u = base + t
            w = etake(m, "e13", u)
            k0 = pl.multiple_of(t * tk["e13"], tk["e13"])
            acc[:, : 2 * IR] += stream.mm(xb[:, pl.ds(k0, tk["e13"])], w[...])
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T13, b13, 0)
        h = acc[:, : 2 * IR]
    else:
        TK13 = tk["e13"]
        if not DIRECT0:
            acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)
        for t in range(T13):
            u = base + t
            w = etake(m, "e13", u)
            m13 = stream.mm(xb[:, t * TK13 : (t + 1) * TK13], w[...])
            if DIRECT0 and t == 0:
                acc[:, : 2 * IR] = m13
            else:
                acc[:, : 2 * IR] += m13
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, t - 1), (u, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, u, l_idx, t)
            else:
                egive(m, u, l_idx, t)
        h = acc[:, : 2 * IR]
    es_wait(m, i)
    h = h * scr["es"][s, :, : 2 * IR]
    mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
    xb2[:, :IR] = mid_.astype(BF16)
    wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
    pre_down(e)
    u2 = base + T13
    if T2 == 1:
        w2 = etake(m, "e2", u2)[...]
        xm = xb2[:, :IR]
        for j in range(H // CH):
            lo = 2 * IR + j * CH
            y = stream.mm(xm, w2[:, j * CH : (j + 1) * CH])[0:MR]
            y = rnd(y * scr["es"][s, :, lo : lo + CH])
            sink(j * CH, (j + 1) * CH, wcol, y)
        egive(m, u2, l_idx)
    elif not mp.unroll:
        acc[:, :H] = jnp.zeros((MB, H), F32)

        def b2(t, cc):
            u = u2 + t
            w = etake(m, "e2", u)
            k0 = pl.multiple_of(t * tk["e2"], tk["e2"])
            acc[:, :H] += stream.mm(xb2[:, pl.ds(k0, tk["e2"])], w[...])
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T2, b2, 0)
        y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
        sink(0, H, wcol, y)
    else:
        TK2 = tk["e2"]
        if not DIRECT0:
            acc[:, :H] = jnp.zeros((MB, H), F32)
        for t in range(T2):
            u = u2 + t
            w = etake(m, "e2", u)
            m2 = stream.mm(xb2[:, t * TK2 : (t + 1) * TK2], w[...])
            if DIRECT0 and t == 0:
                acc[:, :H] = m2
            else:
                acc[:, :H] += m2
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, T13 + t - 1), (u, T13 + t)], l_idx)
                elif t == T2 - 1:
                    egive(m, u, l_idx, T13 + t)
            else:
                egive(m, u, l_idx, T13 + t)
        y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
        sink(0, H, wcol, y)


def _drain_slot(m: MoEEmit, l_idx: Any, j: int, issued: bool) -> None:
    """A prefetched slot routing did not use: wait its units and scales if issued (semaphore balance), release all."""
    mp = m.mp
    T13, TE = mp.T13, mp.TE
    for t in range(TE):
        u = j * TE + t
        if issued:
            etake(m, "e13" if t < T13 else "e2", u)
        egive(m, u, l_idx, t)
    if issued:
        es_wait(m, j)


def experts_pg(m: MoEEmit, l_idx: Any, n_comb: Any) -> None:
    """Phase A (prefetched slots, static), phase B (the rest, true order) and the final merge; then the engagement
    output row of this layer."""
    mp, scr = m.mp, m.scr
    MR, PG = m.ex.MR, mp.PG
    ids, part = scr["pg_ids"], scr["part"]
    for j in range(PG):
        used = ids[_POS + j] < _BIG
        issued = ids[_ST + j] == 1

        def store(lo, hi, wcol, y, j=j):
            del wcol
            scr["pg_y"][j, 0:MR, lo:hi] = y.astype(BF16)

        def hit_slot(j=j, store=store):
            with m.sc("pg_hit", 2):  # region-trace scopes (VLLM_TPU_MK_SCOPES >= 2): the hit rate from a traced arm
                _body(m, l_idx, j, store, lambda e: None)

        def miss_slot(j=j):
            with m.sc("pg_miss", 2):
                _drain_slot(m, l_idx, j, True)

        pl.when(used)(hit_slot)
        pl.when(jnp.logical_and(issued, jnp.logical_not(used)))(miss_slot)
        pl.when(jnp.logical_not(issued))(lambda j=j: _drain_slot(m, l_idx, j, False))

    def add(lo, hi, wcol, y):
        part[0:MR, lo:hi] += wcol * y

    def phase_b(i, carry):
        q = scr["pg_tpos"][i]
        _body(m, l_idx, i, add, lambda e: _merge_upto(m, q))
        return carry

    lax.fori_loop(PG, n_comb, phase_b, 0)
    _merge_upto(m, jnp.int32(_BIG))
    o = m.sp.block_ord(l_idx, "moe")
    c = pltpu.make_async_copy(scr["pg_eng"], m.ctx.O[PG_ENG].at[o], scr["pg_sem"])
    c.start()
    c.wait()


# ---------------------------------------------------------------- in-kernel predictor (PG_SRC "pred")
def pg_router_window(mp: Any, me: Any) -> tuple[Any, Any]:
    """(first router column of this rank's 128-lane window, this rank's lane offset inside it)."""
    start = lax.div(me * mp.ER, jnp.int32(128)) * 128
    return start, me * mp.ER - start


def pregate_router_fetch(ctx: Any, mp: Any, l_idx: Any) -> None:
    """Layer start: DMA this rank's 128-lane window of layer `l_idx`'s router (its own E / R columns) into `pg_wr`,
    one copy per router K tile (the router's own HBM tiles; no new weight)."""
    scr, W = ctx.scr, ctx.W
    o = ctx.stream.sp.block_ord(l_idx, "moe")
    start, _ = pg_router_window(mp, ctx.me)
    for t in range(mp.PG_RNT):
        pltpu.make_async_copy(
            W["router"].at[o, t, :, pl.ds(pl.multiple_of(start, 128), 128)],
            scr["pg_wr"].at[pl.ds(t * mp.PG_RTK, mp.PG_RTK)],
            scr["pg_wsem"],
        ).start()


def pg_scores(x: Any, wr: Any, B: int, off: Any, ER: int) -> Any:
    """[1, 128] f32 per-expert score of this rank's window: the sum over the B rows of exp(logit - the row's best own
    logit) (each row's softmax numerators over this rank's experts: a soft count of the rows that rank the expert
    high); lanes outside [off, off + ER) are -inf. Pure jnp."""
    lg = jnp.dot(x, wr, preferred_element_type=F32)
    lane = lax.broadcasted_iota(I32, lg.shape, 1)
    row = lax.broadcasted_iota(I32, lg.shape, 0)
    own = jnp.logical_and(lane >= off, lane < off + ER)
    ok = jnp.logical_and(own, row < B)
    neg = jnp.float32(-jnp.inf)
    lg = jnp.where(ok, lg, neg)
    rmax = jnp.max(lg, axis=1, keepdims=True)
    p = jnp.where(ok, jnp.exp(lg - jnp.where(row < B, rmax, 0.0)), 0.0)
    return jnp.where(own[0:1], jnp.sum(p, axis=0, keepdims=True), neg)


def pg_top(sc: Any, PG: int) -> list:
    """The PG best lanes of a [1, 128] score (ties: the lowest lane), as [1, 1] int32 values. Pure jnp."""
    lane = lax.broadcasted_iota(I32, sc.shape, 1)
    out = []
    for _ in range(PG):
        mx = jnp.max(sc, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(sc == mx, lane, jnp.int32(1 << 20)), axis=1, keepdims=True)
        out.append(idx)
        sc = jnp.where(lane == idx, jnp.float32(-jnp.inf), sc)
    return out


def pg_input(res: Any, nhc: int, h: int, eps: float) -> Any:
    """The predictor's input: the mean over the NHC residual streams of each stream RMS-normalised (unit gain), BF16.
    Pure jnp (also the CPU proof's host reference)."""
    acc = None
    for b in range(nhc):
        seg = res[:, b * h : (b + 1) * h]
        xn = seg * lax.rsqrt(jnp.mean(seg * seg, axis=-1, keepdims=True) + eps)
        acc = xn if acc is None else acc + xn
    return (acc / nhc).astype(BF16)


def pregate_predict(block: Any, ctx: Any, l_idx: Any, owner: Any) -> None:
    """As the mixer HC sublayer starts (the residual is the layer input): score this rank's experts on `pg_input` and
    start the copy of the PG predicted global expert ids to SMEM (waited at the issue, before the mixer all-reduce)."""
    mp = ctx.block_params(block.name)
    scr = ctx.scr
    with ctx.sc("pg_predict", 2):
        for _ in range(mp.PG_RNT):
            pltpu.make_async_copy(scr["pg_wr"].at[pl.ds(0, mp.PG_RTK)], scr["pg_wr"].at[pl.ds(0, mp.PG_RTK)],
                                  scr["pg_wsem"]).wait()
        start, off = pg_router_window(mp, ctx.me)
        H = mp.H
        x = pg_input(scr["res"][...], owner.nhc, owner.h, float(ctx.g.eps))
        if block.debug_flags(ctx.ex).pregate_xtrace:
            scr["pg_xt"][...] = x
        sc = pg_scores(x, scr["pg_wr"][0:H, :], ctx.ex.B, off, mp.ER)
        lane = lax.broadcasted_iota(I32, (8, 128), 1)
        row = lax.broadcasted_iota(I32, (8, 128), 0)
        ids = jnp.full((8, 128), -1, I32)
        for j, idx in enumerate(pg_top(sc, mp.PG)):
            ids = jnp.where(jnp.logical_and(row == 0, lane == j), idx + start, ids)
        scr["pg_idv"][...] = ids
        pltpu.make_async_copy(scr["pg_idv"], scr["pg_ids2"], scr["pg_isem"]).start()
        if block.debug_flags(ctx.ex).pregate_xtrace:  # tests only: the predictor's input of this layer
            o = ctx.stream.sp.block_ord(l_idx, "moe")
            c = pltpu.make_async_copy(scr["pg_xt"], ctx.O[PGX_TRACE].at[o], scr["pg_sem"])
            c.start()
            c.wait()


def fire_predict(ctx: Any, blk: Any, l_idx: Any, flag: Any) -> Any:
    """Called as an HC sublayer starts; runs the predictor when `blk` owns the prefetch. Returns `flag`."""
    fn = getattr(ctx, "pregate_pred", None)
    if fn is not None and any(blk is o for o in getattr(ctx, "pregate_owners", ())):
        fn(ctx, l_idx, blk)
    return flag

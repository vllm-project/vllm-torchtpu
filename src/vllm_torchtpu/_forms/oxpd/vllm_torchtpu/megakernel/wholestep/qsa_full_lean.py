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
"""q38 L1-D08S1: D08 package S1 "UNION-LEAN" on the whole step's in-kernel indexer (union attention path).

Three stages, each behind its own default-off option of the indexed block (envs in `qwen38_wholestep_call`):

* B1 `union_bounded` (VLLM_TPU_MK_UNION_BOUNDED, D08 item 1h): union compaction stops at the union count. MM's
  `compact_union` runs all `KU / 256` output passes of every union half whatever the union size; here the count
  is stored first and pass `j0` runs only when `j0 < n` (`pl.when`). Each pass is MM's arithmetic unchanged, and
  passes are independent, so every lane a tile reads (`< ceil(n / 64) * 64 <= ceil(n / 256) * 256`) is
  bit-identical; lanes past the last live pass keep stale values that nothing reads.
* B2 `union_lean` (VLLM_TPU_MK_UNION_LEAN, items 1a + 1b + 1c): under union attention the per-row compacted ids
  (`qi_id` / `qi_ids`) are read only by the per-row path and the debug output, so rowsel stops building them
  (no `compact_row`, no per-row SMEM copies). With the sharded indexer the selection exchange (`qsa_selx`) is
  folded into the union exchange: each owner compacts every union half straight into a per-destination send
  slot whose 128-lane tail carries `qu_n` (D lanes), `qi_kvl` and the `g_a` flag of its T rows; each rank
  receives only its own DCP half (1 / D of MM's union bytes) and attention reads it in place from the receive
  buffer, so only the tails are unpacked (into `qu_n`, `qi_kvl`, `qc_sga`). `qi_z` is still zeroed on every rank
  (u_ids copies it). int32 throughout: exact.
* B3 `slot_batch` (VLLM_TPU_MK_SLOT_BATCH, items 1d + 1e + 1f): the per-slot DMA round trips become per-layer
  batches. Completed groups: one DMA each for every slot's ring rows and RoPE rows, MM's per-slot arithmetic
  unchanged (same f32 sum order) into batched buffers, one output DMA each for `qsa_ck_new` / `qsa_ring_new`,
  waited at the end of the selection. Recent tiles: every slot's pool rows are issued right after the layer's
  tables load and waited (per copy) only after the owner's scoring and selection. Union ids: double-buffered
  SMEM rows, request q + 1's copy in flight behind request q's attention.

q38 L2-UAISSUE (X-UA-ISSUE, `union_issue` = VLLM_TPU_MK_UNION_ISSUE 1 / 2; union_issue.py) also rides this class: physical
micro-page ids from a per-layer table slab (stage a) and MM D12 F0a's issue recipe in the union walk (stage b), with or
without the S1 stages; bitwise MM's outputs (tests/megakernel/test_l2_uaissue.py).

q38 L2-UNPACK (X-UA-UNPACK, `union_unpack` = VLLM_TPU_MK_UNION_UNPACK; union_unpack.py) rides it too: the union tile body
reads its ring slot with full-vreg strided loads (head pair 0, static offset 0), with or without X-UA-ISSUE and the S1
stages; bitwise (tests/megakernel/test_l2_unpack.py).

Every stage keeps MM's values and orders, so the outputs are bitwise identical to `QSAFullStepBlock` /
`QSAFullStepGroupBlock` (D14) at DCP 1 / 2 / 4. The classes here are selected by `wholestep_spec` only when an
S1 option (or the test probe) is present (`lean_step_block`); with every option off the whole step builds MM's /
D14's block exactly as before (no line of qsa_full_b.py / qsa_full_c.py / union_attn.py changes).

`union_probe` (tests, with `debug`): every QSA layer dumps the union state attention is about to read (this rank's
union halves, `qu_n`, `qi_kvl`, `qc_sga`, `qi_z`) to `qsa_dbg_union` / `qsa_dbg_rows`, for MM's block too
(`QSAFullStep*ProbeBlock`), so the identity suite compares the internal state as well as the outputs.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, _unpack_kv
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rmsnorm, rnd
from ..spec import ScratchReq, StateDecl
from . import indexer as ix
from . import union_attn as ua
from . import union_issue as ui
from .indexer import exclusive_prefix
from .qsa_full import POOL, SCALES
from .qsa_full_b import IDX_N, IDX_W, MTAB, NVIS
from .qsa_full_c import (
    CK_NEW,
    CTXQ,
    IR_NEW,
    KR_NEW,
    RING,
    RING_NEW,
    ROPE_CK,
    UNION_STATS,
    UNION_STATS_REQS,
    QSAFullStepBlock,
    QSAFullStepGroupBlock,
    _scoped,
    pick_row,
    pool_units,
)

DBG_UNION = "qsa_dbg_union"  # [requests, 1, 2 D KU + 128] int32 per QSA layer (`union_probe`)
DBG_ROWS = "qsa_dbg_rows"  # [M, 1, 256] int32 per QSA layer: lane 0 qi_kvl, 1 qc_sga, 128.. qi_z
LEAN_KEYS = ("union_bounded", "union_lean", "slot_batch")
PROBE_KEY = "union_probe"
ISSUE_KEY = "union_issue"  # q38 L2-UAISSUE (passed only when on, like the S1 options)
UNPACK_KEY = "union_unpack"  # q38 L2-UNPACK (passed only when on)


def lean_step_block(indexed: dict, qsa: dict | None) -> Any:
    """The stage-C QSA block for `indexed` when an S1 option or the probe is present, else None (then
    `wholestep_spec` builds MM's / D14's block exactly as before). Any S1 key present (even False) selects the
    lean class, whose all-off path is MM's code (the identity suite's copy check)."""
    if not any(k in indexed for k in LEAN_KEYS + (PROBE_KEY, ISSUE_KEY, UNPACK_KEY)):
        return None
    group = int(indexed.get("dcp", 1) or 1) > 2
    lean = any(k in indexed for k in LEAN_KEYS + (ISSUE_KEY, UNPACK_KEY))
    cls = {
        (False, False): QSAFullStepProbeBlock,
        (True, False): QSAFullStepGroupProbeBlock,
        (False, True): QSAFullStepLeanBlock,
        (True, True): QSAFullStepGroupLeanBlock,
    }[(group, lean)]
    return cls(**indexed, **(qsa or {}))


def _union_prologue(code: jax.Array, consts: tuple) -> tuple:
    """`ua.compact_union`'s pass-independent part (row ranks, row offsets, count), MM's arithmetic."""
    U, Us, jcol0, srow, lane = consts
    S = code.shape[0]
    sel = code > 0
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (S, 128)))[0:1, :]
    O_row = exclusive_prefix(c_row, Us)
    E_row = O_row + c_row
    Pb, codeb = P.astype(BF16), code.astype(F32).astype(BF16)
    n = jnp.sum(sel.astype(I32), axis=(0, 1), keepdims=True)
    return E_row, O_row, Pb, codeb, n


def _union_pass(pro: tuple, consts: tuple, j0: int) -> tuple[jax.Array, jax.Array]:
    """`ua.compact_union`'s pass at output slots `[j0, j0 + W)`: `[1, W]` ids and codes, MM's arithmetic."""
    U, Us, jcol0, srow, lane = consts
    E_row, O_row, Pb, codeb, n = pro
    W = jcol0.shape[0]
    jcol = jcol0 + j0
    jf = jcol.astype(F32)
    r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
    hit = srow == r_j
    oh = hit.astype(F32).astype(BF16)
    Pg = jnp.dot(oh, Pb, preferred_element_type=F32)
    O_rj = jnp.sum(jnp.where(hit, O_row, 0.0), axis=1, keepdims=True)
    lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
    Cg = jnp.dot(oh, codeb, preferred_element_type=F32)
    code_j = jnp.sum(jnp.where(lane == lane_j, Cg, 0.0), axis=1, keepdims=True)
    live = jcol < n
    ids = jnp.where(live, r_j * 128 + lane_j, 0).astype(F32)
    codes = jnp.where(live, code_j, 0.0)
    return (
        jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32),
        jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32),
    )


class _ProbeMixin:
    """`union_probe`: dump the union state attention reads, per QSA layer (tests only; needs `debug`)."""

    def __init__(self, *, union_probe: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.union_probe = bool(union_probe)
        assert not self.union_probe or (self.union_attn and self.debug), "the probe dumps the union (debug)"

    def _probe_width(self, ex: ExecSpec) -> int:
        return 2 * max(self.dcp, 1) * self._union_width(ex) + 128

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        out = super().outputs(g, ex)
        if self.union_probe:
            out = out + [
                StateDecl(DBG_UNION, (ex.M // ex.T, 1, self._probe_width(ex)), I32, "w"),
                StateDecl(DBG_ROWS, (ex.M, 1, 256), I32, "w"),
            ]
        return out

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        if self.union_probe:
            out.update({k: raw_outputs[k] for k in (DBG_UNION, DBG_ROWS)})
        return out

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        out = super().scratch(g, ex, plan)
        if self.union_probe:
            out = out + [
                ScratchReq("qp_u", pltpu.VMEM((ex.MB // ex.T, 1, self._probe_width(ex)), I32)),
                ScratchReq("qp_r", pltpu.VMEM((ex.MB, 1, 256), I32)),
                ScratchReq("qp_sem", pltpu.SemaphoreType.DMA((2,))),
            ]
        return out

    def _probe_halves(self, ctx: Any, q: int) -> tuple[jax.Array, jax.Array]:
        """Request `q`'s union ids / codes `[1, D KU]` as this rank holds them before attention."""
        return ctx.scr["qu_id"][q], ctx.scr["qu_cd"][q]

    def _probe_dump(self, ctx: Any, l: Any) -> None:  # noqa: E741
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        nreq, D = ex.MB // T, max(self.dcp, 1)
        lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)
        for q in range(nreq):
            ids, codes = self._probe_halves(ctx, q)
            tail = jnp.zeros((1, 128), I32)
            for r in range(D):
                tail = jnp.where(lane == r, scr["qu_n"][q, r], tail)
            scr["qp_u"][q] = jnp.concatenate([ids, codes, tail], axis=1)
        lane2 = jax.lax.broadcasted_iota(I32, (1, 256), 1)
        z = jnp.concatenate([jnp.zeros((1, 128), I32), scr["qi_z"][...]], axis=1)
        for r in range(ex.MB):
            scr["qp_r"][r] = jnp.where(lane2 == 0, scr["qi_kvl"][r], jnp.where(lane2 == 1, scr["qc_sga"][r], z))
        o = ctx.stream.sp.block_ord(l, self.name)
        cps = [
            pltpu.make_async_copy(scr["qp_u"], ctx.O[DBG_UNION].at[o, pl.ds(0, nreq)], scr["qp_sem"].at[0]),
            pltpu.make_async_copy(scr["qp_r"], ctx.O[DBG_ROWS].at[o, pl.ds(0, ex.MB)], scr["qp_sem"].at[1]),
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()

    def _attend_unions(self, ctx: Any, l: Any, zeros: tuple) -> list[jax.Array]:  # noqa: E741
        if self.union_probe:
            self._probe_dump(ctx, l)
        return self._attend_body(ctx, l, zeros)

    def _attend_body(self, ctx: Any, l: Any, zeros: tuple) -> list[jax.Array]:  # noqa: E741
        return super()._attend_unions(ctx, l, zeros)


class _LeanMixin(_ProbeMixin):
    """D08 S1 stages B1 / B2 / B3 (module docstring) on MM's union path; all off = MM's code."""

    def __init__(
        self, *, union_bounded: bool = False, union_lean: bool = False, slot_batch: bool = False,
        slot_batch_parts: str = "", union_issue: int = 0, union_unpack: bool = False, **kw: Any
    ) -> None:
        super().__init__(**kw)
        # q38 L2-UNPACK: the union tile body's strided full-vreg ring read (union_unpack.py), head pair 0 only.
        self.union_unpack = bool(union_unpack)
        if self.union_unpack:
            assert self.union_attn, "X-UA-UNPACK acts on the union tile body"
        # q38 L2-UAISSUE: 0 off, 1 stage (a) slab-bounded physical ids, 2 stages (a) + (b) (union_issue.py).
        assert int(union_issue) in ui.LEVELS, union_issue
        self.union_issue = int(union_issue)
        if self.union_issue:
            assert self.union_attn and self.hbm_tables, "X-UA-ISSUE: union attention with HBM page tables"
            assert not self.union_phys, "X-UA-ISSUE builds its own physical ids (wholephys off)"
        self.union_bounded, self.union_lean, self.slot_batch = bool(union_bounded), bool(union_lean), bool(slot_batch)
        # B3's three parts (all with `slot_batch`; `slot_batch_parts` = a comma subset, for the tests' localisation).
        parts = set((slot_batch_parts or "newgrp,recent,uids").split(",")) if self.slot_batch else set()
        assert parts <= {"newgrp", "recent", "uids"}, parts
        self._bg, self._br, self._bu = "newgrp" in parts, "recent" in parts, "uids" in parts
        assert self.union_attn and self.per_request_scores, "D08 S1 acts on the union path"
        assert not self.compressed_dcp, "D08 S1: main-KV DCP only (compressed-key DCP stays on MM's path)"
        assert not (self._bu and self.union_phys), "D08 S1 B3 double-buffers the union ids (not wholephys)"

    @property
    def _shard_lean(self) -> bool:
        return self.union_lean and self.shard_index

    # ------------------------------------------------------------ scratch
    def _union_scratch(self, ex: ExecSpec) -> list[ScratchReq]:
        out = super()._union_scratch(ex)
        if not self._shard_lean:
            return out
        # B2: owners compact straight into per-destination send slots (`qgl_send [D, npr, 1, 2 KU + 128]`, slot h =
        # union half h + tail); each rank receives its own half from every run (`qgl_recv`), attention reads it there.
        drop = {"qu_id", "qu_cd", "qgu_send", "qgu_recv", "qgu_ssem", "qgu_rsem"}
        npr, n = self._shard_rows(ex)[0], max(ex.R - 1, 1)
        D, W = max(self.dcp, 1), 2 * self._union_width(ex) + 128
        return [s for s in out if s.name not in drop] + [
            ScratchReq("qgl_send", pltpu.VMEM((D, npr, 1, W), I32)),
            ScratchReq("qgl_recv", pltpu.VMEM((self._runs(ex), npr, 1, W), I32)),
            ScratchReq("qgl_ssem", pltpu.SemaphoreType.DMA((n,))),
            ScratchReq("qgl_rsem", pltpu.SemaphoreType.DMA((n,))),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        out = super().scratch(g, ex, plan)
        drop: set[str] = set()
        if self.union_lean:
            drop |= {"qi_id", "qi_ids"}
            if self.shard_index:
                drop |= {"qg_send", "qg_recv", "qg_ssem", "qg_rsem"}
        if self._bg:
            drop |= {"qc_ring", "qc_rope", "qc_ck", "qc_rnew"}
        if self._br:
            drop |= {"qc_old"}
        if self._bu:
            drop |= {"qu_row"}
        out = [s for s in out if s.name not in drop]
        nreq, KU = ex.MB // ex.T, self._union_width(ex)
        if self._bg:
            out = out + [
                ScratchReq("qb_ring", pltpu.VMEM((nreq, 8, ix.IDX_DIM), BF16)),
                ScratchReq("qb_rope", pltpu.VMEM((nreq, 1, 128), F32)),
                ScratchReq("qb_ck", pltpu.VMEM((nreq, 1, ix.IDX_DIM), BF16)),
                ScratchReq("qb_rnew", pltpu.VMEM((nreq, 8, ix.IDX_DIM), BF16)),
            ]
        if self._br:
            out = out + [ScratchReq("qb_old", pltpu.VMEM((nreq, MICRO, 1, 4, 256), jnp.float8_e4m3fn))]
        if self._bg or self._br:
            out = out + [ScratchReq("qb_sem", pltpu.SemaphoreType.DMA((4,)))]
        if self._bu:
            out = out + [
                ScratchReq("qu_rowb", pltpu.SMEM((2, KU + 128), I32)),
                ScratchReq("qb_usem", pltpu.SemaphoreType.DMA((2,))),
            ]
        if self.union_issue:  # q38 L2-UAISSUE: the layer's table slab and every request's physical ids
            out = out + [
                ScratchReq("qx_slab", pltpu.VMEM((nreq, ui.slab_width(self.pps, self.dcp)), I32)),
                ScratchReq("qx_pv", pltpu.VMEM((nreq, KU), I32)),
                ScratchReq("qx_sem", pltpu.SemaphoreType.DMA((1,))),
            ]
        return out

    # ------------------------------------------------------------ q38 L2-UAISSUE (stage a)
    def _issue_cols(self, ctx: Any) -> int:
        """Slab columns a union id can address at this pool's page size (<= the planned slab width)."""
        per_page = ctx.extra[POOL].shape[1] // MICRO
        cols = ui.slab_cols(self.max_groups, per_page, max(self.dcp, 1), self.pps)
        assert cols <= ui.slab_width(self.pps, self.dcp), (cols, self.pps, self.dcp, per_page)
        return cols

    def _issue_slab(self, ctx: Any, l: Any) -> Any:  # noqa: E741
        """This layer's slab copy: every request's main-table row, the columns a union id can address."""
        ex, mt = ctx.ex, ctx.extra[MTAB]
        assert mt.shape[1] == ex.MB // ex.T and mt.shape[2] == self.pps, (mt.shape, ex.MB // ex.T, self.pps)
        o = ctx.stream.sp.block_ord(l, self.name) if self.n_layers else 0
        return ui.slab_copy(mt, o, self._issue_cols(ctx), ctx.scr["qx_slab"], ctx.scr["qx_sem"].at[0])

    def load_tables(self, ctx: Any, l: Any) -> None:  # noqa: E741
        """MM's SMEM table windows, then (X-UA-ISSUE) this layer's slab copy, waited in `_issue_prep` (once per layer:
        `_select` calls this once before the layer's attention). q38 L2-XUAV5: the start is recorded for this layer
        (`ctx._xua_slab_l = l`), so `_issue_prep` starts the copy itself only when this method was not reached for the
        layer (an override that does not call it: qsa_full_idx's `table_prefetch` windows, VLLM_TPU_MK_TABLE_SLIM=2)."""
        super().load_tables(ctx, l)
        if self.union_issue:
            self._issue_slab(ctx, l).start()
            ctx._xua_slab_l = l  # q38 L2-XUAV5: this layer's slab copy is in flight

    def _issue_prep(self, ctx: Any, l: Any, union: Any, rk: Any) -> None:  # noqa: E741
        """Wait the slab, then every request's union ids -> physical micro-page ids (`qx_pv[q]`). q38 L2-XUAV5: if
        `load_tables` did not start this layer's slab copy (qsa_full_idx's prefetch override never calls it), start it here
        first (not hidden behind the selection, still exact); the per-layer record is consumed, so the next layer decides
        afresh and a copy is never started twice."""
        ex, scr = ctx.ex, ctx.scr
        if getattr(ctx, "_xua_slab_l", None) is not l:  # q38 L2-XUAV5: no start recorded for this layer
            self._issue_slab(ctx, l).start()
        ctx._xua_slab_l = None  # q38 L2-XUAV5: consumed (per layer)
        self._issue_slab(ctx, l).wait()
        ui.prep(nreq=ex.MB // ex.T, union_ids=lambda q: union(q)[0], n_union=lambda q: scr["qu_n"][q, rk],
                slab=scr["qx_slab"], cols=self._issue_cols(ctx), per_page=ctx.extra[POOL].shape[1] // MICRO,
                dcp=max(self.dcp, 1), pv=scr["qx_pv"])

    def _probe_halves(self, ctx: Any, q: int) -> tuple[jax.Array, jax.Array]:
        if not self._shard_lean:
            return super()._probe_halves(ctx, q)
        ex, scr = ctx.ex, ctx.scr
        D, KU = max(self.dcp, 1), self._union_width(ex)
        npr = self._shard_rows(ex)[0]
        own = scr["qgl_recv"][q // npr, q % npr]  # [1, 2 KU + 128]: this rank's half only
        rk = jax.lax.rem(ctx.me, D)
        full = lambda x: jnp.concatenate([jnp.where(rk == h, x, 0) for h in range(D)], axis=1)  # noqa: E731
        return full(own[:, :KU]), full(own[:, KU : 2 * KU])

    # ------------------------------------------------------------ B3 helpers
    def _gc(self, ctx: Any, q: Any) -> Any:
        """Request `q`'s completed group (`_new_group`'s `p3 // 4`, the same scalar ops)."""
        c0 = ctx.extra[CTXQ][q]
        p3 = c0 + jax.lax.rem(3 - jax.lax.rem(c0, 4) + 4, 4)
        return p3 // 4

    def _new_group_one(self, ctx: Any, q: Any, raw_q: jax.Array) -> None:
        """MM's `_new_group` arithmetic for slot `q` from the batched rows (outputs into `qb_ck` / `qb_rnew`)."""
        scr, g, T = ctx.scr, ctx.g, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        ring = scr["qb_ring"][q].astype(F32)
        p3 = c0 + jax.lax.rem(3 - jax.lax.rem(c0, 4) + 4, 4)  # first position = 3 mod 4
        j0 = p3 - 3
        ms = []
        for j in range(4):  # MM's sequential f32 sum order: ((m0 + m1) + m2) + m3
            pos = j0 + j
            cur = pick_row(raw_q, pos - c0)
            old = pick_row(ring, jax.lax.rem(pos, 8))
            ms.append(jnp.where(pos >= c0, cur, old))
        acc = ms[0] + ms[1] + ms[2] + ms[3]
        pooled = rnd(acc * 0.25)
        rope = scr["qb_rope"][q]
        cos, sin = rope[:, : self.rot // 2], rope[:, self.rot // 2 : self.rot]
        ck = ix.index_rope(
            rmsnorm(pooled, ctx.sm(ctx._slot, "qsa_ikn"), g.eps, zero_centered=True), cos, sin, self.rot,
            pltpu.bitcast,
        )
        scr["qb_ck"][q] = ck.astype(BF16)
        end = c0 + T
        lane = jax.lax.broadcasted_iota(I32, (8, 1), 0)
        p_l = end - 8 + jax.lax.rem(lane - jax.lax.rem(end - 8, 8) + 16, 8)
        cur_rows = jnp.zeros((8, ix.IDX_DIM), F32)
        for t in range(T):
            cur_rows = jnp.where(p_l == c0 + t, raw_q[t : t + 1], cur_rows)
        scr["qb_rnew"][q] = jnp.where(p_l >= c0, cur_rows, ring).astype(BF16)

    def _new_group_outs(self, ctx: Any, l: Any) -> list[Any]:  # noqa: E741
        scr, nreq = ctx.scr, ctx.ex.MB // ctx.ex.T
        o = ctx.stream.sp.block_ord(l, self.name)
        return [
            pltpu.make_async_copy(scr[src], ctx.O[dst].at[o, pl.ds(0, nreq)], scr["qb_sem"].at[3])
            for src, dst in (("qb_ck", CK_NEW), ("qb_rnew", RING_NEW))
        ]

    @_scoped("qsa_newgrp")
    def _new_groups(self, ctx: Any, l: Any, raw: jax.Array) -> None:  # noqa: E741
        """B3 1d: every slot's completed group; one DMA each for the ring and RoPE rows, output copies started
        (waited by `_select`'s end)."""
        scr, T = ctx.scr, ctx.ex.T
        nreq = ctx.ex.MB // T
        rb = self.row_base(ctx, l)
        cps = [
            pltpu.make_async_copy(ctx.extra[RING].at[pl.ds(rb, nreq)], scr["qb_ring"], scr["qb_sem"].at[0]),
            pltpu.make_async_copy(ctx.extra[ROPE_CK].at[pl.ds(0, nreq)], scr["qb_rope"], scr["qb_sem"].at[1]),
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        if self.loop_requests:

            def body(q: Any, carry: Any) -> Any:
                self._new_group_one(ctx, q, scr["qc_irq"][q])
                return carry

            jax.lax.fori_loop(0, nreq, body, 0)
        else:
            for q in range(nreq):
                self._new_group_one(ctx, q, raw[q * T : (q + 1) * T])
        for c in self._new_group_outs(ctx, l):
            c.start()

    def _recent_copy(self, ctx: Any, l: Any, q: Any) -> Any:  # noqa: E741
        """MM's `_recent_tile` pool read of slot `q` (the 4 positions from `4 (ctx // 4)`) into `qb_old[q]`."""
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        per_page = ctx.extra[POOL].shape[1] // MICRO
        shift = per_page.bit_length() - 1
        cshift = shift + self.dcp.bit_length() - 1  # DCP: a column covers `dcp * B` tokens
        qb = self.tab_base(ctx, l) + q
        micro = self.tables(ctx)[1][qb, ga >> cshift] * per_page + (ga & (per_page - 1))
        return pltpu.make_async_copy(
            ctx.extra[POOL].at[micro >> shift, pl.ds((micro & (per_page - 1)) * MICRO, MICRO)],
            ctx.scr["qb_old"].at[q],
            ctx.scr["qb_sem"].at[2],
        )

    def _each_slot(self, fn: Any, nreq: int) -> None:
        """`fn(q)` for every slot: a fori_loop with `loop_requests` (traced q), else static."""
        if self.loop_requests:

            def body(q: Any, carry: Any) -> Any:
                fn(q)
                return carry

            jax.lax.fori_loop(0, nreq, body, 0)
        else:
            for q in range(nreq):
                fn(q)

    @_scoped("qsa_recent")
    def _recent_issue(self, ctx: Any, l: Any) -> None:  # noqa: E741
        """B3 1e: every slot's recent-tile pool read in flight at once (the layer's tables are loaded)."""
        self._each_slot(lambda q: self._recent_copy(ctx, l, q).start(), ctx.ex.MB // ctx.ex.T)

    @_scoped("qsa_recent")
    def _recent_finish(self, ctx: Any, l: Any) -> None:  # noqa: E741
        """Every slot's copy waited before any slot is read: the copies share one semaphore, and a wait only says that
        one copy's bytes arrived, not which copy (out-of-order completion; l1-d08s1-diag-i1 j-fc148ed5 caught a per-slot
        wait-then-read with the interpreter's on_wait DMAs, which complete the pending copies last-in first-out)."""
        n = ctx.ex.MB // ctx.ex.T
        self._each_slot(lambda q: self._recent_copy(ctx, l, q).wait(), n)
        self._each_slot(lambda q: self._recent_one(ctx, l, q), n)

    def _recent_one(self, ctx: Any, l: Any, q: Any) -> None:  # noqa: E741
        """MM's `_recent_tile` arithmetic for slot `q` (pool rows already in `qb_old[q]`)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        sk, sv = ctx.extra[SCALES][0], ctx.extra[SCALES][1]
        k8, v8 = _unpack_kv(scr["qb_old"].at[q], self.offset(ctx, l), ex.interpret)
        pad = jnp.zeros((4, 256), F32)
        k_old = jnp.concatenate([k8.astype(F32), pad], axis=0)
        v_old = jnp.concatenate([v8.astype(F32), pad], axis=0)
        if self.loop_requests:
            kv = scr["qc_kvq"][q]
        else:
            kv = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)
        k_cur, v_cur = pool_units(kv[:, :256], sk), pool_units(kv[:, 256:], sv)
        pos = 4 * ga + jax.lax.broadcasted_iota(I32, (8, 1), 0)
        k_t = jnp.where(pos < c0, k_old, 0.0)
        v_t = jnp.where(pos < c0, v_old, 0.0)
        for t in range(T):
            k_t = jnp.where(pos == c0 + t, k_cur[t : t + 1], k_t)
            v_t = jnp.where(pos == c0 + t, v_cur[t : t + 1], v_t)
        if self.dcp > 1:  # the group peers' slots: their pool rows / tokens, scored there
            own = self._dcp_owned(ctx, pos)
            k_t, v_t = jnp.where(own, k_t, 0.0), jnp.where(own, v_t, 0.0)
        scr["qc_rk"][q] = k_t
        scr["qc_rv"][q] = v_t

    # ------------------------------------------------------------ B1 / B2 compaction target
    def _compact(self, ctx: Any, q: Any, r: int, code: jax.Array, KU: int, npr: int) -> None:
        """Union half `r` of request `q` (this rank owns it): MM's compaction, all passes or (B1) only the passes
        below the count, into `qu_id` / `qu_cd` (or, B2 with the sharded indexer, the send slot `qgl_send[r]`)."""
        scr = ctx.scr
        S = code.shape[0]
        consts = ua.union_consts(S, KU)
        if self._shard_lean:
            i = q - self._run(ctx) * npr

            def put(j0: int, w: int, ids: jax.Array, codes: jax.Array) -> None:
                scr["qgl_send"][r, i, :, pl.ds(j0, w)] = ids
                scr["qgl_send"][r, i, :, pl.ds(KU + j0, w)] = codes

        else:

            def put(j0: int, w: int, ids: jax.Array, codes: jax.Array) -> None:
                scr["qu_id"][q, :, pl.ds(r * KU + j0, w)] = ids
                scr["qu_cd"][q, :, pl.ds(r * KU + j0, w)] = codes

        if not self.union_bounded:
            ids, codes, nu = ua.compact_union(code, KU, consts)
            put(0, KU, ids, codes)
            scr["qu_n"][q, r] = jnp.sum(nu)
            return
        pro = _union_prologue(code, consts)
        scr["qu_n"][q, r] = jnp.sum(pro[-1])
        nsc = scr["qu_n"][q, r]
        W = consts[2].shape[0]
        for j0 in range(0, KU, W):

            @pl.when(nsc > j0)
            def _(j0=j0):
                ids, codes = _union_pass(pro, consts, j0)
                put(j0, W, ids, codes)

    # ------------------------------------------------------------ selection
    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        """MM's `QSAFullStepBlock._select` (main-KV paths; compressed-key DCP refused) with the S1 stages."""
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        MB, T, K, G = ex.MB, ex.T, self.budget, self.max_groups
        S, rot = G // 128, self.rot
        ctx._slot = slot
        lean, batch, recent = self.union_lean, self._bg, self._br
        with ctx.stream.sc("qsa_idx_proj", 2), ctx.stream.segment(lazy=False):
            yi = rnd(ctx.stream.dmatmul(IDX_W, scr["xb"], IDX_N))
        cos = scr["qf_rope"][:, : rot // 2]
        sin = scr["qf_rope"][:, rot // 2 : rot]
        iq = ix.index_query(yi, ctx.sm(slot, "qsa_iqn"), cos, sin, rot, g.eps, pltpu.bitcast)
        raw = yi[:, 4 * ix.IDX_DIM : 5 * ix.IDX_DIM]
        scr["qc_ir"][...] = raw.astype(BF16)
        o = ctx.stream.sp.block_ord(l, self.name)
        outs = [
            pltpu.make_async_copy(scr[src], ctx.O[dst].at[o, pl.ds(0, MB)], scr["qc_osem"].at[i])
            for i, (src, dst) in enumerate((("qf_kr", KR_NEW), ("qc_ir", IR_NEW)))
        ]
        for c in outs:
            c.start()
        nreq = MB // T
        for q in range(nreq):
            for t in range(T):
                for h in range(ix.IDX_HEADS):
                    i = t * ix.IDX_HEADS + h
                    scr["qi_q"][q, i : i + 1, :] = iq[h][q * T + t : q * T + t + 1]
        NC = self.n_chunks
        gid = (
            jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 1) * (G // NC)
            + jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 2) * 128
            + jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 3)
        )
        npr = self._shard_rows(ex)[0] if self.shard_index else nreq

        def select_rows(lo: Any, n: int, s0: Any) -> None:
            """Thresholds, row selection and the request's union (rows `[lo, lo + n)`, scores at `qi_s[s0 …]`)."""
            rid = jax.lax.broadcasted_iota(I32, (n, 1, 1), 0) + lo
            nv = jnp.zeros((n, 1, 1), I32)
            for i in range(n):  # `lo` may be traced (`loop_requests`)
                nv = jnp.where(rid == lo + i, ctx.extra[NVIS][lo + i], nv)
            ids = (
                jax.lax.broadcasted_iota(I32, (n, S, 128), 1) * 128
                + jax.lax.broadcasted_iota(I32, (n, S, 128), 2)
            )
            sc = pltpu.bitcast(scr["qi_s"][pl.ds(s0, n)].reshape(n, S, 128), I32)
            bits = jnp.where(ids < nv, sc, -1)
            with ctx.stream.sc("qsa_thr", 2):
                thr, tmax = ix.thresholds(bits, ids, nv, K, G)
            scr["qi_th"][pl.ds(lo, n)] = jnp.broadcast_to(thr, (n, 1, 128))
            scr["qi_tm"][pl.ds(lo, n)] = jnp.broadcast_to(tmax, (n, 1, 128))
            scr["qi_nv"][pl.ds(lo, n)] = jnp.broadcast_to(nv, (n, 1, 128))
            consts = ix.compact_consts(S, K)
            lid = consts[-1]
            scr["qi_z"][...] = jnp.zeros((1, 128), I32)
            scr["qu_code"][...] = jnp.zeros((S, 128), I32)

            def row(r: Any, carry: Any) -> Any:
                nvr = scr["qi_nv"][r][:, :1]
                ga = ctx.extra[CTXQ][r // T] // 4
                b = jnp.where(lid < nvr, pltpu.bitcast(scr["qi_s"][r - lo + s0].reshape(S, 128), I32), -1)
                th, tm = scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1]
                ov = nvr > K
                sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
                # Recent groups (>= g_a) leave the pool path; remember whether g_a was selected.
                scr["qc_sga"][r] = jnp.sum((sel & (lid == ga)).astype(I32))
                kept = jnp.sum((sel & (lid < ga)).astype(I32))
                # Row bit r - lo of every pool-path group it selects.
                scr["qu_code"][...] = scr["qu_code"][...] | jnp.where(sel & (lid < ga), jnp.left_shift(1, r - lo), 0)
                if not lean and self._need_ids():  # MM: the row's compacted ids (read only by the per-row path and the debug output)
                    bp = jnp.where(lid < ga, b, -1)
                    b2 = jnp.where(sel & (lid < ga), bp, -1)
                    pos = ix.compact_row(
                        b2, jnp.zeros((1, 1), I32), jnp.zeros((1, 1), I32) + G,
                        jnp.zeros((1, 1), I32) + kept, K, consts,
                    )
                    scr["qi_id"][r] = pos
                    if not self._skip_qi_ids():
                        cs = [
                            pltpu.make_async_copy(
                                scr["qi_id"].at[r], scr["qi_ids"].at[pl.ds(r, 1), pl.ds(0, K)], scr["qi_sem"].at[1]
                            ),
                            pltpu.make_async_copy(
                                scr["qi_z"], scr["qi_ids"].at[pl.ds(r, 1), pl.ds(K, 128)], scr["qi_sem"].at[2]
                            ),
                        ]
                        for c in cs:
                            c.start()
                        for c in cs:
                            c.wait()
                scr["qi_kvl"][r] = 4 * kept
                return carry

            with ctx.stream.sc("qsa_rowsel", 2):
                jax.lax.fori_loop(lo, lo + n, row, 0)
            KU, D = self._union_width(ex), max(self.dcp, 1)
            with ctx.stream.sc("qsa_ucompact", 2):
                # Main-KV DCP: one union per DCP rank over the groups it owns (group g lives on rank
                # (g // groups-per-page) % D), each compacted ascending.
                code_all = scr["qu_code"][...]
                gpp = ctx.extra[POOL].shape[1] // ua.MICRO
                gidx = jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128 + jax.lax.broadcasted_iota(I32, (S, 128), 1)
                for r in range(D):
                    code = code_all if D == 1 else jnp.where(jax.lax.rem(gidx // gpp, D) == r, code_all, 0)
                    self._compact(ctx, lo // T, r, code, KU, npr)

        self.load_tables(ctx, l)
        if self.loop_requests:
            for q in range(nreq):  # static stores into the request-major copies
                scr["qc_irq"][q] = raw[q * T : (q + 1) * T]
                scr["qc_kvq"][q] = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)
        if recent:
            self._recent_issue(ctx, l)
        if batch:
            self._new_groups(ctx, l, raw)

        def group(q: Any, raw_q: Any) -> tuple[Any, Any]:
            """(fresh key, group id) of request `q`'s completed group."""
            if batch:
                return scr["qb_ck"][q], self._gc(ctx, q)
            return self._new_group(ctx, l, q, raw_q)

        def score_merge(q: Any, qr: Any, ck: Any, gc: Any, first: Any, last: Any) -> None:
            self._score_request(ctx, q, qr, self.tab_base(ctx, l), row0=0, first=first, last=last, c_off=0)
            s_new = ix.score_rows(qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM)))
            scr["qi_s"][0:T] = jnp.where(gid == gc, s_new[:, :1].reshape(T, 1, 1, 1), scr["qi_s"][0:T])
            select_rows(q * T, T, 0)

        exchange = self._exchange_lean if lean else self._exchange_selection
        if self.loop_requests:

            def req(q: Any, carry: Any) -> Any:
                ck, gc = group(q, scr["qc_irq"][q] if not batch else None)

                def own() -> None:
                    score_merge(
                        q, scr["qi_q"][q].astype(BF16), ck, gc, jax.lax.rem(q, npr) == 0,
                        (jax.lax.rem(q, npr) == npr - 1) | (q == nreq - 1),
                    )

                if self.shard_index:
                    pl.when(q // npr == self._run(ctx))(own)
                else:
                    own()
                if not recent:
                    self._recent_tile(ctx, l, q)
                return carry

            jax.lax.fori_loop(0, nreq, req, 0) if not self.select_owned else self._owned_slots(ctx, nreq, npr, group, score_merge)  # q38 L1-SELLOOP
            if recent:
                self._recent_finish(ctx, l)
            if self.shard_index:
                exchange(ctx)
        elif self.shard_index:
            for q in range(nreq):
                ck, gc = group(q, raw[q * T : (q + 1) * T])

                @pl.when(q // npr == self._run(ctx))
                def _(q=q, ck=ck, gc=gc):
                    score_merge(
                        q, scr["qi_q"][q].astype(BF16), ck, gc, q % npr == 0, q % npr == npr - 1 or q == nreq - 1
                    )

                if not recent:
                    self._recent_tile(ctx, l, q)
            if recent:
                self._recent_finish(ctx, l)
            exchange(ctx)
        else:
            for q in range(nreq):
                qr = scr["qi_q"][q].astype(BF16)
                self._score_request(ctx, q, qr, self.tab_base(ctx, l), last=q == nreq - 1, row0=0)
                ck, gc = group(q, raw[q * T : (q + 1) * T])
                s_new = ix.score_rows(qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM)))[:, :1]
                # The completed group's fresh key replaces its stale cached column.
                scr["qi_s"][0:T] = jnp.where(gid == gc, s_new.reshape(T, 1, 1, 1), scr["qi_s"][0:T])
                if not recent:
                    self._recent_tile(ctx, l, q)
                select_rows(q * T, T, 0)
            if recent:
                self._recent_finish(ctx, l)
        for c in outs:
            c.wait()
        if batch:
            for c in self._new_group_outs(ctx, l):
                c.wait()
        if self.debug:  # the kernel's queries (and, without B2, the pool-path ids)
            from .qsa_full_b import DBG_IDS, DBG_IQ

            for h in range(ix.IDX_HEADS):
                scr["qi_dbg"][:, h * ix.IDX_DIM : (h + 1) * ix.IDX_DIM] = iq[h]
            pairs = ((scr["qi_dbg"], DBG_IQ),) + (() if lean else ((scr["qi_id"], DBG_IDS),))
            for src, dst in pairs:
                c = pltpu.make_async_copy(src, ctx.O[dst].at[o, pl.ds(0, MB)], scr["qi_sem"].at[0])
                c.start()
                c.wait()

    @_scoped("qsa_ux")
    def _exchange_lean(self, ctx: Any) -> None:
        """B2 (1b + 1c): the one exchange of the sharded indexer. Each owner's send slot `h` (union half h,
        compacted there by `_compact`) gets the tail `[qu_n (D lanes) | qi_kvl (T) | g_a flags (T)]` of each of its
        requests; rank `me ^ off` receives slot `(me ^ off) % D`, its own half, into `qgl_recv[me]`; then every
        rank unpacks only the tails. int32: the gathered state is the owners' exactly."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        nreq, KU = ex.MB // T, self._union_width(ex)
        npr = self._shard_rows(ex)[0]
        me, D = ctx.me, max(self.dcp, 1)
        lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)
        run = self._run(ctx)
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)  # every rank: u_ids copies it (select_rows zeroes it on owners)

        def pack(i: Any, carry: Any) -> Any:
            q = jnp.minimum(run * npr + i, nreq - 1)  # past the last request: never read
            tail = jnp.zeros((1, 128), I32)
            for r in range(D):
                tail = jnp.where(lane == r, scr["qu_n"][q, r], tail)
            for t in range(T):
                tail = jnp.where(lane == D + t, scr["qi_kvl"][q * T + t], tail)
                tail = jnp.where(lane == D + T + t, scr["qc_sga"][q * T + t], tail)
            for h in range(D):
                scr["qgl_send"][h, i, :, pl.ds(2 * KU, 128)] = tail
            return carry

        jax.lax.fori_loop(0, npr, pack, 0)
        scr["qgl_recv"][run] = scr["qgl_send"][jax.lax.rem(me, D)]
        if ex.R > 1:
            ctx.comm.exchange(ctx.comm.group_copies(
                lambda off: scr["qgl_send"].at[jax.lax.rem(me ^ off, D)], scr["qgl_recv"].at[me],
                scr["qgl_ssem"], scr["qgl_rsem"], ex.R,
            ))

        def unpack(q: Any, carry: Any) -> Any:
            o = q // npr
            tail = scr["qgl_recv"][o, q - o * npr, :, pl.ds(2 * KU, 128)]
            for r in range(D):
                scr["qu_n"][q, r] = jnp.sum(jnp.where(lane == r, tail, 0))
            for t in range(T):
                scr["qi_kvl"][q * T + t] = jnp.sum(jnp.where(lane == D + t, tail, 0))
                scr["qc_sga"][q * T + t] = jnp.sum(jnp.where(lane == D + T + t, tail, 0))
            return carry

        jax.lax.fori_loop(0, nreq, unpack, 0)

    # ------------------------------------------------------------ attention
    def _attend_body(self, ctx: Any, l: Any, zeros: tuple) -> list[jax.Array]:  # noqa: E741
        """MM's `_attend_unions`, reading the union from the receive buffer (B2 with the sharded indexer) and
        with double-buffered union-id rows (B3); MM's own body otherwise."""
        shard_lean, issue, unpack = self._shard_lean, self.union_issue, self.union_unpack
        if not (shard_lean or self._bu or issue or unpack):
            return super()._attend_body(ctx, l, zeros)
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        nh, KU, nreq = self.nh(g, ex), self._union_width(ex), ex.MB // ex.T
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)
        sc = ctx.stream.sc
        rk = jax.lax.rem(ctx.me, self.dcp) if self.dcp > 1 else 0
        uoff = pl.multiple_of(rk * KU, 128) if self.dcp > 1 else 0
        npr = self._shard_rows(ex)[0] if self.shard_index else nreq

        def union(q: Any) -> tuple[Any, Any]:
            """This rank's union of request `q`: (ids `[1, KU]` ref, codes `[1, KU]` ref)."""
            if shard_lean:
                o_ = q // npr
                i_ = q - o_ * npr
                return scr["qgl_recv"].at[o_, i_, :, pl.ds(0, KU)], scr["qgl_recv"].at[o_, i_, :, pl.ds(KU, KU)]
            return scr["qu_id"].at[q, :, pl.ds(uoff, KU)], scr["qu_cd"].at[q, :, pl.ds(uoff, KU)]

        if issue:  # q38 L2-UAISSUE stage (a): the SMEM id rows come from every request's physical ids
            with sc("qsa_u_phys", 2):
                self._issue_prep(ctx, l, union, rk)

        def id_src(q: Any) -> Any:
            return scr["qx_pv"].at[pl.ds(q, 1)] if issue else union(q)[0]

        if self._bu:

            def ucopies(q: Any, b: Any) -> list[Any]:
                return [
                    pltpu.make_async_copy(
                        id_src(q), scr["qu_rowb"].at[pl.ds(b, 1), pl.ds(0, KU)], scr["qb_usem"].at[b]
                    ),
                    pltpu.make_async_copy(
                        scr["qi_z"], scr["qu_rowb"].at[pl.ds(b, 1), pl.ds(KU, 128)], scr["qb_usem"].at[b]
                    ),
                ]

            for c in ucopies(0, 0):
                c.start()

        def req(q: Any, outs: Any) -> Any:
            src, codes = union(q)
            n_u = scr["qu_n"][q, rk]
            with sc("qsa_u_ids", 2):
                if self._bu:
                    b = jax.lax.rem(q, 2)
                    for c in ucopies(q, b):
                        c.wait()

                    @pl.when(q + 1 < nreq)
                    def _():
                        for c in ucopies(q + 1, 1 - b):
                            c.start()

                    ids = scr["qu_rowb"].at[pl.ds(b, 1)]
                else:
                    if issue:
                        src = id_src(q)
                    elif self.union_phys:
                        o = ctx.stream.sp.block_ord(l, self.name) if self.n_layers else 0
                        cp = pltpu.make_async_copy(
                            ctx.extra[MTAB].at[o, pl.ds(q, 1)], scr["qu_tab"], scr["qu_sem"].at[2]
                        )
                        cp.start()
                        cp.wait()
                        per_page = ctx.extra[POOL].shape[1] // ua.MICRO
                        tab = scr["qu_tab"][...]

                        def chunk(c: Any, carry: Any) -> Any:
                            lo = pl.multiple_of(c * 128, 128)
                            scr["qu_pv"][:, pl.ds(lo, 128)] = ua.physical_ids(
                                src[:, pl.ds(lo, 128)], tab, per_page, self.dcp
                            )
                            return carry

                        jax.lax.fori_loop(0, (n_u + 127) // 128, chunk, 0)
                        src = scr["qu_pv"]
                    cs = [
                        pltpu.make_async_copy(src, scr["qu_row"].at[pl.ds(0, 1), pl.ds(0, KU)], scr["qu_sem"].at[0]),
                        pltpu.make_async_copy(
                            scr["qi_z"], scr["qu_row"].at[pl.ds(0, 1), pl.ds(KU, 128)], scr["qu_sem"].at[1]
                        ),
                    ]
                    for c in cs:
                        c.start()
                    for c in cs:
                        c.wait()
                    ids = scr["qu_row"]
            rs = [q * T + t for t in range(T)]
            ok = [self._recent_ok(ctx, r, srow) for r in rs]
            ok_any = self._recent_ok(ctx, rs[0], scol)
            for r in rs[1:]:
                ok_any = ok_any | self._recent_ok(ctx, r, scol)
            if self.dcp > 1:  # every group rank's heads, this rank's first
                qs = [
                    jnp.concatenate([scr["qf_q"][r], scr["qd_q"][r]], axis=0) if self.dcp == 2
                    else self._group_q(ctx, r)
                    for r in rs
                ]
            else:
                qs = [scr["qf_q"][r] for r in rs]
            with sc("qsa_u_attend", 2):
                res = ua.attend_request(
                    T=T, nh=nh * self.dcp, pool=ctx.extra[POOL], ring=scr["qf_ring"],
                    sems=scr["qf_ring_sem"],
                    ids=ids, codes=codes, n_union=n_u,
                    table=self.tables(ctx)[1], table_row=self.tab_base(ctx, l) + q,
                    qs=qs,
                    gates=[scr["qf_g"][r] for r in rs],
                    has=[scr["qi_kvl"][r] > 0 for r in rs],
                    # q38 L2-UNPACK: head pair 0 (static; the whole-step call refuses a paired pool with the option on)
                    off=0 if unpack else self.offset(ctx, l), scale_k=ctx.extra[SCALES][0],
                    scale_v=ctx.extra[SCALES][1], sm_scale=g.HD**-0.5, interpret=ex.interpret,
                    recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok, ok_any), phys=self.union_phys or bool(issue),
                    dcp=self.dcp, sc=ctx.stream.sc, fast=issue >= 2, unpack=unpack,
                )
            if self.dcp > 1:
                for r, (att, lse) in zip(rs, res) if self.dcp == 2 else ():
                    lse = jnp.broadcast_to(lse, (2 * nh, 128))
                    scr["qd_o"][r], scr["qd_os"][r] = att[:nh], att[nh:]
                    scr["qd_l"][r], scr["qd_ls"][r] = lse[:nh], lse[nh:]
                return outs if self.dcp == 2 else self._group_stage(ctx, rs, res, outs)
            if self.union_stats:
                assert nreq <= UNION_STATS_REQS, (nreq, UNION_STATS_REQS)
                kept = [scr["qi_kvl"][r] // 4 for r in rs]
                lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)
                vals = [n_u, sum(kept[1:], kept[0]), kept[0], (kept[0] > 0).astype(I32)]
                for k in kept[1:]:
                    vals[2] = jnp.maximum(vals[2], k)
                    vals[3] = vals[3] + (k > 0).astype(I32)
                st = jnp.zeros((1, 128), I32)
                for i, v in enumerate(vals):
                    st = jnp.where(lane == i, v, st)
                scr["qu_st"][...] = st
                cp = pltpu.make_async_copy(
                    scr["qu_st"],
                    ctx.O[UNION_STATS].at[ctx.stream.sp.block_ord(l, self.name), q],
                    scr["qu_sem"].at[2],
                )
                cp.start()
                cp.wait()
            outs = list(outs)
            for t, r in enumerate(rs):
                for h in range(nh):
                    outs[h] = jnp.where(rows == r, res[t][h : h + 1], outs[h])
            return tuple(outs)

        return list(jax.lax.fori_loop(0, nreq, req, zeros))

    # q38 L1-SELLOOP begin: the owned-slot selection loop (VLLM_TPU_MK_SELECT_OWNED 1 / 2, default off). Appended to the
    # class, with `_select`'s request loop edited only within its existing `fori_loop` line (here and in the generated copy
    # qsa_full_idx._IdxMixin._select), so with the flag off no traced line moves.
    select_owned = 0  # 0: `_select` loops over every request slot (the owned work under `pl.when`)

    def engage_select_owned(self, mode: int) -> None:
        """Select the owned-slot loop (`_owned_slots`), mode 1 or 2 (0 = off). It needs the request loop, the sharded
        indexer and B3's `newgrp` + `recent` parts: then `_select`'s slot loop holds no work for a slot this rank does
        not own (the completed groups and the recent tiles are batched before / after the loop, and `group` is two
        reads), so leaving those slots out changes nothing."""
        mode = int(mode)
        assert mode in (0, 1, 2), ("select_owned: 0, 1 or 2", mode)
        if mode:
            assert self.loop_requests and self.shard_index and self._bg and self._br and not self.compressed_dcp, (
                "select_owned needs loop_requests, shard_index and slot_batch (parts newgrp + recent)",
                self.loop_requests, self.shard_index, self._bg, self._br, self.compressed_dcp,
            )
        self.select_owned = mode

    def _owned_slots(self, ctx: Any, nreq: int, npr: int, group: Any, score_merge: Any) -> None:
        """`_select`'s request loop over this rank's own slots only, in place of `fori_loop(0, nreq)` over every slot
        with the owned work under `pl.when(q // npr == run)`: slot q = run * npr + i for i < npr and q < nreq, the
        same slots in the same order with the same `first` (i == 0) / `last` (i == npr - 1 or q == nreq - 1) flags.
        A skipped slot's iteration only read `qb_ck[q]` and `ctx[q]` (`group`), so the state after the loop is
        bitwise the same. One request per run (npr = 1): one straight-line iteration (no loop, no back-edge). More:
        mode 1 = a fori_loop over this run's owned count, mode 2 = every owned slot unrolled. A slot that some run
        may not own (q >= nreq possible) runs under `pl.when(q < nreq)` (unrolled) or is outside the loop's count."""
        scr, runs, run = ctx.scr, self._runs(ctx.ex), self._run(ctx)

        def one(i: Any, last: Any) -> None:
            q = run * npr + i
            ck, gc = group(q, None)
            score_merge(q, scr["qi_q"][q].astype(BF16), ck, gc, i == 0, last)

        if npr == 1 or self.select_owned == 2:
            _selloop_note("straight" if npr == 1 else "unrolled", nreq, npr, runs)
            for i in range(npr):
                last = True if i == npr - 1 else (run * npr + i == nreq - 1)
                if (runs - 1) * npr + i < nreq:  # every run owns slot i
                    one(i, last)
                else:
                    pl.when(run * npr + i < nreq)(lambda i=i, last=last: one(i, last))
            return
        _selloop_note("loop", nreq, npr, runs)
        n = npr if runs * npr == nreq else jnp.clip(nreq - run * npr, 0, npr)

        def body(i: Any, carry: Any) -> Any:
            one(i, (i == npr - 1) | (run * npr + i == nreq - 1))
            return carry

        jax.lax.fori_loop(0, n, body, 0)

    # q38 L1-SELLOOP end


class QSAFullStepProbeBlock(_ProbeMixin, QSAFullStepBlock):
    """MM's stage-C block (DCP 1 / 2) with the union probe (tests)."""


class QSAFullStepGroupProbeBlock(_ProbeMixin, QSAFullStepGroupBlock):
    """D14's DCP-4 group block with the union probe (tests)."""


class QSAFullStepLeanBlock(_LeanMixin, QSAFullStepBlock):
    """MM's stage-C block (DCP 1 / 2) with the D08 S1 stages."""


class QSAFullStepGroupLeanBlock(_LeanMixin, QSAFullStepGroupBlock):
    """D14's DCP-4 group block with the D08 S1 stages (D14's exchange / merge methods inherited)."""


# q38 L1-SELLOOP begin (module part): the trace-time engagement record of `_LeanMixin._owned_slots` (tests read it; one
# log line per distinct shape and process).
SELLOOP_TRACES: dict = {}


def _selloop_note(path: str, nreq: int, npr: int, runs: int) -> None:
    key = (path, nreq, npr, runs)
    SELLOOP_TRACES[key] = SELLOOP_TRACES.get(key, 0) + 1
    if SELLOOP_TRACES[key] == 1:
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 L1-SELLOOP owned-slot selection traced: %s, %d request slots, %d per run over %d runs", path, nreq, npr,
            runs,
        )
# q38 L1-SELLOOP end
# ===================================================================== q38 L2-UAPIPE (X-UA-PIPE parts (i) and (ii))
# Appended after the lean classes so that no line the flag-off programs trace moves (op locations carry tracebacks).
PIPE_KEY = "union_pipe"


class _PipeMixin:
    """X-UA-PIPE (`union_attn.attend_request_pipe`) on the lean block's union attention: `union_pipe` = 1 co-issue (i),
    2 cross-request (ii), 3 both, 5 / 7 = 1 / 3 with the starts after the unpack. Bitwise identical to the same block
    without it (union_attn's note). (ii) extends D08 S1 B3's union-id buffers (`slot_batch` with its `uids` part) from
    two SMEM rows to `PIPE_IDS_BUFS`, each with its own DMA semaphore, so request q + 1's ids are resident while
    request q streams; it replaces B3's `qu_rowb` / `qb_usem` (D08 S1-1f, subsumed). `wholephys` stays refused (as B3
    refuses it). q38 L2-UAMERGE: X-UA-ISSUE (`union_issue` 1 / 2: the id rows hold `_issue_prep`'s physical ids, the
    copies and the wait are union_issue's inside the pipe) and X-UA-UNPACK (`union_unpack`: the pipe's tile body reads
    with union_unpack's strided loads at the static offset 0) compose with every pipe mode."""

    def __init__(self, *, union_pipe: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.union_pipe = ua.pipe_mode(union_pipe)
        assert self.union_pipe, "the X-UA-PIPE block is built only with a nonzero union_pipe"
        assert not self.union_phys, "X-UA-PIPE: wholephys is not composed (X-UA-ISSUE `union_issue` builds the physical ids)"
        if self.union_pipe & ua.PIPE_XREQ:
            assert self._bu, "X-UA-PIPE (ii) extends D08 S1 B3's union-id buffers (VLLM_TPU_MK_SLOT_BATCH=1, uids)"

    @property
    def _xreq(self) -> bool:
        return bool(self.union_pipe & ua.PIPE_XREQ)

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        out = super().scratch(g, ex, plan)
        if not self._xreq:
            return out
        KU = self._union_width(ex)
        return [s for s in out if s.name not in ("qu_rowb", "qb_usem")] + [
            ScratchReq("qx_rows", pltpu.SMEM((ua.PIPE_IDS_BUFS, KU + 128), I32)),
            ScratchReq("qx_usem", pltpu.SemaphoreType.DMA((ua.PIPE_IDS_BUFS,))),
        ]

    def _attend_body(self, ctx: Any, l: Any, zeros: tuple) -> list[jax.Array]:  # noqa: E741
        """`_LeanMixin._attend_body` (B2 receive-buffer unions, B3 ids, MM's per-request body) with
        `attend_request_pipe`; under (ii) the request loop is `pipe_request_loop` (ids rows + ring position)."""
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        nh, KU, nreq = self.nh(g, ex), self._union_width(ex), ex.MB // ex.T
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)
        sc = ctx.stream.sc
        rk = jax.lax.rem(ctx.me, self.dcp) if self.dcp > 1 else 0
        uoff = pl.multiple_of(rk * KU, 128) if self.dcp > 1 else 0
        shard_lean = self._shard_lean
        npr = self._shard_rows(ex)[0] if self.shard_index else nreq
        xreq = self._xreq
        issue, unpack = self.union_issue, self.union_unpack  # q38 L2-UAMERGE

        def union(q: Any) -> tuple[Any, Any]:
            """This rank's union of request `q`: (ids `[1, KU]` ref, codes `[1, KU]` ref)."""
            if shard_lean:
                o_ = q // npr
                i_ = q - o_ * npr
                return scr["qgl_recv"].at[o_, i_, :, pl.ds(0, KU)], scr["qgl_recv"].at[o_, i_, :, pl.ds(KU, KU)]
            return scr["qu_id"].at[q, :, pl.ds(uoff, KU)], scr["qu_cd"].at[q, :, pl.ds(uoff, KU)]

        if issue:  # q38 L2-UAMERGE: X-UA-ISSUE stage (a), every request's physical ids before the first id copy
            with sc("qsa_u_phys", 2):
                self._issue_prep(ctx, l, union, rk)

        def id_src(q: Any) -> Any:
            """The SMEM id row source of request q: its physical ids under X-UA-ISSUE, else its union ids."""
            return scr["qx_pv"].at[pl.ds(q, 1)] if issue else union(q)[0]

        def ids_copies(q: Any, k: Any) -> list[Any]:
            """(ii): request q's id row (+ the zero tail) -> `qx_rows[k]`, on `qx_usem[k]` (B3's two copies)."""
            return ua.pipe_ids_copies(id_src(q), scr["qi_z"], scr["qx_rows"], scr["qx_usem"], k)

        if self._bu and not xreq:

            def ucopies(q: Any, b: Any) -> list[Any]:
                return [
                    pltpu.make_async_copy(
                        id_src(q), scr["qu_rowb"].at[pl.ds(b, 1), pl.ds(0, KU)], scr["qb_usem"].at[b]
                    ),
                    pltpu.make_async_copy(
                        scr["qi_z"], scr["qu_rowb"].at[pl.ds(b, 1), pl.ds(KU, 128)], scr["qb_usem"].at[b]
                    ),
                ]

            for c in ucopies(0, 0):
                c.start()

        def req(q: Any, ctl: Any, outs: Any) -> Any:
            src, codes = union(q)
            n_u = scr["qu_n"][q, rk]
            if xreq:
                ids, row = scr["qx_rows"], ctl["ids_row"]
            else:
                row = 0
                with sc("qsa_u_ids", 2):
                    if self._bu:
                        b = jax.lax.rem(q, 2)
                        for c in ucopies(q, b):
                            c.wait()

                        @pl.when(q + 1 < nreq)
                        def _():
                            for c in ucopies(q + 1, 1 - b):
                                c.start()

                        ids = scr["qu_rowb"].at[pl.ds(b, 1)]
                    else:
                        src = id_src(q)  # q38 L2-UAMERGE: the physical ids under X-UA-ISSUE
                        cs = [
                            pltpu.make_async_copy(src, scr["qu_row"].at[pl.ds(0, 1), pl.ds(0, KU)], scr["qu_sem"].at[0]),
                            pltpu.make_async_copy(
                                scr["qi_z"], scr["qu_row"].at[pl.ds(0, 1), pl.ds(KU, 128)], scr["qu_sem"].at[1]
                            ),
                        ]
                        for c in cs:
                            c.start()
                        for c in cs:
                            c.wait()
                        ids = scr["qu_row"]
            rs = [q * T + t for t in range(T)]
            ok = [self._recent_ok(ctx, r, srow) for r in rs]
            ok_any = self._recent_ok(ctx, rs[0], scol)
            for r in rs[1:]:
                ok_any = ok_any | self._recent_ok(ctx, r, scol)
            if self.dcp > 1:  # every group rank's heads, this rank's first
                qs = [
                    jnp.concatenate([scr["qf_q"][r], scr["qd_q"][r]], axis=0) if self.dcp == 2
                    else self._group_q(ctx, r)
                    for r in rs
                ]
            else:
                qs = [scr["qf_q"][r] for r in rs]
            pk = {}
            if xreq:
                q1 = jnp.minimum(q + 1, nreq - 1)
                pk = dict(
                    base=ctl["base"], inflight=ctl["inflight"],
                    nxt=ua.PipeNext(ids_row=ctl["next_row"], n_union=scr["qu_n"][q1, rk],
                                    table_row=self.tab_base(ctx, l) + q1, has=ctl["has_next"]),
                )
            with sc("qsa_u_attend", 2):
                res, nt = ua.attend_request_pipe(
                    T=T, nh=nh * self.dcp, pool=ctx.extra[POOL], ring=scr["qf_ring"],
                    sems=scr["qf_ring_sem"],
                    ids=ids, codes=codes, n_union=n_u,
                    table=self.tables(ctx)[1], table_row=self.tab_base(ctx, l) + q,
                    qs=qs,
                    gates=[scr["qf_g"][r] for r in rs],
                    has=[scr["qi_kvl"][r] > 0 for r in rs],
                    # q38 L2-UAMERGE: X-UA-UNPACK reads head pair 0 (static; the whole-step call refuses a paired pool)
                    off=0 if unpack else self.offset(ctx, l), scale_k=ctx.extra[SCALES][0],
                    scale_v=ctx.extra[SCALES][1], sm_scale=g.HD**-0.5, interpret=ex.interpret,
                    recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok, ok_any), pipe=self.union_pipe,
                    ids_row=row, dcp=self.dcp, sc=ctx.stream.sc,
                    phys=bool(issue), fast=issue >= 2, unpack=unpack, **pk,  # q38 L2-UAMERGE
                )
            if self.dcp > 1:
                for r, (att, lse) in zip(rs, res) if self.dcp == 2 else ():
                    lse = jnp.broadcast_to(lse, (2 * nh, 128))
                    scr["qd_o"][r], scr["qd_os"][r] = att[:nh], att[nh:]
                    scr["qd_l"][r], scr["qd_ls"][r] = lse[:nh], lse[nh:]
                return (outs if self.dcp == 2 else self._group_stage(ctx, rs, res, outs)), nt
            if self.union_stats:
                assert nreq <= UNION_STATS_REQS, (nreq, UNION_STATS_REQS)
                kept = [scr["qi_kvl"][r] // 4 for r in rs]
                lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)
                vals = [n_u, sum(kept[1:], kept[0]), kept[0], (kept[0] > 0).astype(I32)]
                for k in kept[1:]:
                    vals[2] = jnp.maximum(vals[2], k)
                    vals[3] = vals[3] + (k > 0).astype(I32)
                st = jnp.zeros((1, 128), I32)
                for i, v in enumerate(vals):
                    st = jnp.where(lane == i, v, st)
                scr["qu_st"][...] = st
                cp = pltpu.make_async_copy(
                    scr["qu_st"],
                    ctx.O[UNION_STATS].at[ctx.stream.sp.block_ord(l, self.name), q],
                    scr["qu_sem"].at[2],
                )
                cp.start()
                cp.wait()
            outs = list(outs)
            for t, r in enumerate(rs):
                for h in range(nh):
                    outs[h] = jnp.where(rows == r, res[t][h : h + 1], outs[h])
            return tuple(outs), nt

        return list(ua.pipe_request_loop(
            nreq=nreq, nslots=scr["qf_ring"].shape[0], pipe=self.union_pipe, ids_copies=ids_copies, body=req,
            init=zeros, sc=sc,
        ))


class QSAFullStepPipeBlock(_PipeMixin, QSAFullStepLeanBlock):
    """MM's stage-C block (DCP 1 / 2) with the D08 S1 stages and X-UA-PIPE."""


class QSAFullStepGroupPipeBlock(_PipeMixin, QSAFullStepGroupLeanBlock):
    """D14's DCP-4 group block with the D08 S1 stages and X-UA-PIPE."""


def pipe_step_block(indexed: dict, qsa: dict | None) -> Any:
    """The X-UA-PIPE block when `indexed` carries `union_pipe` (the whole-step call adds it only when
    VLLM_TPU_MK_UNION_PIPE is nonzero), else None. Built on the lean class (whose all-off path is MM's / D14's code),
    so X-UA-PIPE composes with any D08 S1 stage set."""
    if PIPE_KEY not in indexed:
        return None
    group = int(indexed.get("dcp", 1) or 1) > 2
    return (QSAFullStepGroupPipeBlock if group else QSAFullStepPipeBlock)(**indexed, **(qsa or {}))

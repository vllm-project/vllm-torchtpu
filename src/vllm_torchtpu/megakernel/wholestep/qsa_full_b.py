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
"""Stage B: `QSAFullBlock` with the indexer in-kernel (no selection input).

Per QSA layer, after the stage-A projection:

  iq      = index_query(mixed @ W_idx)                      4 heads x 128, every row
  scores  = per request: its compressed keys DMA'd into VMEM, `score_rows` on the MXU
  ids     = exact top-K per row (`indexer.thresholds` + `compact_row`), lane-major -> SMEM
  pages   = ids + the row's partial group (if `len % 4`), kv_len = 4 min(n_vis, K) + len % 4
  attn    = `attend_row_mapped` (group ids -> pool micro-pages via the main table at DMA issue)

Rows are grouped by request, `ex.T` verify rows each (MTP verify). Prototype key layout:
`qsa_ck [requests, max_groups, 128]` BF16, contiguous per request (the served cache is paged;
see DESIGN.md). Groups completed during this step are stage C.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rnd
from ..spec import ExtraInput, ScratchReq, SmallField, StateDecl, WeightDecl
from . import indexer as ix
from .attend import attend_row_mapped
from .qsa_full import OFFSET, POOL, ROPE, SCALES, QSAFullBlock

CK = "qsa_ck"
NVIS = "qsa_nvis"  # [M] complete groups visible per row (len // 4)
TAIL = "qsa_tail"  # [M] tokens of the row's partial group (len % 4)
MTAB = "qsa_mtab"  # [requests, pages_per_seq] main-pool page table
IDX_W = "idx_w"
DBG_IQ = "qsa_dbg_iq"  # debug outputs: the kernel's index queries and selected ids per layer
DBG_IDS = "qsa_dbg_ids"
IDX_N = 4 * ix.IDX_DIM + ix.IDX_DIM  # 512 query + 128 raw-key columns


class QSAFullIndexedBlock(QSAFullBlock):
    """`QSAFullBlock` + in-kernel indexer and selection."""

    def __init__(
        self, *, max_groups: int, pages_per_seq: int, debug: bool = False, **kw: Any
    ) -> None:
        super().__init__(**kw)
        assert max_groups % 1024 == 0, max_groups  # whole (8, 128) score tiles per row
        self.max_groups, self.pps, self.debug = max_groups, pages_per_seq, debug

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        out = super().outputs(g, ex)
        if self.debug:
            out += [
                StateDecl(DBG_IQ, (ex.M, 4 * ix.IDX_DIM), F32, "w"),
                StateDecl(DBG_IDS, (ex.M, 1, self.budget), I32, "w"),
            ]
        return out

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        if self.debug:
            out.update({k: raw_outputs[k] for k in (DBG_IQ, DBG_IDS)})
        return out

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        # Declaration order is the stream's consumption order: qkvg, idx_w, then o_proj.
        qkvg, o_proj = super().weights(g, ex)
        return [qkvg, WeightDecl(IDX_W, K=g.H, N=IDX_N, dtype=BF16), o_proj]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return super().smalls(g, ex) + [SmallField("qsa_iqn", ix.IDX_DIM)]

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nreq = ex.M // ex.T
        return [
            ExtraInput(POOL, "hbm"),
            ExtraInput(CK, "hbm", (nreq, self.max_groups, ix.IDX_DIM), BF16),
            ExtraInput(NVIS, "smem", (ex.M,), I32),
            ExtraInput(TAIL, "smem", (ex.M,), I32),
            ExtraInput(MTAB, "smem", (nreq, self.pps), I32),
            ExtraInput(OFFSET, "smem", (1,), I32),
            ExtraInput(SCALES, "smem", (2,), F32),
            ExtraInput(ROPE, "hbm", (ex.M, 128), F32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        G, S, K, MB = self.max_groups, self.max_groups // 128, self.budget, ex.MB
        return super().scratch(g, ex, plan) + [
            ScratchReq("qi_ck", pltpu.VMEM((G, ix.IDX_DIM), BF16)),
            ScratchReq("qi_q", pltpu.VMEM((MB // ex.T, ex.T * ix.IDX_HEADS, ix.IDX_DIM), F32)),
            ScratchReq("qi_s", pltpu.VMEM((MB, S, 128), F32)),
            ScratchReq("qi_th", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_tm", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_nv", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_id", pltpu.VMEM((MB, 1, K), I32)),
            ScratchReq("qi_ids", pltpu.SMEM((MB, K + 128), I32)),
            ScratchReq("qi_kvl", pltpu.SMEM((MB,), I32)),
            ScratchReq("qi_z", pltpu.VMEM((1, 128), I32)),
            ScratchReq("qi_dbg", pltpu.VMEM((MB, 4 * ix.IDX_DIM), F32)),
            ScratchReq("qi_sem", pltpu.SemaphoreType.DMA((3,))),
        ]

    # ------------------------------------------------------------ in kernel
    def _select(self, ctx: Any, l: Any, slot: Any) -> None:
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        MB, T, K, G = ex.MB, ex.T, self.budget, self.max_groups
        S, rot = G // 128, self.rot
        with ctx.stream.segment(lazy=False):
            yi = rnd(ctx.stream.dmatmul(IDX_W, scr["xb"], IDX_N))
        cos = scr["qf_rope"][:, : rot // 2]
        sin = scr["qf_rope"][:, rot // 2 : rot]
        iq = ix.index_query(yi, ctx.sm(slot, "qsa_iqn"), cos, sin, rot, g.eps, pltpu.bitcast)
        nreq = MB // T
        for q in range(nreq):  # row-major (row, head) query tile per request, static stores
            for t in range(T):
                for h in range(ix.IDX_HEADS):
                    i = t * ix.IDX_HEADS + h
                    scr["qi_q"][q, i : i + 1, :] = iq[h][q * T + t : q * T + t + 1]
        for q in range(nreq):
            cp = pltpu.make_async_copy(ctx.extra[CK].at[q], scr["qi_ck"], scr["qi_sem"].at[0])
            cp.start()
            cp.wait()
            st = ix.score_rows(scr["qi_q"][q].astype(BF16), scr["qi_ck"][...])  # [T, G]
            scr["qi_s"][q * T : (q + 1) * T] = st.reshape(T, S, 128)
        rid = jax.lax.broadcasted_iota(I32, (MB, 1, 1), 0)
        nv = jnp.zeros((MB, 1, 1), I32)
        for r in range(MB):
            nv = jnp.where(rid == r, ctx.extra[NVIS][r], nv)
        ids = (
            jax.lax.broadcasted_iota(I32, (MB, S, 128), 1) * 128
            + jax.lax.broadcasted_iota(I32, (MB, S, 128), 2)
        )
        bits = jnp.where(ids < nv, pltpu.bitcast(scr["qi_s"][...], I32), -1)
        thr, tmax = ix.thresholds(bits, ids, nv, K, G)
        scr["qi_th"][...] = jnp.broadcast_to(thr, (MB, 1, 128))
        scr["qi_tm"][...] = jnp.broadcast_to(tmax, (MB, 1, 128))
        scr["qi_nv"][...] = jnp.broadcast_to(nv, (MB, 1, 128))
        consts = ix.compact_consts(S, K)
        lid = consts[-1]
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)

        def row(r: Any, carry: Any) -> Any:
            nvr = scr["qi_nv"][r][:, :1]
            b = jnp.where(lid < nvr, pltpu.bitcast(scr["qi_s"][r], I32), -1)
            scr["qi_id"][r] = ix.compact_row(
                b, scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1], nvr, K, consts
            )
            # Columns past K are read by the last tile's copies: keep them at group 0.
            cs = [
                pltpu.make_async_copy(
                    scr["qi_id"].at[r], scr["qi_ids"].at[pl.ds(r, 1), pl.ds(0, K)],
                    scr["qi_sem"].at[1],
                ),
                pltpu.make_async_copy(
                    scr["qi_z"], scr["qi_ids"].at[pl.ds(r, 1), pl.ds(K, 128)], scr["qi_sem"].at[2]
                ),
            ]
            for c in cs:
                c.start()
            for c in cs:
                c.wait()
            n, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
            cnt = jnp.minimum(n, K)

            @pl.when(tail > 0)
            def _():
                scr["qi_ids"][r, cnt] = n  # the partial group follows the selection

            scr["qi_kvl"][r] = 4 * cnt + tail
            return carry

        jax.lax.fori_loop(0, MB, row, 0)
        if self.debug:
            for h in range(ix.IDX_HEADS):
                scr["qi_dbg"][:, h * ix.IDX_DIM : (h + 1) * ix.IDX_DIM] = iq[h]
            o = ctx.stream.sp.block_ord(l, self.name)
            for src, dst in ((scr["qi_dbg"], DBG_IQ), (scr["qi_id"], DBG_IDS)):
                c = pltpu.make_async_copy(src, ctx.O[dst].at[o, pl.ds(0, MB)], scr["qi_sem"].at[0])
                c.start()
                c.wait()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh = self.nh(g, ex)
        kv = self._project(ctx, l, slot)
        self._select(ctx, l, slot)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            out, g0, pre = attend_row_mapped(
                r,
                g0,
                pre,
                nrows=ex.MB,
                pool=ctx.extra[POOL],
                ring=scr["qf_ring"],
                sems=scr["qf_ring_sem"],
                pages=scr["qi_ids"],
                kv_len=scr["qi_kvl"],
                table=ctx.extra[MTAB],
                rows_per_req=ex.T,
                q=scr["qf_q"][r],
                gate=scr["qf_g"][r],
                off=ctx.extra[OFFSET][0],
                scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1],
                sm_scale=g.HD**-0.5,
                interpret=ex.interpret,
            )
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        *outs, _, _ = jax.lax.fori_loop(0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0)))
        for h, o in enumerate(outs):
            scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
        kv.wait()
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)

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
"""Stage C: the step's own tokens, with no in-place cache write inside the kernel.

The stock layer writes this step's K/V (FP8), raw ring rows and any completed group's
compressed key BEFORE scoring and attention. In-kernel writes would need the pool aliased
inside a communicating call (refused by the core, and the pattern behind the fused-KV-writer
hang), so the kernel instead keeps the step's tokens in VMEM and emits the cache rows as
outputs for a post-kernel scatter:

* compressed key of the group completed this step (`g_c`, one per request when T = 4):
  members = this step's raw index-key rows or the request's raw ring, bf16 mean, gemma norm,
  index RoPE at the group's first position; scored in place of the (stale) cached column;
* attention: the pool path reads only selected groups below `g_a = ctx // 4` (wholly older
  than the step); the <= 2 groups holding the step's tokens form a per-request "recent tile"
  of 8 slots (old members from the pool, the step's tokens FP8-quantised like the stock
  write), masked per row: causal, the row's partial group, and `g_a` only if complete for
  the row and selected;
* outputs: `qsa_kv_new` (stage A), `qsa_ck_new [requests, 128]`, `qsa_ring_new [requests, 8, 128]`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, _unpack_kv, attend_row
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rnd, rmsnorm
from ..spec import ExtraInput, ScratchReq, SmallField, StateDecl
from . import indexer as ix
from . import union_attn as ua
from .qsa_full import OFFSET, POOL, ROPE, SCALES
from .qsa_full_b import CK, IDX_N, IDX_W, NVIS, TAIL, QSAFullIndexedBlock

CTXQ = "qsa_ctx"  # [requests] tokens before this step
RING = "qsa_ring"  # [requests, 8, 128] raw index-key ring (lane = position % 8)
ROPE_CK = "qsa_rope_ck"  # [requests, 128] RoPE row at the completed group's first position
CK_NEW = "qsa_ck_new"
RING_NEW = "qsa_ring_new"
# Projections the post-kernel scatter feeds to the stock cache writers
# (`pooled_qsa.qsa_cache_writes`): the un-normed key and the raw index key per row.
KR_NEW = "qsa_kr_new"
IR_NEW = "qsa_ir_new"
FP8_MAX = 448.0


def quant_fp8(x: jax.Array, scale: Any) -> jax.Array:
    """Stock cache write: `clip(x / scale, +-448) -> e4m3`, read back as f32 `* scale`."""
    return pool_units(x, scale) * scale


def pool_units(x: jax.Array, scale: Any) -> jax.Array:
    """The FP8 value the stock write would store (`clip(x / scale) -> e4m3`), as f32: the units
    `attend_row` reads pool tiles in (it applies the scales itself)."""
    return jnp.clip(x / scale, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn).astype(F32)


def pick_row(rows: jax.Array, idx: Any) -> jax.Array:
    """`rows[idx]` for a traced `idx` by a masked sum (exact: one nonzero term)."""
    sel = jax.lax.broadcasted_iota(I32, (rows.shape[0], 1), 0) == idx
    return jnp.sum(jnp.where(sel, rows, 0.0), axis=0, keepdims=True)


class QSAFullStepBlock(QSAFullIndexedBlock):
    """Stage B + the step's own tokens (compressed key, recent tile, cache-row outputs)."""

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return super().smalls(g, ex) + [SmallField("qsa_ikn", ix.IDX_DIM)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        nreq = ex.M // ex.T
        return super().outputs(g, ex) + [
            StateDecl(CK_NEW, (nreq, 1, ix.IDX_DIM), BF16, "w"),
            StateDecl(RING_NEW, (nreq, 8, ix.IDX_DIM), BF16, "w"),
            StateDecl(KR_NEW, (ex.M, g.HD), BF16, "w"),
            StateDecl(IR_NEW, (ex.M, ix.IDX_DIM), BF16, "w"),
        ]

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        out.update({k: raw_outputs[k] for k in (CK_NEW, RING_NEW, KR_NEW, IR_NEW)})
        return out

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nreq = ex.M // ex.T
        return super().extra_inputs(g, ex) + [
            ExtraInput(CTXQ, "smem", (nreq,), I32),
            ExtraInput(RING, "hbm", (self.meta_rows(ex), 8, ix.IDX_DIM), BF16),  # per layer
            ExtraInput(ROPE_CK, "hbm", (nreq, 1, 128), F32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        nreq, f8 = ex.MB // ex.T, jnp.float8_e4m3fn
        return super().scratch(g, ex, plan) + [
            ScratchReq("qc_ring", pltpu.VMEM((8, ix.IDX_DIM), BF16)),
            ScratchReq("qc_rope", pltpu.VMEM((1, 128), F32)),
            ScratchReq("qc_ck", pltpu.VMEM((1, ix.IDX_DIM), BF16)),
            ScratchReq("qc_rnew", pltpu.VMEM((8, ix.IDX_DIM), BF16)),
            ScratchReq("qc_old", pltpu.VMEM((MICRO, 1, 4, 256), f8)),
            ScratchReq("qc_rk", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_rv", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_sga", pltpu.SMEM((ex.MB,), I32)),
            ScratchReq("qc_sem", pltpu.SemaphoreType.DMA((3,))),
            ScratchReq("qf_kr", pltpu.VMEM((ex.MB, g.HD), BF16)),
            ScratchReq("qc_ir", pltpu.VMEM((ex.MB, ix.IDX_DIM), BF16)),
            ScratchReq("qc_osem", pltpu.SemaphoreType.DMA((2,))),
        ] + (self._shard_scratch(ex) if self.shard_index else []) + (
            self._union_scratch(ex) if self.union_attn else []
        )

    def _union_width(self, ex: ExecSpec) -> int:
        """Union slots per request: every row's whole selection, 128-lane aligned."""
        return -(-(ex.T * self.budget) // 128) * 128

    def _union_scratch(self, ex: ExecSpec) -> list[ScratchReq]:
        nreq, KU, S = ex.MB // ex.T, self._union_width(ex), self.max_groups // 128
        out = [
            ScratchReq("qu_code", pltpu.VMEM((S, 128), I32)),
            ScratchReq("qu_id", pltpu.VMEM((nreq, 1, KU), I32)),
            ScratchReq("qu_cd", pltpu.VMEM((nreq, 1, KU), I32)),
            ScratchReq("qu_n", pltpu.SMEM((nreq,), I32)),
            ScratchReq("qu_row", pltpu.SMEM((1, KU + 128), I32)),
            ScratchReq("qu_sem", pltpu.SemaphoreType.DMA((2,))),
        ]
        if self.shard_index:
            npr, n = self._shard_rows(ex)[0], max(ex.R - 1, 1)
            out += [
                ScratchReq("qgu_send", pltpu.VMEM((npr, 1, 2 * KU + 128), I32)),
                ScratchReq("qgu_recv", pltpu.VMEM((ex.R, npr, 1, 2 * KU + 128), I32)),
                ScratchReq("qgu_ssem", pltpu.SemaphoreType.DMA((n,))),
                ScratchReq("qgu_rsem", pltpu.SemaphoreType.DMA((n,))),
            ]
        return out

    def _shard_rows(self, ex: ExecSpec) -> tuple[int, int, int]:
        """(requests per rank, rows per rank, exchange row width) of the sharded indexer."""
        nreq = ex.MB // ex.T
        npr = -(-nreq // ex.R)
        return npr, npr * ex.T, -(-(self.budget + 2) // 128) * 128

    def _shard_scratch(self, ex: ExecSpec) -> list[ScratchReq]:
        _, rows, W = self._shard_rows(ex)
        n = max(ex.R - 1, 1)
        return [
            ScratchReq("qg_send", pltpu.VMEM((rows, 1, W), I32)),
            ScratchReq("qg_recv", pltpu.VMEM((ex.R, rows, 1, W), I32)),
            ScratchReq("qg_ssem", pltpu.SemaphoreType.DMA((n,))),
            ScratchReq("qg_rsem", pltpu.SemaphoreType.DMA((n,))),
        ]

    # ------------------------------------------------------------ in kernel
    def _new_group(self, ctx: Any, l: Any, q: int, raw_q: jax.Array) -> jax.Array:
        """Compressed key of request `q`'s group completed this step; writes its outputs."""
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        for src, dst, i in ((ctx.extra[RING].at[self.row_base(ctx, l) + q], scr["qc_ring"], 0),
                            (ctx.extra[ROPE_CK].at[q], scr["qc_rope"], 1)):
            cp = pltpu.make_async_copy(src, dst, scr["qc_sem"].at[i])
            cp.start()
            cp.wait()
        ring = scr["qc_ring"][...].astype(F32)
        p3 = c0 + jax.lax.rem(3 - jax.lax.rem(c0, 4) + 4, 4)  # first position = 3 mod 4
        j0 = p3 - 3
        acc = None
        for j in range(4):  # sequential f32 sum: one fixed order, mirrored by the reference
            pos = j0 + j
            cur = pick_row(raw_q, pos - c0)
            old = pick_row(ring, jax.lax.rem(pos, 8))
            m = jnp.where(pos >= c0, cur, old)
            acc = m if acc is None else acc + m
        pooled = rnd(acc * 0.25)
        rope = scr["qc_rope"][...]
        cos, sin = rope[:, : self.rot // 2], rope[:, self.rot // 2 : self.rot]
        ck = ix.index_rope(
            rmsnorm(pooled, ctx.sm(ctx._slot, "qsa_ikn"), g.eps, zero_centered=True),
            cos, sin, self.rot, pltpu.bitcast,
        )
        scr["qc_ck"][...] = ck.astype(BF16)
        # Ring: lane l keeps the newest position = l (mod 8) below end = ctx + T.
        end = c0 + T
        lane = jax.lax.broadcasted_iota(I32, (8, 1), 0)
        p_l = end - 8 + jax.lax.rem(lane - jax.lax.rem(end - 8, 8) + 16, 8)
        cur_rows = jnp.zeros((8, ix.IDX_DIM), F32)
        for t in range(T):
            cur_rows = jnp.where(p_l == c0 + t, raw_q[t : t + 1], cur_rows)
        scr["qc_rnew"][...] = jnp.where(p_l >= c0, cur_rows, ring).astype(BF16)
        o = ctx.stream.sp.block_ord(l, self.name)
        for src, dst in ((scr["qc_ck"], CK_NEW), (scr["qc_rnew"], RING_NEW)):
            cp = pltpu.make_async_copy(src, ctx.O[dst].at[o, q], scr["qc_sem"].at[2])
            cp.start()
            cp.wait()
        return ck, p3 // 4

    def _recent_tile(self, ctx: Any, l: Any, q: int) -> None:
        """K / V of the 8 positions from `4 (ctx // 4)` in pool units (FP8 values, unscaled, as
        `attend_row`'s `recent` expects): old members from the pool, the step's tokens
        FP8-quantised from `qf_kv` (stock write-then-read)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        per_page = ctx.extra[POOL].shape[1] // MICRO
        shift = per_page.bit_length() - 1
        qb = self.tab_base(ctx, l) + q
        micro = self.tables(ctx)[1][qb, ga >> shift] * per_page + (ga & (per_page - 1))
        cp = pltpu.make_async_copy(
            ctx.extra[POOL].at[micro >> shift, pl.ds((micro & (per_page - 1)) * MICRO, MICRO)],
            scr["qc_old"],
            scr["qc_sem"].at[0],
        )
        cp.start()
        cp.wait()
        sk, sv = ctx.extra[SCALES][0], ctx.extra[SCALES][1]
        k8, v8 = _unpack_kv(scr["qc_old"], self.offset(ctx, l), ex.interpret)
        pad = jnp.zeros((4, 256), F32)
        k_old = jnp.concatenate([k8.astype(F32), pad], axis=0)
        v_old = jnp.concatenate([v8.astype(F32), pad], axis=0)
        kv = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)
        k_cur, v_cur = pool_units(kv[:, :256], sk), pool_units(kv[:, 256:], sv)
        pos = 4 * ga + jax.lax.broadcasted_iota(I32, (8, 1), 0)
        k_t = jnp.where(pos < c0, k_old, 0.0)
        v_t = jnp.where(pos < c0, v_old, 0.0)
        for t in range(T):
            k_t = jnp.where(pos == c0 + t, k_cur[t : t + 1], k_t)
            v_t = jnp.where(pos == c0 + t, v_cur[t : t + 1], v_t)
        scr["qc_rk"][q] = k_t
        scr["qc_rv"][q] = v_t

    def _select(self, ctx: Any, l: Any, slot: Any) -> None:
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        MB, T, K, G = ex.MB, ex.T, self.budget, self.max_groups
        S, rot = G // 128, self.rot
        ctx._slot = slot
        with ctx.stream.segment(lazy=False):
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

        def select_rows(lo: Any, n: int, s0: Any) -> None:
            """Thresholds and compaction of rows `[lo, lo + n)`; scores at `qi_s[s0 …]`."""
            rid = jax.lax.broadcasted_iota(I32, (n, 1, 1), 0) + lo
            nv = jnp.zeros((n, 1, 1), I32)
            for r in range(lo, lo + n):
                nv = jnp.where(rid == r, ctx.extra[NVIS][r], nv)
            ids = (
                jax.lax.broadcasted_iota(I32, (n, S, 128), 1) * 128
                + jax.lax.broadcasted_iota(I32, (n, S, 128), 2)
            )
            sc = pltpu.bitcast(scr["qi_s"][pl.ds(s0, n)].reshape(n, S, 128), I32)
            bits = jnp.where(ids < nv, sc, -1)
            thr, tmax = ix.thresholds(bits, ids, nv, K, G)
            scr["qi_th"][pl.ds(lo, n)] = jnp.broadcast_to(thr, (n, 1, 128))
            scr["qi_tm"][pl.ds(lo, n)] = jnp.broadcast_to(tmax, (n, 1, 128))
            scr["qi_nv"][pl.ds(lo, n)] = jnp.broadcast_to(nv, (n, 1, 128))
            consts = ix.compact_consts(S, K)
            lid = consts[-1]
            scr["qi_z"][...] = jnp.zeros((1, 128), I32)
            if self.union_attn:
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
                bp = jnp.where(lid < ga, b, -1)
                # Selection of the pool path: the same set restricted to ids < g_a. With `n_vis`
                # clamped to g_a the compaction sees exactly those ids (select-all of the kept).
                kept = jnp.sum((sel & (lid < ga)).astype(I32))
                b2 = jnp.where(sel & (lid < ga), bp, -1)
                if self.union_attn:  # row bit r - lo of every pool-path group it selects
                    scr["qu_code"][...] = scr["qu_code"][...] | jnp.where(
                        sel & (lid < ga), jnp.left_shift(1, r - lo), 0
                    )
                scr["qi_id"][r] = ix.compact_row(
                    b2, jnp.zeros((1, 1), I32), jnp.zeros((1, 1), I32) + G,
                    jnp.zeros((1, 1), I32) + kept, K, consts,
                )
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
                scr["qi_kvl"][r] = 4 * kept
                return carry

            jax.lax.fori_loop(lo, lo + n, row, 0)
            if self.union_attn:  # per-request call: rows [lo, lo + T) are request lo // T
                KU = self._union_width(ex)
                ids, codes, nu = ua.compact_union(
                    scr["qu_code"][...], KU, ua.union_consts(S, KU)
                )
                scr["qu_id"][lo // T] = ids
                scr["qu_cd"][lo // T] = codes
                scr["qu_n"][lo // T] = jnp.sum(nu)
        self.load_tables(ctx, l)
        if self.shard_index:
            # Each rank scores and selects its own run of `npr` requests; every rank still
            # builds every request's completed group (outputs) and recent tile (its KV head).
            npr = self._shard_rows(ex)[0]
            for q in range(nreq):
                ck, gc = self._new_group(ctx, l, q, raw[q * T : (q + 1) * T])

                @pl.when(q // npr == ctx.me)
                def _(q=q, ck=ck, gc=gc):
                    qr = scr["qi_q"][q].astype(BF16)
                    self._score_request(
                        ctx, q, qr, self.tab_base(ctx, l), row0=0,
                        first=q % npr == 0, last=q % npr == npr - 1 or q == nreq - 1,
                    )
                    s_new = ix.score_rows(qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM)))
                    scr["qi_s"][0:T] = jnp.where(
                        gid == gc, s_new[:, :1].reshape(T, 1, 1, 1), scr["qi_s"][0:T]
                    )
                    select_rows(q * T, T, 0)

                self._recent_tile(ctx, l, q)
            self._exchange_selection(ctx)
        for q in range(0 if self.shard_index else nreq):
            qr = scr["qi_q"][q].astype(BF16)
            s0 = 0 if self.per_request_scores else q * T
            self._score_request(ctx, q, qr, self.tab_base(ctx, l), last=q == nreq - 1, row0=s0)
            ck, gc = self._new_group(ctx, l, q, raw[q * T : (q + 1) * T])
            s_new = ix.score_rows(qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM)))[:, :1]
            rows = slice(s0, s0 + T)
            # The completed group's fresh key replaces its stale cached column.
            scr["qi_s"][rows] = jnp.where(
                gid == gc, s_new.reshape(T, 1, 1, 1), scr["qi_s"][rows]
            )
            self._recent_tile(ctx, l, q)
            if self.per_request_scores:  # select this request's rows before the next scores
                select_rows(q * T, T, 0)
        if not self.per_request_scores:
            select_rows(0, MB, 0)
        for c in outs:
            c.wait()
        if self.debug:  # the kernel's queries and pool-path ids; g_a's flag in id slot K-1
            from .qsa_full_b import DBG_IDS, DBG_IQ

            for h in range(ix.IDX_HEADS):
                scr["qi_dbg"][:, h * ix.IDX_DIM : (h + 1) * ix.IDX_DIM] = iq[h]
            o = ctx.stream.sp.block_ord(l, self.name)
            for src, dst in ((scr["qi_dbg"], DBG_IQ), (scr["qi_id"], DBG_IDS)):
                c = pltpu.make_async_copy(src, ctx.O[dst].at[o, pl.ds(0, MB)], scr["qi_sem"].at[0])
                c.start()
                c.wait()

    def _exchange_selection(self, ctx: Any) -> None:
        """All-gather the sharded indexer's per-row results: pool-path ids (`qi_id` and its
        SMEM copy `qi_ids`), `qi_kvl` and the `g_a` flag `qc_sga`. Each rank sends its run of
        rows (lanes `[0, K)` ids, `K` length, `K + 1` flag) to every peer's slot `me`; int32
        throughout, so the gathered selection is the owner's exactly."""
        ex, scr, K, T, MB = ctx.ex, ctx.scr, self.budget, ctx.ex.T, ctx.ex.MB
        npr, rows, W = self._shard_rows(ex)
        me = ctx.me
        lane = jax.lax.broadcasted_iota(I32, (1, W), 1)
        lo = me * rows
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)  # set by select_rows only on owners

        def pack(i: Any, carry: Any) -> Any:
            r = jnp.minimum(lo + i, MB - 1)  # past the last request: never read
            scr["qg_send"][i, :, pl.ds(0, K)] = scr["qi_id"][r]
            head = jnp.where(lane == K, scr["qi_kvl"][r], 0)
            scr["qg_send"][i] = jnp.where(
                lane < K, scr["qg_send"][i], jnp.where(lane == K + 1, scr["qc_sga"][r], head)
            )
            return carry

        jax.lax.fori_loop(0, rows, pack, 0)
        scr["qg_recv"][me] = scr["qg_send"][...]
        if ex.R > 1:
            ctx.comm.exchange(
                ctx.comm.group_copies(
                    lambda off: scr["qg_send"], scr["qg_recv"].at[me],
                    scr["qg_ssem"], scr["qg_rsem"], ex.R,
                )
            )

        def unpack(r: Any, carry: Any) -> Any:
            o = r // T // npr
            row = scr["qg_recv"][o, r - o * rows]
            scr["qi_id"][r] = row[:, :K]
            scr["qi_kvl"][r] = jnp.sum(jnp.where(lane == K, row, 0))
            scr["qc_sga"][r] = jnp.sum(jnp.where(lane == K + 1, row, 0))
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
            return carry

        jax.lax.fori_loop(0, MB, unpack, 0)
        if self.union_attn:
            self._exchange_unions(ctx)

    def _exchange_unions(self, ctx: Any) -> None:
        """All-gather the owners' per-request unions (ids, row codes, count; int32)."""
        ex, scr = ctx.ex, ctx.scr
        nreq, KU = ex.MB // ex.T, self._union_width(ex)
        npr = self._shard_rows(ex)[0]
        me = ctx.me
        lane = jax.lax.broadcasted_iota(I32, (1, 2 * KU + 128), 1)

        def pack(i: Any, carry: Any) -> Any:
            q = jnp.minimum(me * npr + i, nreq - 1)  # past the last request: never read
            scr["qgu_send"][i, :, pl.ds(0, KU)] = scr["qu_id"][q]
            scr["qgu_send"][i, :, pl.ds(KU, KU)] = scr["qu_cd"][q]
            scr["qgu_send"][i, :, pl.ds(2 * KU, 128)] = jnp.zeros((1, 128), I32) + scr["qu_n"][q]
            return carry

        jax.lax.fori_loop(0, npr, pack, 0)
        scr["qgu_recv"][me] = scr["qgu_send"][...]
        if ex.R > 1:
            ctx.comm.exchange(
                ctx.comm.group_copies(
                    lambda off: scr["qgu_send"], scr["qgu_recv"].at[me],
                    scr["qgu_ssem"], scr["qgu_rsem"], ex.R,
                )
            )

        def unpack(q: Any, carry: Any) -> Any:
            o = q // npr
            row = scr["qgu_recv"][o, q - o * npr]
            scr["qu_id"][q] = row[:, :KU]
            scr["qu_cd"][q] = row[:, KU : 2 * KU]
            scr["qu_n"][q] = jnp.sum(jnp.where(lane == 2 * KU, row, 0))
            return carry

        jax.lax.fori_loop(0, nreq, unpack, 0)

    def _recent_ok(self, ctx: Any, r: Any, s: Any) -> Any:
        """Row `r`'s valid recent-tile slots `s`: causal, its partial group, and `g_a` when
        complete for the row and selected (the per-row attention's `ok`)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        q = r // T
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        p_r = c0 + r - q * T
        n_r, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
        pos = 4 * ga + s
        grp = pos // 4
        part = (grp == n_r) & (tail > 0)
        full = (grp == ga) & (ga < n_r) & (scr["qc_sga"][r] > 0)
        return (r < ex.B) & (pos <= p_r) & (part | full)

    def _attend_unions(self, ctx: Any, l: Any, zeros: tuple) -> list[jax.Array]:
        """Every request's rows over the union of their selections (`union_attn`)."""
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        nh, KU, nreq = self.nh(g, ex), self._union_width(ex), ex.MB // ex.T
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)

        def req(q: Any, outs: Any) -> Any:
            cs = [
                pltpu.make_async_copy(
                    scr["qu_id"].at[q], scr["qu_row"].at[pl.ds(0, 1), pl.ds(0, KU)],
                    scr["qu_sem"].at[0],
                ),
                pltpu.make_async_copy(
                    scr["qi_z"], scr["qu_row"].at[pl.ds(0, 1), pl.ds(KU, 128)], scr["qu_sem"].at[1]
                ),
            ]
            for c in cs:
                c.start()
            for c in cs:
                c.wait()
            rs = [q * T + t for t in range(T)]
            ok = jnp.concatenate(
                [jnp.broadcast_to(self._recent_ok(ctx, r, srow), (nh, 8)) for r in rs], axis=0
            )
            ok_any = self._recent_ok(ctx, rs[0], scol)
            for r in rs[1:]:
                ok_any = ok_any | self._recent_ok(ctx, r, scol)
            res = ua.attend_request(
                T=T, nh=nh, pool=ctx.extra[POOL], ring=scr["qf_ring"], sems=scr["qf_ring_sem"],
                ids=scr["qu_row"], codes=scr["qu_cd"].at[q], n_union=scr["qu_n"][q],
                table=self.tables(ctx)[1], table_row=self.tab_base(ctx, l) + q,
                q=jnp.concatenate([scr["qf_q"][r] for r in rs], axis=0),
                gates=[scr["qf_g"][r] for r in rs],
                has=[scr["qi_kvl"][r] > 0 for r in rs],
                off=self.offset(ctx, l), scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1], sm_scale=g.HD**-0.5, interpret=ex.interpret,
                recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok, ok_any),
            )
            outs = list(outs)
            for t, r in enumerate(rs):
                for h in range(nh):
                    outs[h] = jnp.where(rows == r, res[t][h : h + 1], outs[h])
            return tuple(outs)

        return list(jax.lax.fori_loop(0, nreq, req, zeros))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh, T = self.nh(g, ex), ex.T
        kv = self._project(ctx, l, slot)
        self._select(ctx, l, slot)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            q = r // T
            c0 = ctx.extra[CTXQ][q]
            ga = c0 // 4
            p_r = c0 + r - q * T
            n_r, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
            live = r < ex.B

            def ok(s: Any) -> Any:
                pos = 4 * ga + s
                grp = pos // 4
                part = (grp == n_r) & (tail > 0)
                full = (grp == ga) & (ga < n_r) & (scr["qc_sga"][r] > 0)
                return live & (pos <= p_r) & (part | full)

            out, g0, pre = attend_row(
                r, g0, pre, nrows=ex.MB, pool=ctx.extra[POOL], ring=scr["qf_ring"],
                sems=scr["qf_ring_sem"], pages=scr["qi_ids"], kv_len=scr["qi_kvl"],
                table=self.tables(ctx)[1], rows_per_req=T, table_row_base=self.tab_base(ctx, l),
                q=scr["qf_q"][r], gate=scr["qf_g"][r],
                off=self.offset(ctx, l), scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1], sm_scale=g.HD**-0.5, interpret=ex.interpret,
                recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok(srow), ok(scol)),
            )
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        if self.union_attn:
            outs = self._attend_unions(ctx, l, zeros)
        else:
            *outs, _, _ = jax.lax.fori_loop(0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0)))
        for h, o in enumerate(outs):
            scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
        kv.wait()
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)

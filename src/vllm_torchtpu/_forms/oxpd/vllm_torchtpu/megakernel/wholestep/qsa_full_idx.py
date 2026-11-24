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
"""q38 L1-IDX: the in-kernel indexer package on the whole step's union path (D14, on top of D08 S1).

Kernel options of the indexed block (envs in `qwen38_wholestep_call`, all default off; GENERATED file: the selection
`_select` is a copy of `qsa_full_lean._LeanMixin._select` made by scratch/decL1/idx/gen_qsa_full_idx.py):

* `radix_bits` = 2 (VLLM_TPU_MK_SELECT_RADIX, S2P's second half): the selection's thresholds decide two bits per
  counting pass (`thresholds_radix2`): 15 + 1 value passes instead of 31 and ceil(nb / 2) tie passes instead of nb,
  each with three candidate counts. Both searches pick the largest candidate whose count condition holds and the
  counts are monotone in the candidate, so `(thr, tmax)` are identical to `indexer.thresholds` (PD W's
  `thresholds_live` radix, here on MM's whole-tile scores).
* `union_qsplit` = 1 / 2 (VLLM_TPU_MK_UNION_QSPLIT): under main-KV DCP D (2 or 4) with 512-token pool blocks (128
  groups per block = one sublane row of the `[S, 128]` code array), group g's DCP owner is `(g // 128) % D`, i.e. its
  row `s` mod D. Each DCP rank's union is compacted from its own `S / D` rows (rows r, r + D, ...) instead of a masked
  copy of all S rows: 1 = a strided sublane load of `qu_code`, 2 = a one-hot row-gather matmul (codes < 2^T are exact
  in BF16); the compaction is MM's arithmetic on `[S / D, 128]` with `union_consts(S / D, KU)`, and the output ids
  are remapped to `(local row * D + r) * 128 + lane` inside the pass (slots past the count stay 0). The j-th
  selected group in (row, lane) order is the same group in both forms, so ids, codes and counts are bitwise MM's.
* `table_prefetch` (VLLM_TPU_MK_TABLE_SLIM = 2, TSLIM's prefetch; the column slimming is `pages_per_seq` itself):
  the per-layer SMEM table windows are double-buffered (`[2, requests, cols]`, slot = QSA layer ordinal % 2, selected
  through `tables`) and QSA layer o starts layer o + 1's table copies right after its own are waited, so layer
  o + 1 only waits (layer 0 starts and waits its own). Same table values, so same outputs.

With none of these options present `wholestep_spec` builds the D08 S1 lean block (or MM's / D14's block) exactly as
before: `lean_step_block` here only routes to the classes below when an L1-IDX key is in the options.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rnd
from ..spec import ScratchReq
from . import indexer as ix
from . import union_attn as ua
from .qsa_full import POOL
from .qsa_full_b import CKTAB, GPK, IDX_N, IDX_W, MTAB, NVIS
from .qsa_full_c import CTXQ, IR_NEW, KR_NEW
from .qsa_full_lean import (
    QSAFullStepGroupLeanBlock,
    QSAFullStepLeanBlock,
    _union_prologue,
)
from .qsa_full_lean import lean_step_block as _lean_step_block

IDX_KEYS = ("radix_bits", "union_qsplit", "table_prefetch")


def thresholds_radix2(bits: jax.Array, ids: jax.Array, nv: jax.Array, K: int, G: int):
    """`indexer.thresholds` deciding two bits per counting pass: identical per-row `(thr, tmax)` [R, 1, 1].

    Value search: bits (30, 29), (28, 27), ..., (2, 1) in 15 passes, then bit 0. In a pass with low bit b the
    candidates `lo | 1 << b < lo | 2 << b < lo | 3 << b` are counted at once and the largest one with
    `count(bits >= c) >= K` is kept (else `lo`); since that count does not grow with c, this is exactly what the two
    one-bit steps (bit b + 1, then bit b) keep. Tie search: bits (nb - 1, nb - 2), ... and the last bit alone when nb is
    odd, keeping the largest candidate with `count(eq & ids < c) < need` (that count does not shrink with c)."""
    cnt = lambda m: jnp.sum(m.astype(I32), axis=(1, 2), keepdims=True)  # noqa: E731
    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    for b in range(29, 0, -2):
        c1, c2, c3 = lo | (1 << b), lo | (2 << b), lo | (3 << b)
        n1, n2, n3 = cnt(bits >= c1), cnt(bits >= c2), cnt(bits >= c3)
        lo = jnp.where(n3 >= K, c3, jnp.where(n2 >= K, c2, jnp.where(n1 >= K, c1, lo)))
    cand = lo | 1
    lo = jnp.where(cnt(bits >= cand) >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    eq = bits == thr
    need = K - cnt(bits > thr)
    nb = int(G - 1).bit_length()
    t = jnp.zeros(nv.shape, I32)
    b = nb - 2
    while b >= 0:
        c1, c2, c3 = t | (1 << b), t | (2 << b), t | (3 << b)
        n1, n2, n3 = cnt(eq & (ids < c1)), cnt(eq & (ids < c2)), cnt(eq & (ids < c3))
        t = jnp.where(n3 < need, c3, jnp.where(n2 < need, c2, jnp.where(n1 < need, c1, t)))
        b -= 2
    if nb % 2:
        cand = t | 1
        t = jnp.where(cnt(eq & (ids < cand)) < need, cand, t)
    tmax = jnp.where(over & (cnt(eq) > need), t, G)
    return thr, tmax


def _union_pass_rows(pro: tuple, consts: tuple, j0: int, stride: int, off: int) -> tuple[jax.Array, jax.Array]:
    """`qsa_full_lean._union_pass` (MM's pass at output slots `[j0, j0 + W)`) on a row subset: local row `r_j` of the
    `[S / stride, 128]` code is row `r_j * stride + off` of the full array, so the ids are remapped before the live
    mask (slots past the count stay 0). Everything else is the lean pass's arithmetic."""
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
    ids = jnp.where(live, (r_j * stride + off) * 128 + lane_j, 0).astype(F32)
    codes = jnp.where(live, code_j, 0.0)
    return (
        jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32),
        jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32),
    )


def compact_union_rows(code: jax.Array, KU: int, consts: tuple, stride: int, off: int) -> tuple:
    """`union_attn.compact_union` of a row subset `code [S / stride, 128]` (rows off, off + stride, ... of the full
    array), ids in full-array numbering: `(ids [1, KU], codes [1, KU], n [1, 1])`."""
    pro = _union_prologue(code, consts)
    W = consts[2].shape[0]
    outs = [_union_pass_rows(pro, consts, j0, stride, off) for j0 in range(0, KU, W)]
    cat = lambda xs: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=1)  # noqa: E731
    return cat([o[0] for o in outs]), cat([o[1] for o in outs]), pro[-1]


def quarter_rows(code_all: jax.Array, D: int, r: int) -> jax.Array:
    """Rows `r, r + D, ...` of `code_all [S, 128]` (int32 codes < 2^T) by a one-hot row-gather matmul (QSPLIT mode 2):
    each output row has exactly one 1.0 weight, the codes are exact in BF16 and the F32 accumulate is exact."""
    S = code_all.shape[0]
    sel = (jax.lax.broadcasted_iota(I32, (S // D, S), 1) == jax.lax.broadcasted_iota(I32, (S // D, S), 0) * D + r)
    g = jnp.dot(sel.astype(F32).astype(BF16), code_all.astype(F32).astype(BF16), preferred_element_type=F32)
    return g.astype(I32)


class _IdxMixin:
    """L1-IDX options (module docstring) on the D08 S1 lean block; all off = the lean block's selection."""

    def __init__(self, *, radix_bits: int = 1, union_qsplit: int = 0, table_prefetch: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.radix_bits, self.union_qsplit, self.table_prefetch = int(radix_bits), int(union_qsplit), bool(table_prefetch)
        assert self.radix_bits in (1, 2), radix_bits
        assert self.union_qsplit in (0, 1, 2), union_qsplit
        assert not self.table_prefetch or (self.hbm_tables and self.n_layers > 0), "prefetch: per-layer HBM tables"

    # ------------------------------------------------------------ S2P radix-2
    def _idx_thresholds(self, bits: jax.Array, ids: jax.Array, nv: jax.Array, K: int, G: int):
        if self.radix_bits == 2:
            return thresholds_radix2(bits, ids, nv, K, G)
        return ix.thresholds(bits, ids, nv, K, G)

    # ------------------------------------------------------------ QSPLIT
    def _qsplit_compact(self, ctx: Any, q: Any, KU: int, npr: int) -> None:
        """Every DCP rank's union half of request `q` from its own rows of `qu_code` (512-token pool blocks only)."""
        scr = ctx.scr
        D = max(self.dcp, 1)
        S = self.max_groups // 128
        gpp = ctx.extra[POOL].shape[1] // ua.MICRO
        assert gpp == 128 and S % D == 0, ("QSPLIT needs 128 groups per pool block (512 tokens)", gpp, S, D)
        code_all = scr["qu_code"][...] if self.union_qsplit == 2 else None
        for r in range(D):
            if self.union_qsplit == 1:
                code = scr["qu_code"][pl.ds(r, S // D, stride=D), :]
            else:
                code = quarter_rows(code_all, D, r)
            self._compact_rows(ctx, q, r, code, KU, npr, D)

    def _compact_rows(self, ctx: Any, q: Any, r: int, code: jax.Array, KU: int, npr: int, D: int) -> None:
        """`_LeanMixin._compact` for union half `r` from its rows `code [S / D, 128]` (ids remapped in the pass): all
        passes, or (B1) only the passes below the count; into `qu_id` / `qu_cd` or (B2, sharded) `qgl_send[r]`."""
        scr = ctx.scr
        consts = ua.union_consts(code.shape[0], KU)
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
            ids, codes, nu = compact_union_rows(code, KU, consts, D, r)
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
                ids, codes = _union_pass_rows(pro, consts, j0, D, r)
                put(j0, W, ids, codes)

    # ------------------------------------------------------------ TSLIM prefetch
    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        """The prefetch's windows are 3-D `[2 slots, requests, cols]`: a slot is a whole leading index, so neither the
        DMAs nor the reads slice a tiled dimension. (A 2-D `[2 requests, cols]` window failed Mosaic at 12 requests,
        48 rows: l1-idx-aot-fit-i2 j-fe1ccff8 refused the row offset 12 and l1-idx-aot-fit-i3 j-f83ded87 the 12-row
        slice size of the SMEM (8, 128) tile.)"""
        out = super().scratch(g, ex, plan)
        if not self.table_prefetch:
            return out
        nreq = ex.MB // ex.T
        out = [s for s in out if s.name not in ("qi_ckw", "qi_mtw", "qi_tsem")]
        return out + [
            ScratchReq("qi_ckw", pltpu.SMEM((2, nreq, self.max_groups // GPK // (self.mp or 1)), I32)),
            ScratchReq("qi_mtw", pltpu.SMEM((2, nreq, self.pps), I32)),
            ScratchReq("qi_tsem", pltpu.SemaphoreType.DMA((4,))),
        ]

    def tables(self, ctx: Any) -> tuple[Any, Any]:
        """With the prefetch: this QSA layer's slot of the windows (set by `load_tables` at the layer's start; rows are
        requests, `tab_base` stays 0 as for every HBM-table window)."""
        if not self.table_prefetch:
            return super().tables(ctx)
        b = ctx._idx_tslot
        return ctx.scr["qi_ckw"].at[b], ctx.scr["qi_mtw"].at[b]

    def load_tables(self, ctx: Any, l: Any) -> None:  # noqa: E741
        """Prefetch: QSA layer o's window rows are in slot o % 2; layer 0 copies and waits its own, every layer waits
        its own and then starts layer o + 1's copies into the other slot (layer o - 1's, which is done)."""
        if not self.table_prefetch:
            return super().load_tables(ctx, l)
        scr = ctx.scr
        o = ctx.stream.sp.block_ord(l, self.name)
        ctx._idx_tslot = jax.lax.rem(o, 2)

        def copies(oo: Any) -> list[Any]:
            b = jax.lax.rem(oo, 2)
            return [
                pltpu.make_async_copy(ctx.extra[src].at[oo], scr[dst].at[b], scr["qi_tsem"].at[2 * b + i])
                for i, (src, dst) in enumerate(((CKTAB, "qi_ckw"), (MTAB, "qi_mtw")))
            ]

        @pl.when(o == 0)
        def _():
            for c in copies(o):
                c.start()

        for c in copies(o):
            c.wait()

        @pl.when(o + 1 < self.n_layers)
        def _():
            for c in copies(o + 1):
                c.start()

    # ------------------------------------------------------------ selection (generated copy)
    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        """`_LeanMixin._select` (generated copy, scratch/decL1/idx/gen_qsa_full_idx.py) with the L1-IDX options: the
        thresholds through `_idx_thresholds` (radix-2) and, under main-KV DCP with `union_qsplit`, each DCP rank's
        union compacted from its own rows of the code array (`_qsplit_compact`)."""
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
                thr, tmax = self._idx_thresholds(bits, ids, nv, K, G)  # q38 L1-IDX S2P radix-2
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
                if self.union_qsplit and D > 1:  # q38 L1-IDX QSPLIT: each DCP rank's own rows of the code array
                    self._qsplit_compact(ctx, lo // T, KU, npr)
                else:
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


class QSAFullStepIdxBlock(_IdxMixin, QSAFullStepLeanBlock):
    """MM's stage-C block (DCP 1 / 2) with the D08 S1 stages and the L1-IDX options."""


class QSAFullStepGroupIdxBlock(_IdxMixin, QSAFullStepGroupLeanBlock):
    """D14's DCP-4 group block with the D08 S1 stages and the L1-IDX options."""


def lean_step_block(indexed: dict, qsa: dict | None) -> Any:
    """`qsa_full_lean.lean_step_block`, except that an L1-IDX option (any `IDX_KEYS` key present) selects the L1-IDX
    class for the DCP size (the D08 S1 stages default off in it). Without one it is the lean function itself.
    q38 L1-SELLOOP: a `select_owned` option (1 / 2) is not a constructor option; it is taken out and engaged on the
    built block (`_LeanMixin.engage_select_owned`, which checks the request loop / sharded indexer / B3 parts). It
    needs the D08 S1 lean (or L1-IDX) block: without one it is refused.
    q38 L1-V6 composition: `select_owned` is taken out before the L1-QX routing, so a QX block over the lean / IDX
    class (both carry `_LeanMixin`) engages it the same way; QX over MM's / D14's plain block refuses it."""
    owned = int(indexed.get("select_owned", 0) or 0)  # q38 L1-SELLOOP
    indexed = {k: v for k, v in indexed.items() if k != "select_owned"}
    if "qx_early" in indexed:  # q38 L1-QX: X-DCP-QX-EARLY over the class below (end of module)
        blk = qx_step_block(indexed, qsa)
    elif not any(k in indexed for k in IDX_KEYS):
        blk = _lean_step_block(indexed, qsa)
    else:
        cls = QSAFullStepGroupIdxBlock if int(indexed.get("dcp", 1) or 1) > 2 else QSAFullStepIdxBlock
        blk = cls(**indexed, **(qsa or {}))
    if owned:
        assert blk is not None and hasattr(blk, "engage_select_owned"), (
            "select_owned (VLLM_TPU_MK_SELECT_OWNED) needs the D08 S1 lean block (VLLM_TPU_MK_SLOT_BATCH=1)", type(blk))
        blk.engage_select_owned(owned)
    return blk


# q38 L1-QX begin
# X-DCP-QX-EARLY (cycle 6; reports/decode_levers_v6_delta.md): under main-KV DCP the stage-C emit runs `_project`, `_select`, then
# `_dcp_attend`, which begins with a blocking start-then-wait q exchange (`qf_q` to the DCP peers: MM's pair copy at DCP 2, D14's
# group copies at DCP 4). `qf_q` is written only in `_project` (qsa_full.py), so with `qx_early` the SAME copies start right after
# `_project` and are waited (recv, then send, as `exchange` does) at the start of `_dcp_attend`, just before the union attention:
# the transfer hides under the indexer chain. Same descriptors, same values, same partial exchange afterwards: bitwise. Hazards
# (C): the receive buffers `qd_q` / `qdg_q` and the semaphores `qd_[sr]sem0` / `qdg_[sr]sem0` are used by nothing else (the
# scratch overlay merges only same-named requests, vmem._merge_scratch), a peer's last read of its receive buffer (the previous
# QSA layer's attention) precedes its partial exchange and the all-reduces this rank must finish before its next `_project`,
# and the source `qf_q` is not written again before the wait (next QSA layer). Appended at the end of the module (one in-place
# routing line pair in `lean_step_block`), so every traced line above keeps its number.
from .qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock  # noqa: E402
from .qsa_full_lean import LEAN_KEYS, PROBE_KEY, QSAFullStepGroupProbeBlock, QSAFullStepProbeBlock  # noqa: E402


class _QxEarlyMixin:
    """`qx_early` (VLLM_TPU_MK_DCP_QX_EARLY) on any stage-C block at main-KV DCP 2 / 4; off = the block's own code."""

    def __init__(self, *, qx_early: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.qx_early = bool(qx_early)
        assert not self.qx_early or self.dcp in (2, 4), ("X-DCP-QX-EARLY needs main-KV DCP 2 or 4", self.dcp)
        assert not (self.qx_early and self.compressed_dcp), "X-DCP-QX-EARLY: main-KV DCP only (no compressed-key DCP)"

    def _qx_copies(self, ctx: Any) -> list[Any]:
        """The q exchange descriptors `_dcp_attend` builds: DCP 2 = `_dcp_pair(qf_q -> qd_q, sems 0)`, DCP 4 = D14's
        `_group_copies(qf_q -> rank me ^ off's qdg_q[off - 1], sems 0)`."""
        scr = ctx.scr
        if self.dcp == 2:
            return ctx.comm.group_copies(lambda off: scr["qf_q"], scr["qd_q"], scr["qd_ssem0"], scr["qd_rsem0"], 2)
        return self._group_copies(ctx, lambda off: scr["qf_q"], lambda off: scr["qdg_q"].at[off - 1], 0)

    def _project(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        kv = super()._project(ctx, l, slot)
        if self.qx_early:  # qf_q is complete here: start the exchange (it runs under `_select`)
            with ctx.stream.sc("qsa_dcp_qxs", 2):
                for d in self._qx_copies(ctx):
                    d.start()
        return kv

    def _dcp_attend(self, ctx: Any, l: Any, zeros: tuple) -> None:  # noqa: E741
        if not self.qx_early:
            return super()._dcp_attend(ctx, l, zeros)
        scr, sc = ctx.scr, ctx.stream.sc
        with sc("qsa_dcp_qx", 2):  # wait for the exchange started in `_project` (recv, then send: `exchange`'s order)
            ds = self._qx_copies(ctx)
            for d in ds:
                d.wait_recv()
            for d in ds:
                d.wait_send()
        self._attend_unions(ctx, l, zeros)
        with sc("qsa_dcp_px", 2):  # the partial exchange, as the base `_dcp_attend` (MM's pair / D14's group)
            if self.dcp == 2:
                self._dcp_pair(ctx, scr["qd_os"], scr["qd_or"], 1)
                self._dcp_pair(ctx, scr["qd_ls"], scr["qd_lr"], 2)
            else:
                ctx.comm.exchange(
                    self._group_copies(
                        ctx, lambda off: scr["qdg_os"].at[off - 1], lambda off: scr["qdg_or"].at[off], 1
                    )
                    + self._group_copies(
                        ctx, lambda off: scr["qdg_ls"].at[off - 1], lambda off: scr["qdg_lr"].at[off], 2
                    )
                )
        return None


class QSAFullStepQxBlock(_QxEarlyMixin, QSAFullStepBlock):
    """MM's stage-C block (DCP 2 pair) with X-DCP-QX-EARLY."""


class QSAFullStepGroupQxBlock(_QxEarlyMixin, QSAFullStepGroupBlock):
    """D14's DCP-4 group block with X-DCP-QX-EARLY."""


class QSAFullStepProbeQxBlock(_QxEarlyMixin, QSAFullStepProbeBlock):
    """MM's block with the union probe (tests) and X-DCP-QX-EARLY."""


class QSAFullStepGroupProbeQxBlock(_QxEarlyMixin, QSAFullStepGroupProbeBlock):
    """D14's block with the union probe (tests) and X-DCP-QX-EARLY."""


class QSAFullStepLeanQxBlock(_QxEarlyMixin, QSAFullStepLeanBlock):
    """MM's block with the D08 S1 stages and X-DCP-QX-EARLY."""


class QSAFullStepGroupLeanQxBlock(_QxEarlyMixin, QSAFullStepGroupLeanBlock):
    """D14's block with the D08 S1 stages and X-DCP-QX-EARLY."""


class QSAFullStepIdxQxBlock(_QxEarlyMixin, QSAFullStepIdxBlock):
    """MM's block with the D08 S1 stages, the L1-IDX options and X-DCP-QX-EARLY."""


class QSAFullStepGroupIdxQxBlock(_QxEarlyMixin, QSAFullStepGroupIdxBlock):
    """D14's block with the D08 S1 stages, the L1-IDX options and X-DCP-QX-EARLY."""


QX_CLASSES = {
    QSAFullStepBlock: QSAFullStepQxBlock, QSAFullStepGroupBlock: QSAFullStepGroupQxBlock,
    QSAFullStepProbeBlock: QSAFullStepProbeQxBlock, QSAFullStepGroupProbeBlock: QSAFullStepGroupProbeQxBlock,
    QSAFullStepLeanBlock: QSAFullStepLeanQxBlock, QSAFullStepGroupLeanBlock: QSAFullStepGroupLeanQxBlock,
    QSAFullStepIdxBlock: QSAFullStepIdxQxBlock, QSAFullStepGroupIdxBlock: QSAFullStepGroupIdxQxBlock,
}


def qx_base_class(indexed: dict) -> type:
    """The class `wholestep_spec` builds for `indexed` without `qx_early`: `lean_step_block`'s routing (an L1-IDX key ->
    the L1-IDX class, else an S1 key -> the lean class, else the probe -> the probe class), else MM's / D14's block."""
    group = int(indexed.get("dcp", 1) or 1) > 2
    if any(k in indexed for k in IDX_KEYS):
        return QSAFullStepGroupIdxBlock if group else QSAFullStepIdxBlock
    if any(k in indexed for k in LEAN_KEYS):
        return QSAFullStepGroupLeanBlock if group else QSAFullStepLeanBlock
    if PROBE_KEY in indexed:
        return QSAFullStepGroupProbeBlock if group else QSAFullStepProbeBlock
    return QSAFullStepGroupBlock if group else QSAFullStepBlock


def qx_step_block(indexed: dict, qsa: dict | None) -> Any:
    """The X-DCP-QX-EARLY subclass of `qx_base_class(indexed)` (the key present, even False, selects it; False = the base
    class's code)."""
    return QX_CLASSES[qx_base_class(indexed)](**indexed, **(qsa or {}))
# q38 L1-QX end

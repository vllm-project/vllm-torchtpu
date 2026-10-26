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
"""Fused DeepSeek-V4 HCA-layer attention for mixed (prefill-chunk) steps, T > 32.

The mixed-step counterpart of `fused_hca_decode`: one grid-less ``pallas_call``
per HCA layer replaces the SWA kernel, the MLA-p (verify) and MLA-m (prefill
chunk) kernels and their (acc, l, m) hand-off glue.

Per request:
  1. its compressed rows ``[0, n_c)`` stream from the HCA pages through a ring of
     ``blk_pages``-page blocks (issue cursor running ``slots - 1`` blocks ahead,
     across requests) and are decoded ONCE into a VMEM-resident bf16 array;
  2. its query rows run in blocks of ``chunk`` (decode / verify requests) or
     ``bq`` (prefill chunks) tokens: one online softmax over the sliding window
     (cached rows from the SWA pages + this step's rows) and then the resident
     compressed rows in tiles of ``kt`` rows -- tiles every row of the block can
     see run unmasked, only the tiles at the visibility edge are masked;
  3. q blocks are DMA'd in double-buffered, outputs DMA'd out asynchronously.
This removes stock MLA-m's re-read and re-decode of the whole prefix per
32-token query block, its exposed per-block page DMA, and the full-tile causal
mask (decomposition row 10).

Arithmetic follows `fused_hca_decode` / stock: scores scaled, masked with
``finfo(f32).min``, ``p = exp(s - m)`` in f32, PV at DEFAULT precision, f32
accumulation, sink added to the denominator at the end, approximate reciprocal.
New rows of every request are stored to their SWA pages (as stock does).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention.fused_hca_decode import (
    HEAD_DIM,
    LANES,
    NEG,
    NN,
    NT,
    SWA_PAGES,
    _decode_tokens,
    _recip,
    _tokens_to_words,
    _words,
    cdiv,
)

MIN_TOKENS = 64  # smallest step bucket served here (fused_hca_decode owns T <= 32)


def _round_up(x, m):
    return (x + m - 1) // m * m


@dataclasses.dataclass(frozen=True)
class Config:
    tokens: int
    heads: int
    chunk: int  # q-block tokens for decode / verify requests (verify window length)
    bq: int  # q-block tokens for prefill (mixed) requests
    kt: int  # compressed rows per compute tile
    window: int
    swa_page: int
    hca_page: int
    sm_scale: float
    max_req: int
    hca_ppr: int
    num_hca_pages: int
    nw: int  # SWA write pages per request
    na: int = 0  # SWA pages per request covering [s0, pf + q_len) (window-from-cache table)
    blk_pages: int = 128
    slots: int = 3
    issue_chunk: int = 16
    edge_mask_only: bool = True  # unmasked interior tiles (else every tile masked)
    unroll: int = 1  # interior-tile loop unroll
    kt_tail: int = 0  # >0: tail / edge tiles of this many rows (wide kt tiles only where all rows see them)
    kt_mid: int = 0  # >0: a middle tile width between kt and kt_tail
    swa_skip_cached: bool = False  # q blocks past the first W tokens of a request skip the cached-window keys
    swa_fc: bool = False  # window keys of each q block DMA'd back from the SWA pages (exact rows, lane-padded width)
    swa_merge: bool = False  # window keys + the first compressed tile in one (masked) update
    stats128: bool = False
    row_split: int = 1  # wide/narrow unmasked tiles processed as this many independent row groups  # m / l kept lane-replicated [rows, 128] (upstream flash_attention form)
    # timing-only removal arms (wrong numbers by construction)
    no_hca_dma: bool = False
    no_hca_compute: bool = False
    no_swa_compute: bool = False
    no_softmax: bool = False

    @property
    def blk_rows(self):
        return self.blk_pages * self.hca_page

    @property
    def ncmax(self):
        return self.hca_ppr * self.hca_page


def _kernel(cfg: Config, dist_ref, seq_ref, cu_ref, ext_ref, swa_win_ref, swa_wr_ref, swa_all_ref, hca_pt_ref,
            nkv_ref, sink_ref, q_hbm, _swa_in, hca_in, out_hbm, swa_hbm,
            swa_buf, swa_kv, wbuf, ring, res, qbuf, obuf, m_scr, l_scr, acc_scr, osz, winbuf, sems):
    T, H, W, P, HP = cfg.tokens, cfg.heads, cfg.window, cfg.swa_page, cfg.hca_page
    B, S, BP, KT = cfg.blk_rows, cfg.slots, cfg.blk_pages, cfg.kt
    ROWS2 = 2 * HP
    if hca_in.dtype == jnp.int32:  # int32 words [N_h * 2 * hca_page, 128] (flattened by the wrapper)
        hca2d = hca_in
    else:  # uint8 slabs [N_h, rows, 4, 128] -> int32 rows, a free bitcast
        n_, rr_, _, lanes_ = hca_in.shape
        hca2d = hca_in.bitcast(jnp.int32).reshape(n_ * rr_, lanes_)
    swa_hbm = _words(swa_hbm)
    nreq = dist_ref[2]
    nsmall = dist_ref[1]  # requests [0, nsmall) are decode / verify
    SEM_SWA_W, SEM_RING, SEM_Q, SEM_O, SEM_WIN = 2, 3, 3 + S, 5 + S, 7 + S

    def req_scalars(r):
        cu_r = cu_ref[r]
        qlen = cu_ref[r + 1] - cu_r
        pf = seq_ref[r] - qlen
        s0 = jnp.maximum(pf - (W - 1), 0)
        return cu_r, qlen, pf, s0

    def req_nc(r):
        """Compressed rows visible to the request's last token (>= every token's)."""
        end = cu_ref[r + 1]
        return jnp.where(end > cu_ref[r], ext_ref[jnp.maximum(end - 1, 0)], 0)

    def n_blocks(nc):
        return jnp.maximum(1, cdiv(nc, B))

    # ------------------------------------------------------------ SWA window (<= 2 DMAs) and writes
    def swa_window(r, slot, *, wait):
        _, _, pf, s0 = req_scalars(r)
        n = pf - s0
        off = s0 % P
        na = jnp.minimum(n, P - off)
        nb = n - na
        buf, sem = swa_buf.at[slot], sems.at[slot]
        for page, src, dst, cnt in ((swa_win_ref[SWA_PAGES * r], 2 * off, 0, na),
                                    (swa_win_ref[SWA_PAGES * r + 1], 0, 2 * na, nb)):
            @pl.when(cnt > 0)
            def _(page=page, src=src, dst=dst, cnt=cnt):
                cp = pltpu.make_async_copy(swa_hbm.at[page, pl.ds(src, 2 * cnt)],
                                           buf.at[pl.ds(dst, 2 * cnt)], sem)
                cp.wait() if wait else cp.start()

    def swa_writes(*, wait):
        """Every request's new rows -> its SWA pages (one DMA per page run)."""
        def per_req(r, c):
            cu_r, qlen, pf, _ = req_scalars(r)
            a0 = pf % P
            npg = cdiv(a0 + qlen, P)

            def per_page(j, c2):
                a = jnp.where(j == 0, a0, 0)
                done = jnp.where(j == 0, 0, P - a0 + (j - 1) * P)
                cnt = jnp.minimum(P - a, qlen - done)
                page = swa_wr_ref[cfg.nw * r + j]
                cp = pltpu.make_async_copy(wbuf.at[pl.ds(2 * (cu_r + done), 2 * cnt)],
                                           swa_hbm.at[page, pl.ds(2 * a, 2 * cnt)], sems.at[SEM_SWA_W])
                cp.wait() if wait else cp.start()
                return c2
            return lax.fori_loop(0, jnp.where(qlen > 0, npg, 0), per_page, c)
        lax.fori_loop(0, nreq, per_req, 0)

    # ------------------------------------------------------------ HCA ring -> resident
    CH = cfg.issue_chunk

    def live_pages(nc, b):
        return jnp.clip(cdiv(nc, HP) - b * BP, 0, BP)

    def issued_pages(nc, b):
        nl = live_pages(nc, b)
        return jnp.minimum(cdiv(nl, CH) * CH, BP) if CH else nl

    def page_copy(page, p, slot):
        pltpu.make_async_copy(hca2d.at[pl.ds(pl.multiple_of(page * ROWS2, ROWS2), ROWS2)],
                              ring.at[slot, pl.ds(p * ROWS2, ROWS2)], sems.at[SEM_RING + slot]).start()

    def hca_start(r, nc, b, slot):
        if cfg.no_hca_dma:
            return
        base = r * cfg.hca_ppr + b * BP
        nl = live_pages(nc, b)
        if CH:
            for c in range(cdiv(BP, CH)):
                @pl.when(c * CH < nl)
                def _(c=c):
                    for p in range(c * CH, min((c + 1) * CH, BP)):
                        page_copy(hca_pt_ref[base + jnp.minimum(p, nl - 1)], p, slot)
            return

        def body(p, c):
            page_copy(hca_pt_ref[base + p], p, slot)
            return c
        lax.fori_loop(0, nl, body, 0)

    def hca_wait(nc, b, slot):
        if cfg.no_hca_dma:
            return
        nl = issued_pages(nc, b)
        blk = ring.at[slot, pl.ds(0, nl * ROWS2)]

        @pl.when(nl > 0)
        def _():
            pltpu.make_async_copy(blk, blk, sems.at[SEM_RING + slot]).wait()

    # issue cursor over (request, block)
    def advance(c):
        cr, cb, cv = c
        crc = jnp.minimum(cr, nreq - 1)
        lastb = cb + 1 >= n_blocks(req_nc(crc))
        morer = cr + 1 < nreq
        return jnp.where(lastb, cr + 1, cr), jnp.where(lastb, 0, cb + 1), cv & (~lastb | morer)

    def issue(c, slot):
        cr, cb, cv = c

        @pl.when(cv)
        def _():
            crc = jnp.minimum(cr, nreq - 1)
            hca_start(crc, req_nc(crc), cb, slot)

    def load_resident(r, carry):
        nc = req_nc(r)
        kr = lax.broadcasted_iota(jnp.int32, (B, 1), 0)

        def blk(b, c):
            g, cur = c[0], c[1:]
            slot = g % S
            hca_wait(nc, b, slot)
            issue(cur, (g + S - 1) % S)
            cur = advance(cur)
            kvb = _decode_tokens(ring.at[slot], B)
            kvb = jnp.where(kr < live_pages(nc, b) * HP, kvb, jnp.zeros((), kvb.dtype))
            res[pl.ds(pl.multiple_of(b * B, B), B)] = kvb
            return (g + 1,) + tuple(cur)
        return lax.fori_loop(0, n_blocks(nc), blk, carry)

    # ------------------------------------------------------------ online softmax
    SL = LANES if cfg.stats128 else 1

    def update(q, kv, vis_mask, *, first, rows, r0=0):
        m_ref, l_ref, a_ref = m_scr.at[r0:r0 + rows], l_scr.at[r0:r0 + rows], acc_scr.at[r0:r0 + rows]
        s = lax.dot_general(q, kv, NT, preferred_element_type=jnp.float32)
        if cfg.no_softmax and not first:
            a_ref[...] = a_ref[...] + lax.dot_general(s.astype(jnp.bfloat16), kv, NN,
                                                      preferred_element_type=jnp.float32)
            return
        s = s * cfg.sm_scale if vis_mask is None else jnp.where(vis_mask, s * cfg.sm_scale, NEG)
        n = s.shape[1]
        m_blk = jnp.max(s, axis=1, keepdims=True)  # [rows, 1]
        if first:
            m_cur = m_blk if SL == 1 else jnp.broadcast_to(m_blk, (rows, SL))
            p = jnp.exp(s - m_blk)
        else:
            m_prev = m_ref[...]
            m_cur = jnp.maximum(m_prev, m_blk)  # [rows, SL]
            if SL > 1 and n % SL == 0:
                p = jnp.exp(s - jnp.tile(m_cur, (1, n // SL)))
            else:
                p = jnp.exp(s - m_cur[:, :1])
        l_blk = jnp.sum(p, axis=1, keepdims=True)
        pv = lax.dot_general(p.astype(jnp.bfloat16), kv, NN, preferred_element_type=jnp.float32)
        if first:
            l_ref[...] = l_blk if SL == 1 else jnp.broadcast_to(l_blk, (rows, SL))
            a_ref[...] = pv
        else:
            alpha = jnp.exp(m_prev - m_cur)
            l_ref[...] = alpha * l_ref[...] + l_blk
            a_ref[...] = (jnp.tile(alpha, (1, HEAD_DIM // SL)) if SL > 1 else alpha) * a_ref[...] + pv
        m_ref[...] = m_cur

    # ------------------------------------------------------------ q blocks
    def q_copy(off, QB, slot):
        return pltpu.make_async_copy(q_hbm.at[pl.ds(off, QB)], qbuf.at[slot, pl.ds(0, QB)],
                                     sems.at[SEM_Q + slot])

    def out_copy(slot, src0, dst0, n):
        return pltpu.make_async_copy(obuf.at[slot, pl.ds(src0, n)], out_hbm.at[pl.ds(dst0, n)],
                                     sems.at[SEM_O + slot])

    def out_wait(slot):
        n = osz[3 * slot]

        @pl.when(n > 0)
        def _():
            out_copy(slot, osz[3 * slot + 1], osz[3 * slot + 2], n).wait()
            osz[3 * slot] = 0

    def compute_request(r, QB, ocount):
        """All q blocks of request r with QB tokens per block; ocount = running out-slot counter."""
        rows = QB * H
        LN = min(_round_up(W + QB + 16, 16), T)
        if cfg.swa_merge and _round_up(W + LN, LANES) - W <= T:
            LN = _round_up(W + LN, LANES) - W  # lane-aligned window width so the merged key/mask concat lowers
        cu_r, qlen, pf, s0 = req_scalars(r)
        n_cached = pf - s0
        nq = cdiv(qlen, QB)
        row_tok = lax.broadcasted_iota(jnp.int32, (rows, 1), 0) // H
        jc = lax.broadcasted_iota(jnp.int32, (1, W), 1)
        jn = lax.broadcasted_iota(jnp.int32, (1, LN), 1)

        def offs(k):
            tok0 = cu_r + QB * k
            return tok0, jnp.minimum(QB, qlen - QB * k), jnp.minimum(tok0, T - QB)

        LW = _round_up(W + QB, 16)
        lp0 = s0 // P

        def win_span(k):
            first = pf + QB * k
            a = jnp.maximum(first - (W - 1), 0)
            return a, first + jnp.minimum(QB, qlen - QB * k) - a

        def win_copy(k, slot, *, wait):
            """Window rows [a, a + n) of q block k from the SWA cache pages (written above) -> winbuf[slot]."""
            a, n = win_span(k)
            off = a % P
            c1 = jnp.minimum(n, P - off)
            c2 = jnp.minimum(n - c1, P)
            c3 = n - c1 - c2
            lp = a // P - lp0
            for i, (src, dst, cnt) in enumerate(((off, 0, c1), (0, c1, c2), (0, c1 + c2, c3))):
                @pl.when(cnt > 0)
                def _(i=i, src=src, dst=dst, cnt=cnt):
                    page = swa_all_ref[cfg.na * r + lp + i]
                    cp = pltpu.make_async_copy(swa_hbm.at[page, pl.ds(2 * src, 2 * cnt)],
                                               winbuf.at[slot, pl.ds(2 * dst, 2 * cnt)], sems.at[SEM_WIN + slot])
                    cp.wait() if wait else cp.start()

        @pl.when(nq > 0)
        def _():
            q_copy(offs(0)[2], QB, 0).start()
            if cfg.swa_fc:
                win_copy(0, 0, wait=False)

        def qblock(k, oc):
            tok0, nrow, off = offs(k)
            qslot = k % 2
            q_copy(off, QB, qslot).wait()

            @pl.when(k + 1 < nq)
            def _():
                q_copy(offs(k + 1)[2], QB, 1 - qslot).start()

            if cfg.swa_fc:
                win_copy(k, qslot, wait=True)

                @pl.when(k + 1 < nq)
                def _():
                    win_copy(k + 1, 1 - qslot, wait=False)

            q = qbuf[qslot, :QB].reshape(rows, HEAD_DIM)
            qpos = pf + (off - cu_r) + row_tok  # [rows, 1]
            vis = jnp.zeros((rows, 1), jnp.int32)
            for j in range(QB):
                vis = jnp.where(row_tok == j, ext_ref[jnp.minimum(off + j, T - 1)], vis)

            # --- sliding window: cached rows [s0, pf) then this request's new rows
            st = jnp.clip(tok0 - W, 0, T - LN)
            st = pl.multiple_of(st // 16 * 16, 16)
            newk = nkv_ref[pl.ds(st, LN)]
            kv = jnp.concatenate([swa_kv[...], newk], axis=0)  # [W + LN, 512]
            cmask = (jc < n_cached) & (qpos - (s0 + jc) < W)
            kt_ = st + jn
            kpos = pf + kt_ - cu_r
            nmask = (kt_ >= cu_r) & (kt_ < cu_r + qlen) & (kpos <= qpos) & (qpos - kpos < W)
            if cfg.no_swa_compute:
                m_scr[:rows] = jnp.full((rows, SL), -1e30, jnp.float32)
                l_scr[:rows] = jnp.zeros((rows, SL), jnp.float32)
                acc_scr[:rows] = jnp.zeros((rows, HEAD_DIM), jnp.float32)
            vis_lo = ext_ref[tok0]
            vis_hi = ext_ref[tok0 + nrow - 1]
            KS = cfg.kt_tail or KT  # narrow tiles for the tail / visibility edge
            merged = (cfg.swa_merge and not cfg.no_swa_compute and not cfg.no_hca_compute
                      and (W + LN) % LANES == 0)

            def tile_mask(r0, K, kvt):
                kr = lax.broadcasted_iota(jnp.int32, (K, 1), 0)
                kc = lax.broadcasted_iota(jnp.int32, (1, K), 1)
                # rows past the request's last decoded block may be stale/uninitialized VMEM: zero them
                return jnp.where(r0 + kr < vis_hi, kvt, jnp.zeros((), kvt.dtype)), r0 + kc < vis

            if merged:  # window + compressed rows [0, KS) in one update
                kv0, m0_ = tile_mask(0, KS, res[pl.ds(0, KS)])
                update(q, jnp.concatenate([kv, kv0], axis=0), jnp.concatenate([cmask, nmask, m0_], axis=1),
                       first=True, rows=rows)
            elif not cfg.no_swa_compute and cfg.swa_fc:
                a_w, n_w = win_span(k)
                jw = lax.broadcasted_iota(jnp.int32, (1, LW), 1)
                jr = lax.broadcasted_iota(jnp.int32, (LW, 1), 0)
                wk = _decode_tokens(winbuf.at[qslot], LW)
                wk = jnp.where(jr < n_w, wk, jnp.zeros((), wk.dtype))
                dpos = qpos - (a_w + jw)
                update(q, wk, (jw < n_w) & (dpos >= 0) & (dpos < W), first=True, rows=rows)
            elif not cfg.no_swa_compute and cfg.swa_skip_cached:
                # blocks starting >= W tokens into the request never see a cached window row: new rows only
                @pl.when(tok0 - cu_r < W)
                def _():
                    update(q, kv, jnp.concatenate([cmask, nmask], axis=1), first=True, rows=rows)

                @pl.when(tok0 - cu_r >= W)
                def _():
                    update(q, newk, nmask, first=True, rows=rows)
            elif not cfg.no_swa_compute:
                update(q, kv, jnp.concatenate([cmask, nmask], axis=1), first=True, rows=rows)

            # --- compressed rows from the resident array: wide unmasked tiles, narrow unmasked, narrow masked
            if not cfg.no_hca_compute:
                r0 = KS if merged else 0
                lo = vis_lo if cfg.edge_mask_only else r0
                n_big = jnp.maximum(lo - r0, 0) // KT
                r1 = r0 + n_big * KT
                KM = cfg.kt_mid
                n_mid = jnp.maximum(lo - r1, 0) // KM if KM else 0
                r1m = r1 + n_mid * KM if KM else r1
                n_small = jnp.maximum(lo - r1m, 0) // KS if KS != KT else 0
                r2 = r1m + n_small * KS
                n_edge = cdiv(jnp.maximum(vis_hi - r2, 0), KS)

                RS = cfg.row_split if rows % (cfg.row_split * 16) == 0 else 1

                def full_tile(K, base):
                    def body(t, c):
                        kvt = res[pl.ds(pl.multiple_of(base + t * K, 16), K)]
                        rh = rows // RS
                        for h in range(RS):  # independent row groups: MXU of one can overlap VPU of the other
                            update(q[h * rh:(h + 1) * rh], kvt, None, first=False, rows=rh, r0=h * rh)
                        return c
                    return body

                lax.fori_loop(0, n_big, full_tile(KT, r0), 0)
                if KS != KT:
                    if KM:
                        lax.fori_loop(0, n_mid, full_tile(KM, r1), 0)
                    lax.fori_loop(0, n_small, full_tile(KS, r1m), 0)

                def edge_tile(t, c):
                    rs = pl.multiple_of(r2 + t * KS, 16)
                    kvt, msk = tile_mask(rs, KS, res[pl.ds(rs, KS)])
                    update(q, kvt, msk, first=False, rows=rows)
                    return c

                lax.fori_loop(0, n_edge, edge_tile, 0)

            # --- finalize -> out block
            sink = jnp.concatenate([sink_ref[...].astype(jnp.float32)] * QB, axis=0)  # [rows, 1]
            m = m_scr[:rows, :1]
            l_tot = l_scr[:rows, :1] + jnp.exp(sink - m)
            o = (acc_scr[:rows] * _recip(l_tot)).astype(obuf.dtype).reshape(QB, H, HEAD_DIM)
            oslot = oc % 2
            out_wait(oslot)
            obuf[oslot, :QB] = o
            src0 = tok0 - off
            out_copy(oslot, src0, tok0, nrow).start()
            osz[3 * oslot] = nrow
            osz[3 * oslot + 1] = src0
            osz[3 * oslot + 2] = tok0
            return oc + 1

        return lax.fori_loop(0, nq, qblock, ocount)

    # ------------------------------------------------------------ main
    osz[0] = 0
    osz[3] = 0

    @pl.when(nreq > 0)
    def _():
        swa_buf[...] = jnp.zeros(swa_buf.shape, swa_buf.dtype)  # never-written window rows stay finite
        if cfg.no_hca_dma:
            ring[...] = jnp.zeros(ring.shape, ring.dtype)
        WT = min(T, 256)
        for c0 in range(0, T, WT):
            wbuf[2 * c0:2 * (c0 + WT)] = _tokens_to_words(nkv_ref[c0:c0 + WT])
        swa_writes(wait=False)
        if cfg.swa_fc:  # q blocks read their window (incl. this step's rows) back from the SWA pages
            winbuf[...] = jnp.zeros(winbuf.shape, winbuf.dtype)
            swa_writes(wait=True)
        swa_window(0, 0, wait=False)
        cur = (jnp.int32(0), jnp.int32(0), jnp.bool_(True))
        for j in range(S - 1):
            issue(cur, j)
            cur = advance(cur)

        def req_body(r, carry):
            g_cur, oc = carry[:4], carry[4]
            slot = r % 2
            swa_window(r, slot, wait=True)

            @pl.when(r + 1 < nreq)
            def _():
                swa_window(r + 1, 1 - slot, wait=False)

            swa_kv[...] = _decode_tokens(swa_buf.at[slot], W)
            g_cur = load_resident(r, g_cur)
            oc = lax.cond(r < nsmall, lambda o: compute_request(r, cfg.chunk, o),
                          lambda o: compute_request(r, cfg.bq, o), oc)
            return tuple(g_cur) + (oc,)

        lax.fori_loop(0, nreq, req_body, (jnp.int32(0),) + tuple(cur) + (jnp.int32(0),))
        if not cfg.swa_fc:
            swa_writes(wait=True)

    out_wait(0)
    out_wait(1)

    # rows past the step's last token: zeros (never garbage into o_proj / MoE)
    QZ = cfg.bq
    obuf[0, :QZ] = jnp.zeros((QZ, H, HEAD_DIM), obuf.dtype)
    t_end = cu_ref[nreq]
    nz = cdiv(T - t_end, QZ)

    def zcopy(i, *, wait):
        a = t_end + i * QZ
        n = jnp.minimum(QZ, T - a)
        cp = pltpu.make_async_copy(obuf.at[0, pl.ds(0, n)], out_hbm.at[pl.ds(a, n)], sems.at[SEM_O])
        cp.wait() if wait else cp.start()

    lax.fori_loop(0, nz, lambda i, c: (zcopy(i, wait=False), c)[1], 0)
    lax.fori_loop(0, nz, lambda i, c: (zcopy(i, wait=True), c)[1], 0)


def supported(*, tokens, heads, chunk, bq=16, two_caches_same_buffer=False, swa_only=False,
              non_causal_block=False) -> bool:
    return (not swa_only and not two_caches_same_buffer and not non_causal_block
            and tokens >= MIN_TOKENS and tokens >= bq and heads % 16 == 0 and chunk <= bq)


def default_bq(heads):
    return max(8, 512 // heads)  # 512 query rows per prefill block


def fused_hca_mixed_attention(
    q,  # bf16 [T, H, 512] (post qnorm_rope)
    new_kv,  # bf16 [T, 512] (post rope)
    sw_cache,  # uint8 [N, 2 * swa_page, 4, 128] slabs (or int32 words)
    seq_lens,  # int32 [max_req]
    swa_page_indices,  # int32 [max_req, P_swa] (or flat)
    cu_q_lens,  # int32 [max_req + 1]
    distribution,  # int32 [3]
    hca_cache,  # uint8 [N_h, 2 * hca_page, 4, 128]
    kv_lens_to_attend,  # int32 [T]
    hca_page_indices,  # int32 [max_req, P_hca] (or flat)
    attention_sinks,  # [H]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    chunk: int = 7,
    bq: int | None = None,
    kt: int = 512,
    blk_pages: int = 128,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
    **knobs,
):
    T, H, D = q.shape
    assert D == HEAD_DIM, q.shape
    bq = bq or default_bq(H)
    assert T % min(T, 256) == 0
    max_req = cu_q_lens.shape[0] - 1
    hca_page = hca_cache.shape[1] // 2
    num_hca_pages = hca_cache.shape[0]
    swa_pt = swa_page_indices.reshape(max_req, -1).astype(jnp.int32)
    hca_pt = hca_page_indices.reshape(-1).astype(jnp.int32)
    hca_ppr = hca_pt.shape[0] // max_req
    P = logical_page_size
    nw = cdiv(T, P) + 1
    na = cdiv(T + sliding_window, P) + 2
    cfg = Config(tokens=T, heads=H, chunk=chunk, bq=bq, kt=kt, window=sliding_window, swa_page=P,
                 hca_page=hca_page, sm_scale=float(sm_scale), max_req=max_req, hca_ppr=hca_ppr,
                 num_hca_pages=num_hca_pages, nw=nw, na=na, blk_pages=min(blk_pages, hca_ppr), **knobs)
    cu = cu_q_lens.astype(jnp.int32)
    qlens = cu[1:] - cu[:-1]
    pf = seq_lens.astype(jnp.int32) - qlens
    width = swa_pt.shape[1]
    lp = jnp.maximum(pf - (cfg.window - 1), 0) // P
    cols = jnp.minimum(lp[:, None] + jnp.arange(SWA_PAGES, dtype=jnp.int32)[None, :], width - 1)
    swa_win = jnp.take_along_axis(swa_pt, cols, axis=1).reshape(-1)
    wcols = jnp.minimum((pf // P)[:, None] + jnp.arange(nw, dtype=jnp.int32)[None, :], width - 1)
    swa_wr = jnp.take_along_axis(swa_pt, wcols, axis=1).reshape(-1)
    acols = jnp.minimum(lp[:, None] + jnp.arange(na, dtype=jnp.int32)[None, :], width - 1)
    swa_all = jnp.take_along_axis(swa_pt, acols, axis=1).reshape(-1)
    sinks = attention_sinks.reshape(H, 1)
    if hca_cache.dtype == jnp.int32:  # int32 words (CPU interpreter dry runs)
        hca_cache = hca_cache.reshape(-1, LANES)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    rows_max = bq * H
    scratch = [
        pltpu.VMEM((2, 2 * cfg.window, LANES), jnp.int32),  # SWA window words x2
        pltpu.VMEM((cfg.window, HEAD_DIM), jnp.bfloat16),  # decoded window of the current request
        pltpu.VMEM((2 * T, LANES), jnp.int32),  # this step's rows as cache words
        pltpu.VMEM((cfg.slots, 2 * cfg.blk_rows, LANES), jnp.int32),  # HCA block ring
        pltpu.VMEM((cfg.ncmax + max(cfg.blk_rows, kt), HEAD_DIM), jnp.bfloat16),  # resident compressed rows
        pltpu.VMEM((2, bq, H, HEAD_DIM), q.dtype),  # q blocks
        pltpu.VMEM((2, bq, H, HEAD_DIM), q.dtype),  # out blocks
        pltpu.VMEM((rows_max, LANES if knobs.get("stats128") else 1), jnp.float32),  # m
        pltpu.VMEM((rows_max, LANES if knobs.get("stats128") else 1), jnp.float32),  # l
        pltpu.VMEM((rows_max, HEAD_DIM), jnp.float32),
        pltpu.SMEM((6,), jnp.int32),  # out DMA (n, src, dst) per slot
        pltpu.VMEM((2, 2 * _round_up(cfg.window + bq, 16), LANES), jnp.int32),  # per-q-block window words x2
        pltpu.SemaphoreType.DMA((9 + cfg.slots,)),  # SWA win x2, SWA write, ring x S, q x2, out x2, block window x2
    ]
    if vmem_limit_bytes is None:
        est = (2 * 2 * cfg.window * LANES * 4 + cfg.window * HEAD_DIM * 2 + 2 * T * LANES * 4
               + T * HEAD_DIM * 2 + cfg.slots * 2 * cfg.blk_rows * LANES * 4
               + (cfg.ncmax + cfg.blk_rows) * HEAD_DIM * 2 + 4 * rows_max * HEAD_DIM * 2
               + rows_max * HEAD_DIM * 4 + 3 * rows_max * max(kt, 2 * cfg.window + bq + 16) * 4)
        vmem_limit_bytes = min(int(est * 1.5) + (8 << 20), 60 << 20)
    operands = [distribution.astype(jnp.int32), seq_lens.astype(jnp.int32), cu,
                kv_lens_to_attend.astype(jnp.int32), swa_win, swa_wr, swa_all, hca_pt, new_kv, sinks, q,
                sw_cache, hca_cache]
    out, swa = pl.pallas_call(
        functools.partial(_kernel, cfg),
        out_shape=[jax.ShapeDtypeStruct((T, H, HEAD_DIM), q.dtype),
                   jax.ShapeDtypeStruct(sw_cache.shape, sw_cache.dtype)],
        in_specs=[smem] * 8 + [vmem, vmem, hbm, hbm, hbm],
        out_specs=[hbm, hbm],
        input_output_aliases={11: 1},
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, vmem_limit_bytes=vmem_limit_bytes),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fused_hca_mixed_t{T}_h{H}_q{bq}_k{kt}_b{cfg.blk_rows}",
    )(*operands)
    return out, swa

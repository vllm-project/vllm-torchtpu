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
"""Fused DeepSeek-V4 HCA-layer attention for small (decode-bucket) steps.

One grid-less ``pallas_call`` per HCA layer replaces the SWA kernel, the HCA
MLA kernels, their partial (acc, l, m) hand-off glue and the new-row packing:
per request, one online softmax runs over the sliding window (cached rows from
the SWA pages plus this step's rows) and then the compressed rows ``k <
kv_lens_to_attend[t]`` streamed from the HCA pages through a 2-slot VMEM ring
that runs ahead across requests, finishing with the attention sink.

Every request of the step is handled, whatever its query length (decode
``q_len = 1``, speculative verify windows, short prefill tails): a request is
processed as consecutive chunks of at most ``chunk`` query rows, so the call
needs no runtime branch around it (STRAT-DSV4-01) — the only gate is static
(the step's token bucket).

Arithmetic follows the stock kernels (`mla_swa`, `mla`): scores scaled, masked
with ``finfo(f32).min``, ``p = exp(s - m)`` in f32, PV at DEFAULT precision
(p rounded to bf16 by the MXU), f32 accumulation, sink added to the
denominator at the end, approximate reciprocal. Unlike stock, the window
accumulator is not rounded to bf16 between the two parts (one softmax).

Cache layouts (served, block size 1024):
  SWA   int32 ``[N, 2 * swa_page, 128]``: token = 2 word rows; word ``w`` of
        row ``s`` = bf16 ``256 s + w`` (low half) | bf16 ``256 s + 128 + w``.
  HCA   uint8 ``[N, 2 * hca_page, 4, 128]`` slabs of the same words (viewed as
        int32 in-kernel, a free bitcast).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _rope_lib():
    from vllm_torchtpu.kernels.deepseek_v4 import rope  # lazy: pulls the model's quantization deps
    return rope

NEG = float(jnp.finfo(jnp.float32).min)
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
LANES = 128
HEAD_DIM = 512
SWA_PAGES = 3  # window pages per request (<= 127 cached + <= 32 new rows)
LOG2E = 1.4426950408889634
_HI16 = -65536  # 0xFFFF0000 as int32
MAX_TOKENS = 32  # largest step bucket the kernel is built for


def cdiv(a, b):
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class Config:
    tokens: int  # T, the step's token bucket
    heads: int  # H, query heads on this device
    chunk: int  # query rows per chunk (verify window length)
    window: int
    swa_page: int  # SWA tokens per page
    hca_page: int  # compressed rows per HCA page
    sm_scale: float
    max_req: int
    hca_ppr: int  # HCA page-table row length
    num_hca_pages: int
    blk_pages: int = 128  # HCA pages per KV block
    slots: int = 3  # HCA ring depth: blocks in flight = slots - 1 ahead of the one consumed
    trim: bool = True  # issue only the live pages of a request's last block (else clamp + re-read)
    coalesce: bool = False  # one DMA per run of consecutive physical pages
    rope_in_kernel: bool = False  # stage B (B2+B4): q per-head RMSNorm + RoPE and kv RoPE in the prologue
    q_eps: float = 1e-6
    compute_tile: int = 0  # >0: consume a KV block in sub-tiles of this many rows (live ones only, online softmax)
    zsize_issue: bool = False  # branch-free issue of all pages, dead ones as zero-length DMAs
    issue_chunk: int = 16  # >0: trimmed issue as unrolled chunks of this many pages (tail clamped within a chunk)
    scale_fold: bool = False  # exp2 with sm_scale*log2e folded (changes rounding)
    tail_mask_only: bool = False  # mask only the block holding the visibility edge
    # timing-only removal arms (wrong numbers by construction)
    no_hca_dma: bool = False
    no_hca_compute: bool = False
    no_softmax: bool = False
    no_swa_compute: bool = False

    @property
    def rows(self) -> int:
        return self.chunk * self.heads

    @property
    def blk_rows(self) -> int:
        return self.blk_pages * self.hca_page


def _f32(w):
    return lax.bitcast_convert_type(w, jnp.float32)


def _decode_tokens(buf, n_tok):
    """int32 ``[2 n, 128]`` ref of bf16-latent tokens -> bf16 ``[n, 512]``."""
    even = buf[pl.ds(0, n_tok, stride=2), :]
    odd = buf[pl.ds(1, n_tok, stride=2), :]
    parts = (even << 16, even & _HI16, odd << 16, odd & _HI16)
    return jnp.concatenate([_f32(p) for p in parts], axis=-1).astype(jnp.bfloat16)


def _tokens_to_words(x):
    """bf16 ``[n, 512]`` -> int32 ``[2 n, 128]`` cache rows (inverse of the above)."""
    bits = lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
    words = []
    for s in range(2):
        lo = bits[:, 256 * s: 256 * s + 128]
        hi = bits[:, 256 * s + 128: 256 * s + 256]
        words.append(((lo >> 16) & 0xFFFF) | (hi & _HI16))
    n = x.shape[0]
    return jnp.stack(words, axis=1).reshape(2 * n, LANES)


def _words(ref):
    """A cache ref as int32 words: uint8 slabs [N, R, 4, 128] -> int32 [N, R, 128] (free bitcast); int32 passes."""
    if ref.dtype == jnp.int32:
        return ref
    n, rr, _, lanes = ref.shape
    return ref.bitcast(jnp.int32).reshape(n, rr, lanes)


def _recip(x):
    r = getattr(pltpu, "reciprocal", None)
    if r is None:  # older jax (CPU dry runs)
        return 1.0 / x
    return r(x, approx=True)


def _kernel(cfg: Config, dist_ref, seq_ref, cu_ref, ext_ref, swa_pt_ref, hca_pt_ref,
            q_ref, nkv_ref, sink_ref, *rest):
    if cfg.rope_in_kernel:
        (cs_ref, _swa_in, hca_in, out_ref, swa_hbm,
         swa_buf, wbuf, kv_buf, m_scr, l_scr, acc_scr, sems, qs_scr, nkvs_scr) = rest
        rope_lib = _rope_lib()
        # q: projection tiles -> [T, H, 512], per-head RMSNorm + RoPE (rope.qnorm_rope_flat's body, same ops/order)
        rope_lib._untile_heads(q_ref.bitcast(jnp.uint32), qs_scr)
        rope_lib._qnorm_rope_body(qs_scr, cs_ref, qs_scr, rotary_dim=cs_ref.shape[-1], eps=cfg.q_eps,
                                  inverse=False, out_dtype=qs_scr.dtype)
        # kv: RoPE on the last lane block only (rope.rope's body); the other lanes pass through untouched
        tail = nkv_ref[:, HEAD_DIM - LANES:].astype(jnp.float32)
        cos, sin = rope_lib._cos_sin_lanes(cs_ref[...], rotary_dim=cs_ref.shape[-1], inverse=False)
        nkvs_scr[:, :HEAD_DIM - LANES] = nkv_ref[:, :HEAD_DIM - LANES]
        nkvs_scr[:, HEAD_DIM - LANES:] = (tail * cos + rope_lib._rotate_gptj(tail) * sin).astype(nkvs_scr.dtype)
        q_ref, nkv_ref = qs_scr, nkvs_scr
    else:
        (_swa_in, hca_in, out_ref, swa_hbm,
         swa_buf, wbuf, kv_buf, m_scr, l_scr, acc_scr, sems) = rest
    T, H, C, W, P = cfg.tokens, cfg.heads, cfg.chunk, cfg.window, cfg.swa_page
    R, B, HP = cfg.rows, cfg.blk_rows, cfg.hca_page
    if hca_in.dtype == jnp.int32:  # int32 [N_h * 2 * hca_page, 128] (flattened by the wrapper)
        hca2d = hca_in
    else:  # uint8 slabs [N_h, rows, 4, 128] -> int32 rows, a free bitcast
        n_, rr_, _, lanes_ = hca_in.shape
        hca2d = hca_in.bitcast(jnp.int32).reshape(n_ * rr_, lanes_)
    swa_hbm = _words(swa_hbm)
    nreq = dist_ref[2]
    sem_swa_w = sems.at[2]

    # ------------------------------------------------------------ scalars
    def req_scalars(r):
        cu_r = cu_ref[r]
        qlen = cu_ref[r + 1] - cu_r
        pf = seq_ref[r] - qlen  # position of the request's first new row
        s0 = jnp.maximum(pf - (W - 1), 0)
        return cu_r, qlen, pf, s0

    def vreq_nc(r, k):
        """Compressed rows visible to the last query row of chunk k of request r."""
        cu_r = cu_ref[r]
        end = jnp.minimum(cu_r + C * (k + 1), cu_ref[r + 1])
        return ext_ref[jnp.maximum(end - 1, 0)]

    def n_blocks(nc):
        return jnp.maximum(1, cdiv(nc, B))

    # ------------------------------------------------------------ SWA window (<= 2 DMAs)
    def swa_window(r, slot, *, wait):
        _, _, pf, s0 = req_scalars(r)
        n = pf - s0
        off = s0 % P
        na = jnp.minimum(n, P - off)
        nb = n - na
        buf, sem = swa_buf.at[slot], sems.at[slot]
        for page, src, dst, cnt in ((swa_pt_ref[SWA_PAGES * r], 2 * off, 0, na),
                                    (swa_pt_ref[SWA_PAGES * r + 1], 0, 2 * na, nb)):
            @pl.when(cnt > 0)
            def _(page=page, src=src, dst=dst, cnt=cnt):
                cp = pltpu.make_async_copy(swa_hbm.at[page, pl.ds(src, 2 * cnt)],
                                           buf.at[pl.ds(dst, 2 * cnt)], sem)
                cp.wait() if wait else cp.start()

    def swa_write(r, *, wait):
        """The request's new rows -> its SWA pages (<= 2 page-contiguous runs)."""
        cu_r, qlen, pf, s0 = req_scalars(r)
        lp0 = s0 // P
        page_a = pf // P
        off = pf % P
        na = jnp.minimum(qlen, P - off)
        nb = qlen - na
        pa = swa_pt_ref[SWA_PAGES * r + jnp.clip(page_a - lp0, 0, SWA_PAGES - 1)]
        pb = swa_pt_ref[SWA_PAGES * r + jnp.clip(page_a + 1 - lp0, 0, SWA_PAGES - 1)]
        for page, dst, src, cnt in ((pa, 2 * off, 2 * cu_r, na), (pb, 0, 2 * (cu_r + na), nb)):
            @pl.when(cnt > 0)
            def _(page=page, dst=dst, src=src, cnt=cnt):
                cp = pltpu.make_async_copy(wbuf.at[pl.ds(src, 2 * cnt)],
                                           swa_hbm.at[page, pl.ds(dst, 2 * cnt)], sem_swa_w)
                cp.wait() if wait else cp.start()

    # ------------------------------------------------------------ HCA ring
    S, BP, ROWS = cfg.slots, cfg.blk_pages, 2 * HP
    n_pages_hbm = cfg.num_hca_pages

    def live_pages(nc, b):
        if cfg.trim:
            return jnp.clip(cdiv(nc, HP) - b * BP, 0, BP)
        return jnp.int32(BP)

    CH = cfg.issue_chunk

    def issued_pages(nc, b):
        nl = live_pages(nc, b)
        if cfg.trim and CH and not cfg.zsize_issue:
            return jnp.minimum(cdiv(nl, CH) * CH, BP)
        return nl

    def page_copy(page, p, slot):
        pltpu.make_async_copy(hca2d.at[pl.ds(pl.multiple_of(page * ROWS, ROWS), ROWS)],
                              kv_buf.at[slot, pl.ds(p * ROWS, ROWS)], sems.at[3 + slot]).start()

    def hca_start(r, nc, b, slot):
        if cfg.no_hca_dma:
            return
        base = r * cfg.hca_ppr + b * BP
        if not cfg.trim:  # clamp tail pages to the last live page (re-read, masked)
            last = jnp.maximum(cdiv(nc, HP) - 1, 0) - b * BP
            for p in range(BP):
                page_copy(jnp.clip(hca_pt_ref[base + jnp.minimum(p, last)], 0, n_pages_hbm - 1), p, slot)
            return
        nl = live_pages(nc, b)
        if cfg.zsize_issue:
            for p in range(BP):
                live = (p < nl).astype(jnp.int32)
                page = hca_pt_ref[base + jnp.maximum(jnp.minimum(p, nl - 1), 0)]
                sz = ROWS * live
                pltpu.make_async_copy(hca2d.at[pl.ds(pl.multiple_of(page * ROWS, ROWS), sz)],
                                      kv_buf.at[slot, pl.ds(p * ROWS, sz)], sems.at[3 + slot]).start()
            return
        if CH:
            for c in range(cdiv(BP, CH)):
                @pl.when(c * CH < nl)
                def _(c=c):
                    for p in range(c * CH, min((c + 1) * CH, BP)):
                        page_copy(hca_pt_ref[base + jnp.minimum(p, nl - 1)], p, slot)
            return
        if cfg.coalesce:
            def run_dma(p0, ln):
                pg0 = hca_pt_ref[base + p0]
                pltpu.make_async_copy(hca2d.at[pl.ds(pl.multiple_of(pg0 * ROWS, ROWS), ln * ROWS)],
                                      kv_buf.at[slot, pl.ds(p0 * ROWS, ln * ROWS)], sems.at[3 + slot]).start()

            def body(i, c):
                p0, prev = c
                pg = hca_pt_ref[base + i]
                brk = pg != prev + 1

                @pl.when(brk & (i > 0))
                def _():
                    run_dma(p0, i - p0)
                return jnp.where(brk, i, p0), pg

            @pl.when(nl > 0)
            def _():
                p0, _ = lax.fori_loop(0, nl, body, (jnp.int32(0), jnp.int32(-2)))
                run_dma(p0, nl - p0)
            return

        @pl.when(nl == BP)
        def _():
            for p in range(BP):
                page_copy(hca_pt_ref[base + p], p, slot)

        @pl.when((nl > 0) & (nl < BP))
        def _():
            def body(p, c):
                page_copy(hca_pt_ref[base + p], p, slot)
                return c
            lax.fori_loop(0, nl, body, 0)

    def hca_wait(nc, b, slot):
        if cfg.no_hca_dma:
            return
        nl = issued_pages(nc, b)
        blk = kv_buf.at[slot, pl.ds(0, nl * ROWS)]

        @pl.when(nl > 0)
        def _():
            pltpu.make_async_copy(blk, blk, sems.at[3 + slot]).wait()

    # Issue cursor over the consume order (request, chunk, block); runs S - 1 blocks ahead.
    def cur_nc(c):
        cr, ck, _, _ = c
        return vreq_nc(jnp.minimum(cr, nreq - 1), ck)

    def advance(c):
        cr, ck, cb, cv = c
        crc = jnp.minimum(cr, nreq - 1)
        nb = n_blocks(vreq_nc(crc, ck))
        nch = cdiv(cu_ref[crc + 1] - cu_ref[crc], C)
        lastb = cb + 1 >= nb
        morek = ck + 1 < nch
        morer = cr + 1 < nreq
        nr = jnp.where(lastb & ~morek, cr + 1, cr)
        nk = jnp.where(lastb, jnp.where(morek, ck + 1, 0), ck)
        nbb = jnp.where(lastb, 0, cb + 1)
        return nr, nk, nbb, cv & (~lastb | morek | morer)

    def issue(c, slot):
        cr, ck, cb, cv = c

        @pl.when(cv)
        def _():
            hca_start(jnp.minimum(cr, nreq - 1), cur_nc(c), cb, slot)

    # ------------------------------------------------------------ online softmax
    def update(q, kv, vis, *, first):
        s = lax.dot_general(q, kv, NT, preferred_element_type=jnp.float32)  # [R, n]
        if cfg.no_softmax and not first:
            pv = lax.dot_general(s.astype(jnp.bfloat16), kv, NN, preferred_element_type=jnp.float32)
            acc_scr[...] = acc_scr[...] + pv
            return
        if cfg.scale_fold:
            s = jnp.where(vis, s, NEG)  # unscaled
            c = cfg.sm_scale * LOG2E
        else:
            s = jnp.where(vis, s * cfg.sm_scale, NEG) if vis is not None else s * cfg.sm_scale
        m_blk = jnp.max(s, axis=1, keepdims=True)
        m_prev = None if first else m_scr[...]
        m_cur = m_blk if first else jnp.maximum(m_prev, m_blk)
        if cfg.scale_fold:
            p = jnp.exp2((s - m_cur) * c)
        else:
            p = jnp.exp(s - m_cur)
        l_blk = jnp.sum(p, axis=1, keepdims=True)
        pv = lax.dot_general(p.astype(jnp.bfloat16), kv, NN, preferred_element_type=jnp.float32)
        if first:
            l_scr[...] = l_blk
            acc_scr[...] = pv
        else:
            alpha = jnp.exp2((m_prev - m_cur) * c) if cfg.scale_fold else jnp.exp(m_prev - m_cur)
            l_scr[...] = alpha * l_scr[...] + l_blk
            acc_scr[...] = alpha * acc_scr[...] + pv
        m_scr[...] = m_cur

    row_tok = lax.broadcasted_iota(jnp.int32, (R, 1), 0) // H  # chunk-local token of each query row

    def chunk(r, k, swa_slot, carry):
        cu_r, qlen, pf, s0 = req_scalars(r)
        tok0 = cu_r + C * k
        nrow = jnp.minimum(C, qlen - C * k)
        off = jnp.minimum(tok0, T - C)  # rows [off, off + C) are computed; [tok0, tok0 + nrow) kept
        q = q_ref[pl.ds(off, C)].reshape(R, HEAD_DIM)
        qpos = pf + (off - cu_r) + row_tok  # [R, 1]
        vis = jnp.zeros((R, 1), jnp.int32)
        for j in range(C):
            vis = jnp.where(row_tok == j, ext_ref[off + j], vis)

        # --- sliding window: cached rows [s0, pf) then this step's rows of the request
        n_cached = pf - s0
        kc = _decode_tokens(swa_buf.at[swa_slot], W)
        tk = lax.broadcasted_iota(jnp.int32, (T, 1), 0)
        own = (tk >= cu_r) & (tk < cu_r + qlen)
        new = jnp.where(own, nkv_ref[...], jnp.zeros((), jnp.bfloat16))
        kv = jnp.concatenate([kc, new], axis=0)  # [W + T, 512]
        j = lax.broadcasted_iota(jnp.int32, (1, W + T), 1)
        cached = (j < n_cached) & (qpos - (s0 + j) < W)
        kt = j - W
        fresh = (j >= W) & (kt >= cu_r) & (kt < cu_r + qlen) & (pf + kt - cu_r <= qpos)
        if cfg.no_swa_compute:
            m_scr[...] = jnp.full(m_scr.shape, -1e30, jnp.float32)
            l_scr[...] = jnp.zeros(l_scr.shape, jnp.float32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, jnp.float32)
        else:
            update(q, kv, cached | fresh, first=True)

        # --- compressed rows, through the ring
        nc = vreq_nc(r, k)
        nb = n_blocks(nc)
        kb = lax.broadcasted_iota(jnp.int32, (1, B), 1)

        def blk_body(b, carry):
            g, cur = carry[0], carry[1:]
            slot = g % S
            hca_wait(nc, b, slot)
            issue(cur, (g + S - 1) % S)
            cur = advance(cur)
            if not cfg.no_hca_compute and cfg.compute_tile:
                CT = cfg.compute_tile
                n_live_rows = live_pages(nc, b) * HP
                kt_ = lax.broadcasted_iota(jnp.int32, (1, CT), 1)
                kr_ = lax.broadcasted_iota(jnp.int32, (CT, 1), 0)
                buf = kv_buf.at[slot]

                def sub(st, c):
                    r0 = pl.multiple_of(st * CT, CT)
                    kvt = _decode_tokens(buf.at[pl.ds(pl.multiple_of(2 * r0, 2 * CT), 2 * CT)], CT)
                    kvt = jnp.where(r0 + kr_ < n_live_rows, kvt, jnp.zeros((), kvt.dtype))
                    update(q, kvt, b * B + r0 + kt_ < vis, first=False)
                    return c

                lax.fori_loop(0, jnp.maximum(cdiv(jnp.minimum(n_live_rows, B), CT), 1), sub, 0)
            elif not cfg.no_hca_compute:
                kvb = _decode_tokens(kv_buf.at[slot], B)
                if cfg.trim:
                    kr = lax.broadcasted_iota(jnp.int32, (B, 1), 0)
                    kvb = jnp.where(kr < live_pages(nc, b) * HP, kvb, jnp.zeros((), kvb.dtype))
                if cfg.tail_mask_only:
                    edge = (b + 1) * B > jnp.min(jnp.where(row_tok < nrow, vis, jnp.int32(2**30)))

                    @pl.when(edge)
                    def _():
                        update(q, kvb, b * B + kb < vis, first=False)

                    @pl.when(~edge)
                    def _():
                        update(q, kvb, None, first=False)
                else:
                    update(q, kvb, b * B + kb < vis, first=False)
            return (g + 1,) + tuple(cur)

        carry = lax.fori_loop(0, nb, blk_body, carry)

        # --- finalize: sink, reciprocal, store rows [tok0, tok0 + nrow)
        sink = jnp.concatenate([sink_ref[...].astype(jnp.float32)] * C, axis=0)  # [R, 1]
        m = m_scr[...]
        if cfg.scale_fold:
            l_tot = l_scr[...] + jnp.exp(sink - m * cfg.sm_scale)
        else:
            l_tot = l_scr[...] + jnp.exp(sink - m)
        o = (acc_scr[...] * _recip(l_tot)).astype(out_ref.dtype).reshape(C, H, HEAD_DIM)
        tt = off + lax.broadcasted_iota(jnp.int32, (C, 1, 1), 0)
        keep = (tt >= tok0) & (tt < tok0 + nrow)
        out_ref[pl.ds(off, C)] = jnp.where(keep, o, out_ref[pl.ds(off, C)])
        return carry

    def req_body(r, carry):
        slot = r % 2
        swa_window(r, slot, wait=True)

        @pl.when(r + 1 < nreq)
        def _():
            swa_window(r + 1, 1 - slot, wait=False)

        swa_write(r, wait=False)
        _, qlen, _, _ = req_scalars(r)
        carry = lax.fori_loop(0, cdiv(qlen, C), lambda k, c: chunk(r, k, slot, c), carry)
        swa_write(r, wait=True)
        return carry

    @pl.when(nreq > 0)
    def _():
        swa_buf[...] = jnp.zeros(swa_buf.shape, swa_buf.dtype)  # never-written rows stay finite
        if cfg.no_hca_dma:
            kv_buf[...] = jnp.zeros(kv_buf.shape, kv_buf.dtype)
        wbuf[...] = _tokens_to_words(nkv_ref[...])
        swa_window(0, 0, wait=False)
        cur = (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(True))
        for j in range(S - 1):
            issue(cur, j)
            cur = advance(cur)
        lax.fori_loop(0, nreq, req_body, (jnp.int32(0),) + tuple(cur))

    # rows past the step's last token: zeros (never garbage into o_proj / MoE)
    def zero_row(t, c):
        out_ref[t] = jnp.zeros((H, HEAD_DIM), out_ref.dtype)
        return c

    lax.fori_loop(cu_ref[nreq], T, zero_row, 0)


def supported(*, tokens, heads, chunk, two_caches_same_buffer=False, swa_only=False,
              non_causal_block=False) -> bool:
    """Static gate for the fused call."""
    return (not swa_only and not two_caches_same_buffer and not non_causal_block
            and chunk <= tokens <= MAX_TOKENS and heads % 16 == 0)


def fused_hca_attention(
    q,  # bf16 [T, H, 512] (post qnorm_rope)
    new_kv,  # bf16 [T, 512] (post rope)
    sw_cache,  # int32 [N, 2 * swa_page, 128] words or uint8 [N, 2 * swa_page, 4, 128] slabs
    seq_lens,  # int32 [max_req] (incl. this step's rows)
    swa_page_indices,  # int32 [max_req, P_swa] (or flat)
    cu_q_lens,  # int32 [max_req + 1]
    distribution,  # int32 [3]; [2] = number of requests
    hca_cache,  # uint8 [N_h, 2 * hca_page, 4, 128] slabs or int32 [N_h, 2 * hca_page, 128] words
    kv_lens_to_attend,  # int32 [T]: compressed rows visible to each token
    hca_page_indices,  # int32 [max_req, P_hca] (or flat)
    attention_sinks,  # [H]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    chunk: int = 7,
    blk_pages: int = 128,
    interpret: bool = False,
    cs_rows=None,
    **knobs,
):
    """Returns (out bf16 [T, H, 512], sw_cache with this step's rows stored).

    With ``rope_in_kernel=True`` (stage B), ``q`` is the raw projection output in its token tiles
    (``rope.token_tiles(q_flat, H)``, from `fused_hca_attention_rope`) and ``new_kv`` the pre-RoPE kv;
    ``cs_rows`` [T, rotary_dim] f32 are the tokens' cos/sin rows."""
    rope_in = knobs.get("rope_in_kernel", False)
    if rope_in:
        T, H = q.shape[0] * q.shape[3], q.shape[1]
        D = q.shape[2] * q.shape[4]
    else:
        T, H, D = q.shape
    out_dtype = q.dtype
    assert D == HEAD_DIM, q.shape
    max_req = cu_q_lens.shape[0] - 1
    hca_page = hca_cache.shape[1] // 2
    num_hca_pages = hca_cache.shape[0]
    if hca_cache.dtype == jnp.int32:
        hca_cache = hca_cache.reshape(-1, LANES)
    swa_pt = swa_page_indices.reshape(max_req, -1).astype(jnp.int32)
    hca_pt = hca_page_indices.reshape(-1).astype(jnp.int32)
    hca_ppr = hca_pt.shape[0] // max_req
    cfg = Config(tokens=T, heads=H, chunk=chunk, window=sliding_window, swa_page=logical_page_size,
                 hca_page=hca_page, sm_scale=float(sm_scale), max_req=max_req, hca_ppr=hca_ppr,
                 num_hca_pages=num_hca_pages, blk_pages=min(blk_pages, hca_ppr),
                 **knobs)
    assert sw_cache.shape[1] == 2 * cfg.swa_page, (sw_cache.shape, cfg.swa_page)
    assert swa_page_indices.size % max_req == 0 and hca_page_indices.size % max_req == 0
    # Window pages of each request: [s0, pf + q_len) spans <= 3 logical pages.
    cu = cu_q_lens.astype(jnp.int32)
    pf = seq_lens.astype(jnp.int32) - (cu[1:] - cu[:-1])
    lp = jnp.maximum(pf - (cfg.window - 1), 0) // cfg.swa_page
    cols = jnp.minimum(lp[:, None] + jnp.arange(SWA_PAGES, dtype=jnp.int32)[None, :], swa_pt.shape[1] - 1)
    swa_win = jnp.take_along_axis(swa_pt, cols, axis=1).reshape(-1)
    sinks = attention_sinks.reshape(H, 1)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [smem] * 6 + [vmem, vmem, vmem] + ([vmem] if rope_in else []) + [hbm, hbm]
    scratch = [
        pltpu.VMEM((2, 2 * cfg.window, LANES), jnp.int32),  # SWA window ring
        pltpu.VMEM((2 * T, LANES), jnp.int32),  # this step's rows as cache words
        pltpu.VMEM((cfg.slots, 2 * cfg.blk_rows, LANES), jnp.int32),  # HCA block ring
        pltpu.VMEM((cfg.rows, 1), jnp.float32),  # m
        pltpu.VMEM((cfg.rows, 1), jnp.float32),  # l
        pltpu.VMEM((cfg.rows, HEAD_DIM), jnp.float32),  # acc
        pltpu.SemaphoreType.DMA((3 + cfg.slots,)),  # SWA window x2, SWA write, HCA ring x slots
    ]
    if rope_in:
        scratch += [pltpu.VMEM((T, H, HEAD_DIM), out_dtype), pltpu.VMEM((T, HEAD_DIM), new_kv.dtype)]
    operands = [distribution.astype(jnp.int32), seq_lens.astype(jnp.int32), cu,
                kv_lens_to_attend.astype(jnp.int32), swa_win, hca_pt, q, new_kv, sinks]
    if rope_in:
        operands.append(cs_rows)
    operands += [sw_cache, hca_cache]
    out, swa = pl.pallas_call(
        functools.partial(_kernel, cfg),
        out_shape=[jax.ShapeDtypeStruct((T, H, HEAD_DIM), out_dtype),
                   jax.ShapeDtypeStruct(sw_cache.shape, sw_cache.dtype)],
        in_specs=in_specs,
        out_specs=[vmem, hbm],
        input_output_aliases={len(operands) - 2: 1},
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fused_hca_decode_t{T}_h{H}_b{cfg.blk_rows}_s{cfg.slots}{'_rope' if rope_in else ''}",
    )(*operands)
    return out, swa


def fused_hca_attention_rope(q_flat, kv, positions, cos_sin_cache, *args, eps=1e-6, **kw):
    """Stage B (B2+B4): `fused_hca_attention` taking the raw q projection ``q_flat`` [T, H * 512] and the
    pre-RoPE kv [T, 512]; the per-head q RMSNorm + RoPE and the kv RoPE run in the kernel's prologue with
    the stock ops (rope.qnorm_rope_flat / rope.rope bodies). ``kw['heads']`` = H."""
    rope_lib = _rope_lib()
    H = kw.pop("heads")
    cs_rows = jnp.take(cos_sin_cache, positions, axis=0)  # layer-invariant: one gather per step in served
    q_tiles = rope_lib.token_tiles(q_flat, H)  # the projection's own tiling: an XLA bitcast
    return fused_hca_attention(q_tiles, kv, *args, cs_rows=cs_rows, rope_in_kernel=True, q_eps=eps, **kw)

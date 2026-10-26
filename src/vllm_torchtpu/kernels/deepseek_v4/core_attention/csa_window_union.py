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
"""Union-of-window CSA sparse attention for DeepSeek-V4 decode buckets.

The sparse path gathers every query token's own 1024 top-k compressed rows on
SparseCore and pushes them through the MXU one token at a time. The tokens of a
speculative verify window (1 + K tokens of one request) select heavily
overlapping rows (served union ~1.8k of 7168), so both the gather and the MXU
pushes scale with 7168 rows where ~1.8k distinct rows exist.

Here every scheduled request is cut into windows of at most `WQ` (8) tokens
(a verify window is one window). Per window:

* `union_gather` (one SparseCore call, all windows): dedups the window's top-k
  rows with 4-bit per-row counters (S/8 words; no row-count array, so any
  context up to max-model-len works without a fallback), ranks the union in row
  order, gathers the union rows into the window's region of a gathered buffer
  (`csa_gather` layout, 1024-row tiles), writes each token's keep bitmask over
  the union ranks and the union size.
* `csa_window_attention.window_attention` (one TensorCore call): attends each
  window's union tiles once for all its tokens (M = 8 x heads rows per pushed
  tile), each token keeping only its own rows, folded with the SWA partials and
  the sink. Same keys and values per token as the sparse kernel; only the
  summation order differs.

Interface (frozen between the two kernels):

  meta   i32[(1 + nw_cap + workers) * L]  row 0 lane 0 = nw (live windows);
         row 1 + w = (q0, qn, seq, S, ...) of window w (S = selectable rows);
         row 1 + nw_cap + wid = SparseCore worker wid's (w, j, G, magic) --
         the TensorCore reads rows 0 .. nw_cap only.
  nope_g i32[T * TILE, 128]  window w's union rank r at row q0_w * TILE + r.
  rope_g i32[T * TILE // 4, 128]  `csa_gather`'s paired layout per TILE-row
         period: viewed as bf16[T * TILE // 2, 128], row g * TILE/2 + i holds
         rank (g - q0_w) * TILE + i in lanes 0:64 and rank + TILE/2 in 64:128.
  mask   i32[T, W]  token t's keep bit of rank r: word (r >> 12) * 128 +
         (r & 127), bit (r >> 7) & 31 (the `csa_union_dense` layout).
  ucnt   i32[nw_cap * L]  lane 0 of row w = U_w, the window's union size.
Rows of nope_g / rope_g at ranks >= U_w, and mask rows of tokens outside every
window, are not written. A token's valid top-k ids must be distinct (the
indexer's exact top-k emits distinct column indices): the 4-bit counters and
the mask bits are built with adds.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 128
TILE = 1024  # union rows per KV tile == the gathered buffer's period
WQ = 8  # tokens per window
TOPK = 1024
W_WORDS = 256  # mask words per token: ranks < 8192 >= WQ_MAX_U
MAX_U = 7 * TOPK  # a verify window of 7 tokens; an 8-token chunk can reach 8192
WORD_SHIFT = 3  # 8 rows per counter word (4-bit counters, <= 8 tokens)
NBUF = 4  # SparseCore gather ring depth


def cdiv(a, b):
    return (a + b - 1) // b


def sc_geometry():
    sc = pltpu.get_tpu_info().sparse_core
    assert sc is not None, "SparseCore info is missing."
    return sc.num_lanes, sc.num_cores, sc.num_subcores


def nw_cap_for(num_tokens: int, max_num_seqs: int, wq: int = WQ) -> int:
    return cdiv(num_tokens, wq) + max_num_seqs


def mask_words(wq: int = WQ) -> int:
    """Mask words per token: ranks < wq x TOPK (4096 ranks per 128 words)."""
    return max(W_WORDS, cdiv(wq * TOPK, 4096) * LANES)


def supported(num_tokens: int, max_num_seqs: int) -> bool:
    """Trace-time route condition: one SparseCore worker per window slot is
    needed (T <= 128 with max_num_seqs <= 16 fits the 32 workers of v7x)."""
    L, nc, ns = sc_geometry()
    return num_tokens % 8 == 0 and nw_cap_for(num_tokens, max_num_seqs) <= nc * ns


def waves_for(num_tokens: int, max_num_seqs: int, workers: int) -> int:
    """SparseCore assignment waves: 1 when every window slot has a worker (the
    decode layout), else ceil(nw_cap / workers) (mixed buckets)."""
    return cdiv(nw_cap_for(num_tokens, max_num_seqs), workers)


def supported_mixed(num_tokens: int, max_num_seqs: int, max_tokens: int = 2048) -> bool:
    """Trace-time route condition for the mixed-bucket route (waves of windows;
    the gathered buffer is T x 1024 rows, so T is capped)."""
    return num_tokens % 8 == 0 and num_tokens <= max_tokens


# ----------------------------------------------------------------------------
# Window metadata (XLA; layer-invariant -> CSE'd once per step in the model).


def window_meta(cu_q_lens, distribution, kv_lens, *, num_tokens, s_cap, lanes=None,
                workers=None, max_team=None):
    """meta i32[(1 + nw_cap + workers) * L] (see the module docstring)."""
    if lanes is None or workers is None:
        L, nc, ns = sc_geometry()
        lanes, workers = L, nc * ns
    L = lanes
    # max_num_seqs as the sparse kernel takes it (cu_q_lens has one more entry).
    nseq = cu_q_lens.shape[0] - 1
    kv_lens = kv_lens[:nseq]
    nw_cap = nw_cap_for(num_tokens, nseq)
    waves = cdiv(nw_cap, workers)
    seq_ids = jnp.arange(nseq, dtype=jnp.int32)
    qlen = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_ids < distribution[2]) & (qlen > 0)
    nwin = jnp.where(live, (qlen + WQ - 1) // WQ, 0)
    wend = jnp.cumsum(nwin)
    nw = wend[-1]
    b = jnp.arange(nw_cap, dtype=jnp.int32)
    seq_b = jnp.clip(jnp.searchsorted(wend, b, side="right"), 0, nseq - 1).astype(jnp.int32)
    j = b - (wend[seq_b] - nwin[seq_b])
    q0 = cu_q_lens[seq_b] + j * WQ
    qn = jnp.where(b < nw, jnp.clip(cu_q_lens[seq_b + 1] - q0, 0, WQ), 0)
    q0 = jnp.where(qn > 0, q0, 0)
    S = jnp.where(qn > 0, jnp.minimum(kv_lens[seq_b] + 1, s_cap), 0)
    wrows = jnp.zeros((nw_cap, L), jnp.int32).at[:, :4].set(
        jnp.stack([q0, qn, seq_b, S], axis=1).astype(jnp.int32))
    # SparseCore assignment: G = workers // nw subcores per window, worker
    # wid -> (window wid // G, member wid % G); magic = ceil(2^16 / G) lets the
    # kernel divide small counts by G with a multiply and a shift.
    if waves > 1:
        arows = _wave_rows(nw, q0, qn, seq_b, S, nw_cap=nw_cap, workers=workers, waves=waves,
                           L=L, max_team=max_team)
        row0 = jnp.zeros((1, L), jnp.int32).at[0, 0].set(nw)
        return jnp.concatenate([row0, wrows, arows], axis=0).reshape(-1)
    G = jnp.maximum(workers // jnp.maximum(nw, 1), 1)
    if max_team is not None:
        G = jnp.minimum(G, max_team)
    wid = jnp.arange(workers, dtype=jnp.int32)
    aw = wid // G
    aj = wid % G
    magic = (65536 + G - 1) // G
    # Lanes 4..8 repeat the worker's window row and nw, so a worker needs one
    # 16-word DMA (version >= 2) instead of the whole meta.
    wsafe = jnp.minimum(aw, nw_cap - 1)
    live_w = aw < nw
    arows = jnp.zeros((workers, L), jnp.int32).at[:, :9].set(
        jnp.stack([aw, aj, jnp.full_like(wid, G), jnp.full_like(wid, magic),
                   jnp.where(live_w, q0[wsafe], 0), jnp.where(live_w, qn[wsafe], 0),
                   jnp.where(live_w, seq_b[wsafe], 0), jnp.where(live_w, S[wsafe], 0),
                   jnp.full_like(wid, nw)], axis=1).astype(jnp.int32))
    row0 = jnp.zeros((1, L), jnp.int32).at[0, 0].set(nw)
    return jnp.concatenate([row0, wrows, arows], axis=0).reshape(-1)


def _wave_rows(nw, q0, qn, seq_b, S, *, nw_cap, workers, waves, L, max_team=None, slot=None):
    """Assignment rows of `waves` waves (i32[waves * workers, L]): wave k takes
    windows [k * workers, k * workers + n_k), n_k = clip(nw - k * workers, 0,
    workers), G_k = workers // n_k subcores per window (a short last wave splits
    its windows over teams, as the decode layout does). Dead assignments get
    window nw_cap (>= nw: skipped)."""
    rows = []
    wid = jnp.arange(workers, dtype=jnp.int32)
    for k in range(waves):
        n_k = jnp.clip(nw - k * workers, 0, workers)
        G = jnp.maximum(workers // jnp.maximum(n_k, 1), 1)
        if max_team is not None:
            G = jnp.minimum(G, max_team)
        loc = wid // G
        live_w = loc < n_k
        aw = jnp.where(live_w, k * workers + loc, nw_cap)
        aj = wid % G
        magic = (65536 + G - 1) // G
        wsafe = jnp.minimum(aw, nw_cap - 1)
        cols = [aw, aj, jnp.full_like(wid, G), jnp.full_like(wid, magic),
                jnp.where(live_w, q0[wsafe], 0), jnp.where(live_w, qn[wsafe], 0),
                jnp.where(live_w, seq_b[wsafe], 0), jnp.where(live_w, S[wsafe], 0),
                jnp.full_like(wid, nw)]
        if slot is not None:
            cols.append(jnp.where(live_w, slot[wsafe], 0))
        rows.append(jnp.zeros((workers, L), jnp.int32).at[:, :len(cols)].set(
            jnp.stack(cols, axis=1).astype(jnp.int32)))
    return jnp.concatenate(rows, axis=0)


# ----------------------------------------------------------------------------
# Host reference (numpy): what union_gather must produce.


def mask_word_bit(rank):
    return (rank >> 12) * LANES + (rank & (LANES - 1)), (rank >> 7) & 31


def plan_host(topk, cu_q_lens, distribution, kv_lens, page_table, page_size, *, s_cap, wq=WQ):
    """Reference union plan. Returns dict: windows [(q0, qn, seq, S)], U list,
    urow list of int arrays (physical rows by rank), mask u32[T, W_WORDS],
    gather_idx i32[T * TILE] (physical row per gathered-buffer row; rows past a
    window's union spread over its own rows, unused rows 0)."""
    T = topk.shape[0]
    nseq = len(kv_lens)
    wins, Us, urows = [], [], []
    mask = np.zeros((T, mask_words(wq)), np.uint32)
    gidx = np.zeros((T * TILE,), np.int64)
    for s in range(int(distribution[2])):
        a, b = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        if b <= a:
            continue
        S = min(int(kv_lens[s]) + 1, s_cap)
        for q0 in range(a, b, wq):
            qn = min(wq, b - q0)
            sel = topk[q0:q0 + qn]
            v = sel[(sel >= 0) & (sel < S)]
            u = np.unique(v)
            rank = {int(p): r for r, p in enumerate(u)}
            for t in range(q0, q0 + qn):
                row = topk[t]
                for p in row[(row >= 0) & (row < S)]:
                    wd, bt = mask_word_bit(rank[int(p)])
                    mask[t, wd] |= np.uint32(1) << np.uint32(bt)
            phys = page_table[s, u // page_size] * page_size + u % page_size
            wins.append((q0, qn, s, S))
            Us.append(len(u))
            urows.append(phys.astype(np.int64))
            if len(u):
                nt = cdiv(len(u), TILE)
                reg = np.concatenate([phys, phys[np.arange(nt * TILE - len(u)) % len(u)]])
                gidx[q0 * TILE:q0 * TILE + nt * TILE] = reg
    del nseq
    return dict(windows=wins, U=Us, urow=urows, mask=mask.view(np.int32), gather_idx=gidx.astype(np.int32))


def reference_union_gather(cache_kv_nope, cache_kv_rope, plan, num_tokens):
    """L1 reference for testing / component arms: the host plan's rows gathered
    with the existing `csa_gather` kernel into the frozen layout."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

    nope_g, rope_g = csa_gather.csa_gather(
        cache_kv_nope, cache_kv_rope, jnp.asarray(plan["gather_idx"]), top_k=TILE)
    return nope_g, rope_g, jnp.asarray(plan["mask"])


# ----------------------------------------------------------------------------
# L1: SparseCore union + gather (one call for every window of the step).


def _union_gather_kernel(
    meta_hbm,  # i32[(1 + nw_cap + workers) * L]
    topk_hbm,  # i32[T * K]
    pages_hbm,  # i32[max_num_seqs * pps + pps_dma]
    nope_hbm,  # i32[rows, 128]
    rope_hbm,  # i32[rows, 32]
    nope_out,  # i32[T * TILE, 128]
    rope_out,  # i32[T * TILE // 4, 128]
    mask_out,  # i32[T * W_WORDS]
    ucnt_out,  # i32[nw_cap * L]
    metab,  # VMEM i32[(1 + nw_cap + workers) * L]
    tkb,  # VMEM i32[WQ * K]
    pageb,  # VMEM i32[pps_pad]
    cnt,  # VMEM i32[s_words + L]: 4-bit per-row counters, then nonzero nibbles
    base,  # VMEM i32[s_words + L]: union rank of each word's first row
    urow,  # VMEM i32[WQ * K + L]: physical row by union rank
    mrow,  # VMEM i32[W_WORDS]
    ubuf,  # VMEM i32[L]
    gnope,  # VMEM i32[NBUF, L, 128]
    grope,  # VMEM i32[NBUF, L, 32]
    rbuf,  # VMEM i32[NBUF, L // 2, 64]
    sg,  # DMA[2, NBUF] gathers (nope, rope)
    so,  # DMA[2, NBUF] writes (nope, rope)
    sl,  # DMA[2] top-k rows, page-table row
    *,
    nw_cap: int,
    K: int,
    pps: int,
    pps_dma: int,
    page_size: int,
    L: int,
    num_subcores: int,
    skip_gather: bool,
    stop_after: str | None = None,
    version: int = 0,
    nbuf: int = NBUF,
    sb_major_T: int = 0,
    nz_shifts: int = 4,
    waves: int = 1,
    workers: int = 32,
    slot_meta: bool = False,
    wq: int = WQ,
):
    WQ = wq  # noqa: N806 -- 8, or 16 on the mixed route (two nibble count halves)
    W_WORDS = mask_words(wq)  # noqa: N806
    # version 0: as first measured (j-6e9128d7). version 1: top-k / page-table
    # DMAs concurrent with the zero pass, zero pass 8 stores per iteration,
    # unrolled parallel loops, a deeper gather ring (nbuf 8).
    unroll = 4 if version >= 1 else 1
    # version 2: one 16-word meta DMA per worker (its assignment row carries
    # its window's q0 / qn / seq / S and nw).
    page_shift = page_size.bit_length() - 1
    assert 1 << page_shift == page_size
    assert L == 16 and TILE % (2 * L) == 0
    srl = lax.shift_right_logical
    wid = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    zeros = jnp.zeros((L,), jnp.int32)
    ones = jnp.ones((L,), jnp.int32)

    def _wave(v):
        # `waves` > 1 (mixed buckets, more windows than workers): wave v's
        # assignment rows follow wave v - 1's; waves == 1 is the decode layout.
        if version >= 2:
            pltpu.sync_copy(meta_hbm.at[pl.ds((1 + nw_cap + v * workers + wid) * L, L)],
                            metab.at[pl.ds(0, L)])
            a = metab[pl.ds(0, L)]
            nw = a[8]
        else:
            pltpu.sync_copy(meta_hbm, metab)
            nw = metab[pl.ds(0, L)][0]
            a = metab[pl.ds((1 + nw_cap + v * workers + wid) * L, L)]
        w, j, G, magic = a[0], a[1], a[2], a[3]

        def div_g(x):  # floor(x / G) for 0 <= x < 2^16 / 32
            return srl(x * magic, 16)

        @pl.when(w < nw)
        def _window():
            if version >= 2:
                q0, qn, seq, S = a[4], a[5], a[6], a[7]
            else:
                m = metab[pl.ds((1 + w) * L, L)]
                q0, qn, seq, S = m[0], m[1], m[2], m[3]
            p_lo = srl(seq * pps, 3) * 8  # 32-byte-aligned superset of the row
            p_off = seq * pps - p_lo
            tk_dma = pltpu.make_async_copy(
                topk_hbm.at[pl.ds(q0 * K, qn * K)], tkb.at[pl.ds(0, qn * K)], sl.at[0])
            pg_dma = pltpu.make_async_copy(
                pages_hbm.at[pl.ds(p_lo, pps_dma)], pageb.at[pl.ds(0, pps_dma)], sl.at[1])
            tk_dma.start()
            pg_dma.start()
            if version == 0:
                tk_dma.wait()
                pg_dma.wait()
            nvec = srl(srl(S + 7, WORD_SHIFT) + L - 1, 4)
            if stop_after == "load":
                if version >= 1:
                    tk_dma.wait()
                    pg_dma.wait()
                return

            # Sequential: zero stores under a parallel_loop are not ordered against
            # the scatters that follow (csa_union_dense).
            if version == 0:
                @pl.loop(0, nvec)
                def _(i):
                    cnt[pl.ds(i * L, L)] = zeros
            else:
                @pl.loop(0, srl(nvec + 7, 3))
                def _(i):
                    for u in range(8):
                        cnt[pl.ds((i * 8 + u) * L, L)] = zeros

                tk_dma.wait()
                pg_dma.wait()

            def ids(t, k):
                c = tkb[pl.ds(t * K + k, L)]
                v = (c >= 0) & (c < S)
                return jnp.where(v, c, 0), v

            # Count pass: +1 in row c's nibble. At most WQ tokens select a row and
            # a token's top-k rows are distinct, so a nibble never overflows.
            def count(t0, t1):
                @pl.loop(t0, t1)
                def _(t):
                    @plsc.parallel_loop(0, K, step=L, unroll=unroll)
                    def _(k):
                        c, v = ids(t, k)
                        plsc.addupdate_scatter(
                            cnt, (srl(c, WORD_SHIFT),), jnp.left_shift(ones, (c & 7) * 4), mask=v
                        )

            if WQ <= 8:
                count(0, qn)
            else:
                # 16-token windows: a nibble holds a count <= 15. Count tokens
                # [0, 8), squash every nibble to 0/1 (sequential: ordered against
                # the scatters on both sides), then count [8, qn): <= 1 + 8.
                count(0, jnp.minimum(qn, 8))

                @pl.loop(0, nvec)
                def _(i):
                    x = cnt[pl.ds(i * L, L)]
                    cnt[pl.ds(i * L, L)] = (x | srl(x, 1) | srl(x, 2) | srl(x, 3)) & 0x11111111

                count(8, jnp.maximum(qn, 8))

            if stop_after == "count":
                return

            # Prefix pass: per word the nonzero-nibble flags (bit 4i) and the rank
            # of its first union row; U = total.
            def prefix(i, carry):
                x = cnt[pl.ds(i * L, L)]
                # counts reach WQ = 8 (0b1000): all four nibble bits matter
                nz = x
                for sh in range(1, nz_shifts):  # 4 (test-only 3 reproduces the count-8 bug)
                    nz = nz | srl(x, sh)
                nz = nz & 0x11111111
                pc = srl(nz * 0x11111111, 28)
                cs = plsc.cumsum(pc)
                base[pl.ds(i * L, L)] = cs - pc + carry
                cnt[pl.ds(i * L, L)] = nz
                return carry + cs[L - 1]

            U = plsc.parallel_loop(0, nvec, unroll=2 if version >= 1 else 1, carry=jnp.int32(0))(prefix)

            if stop_after == "prefix":
                ubuf[...] = jnp.full((L,), U, jnp.int32)
                pltpu.sync_copy(ubuf, ucnt_out.at[pl.ds(w * L, L)])
                return

            def rank_of(c, v):
                wi = srl(c, WORD_SHIFT)
                nzw = plsc.load_gather(cnt, [wi], mask=v)
                bw = plsc.load_gather(base, [wi], mask=v)
                below = nzw & (jnp.left_shift(ones, (c & 7) * 4) - 1)
                return jnp.where(v, bw + srl(below * 0x11111111, 28), 0)

            # Union rows by rank (every token; duplicates write equal values).
            @pl.loop(0, qn)
            def _(t):
                @plsc.parallel_loop(0, K, step=L, unroll=unroll)
                def _(k):
                    c, v = ids(t, k)
                    r = rank_of(c, v)
                    pg = plsc.load_gather(pageb, [srl(c, page_shift) + p_off], mask=v)
                    phys = jnp.left_shift(pg, page_shift) | (c & (page_size - 1))
                    plsc.store_scatter(urow, [r], phys, mask=v)

            if stop_after == "rank":
                return

            # Keep bitmasks of this member's tokens t = j, j + G, ...
            ntok = jnp.where(j < qn, div_g(qn - j + G - 1), 0)

            @pl.loop(0, ntok)
            def _(kt):
                t = j + kt * G

                @pl.loop(0, W_WORDS // L)
                def _(i):
                    mrow[pl.ds(i * L, L)] = zeros

                @plsc.parallel_loop(0, K, step=L, unroll=unroll)
                def _(k):
                    c, v = ids(t, k)
                    r = rank_of(c, v)
                    word = jnp.left_shift(srl(r, 12), 7) | (r & (LANES - 1))
                    bit = srl(r, 7) & 31
                    plsc.addupdate_scatter(mrow, (word,), jnp.left_shift(ones, bit), mask=v)

                if sb_major_T:  # mask laid out [W / 128, T, 128]: super-block major
                    for sb in range(W_WORDS // LANES):
                        pltpu.sync_copy(
                            mrow.at[pl.ds(sb * LANES, LANES)],
                            mask_out.at[pl.ds((sb * sb_major_T + q0 + t) * LANES, LANES)],
                        )
                else:
                    pltpu.sync_copy(mrow, mask_out.at[pl.ds((q0 + t) * W_WORDS, W_WORDS)])

            @pl.when(j == 0)
            def _():
                ubuf[...] = jnp.full((L,), U, jnp.int32)
                pltpu.sync_copy(ubuf, ucnt_out.at[pl.ds(w * L, L)])

            if skip_gather:
                return

            # Pad the last 16-row chunk with distinct real rows of the window.
            @pl.when(U > 0)
            def _():
                u0 = srl(U, 4) * L
                r = u0 + lax.broadcasted_iota(jnp.int32, (L,), 0)
                src = jnp.where(r < U, r, lax.rem(r - U, U))
                urow[pl.ds(u0, L)] = plsc.load_gather(urow, [src])

            # Gather this member's contiguous share of the union's 16-row chunks.
            nch = srl(U + L - 1, 4)
            cpm = div_g(nch + G - 1)
            c0 = jnp.minimum(j * cpm, nch)
            n = jnp.minimum(c0 + cpm, nch) - c0
            ts = a[9] if slot_meta else q0  # first gathered tile of this window
            row0 = ts * TILE

            def g_descs(i, s):
                off = pl.multiple_of((c0 + i) * L, L)
                idx = urow.at[pl.ds(off, L)]
                return (
                    pltpu.make_async_copy(nope_hbm.at[idx], gnope.at[s], sg.at[0, s]),
                    pltpu.make_async_copy(rope_hbm.at[idx], grope.at[s], sg.at[1, s]),
                )

            def o_descs(i, s, half=0):
                r0 = (c0 + i) * L
                rr = (ts + srl(r0, 10)) * (TILE // 4) + srl(r0 & (TILE // 2 - 1), 1)
                return (
                    pltpu.make_async_copy(
                        gnope.at[s], nope_out.at[pl.ds(row0 + r0, L)], so.at[0, s]
                    ),
                    pltpu.make_async_copy(
                        rbuf.at[s],
                        rope_out.at[pl.ds(rr, L // 2), pl.ds(half * 64, 64)],
                        so.at[1, s],
                    ),
                )

            def issue(i, s):
                for d in g_descs(i, s):
                    d.start()

            for s in range(nbuf - 1):
                @pl.when(s < n)
                def _():
                    issue(s, s)

            def chunk(i, s):
                for d in g_descs(i, s):
                    d.wait()
                on, _ = o_descs(i, s)
                on.start()
                # bf16 rope rows 2t / 2t + 1 = low / high halves of int32 row t;
                # cache word k holds values k (low) and k + 32 (high).
                for t in range(L // 2 if stop_after != "norope" else 0):
                    for hh in range(2):
                        e = grope[s, 2 * t, pl.ds(16 * hh, 16)]
                        o = grope[s, 2 * t + 1, pl.ds(16 * hh, 16)]
                        rbuf[s, t, pl.ds(16 * hh, 16)] = (e & 0xFFFF) | jnp.left_shift(o, 16)
                        rbuf[s, t, pl.ds(32 + 16 * hh, 16)] = srl(e, 16) | (o & jnp.int32(-65536))
                half = srl((c0 + i) * L, 9) & 1
                for hv in range(2):
                    @pl.when(half == hv)
                    def _():
                        o_descs(i, s, hv)[1].start()

                ps = (s - 1) % nbuf

                @pl.when(i >= 1)
                def _():
                    for d in o_descs(i - 1, ps):
                        d.wait()

                @pl.when(i + nbuf - 1 < n)
                def _():
                    issue(i + nbuf - 1, ps)

            @pl.loop(0, srl(n + nbuf - 1, nbuf.bit_length() - 1))
            def _(ip):
                for s in range(nbuf):
                    @pl.when(ip * nbuf + s < n)
                    def _():
                        chunk(ip * nbuf + s, s)

            for s in range(nbuf):
                @pl.when(jnp.logical_and(n > 0, (n - 1) % nbuf == s))
                def _():
                    for d in o_descs(n - 1, s):
                        d.wait()

    if waves == 1:
        _wave(0)
    else:
        @pl.loop(0, waves)
        def _(v):
            _wave(v)


@functools.partial(
    jax.jit,
    static_argnames=("num_tokens", "max_num_seqs", "skip_gather", "stop_after", "version",
                     "mask_sb_major", "_nz_shifts", "n_tiles", "nw_cap_override", "wq"),
)
def union_gather(
    meta: jax.Array,
    topk_indices: jax.Array,  # i32[T, K] seq-local compressed rows, -1 padded
    page_indices: jax.Array,  # i32[max_num_seqs * pps]
    cache_kv_nope: jax.Array,  # i32[pages, page_size, 128]
    cache_kv_rope: jax.Array,  # i32[pages, page_size // 4, 128]
    *,
    num_tokens: int,
    max_num_seqs: int,
    skip_gather: bool = False,
    stop_after: str | None = None,
    version: int = 2,
    mask_sb_major: bool = False,
    _nz_shifts: int = 4,
    n_tiles: int | None = None,
    nw_cap_override: int | None = None,
    wq: int = WQ,
):
    """L1. Returns nope_g, rope_g, mask i32[T, W_WORDS], ucnt (module docstring).

    `skip_gather`: timing-only removal arm (dedup, ranks and masks; no rows)."""
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

    T, K = topk_indices.shape
    assert T == num_tokens and K == TOPK
    assert num_tokens * TILE // 4 * 4 == num_tokens * TILE
    L, nc, ns = sc_geometry()
    workers = nc * ns
    # n_tiles / nw_cap_override: one group of a grouped mixed step (`group_meta`):
    # the gathered buffer holds n_tiles tiles, window w's at its slot (lane 9).
    nw_cap = nw_cap_for(T, max_num_seqs) if nw_cap_override is None else nw_cap_override
    waves = cdiv(nw_cap, workers)
    slot_meta = n_tiles is not None
    n_tiles = T if n_tiles is None else n_tiles
    assert meta.shape == ((1 + nw_cap + waves * workers) * L,), meta.shape
    page_size = cache_kv_nope.shape[1]
    pps = page_indices.shape[0] // max_num_seqs
    s_cap = pps * page_size
    s_words = cdiv(cdiv(s_cap, 8), 8 * L) * 8 * L
    nbuf = 8 if version >= 1 else NBUF
    pps_dma = cdiv(pps + 8, 8) * 8
    pps_pad = cdiv(pps_dma, L) * L
    page_indices = jnp.pad(page_indices, (0, pps_dma))
    nope = cache_kv_nope.reshape(-1, csa_cache_layout.ROW_WORDS)
    rope = cache_kv_rope.reshape(-1, csa_cache_layout.ROPE_WORDS)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=nc, num_subcores=ns, core_axis_name="core", subcore_axis_name="subcore"
    )
    nope_g, rope_g, mask, ucnt = pl.kernel(
        functools.partial(
            _union_gather_kernel,
            nw_cap=nw_cap,
            K=K,
            pps=pps,
            pps_dma=pps_dma,
            page_size=page_size,
            L=L,
            num_subcores=ns,
            skip_gather=skip_gather,
            stop_after=stop_after,
            version=version,
            nbuf=nbuf,
            sb_major_T=T if mask_sb_major else 0,
            nz_shifts=_nz_shifts,
            waves=waves,
            workers=workers,
            slot_meta=slot_meta,
            **({} if wq == WQ else {"wq": wq}),
        ),
        out_type=(
            jax.ShapeDtypeStruct((n_tiles * TILE, LANES), jnp.int32),
            jax.ShapeDtypeStruct((n_tiles * TILE // 4, LANES), jnp.int32),
            jax.ShapeDtypeStruct((T * mask_words(wq),), jnp.int32),
            jax.ShapeDtypeStruct((nw_cap * L,), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM(meta.shape, jnp.int32),
            pltpu.VMEM((wq * K,), jnp.int32),
            pltpu.VMEM((pps_pad,), jnp.int32),
            pltpu.VMEM((s_words + L,), jnp.int32),
            pltpu.VMEM((s_words + L,), jnp.int32),
            pltpu.VMEM((wq * K + L,), jnp.int32),
            pltpu.VMEM((mask_words(wq),), jnp.int32),
            pltpu.VMEM((L,), jnp.int32),
            pltpu.VMEM((nbuf, L, LANES), jnp.int32),
            pltpu.VMEM((nbuf, L, 32), jnp.int32),
            pltpu.VMEM((nbuf, L // 2, 64), jnp.int32),
            pltpu.SemaphoreType.DMA((2, nbuf)),
            pltpu.SemaphoreType.DMA((2, nbuf)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        mesh=mesh,
        name="sc_csa_window_union_gather",
    )(meta, topk_indices.reshape(-1), page_indices, nope, rope)
    if mask_sb_major:
        # A 128-lane int32 array's (8, 128) tiling is row-major, so this view
        # is free for T % 8 == 0: mask[sb, t] = words [128 sb, 128 sb + 128).
        return nope_g, rope_g, mask.reshape(mask_words(wq) // LANES, T, LANES), ucnt
    return nope_g, rope_g, mask.reshape(T, mask_words(wq)), ucnt


def group_meta(cu_q_lens, distribution, kv_lens, *, num_tokens, s_cap, groups, g, wq=WQ):
    """Meta of group g of `groups` (windows [g * nw // groups, (g + 1) * nw // groups)
    of the step), for a grouped mixed step: the SparseCore call of group g + 1
    runs under the TensorCore call of group g. Window rows are the group's
    (local index); window i's gathered tiles start at slot i * MAX_TILES of the
    group's buffer (lane 9 of its assignment row, `tslot` for the TensorCore).
    Returns (meta, tslot i32[T], nwc) with nwc the group's window capacity."""
    L, nc, ns = sc_geometry()
    workers = nc * ns
    nseq = cu_q_lens.shape[0] - 1
    kv_lens = kv_lens[:nseq]
    nw_cap = nw_cap_for(num_tokens, nseq, wq)
    nwc = cdiv(nw_cap, groups) + 1
    seq_ids = jnp.arange(nseq, dtype=jnp.int32)
    qlen = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_ids < distribution[2]) & (qlen > 0)
    nwin = jnp.where(live, (qlen + wq - 1) // wq, 0)
    wend = jnp.cumsum(nwin)
    nw = wend[-1]
    lo = (g * nw) // groups
    hi = ((g + 1) * nw) // groups
    n_g = hi - lo
    b = lo + jnp.arange(nwc, dtype=jnp.int32)
    seq_b = jnp.clip(jnp.searchsorted(wend, b, side="right"), 0, nseq - 1).astype(jnp.int32)
    j = b - (wend[seq_b] - nwin[seq_b])
    q0 = cu_q_lens[seq_b] + j * wq
    qn = jnp.where(b < hi, jnp.clip(cu_q_lens[seq_b + 1] - q0, 0, wq), 0)
    q0 = jnp.where(qn > 0, q0, 0)
    S = jnp.where(qn > 0, jnp.minimum(kv_lens[seq_b] + 1, s_cap), 0)
    slot = jnp.arange(nwc, dtype=jnp.int32) * wq  # a window has <= wq x TOPK union rows = wq tiles
    wrows = jnp.zeros((nwc, L), jnp.int32).at[:, :4].set(
        jnp.stack([q0, qn, seq_b, S], axis=1).astype(jnp.int32))
    waves = cdiv(nwc, workers)
    arows = _wave_rows(n_g, q0, qn, seq_b, S, nw_cap=nwc, workers=workers, waves=waves, L=L, slot=slot)
    row0 = jnp.zeros((1, L), jnp.int32).at[0, 0].set(n_g)
    meta = jnp.concatenate([row0, wrows, arows], axis=0).reshape(-1)
    tslot = jnp.zeros((num_tokens,), jnp.int32).at[jnp.where(qn > 0, q0, num_tokens)].set(
        slot, mode="drop")
    return meta, tslot, nwc


MAX_TILES_MIXED = 8  # tiles per window slot (U <= 8192 = WQ x TOPK)


def _group_l1(args, g, groups, wq=WQ):
    topk_indices, kv_lens, page_indices, cu_q_lens, distribution, cache_kv_nope, cache_kv_rope = args
    T = topk_indices.shape[0]
    max_num_seqs = cu_q_lens.shape[0] - 1
    pps = page_indices.shape[0] // max_num_seqs
    s_cap = pps * cache_kv_nope.shape[1]
    meta, tslot, nwc = group_meta(cu_q_lens, distribution, kv_lens, num_tokens=T,
                                  s_cap=s_cap, groups=groups, g=g, wq=wq)
    nope_g, rope_g, mask, ucnt = union_gather(
        meta, topk_indices, page_indices, cache_kv_nope, cache_kv_rope,
        num_tokens=T, max_num_seqs=max_num_seqs, mask_sb_major=MASK_SB_MAJOR,
        n_tiles=nwc * wq, nw_cap_override=nwc, wq=wq)
    return dict(meta=meta, tslot=tslot, nope_g=nope_g, rope_g=rope_g, mask=mask, ucnt=ucnt)


def prepare_grouped(topk_indices, kv_lens, page_indices, cu_q_lens, distribution,
                    cache_kv_nope, cache_kv_rope, *, groups, wq=WQ):
    """L1 of group 0 of a grouped mixed step (issued ahead, like `prepare`); the
    other groups' SparseCore calls are issued by `attend_grouped`, each fenced
    behind the previous group's attention input so it runs under it (left
    free, XLA issues every SparseCore call first and the TensorCore waits on
    all of them before the first attention)."""
    args = (topk_indices, kv_lens, page_indices, cu_q_lens, distribution, cache_kv_nope,
            cache_kv_rope)
    # No static entries: the plan crosses an optimization_barrier in the op.
    return dict(first=_group_l1(args, 0, groups, wq), args=args)


def attend_grouped(q, plan, attention_sinks, swa_accumution, swa_l, swa_m, *, sm_scale, groups,
                   l2_config=None, wq=WQ):
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_attention

    L, nc, ns = sc_geometry()
    args = plan["args"]
    rows = plan["first"]
    for g in range(groups):
        nxt = None
        if g + 1 < groups:
            # Group g + 1's SparseCore call may start only once group g's
            # attention can (its input q), so it runs under that attention.
            q, topk = jax.lax.optimization_barrier((q, args[0]))
            nxt = _group_l1((topk,) + tuple(args[1:]), g + 1, groups, wq)
        q = csa_window_attention.window_attention(
            q, rows["nope_g"], rows["rope_g"], rows["mask"], rows["meta"], rows["ucnt"],
            attention_sinks, swa_accumution, swa_l, swa_m, sm_scale=sm_scale,
            nw_cap=rows["ucnt"].shape[0] // L, tslot=rows["tslot"],
            **(L2_CONFIG if l2_config is None else l2_config), **({} if wq == WQ else {"wq": wq}))
        rows = nxt
    return q


# ----------------------------------------------------------------------------
# Entry points used by the CSA attention op.


# L2 configuration attend() runs (L2 round 3, "v8rpsn"): tile pairs + one narrow
# (<= 512-row) last-tile body, operands pinned to HBM except l/m, SWA l/m
# broadcast in-kernel, super-block-major mask (no wrapper reshape/copy).
L2_CONFIG = dict(version=7, tails=(512,), lm_raw=True, hbm_pin=True, pin_lm=False)
MASK_SB_MAJOR = True


def prepare(topk_indices, kv_lens, page_indices, cu_q_lens, distribution,
            cache_kv_nope, cache_kv_rope, *, mask_sb_major=MASK_SB_MAJOR):
    """L1 for one CSA layer: window meta (layer-invariant XLA) + the SparseCore
    union/gather call. Needs no q and no SWA output, so the caller can issue it
    ahead of the SWA pass."""
    T = topk_indices.shape[0]
    max_num_seqs = cu_q_lens.shape[0] - 1
    pps = page_indices.shape[0] // max_num_seqs
    s_cap = pps * cache_kv_nope.shape[1]
    meta = window_meta(cu_q_lens, distribution, kv_lens, num_tokens=T, s_cap=s_cap)
    nope_g, rope_g, mask, ucnt = union_gather(
        meta, topk_indices, page_indices, cache_kv_nope, cache_kv_rope,
        num_tokens=T, max_num_seqs=max_num_seqs, mask_sb_major=mask_sb_major)
    return dict(meta=meta, nope_g=nope_g, rope_g=rope_g, mask=mask, ucnt=ucnt)


def attend(q, rows, attention_sinks, swa_accumution, swa_l, swa_m, *, sm_scale, l2_config=None):
    """L2: every live window's tokens attend its union rows (+ SWA partials,
    sink); rows outside every window return q."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_attention

    L, nc, ns = sc_geometry()
    nw_cap = rows["ucnt"].shape[0] // L
    return csa_window_attention.window_attention(
        q, rows["nope_g"], rows["rope_g"], rows["mask"], rows["meta"], rows["ucnt"],
        attention_sinks, swa_accumution, swa_l, swa_m, sm_scale=sm_scale,
        nw_cap=nw_cap, **(L2_CONFIG if l2_config is None else l2_config))

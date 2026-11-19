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
"""GQA attention over the KV worklist: per-item scores and online softmax."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, F32, FP8, I32, U32
from ...pool.config import cache_exec
from .context import GQAEmit
from .kv import KVRing
from .params import NN, NT


def item_front(cx: GQAEmit, kv: KVRing, col: Any, col_v: Any, i: Any) -> Any:
    """Scores, row maxima and V tiles of worklist item `i` (per sub-block)."""
    ex, ap, scr, pos_ref, stream = cx.ex, cx.ap, cx.scr, cx.pos_ref, cx.stream
    interpret, KR = cx.interpret, cx.KR
    T, VER = ex.T, ex.VER
    KVE, BLK, SUB, AMASK, HD = ap.KVE, ap.BLK, ap.SUB, ap.AMASK, ap.HD
    kv_slot, kvbuf, kbuf, vbuf = kv.slot, kv.kvbuf, kv.kbuf, kv.vbuf
    s = i % KR
    b, t = scr["wl_b"][i], scr["wl_t"][i]
    plen = pos_ref[b * T] if VER else pos_ref[b]
    qrows = scr["qb"][b]
    k_full: Any = None
    v_full: Any = None
    v_u32: Any = None
    if cache_exec(ex).paged_kv:
        if interpret:
            kst = kv_slot(s) if KVE else kvbuf.at[s]
            k_u32 = lax.bitcast_convert_type(jnp.swapaxes(kst[0], 1, 2), U32)
            v_u32 = lax.bitcast_convert_type(jnp.swapaxes(kst[1], 1, 2), U32)
        elif KVE:
            kst = kv_slot(s)
            k_u32 = kst[0]
            v_u32 = kst[1]
        else:
            k_u32 = kvbuf.at[s, 0].bitcast(U32).reshape(HD // 4, BLK)[...]
            v_u32 = kvbuf.at[s, 1].bitcast(U32).reshape(HD // 4, BLK)[...]
        k_full = pltpu.bitcast(k_u32, FP8)
        if AMASK in (0, 3, 4):
            v_full = pltpu.bitcast(v_u32, FP8)
    outs = []
    for j in range(BLK // SUB):
        off = t * BLK + j * SUB
        rem = plen - off
        full_blk = off + SUB <= plen
        if cache_exec(ex).paged_kv:
            kj = k_full if SUB == BLK else k_full[:, j * SUB : (j + 1) * SUB]
            if AMASK == 0:
                vj = v_full if SUB == BLK else v_full[:, j * SUB : (j + 1) * SUB]
                vj = jnp.where(col_v + off < plen, vj, jnp.zeros_like(vj))
            elif AMASK == 1:
                vu = v_u32 if SUB == BLK else v_u32[:, j * SUB : (j + 1) * SUB]
                vu = jnp.where(col < rem, vu, jnp.uint32(0))
                vj = pltpu.bitcast(vu, FP8)
            else:
                assert AMASK == 2
                vu = v_u32 if SUB == BLK else v_u32[:, j * SUB : (j + 1) * SUB]
                vu = lax.cond(
                    full_blk,
                    lambda v_val=vu: v_val,
                    lambda v_val=vu, r_val=rem: jnp.where(
                        col < r_val, v_val, jnp.uint32(0)
                    ),
                )
                vj = pltpu.bitcast(vu, FP8)
            scores = stream.mm(qrows, kj, NN)
        else:
            kj = kbuf[s] if SUB == BLK else kbuf[s, j * SUB : (j + 1) * SUB]
            vj = vbuf[s] if SUB == BLK else vbuf[s, j * SUB : (j + 1) * SUB]
            scores = stream.mm(qrows, kj, NT)
        if AMASK == 0:
            scores = jnp.where(col + off < plen, scores, -jnp.inf)
        elif AMASK in (1, 4):
            scores = jnp.where(col < rem, scores, -jnp.inf)
        else:
            scores = lax.cond(
                full_blk,
                lambda s_val=scores: s_val,
                lambda s_val=scores, r_val=rem: jnp.where(col < r_val, s_val, -jnp.inf),
            )
        rmax = jnp.max(scores, axis=-1, keepdims=True)
        outs.append((scores, rmax, vj))
    return b, outs


def item_back(cx: GQAEmit, b: Any, outs: Any) -> None:
    """Fold one item's sub-blocks into sequence `b`'s online softmax state."""
    ex, stream, am, al, aacc = cx.ex, cx.stream, cx.am, cx.al, cx.aacc
    for scores, rmax, vj in outs:
        m_old = am[b]
        m_new = jnp.maximum(m_old, rmax)
        alpha = jnp.exp(m_old - m_new)
        pr = jnp.exp(scores - m_new[:, 0:1])
        pv = stream.mm(pr.astype(BF16), vj, NT if cache_exec(ex).paged_kv else NN)
        al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
        aacc[b] = aacc[b] * alpha[:, 0:1] + pv
        am[b] = m_new


def att_core(cx: GQAEmit, kv: KVRing) -> None:
    """Online softmax over this rank's KV worklist (optionally item pairs)."""
    ex, ap, skip, sc, n_items, KR = cx.ex, cx.ap, cx.skip, cx.sc, cx.n_items, cx.KR
    am, al, aacc = cx.am, cx.al, cx.aacc
    PAIR_KV, SUB, AMASK, HD, TQ = ap.PAIR_KV, ap.SUB, ap.AMASK, ap.HD, ap.TQ
    kv_start, kv_wait = kv.start, kv.wait
    with sc("att_core"):
        col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
        col_v: Any = (
            lax.broadcasted_iota(I32, (HD, SUB), 1)
            if (cache_exec(ex).paged_kv and AMASK == 0)
            else None
        )
        am[...] = jnp.full(am.shape, -1e30, F32)
        al[...] = jnp.zeros(al.shape, F32)
        aacc[...] = jnp.zeros(aacc.shape, F32)

        def item(i, carry):
            kv_wait(i)
            if "attn" not in skip:
                item_back(cx, *item_front(cx, kv, col, col_v, i))
            kv_start(i + KR)
            return carry

        def item_pair(p, carry):
            i0 = 2 * p
            i1 = i0 + 1
            kv_wait(i0)
            kv_wait(i1)
            if "attn" not in skip:
                f0 = item_front(cx, kv, col, col_v, i0)
                f1 = item_front(cx, kv, col, col_v, i1)
                item_back(cx, *f0)
                item_back(cx, *f1)
            kv_start(i0 + KR)
            kv_start(i1 + KR)
            return carry

        if PAIR_KV:
            lax.fori_loop(0, lax.div(n_items, 2), item_pair, 0)

            @pl.when(lax.rem(n_items, 2) == 1)
            def _():
                item(n_items - 1, 0)
        else:
            lax.fori_loop(0, n_items, item, 0)

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
"""MoE router top-k: packed-key / Batcher lane top-k and the routing tables."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax

from ...ops import BF16, F32, I32
from .context import MoEEmit

NT = (((1,), (1,)), ((), ()))
TOPK_NL = 16


def _batcher_net(n: int) -> list[tuple[int, int]]:
    """Batcher odd-even mergesort comparators `(i, j)` (`i < j`, `i` takes max)."""
    out: list[tuple[int, int]] = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        out.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return out


def topk_rows_on_lanes(
    key, topk: int, em: int, need_ids: bool = True, nl: int = TOPK_NL
):
    """Exact top-k of distinct f32 keys `[MR, E]` (`MR <= 128`)."""
    mr, e = key.shape
    kp = (
        key
        if mr == 128
        else jnp.concatenate([key, jnp.zeros((128 - mr, e), F32)], axis=0)
    )
    kt = kp.T  # [E, 128]
    sl = 128 // nl
    lev = [
        jnp.concatenate(
            [kt[bk * 128 + i * sl : bk * 128 + (i + 1) * sl] for bk in range(e // 128)],
            axis=0,
        )
        for i in range(nl)
    ]
    for i, j in _batcher_net(nl):
        lev[i], lev[j] = jnp.maximum(lev[i], lev[j]), jnp.minimum(lev[i], lev[j])
    ks = []
    for k in range(topk):
        m = jnp.max(lev[0], axis=0, keepdims=True)
        ks.append(m)
        nup = min(nl, topk - 1 - k)
        if nup == 0:
            break
        hit = lev[0] == m
        for j in range(nup):
            lev[j] = jnp.where(hit, lev[j + 1] if j + 1 < nl else -1.0, lev[j])
    thr = jnp.broadcast_to(ks[topk - 1], (128, 128)).T[0:mr, :]
    sel = key >= jnp.concatenate([thr] * (e // 128), axis=1)
    ids = None
    if need_ids:
        rows = jnp.concatenate(ks + [jnp.zeros((128 - topk, 128), F32)], axis=0)
        kk = rows.T[0:mr, :]
        lane = lax.broadcasted_iota(I32, (mr, 128), 1)
        ids = jnp.where(lane < topk, em - (lax.bitcast_convert_type(kk, I32) & em), 0)
    return ids, sel


def route_topk(
    m: MoEEmit, choice: Any, p8: Any, VEC: bool, slot_ec: Any, NCP: int, MROW: int
) -> None:
    """Pick each row's top-k experts; write `hitv` (ids or VEC meta) and `wn`."""
    ctx, ex, mp, stream, scr = m.ctx, m.ex, m.mp, m.stream, m.scr
    EM, FLT_MIN, lane128, lane_e8, row8 = m.EM, m.FLT_MIN, m.lane128, m.lane_e8, m.row8
    B, MR, R = ex.B, ex.MR, ex.R
    E, ER, TOPK = mp.E, mp.ER, mp.TOPK
    ids = jnp.zeros((MR, 128), I32)
    if mp.topk_packed:
        kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
        key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e8), F32)
        if mp.topk_lanes and E % 128 == 0 and MR <= 128 and TOPK <= 128:
            ids_l, sel = topk_rows_on_lanes(key, TOPK, EM, need_ids=not VEC)
            if not VEC:
                ids = ids_l
        else:
            for k in range(TOPK):
                kmax = jnp.max(key, axis=-1, keepdims=True)
                if not VEC:
                    am_ = EM - (lax.bitcast_convert_type(kmax, I32) & EM)
                    ids = jnp.where(lane128 == k, am_, ids)
                key = jnp.where(key == kmax, -1.0, key)
            sel = key < 0.0
    else:
        sel = jnp.zeros((MR, E), jnp.bool_)
        for k in range(TOPK):
            cmax = jnp.max(choice, axis=-1, keepdims=True)
            am_ = jnp.min(
                jnp.where(choice == cmax, lane_e8, E),
                axis=-1,
                keepdims=True,
            )
            ids = jnp.where(lane128 == k, am_, ids)
            hit = lane_e8 == am_
            sel = jnp.logical_or(sel, hit)
            choice = jnp.where(hit, -jnp.inf, choice)
    if VEC:
        hitf = jnp.max(
            jnp.where(jnp.logical_and(sel, row8 < B), 1, 0).astype(I32),
            axis=0,
            keepdims=True,
        )
        cpi = ctx.sm(slot_ec, "ecopy").astype(I32)
        fixed = jnp.where(cpi > 0, 0, hitf)
        for j in range(MROW):
            scr["hitv"][j : j + 1, :] = fixed[:, j * 128 : (j + 1) * 128]
        lane_1e = lax.broadcasted_iota(I32, (1, E), 1)
        v1 = (lane_1e + 1) * hitf
        row_a = lax.broadcasted_iota(I32, (16, E), 0)
        a = jnp.where(
            row_a == 0,
            lax.shift_right_logical(v1, 5),
            jnp.where(
                row_a == 1,
                lax.bitwise_and(v1, 31),
                jnp.where(row_a == 2, hitf, 0),
            ),
        )
        a = a.astype(F32).astype(BF16)
        row_b = lax.broadcasted_iota(I32, (128, E), 0)
        lane_b = lax.broadcasted_iota(I32, (128, E), 1)
        rr = row_b - NCP
        in_rank = jnp.logical_and(lane_b >= rr * ER, lane_b < (rr + 1) * ER)
        bm = jnp.where(
            row_b < NCP,
            jnp.where(cpi == row_b + 1, 1.0, 0.0),
            jnp.where(
                rr < R,
                jnp.where(in_rank, 1.0, 0.0),
                jnp.where(rr == R, 1.0, 0.0),
            ),
        )
        bm = bm.astype(BF16)
        res = stream.mm(a, bm, NT)
        lane_r = lax.broadcasted_iota(I32, (1, 128), 1)
        meta = jnp.where(lane_r < NCP, res[0:1] * 32.0 + res[1:2], res[2:3])
        scr["hitv"][MROW : MROW + 1, :] = meta.astype(I32)
    else:
        scr["hitv"][...] = ids
    wsel = jnp.where(sel, p8, 0.0)
    scr["wn"][...] = jnp.where(
        row8 < B, wsel / jnp.sum(wsel, axis=-1, keepdims=True), 0.0
    )

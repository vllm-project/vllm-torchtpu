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
"""GQA staging: Q/K norm + RoPE, Q exchange across the KV-head group, new K/V."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, F32, FP8, I32, rmsnorm, rnd, rope_partial
from ...pool.config import cache_exec
from .context import GQAEmit
from .params import GQAParams


def rms_zc(ap: GQAParams, x_in: Any, w_in: Any) -> Any:
    return rmsnorm(x_in, w_in, ap.eps, zero_centered=ap.zero_centered_norm)


def rope(ap: GQAParams, x_in: Any, cos_in: Any, sin_in: Any) -> Any:
    return rope_partial(
        x_in, cos_in, sin_in, ap.ROT, round_intermediates=not ap.rope_fp32
    )


def att_q(cx: GQAEmit, slot: Any) -> tuple[Any, Any, int, int, Any]:
    """QKV projection, then normed / roped Q into the Q staging buffers."""
    ctx, ex, ap, scr, pos_ref, sc = cx.ctx, cx.ex, cx.ap, cx.scr, cx.pos_ref, cx.sc
    stream, gi = cx.stream, cx.gi
    B, NS, VER = ex.B, ex.NS, ex.VER
    nh, HD, NAI, ROT, SPLIT, PACK = ap.nh, ap.HD, ap.NAI, ap.ROT, ap.SPLIT, ap.PACK
    QV, GQ, NHH, SR, QG = ap.QV, ap.GQ, ap.NHH, ap.SR, ap.QG
    with sc("att_in"):
        y = stream.dmatmul("att_in", scr["xb"], NAI)
        ybuf = scr["ybuf"]
        ybuf[:, :NAI] = rnd(y * ctx.sm(slot, "in_s", NAI))
    qn = ctx.sm(slot, "qn")
    kn = ctx.sm(slot, "kn")
    half = ROT // 2
    inv = 1.0 / (
        ap.theta ** (lax.broadcasted_iota(I32, (1, half), 1).astype(F32) * 2.0 / ROT)
    )

    koff = nh * 2 * HD
    rcs = scr["rcs"]
    with sc("att_qk"):
        if QV:
            cos = scr["rope"][0, :B, :half]
            sin = scr["rope"][1, :B, :half]
            for h in range(nh):
                q0_ = h * 2 * HD
                qh = rms_zc(ap, ybuf[:B, q0_ : q0_ + HD], qn)
                qh = rnd(rope(ap, qh, cos, sin) * HD**-0.5)
                for j in range(NHH):
                    scr["qg"][
                        0 if GQ else gi,
                        jnp.int32(j),
                        pl.ds(h, B, stride=SR),
                        :,
                    ] = qh[:, j * 128 : (j + 1) * 128]
            if GQ:
                for sq in range(NS):
                    r0 = sq * QG
                    scr["qxb"][gi, sq] = jnp.concatenate(
                        [scr["qg"][0, j, r0 : r0 + QG, :] for j in range(NHH)],
                        axis=-1,
                    ).astype(BF16)
        else:
            for b in range(B):
                pb = pos_ref[b]
                ang = pb.astype(F32) * inv
                cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                rcs[0, b : b + 1, :] = cos
                rcs[1, b : b + 1, :] = sin
                qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                q = rms_zc(ap, qg[:, :HD], qn)
                q = rnd(rope(ap, q, cos, sin) * HD**-0.5)
                if PACK:
                    t_, o_ = ap.qrow(b)
                    scr["qg"][gi, t_, o_ : o_ + nh] = q
                else:
                    if nh < 8:
                        q = jnp.concatenate((q, jnp.zeros((8 - nh, HD), F32)), axis=0)
                    if SPLIT or VER:
                        scr["qg"][gi, b] = q
                    else:
                        scr["qb"][b] = q.astype(BF16)
    return ybuf, kn, half, koff, rcs


def q_exchange_start(cx: GQAEmit) -> list[Any]:
    """Start the Q all-gather within the KV-head group (`SPLIT`)."""
    ap, scr, sc, comm, gi = cx.ap, cx.scr, cx.sc, cx.comm, cx.gi
    G, SPLIT, GQ = ap.G, ap.SPLIT, ap.GQ
    qx = []
    if SPLIT:
        with sc("att_qgather"):
            qgb = scr["qxb"] if GQ else scr["qg"]
            qx = comm.group_copies(
                lambda off: qgb.at[gi],
                qgb.at[gi],
                scr["sq_ssem"],
                scr["sq_rsem"],
                G,
            )
            for d in qx:
                d.start()
    return qx


def kv_new(
    cx: GQAEmit, ybuf: Any, kn: Any, half: int, koff: int, rcs: Any
) -> tuple[Any, ...]:
    """Output gate, this step's normed / roped K and V; start their writeback."""
    ex, ap, scr, O, sc, a = cx.ex, cx.ap, cx.scr, cx.O, cx.sc, cx.a  # noqa: E741
    B = ex.B
    nh, HD, QV = ap.nh, ap.HD, ap.QV
    with sc("att_kvnew"):
        if QV:
            cos = scr["rope"][0, :B, :half]
            sin = scr["rope"][1, :B, :half]
            for h in range(nh):
                q0_ = h * 2 * HD
                scr["gate"][:B, h * HD : (h + 1) * HD] = ybuf[
                    :B, q0_ + HD : q0_ + 2 * HD
                ]
            k = rms_zc(ap, ybuf[:B, koff : koff + HD], kn)
            v = ybuf[:B, koff + HD : koff + 2 * HD]
            k = rope(ap, k, cos, sin)
            if cache_exec(ex).paged_kv:
                k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                scr["knew"][:B, :] = k_fp8.astype(F32)
                scr["vnew"][:B, :] = v_fp8.astype(F32)
            else:
                scr["knew"][:B, :] = k
                scr["vnew"][:B, :] = v
        else:
            for b in range(B):
                cos, sin = rcs[0, b : b + 1, :], rcs[1, b : b + 1, :]
                qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                k = rms_zc(ap, ybuf[b : b + 1, koff : koff + HD], kn)
                v = ybuf[b : b + 1, koff + HD : koff + 2 * HD]
                k = rope(ap, k, cos, sin)
                if cache_exec(ex).paged_kv:
                    k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                    v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                    scr["knew"][b : b + 1, :] = k_fp8.astype(F32)
                    scr["vnew"][b : b + 1, :] = v_fp8.astype(F32)
                else:
                    scr["knew"][b : b + 1, :] = k
                    scr["vnew"][b : b + 1, :] = v
        if cache_exec(ex).paged_kv:
            scr["knew_o"][...] = scr["knew"][...].astype(FP8)
            scr["vnew_o"][...] = scr["vnew"][...].astype(FP8)
        if not cache_exec(ex).FUSED:
            k_src = scr["knew_o"] if cache_exec(ex).paged_kv else scr["knew"]
            v_src = scr["vnew_o"] if cache_exec(ex).paged_kv else scr["vnew"]
            kv_out = (
                pltpu.make_async_copy(k_src, O["knew"].at[a], scr["kv_out_sem"].at[0]),
                pltpu.make_async_copy(v_src, O["vnew"].at[a], scr["kv_out_sem"].at[1]),
            )
            for d in kv_out:
                d.start()
        else:
            kv_out = ()
    return kv_out


def q_gather_finish(cx: GQAEmit, qx: list[Any]) -> None:
    """Wait for the Q exchange and lay the group's Q rows out per sequence."""
    ex, ap, scr, sc = cx.ex, cx.ap, cx.scr, cx.sc
    T, NS = ex.T, ex.NS
    G, nh, HD, QR, SPLIT, QV, GQ = ap.G, ap.nh, ap.HD, ap.QR, ap.SPLIT, ap.QV, ap.GQ
    NHH, QG = ap.NHH, ap.QG
    with sc("att_qgather"):
        qgb = scr["qg"]
        for d in qx:
            d.wait_recv()
        for d in qx:
            d.wait_send()
        if GQ:
            for sq in range(NS):
                for g_idx in range(G):
                    scr["qb"][sq, g_idx * QG : (g_idx + 1) * QG] = scr["qxb"][g_idx, sq]

        def qg_rows(g_idx, r0, n):
            if not QV:
                t_, o_ = divmod(r0, 8)
                return qgb[g_idx, t_, o_ : o_ + n, :]
            return jnp.concatenate(
                [qgb[g_idx, j, r0 : r0 + n, :] for j in range(NHH)],
                axis=-1,
            )

        def group_q(b):
            if not SPLIT:
                return qg_rows(0, b * 8, 8)
            t_, o_ = ap.qrow(b)
            qa = jnp.concatenate(
                [qg_rows(g_idx, t_ * 8 + o_, nh) for g_idx in range(G)],
                axis=0,
            )
            if G * nh < QR:
                qa = jnp.concatenate((qa, jnp.zeros((QR - G * nh, HD), F32)), 0)
            return qa

        for sq in range(NS if not GQ else 0):
            qs = [group_q(sq * T + t) for t in range(T)]
            qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
            scr["qb"][sq] = qs.astype(BF16)

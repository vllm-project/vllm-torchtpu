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
"""MoE routed experts: rolled / unrolled, `PIPE` and `PIPE_EP` schedules."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, F32, FP8, rnd
from .context import MoEEmit
from .ring import egive, egive_group, es_wait, etake

_FP8_MAX = 448.0


def unpack_nvfp4_tile(
    w_u8: jax.Array, n_logical: int, *, interpret: bool = False
) -> jax.Array:
    """Unpack `[tk_raw, n_raw] uint8` tile into `[2 * tk_raw, n_logical] bf16`."""
    tk_raw = w_u8.shape[0]
    if tk_raw % 8 != 0 or n_logical % 8 != 0:
        raise ValueError(
            f"unpack_nvfp4_tile requires tk_raw % 8 == 0 and n_logical % 8 == 0, "
            f"got tk_raw={tk_raw}, n_logical={n_logical}"
        )
    packed = w_u8[:, :n_logical]
    scale_cols = n_logical // 8
    bscale_u8 = w_u8[:, n_logical : n_logical + scale_cols]
    bscale = (
        lax.bitcast_convert_type(bscale_u8, jnp.float8_e4m3fn)
        .astype(BF16)
        .reshape(tk_raw // 8, 1, n_logical)
    )
    # Interpret mode has no `pltpu.bitcast` rule; the low nibble is the even K row.
    w4 = (
        lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).transpose(0, 2, 1)
        if interpret
        else pltpu.bitcast(packed, jnp.float4_e2m1fn)
    )
    w_bf16 = w4.astype(BF16).reshape(tk_raw // 8, 16, n_logical)
    return (w_bf16 * bscale).reshape(2 * tk_raw, n_logical)


def unpack_nvfp4_tile_fp8(
    w_u8: jax.Array, n_logical: int, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Unpack `[tk_raw, n_raw] uint8` into `(w_fp8, w_s)`."""
    tk_raw = w_u8.shape[0]
    if tk_raw % 8 != 0 or n_logical % 8 != 0:
        raise ValueError(
            f"unpack_nvfp4_tile_fp8 requires tk_raw % 8 == 0 and n_logical % 8 == 0, "
            f"got tk_raw={tk_raw}, n_logical={n_logical}"
        )
    packed = w_u8[:, :n_logical]
    scale_cols = n_logical // 8
    bscale_u8 = w_u8[:, n_logical : n_logical + scale_cols]
    bscale = (
        lax.bitcast_convert_type(bscale_u8, jnp.float8_e4m3fn)
        .astype(BF16)
        .reshape(tk_raw // 8, 1, n_logical)
    )
    s_max = jnp.max(jnp.abs(bscale), axis=0, keepdims=True)
    bscale_norm = jnp.where(s_max > 0, bscale / s_max, jnp.zeros_like(bscale))
    w4 = (
        lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).transpose(0, 2, 1)
        if interpret
        else pltpu.bitcast(packed, jnp.float4_e2m1fn)
    )
    w_bf16 = w4.astype(BF16).reshape(tk_raw // 8, 16, n_logical)
    w_fp8 = (w_bf16 * bscale_norm).reshape(2 * tk_raw, n_logical).astype(FP8)
    return w_fp8, s_max.reshape(1, n_logical).astype(F32)


def _etake_w(m: MoEEmit, kind: str, u, n_logical: int) -> jax.Array:
    w_ref = etake(m, kind, u)
    if m.mp.expert_quant == "nvfp4":
        return unpack_nvfp4_tile(w_ref[...], n_logical, interpret=bool(m.ex.interpret))
    return w_ref[...]


def _mm_etake_w(
    m: MoEEmit,
    kind: str,
    u,
    x_ref: Any,
    k0_logical: Any,
    tk_logical: int,
    n_logical: int,
    acc_ref: Any,
) -> None:
    """Accumulate `x_ref[:, k0:k0+tk_logical] @ tile(u)` into `acc_ref[:, :n]`."""
    w_ref = etake(m, kind, u)
    if m.mp.expert_quant != "nvfp4":
        acc_ref[:, :n_logical] += m.stream.mm(
            x_ref[:, pl.ds(k0_logical, tk_logical)], w_ref[...]
        )
        return
    tk_raw = tk_logical // 2
    if tk_raw > 64 and tk_raw % 64 == 0:
        is_interp = bool(m.ex.interpret)
        use_fp8_dot = (m.ex.MR <= 32) and (not is_interp)
        for s in range(tk_raw // 64):
            r0 = pl.multiple_of(s * 64, 64)
            k_sub = pl.multiple_of(k0_logical + s * 128, 128)
            x_sub = x_ref[:, pl.ds(k_sub, 128)]
            if use_fp8_dot:
                w_fp8, w_s = unpack_nvfp4_tile_fp8(
                    w_ref[pl.ds(r0, 64), :], n_logical, interpret=is_interp
                )
                x_max = jnp.max(jnp.abs(x_sub), axis=-1, keepdims=True)
                x_s = jnp.where(x_max > 0, x_max / _FP8_MAX, jnp.ones_like(x_max))
                x_fp8 = (x_sub / x_s).astype(FP8)
                acc_ref[:, :n_logical] += m.stream.mm(x_fp8, w_fp8) * (
                    x_s.astype(F32) * w_s
                )
            else:
                w_sub = unpack_nvfp4_tile(
                    w_ref[pl.ds(r0, 64), :], n_logical, interpret=is_interp
                )
                acc_ref[:, :n_logical] += m.stream.mm(x_sub, w_sub)
    else:
        w = unpack_nvfp4_tile(w_ref[...], n_logical, interpret=bool(m.ex.interpret))
        acc_ref[:, :n_logical] += m.stream.mm(
            x_ref[:, pl.ds(k0_logical, tk_logical)], w
        )


def gate_up(m: MoEEmit, i, w13):
    mp, scr, stream = m.mp, m.scr, m.stream
    IR, NSS = mp.IR, mp.NSS
    h = stream.mm(scr["xb"][...], w13) * scr["es"][i % NSS, :, : 2 * IR]
    return rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)


def down(m: MoEEmit, i, mid, w2):
    ex, lane_e8, mp, scr, stream = m.ex, m.lane_e8, m.mp, m.scr, m.stream
    MR = ex.MR
    CH, H, IR, NSS = mp.CH, mp.H, mp.IR, mp.NSS
    e, s, part = scr["elist"][i], i % NSS, scr["part"]
    wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
    for j in range(H // CH):
        lo = 2 * IR + j * CH
        y = stream.mm(mid, w2[:, j * CH : (j + 1) * CH])[0:MR]
        y = rnd(y * scr["es"][s, :, lo : lo + CH])
        part[0:MR, j * CH : (j + 1) * CH] += wcol * y


def expert(m: MoEEmit, l_idx, i, carry):
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    CH, H, IR, NSS, T13, T2, TE = mp.CH, mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE
    e = scr["elist"][i]
    s = i % NSS
    base = i * TE
    acc, xb, xb2, part = scr["acc"], scr["xb"], scr["xb2"], scr["part"]
    DIRECT0 = mp.r2_direct_store
    tk13_l = (2 * tk["e13"]) if mp.expert_quant == "nvfp4" else tk["e13"]
    tk2_l = (2 * tk["e2"]) if mp.expert_quant == "nvfp4" else tk["e2"]
    if T13 == 1:
        h = stream.mm(xb[...], _etake_w(m, "e13", base, 2 * IR))
        egive(m, base, l_idx)
    elif not mp.unroll:
        # Default path: rolled tile loop with per-tile refill.
        acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)

        def b13(t, cc):
            u = base + t
            k0 = pl.multiple_of(t * tk13_l, tk13_l)
            _mm_etake_w(m, "e13", u, xb, k0, tk13_l, 2 * IR, acc)
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T13, b13, 0)
        h = acc[:, : 2 * IR]
    else:
        TK13 = tk13_l
        if not DIRECT0:
            acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)
        for t in range(T13):
            u = base + t
            w = _etake_w(m, "e13", u, 2 * IR)
            m13 = stream.mm(xb[:, t * TK13 : (t + 1) * TK13], w)
            if DIRECT0 and t == 0:
                acc[:, : 2 * IR] = m13
            else:
                acc[:, : 2 * IR] += m13
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, t - 1), (u, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, u, l_idx, t)
            else:
                egive(m, u, l_idx, t)
        h = acc[:, : 2 * IR]
    es_wait(m, i)
    h = h * scr["es"][s, :, : 2 * IR]
    mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
    xb2[:, :IR] = mid_.astype(BF16)
    wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
    u2 = base + T13
    if T2 == 1:
        w2 = _etake_w(m, "e2", u2, H)
        xm = xb2[:, :IR]
        for j in range(H // CH):
            lo = 2 * IR + j * CH
            y = stream.mm(xm, w2[:, j * CH : (j + 1) * CH])[0:MR]
            y = rnd(y * scr["es"][s, :, lo : lo + CH])
            part[0:MR, j * CH : (j + 1) * CH] += wcol * y
        egive(m, u2, l_idx)
    elif not mp.unroll:
        acc[:, :H] = jnp.zeros((MB, H), F32)

        def b2(t, cc):
            u = u2 + t
            k0 = pl.multiple_of(t * tk2_l, tk2_l)
            _mm_etake_w(m, "e2", u, xb2, k0, tk2_l, H, acc)
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T2, b2, 0)
        y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
        part[0:MR, :] += wcol * y
    else:
        TK2 = tk2_l
        if not DIRECT0:
            acc[:, :H] = jnp.zeros((MB, H), F32)
        for t in range(T2):
            u = u2 + t
            w = _etake_w(m, "e2", u, H)
            m2 = stream.mm(xb2[:, t * TK2 : (t + 1) * TK2], w)
            if DIRECT0 and t == 0:
                acc[:, :H] = m2
            else:
                acc[:, :H] += m2
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, T13 + t - 1), (u, T13 + t)], l_idx)
                elif t == T2 - 1:
                    egive(m, u, l_idx, T13 + t)
            else:
                egive(m, u, l_idx, T13 + t)
        y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
        part[0:MR, :] += wcol * y
    return carry


def experts_piped(m: MoEEmit, l_idx, n_hit):
    mp, scr = m.mp, m.scr
    IR, T13, TE = mp.IR, mp.T13, mp.TE
    xb2 = scr["xb2"]

    @pl.when(n_hit > 0)
    def _():
        w13 = _etake_w(m, "e13", 0, 2 * IR)
        es_wait(m, 0)
        xb2[:, :IR] = gate_up(m, 0, w13)
        egive(m, 0, l_idx, 0)

        def body(i, carry):
            u13, u2 = (i + 1) * TE, i * TE + T13
            w13 = _etake_w(m, "e13", u13, 2 * IR)
            es_wait(m, i + 1)
            w2 = _etake_w(m, "e2", u2, mp.H)
            mid = xb2[:, :IR]
            nxt = gate_up(m, i + 1, w13)
            down(m, i, mid, w2)
            xb2[:, :IR] = nxt
            egive(m, u13, l_idx, 0)
            egive(m, u2, l_idx, T13)
            return carry

        lax.fori_loop(0, n_hit - 1, body, 0)
        last = n_hit - 1
        u2 = last * TE + T13
        down(m, last, xb2[:, :IR], _etake_w(m, "e2", u2, mp.H))
        egive(m, u2, l_idx, T13)


def experts_piped_ep(m: MoEEmit, l_idx, n_hit):
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    H, IR, NSS, T13, T2, TE = mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE

    @pl.when(n_hit > 0)
    def _():
        acc, acc13 = scr["acc"], scr["acc13"]
        xb, xb2, part = scr["xb"], scr["xb2"], scr["part"]
        BN = 2 * IR
        TK13 = (2 * tk["e13"]) if mp.expert_quant == "nvfp4" else tk["e13"]
        TK2 = (2 * tk["e2"]) if mp.expert_quant == "nvfp4" else tk["e2"]
        DIRECT0 = mp.r2_direct_store

        if not DIRECT0:
            acc13[...] = jnp.zeros((MB, BN), F32)
        for t in range(T13):
            w13 = _etake_w(m, "e13", t, BN)
            k0 = t * TK13
            mm13 = stream.mm(xb[:, k0 : k0 + TK13], w13)
            if DIRECT0 and t == 0:
                acc13[:, :BN] = mm13
            else:
                acc13[:, :BN] += mm13
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(t - 1, t - 1), (t, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, t, l_idx, t)
            else:
                egive(m, t, l_idx, t)
        es_wait(m, 0)
        h0 = acc13[:, :BN] * scr["es"][0, :, :BN]
        mid0 = rnd(rnd(jax.nn.silu(rnd(h0[:, :IR]))) * rnd(h0[:, IR:]))
        xb2[:, :IR] = mid0.astype(BF16)

        def body_ep(i, _):
            e = scr["elist"][i]
            s_cur = i % NSS
            s_nxt = (i + 1) % NSS
            base_w2 = i * TE + T13
            base_nxt13 = (i + 1) * TE

            if not DIRECT0:
                acc[:, :H] = jnp.zeros((MB, H), F32)
                acc13[...] = jnp.zeros((MB, BN), F32)

            for s_step in range(T2):
                u2 = base_w2 + s_step
                t0 = 2 * s_step
                t1 = 2 * s_step + 1
                u13_0 = base_nxt13 + t0
                u13_1 = base_nxt13 + t1

                w2 = _etake_w(m, "e2", u2, H)
                w13_0 = _etake_w(m, "e13", u13_0, BN)
                k2 = s_step * TK2
                k13_0 = t0 * TK13
                mm2 = stream.mm(xb2[:, k2 : k2 + TK2], w2)
                mm13_0 = stream.mm(xb[:, k13_0 : k13_0 + TK13], w13_0)
                if DIRECT0 and s_step == 0:
                    acc[:, :H] = mm2
                    acc13[:, :BN] = mm13_0
                else:
                    acc[:, :H] += mm2
                    acc13[:, :BN] += mm13_0
                if mp.group_rel:
                    egive_group(m, [(u2, T13 + s_step), (u13_0, t0)], l_idx)
                else:
                    egive(m, u2, l_idx, T13 + s_step)
                    egive(m, u13_0, l_idx, t0)

                w13_1 = _etake_w(m, "e13", u13_1, BN)
                k13_1 = t1 * TK13
                acc13[:, :BN] += stream.mm(xb[:, k13_1 : k13_1 + TK13], w13_1)
                egive(m, u13_1, l_idx, t1)

            wcol = jnp.sum(
                jnp.where(lane_e8 == e, scr["wn"][...], 0.0),
                axis=1,
                keepdims=True,
            )
            y = rnd(acc[0:MR, :H] * scr["es"][s_cur, :, BN:])
            part[0:MR, :] += wcol * y

            es_wait(m, i + 1)
            h_nxt = acc13[:, :BN] * scr["es"][s_nxt, :, :BN]
            mid_nxt = rnd(rnd(jax.nn.silu(rnd(h_nxt[:, :IR]))) * rnd(h_nxt[:, IR:]))
            xb2[:, :IR] = mid_nxt.astype(BF16)
            return 0

        lax.fori_loop(0, n_hit - 1, body_ep, 0)

        i_last = n_hit - 1
        e_last = scr["elist"][i_last]
        s_last = i_last % NSS
        base_last_w2 = i_last * TE + T13
        if not DIRECT0:
            acc[:, :H] = jnp.zeros((MB, H), F32)
        for s_step in range(T2):
            u2 = base_last_w2 + s_step
            w2 = _etake_w(m, "e2", u2, H)
            k2 = s_step * TK2
            mm2 = stream.mm(xb2[:, k2 : k2 + TK2], w2)
            if DIRECT0 and s_step == 0:
                acc[:, :H] = mm2
            else:
                acc[:, :H] += mm2
            if mp.group_rel:
                if s_step % 2 == 1:
                    egive_group(
                        m,
                        [(u2 - 1, T13 + s_step - 1), (u2, T13 + s_step)],
                        l_idx,
                    )
                elif s_step == T2 - 1:
                    egive(m, u2, l_idx, T13 + s_step)
            else:
                egive(m, u2, l_idx, T13 + s_step)
        wcol_last = jnp.sum(
            jnp.where(lane_e8 == e_last, scr["wn"][...], 0.0),
            axis=1,
            keepdims=True,
        )
        y_last = rnd(acc[0:MR, :H] * scr["es"][s_last, :, BN:])
        part[0:MR, :] += wcol_last * y_last

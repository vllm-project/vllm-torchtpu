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
"""GDN input stage: state prefetch, `lin_in` projection and the causal conv."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, F32, I32, rnd, rup
from .params import GDNParams
from .states import (
    conv_out_src,
    gpr_conv_copy,
    gpr_conv_start,
    has_gpr_input,
    rec_chunk,
    rec_drain,
    rec_in,
)


def gdn_in(ctx: Any, gp: GDNParams, o: Any, slot: Any) -> Any:
    """Start layer `o`'s state reads (after the previous writes), project `lin_in`."""
    ex, scr, ST, O, skip, sc = ctx.ex, ctx.scr, ctx.ST, ctx.O, ctx.skip, ctx.sc  # noqa: E741
    stream = ctx.stream
    VER = ex.VER
    NLI, GCH, NRB, GPR = gp.NLI, gp.GCH, gp.NRB, gp.GPR
    conv_out_key = "conv_ckpt" if VER else "conv"
    rec_out_key = "rec_ckpt" if VER else "rec"
    convb, recb, sem = scr["convb"], scr["recb"], scr["st_sem"]
    with sc("gdn_in"):
        if "gdn_state" not in skip:
            if GPR:
                gpr_conv_start(ctx, gp, o)
                for s_ in range(NRB):
                    rec_chunk(ctx, gp, o, s_, s_, False).start()

            @pl.when(o > 0)
            def _():
                pltpu.make_async_copy(
                    conv_out_src(ctx, gp),
                    O[conv_out_key].at[0],
                    sem.at[2],
                ).wait()
                if GCH:
                    rec_drain(ctx, gp, 0)
                else:
                    pltpu.make_async_copy(recb, O[rec_out_key].at[0], sem.at[3]).wait()

            if not GPR:
                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).start()
                if GCH:
                    for s_ in range(NRB):
                        rec_chunk(ctx, gp, o, s_, s_, False).start()
                else:
                    for d in rec_in(ctx, gp, o):
                        d.start()

        y = stream.dmatmul("lin_in", scr["xb"], NLI)
        ybuf = scr["ybuf"]
        ybuf[:, :NLI] = rnd(y * ctx.sm(slot, "in_s", NLI))
    return ybuf


def conv_step(ctx: Any, gp: GDNParams, o: Any, slot: Any, ybuf: Any) -> Any:
    """Causal depthwise conv over the conv state; start the conv state write."""
    ex, scr, ST, O, raw_refs = ctx.ex, ctx.scr, ctx.ST, ctx.O, ctx.raw_refs  # noqa: E741
    skip, sc, stream = ctx.skip, ctx.sc, ctx.stream
    B, T, NS, VER = ex.B, ex.T, ex.NS, ex.VER
    QKV, GCH, GPR = gp.QKV, gp.GCH, gp.GPR
    conv_out_key = "conv_ckpt" if VER else "conv"
    convb, sem = scr["convb"], scr["st_sem"]
    with sc("gdn_conv"):
        if "gdn_state" not in skip:
            if GPR:
                hr_rows_128 = gp.HR * gp.qkv_128
                conv_row_iota = lax.broadcasted_iota(I32, (gp.HR, QKV), 0)
                for s_ in range(NS):

                    @pl.when(has_gpr_input(ctx, gp, o, s_))
                    def _(s_=s_):
                        gpr_conv_copy(ctx, gp, raw_refs[0], s_, 0, sem.at[0]).wait()

                for s_ in range(NS):
                    if ex.interpret:
                        raw_hr = pltpu.bitcast(scr["gpr_conv"][s_], BF16).reshape(
                            2 * gp.GCONV_ROWS, 128
                        )
                    else:
                        raw_hr = (
                            scr["gpr_conv"]
                            .at[s_]
                            .bitcast(BF16)
                            .reshape(2 * gp.GCONV_ROWS, 128)
                        )
                    raw_hr = raw_hr[:hr_rows_128, :].reshape(gp.HR, QKV).astype(F32)
                    keep = jnp.logical_and(
                        has_gpr_input(ctx, gp, o, s_), conv_row_iota < 3
                    )
                    convb[s_] = jnp.where(keep, raw_hr, 0.0)
            else:
                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).wait()
        ow, _ = stream.sp.lay["conv"]
        taps = [scr["sm"][slot, :, ow + t * QKV : ow + (t + 1) * QKV] for t in range(4)]
        u = None
        if VER:
            us = []
            for s in range(NS):
                convb[s, 3 : 3 + T, :] = ybuf[s * T : (s + 1) * T, 0:QKV]
                w = [convb[s, k : k + T, :] for k in range(4)]
                u_ = rnd(
                    jax.nn.silu(
                        w[0] * taps[0]
                        + w[1] * taps[1]
                        + w[2] * taps[2]
                        + w[3] * taps[3]
                    )
                )
                if GCH:
                    us.append(u_)
                else:
                    scr["ub"][s, 0:T, :] = u_
            if GCH:
                u = jnp.concatenate(us, axis=0)
        else:
            x = ybuf[0:B, 0:QKV]
            s0 = convb[0:B, :]
            s1 = convb[B : 2 * B, :]
            s2 = convb[2 * B : 3 * B, :]
            u = rnd(
                jax.nn.silu(s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x * taps[3])
            )
            convb[0:B, :] = s1
            convb[B : 2 * B, :] = s2
            convb[2 * B : 3 * B, :] = x
        if GPR:
            scr["convb16"][...] = convb[...].astype(BF16)
        if "gdn_state" not in skip:
            pltpu.make_async_copy(
                conv_out_src(ctx, gp),
                O[conv_out_key].at[o],
                sem.at[2],
            ).start()
        if GCH:
            b_pad = rup(B, 8)
            u_pad = jnp.pad(u, ((0, b_pad - B), (0, 0))) if b_pad > B else u
            ybuf[0:b_pad, 0:QKV] = u_pad
    return u

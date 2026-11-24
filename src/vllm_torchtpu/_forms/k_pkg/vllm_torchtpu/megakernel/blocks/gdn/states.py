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
"""GDN state I/O: checkpoint-slot reads (`GPR`) and the chunked `rec` ring DMAs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, U32, bswitch
from ...pool.config import cache_exec
from .params import GDNParams


def _kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)


def is_seq_active(ctx: Any, gp: GDNParams, s: Any) -> Any:
    ex, ST = ctx.ex, ctx.ST
    if cache_exec(ex).act_mode == "all":
        return jnp.bool_(True)
    if cache_exec(ex).act_mode == "ref":
        return ST["active_seqs"][s] != 0
    return (
        (ST["ckpt_indices"][0, s, 0] > 0)
        if gp.ckpt_is_3d
        else (ST["ckpt_indices"][s, 0] > 0)
    )


def gdn_roff(ctx: Any, gp: GDNParams, sub: Any, s: Any) -> Any:
    ST = ctx.ST
    return ST["read_offsets"][sub, s] if gp.roff_is_2d else ST["read_offsets"][s]


def has_gpr_input(ctx: Any, gp: GDNParams, o: Any, s: Any) -> Any:
    sub = lax.rem(o, gp.G_PER_ATT)
    return jnp.logical_and(
        is_seq_active(ctx, gp, s),
        gdn_roff(ctx, gp, sub, s) >= 0,
    )


def gdn_pb_in(ctx: Any, gp: GDNParams, sub: Any, s: Any) -> Any:
    ex, ST = ctx.ex, ctx.ST
    GDN_PAGES = gp.GDN_PAGES
    roff = jnp.maximum(gdn_roff(ctx, gp, sub, s), jnp.int32(0))
    if gp.ckpt_is_3d:
        pb = gp.pool_layout.page_base(ST["ckpt_indices"][sub, s, roff], sub, 3)
    else:
        pb = gp.pool_layout.page_base(ST["ckpt_indices"][s, roff], sub, 2)
    _kernel_debug_check(
        jnp.logical_and(pb >= 0, pb + GDN_PAGES <= cache_exec(ex).num_pages),
        "GDN gather checkpoint slot maps outside [0, num_pages)",
        interpret=bool(ex.interpret),
    )
    return jnp.clip(pb, 0, cache_exec(ex).num_pages - GDN_PAGES)


def gpr_start(
    ctx: Any,
    gp: GDNParams,
    o: Any,
    seqs: list[Any],
    dsts: list[Any],
    sem: Any,
    first: int,
    npages: int,
) -> None:
    raw_refs = ctx.raw_refs
    sub = lax.rem(o, gp.G_PER_ATT)
    g_pool = lax.div(o, gp.G_PER_ATT)

    def issue(r):
        for sq, dst in zip(seqs, dsts):

            @pl.when(has_gpr_input(ctx, gp, o, sq))
            def _(sq=sq, dst=dst):
                pb = gdn_pb_in(ctx, gp, sub, sq) + first
                pltpu.make_async_copy(r.at[pl.ds(pb, npages)], dst, sem).start()

    bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])


def gpr_wait(
    ctx: Any,
    gp: GDNParams,
    o: Any,
    sq: Any,
    dst: Any,
    sem: Any,
    npages: int,
) -> None:
    raw_refs = ctx.raw_refs

    @pl.when(has_gpr_input(ctx, gp, o, sq))
    def _():
        pltpu.make_async_copy(raw_refs[0].at[pl.ds(0, npages)], dst, sem).wait()


def gpr_conv_copy(ctx: Any, gp: GDNParams, r: Any, sq: int, pb: Any, sem: Any) -> Any:
    return pltpu.make_async_copy(
        r.at[pb, 0, pl.ds(0, gp.GCONV_ROWS)], ctx.scr["gpr_conv"].at[sq], sem
    )


def gpr_conv_start(ctx: Any, gp: GDNParams, o: Any) -> None:
    raw_refs = ctx.raw_refs
    sub = lax.rem(o, gp.G_PER_ATT)
    g_pool = lax.div(o, gp.G_PER_ATT)

    def issue(r):
        for sq in range(ctx.ex.NS):

            @pl.when(has_gpr_input(ctx, gp, o, sq))
            def _(sq=sq):
                gpr_conv_copy(
                    ctx,
                    gp,
                    r,
                    sq,
                    gdn_pb_in(ctx, gp, sub, sq) + gp.pool_layout.conv_page,
                    ctx.scr["st_sem"].at[0],
                ).start()

    bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])


def conv_out_src(ctx: Any, gp: GDNParams) -> Any:
    return ctx.scr["convb16"] if gp.GPR else ctx.scr["convb"]


class _GprRecIn:
    """In-kernel checkpoint-to-`recb` gather handle for `GPR=True`."""

    def __init__(
        self,
        ctx: Any,
        gp: GDNParams,
        o: Any,
        j: Any,
        s: Any,
        n: int,
    ) -> None:
        self.ctx = ctx
        self.gp = gp
        self.o = o
        self.s = s
        self.n = n
        self.nseq = n // (ctx.ex.T * gp.vh)
        self.seqs = [j * (gp.CB // ctx.ex.T) + q for q in range(self.nseq)]
        self.dsts = [ctx.scr["gpr_rec"].at[s, q] for q in range(self.nseq)]

    def start(self) -> None:
        gpr_start(
            self.ctx,
            self.gp,
            self.o,
            self.seqs,
            self.dsts,
            self.ctx.scr["rec_sem"].at[0, self.s],
            0,
            self.gp.REC_PAGES,
        )

    def wait(self) -> None:
        ctx, gp = self.ctx, self.gp
        scr = ctx.scr
        T = ctx.ex.T
        vh, KD, VD = gp.vh, gp.KD, gp.VD
        SSM_U32_CAP, ssm_u32_rows = gp.SSM_U32_CAP, gp.ssm_u32_rows
        s, n = self.s, self.n
        for sq, dst in zip(self.seqs, self.dsts):
            gpr_wait(ctx, gp, self.o, sq, dst, scr["rec_sem"].at[0, s], gp.REC_PAGES)
        for q, sq in enumerate(self.seqs):
            if ctx.ex.interpret:
                u32 = pltpu.bitcast(scr["gpr_rec"][s, q], U32).reshape(
                    SSM_U32_CAP, 128
                )[:ssm_u32_rows, :]
            else:
                u32 = (
                    scr["gpr_rec"]
                    .at[s, q]
                    .bitcast(U32)
                    .reshape(SSM_U32_CAP, 128)[:ssm_u32_rows, :]
                )
            u32 = jnp.where(has_gpr_input(ctx, gp, self.o, sq), u32, jnp.uint32(0))
            scr["recb"][s, pl.ds(n - n // T + q * vh, vh)] = pltpu.bitcast(
                u32, BF16
            ).reshape(vh, KD, VD)


def rec_chunk(ctx: Any, gp: GDNParams, o: Any, j: Any, s: Any, out: bool) -> Any:
    """Issue DMA for row chunk `j` of GDN layer `o` and ring buffer `s`."""
    ex = ctx.ex
    rec_out_key = "rec_ckpt" if ex.VER else "rec"
    n = gp.TR * gp.vh if gp.TR and isinstance(j, int) and j == gp.NF else gp.CBV
    r0 = j * gp.CBV if isinstance(j, int) else pl.multiple_of(j * gp.CBV, gp.CBV)
    buf = ctx.scr["recb"].at[s] if n == gp.CBV else ctx.scr["recb"].at[s, pl.ds(0, n)]
    if out:
        return pltpu.make_async_copy(
            buf,
            ctx.O[rec_out_key].at[o, pl.ds(r0, n)],
            ctx.scr["rec_sem"].at[1, s],
        )
    if gp.GPR:
        return _GprRecIn(ctx, gp, o, j, s, n)
    if ex.VER:
        ni = n // ex.T
        ri = (
            j * (gp.CBV // ex.T)
            if isinstance(j, int)
            else pl.multiple_of(j * (gp.CBV // ex.T), gp.CBV // ex.T)
        )
        return pltpu.make_async_copy(
            ctx.ST["rec"].at[o, pl.ds(ri, ni)],
            ctx.scr["recb"].at[s, pl.ds(n - ni, ni)],
            ctx.scr["rec_sem"].at[0, s],
        )
    return pltpu.make_async_copy(
        ctx.ST["rec"].at[o, pl.ds(r0, n)], buf, ctx.scr["rec_sem"].at[0, s]
    )


def rec_drain(ctx: Any, gp: GDNParams, o: Any) -> None:
    """Wait for layer `o`'s in-flight chunked `rec` write-backs."""
    for j_ in range(gp.NCH - gp.NRB, gp.NCH):
        rec_chunk(ctx, gp, o, j_, j_ % gp.NRB, True).wait()


def rec_in(ctx: Any, gp: GDNParams, o: Any) -> list[Any]:
    """Whole-state `rec` reads of layer `o` (`GCH=False`; per sequence in verify)."""
    ex, scr, ST = ctx.ex, ctx.scr, ctx.ST
    T, NS, VER = ex.T, ex.NS, ex.VER
    vh = gp.vh
    recb, sem = scr["recb"], scr["st_sem"]
    if not VER:
        return [pltpu.make_async_copy(ST["rec"].at[o], recb, sem.at[1])]
    return [
        pltpu.make_async_copy(
            ST["rec"].at[o, pl.ds(s * vh, vh)],
            recb.at[pl.ds(s * T * vh, vh)],
            scr["rin_sem"].at[s],
        )
        for s in range(NS)
    ]

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

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...comm.replication import replicate_groups
from ...ops import BF16, F32, I32, U32, bswitch, rup
from ...pool.backend import kernel_debug_check
from ...pool.config import cache_exec
from .params import GDNParams, gdn_lin_in_layout


def gdn_accept(
    states: dict[str, Any],
    ckpts: dict[str, Any],
    acc: Any,
    *,
    T: int,
    vh: int,
    taps: int = 3,
) -> dict[str, Any]:
    """Jittable on-device selection of post-verify GDN states from checkpoints."""
    c_ckpt = ckpts["conv_ckpt"] if "conv_ckpt" in ckpts else ckpts.get("conv")
    r_ckpt = ckpts["rec_ckpt"] if "rec_ckpt" in ckpts else ckpts.get("rec")
    out = {k: v for k, v in states.items() if k not in ("conv_ckpt", "rec_ckpt")}
    if c_ckpt is None or r_ckpt is None:
        return out

    NS = c_ckpt.shape[-3]
    acc_arr = jnp.asarray(acc, dtype=I32)
    assert acc_arr.shape == (NS,), (acc_arr.shape, NS)

    conv_next = jax.vmap(
        lambda c_seq, a_s: c_seq.at[..., 0:taps, :].set(
            lax.dynamic_slice_in_dim(c_seq, a_s, taps, axis=-2)
        ),
        in_axes=(-3, 0),
        out_axes=-3,
    )(c_ckpt, acc_arr)

    lead = r_ckpt.shape[:-3]
    KD, VD = r_ckpt.shape[-2], r_ckpt.shape[-1]
    r_unflat = r_ckpt.reshape((*lead, NS, T, vh, KD, VD))
    r_sel = jax.vmap(
        lambda r_seq, a_s: lax.dynamic_index_in_dim(
            r_seq, a_s - 1, axis=-4, keepdims=False
        ),
        in_axes=(-5, 0),
        out_axes=-4,
    )(r_unflat, acc_arr)
    out["conv"] = conv_next
    out["rec"] = r_sel.reshape((*lead, NS * vh, KD, VD))
    return out


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
    raw_roff = gdn_roff(ctx, gp, sub, s)
    kernel_debug_check(
        jnp.logical_and(raw_roff >= 0, raw_roff < gp.K_ckpt),
        "GDN gather read_offsets outside [0, K)",
        interpret=bool(ex.interpret),
    )
    roff = jnp.maximum(raw_roff, jnp.int32(0))
    if gp.ckpt_is_3d:
        pb = gp.pool_layout.page_base(ST["ckpt_indices"][sub, s, roff], sub, 3)
    else:
        pb = gp.pool_layout.page_base(ST["ckpt_indices"][s, roff], sub, 2)
    kernel_debug_check(
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


def gdn_init_logical_states(
    key: Any, g: Any, ex: Any, n_layers: int, *, conv_taps: int
) -> dict[str, Any]:
    """Initialize unfolded `[R, n_layers, ...]` GDN `conv` and `rec` states."""
    lay = gdn_lin_in_layout(g, ex)
    k_conv, k_rec = jax.random.split(key)
    conv = (
        jax.random.normal(
            k_conv, (ex.R, n_layers, ex.NS, conv_taps, lay.QKV), dtype=F32
        )
        * 0.05
    )
    kh_rep = max(1, int(ex.R) // int(g.KH))
    if kh_rep > 1:
        qk = 2 * lay.kh * lay.KD
        conv = conv.at[..., :qk].set(replicate_groups(conv[..., :qk], kh_rep))
    return {
        "conv": conv,
        "rec": (
            jax.random.normal(
                k_rec, (ex.R, n_layers, ex.NS, lay.vh, lay.KD, lay.VD), dtype=BF16
            )
            * 0.05
        ),
    }


def gdn_pack_logical_states(
    logical_states: Mapping[str, Any], g: Any, ex: Any, *, conv_taps: int
) -> dict[str, Any]:
    """Pack unfolded GDN `conv` and `rec` states into rank-folded kernel layout."""
    if "conv" not in logical_states or "rec" not in logical_states:
        return {}
    conv_l, rec_l = logical_states["conv"], logical_states["rec"]
    lay = gdn_lin_in_layout(g, ex)
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    n_layers = conv_l.shape[1]
    if conv_l.ndim == 5 and conv_l.shape[3] == conv_taps:
        if T == 1:
            cr = rup(conv_taps * B, 8)
            conv_tb = jnp.swapaxes(conv_l, 2, 3).reshape(
                R, n_layers, conv_taps * B, lay.QKV
            )
            conv_p = jnp.pad(conv_tb, ((0, 0), (0, 0), (0, cr - conv_taps * B), (0, 0)))
        else:
            hr = rup(conv_taps + T, 8)
            conv_p = jnp.pad(
                conv_l, ((0, 0), (0, 0), (0, 0), (0, hr - conv_taps), (0, 0))
            )
    else:
        conv_p = conv_l
    rec_p = (
        rec_l.reshape(R, n_layers, NS * lay.vh, lay.KD, lay.VD)
        if rec_l.ndim == 6
        else rec_l
    )
    return {
        "conv": conv_p.astype(F32).reshape(
            (conv_p.shape[0] * conv_p.shape[1], *conv_p.shape[2:])
        ),
        "rec": rec_p.astype(BF16).reshape(
            (rec_p.shape[0] * rec_p.shape[1], *rec_p.shape[2:])
        ),
    }


def gdn_unpack_kernel_states(
    raw_states: Mapping[str, Any], g: Any, ex: Any, *, conv_taps: int
) -> dict[str, Any]:
    """Unpack rank-folded GDN state outputs into `[R, ...]` reference states."""
    del g, conv_taps
    return {
        k: raw_states[k].reshape(
            (ex.R, raw_states[k].shape[0] // ex.R, *raw_states[k].shape[1:])
        )
        for k in ("conv", "rec", "conv_ckpt", "rec_ckpt")
        if k in raw_states
    }


def gdn_next_logical_states(
    out_states: Mapping[str, Any], g: Any, ex: Any, *, conv_taps: int
) -> dict[str, Any]:
    """Extract final-token unfolded GDN `conv` and `rec` states from step outputs."""
    lay = gdn_lin_in_layout(g, ex)
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    out: dict[str, Any] = {}
    c_src = out_states.get("conv_ckpt" if T > 1 else "conv")
    if c_src is not None:
        n_l = c_src.shape[1]
        if T == 1 and c_src.ndim == 4:
            c_tb = c_src[:, :, : conv_taps * B, :].reshape(
                R, n_l, conv_taps, B, lay.QKV
            )
            out["conv"] = jnp.swapaxes(c_tb, 2, 3)
        elif c_src.ndim == 5 and c_src.shape[3] >= T + conv_taps:
            out["conv"] = c_src[:, :, :, T : T + conv_taps, :]
        else:
            out["conv"] = c_src
    r_src = out_states.get("rec_ckpt" if T > 1 else "rec")
    if r_src is not None:
        n_l = r_src.shape[1]
        if T > 1 and r_src.shape[2] == NS * T * lay.vh:
            r_t = r_src.reshape(R, n_l, NS, T, lay.vh, lay.KD, lay.VD)
            out["rec"] = r_t[:, :, :, -1]
        else:
            out["rec"] = r_src.reshape(R, n_l, NS, lay.vh, lay.KD, lay.VD)
    return out

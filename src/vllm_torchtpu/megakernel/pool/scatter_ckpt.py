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
"""Pool scatter: GDN conv / recurrent checkpoints into their checkpoint slots."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import BF16, FP8, U32
from .backend import kernel_debug_check
from .config import cache_exec
from .layout import LANES, PoolLayout
from .scatter_ctx import ScatterKernel, is_active


def prepare_ckpt(
    layout: PoolLayout,
    ex: Any,
    conv: Any,
    rec: Any,
    ckpt_indices: Any,
    *,
    has_gdn: bool,
    interpret: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Kernel conv input, its VMEM buffer shape / dtype and the GDN geometry."""
    NS, B, T = ex.NS, ex.B, ex.T
    conv_in: Any = None
    conv_buf_shape: Any = None
    if has_gdn:
        assert conv is not None and rec is not None and ckpt_indices is not None
        vh, KD, VD = layout.vh_per_rank(ex.R), layout.KD, layout.VD
        QKV = layout.qkv_per_rank(ex.R)
        taps = layout.conv_taps
        ssm_u32_rows = (vh * KD) // 2
        conv_rows = (taps * QKV) // LANES
        conv_rows_pad = ((conv_rows + 15) // 16) * 16
        qkv_128 = QKV // LANES
        # The direct path reads `taps + 1` rows per token (the last one masked
        # to zero) when they fill the padded window exactly, else `taps` rows
        # padded with zero rows; it falls back to the packed copy when the
        # caller's conv tensor is too short.
        tail_row = (taps + 1) * qkv_128 == conv_rows_pad
        min_direct_rows = taps + 1 + T if tail_row else taps + T
        direct_conv = (
            T > 1
            and conv.ndim == 4
            and conv.shape[2] >= min_direct_rows
            and ((conv.shape[2] * qkv_128) % 16 == 0 or bool(interpret))
        )
        if direct_conv:
            conv_in = conv
            conv_buf_shape = (2, 1, conv.shape[2], QKV)
            conv_buf_dt = conv.dtype
        else:
            if conv.ndim == 3:
                conv_4d = jnp.swapaxes(
                    conv[:, : taps * B, :].reshape(conv.shape[0], taps, B, QKV), 1, 2
                )
            else:
                conv_4d = conv
            conv_has_hist = conv_4d.shape[2] >= taps + T
            if conv_has_hist:
                conv_win = jnp.stack(
                    [conv_4d[:, :, 1 + t : 1 + taps + t, :] for t in range(T)], axis=2
                )
            else:
                conv_win = jnp.broadcast_to(
                    conv_4d[:, :, None, 0:taps, :],
                    (conv_4d.shape[0], NS, T, taps, QKV),
                )
            conv_flat = conv_win.astype(BF16).reshape(
                conv_4d.shape[0], NS * T, conv_rows, LANES
            )
            if conv_rows_pad > conv_rows:
                conv_flat = jnp.pad(
                    conv_flat,
                    ((0, 0), (0, 0), (0, conv_rows_pad - conv_rows), (0, 0)),
                )
            conv_in = conv_flat.reshape(conv_4d.shape[0], NS * T * conv_rows_pad, LANES)
            conv_buf_shape = (2, T * conv_rows_pad, LANES)
            conv_buf_dt = BF16
        ckpt_is_3d = ckpt_indices.ndim == 3
        ckpt_k = ckpt_indices.shape[-1]
        ckpt_t_off = 0 if ckpt_k == T else 1
        assert ckpt_t_off + T <= ckpt_k, (ckpt_k, T)
    else:
        vh = KD = VD = QKV = 0
        ssm_u32_rows = conv_rows = conv_rows_pad = qkv_128 = 0
        direct_conv = tail_row = False
        conv_buf_dt = BF16
        ckpt_is_3d = False
        ckpt_t_off = 0
        taps = 0
    geom = dict(
        vh=vh,
        KD=KD,
        VD=VD,
        QKV=QKV,
        taps=taps,
        ssm_u32_rows=ssm_u32_rows,
        conv_rows=conv_rows,
        conv_rows_pad=conv_rows_pad,
        qkv_128=qkv_128,
        direct_conv=direct_conv,
        conv_tail_row=direct_conv and tail_row,
        ckpt_is_3d=ckpt_is_3d,
        ckpt_t_off=ckpt_t_off,
        ckpt_ndim=ckpt_indices.ndim if has_gdn else 0,
    )
    return conv_in, conv_buf_shape, conv_buf_dt, geom


def pb_for(kc: ScatterKernel, sub: Any, s: Any, t: Any, span: int) -> Any:
    """First page of sequence `s`'s checkpoint slot `t` in recurrent slot `sub`."""
    sg, ckpt_ref = kc.sg, kc.ckpt_ref
    layout, ex, interpret, ckpt_is_3d = sg.layout, sg.ex, sg.interpret, sg.ckpt_is_3d
    ckpt_t_off, ckpt_ndim = sg.ckpt_t_off, sg.ckpt_ndim
    slot = (
        ckpt_ref[sub, s, t + ckpt_t_off] if ckpt_is_3d else ckpt_ref[s, t + ckpt_t_off]
    )
    pb = layout.page_base(slot, sub, ckpt_ndim)
    kernel_debug_check(
        jnp.logical_and(pb >= 0, pb + span <= cache_exec(ex).num_pages),
        "GDN scatter checkpoint slot maps outside [0, num_pages)",
        interpret=bool(interpret),
    )
    return jnp.clip(pb, 0, cache_exec(ex).num_pages - span)


def is_contig_ckpt(kc: ScatterKernel, sub: Any, s: Any) -> Any:
    """Whether sequence `s`'s `T` checkpoint slots are consecutive."""
    sg, ckpt_ref = kc.sg, kc.ckpt_ref
    T, ckpt_is_3d, ckpt_t_off = sg.T, sg.ckpt_is_3d, sg.ckpt_t_off
    if not ckpt_is_3d or T <= 1:
        return jnp.bool_(T == 1)
    s0 = ckpt_ref[sub, s, ckpt_t_off]
    ok = jnp.bool_(True)
    for t in range(1, T):
        ok = jnp.logical_and(ok, ckpt_ref[sub, s, t + ckpt_t_off] == s0 + t)
    return ok


def slot_copy(kc: ScatterKernel, raw_out_ref: Any, slot: Any, t: Any, pb_t: Any) -> Any:
    """Copy of slot `t`'s live rows only (recurrent pages + conv rows)."""
    sg, page_buf, sem_p, SLOT_ROWS = kc.sg, kc.page_buf, kc.sem_p, kc.SLOT_ROWS
    ex, T, page_size, pps, page_rows = sg.ex, sg.T, sg.page_size, sg.pps, sg.page_rows
    src = page_buf.at[slot].reshape(T * pps * page_rows, 4, page_size)
    dst = raw_out_ref.reshape(cache_exec(ex).num_pages * page_rows, 4, page_size)
    r0 = pl.multiple_of(pb_t * page_rows, page_rows)
    return pltpu.make_async_copy(
        src.at[pl.ds(t * pps * page_rows, SLOT_ROWS)],
        dst.at[pl.ds(r0, SLOT_ROWS)],
        sem_p.at[slot, t],
    )


def start_scatter_seq(
    kc: ScatterKernel, raw_out_ref: Any, sub: Any, s: Any, slot: Any
) -> None:
    """Pack sequence `s`'s states into slot pages; start their pool writes."""
    sg, page_buf, rec_buf, conv_buf = kc.sg, kc.page_buf, kc.rec_buf, kc.conv_buf
    sem_p, w4_row_iota = kc.sem_p, kc.w4_row_iota
    interpret, T, page_shape, pps = sg.interpret, sg.T, sg.page_shape, sg.pps
    rec_pages, rec_u32_cap, conv_page = sg.rec_pages, sg.rec_u32_cap, sg.conv_page
    conv_page_rows, use_conv_partial, vh = sg.conv_page_rows, sg.use_conv_partial, sg.vh
    KD, VD, taps, ssm_u32_rows = sg.KD, sg.VD, sg.taps, sg.ssm_u32_rows
    conv_rows, conv_rows_pad = sg.conv_rows, sg.conv_rows_pad
    direct_conv, conv_tail_row = sg.direct_conv, sg.conv_tail_row

    @pl.when(is_active(kc, s))
    def _():
        conv_bf16: Any = conv_buf[slot, 0].astype(BF16) if conv_tail_row else None
        for t in range(T):
            s_km = rec_buf[slot, pl.ds(t * vh, vh)]
            if bool(interpret):
                bf16_pairs = s_km.reshape(ssm_u32_rows, 2, VD).swapaxes(1, 2)
                bf16_pad = (
                    jnp.pad(
                        bf16_pairs,
                        ((0, rec_u32_cap - ssm_u32_rows), (0, 0), (0, 0)),
                    )
                    if ssm_u32_rows < rec_u32_cap
                    else bf16_pairs
                )
                page_buf[slot, pl.ds(t * pps, rec_pages)] = (
                    lax.bitcast_convert_type(bf16_pad, FP8)
                    .reshape(rec_u32_cap, VD, 4)
                    .swapaxes(1, 2)
                    .reshape(rec_pages, *page_shape)
                )
            else:
                u32_val = pltpu.bitcast(s_km.reshape(vh * KD, VD), U32)
                page_buf.at[slot, pl.ds(t * pps, rec_pages)].bitcast(U32).reshape(
                    rec_u32_cap, LANES
                )[:ssm_u32_rows, :] = u32_val

            if direct_conv:
                if conv_tail_row:
                    w4 = jnp.where(
                        w4_row_iota < taps,
                        conv_bf16[1 + t : 2 + taps + t, :],
                        jnp.bfloat16(0.0),
                    )
                    conv_taps = w4.reshape(conv_rows_pad, LANES)
                else:
                    taps_128 = (
                        conv_buf[slot, 0, 1 + t : 1 + taps + t, :]
                        .astype(BF16)
                        .reshape(conv_rows, LANES)
                    )
                    if conv_rows_pad > conv_rows:
                        taps_128 = jnp.pad(
                            taps_128,
                            ((0, conv_rows_pad - conv_rows), (0, 0)),
                        )
                    conv_taps = taps_128
            else:
                conv_taps = conv_buf[slot, pl.ds(t * conv_rows_pad, conv_rows_pad), :]
            if bool(interpret):
                conv_pad = (
                    jnp.pad(
                        conv_taps,
                        ((0, conv_page_rows - conv_rows_pad), (0, 0)),
                    )
                    if conv_rows_pad < conv_page_rows
                    else conv_taps
                )
                page_buf[slot, t * pps + conv_page] = (
                    lax.bitcast_convert_type(
                        conv_pad.reshape(
                            page_shape[0],
                            page_shape[1],
                            page_shape[2] // 2,
                            page_shape[3],
                        ),
                        FP8,
                    )
                    .swapaxes(3, 4)
                    .reshape(page_shape)
                )
            else:
                page_buf.at[slot, t * pps + conv_page].bitcast(BF16).reshape(
                    conv_page_rows, LANES
                )[:conv_rows_pad, :] = conv_taps

        if use_conv_partial:
            for t in range(T):
                slot_copy(
                    kc, raw_out_ref, slot, t, pb_for(kc, sub, s, t, span=pps)
                ).start()
        elif T == 1:
            pb0 = pb_for(kc, sub, s, 0, span=pps)
            pltpu.make_async_copy(
                page_buf.at[slot],
                raw_out_ref.at[pl.ds(pb0, pps)],
                sem_p.at[slot, 0],
            ).start()
        else:
            contig = is_contig_ckpt(kc, sub, s)

            @pl.when(contig)
            def _():
                pb0 = pb_for(kc, sub, s, 0, span=pps * T)
                pltpu.make_async_copy(
                    page_buf.at[slot],
                    raw_out_ref.at[pl.ds(pb0, pps * T)],
                    sem_p.at[slot, 0],
                ).start()

            @pl.when(jnp.logical_not(contig))
            def _():
                for t in range(T):
                    pb_t = pb_for(kc, sub, s, t, span=pps)
                    pltpu.make_async_copy(
                        page_buf.at[slot, pl.ds(t * pps, pps)],
                        raw_out_ref.at[pl.ds(pb_t, pps)],
                        sem_p.at[slot, t],
                    ).start()


def wait_scatter_seq(
    kc: ScatterKernel, raw_out_ref: Any, sub: Any, s: Any, slot: Any
) -> None:
    """Wait for `start_scatter_seq(raw_out_ref, sub, s, slot)`."""
    sg, page_buf, sem_p = kc.sg, kc.page_buf, kc.sem_p
    T, pps, use_conv_partial = sg.T, sg.pps, sg.use_conv_partial

    @pl.when(is_active(kc, s))
    def _():
        if use_conv_partial:
            for t in range(T):
                slot_copy(
                    kc, raw_out_ref, slot, t, pb_for(kc, sub, s, t, span=pps)
                ).wait()
        elif T == 1:
            pb0 = pb_for(kc, sub, s, 0, span=pps)
            pltpu.make_async_copy(
                page_buf.at[slot],
                raw_out_ref.at[pl.ds(pb0, pps)],
                sem_p.at[slot, 0],
            ).wait()
        else:
            contig = is_contig_ckpt(kc, sub, s)

            @pl.when(contig)
            def _():
                pb0 = pb_for(kc, sub, s, 0, span=pps * T)
                pltpu.make_async_copy(
                    page_buf.at[slot],
                    raw_out_ref.at[pl.ds(pb0, pps * T)],
                    sem_p.at[slot, 0],
                ).wait()

            @pl.when(jnp.logical_not(contig))
            def _():
                for t in range(T):
                    pb_t = pb_for(kc, sub, s, t, span=pps)
                    pltpu.make_async_copy(
                        page_buf.at[slot, pl.ds(t * pps, pps)],
                        raw_out_ref.at[pl.ds(pb_t, pps)],
                        sem_p.at[slot, t],
                    ).wait()

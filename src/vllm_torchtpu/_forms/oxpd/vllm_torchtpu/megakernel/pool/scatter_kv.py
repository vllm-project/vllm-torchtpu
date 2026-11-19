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
"""Pool scatter: new K / V rows into their KV pages (read-modify-write)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import FP8, I32
from .config import cache_exec
from .scatter_ctx import ScatterKernel, batch_to_i32_cols, is_active


def kv_cols(kc: ScatterKernel) -> None:
    """Pre-transpose every pool's new K / V rows into lane columns (`NS > 1`)."""
    sg, knew_buf, vnew_buf, k_cols_buf = kc.sg, kc.knew_buf, kc.vnew_buf, kc.k_cols_buf
    v_cols_buf = kc.v_cols_buf
    NATT, NS = sg.NATT, sg.NS
    if NS > 1:
        for g_idx in range(NATT):
            k_cols_buf[g_idx] = batch_to_i32_cols(kc, knew_buf[g_idx])
            v_cols_buf[g_idx] = batch_to_i32_cols(kc, vnew_buf[g_idx])


def scatter_single_token_across_att(
    kc: ScatterKernel, s_idx: Any, b_idx: Any, t_idx: Any
) -> None:
    """Write row `b_idx`'s K / V into its page of every attention pool."""
    sg, pos_ref, kv_pages_ref = kc.sg, kc.pos_ref, kc.kv_pages_ref
    raw_out_refs, kv_page, knew_buf = kc.raw_out_refs, kc.kv_page, kc.knew_buf
    vnew_buf, sem_kv_rd, sem_kv_wr = kc.vnew_buf, kc.sem_kv_rd, kc.sem_kv_wr
    k_cols_buf, v_cols_buf, lane_iota = kc.k_cols_buf, kc.v_cols_buf, kc.lane_iota
    ex, interpret, NATT, NS = sg.ex, sg.interpret, sg.NATT, sg.NS
    page_size, h4 = sg.page_size, sg.h4
    p = pos_ref[b_idx]
    ln = lax.rem(p, page_size)
    kp = jnp.clip(kv_pages_ref[s_idx, t_idx], 0, cache_exec(ex).num_pages - 1)
    mask_ln = lane_iota == ln
    sh_b = lax.rem(ln + (page_size - lax.rem(b_idx, page_size)), page_size)

    pltpu.make_async_copy(
        raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
    ).start()

    for g_idx in range(NATT):
        k_slot = g_idx & 1
        nxt_slot = 1 - k_slot
        if g_idx >= 1:
            pltpu.make_async_copy(
                kv_page.at[nxt_slot],
                raw_out_refs[g_idx - 1].at[kp],
                sem_kv_wr.at[nxt_slot],
            ).wait()
        if g_idx + 1 < NATT:
            pltpu.make_async_copy(
                raw_out_refs[g_idx + 1].at[kp],
                kv_page.at[nxt_slot],
                sem_kv_rd.at[nxt_slot],
            ).start()
        pltpu.make_async_copy(
            raw_out_refs[g_idx].at[kp],
            kv_page.at[k_slot],
            sem_kv_rd.at[k_slot],
        ).wait()

        k_cols = k_cols_buf[g_idx] if NS > 1 else batch_to_i32_cols(kc, knew_buf[g_idx])
        v_cols = v_cols_buf[g_idx] if NS > 1 else batch_to_i32_cols(kc, vnew_buf[g_idx])
        k_col = pltpu.roll(k_cols, sh_b, 1)
        v_col = pltpu.roll(v_cols, sh_b, 1)
        k_view = kv_page.at[k_slot, 0].bitcast(I32).reshape(h4, page_size)
        v_view = kv_page.at[k_slot, 1].bitcast(I32).reshape(h4, page_size)
        k_new = jnp.where(mask_ln, k_col, k_view[...])
        v_new = jnp.where(mask_ln, v_col, v_view[...])
        if bool(interpret):
            kv_page[k_slot, 0] = lax.bitcast_convert_type(k_new, FP8).swapaxes(1, 2)
            kv_page[k_slot, 1] = lax.bitcast_convert_type(v_new, FP8).swapaxes(1, 2)
        else:
            k_view[...] = k_new
            v_view[...] = v_new

        pltpu.make_async_copy(
            kv_page.at[k_slot],
            raw_out_refs[g_idx].at[kp],
            sem_kv_wr.at[k_slot],
        ).start()

    if NATT >= 1:
        last_slot = (NATT - 1) & 1
        pltpu.make_async_copy(
            kv_page.at[last_slot],
            raw_out_refs[NATT - 1].at[kp],
            sem_kv_wr.at[last_slot],
        ).wait()


def scatter_kv(kc: ScatterKernel) -> None:
    """Scatter every active row (`T == 1`) or sequence (`T > 1`) into its pages."""
    sg, pos_ref, kv_pages_ref = kc.sg, kc.pos_ref, kc.kv_pages_ref
    raw_out_refs, kv_page, knew_buf = kc.raw_out_refs, kc.kv_page, kc.knew_buf
    vnew_buf, sem_kv_rd, sem_kv_wr = kc.vnew_buf, kc.sem_kv_rd, kc.sem_kv_wr
    k_cols_buf, v_cols_buf, lane_iota = kc.k_cols_buf, kc.v_cols_buf, kc.lane_iota
    ex, interpret, NATT, NS, B, T = sg.ex, sg.interpret, sg.NATT, sg.NS, sg.B, sg.T
    page_size, h4 = sg.page_size, sg.h4
    if T == 1:

        def scatter_kv_row(b, _):
            @pl.when(is_active(kc, b))
            def _():
                scatter_single_token_across_att(kc, b, b, 0)

            return 0

        if B == 1:
            scatter_kv_row(0, 0)
        else:
            lax.fori_loop(0, B, scatter_kv_row, 0)
    else:

        def scatter_kv_seq(s, _):
            @pl.when(is_active(kc, s))
            def _():
                b0 = s * T
                kp0 = jnp.clip(kv_pages_ref[s, 0], 0, cache_exec(ex).num_pages - 1)
                same_page = jnp.bool_(True)
                for t in range(1, T):
                    same_page = jnp.logical_and(
                        same_page,
                        jnp.clip(kv_pages_ref[s, t], 0, cache_exec(ex).num_pages - 1)
                        == kp0,
                    )

                @pl.when(same_page)
                def _():
                    kp = kp0
                    ln_list = [lax.rem(pos_ref[b0 + t], page_size) for t in range(T)]
                    mask_ln_list = [lane_iota == ln_list[t] for t in range(T)]
                    sh_list = [
                        lax.rem(
                            ln_list[t] + (page_size - lax.rem(b0 + t, page_size)),
                            page_size,
                        )
                        for t in range(T)
                    ]

                    pltpu.make_async_copy(
                        raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
                    ).start()

                    for g_idx in range(NATT):
                        k_slot = g_idx & 1
                        nxt_slot = 1 - k_slot
                        if g_idx >= 1:
                            pltpu.make_async_copy(
                                kv_page.at[nxt_slot],
                                raw_out_refs[g_idx - 1].at[kp],
                                sem_kv_wr.at[nxt_slot],
                            ).wait()
                        if g_idx + 1 < NATT:
                            pltpu.make_async_copy(
                                raw_out_refs[g_idx + 1].at[kp],
                                kv_page.at[nxt_slot],
                                sem_kv_rd.at[nxt_slot],
                            ).start()
                        pltpu.make_async_copy(
                            raw_out_refs[g_idx].at[kp],
                            kv_page.at[k_slot],
                            sem_kv_rd.at[k_slot],
                        ).wait()

                        k_cols = (
                            k_cols_buf[g_idx]
                            if NS > 1
                            else batch_to_i32_cols(kc, knew_buf[g_idx])
                        )
                        v_cols = (
                            v_cols_buf[g_idx]
                            if NS > 1
                            else batch_to_i32_cols(kc, vnew_buf[g_idx])
                        )
                        k_view = (
                            kv_page.at[k_slot, 0].bitcast(I32).reshape(h4, page_size)
                        )
                        v_view = (
                            kv_page.at[k_slot, 1].bitcast(I32).reshape(h4, page_size)
                        )
                        k_cur = k_view[...]
                        v_cur = v_view[...]
                        for t in range(T):
                            k_col = pltpu.roll(k_cols, sh_list[t], 1)
                            v_col = pltpu.roll(v_cols, sh_list[t], 1)
                            k_cur = jnp.where(mask_ln_list[t], k_col, k_cur)
                            v_cur = jnp.where(mask_ln_list[t], v_col, v_cur)
                        if bool(interpret):
                            kv_page[k_slot, 0] = lax.bitcast_convert_type(
                                k_cur, FP8
                            ).swapaxes(1, 2)
                            kv_page[k_slot, 1] = lax.bitcast_convert_type(
                                v_cur, FP8
                            ).swapaxes(1, 2)
                        else:
                            k_view[...] = k_cur
                            v_view[...] = v_cur

                        pltpu.make_async_copy(
                            kv_page.at[k_slot],
                            raw_out_refs[g_idx].at[kp],
                            sem_kv_wr.at[k_slot],
                        ).start()

                    if NATT >= 1:
                        last_slot = (NATT - 1) & 1
                        pltpu.make_async_copy(
                            kv_page.at[last_slot],
                            raw_out_refs[NATT - 1].at[kp],
                            sem_kv_wr.at[last_slot],
                        ).wait()

                @pl.when(jnp.logical_not(same_page))
                def _():
                    for t in range(T):
                        scatter_single_token_across_att(kc, s, b0 + t, t)

            return 0

        if NS == 1:
            scatter_kv_seq(0, 0)
        else:
            lax.fori_loop(0, NS, scatter_kv_seq, 0)

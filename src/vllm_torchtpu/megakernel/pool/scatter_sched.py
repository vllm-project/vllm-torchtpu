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
"""Pool scatter: GDN state reads and the double-buffered checkpoint pipeline."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import BF16, U32
from .layout import LANES
from .scatter_ckpt import start_scatter_seq, wait_scatter_seq
from .scatter_ctx import ScatterKernel, is_active


def start_seq_in(kc: ScatterKernel, o: Any, s: Any, in_slot: Any) -> None:
    """Start reading GDN layer `o`'s conv / rec state of sequence `s`."""
    sg, conv_ref, rec_ref, rec_buf = kc.sg, kc.conv_ref, kc.rec_ref, kc.rec_buf
    conv_buf, sem_in = kc.conv_buf, kc.sem_in
    T, vh, conv_rows_pad, direct_conv = sg.T, sg.vh, sg.conv_rows_pad, sg.direct_conv

    @pl.when(is_active(kc, s))
    def _():
        r_base = (
            s * T * vh if isinstance(s, int) else pl.multiple_of(s * T * vh, T * vh)
        )
        if direct_conv:
            pltpu.make_async_copy(
                conv_ref.at[o, pl.ds(s, 1)],
                conv_buf.at[in_slot],
                sem_in.at[in_slot, 0],
            ).start()
        else:
            c_base = (
                s * T * conv_rows_pad
                if isinstance(s, int)
                else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
            )
            pltpu.make_async_copy(
                conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                conv_buf.at[in_slot],
                sem_in.at[in_slot, 0],
            ).start()
        pltpu.make_async_copy(
            rec_ref.at[o, pl.ds(r_base, T * vh)],
            rec_buf.at[in_slot],
            sem_in.at[in_slot, 1],
        ).start()


def wait_seq_in(kc: ScatterKernel, o: Any, s: Any, in_slot: Any) -> None:
    """Wait for `start_seq_in(o, s, in_slot)`."""
    sg, conv_ref, rec_ref, rec_buf = kc.sg, kc.conv_ref, kc.rec_ref, kc.rec_buf
    conv_buf, sem_in = kc.conv_buf, kc.sem_in
    T, vh, conv_rows_pad, direct_conv = sg.T, sg.vh, sg.conv_rows_pad, sg.direct_conv

    @pl.when(is_active(kc, s))
    def _():
        r_base = (
            s * T * vh if isinstance(s, int) else pl.multiple_of(s * T * vh, T * vh)
        )
        if direct_conv:
            pltpu.make_async_copy(
                conv_ref.at[o, pl.ds(s, 1)],
                conv_buf.at[in_slot],
                sem_in.at[in_slot, 0],
            ).wait()
        else:
            c_base = (
                s * T * conv_rows_pad
                if isinstance(s, int)
                else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
            )
            pltpu.make_async_copy(
                conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                conv_buf.at[in_slot],
                sem_in.at[in_slot, 0],
            ).wait()
        pltpu.make_async_copy(
            rec_ref.at[o, pl.ds(r_base, T * vh)],
            rec_buf.at[in_slot],
            sem_in.at[in_slot, 1],
        ).wait()


def ckpt_prologue(kc: ScatterKernel) -> None:
    """Start the first state reads; zero the slot pages' unused tails."""
    sg, page_buf, total_gdn = kc.sg, kc.page_buf, kc.total_gdn
    interpret, NS, T, G_PER_ATT, pps = sg.interpret, sg.NS, sg.T, sg.G_PER_ATT, sg.pps
    rec_pages, rec_u32_cap, conv_page = sg.rec_pages, sg.rec_u32_cap, sg.conv_page
    conv_page_rows, use_conv_partial = sg.conv_page_rows, sg.use_conv_partial
    ssm_u32_rows = sg.ssm_u32_rows
    if total_gdn > 0 and NS >= 1:
        start_seq_in(kc, 0, 0, 0)
        if NS >= 2:
            start_seq_in(kc, 0, 1, 1)
        elif total_gdn >= 2:
            start_seq_in(kc, 1, 0, 1)

    if G_PER_ATT > 0:
        for slot in range(2):
            if bool(interpret):
                page_buf[slot] = jnp.zeros_like(page_buf[slot])
            else:
                if ssm_u32_rows < rec_u32_cap:
                    for t in range(T):
                        page_buf.at[slot, pl.ds(t * pps, rec_pages)].bitcast(
                            U32
                        ).reshape(rec_u32_cap, LANES)[...] = jnp.zeros(
                            (rec_u32_cap, LANES), U32
                        )
                if not use_conv_partial:
                    for t in range(T):
                        page_buf.at[slot, t * pps + conv_page].bitcast(BF16).reshape(
                            conv_page_rows, LANES
                        )[...] = jnp.zeros((conv_page_rows, LANES), BF16)


def scatter_ckpt_states(kc: ScatterKernel) -> None:
    """Read, pack and write every GDN layer's states (`NS`-specialized order)."""
    sg, raw_out_refs, total_gdn = kc.sg, kc.raw_out_refs, kc.total_gdn
    NATT, NS, G_PER_ATT, n_pairs = sg.NATT, sg.NS, sg.G_PER_ATT, sg.n_pairs
    if total_gdn > 0:
        if NS == 1:
            for g_idx in range(NATT):
                raw_out_ref = raw_out_refs[g_idx]
                for sub in range(G_PER_ATT):
                    o = g_idx * G_PER_ATT + sub
                    slot = o & 1
                    if o >= 2:
                        prev_o = o - 2
                        wait_scatter_seq(
                            kc,
                            raw_out_refs[prev_o // G_PER_ATT],
                            prev_o % G_PER_ATT,
                            0,
                            slot,
                        )
                    wait_seq_in(kc, o, 0, slot)
                    start_scatter_seq(kc, raw_out_ref, sub, 0, slot)
                    if o + 2 < total_gdn:
                        start_seq_in(kc, o + 2, 0, slot)

            for o_tail in range(max(0, total_gdn - 2), total_gdn):
                wait_scatter_seq(
                    kc,
                    raw_out_refs[o_tail // G_PER_ATT],
                    o_tail % G_PER_ATT,
                    0,
                    o_tail & 1,
                )
        elif NS == 2:
            for g_idx in range(NATT):
                raw_out_ref = raw_out_refs[g_idx]
                for sub in range(G_PER_ATT):
                    o = g_idx * G_PER_ATT + sub
                    if o >= 1:
                        prev_o = o - 1
                        wait_scatter_seq(
                            kc,
                            raw_out_refs[prev_o // G_PER_ATT],
                            prev_o % G_PER_ATT,
                            0,
                            0,
                        )
                    wait_seq_in(kc, o, 0, 0)
                    start_scatter_seq(kc, raw_out_ref, sub, 0, 0)
                    if o + 1 < total_gdn:
                        start_seq_in(kc, o + 1, 0, 0)

                    if o >= 1:
                        prev_o = o - 1
                        wait_scatter_seq(
                            kc,
                            raw_out_refs[prev_o // G_PER_ATT],
                            prev_o % G_PER_ATT,
                            1,
                            1,
                        )
                    wait_seq_in(kc, o, 1, 1)
                    start_scatter_seq(kc, raw_out_ref, sub, 1, 1)
                    if o + 1 < total_gdn:
                        start_seq_in(kc, o + 1, 1, 1)

            last_o = total_gdn - 1
            wait_scatter_seq(
                kc, raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 0, 0
            )
            wait_scatter_seq(
                kc, raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
            )
        elif NS == 3:
            for g_idx in range(NATT):
                raw_out_ref = raw_out_refs[g_idx]
                for sub in range(G_PER_ATT):
                    o = g_idx * G_PER_ATT + sub
                    if o >= 1:
                        prev_o = o - 1
                        wait_scatter_seq(
                            kc,
                            raw_out_refs[prev_o // G_PER_ATT],
                            prev_o % G_PER_ATT,
                            2,
                            0,
                        )
                    wait_seq_in(kc, o, 0, 0)
                    start_scatter_seq(kc, raw_out_ref, sub, 0, 0)
                    start_seq_in(kc, o, 2, 0)

                    if o >= 1:
                        prev_o = o - 1
                        wait_scatter_seq(
                            kc,
                            raw_out_refs[prev_o // G_PER_ATT],
                            prev_o % G_PER_ATT,
                            1,
                            1,
                        )
                    wait_seq_in(kc, o, 1, 1)
                    start_scatter_seq(kc, raw_out_ref, sub, 1, 1)
                    if o + 1 < total_gdn:
                        start_seq_in(kc, o + 1, 1, 1)

                    wait_scatter_seq(kc, raw_out_ref, sub, 0, 0)
                    wait_seq_in(kc, o, 2, 0)
                    start_scatter_seq(kc, raw_out_ref, sub, 2, 0)
                    if o + 1 < total_gdn:
                        start_seq_in(kc, o + 1, 0, 0)

            last_o = total_gdn - 1
            wait_scatter_seq(
                kc, raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
            )
            wait_scatter_seq(
                kc, raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 2, 0
            )
        else:
            for g_idx in range(NATT):
                raw_out_ref = raw_out_refs[g_idx]
                for sub in range(G_PER_ATT):
                    o = g_idx * G_PER_ATT + sub

                    def scatter_pair(pair, _, raw_out_ref=raw_out_ref, sub=sub, o=o):
                        s0 = 2 * pair
                        s1 = s0 + 1

                        @pl.when(pair > 0)
                        def _():
                            wait_scatter_seq(kc, raw_out_ref, sub, s0 - 2, 0)

                        wait_seq_in(kc, o, s0, 0)
                        start_scatter_seq(kc, raw_out_ref, sub, s0, 0)

                        @pl.when(s0 + 2 < NS)
                        def _():
                            start_seq_in(kc, o, s0 + 2, 0)

                        @pl.when(pair > 0)
                        def _():
                            wait_scatter_seq(kc, raw_out_ref, sub, s1 - 2, 1)

                        wait_seq_in(kc, o, s1, 1)
                        start_scatter_seq(kc, raw_out_ref, sub, s1, 1)

                        @pl.when(s1 + 2 < NS)
                        def _():
                            start_seq_in(kc, o, s1 + 2, 1)

                        return 0

                    lax.fori_loop(0, n_pairs, scatter_pair, 0)

                    if NS % 2 == 1:
                        wait_scatter_seq(kc, raw_out_ref, sub, NS - 3, 0)
                        wait_seq_in(kc, o, NS - 1, 0)
                        start_scatter_seq(kc, raw_out_ref, sub, NS - 1, 0)
                        if o + 1 < total_gdn:
                            start_seq_in(kc, o + 1, 0, 0)
                            start_seq_in(kc, o + 1, 1, 1)
                        wait_scatter_seq(kc, raw_out_ref, sub, NS - 2, 1)
                        wait_scatter_seq(kc, raw_out_ref, sub, NS - 1, 0)
                    else:
                        if o + 1 < total_gdn:
                            start_seq_in(kc, o + 1, 0, 0)
                            start_seq_in(kc, o + 1, 1, 1)
                        wait_scatter_seq(kc, raw_out_ref, sub, NS - 2, 0)
                        wait_scatter_seq(kc, raw_out_ref, sub, NS - 1, 1)

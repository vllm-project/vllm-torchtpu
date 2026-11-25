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
"""Pallas GDN gather kernel: per-rank `conv` / `rec` states from raw pools."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, FP8, I32, U32, rup
from .backend import check_ckpt_bounds, kernel_debug_check, resolve_active_seqs
from .config import cache_exec
from .layout import LANES, PoolLayout

# Pallas kernel name (`GATHER_KERNEL_NAME` is a substring of it).
_PALLAS_NAME = "qwen3_5_gather_gdn_states"


# ---------------------------------------------------------------------------
# Pallas GDN Gather Kernel
# ---------------------------------------------------------------------------
def gather_gdn_states(
    layout: PoolLayout,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    ckpt_indices: jax.Array,
    read_offsets: jax.Array,
    active_seqs: jax.Array | str | None = None,
    *,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gather per-rank GDN `conv` and `rec` states from per-attention `raw_pools`.

    Args:
      layout: Pool layout (`num_pools`, `slots_per_pool`, page and slot geometry,
        recurrent / conv state geometry).
      ex: Execution spec (`B`, `T`, `NS`, `R`, `page_size`, `num_pages`).
      raw_pools: Tuple of `layout.num_pools` per-rank raw pool arrays, each of
        shape `(num_pages, *layout.kv_page_shape) fp8_e4m3fn`.
      ckpt_indices: `int32` checkpoint slot table, shape `(G_PER_ATT, NS, K_ckpt)`
        or `(NS, K_ckpt)`, mapped to pages by `layout.page_base` (3D:
        `slot * pps`; 2D: `slot * (pps * G_PER_ATT) + sub * pps`).
      read_offsets: `int32` index into `ckpt_indices` axis `-1` for the input
        checkpoint slot per sequence, shape `(G_PER_ATT, NS)` or `(NS,)`. A
        negative offset (`read_offsets < 0`) indicates a cold-start sequence
        with no prior GDN state in the pool; the gather skips the pool DMA for
        that `(sub, s)` and outputs zero `conv` and `rec` state.
      active_seqs: Optional activity mask (`None` or `"from_ckpt"` to check
        `ckpt_indices[..., 0] > 0`, `"all"` for all sequences active, or
        `(NS,) int32` mask).
      interpret: Override `ex.interpret`.

    Returns:
      `(conv, rec)` per-rank recurrent and causal-conv state tensors.
    """
    layout.validate(ex)
    if interpret is None:
        interpret = ex.interpret
    NATT = layout.num_pools
    assert len(raw_pools) == NATT, (len(raw_pools), NATT)
    G_PER_ATT = layout.slots_per_pool
    act_mode, active_mask = resolve_active_seqs(
        active_seqs, default_mode="from_ckpt" if G_PER_ATT > 0 else "all"
    )
    check_ckpt_bounds(
        ckpt_indices,
        cache_exec(ex).num_pages,
        layout,
        act_mode=act_mode,
        active_mask=active_mask,
    )

    vh, KD, VD = layout.vh_per_rank(ex.R), layout.KD, layout.VD
    QKV = layout.qkv_per_rank(ex.R)
    NS, B, T = ex.NS, ex.B, ex.T
    taps = layout.conv_taps
    CR = rup(taps * B, 8)
    HR = rup(taps + T, 8)
    ngdn = max(1, layout.num_recurrent_layers)
    conv_shape = (ngdn, CR, QKV) if T == 1 else (ngdn, NS, HR, QKV)
    conv_dt = F32
    rec_shape = (ngdn, NS * vh, KD, VD)
    rec_dt = BF16

    pps = layout.pages_per_slot
    rec_pages = layout.rec_pages_per_slot
    rec_u32_cap = layout.rec_u32_rows
    conv_page = layout.conv_page
    conv_page_rows = layout.conv_page_rows
    ssm_u32_rows = (vh * KD) // 2
    conv_rows = (taps * QKV) // LANES
    conv_rows_pad = ((conv_rows + 15) // 16) * 16
    qkv_128 = QKV // LANES
    direct_conv = T > 1 and ((HR * qkv_128) % 16 == 0 or bool(interpret))
    hr_rows_128 = HR * qkv_128
    conv_raw_shape = conv_shape if direct_conv else (ngdn, NS * conv_rows_pad, LANES)
    conv_raw_dt = conv_dt if direct_conv else BF16
    ckpt_is_3d = ckpt_indices.ndim == 3
    roff_is_2d = read_offsets.ndim == 2
    n_pairs = NS // 2

    coal_rec = (G_PER_ATT > 0) and (NS * vh <= 128)
    rec_coal = max(1, G_PER_ATT) if coal_rec else 1

    scratch = {
        "page_buf": pltpu.VMEM((2, pps, *layout.kv_page_shape), FP8),
        "rec_buf": pltpu.VMEM((2, rec_coal, *rec_shape[1:]), rec_dt),
        "conv_buf": pltpu.VMEM(
            (2, max(1, G_PER_ATT), *conv_raw_shape[1:]), conv_raw_dt
        ),
        "sem_in": pltpu.SemaphoreType.DMA((2,)),
        "sem_out": pltpu.SemaphoreType.DMA((2, 2)),
    }
    s_names = list(scratch)
    n_smem = 3 if act_mode == "ref" else 2

    def _kernel(*refs):
        raw_refs = refs[:NATT]
        act_ref: Any = None
        if act_mode == "ref":
            ckpt_ref, roff_ref, act_ref = refs[NATT : NATT + 3]
        else:
            ckpt_ref, roff_ref = refs[NATT : NATT + 2]
        conv_out_ref, rec_out_ref = refs[NATT + n_smem : NATT + n_smem + 2]
        scr = dict(zip(s_names, refs[NATT + n_smem + 2 :]))
        page_buf = scr["page_buf"]
        rec_buf = scr["rec_buf"]
        conv_buf = scr["conv_buf"]
        sem_in = scr["sem_in"]
        sem_out = scr["sem_out"]
        conv_row_iota: Any = (
            lax.broadcasted_iota(I32, (HR, QKV), 0) if direct_conv else None
        )

        for slot in range(2):
            conv_buf[slot] = jnp.zeros(conv_buf.shape[1:], conv_raw_dt)
            rec_buf[slot] = jnp.zeros(rec_buf.shape[1:], rec_dt)

        def is_active(s):
            if act_mode == "all":
                return jnp.bool_(True)
            if act_mode == "from_ckpt":
                return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
            return act_ref[s] != 0

        def has_input_state(sub, s):
            roff = roff_ref[sub, s] if roff_is_2d else roff_ref[s]
            return jnp.logical_and(is_active(s), roff >= 0)

        def page_base_for(sub, s):
            roff = jnp.maximum(
                roff_ref[sub, s] if roff_is_2d else roff_ref[s], jnp.int32(0)
            )
            slot = ckpt_ref[sub, s, roff] if ckpt_is_3d else ckpt_ref[s, roff]
            pb = layout.page_base(slot, sub, ckpt_indices.ndim)
            kernel_debug_check(
                jnp.logical_and(pb >= 0, pb + pps <= cache_exec(ex).num_pages),
                "GDN gather checkpoint slot maps outside [0, num_pages)",
                interpret=bool(interpret),
            )
            return jnp.clip(pb, 0, cache_exec(ex).num_pages - pps)

        def start_page_read(raw_ref, sub, s, p_slot):
            @pl.when(has_input_state(sub, s))
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, pps)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).start()

        def finish_page_read(raw_ref, sub, s, p_slot, g_slot, r_slot, r_sub):
            has_st = has_input_state(sub, s)
            h0 = s * vh if isinstance(s, int) else pl.multiple_of(s * vh, vh)

            @pl.when(has_st)
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, pps)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).wait()

                u32_val = (
                    page_buf.at[p_slot, pl.ds(0, rec_pages)]
                    .bitcast(U32)
                    .reshape(rec_u32_cap, LANES)[:ssm_u32_rows, :]
                )
                s_km = pltpu.bitcast(u32_val, BF16).reshape(vh, KD, VD)
                rec_buf[r_slot, r_sub, pl.ds(h0, vh)] = s_km

                if direct_conv:
                    raw_hr = (
                        page_buf.at[p_slot, conv_page]
                        .bitcast(BF16)
                        .reshape(conv_page_rows, LANES)[:hr_rows_128, :]
                        .reshape(HR, QKV)
                        .astype(conv_dt)
                    )
                    conv_buf[g_slot, sub, s] = jnp.where(
                        conv_row_iota < taps, raw_hr, 0.0
                    )
                else:
                    c0 = (
                        s * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * conv_rows_pad, conv_rows_pad)
                    )
                    conv_buf[g_slot, sub, pl.ds(c0, conv_rows_pad), :] = (
                        page_buf.at[p_slot, conv_page]
                        .bitcast(BF16)
                        .reshape(conv_page_rows, LANES)[:conv_rows_pad, :]
                    )

            @pl.when(jnp.logical_not(has_st))
            def _():
                rec_buf[r_slot, r_sub, pl.ds(h0, vh)] = jnp.zeros((vh, KD, VD), rec_dt)
                if direct_conv:
                    conv_buf[g_slot, sub, s] = jnp.zeros((HR, QKV), conv_raw_dt)
                else:
                    c0 = (
                        s * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * conv_rows_pad, conv_rows_pad)
                    )
                    conv_buf[g_slot, sub, pl.ds(c0, conv_rows_pad), :] = jnp.zeros(
                        (conv_rows_pad, LANES), conv_raw_dt
                    )

        total_gdn = NATT * G_PER_ATT
        if total_gdn > 0 and NS >= 1:
            start_page_read(raw_refs[0], 0, 0, 0)

        for g_idx in range(NATT):
            raw_ref = raw_refs[g_idx]
            g_slot = g_idx & 1
            if G_PER_ATT > 0 and g_idx >= 2:
                o_prev = (g_idx - 2) * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).wait()

            for sub in range(G_PER_ATT):
                o = g_idx * G_PER_ATT + sub
                r_slot = g_slot if coal_rec else (o & 1)
                r_sub = sub if coal_rec else 0
                if not coal_rec and o >= 2:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o - 2],
                        sem_out.at[r_slot, 1],
                    ).wait()

                next_o = o + 1
                has_next_o = next_o < total_gdn
                next_raw_ref = raw_refs[next_o // G_PER_ATT] if has_next_o else None
                next_sub = (next_o % G_PER_ATT) if has_next_o else 0

                if NS == 1:
                    p_cur = o & 1
                    p_nxt = (o + 1) & 1
                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p_nxt)
                    finish_page_read(raw_ref, sub, 0, p_cur, g_slot, r_slot, r_sub)
                elif NS % 2 == 0:
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, 1)
                        finish_page_read(raw_ref, sub, 0, 0, g_slot, r_slot, r_sub)
                        if has_next_o:
                            start_page_read(next_raw_ref, next_sub, 0, 0)
                        finish_page_read(raw_ref, sub, 1, 1, g_slot, r_slot, r_sub)
                    else:

                        def gather_pair_even(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            has_next_o=has_next_o,
                            next_raw_ref=next_raw_ref,
                            next_sub=next_sub,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, 1)
                            finish_page_read(raw_ref, sub, s0, 0, g_slot, r_slot, r_sub)

                            @pl.when(s0 + 2 < NS)
                            def _():
                                start_page_read(raw_ref, sub, s0 + 2, 0)

                            if has_next_o:

                                @pl.when(s0 + 2 == NS)
                                def _():
                                    start_page_read(next_raw_ref, next_sub, 0, 0)

                            finish_page_read(raw_ref, sub, s1, 1, g_slot, r_slot, r_sub)
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_even, 0)
                else:
                    p0 = o & 1
                    p1 = 1 - p0
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, p1)
                        finish_page_read(raw_ref, sub, 0, p0, g_slot, r_slot, r_sub)
                        start_page_read(raw_ref, sub, 2, p0)
                        finish_page_read(raw_ref, sub, 1, p1, g_slot, r_slot, r_sub)
                    else:

                        def gather_pair_odd(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            p0=p0,
                            p1=p1,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, p1)
                            finish_page_read(
                                raw_ref, sub, s0, p0, g_slot, r_slot, r_sub
                            )
                            start_page_read(raw_ref, sub, s0 + 2, p0)
                            finish_page_read(
                                raw_ref, sub, s1, p1, g_slot, r_slot, r_sub
                            )
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_odd, 0)

                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p1)
                    finish_page_read(raw_ref, sub, NS - 1, p0, g_slot, r_slot, r_sub)

                if not coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o],
                        sem_out.at[r_slot, 1],
                    ).start()

            if G_PER_ATT > 0:
                o_base = g_idx * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).start()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).start()

        if G_PER_ATT > 0:
            for g_tail in range(max(0, NATT - 2), NATT):
                tail_slot = g_tail & 1
                o_tail = g_tail * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[tail_slot],
                    conv_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                    sem_out.at[tail_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[tail_slot],
                        rec_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                        sem_out.at[tail_slot, 1],
                    ).wait()
            if not coal_rec:
                for o_tail in range(max(0, total_gdn - 2), total_gdn):
                    r_tail = o_tail & 1
                    pltpu.make_async_copy(
                        rec_buf.at[r_tail, 0],
                        rec_out_ref.at[o_tail],
                        sem_out.at[r_tail, 1],
                    ).wait()
        else:
            cp_c = pltpu.make_async_copy(
                conv_buf.at[0, 0], conv_out_ref.at[0], sem_out.at[0, 0]
            )
            cp_r = pltpu.make_async_copy(
                rec_buf.at[0, 0], rec_out_ref.at[0], sem_out.at[0, 1]
            )
            cp_c.start()
            cp_r.start()
            cp_c.wait()
            cp_r.wait()

    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * NATT + [
        pl.BlockSpec(memory_space=pltpu.SMEM)
    ] * n_smem
    out_shape = (
        [
            jax.ShapeDtypeStruct(conv_raw_shape, conv_raw_dt),
            jax.ShapeDtypeStruct(rec_shape, rec_dt),
        ]
        if interpret
        else [
            pltpu.HBM(conv_raw_shape, conv_raw_dt),
            pltpu.HBM(rec_shape, rec_dt),
        ]
    )
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    call = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
            vmem_limit_bytes=ex.resolved_tiles.vmem_mib << 20,
        ),
        interpret=bool(interpret),
        name=_PALLAS_NAME,
    )
    smem_args = (
        (ckpt_indices, read_offsets, active_mask)
        if act_mode == "ref"
        else (ckpt_indices, read_offsets)
    )
    conv_raw, rec_out = call(*raw_pools, *smem_args)
    if direct_conv:
        return conv_raw, rec_out
    taps_all = (
        conv_raw.reshape(ngdn, NS, conv_rows_pad, LANES)[:, :, :conv_rows, :]
        .reshape(ngdn, NS, taps, QKV)
        .astype(conv_dt)
    )
    if T > 1:
        conv_out = jnp.pad(taps_all, ((0, 0), (0, 0), (0, HR - taps), (0, 0)))
    else:
        conv_3b = jnp.swapaxes(taps_all, 1, 2).reshape(ngdn, taps * B, QKV)
        conv_out = jnp.pad(conv_3b, ((0, 0), (0, CR - taps * B), (0, 0)))
    return conv_out, rec_out

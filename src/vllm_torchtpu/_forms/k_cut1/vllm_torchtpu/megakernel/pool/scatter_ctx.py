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
"""Pool scatter kernel state: static geometry and the kernel's refs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from ..ops import BF16, FP8, I32
from .layout import PoolLayout


@dataclass(frozen=True)
class ScatterGeom:
    """Static (trace-time) geometry of one `scatter_pool_states` call.

    The GDN fields (`vh` .. `ckpt_ndim`) are zero when the pools hold no
    recurrent slots; `direct_conv` streams the conv window straight from the
    caller's `(.., NS, rows, QKV)` conv tensor instead of a packed copy.
    """

    layout: PoolLayout
    ex: Any
    interpret: bool
    NATT: int
    NS: int
    B: int
    T: int
    MR: int
    G_PER_ATT: int
    page_size: int
    HD: int
    h4: int
    page_shape: tuple[int, ...]
    pps: int
    rec_pages: int
    rec_u32_cap: int
    conv_page: int
    conv_page_rows: int
    page_rows: int
    has_gdn: bool
    use_conv_partial: bool
    act_mode: str
    n_smem: int
    n_hbm: int
    s_names: tuple[str, ...]
    n_pairs: int
    vh: int = 0
    KD: int = 0
    VD: int = 0
    QKV: int = 0
    taps: int = 0
    ssm_u32_rows: int = 0
    conv_rows: int = 0
    conv_rows_pad: int = 0
    qkv_128: int = 0
    direct_conv: bool = False
    # Direct path reading `conv_taps + 1` rows per token (exactly the padded
    # window); False: `conv_taps` rows plus zero padding.
    conv_tail_row: bool = False
    ckpt_is_3d: bool = False
    ckpt_t_off: int = 0
    ckpt_ndim: int = 0


@dataclass(frozen=True)
class ScatterKernel:
    """The scatter kernel's refs (inputs, outputs, scratch) and lane iotas."""

    sg: ScatterGeom
    pos_ref: Any
    kv_pages_ref: Any
    ckpt_ref: Any
    act_ref: Any
    knew_ref: Any
    vnew_ref: Any
    conv_ref: Any
    rec_ref: Any
    raw_out_refs: Any
    kv_page: Any
    knew_buf: Any
    vnew_buf: Any
    page_buf: Any
    rec_buf: Any
    conv_buf: Any
    sem_kv_rd: Any
    sem_kv_wr: Any
    sem_kv_in: Any
    sem_in: Any
    sem_p: Any
    k_cols_buf: Any
    v_cols_buf: Any
    lane_iota: Any
    w4_row_iota: Any
    total_gdn: int
    # `(rows, 4, page_size)` fp8 view of the slot pages: the recurrent pages
    # plus the live conv rows (one such row holds two bf16 lane-rows).
    CP_ROWS: int
    SLOT_ROWS: int

    @classmethod
    def build(cls, sg: ScatterGeom, refs: Any) -> ScatterKernel:
        """Split the kernel's `refs`; trace the lane iotas (in kernel order)."""
        NATT, NS, QKV, h4, page_size = sg.NATT, sg.NS, sg.QKV, sg.h4, sg.page_size
        has_gdn, act_mode, n_smem, n_hbm = sg.has_gdn, sg.act_mode, sg.n_smem, sg.n_hbm
        direct_conv, s_names, G_PER_ATT = sg.direct_conv, sg.s_names, sg.G_PER_ATT
        taps = sg.taps
        smem_refs = refs[NATT : NATT + n_smem]
        pos_ref, kv_pages_ref = smem_refs[0], smem_refs[1]
        ckpt_ref: Any = smem_refs[2] if has_gdn else None
        act_ref: Any = smem_refs[-1] if act_mode == "ref" else None
        hbm_refs: tuple[Any, ...] = (
            refs[NATT + n_smem : NATT + n_smem + n_hbm] + (None, None)
        )[:4]
        knew_ref, vnew_ref, conv_ref, rec_ref = hbm_refs
        raw_out_refs = refs[NATT + n_smem + n_hbm : 2 * NATT + n_smem + n_hbm]
        scr = dict(zip(s_names, refs[2 * NATT + n_smem + n_hbm :]))
        kv_page = scr["kv_page"]
        knew_buf = scr["knew_buf"]
        vnew_buf = scr["vnew_buf"]
        page_buf: Any = scr.get("page_buf")
        rec_buf: Any = scr.get("rec_buf")
        conv_buf: Any = scr.get("conv_buf")
        sem_kv_rd = scr["sem_kv_rd"]
        sem_kv_wr = scr["sem_kv_wr"]
        sem_kv_in = scr["sem_kv_in"]
        sem_in: Any = scr.get("sem_in")
        sem_p: Any = scr.get("sem_p")
        k_cols_buf: Any = scr["k_cols_buf"] if NS > 1 else None
        v_cols_buf: Any = scr["v_cols_buf"] if NS > 1 else None
        lane_iota = lax.broadcasted_iota(I32, (h4, page_size), 1)
        w4_row_iota: Any = (
            lax.broadcasted_iota(I32, (taps + 1, QKV), 0) if direct_conv else None
        )
        total_gdn = NATT * G_PER_ATT
        CP_ROWS = sg.conv_rows_pad // 2
        return cls(
            sg=sg,
            pos_ref=pos_ref,
            kv_pages_ref=kv_pages_ref,
            ckpt_ref=ckpt_ref,
            act_ref=act_ref,
            knew_ref=knew_ref,
            vnew_ref=vnew_ref,
            conv_ref=conv_ref,
            rec_ref=rec_ref,
            raw_out_refs=raw_out_refs,
            kv_page=kv_page,
            knew_buf=knew_buf,
            vnew_buf=vnew_buf,
            page_buf=page_buf,
            rec_buf=rec_buf,
            conv_buf=conv_buf,
            sem_kv_rd=sem_kv_rd,
            sem_kv_wr=sem_kv_wr,
            sem_kv_in=sem_kv_in,
            sem_in=sem_in,
            sem_p=sem_p,
            k_cols_buf=k_cols_buf,
            v_cols_buf=v_cols_buf,
            lane_iota=lane_iota,
            w4_row_iota=w4_row_iota,
            total_gdn=total_gdn,
            CP_ROWS=CP_ROWS,
            SLOT_ROWS=sg.rec_pages * sg.page_rows + CP_ROWS,
        )


def is_active(kc: ScatterKernel, s: Any) -> Any:
    """Whether sequence `s` is active (all / from checkpoints / mask ref)."""
    sg, ckpt_ref, act_ref = kc.sg, kc.ckpt_ref, kc.act_ref
    act_mode, ckpt_is_3d = sg.act_mode, sg.ckpt_is_3d
    if act_mode == "all":
        return jnp.bool_(True)
    if act_mode == "from_ckpt":
        return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
    return act_ref[s] != 0


def batch_to_i32_cols(kc: ScatterKernel, tok_mr_hd: Any) -> Any:
    """`[MR, HD]` new K or V rows as `[HD // 4, page_size]` i32 fp8 columns."""
    sg = kc.sg
    MR, page_size = sg.MR, sg.page_size
    rows_bf16 = tok_mr_hd.astype(BF16)
    if page_size > MR:
        rows_bf16 = jnp.pad(rows_bf16, ((0, page_size - MR), (0, 0)))
    cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
    return pltpu.bitcast(cols_fp8, I32)

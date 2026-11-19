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
"""Experimental in-kernel fused KV and GDN pool I/O helpers (`CacheExec.FUSED`).

`GQABlock` / `GDNBlock` reach these helpers only through the block hook
(`fused_io_hook.fused_io()`); this module registers itself there on
import, so the blocks carry no import-time dependency on experimental code.

Contract & Aliasing Model (TPU vs. CPU `interpret=True`):
---------------------------------------------------------
Mosaic TPU rejects `input_output_aliases` on large HBM pool operands inside a
communicating (`collective_id != None`) `pallas_call` by materializing a full-pool
HBM copy before the kernel. Therefore, `cache_exec(ex).FUSED` uses two backend-specific
aliasing contracts:

1. **TPU (`ex.interpret == False`)**:
   - The pool tuple `ST[pool]` (`pool = POOL_STATE_NAME`) is
     declared `access="r"` on the communicating `pallas_call` (`O[pool]` is
     omitted, and `pallas_call` sets no pool `input_output_aliases`).
   - Inside the kernel, `ctx.raw_refs = ST[pool]`; `GQABlock` and `GDNBlock`
     issue async HBM DMA reads and writes directly against `ctx.raw_refs`.
   - The outer `jax.jit` (`donate_argnums` on `raw_pools`) returns the donated
     `raw_pools` input references directly as outputs, so XLA aliases the outer
     input and output buffers in-place without a Pallas-level copy.
   - **Safety requirement**: Correctness on TPU depends on outer buffer donation
     (`jax.buffer_donor = true` / `HloModule input_output_alias`) being honored
     with zero intermediate full-pool copies. See `cache_experimental.py` for
     the opt-in wrappers and HLO checks.

2. **CPU interpret mode (`ex.interpret != False`)**:
   - CPU `interpret=True` does not mutate `access="r"` inputs in-place across the
     `pallas_call` boundary unless aliased to an output.
   - `GQABlock.outputs` declares `O[pool]` (`access="w"`), `kernel.build`
     aliases `ST[pool] -> O[pool]` in `pallas_call(input_output_aliases=...)`,
     and `ctx.raw_refs = O[pool]`.

CPU vs TPU code-path differences (beyond aliasing):
---------------------------------------------------
The `if interpret:` branches below emulate TPU ref views with value ops, so CPU
tests cover the arithmetic but not every TPU lowering:
- FP8 page access: CPU uses value-level `lax.bitcast_convert_type` + `swapaxes`
  on whole pages; TPU uses ref `.bitcast(...).reshape(...)` views (sublane-packed
  `(4, page_size)` layout).
- KV column placement (RMW of the new token's KV lane): CPU selects the column
  with a masked lane sum; TPU rotates it with `pltpu.roll` (same bit pattern).
- GDN recurrent state: CPU materializes the whole `[vh, KD, VD]` state as a value
  and writes all `REC_PAGES` at once; TPU streams per-head reads (from the input
  page at t=0, else from the previous token's output page) and per-head writes.
- GDN input refill: CPU issues the next sequence's gather before the recurrence;
  TPU issues it after (DMA scheduling only).
- Conv checkpoint page: CPU zero-pads the staged taps to the full conv page
  (`PoolLayout.conv_page_rows`);
  TPU writes the live rows only and relies on the zero-initialized staging page.
Only the TPU paths are exercised by the TPU benches (`test_tpu_v7x8_bench.py`).

The fused GDN recurrence is always head-major (`GDNTunables.gdn_head_major=True`
numerics); compare it against the split path with that flag enabled.
"""

from __future__ import annotations

import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.gdn.recurrence import gate_act
from ..exec_spec import ExecSpec
from ..fused_io_hook import FusedPoolIO, register_fused_io
from ..ops import BF16, F32, FP8, I32, U32, bswitch, rnd, rup
from ..pool.config import POOL_STATE_NAME, cache_exec
from ..pool.layout import LANES
from ..spec import StateDecl


def _kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)


# ---------------------------------------------------------------------------
# GQA Fused Pool I/O (`GQABlock` when `CacheExec.FUSED`)
# ---------------------------------------------------------------------------
def gqa_fused_states(ex: ExecSpec) -> list[StateDecl]:
    """Extra SMEM state declarations for `GQABlock` when `CacheExec.FUSED`."""
    if cache_exec(ex).act_mode == "ref":
        return [
            StateDecl(
                "active_seqs",
                (ex.NS,),
                I32,
                access="r",
                per_layer=False,
                memory_space="smem",
            )
        ]
    return []


def gqa_fused_outputs(g: Any, ex: ExecSpec) -> list[StateDecl]:
    """Output declarations for `GQABlock` when `CacheExec.FUSED`."""
    if bool(ex.interpret):
        return [
            StateDecl(
                POOL_STATE_NAME,
                (cache_exec(ex).num_pages, 2, g.HD // 4, 4, cache_exec(ex).page_size),
                FP8,
                access="w",
                per_layer=True,
                memory_space="hbm",
                tupled=True,
            )
        ]
    return []


def gqa_fused_scratch(ex: ExecSpec, hd: int) -> dict[str, Any]:
    """Scratch buffers for in-kernel KV page RMW (`CacheExec.FUSED`)."""
    return {
        "kv_scat_page": pltpu.VMEM(
            (ex.NS, 2, hd // 4, 4, cache_exec(ex).page_size), FP8
        ),
        "kv_rd_sem": pltpu.SemaphoreType.DMA((ex.NS,)),
        "kv_wr_sem": pltpu.SemaphoreType.DMA((ex.NS,)),
        "kv_rd_pend": pltpu.SMEM((ex.NS,), I32),
        "kv_wr_pend": pltpu.SMEM((ex.NS,), I32),
    }


def gqa_fused_init_buffers(ctx: Any) -> None:
    """Zero per-sequence KV RMW pending flags in SMEM."""
    scr = ctx.scr
    for s in range(ctx.ex.NS):
        scr["kv_rd_pend"][s] = jnp.int32(0)
        scr["kv_wr_pend"][s] = jnp.int32(0)


def _gqa_seq_active(ctx: Any, s: Any) -> Any:
    ex, ST = ctx.ex, ctx.ST
    if cache_exec(ex).act_mode == "all":
        return jnp.bool_(True)
    if cache_exec(ex).act_mode == "ref":
        return (ST["active_seqs"][s] != 0) if "active_seqs" in ST else jnp.bool_(True)
    if "ckpt_indices" not in ST:
        return jnp.bool_(True)
    ck_ref = ST["ckpt_indices"]
    return (ck_ref[s, 0] > 0) if cache_exec(ex).ckpt_2d else (ck_ref[0, s, 0] > 0)


def _gqa_seq_same_page(ctx: Any, s: Any) -> Any:
    ex, pos_ref = ctx.ex, ctx.pos_ref
    T = ex.T
    if T == 1:
        return jnp.bool_(True)
    b0 = s * T
    lp0 = lax.div(jnp.maximum(pos_ref[b0], jnp.int32(0)), cache_exec(ex).page_size)
    ok = jnp.bool_(True)
    for t in range(1, T):
        lp_t = lax.div(
            jnp.maximum(pos_ref[b0 + t], jnp.int32(0)), cache_exec(ex).page_size
        )
        ok = jnp.logical_and(ok, lp_t == lp0)
    return ok


def _gqa_wait_kv_rd(raw_refs: Any, scr: dict[str, Any], s: int) -> None:
    pltpu.make_async_copy(
        raw_refs[0].at[0],
        scr["kv_scat_page"].at[s],
        scr["kv_rd_sem"].at[s],
    ).wait()


def _gqa_wait_kv_wr(
    raw_refs: Any, scr: dict[str, Any], skip: frozenset[str], s: int
) -> None:
    if "kv" in skip:
        return
    wr_pend = scr["kv_wr_pend"]
    pend = wr_pend[s]

    @pl.when(pend != 0)
    def _():
        pltpu.make_async_copy(
            scr["kv_scat_page"].at[s],
            raw_refs[0].at[0],
            scr["kv_wr_sem"].at[s],
        ).wait()

    wr_pend[s] = jnp.int32(0)


def gqa_fused_kvrd_issue(ctx: Any, a_cur: Any) -> None:
    """Prefetch the target KV page for each active sequence into `kv_scat_page`."""
    if "kv" in ctx.skip:
        return
    ex, scr, ST, pos_ref = ctx.ex, ctx.scr, ctx.ST, ctx.pos_ref
    raw_refs = ctx.raw_refs
    NS, T = ex.NS, ex.T
    pt_ref = ST["page_table"]
    kv_scat = scr["kv_scat_page"]
    rd_sem = scr["kv_rd_sem"]
    rd_pend = scr["kv_rd_pend"]
    for s in range(NS):
        _gqa_wait_kv_wr(raw_refs, scr, ctx.skip, s)
        act_s = _gqa_seq_active(ctx, s)
        same_s = _gqa_seq_same_page(ctx, s)
        do_rd = jnp.logical_and(act_s, same_s)
        rd_pend[s] = jnp.where(do_rd, 1, 0).astype(I32)

        @pl.when(do_rd)
        def _(s=s):
            p0 = jnp.maximum(pos_ref[s * T], jnp.int32(0))
            lp0 = jnp.minimum(
                lax.div(p0, cache_exec(ex).page_size), pt_ref.shape[1] - 1
            )
            kp0 = jnp.clip(pt_ref[s, lp0], 0, cache_exec(ex).num_pages - 1)
            bswitch(
                a_cur,
                [
                    lambda r=r, s=s: pltpu.make_async_copy(
                        r.at[kp0], kv_scat.at[s], rd_sem.at[s]
                    ).start()
                    for r in raw_refs
                ],
            )


def gqa_fused_kv_rmw(ctx: Any, ap: Any, a_cur: Any) -> None:
    """Modify prefetched KV pages with `knew_o`/`vnew_o` and start DMA writeback."""
    if "kv" in ctx.skip:
        return
    ex, scr, ST, pos_ref = ctx.ex, ctx.scr, ctx.ST, ctx.pos_ref
    raw_refs = ctx.raw_refs
    NS, T = ex.NS, ex.T
    HD = ap.HD
    h4 = HD // 4
    interpret = bool(ex.interpret)
    pt_ref = ST["page_table"]
    kv_scat = scr["kv_scat_page"]
    rd_sem, wr_sem = scr["kv_rd_sem"], scr["kv_wr_sem"]
    rd_pend, wr_pend = scr["kv_rd_pend"], scr["kv_wr_pend"]
    lane_iota_kv = lax.broadcasted_iota(I32, (h4, cache_exec(ex).page_size), 1)

    def _batch_to_i32_cols(tok_mr_hd):
        rows_bf16 = tok_mr_hd.astype(BF16)
        if cache_exec(ex).page_size > ex.MR:
            rows_bf16 = jnp.pad(
                rows_bf16, ((0, cache_exec(ex).page_size - ex.MR), (0, 0))
            )
        cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
        return pltpu.bitcast(cols_fp8, I32)

    k_cols = _batch_to_i32_cols(scr["knew_o"][...])
    v_cols = _batch_to_i32_cols(scr["vnew_o"][...])
    for s in range(NS):
        act_s = _gqa_seq_active(ctx, s)
        same_s = _gqa_seq_same_page(ctx, s)
        do_fast = rd_pend[s] != 0
        rd_pend[s] = jnp.int32(0)

        @pl.when(do_fast)
        def _(s=s):
            _gqa_wait_kv_rd(raw_refs, scr, s)
            b0 = s * T
            k_view: Any = None
            v_view: Any = None
            if interpret:
                k_cur = lax.bitcast_convert_type(jnp.swapaxes(kv_scat[s, 0], 1, 2), I32)
                v_cur = lax.bitcast_convert_type(jnp.swapaxes(kv_scat[s, 1], 1, 2), I32)
            else:
                k_view = (
                    kv_scat.at[s, 0].bitcast(I32).reshape(h4, cache_exec(ex).page_size)
                )
                v_view = (
                    kv_scat.at[s, 1].bitcast(I32).reshape(h4, cache_exec(ex).page_size)
                )
                k_cur = k_view[...]
                v_cur = v_view[...]
            for t in range(T):
                b_t = b0 + t
                p_t = jnp.maximum(pos_ref[b_t], jnp.int32(0))
                ln_t = lax.rem(p_t, cache_exec(ex).page_size)
                mask_ln = lane_iota_kv == ln_t
                if interpret:
                    mask_b = lane_iota_kv == b_t
                    k_col = jnp.sum(
                        jnp.where(mask_b, k_cols, jnp.int32(0)),
                        axis=1,
                        keepdims=True,
                    )
                    v_col = jnp.sum(
                        jnp.where(mask_b, v_cols, jnp.int32(0)),
                        axis=1,
                        keepdims=True,
                    )
                else:
                    sh_t = lax.rem(
                        ln_t
                        + (cache_exec(ex).page_size - b_t % cache_exec(ex).page_size),
                        cache_exec(ex).page_size,
                    )
                    k_col = pltpu.roll(k_cols, sh_t, 1)
                    v_col = pltpu.roll(v_cols, sh_t, 1)
                k_cur = jnp.where(mask_ln, k_col, k_cur)
                v_cur = jnp.where(mask_ln, v_col, v_cur)
            if interpret:
                kv_scat[s, 0] = lax.bitcast_convert_type(k_cur, FP8).swapaxes(1, 2)
                kv_scat[s, 1] = lax.bitcast_convert_type(v_cur, FP8).swapaxes(1, 2)
            else:
                k_view[...] = k_cur
                v_view[...] = v_cur
            p0 = jnp.maximum(pos_ref[b0], jnp.int32(0))
            lp0 = jnp.minimum(
                lax.div(p0, cache_exec(ex).page_size), pt_ref.shape[1] - 1
            )
            kp0 = jnp.clip(pt_ref[s, lp0], 0, cache_exec(ex).num_pages - 1)
            bswitch(
                a_cur,
                [
                    lambda r=r, s=s: pltpu.make_async_copy(
                        kv_scat.at[s], r.at[kp0], wr_sem.at[s]
                    ).start()
                    for r in raw_refs
                ],
            )

        wr_pend[s] = jnp.where(do_fast, 1, 0).astype(I32)
        if T > 1:

            @pl.when(jnp.logical_and(act_s, jnp.logical_not(same_s)))
            def _(s=s):
                b0 = s * T
                for t in range(T):
                    b_t = b0 + t
                    p_t = jnp.maximum(pos_ref[b_t], jnp.int32(0))
                    lp_t = jnp.minimum(
                        lax.div(p_t, cache_exec(ex).page_size), pt_ref.shape[1] - 1
                    )
                    ln_t = lax.rem(p_t, cache_exec(ex).page_size)
                    kp_t = jnp.clip(pt_ref[s, lp_t], 0, cache_exec(ex).num_pages - 1)
                    bswitch(
                        a_cur,
                        [
                            lambda r=r, s=s, kp_t=kp_t: pltpu.make_async_copy(
                                r.at[kp_t], kv_scat.at[s], rd_sem.at[s]
                            ).start()
                            for r in raw_refs
                        ],
                    )
                    mask_ln = lane_iota_kv == ln_t
                    if interpret:
                        mask_b = lane_iota_kv == b_t
                        k_col = jnp.sum(
                            jnp.where(mask_b, k_cols, jnp.int32(0)),
                            axis=1,
                            keepdims=True,
                        )
                        v_col = jnp.sum(
                            jnp.where(mask_b, v_cols, jnp.int32(0)),
                            axis=1,
                            keepdims=True,
                        )
                    else:
                        sh_t = lax.rem(
                            ln_t
                            + (
                                cache_exec(ex).page_size
                                - b_t % cache_exec(ex).page_size
                            ),
                            cache_exec(ex).page_size,
                        )
                        k_col = pltpu.roll(k_cols, sh_t, 1)
                        v_col = pltpu.roll(v_cols, sh_t, 1)
                    _gqa_wait_kv_rd(raw_refs, scr, s)
                    if interpret:
                        k_old = lax.bitcast_convert_type(
                            jnp.swapaxes(kv_scat[s, 0], 1, 2), I32
                        )
                        v_old = lax.bitcast_convert_type(
                            jnp.swapaxes(kv_scat[s, 1], 1, 2), I32
                        )
                        k_cur = jnp.where(mask_ln, k_col, k_old)
                        v_cur = jnp.where(mask_ln, v_col, v_old)
                        kv_scat[s, 0] = lax.bitcast_convert_type(k_cur, FP8).swapaxes(
                            1, 2
                        )
                        kv_scat[s, 1] = lax.bitcast_convert_type(v_cur, FP8).swapaxes(
                            1, 2
                        )
                    else:
                        k_view = (
                            kv_scat.at[s, 0]
                            .bitcast(I32)
                            .reshape(h4, cache_exec(ex).page_size)
                        )
                        v_view = (
                            kv_scat.at[s, 1]
                            .bitcast(I32)
                            .reshape(h4, cache_exec(ex).page_size)
                        )
                        k_cur = jnp.where(mask_ln, k_col, k_view[...])
                        v_cur = jnp.where(mask_ln, v_col, v_view[...])
                        k_view[...] = k_cur
                        v_view[...] = v_cur
                    bswitch(
                        a_cur,
                        [
                            lambda r=r, s=s, kp_t=kp_t: pltpu.make_async_copy(
                                kv_scat.at[s], r.at[kp_t], wr_sem.at[s]
                            ).start()
                            for r in raw_refs
                        ],
                    )
                    pltpu.make_async_copy(
                        kv_scat.at[s], raw_refs[0].at[0], wr_sem.at[s]
                    ).wait()


def gqa_fused_epilogue(ctx: Any) -> None:
    """Drain any in-flight KV page RMW DMA writes at the end of the layer loop."""
    if "kv" in ctx.skip:
        return
    ex, scr = ctx.ex, ctx.scr
    raw_refs = ctx.raw_refs
    for s in range(ex.NS):
        _gqa_wait_kv_wr(raw_refs, scr, ctx.skip, s)


# ---------------------------------------------------------------------------
# GDN Fused Pool I/O (`GDNBlock` when `CacheExec.FUSED`)
# ---------------------------------------------------------------------------
def gdn_fused_scratch(ex: ExecSpec, gp: Any, plan: Any) -> dict[str, Any]:
    """Scratch buffers for in-kernel GDN checkpoint gather/scatter (`CacheExec.FUSED`).

    The staged checkpoint pages follow the pool backend's `PoolLayout`
    (`gp.pool_layout`, checked by `GDNBlock.plan_params`), which also sets the
    block's slot geometry (`GDN_PAGES` / `REC_PAGES`).
    """
    del plan
    page = gp.pool_layout.kv_page_shape
    return {
        "sbuf": pltpu.VMEM((gp.vh, gp.KD, gp.VD), F32),
        "gdn_in_page": pltpu.VMEM((gp.NGI, gp.GDN_PAGES, *page), FP8),
        "gdn_out_page": pltpu.VMEM((gp.NGO, ex.T * gp.GDN_PAGES, *page), FP8),
        "gdn_in_sem": pltpu.SemaphoreType.DMA((gp.NGI,)),
        "gdn_out_sem": pltpu.SemaphoreType.DMA((gp.NGO, ex.T)),
        "gdn_in_pend": pltpu.SMEM((gp.NGI,), I32),
        "gdn_out_mode": pltpu.SMEM((gp.NGO,), I32),
    }


def gdn_fused_init_buffers(ctx: Any, gp: Any) -> None:
    """Initialize fused GDN gather/scatter SMEM flags and output page staging."""
    scr = ctx.scr
    for s_i in range(gp.NGI):
        scr["gdn_in_pend"][s_i] = jnp.int32(0)
    for s_o in range(gp.NGO):
        scr["gdn_out_mode"][s_o] = jnp.int32(0)
    scr["gdn_out_page"][...] = jnp.zeros(scr["gdn_out_page"].shape, FP8)


def gdn_wait_out(ctx: Any, gp: Any, slot_o: int) -> None:
    """Wait for any in-flight fused GDN checkpoint write on `slot_o`."""
    ex = ctx.ex
    scr = ctx.scr
    raw_refs = ctx.raw_refs
    GDN_PAGES, T = gp.GDN_PAGES, ex.T
    out_page = scr["gdn_out_page"]
    out_sem = scr["gdn_out_sem"]
    out_mode = scr["gdn_out_mode"]
    mode = out_mode[slot_o]

    @pl.when(mode == 1)
    def _():
        pltpu.make_async_copy(
            out_page.at[slot_o],
            raw_refs[0].at[pl.ds(0, GDN_PAGES * T)],
            out_sem.at[slot_o, 0],
        ).wait()

    if T > 1:

        @pl.when(mode == 2)
        def _():
            for t in range(T):
                pltpu.make_async_copy(
                    out_page.at[slot_o, pl.ds(t * GDN_PAGES, GDN_PAGES)],
                    raw_refs[0].at[pl.ds(0, GDN_PAGES)],
                    out_sem.at[slot_o, t],
                ).wait()

    out_mode[slot_o] = jnp.int32(0)


def gdn_fused_emit(ctx: Any, gp: Any, o: Any, slot: Any) -> Any:
    """Fused in-kernel GDN checkpoint gather + recurrence + checkpoint scatter."""
    ex = ctx.ex
    scr, ST = ctx.scr, ctx.ST
    raw_refs = ctx.raw_refs
    skip, sc, stream = ctx.skip, ctx.sc, ctx.stream
    B, T, NS = ex.B, ex.T, ex.NS
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF, NLI, H = gp.QKV, gp.Z, gp.BA, gp.A_OFF, gp.NLI, gp.H
    NGI, NGO, G_PER_ATT = gp.NGI, gp.NGO, gp.G_PER_ATT
    REC_PAGES, GDN_PAGES = gp.REC_PAGES, gp.GDN_PAGES
    SSM_U32_CAP, ssm_u32_rows = gp.SSM_U32_CAP, gp.ssm_u32_rows
    A_conv, qkv_128, conv8_rows = gp.A_conv, gp.qkv_128, gp.conv8_rows
    l2_eps = gp.l2_eps
    interpret = bool(ex.interpret)
    layout = gp.pool_layout
    page = layout.kv_page_shape
    CPR = layout.conv_page_rows

    in_page = scr["gdn_in_page"]
    out_page = scr["gdn_out_page"]
    in_sem = scr["gdn_in_sem"]
    out_sem = scr["gdn_out_sem"]
    in_pend = scr["gdn_in_pend"]
    out_mode = scr["gdn_out_mode"]
    sb = scr["sbuf"]

    def _is_seq_active(s):
        if cache_exec(ex).act_mode == "all":
            return jnp.bool_(True)
        if cache_exec(ex).act_mode == "ref":
            return ST["active_seqs"][s] != 0
        return (
            (ST["ckpt_indices"][0, s, 0] > 0)
            if gp.ckpt_is_3d
            else (ST["ckpt_indices"][s, 0] > 0)
        )

    def _gdn_roff(sub, s):
        return ST["read_offsets"][sub, s] if gp.roff_is_2d else ST["read_offsets"][s]

    def _gdn_pb_in(sub, s):
        roff = jnp.maximum(_gdn_roff(sub, s), jnp.int32(0))
        if gp.ckpt_is_3d:
            pb = layout.page_base(ST["ckpt_indices"][sub, s, roff], sub, 3)
        else:
            pb = layout.page_base(ST["ckpt_indices"][s, roff], sub, 2)
        _kernel_debug_check(
            jnp.logical_and(pb >= 0, pb + GDN_PAGES <= cache_exec(ex).num_pages),
            "GDN gather checkpoint slot maps outside [0, num_pages)",
            interpret=interpret,
        )
        return jnp.clip(pb, 0, cache_exec(ex).num_pages - GDN_PAGES)

    def _gdn_pb_out(sub, s, t, span: int = GDN_PAGES):
        if gp.ckpt_is_3d:
            pb = layout.page_base(ST["ckpt_indices"][sub, s, t + gp.ckpt_t_off], sub, 3)
        else:
            pb = layout.page_base(ST["ckpt_indices"][s, t + gp.ckpt_t_off], sub, 2)
        _kernel_debug_check(
            jnp.logical_and(pb >= 0, pb + span <= cache_exec(ex).num_pages),
            "GDN scatter checkpoint slot maps outside [0, num_pages)",
            interpret=interpret,
        )
        return jnp.clip(pb, 0, cache_exec(ex).num_pages - span)

    def _gdn_contig_ckpt(sub, s):
        if not gp.ckpt_is_3d or T <= 1:
            return jnp.bool_(T == 1)
        s0 = ST["ckpt_indices"][sub, s, gp.ckpt_t_off]
        ok = jnp.bool_(True)
        for t in range(1, T):
            ok = jnp.logical_and(
                ok, ST["ckpt_indices"][sub, s, t + gp.ckpt_t_off] == s0 + t
            )
        return ok

    def _start_gdn_in(o_, s):
        slot_i = s % NGI
        sub = lax.rem(o_, G_PER_ATT)
        roff = _gdn_roff(sub, s)
        act_in = jnp.logical_and(_is_seq_active(s), roff >= 0)
        in_pend[slot_i] = jnp.where(act_in, 1, 0).astype(I32)

        @pl.when(act_in)
        def _():
            g_pool = lax.div(o_, G_PER_ATT)
            pb = _gdn_pb_in(sub, s)
            bswitch(
                g_pool,
                [
                    lambda r=r: pltpu.make_async_copy(
                        r.at[pl.ds(pb, GDN_PAGES)],
                        in_page.at[slot_i],
                        in_sem.at[slot_i],
                    ).start()
                    for r in raw_refs
                ],
            )

    def _wait_gdn_in(slot_i):
        pend = in_pend[slot_i]

        @pl.when(pend != 0)
        def _():
            pltpu.make_async_copy(
                raw_refs[0].at[pl.ds(0, GDN_PAGES)],
                in_page.at[slot_i],
                in_sem.at[slot_i],
            ).wait()

        in_pend[slot_i] = jnp.int32(0)
        return pend != 0

    def _start_gdn_out(o_, s):
        slot_o = s % NGO
        act_out = _is_seq_active(s)
        sub = lax.rem(o_, G_PER_ATT)
        g_pool = lax.div(o_, G_PER_ATT)
        if T == 1:
            out_mode[slot_o] = jnp.where(act_out, 1, 0).astype(I32)

            @pl.when(act_out)
            def _():
                pb0 = _gdn_pb_out(sub, s, 0, span=GDN_PAGES)
                bswitch(
                    g_pool,
                    [
                        lambda r=r: pltpu.make_async_copy(
                            out_page.at[slot_o],
                            r.at[pl.ds(pb0, GDN_PAGES)],
                            out_sem.at[slot_o, 0],
                        ).start()
                        for r in raw_refs
                    ],
                )
        else:
            contig = _gdn_contig_ckpt(sub, s)
            out_mode[slot_o] = jnp.where(act_out, jnp.where(contig, 1, 2), 0).astype(
                I32
            )

            @pl.when(jnp.logical_and(act_out, contig))
            def _():
                pb0 = _gdn_pb_out(sub, s, 0, span=GDN_PAGES * T)
                bswitch(
                    g_pool,
                    [
                        lambda r=r: pltpu.make_async_copy(
                            out_page.at[slot_o],
                            r.at[pl.ds(pb0, GDN_PAGES * T)],
                            out_sem.at[slot_o, 0],
                        ).start()
                        for r in raw_refs
                    ],
                )

            @pl.when(jnp.logical_and(act_out, jnp.logical_not(contig)))
            def _():
                def do_out_nc(r):
                    for t in range(T):
                        pb_t = _gdn_pb_out(sub, s, t, span=GDN_PAGES)
                        pltpu.make_async_copy(
                            out_page.at[slot_o, pl.ds(t * GDN_PAGES, GDN_PAGES)],
                            r.at[pl.ds(pb_t, GDN_PAGES)],
                            out_sem.at[slot_o, t],
                        ).start()

                bswitch(g_pool, [lambda r=r: do_out_nc(r) for r in raw_refs])

    with stream.segment(lazy=False):
        with sc("gdn_in"):
            if "gdn_state" not in skip:
                for s_ in range(min(NS, NGI)):
                    _start_gdn_in(o, s_)
            y = stream.dmatmul("lin_in", scr["xb"], NLI)
            ybuf = scr["ybuf"]
            ybuf[:, :NLI] = rnd(y * ctx.sm(slot, "in_s", NLI))

        ow, _ = stream.sp.lay["conv"]
        taps = [scr["sm"][slot, :, ow + t * QKV : ow + (t + 1) * QKV] for t in range(4)]
        alog = ctx.sm(slot, "alog")[0, :vh]
        dtb = ctx.sm(slot, "dtb")[0, :vh]
        gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
        expand = vh // kh
        row_iota_conv = lax.broadcasted_iota(I32, (A_conv, QKV), 0)
        conv_live = rup(3 * qkv_128, 16)
        conv_fast = (
            gp.conv_stage
            and QKV % 128 == 0
            and VD == 128
            and conv_live <= KD
            and conv_live <= conv8_rows
        )

        def colsk(x):
            return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

        assert 2 * kh <= vh, (kh, vh)
        NCB = 2 if 4 * kh <= vh else 1
        hrow = lax.broadcasted_iota(I32, (vh, VD), 0)
        HR2 = KD // 2

        for s in range(NS):
            slot_i = s % NGI
            slot_o = s % NGO
            with sc("gdn_rec_wait"):
                act_in = (
                    _wait_gdn_in(slot_i)
                    if "gdn_state" not in skip
                    else jnp.bool_(False)
                )
            in_view: Any = None
            s_bf16: Any = None
            if interpret:
                raw_ssm_u32 = pltpu.bitcast(
                    in_page[slot_i, pl.ds(0, REC_PAGES)], U32
                ).reshape(SSM_U32_CAP, 128)
                u32_in = jnp.where(act_in, raw_ssm_u32[:ssm_u32_rows, :], jnp.uint32(0))
                s_bf16 = pltpu.bitcast(u32_in, BF16).reshape(vh, KD, VD)
                raw_conv = (
                    pltpu.bitcast(in_page[slot_i, REC_PAGES], BF16)
                    .reshape(CPR, LANES)[:conv8_rows, :]
                    .reshape(A_conv, QKV)
                    .astype(F32)
                )
            else:
                in_view = (
                    in_page.at[slot_i, pl.ds(0, REC_PAGES)]
                    .bitcast(U32)
                    .reshape(SSM_U32_CAP, 128)
                )
                raw_conv = (
                    in_page.at[slot_i, REC_PAGES]
                    .bitcast(BF16)
                    .reshape(CPR, LANES)[:conv8_rows, :]
                    .reshape(A_conv, QKV)
                    .astype(F32)
                )
            raw_conv = jnp.where(act_in, raw_conv, 0.0)
            c0 = raw_conv[0:1, :]
            c1 = raw_conv[1:2, :]
            c2 = raw_conv[2:3, :]
            refill_in = s + NGI < NS
            if refill_in and interpret and "gdn_state" not in skip:
                _start_gdn_in(o, s + NGI)
            if "gdn_state" not in skip:
                gdn_wait_out(ctx, gp, slot_o)
            if "gdn" not in skip:
                with sc("gdn_rec"):
                    for t in range(T):
                        b = s * T + t
                        x_t = ybuf[b : b + 1, 0:QKV]
                        u_t = rnd(
                            jax.nn.silu(
                                c0 * taps[0]
                                + c1 * taps[1]
                                + c2 * taps[2]
                                + x_t * taps[3]
                            )
                        )
                        c0, c1, c2 = c1, c2, x_t
                        q = u_t[0, : kh * KD].reshape(kh, KD)
                        k = u_t[0, kh * KD : 2 * kh * KD].reshape(kh, KD)
                        v = u_t[0, 2 * kh * KD :].reshape(vh, VD)
                        q = q * lax.rsqrt(
                            jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps
                        )
                        q = q * KD**-0.5
                        k = k * lax.rsqrt(
                            jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps
                        )
                        cb = (t % NCB) * 2 * kh
                        sb[cb : cb + kh] = colsk(k)
                        sb[cb + kh : cb + 2 * kh] = colsk(q)
                        beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
                        a = ybuf[b, A_OFF : A_OFF + vh]
                        g_dec = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                        eg = jnp.exp(g_dec)[:, None, None]
                        beta2 = beta[:, None]
                        ov: Any = None
                        pv: Any = None
                        if not interpret:
                            ov = (
                                out_page.at[slot_o, pl.ds(t * GDN_PAGES, REC_PAGES)]
                                .bitcast(U32)
                                .reshape(SSM_U32_CAP, 128)
                            )
                            if t > 0:
                                pv = (
                                    out_page.at[
                                        slot_o,
                                        pl.ds((t - 1) * GDN_PAGES, REC_PAGES),
                                    ]
                                    .bitcast(U32)
                                    .reshape(SSM_U32_CAP, 128)
                                )
                        out_acc = jnp.zeros((vh, VD), F32)
                        new_heads = []
                        for h in range(vh):
                            j = h // expand
                            if interpret:
                                st_h = s_bf16[h : h + 1]
                            elif t == 0:
                                u_h = jnp.where(
                                    act_in,
                                    in_view[h * HR2 : (h + 1) * HR2, :],
                                    jnp.uint32(0),
                                )
                                st_h = pltpu.bitcast(u_h, BF16).reshape(1, KD, VD)
                            else:
                                st_h = pltpu.bitcast(
                                    pv[h * HR2 : (h + 1) * HR2, :], BF16
                                ).reshape(1, KD, VD)
                            kc_h = sb[cb + j : cb + j + 1]
                            s_h = st_h.astype(F32) * eg[h : h + 1]
                            pred_h = jnp.sum(s_h * kc_h, axis=1)
                            delta_h = (v[h : h + 1] - pred_h) * beta2[h : h + 1]
                            s_h = s_h + kc_h * delta_h[:, None, :]
                            qc_h = sb[cb + kh + j : cb + kh + j + 1]
                            out_acc = jnp.where(
                                hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc
                            )
                            nb_h = s_h.astype(BF16)
                            if interpret:
                                new_heads.append(nb_h)
                            else:
                                ov[h * HR2 : (h + 1) * HR2, :] = pltpu.bitcast(
                                    nb_h.reshape(KD, VD), U32
                                )
                        out = rnd(out_acc)
                        if interpret:
                            s_bf16 = jnp.concatenate(new_heads, axis=0)
                        val = out * lax.rsqrt(
                            jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps
                        )
                        val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                        zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
                        scr["mix"][b, :Z] = rnd(val * gate_act(gp.gate)(zz)).reshape(Z)

                        if interpret:
                            u32_out = pltpu.bitcast(s_bf16.reshape(vh * KD, VD), U32)
                            u32_p = (
                                jnp.pad(
                                    u32_out,
                                    ((0, SSM_U32_CAP - ssm_u32_rows), (0, 0)),
                                )
                                if ssm_u32_rows < SSM_U32_CAP
                                else u32_out
                            )
                            out_page[
                                slot_o,
                                pl.ds(t * GDN_PAGES, REC_PAGES),
                            ] = (
                                lax.bitcast_convert_type(u32_p, FP8)
                                .swapaxes(1, 2)
                                .reshape(REC_PAGES, *page)
                            )
                            if conv_fast:
                                cst = sb.at[0]
                                for r_, cr in enumerate((c0, c1, c2)):
                                    for j in range(qkv_128):
                                        rr = r_ * qkv_128 + j
                                        cst[rr : rr + 1, :] = cr[
                                            :, j * 128 : (j + 1) * 128
                                        ]
                                if conv_live > 3 * qkv_128:
                                    cst[3 * qkv_128 : conv_live, :] = jnp.zeros(
                                        (conv_live - 3 * qkv_128, 128), F32
                                    )
                                conv_taps = cst[0:conv_live, :].astype(BF16)
                                conv_p = (
                                    jnp.pad(
                                        conv_taps,
                                        ((0, CPR - conv_live), (0, 0)),
                                    )
                                    if conv_live < CPR
                                    else conv_taps
                                )
                            else:
                                c0_f = jnp.broadcast_to(c0, (A_conv, QKV))
                                c1_f = jnp.broadcast_to(c1, (A_conv, QKV))
                                c2_f = jnp.broadcast_to(c2, (A_conv, QKV))
                                w_conv = (
                                    jnp.where(
                                        row_iota_conv == 0,
                                        c0_f,
                                        jnp.where(
                                            row_iota_conv == 1,
                                            c1_f,
                                            jnp.where(row_iota_conv == 2, c2_f, 0.0),
                                        ),
                                    )
                                    .astype(BF16)
                                    .reshape(conv8_rows, 128)
                                )
                                conv_p = (
                                    jnp.pad(
                                        w_conv,
                                        ((0, CPR - conv8_rows), (0, 0)),
                                    )
                                    if conv8_rows < CPR
                                    else w_conv
                                )
                            out_page[slot_o, t * GDN_PAGES + REC_PAGES] = (
                                lax.bitcast_convert_type(
                                    conv_p.reshape(
                                        page[0], page[1], page[2] // 2, page[3]
                                    ),
                                    FP8,
                                )
                                .swapaxes(3, 4)
                                .reshape(page)
                            )
                        else:
                            conv_head: Any = None
                            w_conv: Any = None
                            if conv_fast:
                                cst = sb.at[0]
                                for r_, cr in enumerate((c0, c1, c2)):
                                    for j in range(qkv_128):
                                        rr = r_ * qkv_128 + j
                                        cst[rr : rr + 1, :] = cr[
                                            :, j * 128 : (j + 1) * 128
                                        ]
                                if conv_live > 3 * qkv_128:
                                    cst[3 * qkv_128 : conv_live, :] = jnp.zeros(
                                        (conv_live - 3 * qkv_128, 128), F32
                                    )
                                conv_head = cst[0:conv_live, :].astype(BF16)
                            else:
                                c0_f = jnp.broadcast_to(c0, (A_conv, QKV))
                                c1_f = jnp.broadcast_to(c1, (A_conv, QKV))
                                c2_f = jnp.broadcast_to(c2, (A_conv, QKV))
                                w_conv = jnp.where(
                                    row_iota_conv == 0,
                                    c0_f,
                                    jnp.where(
                                        row_iota_conv == 1,
                                        c1_f,
                                        jnp.where(row_iota_conv == 2, c2_f, 0.0),
                                    ),
                                ).astype(BF16)
                            cv = (
                                out_page.at[slot_o, t * GDN_PAGES + REC_PAGES]
                                .bitcast(BF16)
                                .reshape(CPR, LANES)
                            )
                            if conv_fast:
                                # Rows [conv_live, CPR) stay zero: `gdn_out_page`
                                # is zero-filled in `gdn_fused_init_buffers` and
                                # only this conv write touches the conv page, so
                                # the HBM page matches split's zero-padded page.
                                cv[0:conv_live, :] = conv_head
                            else:
                                cv[:conv8_rows, :] = w_conv.reshape(conv8_rows, 128)
            if refill_in and not interpret and "gdn_state" not in skip:
                _start_gdn_in(o, s + NGI)
            if "gdn_state" not in skip:
                _start_gdn_out(o, s)

        with sc("gdn_out"):
            if B < ex.MB:
                scr["mix"][B:, :Z] = jnp.zeros((ex.MB - B, Z), F32)
            scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
            yo = stream.dmatmul("lin_out", scr["xb2"], H)
            return yo * ctx.sm(slot, "out_s")


def gdn_fused_epilogue(ctx: Any, gp: Any) -> None:
    """Drain any in-flight fused GDN checkpoint writes at the end of the layer loop."""
    if gp.NGDN > 0 and "gdn_state" not in ctx.skip:
        for slot_o in range(min(ctx.ex.NS, gp.NGO)):
            gdn_wait_out(ctx, gp, slot_o)


register_fused_io(cast(FusedPoolIO, sys.modules[__name__]))

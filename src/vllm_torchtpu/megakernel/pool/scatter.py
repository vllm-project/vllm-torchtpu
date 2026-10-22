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
"""Pallas pool scatter kernel: new KV + GDN checkpoints into raw pools."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, FP8, I32
from .backend import check_ckpt_bounds, resolve_active_seqs
from .config import cache_exec
from .layout import PoolLayout
from .scatter_ckpt import prepare_ckpt
from .scatter_ctx import ScatterGeom, ScatterKernel
from .scatter_kv import kv_cols, scatter_kv
from .scatter_sched import ckpt_prologue, scatter_ckpt_states

# Pallas kernel name (`SCATTER_KERNEL_NAME` is a substring of it).
_PALLAS_NAME = "qwen3_5_scatter_pool_states"


# ---------------------------------------------------------------------------
# Pallas Pool Scatter Kernel (KV + GDN Checkpoint States)
# ---------------------------------------------------------------------------
def scatter_pool_states(
    layout: PoolLayout,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    pos: jax.Array,
    page_table: jax.Array,
    knew: jax.Array,
    vnew: jax.Array,
    conv: jax.Array | None = None,
    rec: jax.Array | None = None,
    ckpt_indices: jax.Array | None = None,
    active_seqs: jax.Array | str | None = None,
    *,
    interpret: bool | None = None,
    conv_partial: bool | None = None,
) -> tuple[jax.Array, ...]:
    """Scatter `knew`/`vnew` and GDN `conv`/`rec` checkpoints into `raw_pools` in-place.

    Only the per-row KV page indices `kv_pages` of shape `(NS, T)` are passed into
    SMEM (rather than the full `(NS, max_pages_per_seq)` `page_table`), and all
    page/checkpoint indices are clamped to `[0, num_pages)` before DMA.

    Note on `interpret=True` vs TPU `Ref.bitcast` stores:
      On TPU (`interpret=False`), sub-slice writes into `page_buf` and `kv_page`
      use zero-copy `Ref.bitcast(U32/BF16/I32).reshape(...)[...] = ...`, which
      packs the 2nd-to-last sublane axis (`4` for `fp8->u32/i32`, `2` for
      `fp8->bf16`) into 32-bit/16-bit words in VMEM. Because JAX's CPU Pallas
      `state_discharge` does not invert `Ref.bitcast(...).reshape(...)` on LHS
      indexed stores, `interpret=True` uses `lax.bitcast_convert_type(..., FP8)`
      followed by `swapaxes` to move the unpacked byte axis back to the sublane
      axis, producing the exact same byte layout in `page_buf`/`kv_page`.
    """
    layout.validate(ex)
    if interpret is None:
        interpret = ex.interpret
    NATT = layout.num_pools
    assert len(raw_pools) == NATT, (len(raw_pools), NATT)

    NS, B, T, MR = ex.NS, ex.B, ex.T, ex.MR
    G_PER_ATT = layout.slots_per_pool
    page_size = cache_exec(ex).page_size
    HD = layout.hd
    h4 = HD // 4
    page_shape = layout.kv_page_shape
    pps = layout.pages_per_slot
    rec_pages = layout.rec_pages_per_slot
    rec_u32_cap = layout.rec_u32_rows
    conv_page = layout.conv_page
    conv_page_rows = layout.conv_page_rows
    page_rows = layout.page_rows
    assert page_size >= MR, f"MR={MR} exceeds page_size={page_size}"
    if page_table.ndim != 2 or page_table.shape[0] != NS or page_table.shape[1] < 1:
        raise ValueError(
            f"page_table has shape {page_table.shape}, expected ({NS}, >=1)"
        )
    has_gdn = G_PER_ATT > 0
    use_conv_partial = (
        bool(cache_exec(ex).GPR if conv_partial is None else conv_partial)
        and has_gdn
        and not bool(interpret)
    )
    act_mode, active_mask = resolve_active_seqs(
        active_seqs, default_mode="from_ckpt" if has_gdn else "all"
    )
    if has_gdn:
        check_ckpt_bounds(
            ckpt_indices,
            cache_exec(ex).num_pages,
            layout,
            act_mode=act_mode,
            active_mask=active_mask,
        )

    # Pre-gather and clamp only the (NS, T) active KV page indices for SMEM
    pt_cols = page_table.shape[1]
    pos_st = pos[: NS * T].reshape(NS, T)
    lp_idx = jnp.clip(lax.div(pos_st, page_size), 0, pt_cols - 1)
    s_idx_arr = jnp.arange(NS, dtype=I32)[:, None]
    kv_pages = jnp.clip(
        page_table[s_idx_arr, lp_idx], 0, cache_exec(ex).num_pages - 1
    ).astype(I32)

    conv_in, conv_buf_shape, conv_buf_dt, ckpt_geom = prepare_ckpt(
        layout,
        ex,
        conv,
        rec,
        ckpt_indices,
        has_gdn=has_gdn,
        interpret=bool(interpret),
    )
    vh, KD, VD = ckpt_geom["vh"], ckpt_geom["KD"], ckpt_geom["VD"]

    n_pairs = NS // 2
    scratch: dict[str, Any] = {
        "kv_page": pltpu.VMEM((2, *page_shape), FP8),
        "knew_buf": pltpu.VMEM((NATT, MR, HD), FP8),
        "vnew_buf": pltpu.VMEM((NATT, MR, HD), FP8),
        "sem_kv_rd": pltpu.SemaphoreType.DMA((2,)),
        "sem_kv_wr": pltpu.SemaphoreType.DMA((2,)),
        "sem_kv_in": pltpu.SemaphoreType.DMA((2,)),
    }
    if has_gdn:
        scratch.update(
            {
                "page_buf": pltpu.VMEM((2, pps * T, *page_shape), FP8),
                "rec_buf": pltpu.VMEM((2, T * vh, KD, VD), BF16),
                "conv_buf": pltpu.VMEM(conv_buf_shape, conv_buf_dt),
                "sem_in": pltpu.SemaphoreType.DMA((2, 2)),
                "sem_p": pltpu.SemaphoreType.DMA((2, T)),
            }
        )
    if NS > 1:
        scratch["k_cols_buf"] = pltpu.VMEM((NATT, h4, page_size), I32)
        scratch["v_cols_buf"] = pltpu.VMEM((NATT, h4, page_size), I32)

    s_names = list(scratch)
    n_smem = 2 + (1 if has_gdn else 0) + (1 if act_mode == "ref" else 0)
    n_hbm = 4 if has_gdn else 2
    sg = ScatterGeom(
        layout=layout,
        ex=ex,
        interpret=bool(interpret),
        NATT=NATT,
        NS=NS,
        B=B,
        T=T,
        MR=MR,
        G_PER_ATT=G_PER_ATT,
        page_size=page_size,
        HD=HD,
        h4=h4,
        page_shape=tuple(page_shape),
        pps=pps,
        rec_pages=rec_pages,
        rec_u32_cap=rec_u32_cap,
        conv_page=conv_page,
        conv_page_rows=conv_page_rows,
        page_rows=page_rows,
        has_gdn=has_gdn,
        use_conv_partial=use_conv_partial,
        act_mode=act_mode,
        n_smem=n_smem,
        n_hbm=n_hbm,
        s_names=tuple(s_names),
        n_pairs=n_pairs,
        **ckpt_geom,
    )

    def _kernel(*refs: Any):
        kc = ScatterKernel.build(sg, refs)
        cp_knew = pltpu.make_async_copy(kc.knew_ref, kc.knew_buf, kc.sem_kv_in.at[0])
        cp_vnew = pltpu.make_async_copy(kc.vnew_ref, kc.vnew_buf, kc.sem_kv_in.at[1])
        cp_knew.start()
        cp_vnew.start()
        ckpt_prologue(kc)
        cp_knew.wait()
        cp_vnew.wait()
        kv_cols(kc)
        scatter_kv(kc)
        scatter_ckpt_states(kc)

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * NATT
        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * n_smem
        + [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ][:n_hbm]
    )
    out_shape = [jax.ShapeDtypeStruct(r.shape, r.dtype) for r in raw_pools]
    out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * NATT
    call = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases={r: r for r in range(NATT)},
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
            vmem_limit_bytes=ex.resolved_tiles.vmem_mib << 20,
        ),
        interpret=bool(interpret),
        name=_PALLAS_NAME,
    )
    smem_args = (
        (pos, kv_pages)
        + ((ckpt_indices,) if has_gdn else ())
        + ((active_mask,) if act_mode == "ref" else ())
    )
    if knew.dtype != FP8:
        knew = jnp.clip(knew, -448.0, 448.0).astype(FP8)
    if vnew.dtype != FP8:
        vnew = jnp.clip(vnew, -448.0, 448.0).astype(FP8)
    if not bool(interpret):
        knew = pltpu.with_memory_space_constraint(knew, pltpu.HBM)
        vnew = pltpu.with_memory_space_constraint(vnew, pltpu.HBM)
        if has_gdn:
            assert conv_in is not None and rec is not None
            conv_in = pltpu.with_memory_space_constraint(conv_in, pltpu.HBM)
            rec = pltpu.with_memory_space_constraint(rec, pltpu.HBM)
    gdn_args = (conv_in, rec) if has_gdn else ()
    out = call(*raw_pools, *smem_args, knew, vnew, *gdn_args)
    return tuple(out)

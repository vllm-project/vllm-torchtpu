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
"""`make_pool_backend(layout)` plus the shared pool checks.

`check_ckpt_bounds` is the single host-side checkpoint-slot bounds check, used by
the pool kernels, the host packers and the paged-pool callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from ..errors import MegakernelConfigError
from ..exec_spec import ExecSpec
from ..ops import I32
from .config import PoolBackend, cache_exec
from .layout import PoolLayout

GATHER_KERNEL_NAME = "gather_gdn_states"
SCATTER_KERNEL_NAME = "scatter_pool_states"
POOL_KERNEL_NAMES = (GATHER_KERNEL_NAME, SCATTER_KERNEL_NAME)


def resolve_active_seqs(
    active_seqs: jax.Array | str | None,
    *,
    default_mode: str = "from_ckpt",
) -> tuple[str, jax.Array | None]:
    """Normalize `active_seqs` into `(act_mode, active_mask_array)`."""
    if active_seqs is None:
        return default_mode, None
    if isinstance(active_seqs, str):
        if active_seqs in ("from_ckpt", "all"):
            return active_seqs, None
        raise ValueError(
            f"Unsupported active_seqs mode {active_seqs!r}; "
            "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
        )
    return "ref", jnp.asarray(active_seqs, dtype=I32)


def check_ckpt_bounds(
    ckpt_indices: Any,
    num_pages: int,
    layout: PoolLayout,
    *,
    act_mode: str = "from_ckpt",
    active_mask: Any = None,
) -> None:
    """Validate concrete (eager) active checkpoint slots fit in `[0, num_pages)`.

    Accepts 2D `[NS, K]`, 3D `[slots_per_pool, NS, K]` or rank-leading 4D
    `ckpt_indices` (see `PoolLayout.page_base`). 3D indices must use disjoint
    slots across sub-layers of an active sequence. Under `jax.jit` (`Tracer`
    inputs), this eager host check is a no-op and bounds are checked inside the
    Pallas kernels via `pl.debug_check` when Pallas debug checks are enabled
    (`jax_pallas_enable_debug_checks=True`).
    """
    g_per_att = layout.slots_per_pool
    if (
        ckpt_indices is None
        or g_per_att <= 0
        or isinstance(ckpt_indices, jax.core.Tracer)
    ):
        return
    ckpt_np = np.asarray(ckpt_indices)
    if ckpt_np.ndim == 4:
        ckpt_np = ckpt_np[0]
    is_3d = ckpt_np.ndim == 3
    ns = ckpt_np.shape[1] if is_3d else ckpt_np.shape[0]
    mask_np = (
        None
        if (active_mask is None or isinstance(active_mask, jax.core.Tracer))
        else np.asarray(active_mask).reshape(-1)
    )
    pps = layout.pages_per_slot
    for s in range(ns):
        if act_mode == "from_ckpt":
            active = int(ckpt_np[0, s, 0] if is_3d else ckpt_np[s, 0]) > 0
        elif act_mode == "ref" and mask_np is not None:
            active = int(mask_np[s]) != 0
        else:
            active = True
        if not active:
            continue
        seen_slots: dict[int, int] = {}
        for sub in range(g_per_att):
            slots = ckpt_np[sub, s] if is_3d else ckpt_np[s]
            for slot in np.ravel(slots):
                slot_i = int(slot)
                pb = layout.page_base(slot_i, sub, ckpt_np.ndim)
                if pb < 0 or pb + pps > num_pages:
                    raise ValueError(
                        f"GDN checkpoint slot {slot_i} (sub={sub}, s={s}) "
                        f"maps to pages [{pb}, {pb + pps}), which exceeds pool "
                        f"capacity [0, {num_pages})"
                    )
                if is_3d:
                    prev_sub = seen_slots.get(slot_i)
                    if prev_sub is not None and prev_sub != sub:
                        raise ValueError(
                            f"3D ckpt_indices slot {slot_i} is shared between "
                            f"sub={prev_sub} and sub={sub} for active sequence "
                            f"s={s}; 3D ckpt_indices requires disjoint slots "
                            f"across sub-layers (use 2D ckpt_indices if slots "
                            f"are shared across sub-layers)"
                        )
                    seen_slots[slot_i] = sub


def kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)


# `ex -> (chunk rows, state ring depth)` of the recurrent block's chunked path.
RecChunk = Callable[[ExecSpec], tuple[int, int]]


def can_pool_gather(
    layout: PoolLayout, ex: ExecSpec, rec_chunk: RecChunk | None = None
) -> bool:
    """Whether verify bucket `ex` can gather recurrent state in-kernel (`GPR`).

    Depends only on the pool layout and the recurrent chunking callback
    (`rec_chunk(ex) -> (chunk_rows, ring_buffers)`; `None`: no in-kernel gather):
    `GPR` needs the chunked verify path and a conv window that fits the K half
    of one pool page as `(rows, 4, page_size)` fp8 rows.
    """
    if (
        not layout.has_recurrent
        or cache_exec(ex).page_size != layout.page_size
        or ex.T <= 1
    ):
        return False
    if cache_exec(ex).pool_gather in (False, "split"):
        return False
    if cache_exec(ex).pool_gather is None and not ex.WIDE:
        return False
    if rec_chunk is None:
        return False
    cb, bufs = rec_chunk(ex)
    if cb <= 0 or cb % 8 != 0 or cb % ex.T != 0:
        return False
    _, tr = divmod(ex.B, cb)
    nch = ex.B // cb + int(tr > 0)
    nrb = min(bufs, nch)
    if nch > 1 and nrb < 2:
        return False
    return layout.gpr_conv_fits(ex.T, ex.R)


def make_pool_backend(
    layout: PoolLayout,
    *,
    check: Callable[[ExecSpec], None] | None = None,
    rec_chunk: RecChunk | None = None,
    supports_fused_io: bool = True,
    supports_pool_gather: bool | None = None,
) -> PoolBackend:
    """Build the `PoolBackend` (gather / scatter callbacks) for `layout`.

    `check(ex)`, if given, runs before each gather / scatter launch (e.g. the
    model geometry's `check`). `rec_chunk` enables in-kernel gather (see
    `can_pool_gather`); `supports_pool_gather` defaults to whether it is set,
    and `True` without `rec_chunk` raises `MegakernelConfigError`.
    """
    if supports_pool_gather is None:
        supports_pool_gather = rec_chunk is not None
    elif supports_pool_gather and rec_chunk is None:
        raise MegakernelConfigError(
            "make_pool_backend(supports_pool_gather=True) needs rec_chunk (the "
            "recurrent block's chunking, e.g. GDNBlock.chunk_rows): without it "
            "in-kernel gather is never eligible"
        )
    # Local imports: the kernels import the shared checks from this module.
    from .gather import gather_gdn_states
    from .scatter import scatter_pool_states

    has_rec = layout.has_recurrent

    def _gather(
        ex_paged: ExecSpec,
        raw_pools: tuple[jax.Array, ...],
        ckpt_indices: jax.Array | None,
        read_offsets: jax.Array | None,
        active_seqs: Any,
    ) -> dict[str, Any]:
        if not has_rec or cache_exec(ex_paged).GPR:
            return {}
        assert ckpt_indices is not None and read_offsets is not None
        if check is not None:
            check(ex_paged)
        conv, rec = gather_gdn_states(
            layout,
            ex_paged,
            raw_pools,
            ckpt_indices,
            read_offsets,
            active_seqs,
            interpret=ex_paged.interpret,
        )
        return {"conv": conv, "rec": rec}

    def _scatter(
        ex_paged: ExecSpec,
        raw_pools: tuple[jax.Array, ...],
        pos: jax.Array,
        page_table: jax.Array,
        out_map: dict[str, Any],
        ckpt_indices: jax.Array | None,
        active_seqs: Any,
    ) -> tuple[jax.Array, ...]:
        if check is not None:
            check(ex_paged)
        conv_out = out_map.get("conv_ckpt" if ex_paged.VER else "conv")
        rec_out = out_map.get("rec_ckpt" if ex_paged.VER else "rec")
        return scatter_pool_states(
            layout,
            ex_paged,
            raw_pools,
            pos,
            page_table,
            out_map["knew"],
            out_map["vnew"],
            conv_out if has_rec else None,
            rec_out if has_rec else None,
            ckpt_indices if has_rec else None,
            active_seqs,
            interpret=ex_paged.interpret,
            conv_partial=cache_exec(ex_paged).GPR,
        )

    def _can_pool_gather(ex_paged: ExecSpec) -> bool:
        return can_pool_gather(layout, ex_paged, rec_chunk)

    return PoolBackend(
        layout=layout,
        kernel_names=POOL_KERNEL_NAMES,
        gather_states=_gather,
        scatter_states=_scatter,
        can_pool_gather=_can_pool_gather,
        page_table_state_name="page_table",
        supports_fused_io=supports_fused_io,
        supports_pool_gather=supports_pool_gather,
    )


__all__ = [
    "GATHER_KERNEL_NAME",
    "POOL_KERNEL_NAMES",
    "SCATTER_KERNEL_NAME",
    "can_pool_gather",
    "check_ckpt_bounds",
    "kernel_debug_check",
    "make_pool_backend",
    "resolve_active_seqs",
]

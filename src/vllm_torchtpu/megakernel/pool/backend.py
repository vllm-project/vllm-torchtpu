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
"""Shared pool checkpoint-slot bounds check (`check_ckpt_bounds`)."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .layout import PoolLayout


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


__all__ = ["check_ckpt_bounds"]

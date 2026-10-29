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
"""Null-page routing and jit-safe pool metadata uniqueness/disjointness checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from ..ops import I32
from .config import cache_exec, dcp_group

if TYPE_CHECKING:
    from ..exec_spec import ExecSpec
    from .layout import PoolLayout


def safe_page_base(pb: Any, span: int, num_pages: int) -> Any:
    """Route out-of-range checkpoint page bases to shared null page `0`.

    Page `0` is vLLM's shared null block (`[0, reserved_pages)` are reserved in
    `PoolLayout`). Callers issuing contiguous multi-token checkpoint writes with
    `span = pages_per_slot * T > reserved_pages` must ensure `pb` is in range
    (or fall back to per-token writes of length `pages_per_slot <= reserved_pages`)
    so null-page routing never spills past `reserved_pages`.
    """
    valid = jnp.logical_and(pb >= 0, pb + int(span) <= int(num_pages))
    return jnp.where(valid, pb, jnp.int32(0))


def safe_kv_page(kp: Any, num_pages: int) -> Any:
    """Route out-of-range KV page indices to shared null page `0`."""
    valid = jnp.logical_and(kp >= 0, kp < int(num_pages))
    return jnp.where(valid, kp, jnp.int32(0)).astype(I32)


def _validate_pool_metadata_host(
    layout: PoolLayout,
    ex: ExecSpec,
    pt_arr: Any,
    pos_arr: Any,
    ckpt_arr: Any,
    roff_arr: Any,
    act_arr: Any,
    *,
    check_bounds: bool = True,
) -> None:
    pt = np.asarray(pt_arr, dtype=np.int64)
    pos = np.asarray(pos_arr, dtype=np.int64).reshape(-1)
    NS, T = int(ex.NS), int(ex.T)
    num_pages = int(cache_exec(ex).num_pages)
    page_size = int(layout.page_size)
    pps = int(layout.pages_per_slot)

    has_ckpt = (
        layout.has_recurrent and ckpt_arr is not None and np.asarray(ckpt_arr).size > 0
    )
    ck = np.asarray(ckpt_arr, dtype=np.int64) if has_ckpt else None
    c3 = (ck[None, ...] if ck.ndim == 2 else ck) if ck is not None else None

    if act_arr is not None and np.asarray(act_arr).size > 0:
        act_s = np.asarray(act_arr, dtype=bool).reshape(-1)[:NS]
    elif c3 is not None:
        act_s = c3[0, :NS, 0] > 0
    else:
        act_s = pt[:NS, 0] > 0

    active_kv_pages: set[int] = set()
    kv_write_owner: dict[int, int] = {}
    seen_kv_targets: set[tuple[int, int]] = set()
    dcp_g = dcp_group(ex)

    for s in range(NS):
        if not bool(act_s[s]):
            continue
        seq_pos = pos[s * T : (s + 1) * T]
        for t in range(T):
            p_tok = int(seq_pos[t])
            lp_tok = (p_tok // page_size) // dcp_g
            if p_tok < 0 or lp_tok >= pt.shape[1]:
                if check_bounds:
                    raise ValueError(
                        f"Step pool metadata: active seq {s} token {t} pos={p_tok} "
                        f"maps to logical page {lp_tok} outside cols={pt.shape[1]}"
                    )
                continue
            kp = int(pt[s, lp_tok])
            if kp <= 0 or kp >= num_pages:
                if check_bounds:
                    raise ValueError(
                        f"Step pool metadata: active seq {s} token {t} KV write page "
                        f"{kp} outside valid range (0, {num_pages})"
                    )
                continue
            prev_s = kv_write_owner.get(kp)
            if prev_s is not None and prev_s != s:
                raise ValueError(
                    f"Step pool metadata: duplicate KV write page {kp} shared "
                    f"across active sequences {prev_s} and {s}"
                )
            kv_write_owner[kp] = s
            target = (kp, p_tok % page_size)
            if target in seen_kv_targets:
                raise ValueError(
                    f"Step pool metadata: duplicate KV write (page={kp}, "
                    f"lane={target[1]}) within sequence {s}"
                )
            seen_kv_targets.add(target)

        max_lp = min(
            pt.shape[1], max(0, (int(np.max(seq_pos)) // page_size) // dcp_g + 1)
        )
        for lp in range(max_lp):
            p_id = int(pt[s, lp])
            if p_id <= 0 or p_id >= num_pages:
                if check_bounds:
                    raise ValueError(
                        f"Step pool metadata: active seq {s} logical page {lp} has "
                        f"invalid KV page {p_id} outside (0, {num_pages})"
                    )
                continue
            active_kv_pages.add(p_id)

    if c3 is None:
        return

    K = int(c3.shape[2])
    if K < T:
        raise ValueError(
            f"Step pool metadata: ckpt_indices K={K} < T={T} (shape={ck.shape})"
        )
    ckpt_t_off = K - T
    seen_write_pb: dict[int, tuple[int, int, int]] = {}
    ckpt_pages: set[int] = set()
    has_roff = roff_arr is not None and np.asarray(roff_arr).size > 0
    ro = np.asarray(roff_arr, dtype=np.int64) if has_roff else None
    n_subs = int(layout.slots_per_pool) if ck.ndim == 2 else int(c3.shape[0])

    for sub in range(n_subs):
        c_sub = ck if ck.ndim == 2 else c3[sub]
        for s in range(NS):
            if not bool(act_s[s]):
                continue
            cols = [int(c_sub[s, ckpt_t_off + t]) for t in range(T)]
            if T > 1 and (any(c <= 0 for c in cols) or len(set(cols)) < T):
                raise ValueError(
                    f"Step pool metadata: T={T} requires {T} distinct positive "
                    f"checkpoint write slots at sub={sub}, seq={s} (got {cols})"
                )
            for t, slot in enumerate(cols):
                pb = int(layout.page_base(slot, sub, ck.ndim))
                if slot <= 0 or pb < layout.reserved_pages or pb + pps > num_pages:
                    if check_bounds:
                        raise ValueError(
                            f"Step pool metadata: checkpoint write slot {slot} "
                            f"at sub={sub}, seq={s} (pages [{pb}, {pb + pps})) "
                            f"outside [{layout.reserved_pages}, {num_pages})"
                        )
                    continue
                if pb in seen_write_pb:
                    prev = seen_write_pb[pb]
                    raise ValueError(
                        f"Step pool metadata: duplicate checkpoint write slot "
                        f"pages [{pb}, {pb + pps}) at (sub={sub}, seq={s}, t={t}) "
                        f"and {prev}"
                    )
                seen_write_pb[pb] = (sub, s, t)
                ckpt_pages.update(range(pb, pb + pps))

            if ro is not None:
                r_val = int(ro[sub, s] if ro.ndim == 2 else ro[s])
                if r_val >= 0:
                    if r_val >= K:
                        if check_bounds:
                            raise ValueError(
                                f"Step pool metadata: read_offset {r_val} >= K={K} "
                                f"at sub={sub}, seq={s}"
                            )
                        continue
                    r_slot = int(c_sub[s, r_val])
                    r_pb = int(layout.page_base(r_slot, sub, ck.ndim))
                    if (
                        r_slot <= 0
                        or r_pb < layout.reserved_pages
                        or r_pb + pps > num_pages
                    ):
                        if check_bounds:
                            raise ValueError(
                                f"Step pool metadata: checkpoint read slot {r_slot} "
                                f"at sub={sub}, seq={s} (pages [{r_pb}, {r_pb + pps})) "
                                f"outside [{layout.reserved_pages}, {num_pages})"
                            )
                        continue
                    ckpt_pages.update(range(r_pb, r_pb + pps))

    overlap = active_kv_pages & ckpt_pages
    if overlap:
        raise ValueError(
            f"Step pool metadata: checkpoint slot pages overlap active KV pages "
            f"{sorted(overlap)}"
        )


def check_step_pool_metadata(
    layout: PoolLayout,
    ex: ExecSpec,
    page_table: Any,
    pos: Any,
    *,
    ckpt_indices: Any = None,
    read_offsets: Any = None,
    active_seqs: Any = None,
    check_bounds: bool = True,
    force_under_jit: bool = False,
) -> None:
    """Validate unique write pages/slots, KV/ckpt disjointness, and T>1 columns.

    Eager calls validate immediately on host and raise `ValueError`. Traced calls
    under `jax.jit` attach a `jax.debug.callback` when `force_under_jit=True` or
    `pl.enable_debug_checks.value` is active.
    """
    tensors = (page_table, pos, ckpt_indices, read_offsets, active_seqs)
    is_traced = any(isinstance(x, jax.core.Tracer) for x in tensors if x is not None)
    if not is_traced:
        _validate_pool_metadata_host(
            layout,
            ex,
            page_table,
            pos,
            ckpt_indices,
            read_offsets,
            active_seqs,
            check_bounds=check_bounds,
        )
        return
    if not (force_under_jit or pl.enable_debug_checks.value):
        return

    ck_arg = jnp.zeros((0,), dtype=I32) if ckpt_indices is None else ckpt_indices
    ro_arg = jnp.zeros((0,), dtype=I32) if read_offsets is None else read_offsets
    act_arg = jnp.zeros((0,), dtype=jnp.bool_) if active_seqs is None else active_seqs

    def _cb(pt_v: Any, pos_v: Any, ck_v: Any, ro_v: Any, act_v: Any) -> None:
        _validate_pool_metadata_host(
            layout,
            ex,
            pt_v,
            pos_v,
            None if ckpt_indices is None else ck_v,
            None if read_offsets is None else ro_v,
            None if active_seqs is None else act_v,
            check_bounds=check_bounds,
        )

    jax.debug.callback(_cb, page_table, pos, ck_arg, ro_arg, act_arg, ordered=False)

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
"""Host / test packing of logical states into raw pools (and back).

All helpers follow `PoolLayout`; the per-page helpers default to the validated
page geometry (`PoolLayout()`) when no layout is passed.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, FP8
from .backend import check_ckpt_bounds
from .config import cache_exec, resolved_max_pages_per_seq
from .layout import LANES, PoolLayout

_DEFAULT_LAYOUT = PoolLayout()


def _unfold(x: Any, R: int) -> Any:
    """`[R * N0, ...] -> [R, N0, ...]` (inverse of the rank-folded state layout)."""
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


# ---------------------------------------------------------------------------
# Per-page Host / Test Packing & Unpacking Utilities
# ---------------------------------------------------------------------------
def kv_to_raw_page(k_page: Any, v_page: Any) -> jax.Array:
    """Pack `(page_size, HD)` K and V into `(2, HD // 4, 4, page_size) fp8`."""
    ps, hd = k_page.shape[0], k_page.shape[-1]
    assert hd % 4 == 0, hd
    h4 = hd // 4
    k_fp8 = (
        jnp.clip(k_page, -448.0, 448.0).astype(FP8) if k_page.dtype != FP8 else k_page
    )
    v_fp8 = (
        jnp.clip(v_page, -448.0, 448.0).astype(FP8) if v_page.dtype != FP8 else v_page
    )
    k_pk = jnp.transpose(k_fp8.reshape(ps, h4, 4), (1, 2, 0))
    v_pk = jnp.transpose(v_fp8.reshape(ps, h4, 4), (1, 2, 0))
    return jnp.stack([k_pk, v_pk], axis=0)


def raw_page_to_kv(raw_page: Any) -> tuple[jax.Array, jax.Array]:
    """Unpack `(2, HD // 4, 4, page_size) fp8` into `(k_page, v_page)`."""
    ps = raw_page.shape[-1]
    hd = raw_page.shape[1] * raw_page.shape[2]
    k_page = jnp.transpose(raw_page[0], (2, 0, 1)).reshape(ps, hd)
    v_page = jnp.transpose(raw_page[1], (2, 0, 1)).reshape(ps, hd)
    return k_page, v_page


def ssm_to_raw_pages(rec_slice: Any, layout: PoolLayout | None = None) -> jax.Array:
    """Pack key-major `(vh, KD, VD) bf16` into the slot's recurrent pages.

    Returns `(rec_pages_per_slot, *kv_page_shape) fp8` (`(4, 2, 64, 4, 128)`).
    """
    lay = _DEFAULT_LAYOUT if layout is None else layout
    k2, h4, _, ps = lay.kv_page_shape
    cap = lay.rec_u32_rows
    vh, KD, VD = rec_slice.shape
    ssm_u32_rows = (vh * KD) // 2
    s_t = np.asarray(rec_slice, dtype=jnp.dtype(BF16)).reshape(vh * KD, VD)
    u32_val = (
        s_t.reshape(ssm_u32_rows, 2, LANES)
        .transpose(0, 2, 1)
        .copy()
        .view(np.uint32)
        .reshape(ssm_u32_rows, LANES)
    )
    if ssm_u32_rows < cap:
        u32_val = np.pad(u32_val, ((0, cap - ssm_u32_rows), (0, 0)))
    fp8_4p = (
        u32_val.reshape(lay.rec_pages_per_slot, k2, h4, ps, 1)
        .view(jnp.dtype(FP8))
        .transpose(0, 1, 2, 4, 3)
    )
    return jnp.asarray(fp8_4p)


def raw_pages_to_ssm(
    raw_4pages: Any,
    vh: int,
    KD: int = 128,
    VD: int = 128,
    layout: PoolLayout | None = None,
) -> jax.Array:
    """Unpack `(rec_pages_per_slot, *kv_page_shape) fp8` into `(vh, KD, VD) bf16`."""
    lay = _DEFAULT_LAYOUT if layout is None else layout
    ssm_u32_rows = (vh * KD) // 2
    arr_u8 = np.asarray(raw_4pages).view(np.uint8)
    u32_val = (
        arr_u8.transpose(0, 1, 2, 4, 3)
        .copy()
        .view(np.uint32)
        .reshape(lay.rec_u32_rows, LANES)[:ssm_u32_rows, :]
    )
    bf16_pairs = (
        u32_val.reshape(ssm_u32_rows, LANES, 1)
        .view(jnp.dtype(BF16))
        .transpose(0, 2, 1)
        .reshape(vh, KD, VD)
    )
    return jnp.asarray(bf16_pairs)


def conv_to_raw_page(taps_qkv: Any, layout: PoolLayout | None = None) -> jax.Array:
    """Pack `(conv_taps, QKV)` conv taps into one conv page `kv_page_shape` fp8."""
    lay = _DEFAULT_LAYOUT if layout is None else layout
    k2, h4, w4, ps = lay.kv_page_shape
    QKV = taps_qkv.shape[-1]
    conv_rows = (lay.conv_taps * QKV) // LANES
    taps_bf16 = np.asarray(taps_qkv, dtype=jnp.dtype(BF16)).reshape(conv_rows, LANES)
    padded = np.pad(taps_bf16, ((0, lay.conv_page_rows - conv_rows), (0, 0)))
    fp8_1p = (
        padded.reshape(k2, h4, w4 // 2, ps, 1)
        .view(jnp.dtype(FP8))
        .transpose(0, 1, 2, 4, 3)
    )
    fp8_1p = fp8_1p.reshape(lay.kv_page_shape)
    return jnp.asarray(fp8_1p)


def raw_page_to_conv(
    raw_1page: Any, QKV: int, layout: PoolLayout | None = None
) -> jax.Array:
    """Unpack one conv page `kv_page_shape` fp8 into `(conv_taps, QKV) f32`."""
    lay = _DEFAULT_LAYOUT if layout is None else layout
    k2, h4, w4, ps = lay.kv_page_shape
    conv_rows = (lay.conv_taps * QKV) // LANES
    arr_u8 = np.asarray(raw_1page).view(np.uint8)
    bf16_page = (
        arr_u8.reshape(k2, h4, w4 // 2, 2, ps)
        .transpose(0, 1, 2, 4, 3)
        .copy()
        .view(jnp.dtype(BF16))
        .reshape(lay.conv_page_rows, LANES)[:conv_rows, :]
    )
    return jnp.asarray(bf16_page).reshape(lay.conv_taps, QKV).astype(F32)


def _pages_per_seq(ex: ExecSpec) -> int:
    """Pages per sequence (`ex.ctx // page_size`); requires `ctx % page_size == 0`."""
    page_size = cache_exec(ex).page_size
    assert page_size > 0 and ex.ctx > 0 and ex.ctx % page_size == 0, (
        f"ex.ctx ({ex.ctx}) must be a positive multiple of page_size ({page_size})"
    )
    return ex.ctx // page_size


def allocate_test_layout(
    layout: PoolLayout, ex: ExecSpec
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Allocate disjoint `page_table`, `ckpt_indices`, `read_offsets`, and `min_pages`.

    Pages `[0, layout.reserved_pages)` are reserved as dummy/null pages so padded
    or inactive rows that default to page index `0` or checkpoint slot `0` never
    overwrite active sequence `0` (whose KV pages start at `reserved_pages`).
    Checkpoint slots start at `base_slot >= 1` so `ckpt[0, s, 0] > 0` holds for
    all active sequences under `active_seqs="from_ckpt"`. `ckpt_indices` and
    `min_pages` are sized for the 3D checkpoint table `(slots_per_pool, NS, T)`.
    """
    G_PER_ATT = layout.slots_per_pool
    pps = layout.pages_per_slot
    reserved_pages = layout.reserved_pages
    pages_per_seq = _pages_per_seq(ex)
    kv_pages_total = reserved_pages + ex.NS * pages_per_seq
    pt_cols = max(resolved_max_pages_per_seq(ex), pages_per_seq)
    pt = np.zeros((ex.NS, pt_cols), dtype=np.int32)
    for s in range(ex.NS):
        pt[s, :pages_per_seq] = np.arange(
            reserved_pages + s * pages_per_seq,
            reserved_pages + (s + 1) * pages_per_seq,
            dtype=np.int32,
        )

    if G_PER_ATT > 0:
        base_slot = max(1, (kv_pages_total + pps - 1) // pps)
        ckpt = np.zeros((G_PER_ATT, ex.NS, ex.T), dtype=np.int32)
        for sub in range(G_PER_ATT):
            for s in range(ex.NS):
                for k in range(ex.T):
                    ckpt[sub, s, k] = base_slot + (sub * ex.NS + s) * ex.T + k
        roff = np.zeros((G_PER_ATT, ex.NS), dtype=np.int32)
    else:
        ckpt = np.zeros((1, ex.NS, ex.T), dtype=np.int32)
        roff = np.zeros((1, ex.NS), dtype=np.int32)
    min_num_pages = layout.min_num_pages(ex)

    return jnp.asarray(pt), jnp.asarray(ckpt), jnp.asarray(roff), min_num_pages


def states_to_pool(
    layout: PoolLayout,
    ex: ExecSpec,
    states: dict[str, Any],
    page_table: Any,
    ckpt_indices: Any = None,
    read_offsets: Any = None,
) -> tuple[jax.Array, ...]:
    """Pack unfolded or folded states into `layout.num_pools` raw pools."""
    layout.validate(ex)
    R, NS, B = ex.R, ex.NS, ex.B
    G_PER_ATT = layout.slots_per_pool
    page_shape = layout.kv_page_shape
    pt = np.asarray(page_table)
    pages_per_seq = _pages_per_seq(ex)

    kc = states["kc"]
    vc = states["vc"]
    if kc.ndim == 4:
        kc = _unfold(kc, R)
        vc = _unfold(vc, R)

    ckpt: Any = None
    roff: Any = None
    conv_st: Any = None
    rec_st: Any = None
    if G_PER_ATT > 0:
        assert ckpt_indices is not None
        check_ckpt_bounds(ckpt_indices, cache_exec(ex).num_pages, layout)
        ckpt = np.asarray(ckpt_indices)
        ckpt_is_3d = ckpt.ndim == 3
        if read_offsets is None:
            roff = np.zeros(
                (max(1, G_PER_ATT), NS) if ckpt_is_3d else (NS,), dtype=np.int32
            )
        else:
            roff = np.asarray(read_offsets)
        roff_is_2d = roff.ndim == 2
        vh = layout.vh_per_rank(R)
        conv_st = states["conv"]
        rec_st = states["rec"]
        if conv_st.shape[0] != R:
            conv_st = _unfold(conv_st, R)
        if rec_st.shape[0] != R:
            rec_st = _unfold(rec_st, R)
        if rec_st.ndim == 6:
            rec_st = rec_st.reshape(
                R, layout.num_recurrent_layers, NS * vh, layout.KD, layout.VD
            )
    else:
        ckpt_is_3d = roff_is_2d = False
        vh = 0

    taps = layout.conv_taps
    rec_pages = layout.rec_pages_per_slot
    raw_list = []
    for g_idx in range(layout.num_pools):
        pool_g = np.zeros(
            (R, cache_exec(ex).num_pages, *page_shape), dtype=jnp.dtype(FP8)
        )
        kc_g = np.asarray(jnp.clip(kc[:, g_idx], -448.0, 448.0).astype(FP8))
        vc_g = np.asarray(jnp.clip(vc[:, g_idx], -448.0, 448.0).astype(FP8))
        for r in range(R):
            for s in range(NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * cache_exec(ex).page_size
                    k_sl = kc_g[r, s, t0 : t0 + cache_exec(ex).page_size]
                    v_sl = vc_g[r, s, t0 : t0 + cache_exec(ex).page_size]
                    pool_g[r, kp] = np.asarray(kv_to_raw_page(k_sl, v_sl))

        for sub in range(G_PER_ATT):
            o = g_idx * G_PER_ATT + sub
            for r in range(R):
                for s in range(NS):
                    ro_val = int(roff[sub, s]) if roff_is_2d else int(roff[s])
                    if ro_val < 0:
                        continue
                    slot = int(ckpt[sub, s, ro_val] if ckpt_is_3d else ckpt[s, ro_val])
                    page_base = layout.page_base(slot, sub, ckpt.ndim)
                    rec_sl = rec_st[r, o, s * vh : (s + 1) * vh]
                    pool_g[r, page_base : page_base + rec_pages] = np.asarray(
                        ssm_to_raw_pages(rec_sl, layout)
                    )
                    if conv_st.ndim == 5 and conv_st.shape[3] >= taps:
                        tap_rows = conv_st[r, o, s, 0:taps]
                    else:
                        tap_rows = jnp.stack(
                            [conv_st[r, o, j * B + s] for j in range(taps)], axis=0
                        )
                    pool_g[r, page_base + layout.conv_page] = np.asarray(
                        conv_to_raw_page(tap_rows, layout)
                    )

        raw_list.append(
            jnp.asarray(pool_g.reshape(R * cache_exec(ex).num_pages, *page_shape))
        )
    return tuple(raw_list)


def pool_to_states(
    layout: PoolLayout,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    page_table: Any,
    ckpt_indices: Any = None,
) -> dict[str, jax.Array]:
    """Unpack rank-folded raw pools into unfolded logical states `[R, ...]`."""
    layout.validate(ex)
    R, NS, B, T = ex.R, ex.NS, ex.B, ex.T
    G_PER_ATT = layout.slots_per_pool
    page_shape = layout.kv_page_shape
    pt = np.asarray(page_table)
    pages_per_seq = _pages_per_seq(ex)
    natt, hd = layout.num_pools, layout.hd

    kc_out = np.zeros((R, natt, NS, ex.ctx, hd), dtype=jnp.dtype(FP8))
    vc_out = np.zeros((R, natt, NS, ex.ctx, hd), dtype=jnp.dtype(FP8))

    ckpt: Any = None
    rec_out: Any = None
    conv_out: Any = None
    if G_PER_ATT > 0:
        assert ckpt_indices is not None
        check_ckpt_bounds(ckpt_indices, cache_exec(ex).num_pages, layout)
        ckpt = np.asarray(ckpt_indices)
        ckpt_is_3d = ckpt.ndim == 3
        ckpt_k = ckpt.shape[-1]
        ckpt_t_off = 0 if ckpt_k == T else 1
        vh, KD, VD = layout.vh_per_rank(R), layout.KD, layout.VD
        QKV = layout.qkv_per_rank(R)
        ngdn = layout.num_recurrent_layers
        rec_out = np.zeros((R, ngdn, B * vh, KD, VD), dtype=jnp.dtype(BF16))
        conv_out = np.zeros((R, ngdn, NS, T, layout.conv_taps, QKV), dtype=np.float32)
    else:
        ckpt_is_3d = False
        ckpt_t_off = 0
        vh = KD = VD = QKV = 0

    rec_pages = layout.rec_pages_per_slot
    for g_idx in range(natt):
        pool_g = np.asarray(raw_pools[g_idx]).reshape(
            R, cache_exec(ex).num_pages, *page_shape
        )
        for r in range(R):
            for s in range(NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * cache_exec(ex).page_size
                    k_sl, v_sl = raw_page_to_kv(pool_g[r, kp])
                    kc_out[r, g_idx, s, t0 : t0 + cache_exec(ex).page_size] = (
                        np.asarray(k_sl)
                    )
                    vc_out[r, g_idx, s, t0 : t0 + cache_exec(ex).page_size] = (
                        np.asarray(v_sl)
                    )

        for sub in range(G_PER_ATT):
            o = g_idx * G_PER_ATT + sub
            for r in range(R):
                for s in range(NS):
                    for t in range(T):
                        k_idx = t + ckpt_t_off
                        slot = int(
                            ckpt[sub, s, k_idx] if ckpt_is_3d else ckpt[s, k_idx]
                        )
                        page_base = layout.page_base(slot, sub, ckpt.ndim)
                        b = s * T + t
                        rec_out[r, o, b * vh : (b + 1) * vh] = np.asarray(
                            raw_pages_to_ssm(
                                pool_g[r, page_base : page_base + rec_pages],
                                vh,
                                KD,
                                VD,
                                layout,
                            )
                        )
                        conv_out[r, o, s, t] = np.asarray(
                            raw_page_to_conv(
                                pool_g[r, page_base + layout.conv_page], QKV, layout
                            )
                        )

    out: dict[str, jax.Array] = {
        "kc": jnp.asarray(kc_out),
        "vc": jnp.asarray(vc_out),
    }
    if G_PER_ATT > 0:
        out["rec"] = jnp.asarray(rec_out)
        out["conv"] = jnp.asarray(conv_out)
    return out


__all__ = [
    "allocate_test_layout",
    "conv_to_raw_page",
    "kv_to_raw_page",
    "pool_to_states",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "ssm_to_raw_pages",
    "states_to_pool",
]

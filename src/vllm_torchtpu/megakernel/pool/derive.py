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
"""Derive `PoolLayout` from per-layer module cache declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..decls import KVCacheDecl, RecurrentCacheDecl
from ..errors import MegakernelConfigError
from ..ops import BF16, FP8
from .layout import (
    VALIDATED_CONV_PAGES_PER_SLOT,
    VALIDATED_REC_PAGES_PER_SLOT,
    PoolLayout,
)


def _unwrap_cache_decl(item: Any) -> KVCacheDecl | RecurrentCacheDecl | None:
    if hasattr(item, "cache_decl"):
        return item.cache_decl()
    return item


def derive_pool_layout(
    decls_or_kv: Sequence[KVCacheDecl | RecurrentCacheDecl | Any | None]
    | KVCacheDecl
    | Any,
    rec_decl: RecurrentCacheDecl | Any | None = None,
    *,
    num_pools: int | None = None,
    slots_per_pool: int | None = None,
    R: int | None = None,
) -> PoolLayout:
    """Derive a `PoolLayout` from per-layer module cache declarations.

    In sequence mode, recurrent layers attach to the next attention layer,
    yielding `num_pools` (attention layers) and `slots_per_pool` (recurrent
    layers per pool). Passing `num_pools` or `slots_per_pool` overrides the
    sequence-inferred counts while still extracting and uniformity-checking the
    declarations in the sequence.

    In direct mode `(kv_decl, rec_decl, *, num_pools=..., slots_per_pool=...)`,
    `num_pools` is required, and `slots_per_pool` must be passed explicitly
    whenever `rec_decl` is not None.

    Whenever an active `RecurrentCacheDecl` leaves `rec_pages_per_slot=None`,
    `R` (the tensor-parallel rank count) must be passed explicitly to size
    per-rank recurrent state pages. This function computes layout geometry only;
    call `PoolLayout.validate` to enforce TPU pool-kernel support constraints.
    """
    explicit_rec = _unwrap_cache_decl(rec_decl) if rec_decl is not None else None
    if isinstance(decls_or_kv, Sequence) and not isinstance(decls_or_kv, (str, bytes)):
        items = [_unwrap_cache_decl(x) for x in decls_or_kv]
        kv_decls = [x for x in items if isinstance(x, KVCacheDecl)]
        rec_decls = [x for x in items if isinstance(x, RecurrentCacheDecl)]
        for x in items:
            if (
                x is not None
                and not isinstance(x, KVCacheDecl)
                and not isinstance(x, RecurrentCacheDecl)
            ):
                raise MegakernelConfigError(
                    f"derive_pool_layout got unknown cache decl {x!r}"
                )
        if not kv_decls:
            raise MegakernelConfigError(
                "derive_pool_layout requires at least one KVCacheDecl in the "
                "layer sequence"
            )
        kv_first = kv_decls[0]
        if any(k != kv_first for k in kv_decls[1:]):
            raise MegakernelConfigError(
                f"derive_pool_layout requires uniform KVCacheDecls, got {kv_decls!r}"
            )
        rec_first = rec_decls[0] if rec_decls else None
        if rec_first is not None and any(r != rec_first for r in rec_decls[1:]):
            raise MegakernelConfigError(
                "derive_pool_layout requires uniform RecurrentCacheDecls, "
                f"got {rec_decls!r}"
            )
        pool_slot_counts: list[int] = []
        pending_rec = 0
        for x in items:
            if isinstance(x, RecurrentCacheDecl):
                pending_rec += 1
            elif isinstance(x, KVCacheDecl):
                pool_slot_counts.append(pending_rec)
                pending_rec = 0
        if pending_rec != 0 and slots_per_pool is None:
            raise MegakernelConfigError(
                f"derive_pool_layout found {pending_rec} trailing recurrent layer(s) "
                "not followed by an attention layer"
            )
        if len(set(pool_slot_counts)) > 1 and slots_per_pool is None:
            raise MegakernelConfigError(
                "derive_pool_layout requires uniform recurrent slot counts per "
                f"attention pool, got {pool_slot_counts}"
            )
        resolved_num_pools = (
            len(pool_slot_counts) if num_pools is None else int(num_pools)
        )
        resolved_slots_per_pool = (
            pool_slot_counts[0] if slots_per_pool is None else int(slots_per_pool)
        )
        if explicit_rec is not None and slots_per_pool is None and not rec_decls:
            raise MegakernelConfigError(
                "derive_pool_layout requires explicit slots_per_pool when "
                "rec_decl is provided for a KV-only layer sequence"
            )
        kv_d = kv_first
        rec_d = rec_first if rec_decl is None else explicit_rec
    else:
        kv_d = _unwrap_cache_decl(decls_or_kv)
        rec_d = explicit_rec
        if num_pools is None:
            raise MegakernelConfigError(
                "derive_pool_layout requires a per-layer decl sequence or "
                "explicit num_pools"
            )
        if rec_d is not None and slots_per_pool is None:
            raise MegakernelConfigError(
                "derive_pool_layout requires explicit slots_per_pool when "
                "rec_decl is provided without a layer sequence"
            )
        resolved_num_pools = int(num_pools)
        resolved_slots_per_pool = 0 if slots_per_pool is None else int(slots_per_pool)

    if not isinstance(kv_d, KVCacheDecl):
        raise MegakernelConfigError(
            f"derive_pool_layout requires a KVCacheDecl, got {kv_d!r}"
        )
    if rec_d is not None and not isinstance(rec_d, RecurrentCacheDecl):
        raise MegakernelConfigError(
            f"derive_pool_layout requires a RecurrentCacheDecl or None, got {rec_d!r}"
        )
    if resolved_num_pools < 1:
        raise MegakernelConfigError(
            f"derive_pool_layout requires num_pools >= 1, got {resolved_num_pools}"
        )
    if resolved_slots_per_pool < 0:
        raise MegakernelConfigError(
            "derive_pool_layout requires slots_per_pool >= 0, "
            f"got {resolved_slots_per_pool}"
        )
    if resolved_slots_per_pool > 0 and rec_d is None:
        raise MegakernelConfigError(
            f"derive_pool_layout got slots_per_pool={resolved_slots_per_pool} "
            "without a RecurrentCacheDecl"
        )
    page_dtype = FP8 if kv_d.page_dtype is None else kv_d.page_dtype
    if rec_d is None or resolved_slots_per_pool == 0:
        return PoolLayout(
            num_pools=resolved_num_pools,
            slots_per_pool=0,
            hd=kv_d.hd,
            kvh=kv_d.kvh,
            page_size=kv_d.page_size,
            page_dtype=page_dtype,
        )
    rec_dtype = BF16 if rec_d.rec_dtype is None else rec_d.rec_dtype
    conv_dtype = BF16 if rec_d.conv_dtype is None else rec_d.conv_dtype
    conv_pages = (
        rec_d.conv_pages_per_slot
        if rec_d.conv_pages_per_slot is not None
        else VALIDATED_CONV_PAGES_PER_SLOT
    )
    base = PoolLayout(
        num_pools=resolved_num_pools,
        slots_per_pool=resolved_slots_per_pool,
        hd=kv_d.hd,
        kvh=kv_d.kvh,
        page_size=kv_d.page_size,
        page_dtype=page_dtype,
        kh=rec_d.kh,
        vh=rec_d.vh,
        KD=rec_d.KD,
        VD=rec_d.VD,
        rec_dtype=rec_dtype,
        rec_pages_per_slot=1,
        conv_taps=rec_d.conv_taps,
        conv_dtype=conv_dtype,
        conv_pages_per_slot=conv_pages,
    )
    if rec_d.rec_pages_per_slot is not None:
        rec_pages = rec_d.rec_pages_per_slot
    else:
        if R is None or R <= 0:
            raise MegakernelConfigError(
                "derive_pool_layout requires explicit R > 0 to size "
                "rec_pages_per_slot when RecurrentCacheDecl.rec_pages_per_slot is None"
            )
        u32_rows_per_page = max(1, base.rec_u32_rows)
        # Shrunk geometries (e.g. vh=4) may derive a ModelSpec with a default R
        # (e.g. R=8) before ExecSpec.R is known; ceil-dividing vh by R matches
        # PoolLayout.ssm_u32_rows(R) whenever vh % R == 0 without collapsing
        # 0 < vh < R to 0 heads.
        vh_r = (rec_d.vh + R - 1) // R if rec_d.vh > 0 else 0
        ssm_u32_rows = replace(base, vh=vh_r).ssm_u32_rows(1)
        rec_pages = max(
            VALIDATED_REC_PAGES_PER_SLOT,
            (ssm_u32_rows + u32_rows_per_page - 1) // u32_rows_per_page,
        )
    return replace(base, rec_pages_per_slot=rec_pages)


__all__ = ["derive_pool_layout"]

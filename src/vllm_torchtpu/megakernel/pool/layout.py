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
"""`PoolLayout`: byte layout of the unified raw pools (KV pages + state slots).

Each of the `num_pools` pools (one per attention layer) has shape
`[R, num_pages, *kv_page_shape]` fp8 (stored rank-folded as
`[R * num_pages, *kv_page_shape]`), with `kv_page_shape =
(2 * kvh, hd // 4, 4, page_size)`. A single pool holds:

1. **KV cache pages** (one page per `page_size` tokens per sequence):
   `[0]` = K page `(hd // 4, 4, page_size)`, `[1]` = V page. The `(hd // 4, 4)`
   pair packs `hd` fp8 values into 32-bit words along sublanes, so a `bitcast`
   to `uint32` yields `(hd // 4, page_size)` u32 rows without lane transposes.
2. **Recurrent checkpoint slots** (`pages_per_slot` consecutive pages per slot,
   for each of the `slots_per_pool` recurrent layers that share the pool):
   - Pages `[0, rec_pages_per_slot)`: recurrent state `(vh, KD, VD)` bf16,
     packed as u32 pairs (`rec_u32_rows` rows of `LANES` lanes).
   - Page `rec_pages_per_slot` (`conv_page`): the causal-conv window
     `(conv_taps, QKV)` bf16, as `conv_page_rows` rows of `LANES` lanes.

Checkpoint index conventions (`page_base`): a 3D `ckpt_indices`
`[slots_per_pool, NS, K]` maps slot `c` to pages `[c * pps, (c + 1) * pps)`; a
2D `ckpt_indices` `[NS, K]` shares one slot id across the sub-layers, mapping
`(c, sub)` to `c * (pps * slots_per_pool) + sub * pps`. Pages
`[0, reserved_pages)` are null / dummy pages and slot id `0` is the null slot:
padded or inactive rows that default to page or slot `0` never overwrite live
state, and `ckpt_indices[..., 0] > 0` marks an active sequence (`"from_ckpt"`).

Recurrent head counts are stored **globally** (`kh`, `vh`); the tensor-parallel
gather / scatter kernels see the per-rank values `kh // R`, `vh // R`
(`kh_per_rank`, `vh_per_rank`, `qkv_per_rank`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from ..decls import KVCacheDecl, RecurrentCacheDecl
from ..errors import MegakernelConfigError
from ..ops import BF16, FP8, rup
from .config import cache_exec

if TYPE_CHECKING:
    from ..exec_spec import ExecSpec

# TPU lane width: the u32 / bf16 views of a pool page are `(rows, LANES)`.
LANES = 128

# Geometry the pool I/O paths (split gather / scatter, the in-kernel GPR
# path, and fused pool I/O) are validated for. They are written in terms of
# `PoolLayout`, but these combinations are the only ones exercised on TPU;
# `PoolLayout.validate` rejects anything else instead of guessing.
# - `hd == 256`, `page_size == 128`, `kvh == 1`: KV page `(2, 64, 4, 128)` and
#   one `(K, V)` head pair per rank and page.
# - `rec_pages_per_slot == 4`, `conv_pages_per_slot == 1`: the only slot
#   geometry exercised on TPU.
# - `KD == VD == LANES`, `conv_taps == 3`: recurrent rows are one lane-row wide
#   and the recurrent conv state shapes (`rup(3 * B, 8)`, `rup(3 + T, 8)`)
#   hard-code the 3-tap window.
VALIDATED_HD = 256
VALIDATED_PAGE_SIZE = 128
VALIDATED_KVH = 1
VALIDATED_REC_PAGES_PER_SLOT = 4
VALIDATED_CONV_PAGES_PER_SLOT = 1
VALIDATED_CONV_TAPS = 3


@dataclass(frozen=True)
class PoolLayout:
    """Unified raw-pool layout: KV page packing plus recurrent checkpoint slots.

    The defaults describe a single KV-only pool with the validated page geometry.
    Callers build their layout explicitly for their attention and recurrent head
    counts.
    """

    # Pool topology.
    num_pools: int = 1
    slots_per_pool: int = 0  # recurrent layers per pool; 0 = KV-only pool
    # KV page packing `(2 * kvh, hd // 4, 4, page_size)` of `page_dtype`.
    hd: int = VALIDATED_HD
    kvh: int = VALIDATED_KVH
    page_size: int = VALIDATED_PAGE_SIZE
    page_dtype: Any = FP8
    # Recurrent state `(vh, KD, VD)` of `rec_dtype` (global head counts).
    kh: int = 0
    vh: int = 0
    KD: int = LANES
    VD: int = LANES
    rec_dtype: Any = BF16
    rec_pages_per_slot: int = VALIDATED_REC_PAGES_PER_SLOT
    # Causal-conv window `(conv_taps, QKV)` stored as `conv_dtype`.
    conv_taps: int = VALIDATED_CONV_TAPS
    conv_dtype: Any = BF16
    conv_pages_per_slot: int = VALIDATED_CONV_PAGES_PER_SLOT

    # ------------------------------------------------------------------ slots
    @property
    def pages_per_slot(self) -> int:
        """Pages per recurrent checkpoint slot (recurrent pages + conv pages)."""
        return self.rec_pages_per_slot + self.conv_pages_per_slot

    @property
    def conv_page(self) -> int:
        """Index of the conv page inside a slot (after the recurrent pages)."""
        return self.rec_pages_per_slot

    @property
    def has_recurrent(self) -> bool:
        return self.slots_per_pool > 0

    @property
    def num_recurrent_layers(self) -> int:
        """Total recurrent layers across all pools (`num_pools * slots_per_pool`)."""
        return self.num_pools * self.slots_per_pool

    @property
    def recurrent_slots_per_pool(self) -> int:
        """Leading `ckpt_indices` / `read_offsets` dim (1 for KV-only pools)."""
        return max(1, self.slots_per_pool)

    @property
    def reserved_pages(self) -> int:
        """Null / dummy pages `[0, reserved_pages)` (at least page 0)."""
        return max(1, self.pages_per_slot * self.slots_per_pool)

    def page_base(self, slot: Any, sub: Any, ndim: int) -> Any:
        """First page of checkpoint `slot` for sub-layer `sub` (`ckpt_indices.ndim`).

        Works on Python ints and traced scalars alike.
        """
        pps = self.pages_per_slot
        if ndim == 3:
            return slot * pps
        return slot * (pps * self.slots_per_pool) + sub * pps

    def min_num_pages(self, ex: ExecSpec) -> int:
        """Pages needed for a disjoint test layout (reserved + KV + slots)."""
        pps = self.pages_per_slot
        kv_pages_total = self.reserved_pages + ex.NS * (
            ex.ctx // cache_exec(ex).page_size
        )
        if not self.has_recurrent:
            return kv_pages_total
        base_slot = max(1, (kv_pages_total + pps - 1) // pps)
        return (base_slot + self.slots_per_pool * ex.NS * ex.T) * pps

    # ------------------------------------------------------------------ pages
    @property
    def kv_page_shape(self) -> tuple[int, int, int, int]:
        """Shape of one pool page: `(2 * kvh, hd // 4, 4, page_size)`."""
        return (2 * self.kvh, self.hd // 4, 4, self.page_size)

    @property
    def page_bytes(self) -> int:
        k2, h4, w4, ps = self.kv_page_shape
        return k2 * h4 * w4 * ps * jnp.dtype(self.page_dtype).itemsize

    @property
    def page_rows(self) -> int:
        """Rows of one page viewed as `(rows, 4, page_size)` (`2 * kvh * hd // 4`)."""
        return self.kv_page_shape[0] * self.kv_page_shape[1]

    @property
    def rec_u32_rows(self) -> int:
        """u32 rows of `LANES` lanes across the recurrent pages of one slot."""
        return self.rec_pages_per_slot * self.page_bytes // (4 * LANES)

    @property
    def conv_page_rows(self) -> int:
        """bf16 rows of `LANES` lanes in one conv page (`page_bytes // 256`)."""
        return self.page_bytes // (LANES * 2)

    def gpr_conv_rows(self, T: int, R: int) -> int | None:
        """`(rows, 4, page_size)` fp8 page rows of the in-kernel gather (GPR)
        conv window `(rup(conv_taps + T, 8), QKV)` per rank, or None when the
        window does not fill whole rows. GPR copies only the K half of the conv
        page (`kv_page_shape[1]` rows), so the window must fit there."""
        hr = rup(self.conv_taps + T, 8)
        conv_bytes = hr * self.qkv_per_rank(R) * jnp.dtype(self.conv_dtype).itemsize
        row_bytes = self.kv_page_shape[2] * self.page_size
        if conv_bytes % row_bytes != 0:
            return None
        return conv_bytes // row_bytes

    def gpr_conv_fits(self, T: int, R: int) -> bool:
        """Whether the GPR conv window fits the K half of one conv page."""
        rows = self.gpr_conv_rows(T, R)
        return rows is not None and rows <= self.kv_page_shape[1]

    # ---------------------------------------------------------------- per rank
    def kh_per_rank(self, R: int) -> int:
        return self.kh // R

    def vh_per_rank(self, R: int) -> int:
        return self.vh // R

    def qkv_per_rank(self, R: int) -> int:
        """Conv channel count `2 * kh * KD + vh * VD` on one of `R` ranks."""
        return 2 * self.kh_per_rank(R) * self.KD + self.vh_per_rank(R) * self.VD

    def ssm_u32_rows(self, R: int) -> int:
        """u32 rows of one sequence's per-rank recurrent state."""
        return (self.vh_per_rank(R) * self.KD) // 2

    def conv_rows(self, R: int) -> int:
        """bf16 rows of `LANES` lanes of one sequence's per-rank conv window."""
        return (self.conv_taps * self.qkv_per_rank(R)) // LANES

    # -------------------------------------------------------------- validation
    def validate(self, ex: ExecSpec | None = None) -> None:
        """Raise `MegakernelConfigError` for layouts the pool kernels can't run.

        With `ex`, also checks that its `CacheExec.page_size` matches the layout
        and that the per-rank recurrent state and the conv window of every pool
        I/O path `ex` selects (split, in-kernel gather `GPR`, fused `FUSED`)
        fit their slot pages.
        """

        def _need(ok: bool, msg: str) -> None:
            if not ok:
                raise MegakernelConfigError(f"PoolLayout: {msg} ({self})")

        _need(self.num_pools >= 1, "num_pools must be >= 1")
        _need(self.slots_per_pool >= 0, "slots_per_pool must be >= 0")
        _need(
            jnp.dtype(self.page_dtype) == jnp.dtype(FP8),
            "pool pages must be float8_e4m3fn",
        )
        # Only validated geometry (see the VALIDATED_* constants above).
        _need(
            self.hd == VALIDATED_HD and self.page_size == VALIDATED_PAGE_SIZE,
            f"pool kernels are only validated for hd={VALIDATED_HD}, "
            f"page_size={VALIDATED_PAGE_SIZE}",
        )
        _need(
            self.kvh == VALIDATED_KVH,
            f"pool pages hold kvh={VALIDATED_KVH} (K, V) head pair per rank",
        )
        if ex is not None:
            _need(
                cache_exec(ex).page_size == self.page_size,
                f"CacheExec.page_size={cache_exec(ex).page_size} != layout page_size",
            )
        if not self.has_recurrent:
            return
        rec_hint = (
            "; per-rank recurrent state exceeds 4 pages, increase tensor-parallel R"
            if self.rec_pages_per_slot > VALIDATED_REC_PAGES_PER_SLOT
            else ""
        )
        _need(
            self.rec_pages_per_slot == VALIDATED_REC_PAGES_PER_SLOT
            and self.conv_pages_per_slot == VALIDATED_CONV_PAGES_PER_SLOT,
            "pool kernels are only validated for "
            f"{VALIDATED_REC_PAGES_PER_SLOT} recurrent + "
            f"{VALIDATED_CONV_PAGES_PER_SLOT} conv page(s) per slot "
            f"(got rec_pages_per_slot={self.rec_pages_per_slot}, "
            f"conv_pages_per_slot={self.conv_pages_per_slot}{rec_hint})",
        )
        _need(
            self.KD == LANES and self.VD == LANES,
            f"recurrent state needs KD == VD == {LANES}",
        )
        _need(
            self.conv_taps == VALIDATED_CONV_TAPS,
            f"conv window must have {VALIDATED_CONV_TAPS} taps",
        )
        _need(
            jnp.dtype(self.rec_dtype) == jnp.dtype(BF16)
            and jnp.dtype(self.conv_dtype) == jnp.dtype(BF16),
            "recurrent and conv checkpoints are stored as bf16",
        )
        _need(
            self.kh > 0 and self.vh > 0 and self.vh % self.kh == 0,
            "recurrent heads need kh > 0 and vh % kh == 0",
        )
        if ex is None:
            return
        R = ex.R
        _need(
            self.kh % R == 0 and self.vh % R == 0,
            f"recurrent heads (kh={self.kh}, vh={self.vh}) must be divisible by R={R}",
        )
        _need(
            self.qkv_per_rank(R) % LANES == 0,
            f"per-rank conv width {self.qkv_per_rank(R)} must be a multiple of {LANES}",
        )
        _need(
            self.ssm_u32_rows(R) <= self.rec_u32_rows,
            f"per-rank recurrent state ({self.ssm_u32_rows(R)} u32 rows) "
            f"exceeds {self.rec_pages_per_slot} recurrent pages "
            f"({self.rec_u32_rows} rows)",
        )
        conv_rows_pad = ((self.conv_rows(R) + 15) // 16) * 16
        _need(
            conv_rows_pad <= self.conv_page_rows,
            f"per-rank conv window ({conv_rows_pad} rows) exceeds one conv "
            f"page ({self.conv_page_rows} rows)",
        )
        # In-kernel paths stage the whole `(rup(conv_taps + T, 8), QKV)` verify
        # window: GPR in the K half of the conv page (`gpr_conv_fits`),
        # fused pool I/O (at least 16 rows) in the whole conv page.
        qkv_rows = self.qkv_per_rank(R) // LANES
        hr = rup(self.conv_taps + ex.T, 8)
        _need(
            not cache_exec(ex).GPR or self.gpr_conv_fits(ex.T, R),
            f"in-kernel gather (GPR) conv window ({hr * qkv_rows} rows at "
            f"T={ex.T}) exceeds the K half of one conv page "
            f"({self.conv_page_rows // 2} rows)",
        )
        rows = max(16, hr) * qkv_rows
        _need(
            not cache_exec(ex).FUSED or rows <= self.conv_page_rows,
            f"fused pool I/O conv window ({rows} rows at T={ex.T}) exceeds one "
            f"conv page ({self.conv_page_rows} rows)",
        )

    @classmethod
    def from_cache_decls(
        cls,
        decls_or_kv: Sequence[KVCacheDecl | RecurrentCacheDecl | Any | None]
        | KVCacheDecl
        | Any,
        rec_decl: RecurrentCacheDecl | Any | None = None,
        *,
        num_pools: int | None = None,
        slots_per_pool: int | None = None,
        R: int | None = None,
    ) -> PoolLayout:
        """Construct a `PoolLayout` from per-layer `cache_decl()` descriptors."""
        from .derive import derive_pool_layout

        return derive_pool_layout(
            decls_or_kv,
            rec_decl,
            num_pools=num_pools,
            slots_per_pool=slots_per_pool,
            R=R,
        )


__all__ = [
    "LANES",
    "VALIDATED_CONV_PAGES_PER_SLOT",
    "VALIDATED_CONV_TAPS",
    "VALIDATED_HD",
    "VALIDATED_KVH",
    "VALIDATED_PAGE_SIZE",
    "VALIDATED_REC_PAGES_PER_SLOT",
    "PoolLayout",
]

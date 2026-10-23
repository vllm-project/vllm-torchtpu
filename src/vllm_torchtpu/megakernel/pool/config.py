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
"""Paged cache configuration: the `CacheExec` extension and `PoolBackend`.

`CacheExec` holds the paged-pool execution settings (KV paging, the pool I/O
mode, checkpoint-index formats). It rides on `ExecSpec.cache` (`None` means
`CacheExec()`, i.e. plain non-paged states); read it with `cache_exec(ex)` and
derive a variant with `with_cache(ex, **fields)`. `PoolBackend` wraps a
`PoolLayout` together with its gather / scatter callbacks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..exec_spec import NumericsClass

if TYPE_CHECKING:
    from ..exec_spec import ExecSpec
    from .layout import PoolLayout

# State name of the unified raw-pool tuple (`ST[...]` / `O[...]`), defined
# once here and shared across the pool and execution layers.
POOL_STATE_NAME = "raw"

# Valid `CacheExec.pool_gather` and `CacheExec.act_mode` values.
POOL_GATHER_VALUES: tuple[Any, ...] = (False, "split", None, "eligible", True)
ACT_MODE_VALUES: tuple[str, ...] = ("from_ckpt", "all", "ref")

# Numerics class of each pool-I/O mode.
POOL_IO_NUMERICS: dict[str, NumericsClass] = {
    "fused_pool_io": "approx",  # always runs the head-major recurrence
    # GPR, auto (`pool_gather=None`): WIDE verify buckets only, where
    # chunked recurrence is already on (bitwise vs split at T=4 B=24).
    "pool_gather": "exact",
    # GPR on every eligible bucket (`pool_gather="eligible"`): may reassociate
    # non-WIDE verify buckets under tuned tile schedules.
    "pool_gather_eligible": "approx",
}


@dataclass(frozen=True)
class CacheExec:
    """Paged-pool execution settings (`ExecSpec.cache`)."""

    paged_kv: bool = False  # True: stream KV from paged FP8 pools via page_table
    page_size: int = 128  # Tokens per KV page
    num_pages: int = 4096  # Total pages per attention pool per rank
    max_pages_per_seq: int = 0  # Page-table width per sequence (0 -> ceil(ctx / page))
    fused_pool_io: bool = False  # True: fuse KV RMW and recurrent I/O in-kernel
    # In-kernel recurrent gather (GPR): False (default) = split gather
    # (overridable by caller); "split" = split, explicitly; None = auto
    # (WIDE verify buckets where eligible); "eligible" = every bucket the
    # pool backend can gather on; True = required (`MegakernelConfigError`
    # if the bucket cannot).
    pool_gather: bool | None | Literal["eligible", "split"] = False
    # Sequence activity mode ("from_ckpt", "all", "ref")
    act_mode: Literal["from_ckpt", "all", "ref"] = "from_ckpt"
    ckpt_2d: bool = False  # True: 2D (NS, K); False: 3D (slots_per_pool, NS, K)
    ckpt_k: int = 0  # Last-axis width of ckpt_indices (0 -> T)
    roff_1d: bool = False  # True: 1D (NS,); False: 2D (slots_per_pool, NS)
    # Host-side ckpt_indices bounds validation on every eager pooled-step call
    # (device-to-host sync + Python loop). In-kernel clamps always apply.
    check_ckpt_bounds: bool = False
    dcp_kv_sharded: bool = False
    dcp_size: int = 1  # Ranks sharing one KV head group (`R // NKV`)

    def __post_init__(self) -> None:
        if not any(
            self.pool_gather is v
            or (type(self.pool_gather) is str and self.pool_gather == v)
            for v in POOL_GATHER_VALUES
        ):
            raise ValueError(
                f"CacheExec.pool_gather must be one of {POOL_GATHER_VALUES}, "
                f"got {self.pool_gather!r}"
            )
        if self.act_mode not in ACT_MODE_VALUES:
            raise ValueError(
                f"CacheExec.act_mode must be one of {ACT_MODE_VALUES}, "
                f"got {self.act_mode!r}"
            )

    @property
    def FUSED(self) -> bool:
        return bool(self.paged_kv and self.fused_pool_io)

    @property
    def GPR(self) -> bool:
        """In-kernel recurrent gather on (after `pool_gather` resolves to `True`)."""
        return bool(
            self.paged_kv and not self.fused_pool_io and self.pool_gather is True
        )


DEFAULT_CACHE_EXEC = CacheExec()


def cache_exec(ex: ExecSpec) -> CacheExec:
    """The `CacheExec` of `ex` (`CacheExec()` when `ex.cache` is unset)."""
    cx = ex.cache
    if cx is None:
        return DEFAULT_CACHE_EXEC
    if not isinstance(cx, CacheExec):
        raise TypeError(f"ExecSpec.cache must be a CacheExec, got {type(cx).__name__}")
    return cx


def with_cache(ex: ExecSpec, **fields: Any) -> ExecSpec:
    """`ex` with `CacheExec` `fields` replaced."""
    return dataclasses.replace(ex, cache=dataclasses.replace(cache_exec(ex), **fields))


def resolved_max_pages_per_seq(ex: ExecSpec) -> int:
    """Page-table width per sequence (`max_pages_per_seq`, else `ceil(ctx / page)`)."""
    cx = cache_exec(ex)
    return (
        cx.max_pages_per_seq
        if cx.max_pages_per_seq > 0
        else max(1, (ex.ctx + cx.page_size - 1) // cx.page_size)
    )


@dataclass(frozen=True)
class PoolBackend:
    """Unified pool layout and gather/scatter callbacks.

    Pool count, recurrent-slot count and checkpoint slot stride are derived
    from `layout`.
    """

    layout: PoolLayout
    kernel_names: tuple[str, str]
    gather_states: Any = None
    scatter_states: Any = None
    can_pool_gather: Any = None
    page_table_state_name: str = "page_table"
    supports_fused_io: bool = False
    supports_pool_gather: bool = False

    @property
    def num_pools(self) -> int:
        return self.layout.num_pools

    @property
    def has_recurrent_pool(self) -> bool:
        return self.layout.has_recurrent

    @property
    def recurrent_slots_per_pool(self) -> int:
        return self.layout.recurrent_slots_per_pool

    @property
    def ckpt_pages_per_slot(self) -> int:
        return self.layout.pages_per_slot


__all__ = [
    "ACT_MODE_VALUES",
    "DEFAULT_CACHE_EXEC",
    "POOL_GATHER_VALUES",
    "POOL_IO_NUMERICS",
    "POOL_STATE_NAME",
    "CacheExec",
    "PoolBackend",
    "cache_exec",
    "resolved_max_pages_per_seq",
    "with_cache",
]


def dcp_group(ex: ExecSpec) -> int:
    """Ranks that token-shard one KV head group's pages (1 when unsharded)."""
    cx = cache_exec(ex)
    return cx.dcp_size if cx.dcp_kv_sharded else 1

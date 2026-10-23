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
"""Core execution config for the megakernel: tiles, tunables registry, ExecSpec.

Only core execution knobs live here. A block with its own knobs registers a
frozen `Tunables` dataclass (`register_tunables`, with a per-field `NUMERICS`
class); `Tiles.blocks` holds one instance per block name (`Tiles.block` and
`Tiles.with_blocks`). Optional cache-specific execution settings ride on
`ExecSpec.cache`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from .block_options import (
    BlockOptionsMap,
    DebugBlocksMap,
    NumericsClass,
    TunablesMap,
    debug_class,
    register_debug,
    register_tunables,
    registered_debug,
    registered_tunables,
    tunables_class,
)
from .ops import rup

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tiles:
    """Core execution tiling, ring depths, budgets and collectives.

    Every core field declares its numerics class in `TILE_NUMERICS` (below);
    block knobs live in `blocks` (`{block name: Tunables}`, see `block` and
    `with_blocks`).
    """

    bank: int = 2 << 20  # Bytes per dense ring slot
    nbd: int = 14  # Dense ring slots
    nbe: int = 10  # Expert ring slots
    ebank: int = 1 << 20  # Bytes per expert ring slot
    vmem_mib: int = 62  # VMEM limit in MiB
    smem_mib: int = 1  # SMEM limit in MiB
    bounds_checks: bool = False
    lazy: bool = True  # Refill block dense slots after their streams
    inplace: bool = False  # Alias state inputs to outputs where supported
    rsag: bool = False  # Two-round reduce-scatter + all-gather
    hier: int = 0  # Hierarchical all-reduce on v7x-8 (0, 1, 2, 3)
    war: int = -1  # Unfolded [MR, H] all-reduce payload (-1: auto, 0: off, 1: on)
    mb8: int = 0  # Internal activation row count MB (0: M, 1: max(16, MR), 2: MR)
    # Per-bucket schedule threshold (rows). Blocks never compare `B` to a
    # literal; they read this, and tile rules (`profiles/`) may tune it.
    wide_rows: int = 16  # `B > wide_rows`: WIDE schedule (`ExecSpec.WIDE`)
    # Opt-in performance flags below default to off; the default tile rule
    # leaves them off.
    dense_acc_pair: bool = False  # Unrolled / paired dense matmul accumulation
    dmm_unroll: int = 0  # Max tiles to fully unroll in dmatmul (0: never)
    rel_fetch: bool = False  # Trace-time-decoded dense refills (fetch_rel/flush_rel)
    # Per-block tunables (registered `Tunables` instances keyed by block name).
    blocks: Mapping[str, Any] = TunablesMap()

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, TunablesMap):
            object.__setattr__(self, "blocks", TunablesMap(self.blocks))

    def block(self, name: str, cls: type[_T] | None = None) -> _T:
        """Tunables of block `name` (class defaults when unset).

        Raises `KeyError` for an unregistered block and `TypeError` if the
        stored tunables are not a `cls`.
        """
        want: type = tunables_class(name) if cls is None else cls
        val = self.blocks.get(name)
        if val is None:
            return want()
        if not isinstance(val, want):
            raise TypeError(
                f"Tiles.blocks[{name!r}] is a {type(val).__name__}, not a "
                f"{want.__name__}"
            )
        return val

    def with_blocks(self, updates: Mapping[str, Any]) -> Tiles:
        """Copy with block tunables replaced: `{block name: {field: value}}`.

        A value may also be a `Tunables` instance (replaces the block's whole
        set). Unknown blocks or fields raise.
        """
        blocks = dict(self.blocks)
        for name, upd in updates.items():
            if isinstance(upd, Mapping):
                blocks[name] = dataclasses.replace(self.block(name), **upd)
            else:
                blocks[name] = upd
        return dataclasses.replace(self, blocks=TunablesMap(blocks))


# Numerics class of every core `Tiles` field (combined with each registered
# block's `Tunables.NUMERICS` by `tile_numerics`).
# `exact`: moving the field off its baseline value keeps outputs and states
# bit-identical, because it only changes DMA / ring scheduling, memory budgets
# or debug checks. `approx`: changes the arithmetic (matmul K-tiling,
# reduction order, rounding, routing) or is not proven exact.
TILE_NUMERICS: dict[str, NumericsClass] = {
    "bank": "approx",  # dense K-tiling
    "nbd": "exact",  # dense ring depth (scheduling)
    "nbe": "approx",  # expert ring depth also gates block pipelining paths
    "ebank": "approx",  # expert K-tiling
    "vmem_mib": "exact",  # budget only
    "smem_mib": "exact",  # budget only
    "bounds_checks": "exact",  # debug checks only
    "lazy": "exact",  # dense refill placement (scheduling)
    "inplace": "approx",  # state input/output aliasing (not proven exact)
    "rsag": "approx",  # rounds all-reduce partial sums to bf16
    "hier": "approx",  # all-reduce reduction order
    "war": "approx",  # all-reduce payload layout
    "mb8": "approx",  # activation row count
    "wide_rows": "approx",  # WIDE schedule threshold
    "dense_acc_pair": "approx",  # paired dense accumulation
    "dmm_unroll": "approx",  # unrolled dense matmul accumulation
    "rel_fetch": "exact",  # trace-time-decoded dense refills (scheduling)
}


def tile_numerics() -> dict[str, NumericsClass]:
    """Numerics class of every tile knob: core fields plus `<block>.<field>`."""
    out: dict[str, NumericsClass] = dict(TILE_NUMERICS)
    for name, cls in sorted(registered_tunables().items()):
        out.update({f"{name}.{k}": v for k, v in cls.NUMERICS.items()})  # type: ignore[attr-defined]
    return out


def default_tiles(B: int, **kw: Any) -> Tiles:
    """Model-agnostic default tiles (`profiles/default.py:default_tiles`).

    Used when `ExecSpec.tiles` is `None`; ring depths may be refined at build
    time to fit the VMEM budget. Extra keyword arguments are forwarded to
    `profiles.default.default_tiles`.
    """
    from .profiles.default import default_tiles as _rule

    return _rule(B, **kw)


@dataclass(frozen=True)
class DebugOptions:
    """Debug and ablation flags kept separate from model geometry and execution."""

    skip: frozenset[str] = frozenset()  # Named execution phases to skip during ablation
    scopes: int | bool = False  # Emit per-phase jax.named_scope annotations
    # Per-block debug flags (registered `Debug` dataclasses keyed by block name).
    # Kept out of `Tiles`, so setting them leaves `ExecSpec.tiles=None` (and its
    # build-time VMEM refinement) intact.
    blocks: Mapping[str, Any] = DebugBlocksMap()

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, DebugBlocksMap):
            object.__setattr__(self, "blocks", DebugBlocksMap(self.blocks))


@dataclass(frozen=True)
class ExecSpec:
    """Execution-time configuration (independent of model architecture)."""

    B: int = 4  # Active batch rows (= NS * T)
    T: int = 1  # Tokens per sequence (>1 for MTP verify)
    R: int = 8  # Tensor-parallel / expert-parallel ranks
    ctx: int = 2048  # Max KV sequence length
    # True: 4D stacked expert weights; False: per-layer 3D. The one block
    # weight-layout choice kept in core: the kernel builder, weight packing and
    # the expert ring all depend on it, not just the block that owns experts.
    stack_e: bool = False
    interpret: Any = False  # Run Pallas in CPU interpret mode (bool or InterpretParams)
    # None -> default_tiles(B, R=R), which may be refined at build time to fit VMEM
    tiles: Tiles | None = None
    debug: DebugOptions = DebugOptions()
    # Optional extension object for cache-specific execution settings (None when
    # using plain states without page tables; one equal to its own defaults is
    # stored as None). Core execution config never reads it.
    cache: Any = None

    def __post_init__(self) -> None:
        cx = self.cache
        if (
            dataclasses.is_dataclass(cx)
            and not isinstance(cx, type)
            and cx == type(cx)()
        ):
            object.__setattr__(self, "cache", None)

    @property
    def resolved_tiles(self) -> Tiles:
        return self.tiles if self.tiles is not None else default_tiles(self.B, R=self.R)

    def resolve_tiles(self, g: Any = None) -> Tiles:
        """Same as `resolved_tiles` (`g` kept for callers; the defaults ignore it)."""
        del g
        return self.resolved_tiles

    @property
    def NS(self) -> int:
        assert self.B % self.T == 0, (self.B, self.T)
        return self.B // self.T

    @property
    def MR(self) -> int:
        return rup(self.B, 8)

    @property
    def M(self) -> int:
        return max(16, rup(self.B, 16))

    @property
    def MB(self) -> int:
        t = self.resolved_tiles
        return self.MR if t.mb8 == 2 else (max(16, self.MR) if t.mb8 else self.M)

    @property
    def VER(self) -> bool:
        return self.T > 1

    @property
    def WIDE(self) -> bool:
        return self.resolved_tiles.wide_rows < self.B

    @property
    def skip(self) -> frozenset[str]:
        return frozenset(self.debug.skip)

    @property
    def scopes(self) -> int | bool:
        return self.debug.scopes


__all__ = [
    "BlockOptionsMap",
    "DebugBlocksMap",
    "DebugOptions",
    "ExecSpec",
    "NumericsClass",
    "TILE_NUMERICS",
    "Tiles",
    "TunablesMap",
    "debug_class",
    "default_tiles",
    "register_debug",
    "register_tunables",
    "registered_debug",
    "registered_tunables",
    "tile_numerics",
    "tunables_class",
]

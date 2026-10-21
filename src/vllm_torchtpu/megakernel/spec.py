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
"""Declarative model specification for the megakernel.

Weights, states, extra inputs, smalls and scratch requests that blocks declare,
the `Block` interface, and how blocks compose into layers and a `ModelSpec`.
Execution config (`ExecSpec`, `Tiles`, tunables) lives in `exec_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from .exec_spec import ExecSpec


@dataclass(frozen=True)
class Spec:
    """Shape + dtype descriptor for nested/tupled per-layer HBM tensors."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class WeightDecl:
    """Declarative specification for a block weight tensor.

    The leading layer count and layer-to-index map are derived from the
    slot/branch in `ModelSpec.layer_type` where the owning `Block` appears.
    """

    name: str
    K: int  # Reduction dimension (or row dim for non-streamed HBM weight)
    N: int  # Output column dimension per rank
    dtype: Any
    stream_ring: Literal["static", "dynamic", "hbm"] = "static"
    layout: Literal["stacked", "per_layer"] = "stacked"
    lead_dims: tuple[
        int, ...
    ] = ()  # Extra per-layer leading dims (e.g. `(EW,)` for experts)


@dataclass(frozen=True)
class StateDecl:
    """Declarative specification for a state, cache, or per-layer output tensor.

    `shape` is the per-layer, per-rank shape; when `per_layer=True`, the full
    state shape is prefixed with the owning block's active layer count.
    """

    name: str
    shape: tuple[int, ...]
    dtype: Any
    access: Literal["r", "w", "rw"] = "r"
    per_layer: bool = True
    memory_space: Literal["hbm", "smem", "vmem"] = "hbm"
    tupled: bool = False


@dataclass(frozen=True)
class ExtraInput:
    """Positional input after the states, declared by a block.

    Used for optional debug inputs (e.g. externally forced routing). Passed in
    block declaration order after the state inputs and exposed to the block as
    `ctx.extra[name]`. HBM inputs are sharded across ranks,
    SMEM inputs replicated and counted against the SMEM budget.
    """

    name: str
    memory_space: Literal["hbm", "smem"]
    shape: tuple[int, ...] = ()  # Per-rank shape (SMEM budget accounting)
    dtype: Any = None


@dataclass(frozen=True)
class SmallField:
    """Per-layer 1D vector packed into the double-buffered `smalls` table."""

    name: str
    width: int  # Local elements on each rank


@dataclass(frozen=True)
class ScratchReq:
    """VMEM / SMEM / Semaphore buffer request declared by a Block or Core."""

    name: str
    spec: Any  # pltpu.VMEM, pltpu.SMEM, or pltpu.SemaphoreType.DMA
    phase: str = "persistent"  # "persistent" or phase tag for non-overlapping overlay


class Block:
    """Composable megakernel building block (attention, recurrent, FFN, etc.).

    A block with its own knobs sets `Tunables` to a frozen dataclass (fields
    with defaults plus a `NUMERICS` class attribute); defining the subclass
    registers it under `name`, and the block reads its values with
    `self.tunables(ex)`. Debug-only flags go in a separate frozen dataclass
    `Debug`, read with `self.debug_flags(ex)` from `ExecSpec.debug.blocks`.
    """

    name: str = "block"
    reduction: Literal["allreduce", "none", "custom"] = "allreduce"
    input_norm: str | None = None
    Tunables: ClassVar[type | None] = None
    Debug: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.Tunables is not None and (
            "Tunables" in cls.__dict__ or "name" in cls.__dict__
        ):
            # Deferred: declarations do not import the exec config at load time.
            from .exec_spec import register_tunables

            register_tunables(cls.name, cls.Tunables)

    def tunables(self, ex: ExecSpec) -> Any:
        """This block's `Tunables` in `ex`'s tiles (class defaults when unset)."""
        assert self.Tunables is not None, f"block {self.name!r} declares no Tunables"
        return ex.resolved_tiles.block(self.name, self.Tunables)

    def debug_flags(self, ex: ExecSpec) -> Any:
        """This block's `Debug` flags in `ex.debug.blocks` (defaults when unset)."""
        assert self.Debug is not None, f"block {self.name!r} declares no Debug"
        val = ex.debug.blocks.get(self.name)
        if val is None:
            return self.Debug()
        if not isinstance(val, self.Debug):
            raise TypeError(
                f"DebugOptions.blocks[{self.name!r}] must be a "
                f"{self.Debug.__name__}, got {type(val).__name__}"
            )
        return val

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return []

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        """Positional inputs after the states (see `ExtraInput`)."""
        return []

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return []

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        """Minimum column widths required on shared core VMEM buffers."""
        return {}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> Any:
        """Optional block-specific derived parameters computed at build time."""
        return None

    def scratch(
        self, g: Any, ex: ExecSpec, plan: Any
    ) -> dict[str, Any] | list[ScratchReq]:
        return {}

    def prologue(self, ctx: Any) -> None:
        """Executed once before the inter-rank barrier and main layer loop."""

    def init_buffers(self, ctx: Any) -> None:
        """Executed once after the inter-rank barrier and before the main layer loop."""

    def on_layer_start(self, ctx: Any, lyr: Any, slot: Any) -> None:
        """Executed at the start of layer `lyr` alongside `smalls` double-buffering."""

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        """Execute block for layer `l` and return partial `[MB, H]` output."""
        raise NotImplementedError

    def epilogue(self, ctx: Any) -> None:
        """Executed once after the main layer loop."""

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Host-side state update inside `shard_map` after `pallas_call` returns."""
        return {}


@dataclass(frozen=True)
class SwitchSlot:
    """Conditional slot inside a single layer loop body (e.g. two mixer kinds).

    In each repeating cycle of `period = sum(branch_counts)` layers, `branches[i]`
    executes for `branch_counts[i]` consecutive layers.
    """

    name: str
    branches: tuple[Block, ...]
    branch_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.branch_counts:
            object.__setattr__(self, "branch_counts", (1,) * len(self.branches))
        assert len(self.branches) == len(self.branch_counts), (
            self.branches,
            self.branch_counts,
        )
        assert all(c >= 0 for c in self.branch_counts) and sum(self.branch_counts) > 0

    @property
    def period(self) -> int:
        return sum(self.branch_counts)

    @property
    def active_pairs(self) -> tuple[tuple[Block, int], ...]:
        """Return `(branch, count)` pairs with `count > 0`."""
        return tuple((b, c) for b, c in zip(self.branches, self.branch_counts) if c > 0)

    @property
    def active_branches(self) -> tuple[Block, ...]:
        return tuple(b for b, _ in self.active_pairs)

    @property
    def active_counts(self) -> tuple[int, ...]:
        return tuple(c for _, c in self.active_pairs)


@dataclass(frozen=True)
class LayerType:
    """Sequence of slots executed in a single transformer layer."""

    name: str
    slots: tuple[Block | SwitchSlot, ...]


@dataclass(frozen=True)
class ModelSpec:
    """Complete declarative model architecture specification."""

    name: str
    config: Any
    layer_type: LayerType
    # Optional cache backend (layout + gather / scatter callbacks); None for
    # models without a page-based cache.
    cache_backend: Any = None

    def all_blocks(self) -> list[Block]:
        """Return unique active blocks in deterministic declaration order."""
        seen: set[str] = set()
        out: list[Block] = []
        for s in self.layer_type.slots:
            bs = s.active_branches if isinstance(s, SwitchSlot) else (s,)
            for b in bs:
                if b.name not in seen:
                    seen.add(b.name)
                    out.append(b)
        return out

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

Weights, states, extra inputs, smalls and scratch requests that modules declare,
and how modules compose into layers and a `ModelSpec`. The `Module` base class
lives in `module`, parameter/cache/binding declaration primitives live in
`decls`, and both are re-exported here. Execution config (`ExecSpec`, `Tiles`,
tunables) lives in `exec_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .decls import (  # noqa: F401
    REQUIRED,
    FrozenDict,
    HFParamDecl,
    KVCacheDecl,
    ModuleBinding,
    ParamTransform,
    RecurrentCacheDecl,
    ResolvedModuleGeometry,
    VLLMAlias,
    _resolve_field_spec,
    combined_hf_param_shapes,
    validate_rename_and_transforms,
)
from .module import Module


@dataclass(frozen=True)
class Spec:
    """Shape + dtype descriptor for nested/tupled per-layer HBM tensors."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class WeightDecl:
    """Declarative specification for a module weight tensor.

    The leading layer count and layer-to-index map are derived from the
    slot/branch in `ModelSpec.layer_type` where the owning `Module` appears.
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
    state shape is prefixed with the owning module's active layer count.
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
    """Positional input after the states, declared by a module.

    Used for optional debug inputs (e.g. externally forced routing). Passed in
    module declaration order after the state inputs and exposed to the module as
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
    """VMEM / SMEM / Semaphore buffer request declared by a Module or Core."""

    name: str
    spec: Any  # pltpu.VMEM, pltpu.SMEM, or pltpu.SemaphoreType.DMA
    phase: str = "persistent"  # "persistent" or phase tag for non-overlapping overlay


@dataclass(frozen=True)
class SwitchSlot:
    """Conditional slot inside a single layer loop body (e.g. two mixer kinds).

    In each repeating cycle of `period = sum(branch_counts)` layers, `branches[i]`
    executes for `branch_counts[i]` consecutive layers.
    """

    name: str
    branches: tuple[Module, ...]
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
    def active_pairs(self) -> tuple[tuple[Module, int], ...]:
        """Return `(branch, count)` pairs with `count > 0`."""
        return tuple((b, c) for b, c in zip(self.branches, self.branch_counts) if c > 0)

    @property
    def active_branches(self) -> tuple[Module, ...]:
        return tuple(b for b, _ in self.active_pairs)

    @property
    def active_counts(self) -> tuple[int, ...]:
        return tuple(c for _, c in self.active_pairs)


@dataclass(frozen=True)
class LayerType:
    """Sequence of slots executed in a single transformer layer."""

    name: str
    slots: tuple[Module | SwitchSlot, ...]


@dataclass(frozen=True)
class ModelSpec:
    """Complete declarative model architecture specification."""

    name: str
    config: Any
    layer_type: LayerType
    # Optional cache backend (layout + gather / scatter callbacks); None for
    # models without a page-based cache.
    cache_backend: Any = None

    def all_modules(self) -> list[Module]:
        """Return unique active modules in deterministic declaration order."""
        seen: dict[str, Module] = {}
        out: list[Module] = []
        for s in self.layer_type.slots:
            for b in s.active_branches if isinstance(s, SwitchSlot) else (s,):
                prev = seen.get(b.name)
                if prev is None:
                    seen[b.name] = b
                    out.append(b)
                elif prev != b:
                    raise ValueError(
                        f"ModelSpec {self.name!r} has distinct modules sharing "
                        f"name {b.name!r}: {prev!r} vs {b!r}"
                    )
        return out

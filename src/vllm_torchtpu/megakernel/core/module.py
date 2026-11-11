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
"""Composable megakernel `Module` base class and its declaration/execution hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .decls import (
    FrozenDict,
    HFParamDecl,
    KVCacheDecl,
    ModuleBinding,
    RecurrentCacheDecl,
    ResolvedModuleGeometry,
    _resolve_field_spec,
    validate_rename_and_transforms,
)

if TYPE_CHECKING:
    from .exec_spec import ExecSpec
    from .spec import ExtraInput, ScratchReq, SmallField, StateDecl, WeightDecl


class Module:
    """Composable megakernel module (attention, recurrent, FFN, norm, etc.).

    A module with its own knobs sets `Tunables` to a frozen dataclass (fields
    with defaults plus a `NUMERICS` class attribute); defining the subclass
    registers it under `name`, and the module reads its values with
    `self.tunables(ex)`. Debug-only flags go in a separate frozen dataclass
    `Debug`, read with `self.debug_flags(ex)` from `ExecSpec.debug.modules`.
    """

    name: str = "module"
    reduction: Literal["allreduce", "none", "custom"] = "allreduce"
    input_norm: str | None = None
    Tunables: ClassVar[type | None] = None
    Debug: ClassVar[type | None] = None
    GEOMETRY_FIELDS: ClassVar[dict[str, tuple[str, Any]]] = {}
    FIELD_ALIASES: ClassVar[dict[str, str]] = {}
    GEOMETRY_METHODS: ClassVar[dict[str, Any]] = {}
    OPTIONS: ClassVar[dict[str, Any]] = {}
    PARAM_NAMES: ClassVar[frozenset[str]] = frozenset()
    geometry: Any = None
    prefix: str = ""
    options: Mapping[str, Any] = FrozenDict()

    def __init__(
        self,
        prefix: str = "",
        *,
        geometry: Any = None,
        **options: Any,
    ) -> None:
        unknown = set(options) - set(self.OPTIONS)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}() got unexpected option(s) {sorted(unknown)}"
            )
        merged = {**self.OPTIONS, **options}
        self._validate_options(merged)
        self.prefix = prefix
        self.geometry = geometry
        self.options = FrozenDict(merged)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.Tunables is not None and (
            "Tunables" in cls.__dict__ or "name" in cls.__dict__
        ):
            # Deferred: declarations do not import the exec config at load time.
            from .exec_spec import register_tunables

            register_tunables(cls.name, cls.Tunables)
        if cls.Debug is not None and (
            "Debug" in cls.__dict__ or "name" in cls.__dict__
        ):
            from .exec_spec import register_debug

            register_debug(cls.name, cls.Debug)

    def tunables(self, ex: ExecSpec) -> Any:
        """This module's `Tunables` in `ex`'s tiles (class defaults when unset)."""
        assert self.Tunables is not None, f"module {self.name!r} declares no Tunables"
        return ex.resolved_tiles.module(self.name, self.Tunables)

    def debug_flags(self, ex: ExecSpec) -> Any:
        """This module's `Debug` flags in `ex.debug.modules` (defaults when unset)."""
        assert self.Debug is not None, f"module {self.name!r} declares no Debug"
        val = ex.debug.modules.get(self.name)
        if val is None:
            return self.Debug()
        if not isinstance(val, self.Debug):
            raise TypeError(
                f"DebugOptions.modules[{self.name!r}] must be a "
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
        """Optional module-specific derived parameters computed at build time."""
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
        """Execute module for layer `l` and return partial `[MB, H]` output."""
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

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        """Hook for subclasses to reject unsupported option values."""

    @classmethod
    def bind(cls, prefix: str = "", **hf_fields_or_options: Any) -> ModuleBinding:
        """Bind this module class to a checkpoint `prefix` and HF config fields."""
        if not cls.GEOMETRY_FIELDS and not cls.OPTIONS:
            raise NotImplementedError(f"{cls.__name__}.bind is not implemented")
        rename = dict(hf_fields_or_options.pop("rename", {}) or {})
        transforms = dict(hf_fields_or_options.pop("transforms", {}) or {})
        fields = {
            k: default_hf_key for k, (default_hf_key, _) in cls.GEOMETRY_FIELDS.items()
        }
        options = dict(cls.OPTIONS)
        explicit_fields: set[str] = set()
        for k, v in hf_fields_or_options.items():
            canon = cls.FIELD_ALIASES.get(k, k)
            if canon in cls.GEOMETRY_FIELDS:
                fields[canon] = v
                explicit_fields.add(canon)
            elif k in cls.OPTIONS:
                options[k] = v
            else:
                known = sorted(
                    set(cls.GEOMETRY_FIELDS)
                    | set(cls.FIELD_ALIASES)
                    | set(cls.OPTIONS)
                    | {"rename", "transforms"}
                )
                raise ValueError(
                    f"{cls.__name__}.bind got unknown field or option {k!r}; "
                    f"supported: {known}"
                )
        cls._validate_options(options)
        validate_rename_and_transforms(
            cls, rename, transforms, param_name_override=options.get("param_name")
        )
        return ModuleBinding(
            module_cls=cls,
            prefix=prefix,
            fields=fields,
            options=options,
            rename=rename,
            transforms=transforms,
            explicit_fields=frozenset(explicit_fields),
        )

    @classmethod
    def _resolve_binding(
        cls, binding: ModuleBinding, hf_config: Any
    ) -> ResolvedModuleGeometry:
        cls._validate_options(binding.options)
        validate_rename_and_transforms(
            cls,
            binding.rename,
            binding.transforms,
            param_name_override=binding.options.get("param_name"),
        )
        values: dict[str, Any] = {}
        for canon, (default_hf_key, fallback_default) in cls.GEOMETRY_FIELDS.items():
            spec_val = binding.fields.get(canon, default_hf_key)
            aliases = tuple(a for a, t in cls.FIELD_ALIASES.items() if t == canon)
            values[canon] = _resolve_field_spec(
                hf_config,
                canon,
                spec_val,
                fallback_default,
                aliases,
                explicit=canon in binding.explicit_fields,
            )
        return ResolvedModuleGeometry(
            module_cls=cls,
            prefix=binding.prefix,
            values=values,
            options=binding.options,
            rename=binding.rename,
            transforms=binding.transforms,
        )

    @classmethod
    def _from_resolved(cls, resolved: ResolvedModuleGeometry) -> Module:
        ctor: Any = cls
        return ctor(geometry=resolved, prefix=resolved.prefix, **dict(resolved.options))

    def _require_geometry(self, g: Any = None) -> Any:
        geom = g if g is not None else self.geometry
        if geom is None:
            raise ValueError(
                f"{type(self).__name__} requires a resolved geometry "
                "(pass `geometry=...` or `g=...`, or call via `bind(...).resolve(cfg)`)"
            )
        return geom

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        raise NotImplementedError(f"{type(self).__name__}.hf_params is not implemented")

    def cache_decl(self, g: Any = None) -> KVCacheDecl | RecurrentCacheDecl | None:
        return None

    def reference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.reference is not implemented")


__all__ = ["Module"]

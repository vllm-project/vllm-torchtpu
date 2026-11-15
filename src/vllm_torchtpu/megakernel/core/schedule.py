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
"""Declarative per-layer module scheduling (`Schedule`, `Every`, `FromHFLayerTypes`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .decls import (
    REQUIRED,
    FrozenDict,
    KVCacheDecl,
    ModuleBinding,
    RecurrentCacheDecl,
    _lookup_raw_key,
)
from .errors import MegakernelConfigError
from .spec import Module

ModuleOrBinding = ModuleBinding | Module
_VALID_OVERRIDE_SLOTS = frozenset({"mixer", "ffn", "pre_norm", "post_norm"})


def _is_module_or_binding(x: Any) -> bool:
    return isinstance(x, (ModuleBinding, Module))


def _module_cls_of(m: ModuleOrBinding) -> type[Module]:
    return m.module_cls if isinstance(m, ModuleBinding) else type(m)


def _module_name_of(m: ModuleOrBinding) -> str:
    return m.module_cls.name if isinstance(m, ModuleBinding) else m.name


def _desc_m(m: Any) -> str:
    if not _is_module_or_binding(m):
        return repr(m)
    return f"{_module_cls_of(m).__name__}(prefix={m.prefix!r})"


@dataclass(frozen=True)
class FromHFLayerTypes:
    """Resolve per-layer module bindings from an HF config layer-type list field."""

    field: str = "layer_types"
    mapping: Mapping[str, ModuleOrBinding] = dataclasses.field(
        default_factory=FrozenDict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise MegakernelConfigError(
                f"FromHFLayerTypes: field must be non-empty str, got {self.field!r}"
            )
        if not isinstance(self.mapping, Mapping) or not self.mapping:
            raise MegakernelConfigError(
                "FromHFLayerTypes: mapping must be a non-empty {layer_type: Module} map"
            )
        for k, v in self.mapping.items():
            if not isinstance(k, str) or not k:
                raise MegakernelConfigError(
                    f"FromHFLayerTypes: key must be a non-empty string, got {k!r}"
                )
            if not _is_module_or_binding(v):
                raise MegakernelConfigError(
                    f"FromHFLayerTypes: {k!r} (layer 0) has no valid module ({v!r})"
                )
        object.__setattr__(self, "mapping", FrozenDict(self.mapping))

    def resolve(
        self, hf_config: Any, num_layers: int, *, slot_name: str = "mixer"
    ) -> tuple[ModuleOrBinding, ...]:
        """Resolve a length-`num_layers` tuple of module bindings from `hf_config`."""
        raw = _lookup_raw_key(hf_config, self.field)
        if raw is REQUIRED or raw is None:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: missing {self.field!r} (0..{num_layers - 1})"
            )
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: {self.field!r} not a sequence at layer 0"
            )
        if len(raw) != num_layers:
            bad_l = min(len(raw), num_layers)
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: {self.field!r} has length {len(raw)}, "
                f"expected num_layers={num_layers} (at layer {bad_l})"
            )
        supported = sorted(self.mapping)
        for idx, lt in enumerate(raw):
            if lt not in self.mapping:
                raise MegakernelConfigError(
                    f"Schedule.{slot_name}: unknown layer type {lt!r} at layer {idx}; "
                    f"supported: {supported}"
                )
        return tuple(self.mapping[lt] for lt in raw)


@dataclass(frozen=True)
class Every:
    """Periodic selector: `last` on every `n`-th layer, `rest` on the others."""

    n: int
    last: ModuleOrBinding
    rest: ModuleOrBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise MegakernelConfigError(
                f"Every: period n must be an integer >= 1, got {self.n!r} at layer 0"
            )
        if not _is_module_or_binding(self.last):
            raise MegakernelConfigError(
                f"Every(n={self.n}): layer {self.n - 1} ('last') "
                f"has no valid module ({self.last!r})"
            )
        if self.n > 1 and not _is_module_or_binding(self.rest):
            raise MegakernelConfigError(
                f"Every(n={self.n}): layer 0 ('rest') invalid ({self.rest!r})"
            )
        if self.n == 1 and self.rest is not None:
            raise MegakernelConfigError(
                f"Every(n=1): 'rest' must be None when n=1, got {_desc_m(self.rest)}"
            )

    def resolve(
        self, hf_config: Any, num_layers: int, *, slot_name: str = "mixer"
    ) -> tuple[ModuleOrBinding, ...]:
        del hf_config
        if num_layers <= 0 or num_layers % self.n != 0:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: num_layers={num_layers} is not divisible by "
                f"Every(n={self.n}) period (layer {max(0, num_layers - 1)})"
            )
        rest_mod = self.last if self.n == 1 else self.rest
        assert rest_mod is not None
        return tuple(
            self.last if (idx + 1) % self.n == 0 else rest_mod
            for idx in range(num_layers)
        )


def _validate_slot_spec(spec: Any, slot_name: str, *, allow_none: bool = False) -> None:
    if spec is None:
        if allow_none:
            return
        raise MegakernelConfigError(
            f"Schedule.{slot_name}: layer 0 has no module assigned (got None)"
        )
    if isinstance(spec, (ModuleBinding, Module, FromHFLayerTypes, Every)):
        return
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and spec:
        for idx, item in enumerate(spec):
            if not _is_module_or_binding(item):
                raise MegakernelConfigError(
                    f"Schedule.{slot_name}: layer {idx} has no valid module ({item!r})"
                )
        return
    raise MegakernelConfigError(
        f"Schedule.{slot_name}: unsupported schedule specification {spec!r} at layer 0"
    )


def _resolve_slot_spec(
    spec: Any,
    hf_config: Any,
    num_layers: int,
    slot_name: str,
    *,
    allow_none: bool = False,
) -> list[ModuleOrBinding | None]:
    _validate_slot_spec(spec, slot_name, allow_none=allow_none)
    if spec is None or _is_module_or_binding(spec):
        return [spec] * num_layers
    if isinstance(spec, (FromHFLayerTypes, Every)):
        return list(spec.resolve(hf_config, num_layers, slot_name=slot_name))
    if len(spec) != num_layers:
        raise MegakernelConfigError(
            f"Schedule.{slot_name}: sequence has length {len(spec)}, expected "
            f"num_layers={num_layers} (at layer {min(len(spec), num_layers)})"
        )
    return list(spec)


def _derive_att_every_and_branch_indices(
    mixer: Sequence[ModuleOrBinding], *, declared_period: int | None = None
) -> tuple[int, tuple[int, ...]]:
    """Derive periodic `att_every` and per-layer mixer branch ordinals."""
    num_layers = len(mixer)
    if num_layers <= 0:
        raise MegakernelConfigError("Schedule.mixer: num_layers must be >= 1, got 0")
    counts: dict[str, int] = {}
    branch_indices: list[int] = []
    names: list[str] = []
    for lyr, m in enumerate(mixer):
        if not _is_module_or_binding(m):
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has no valid module (got {m!r})"
            )
        m_name = _module_name_of(m)
        if m_name not in ("gqa", "gdn"):
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has unsupported mixer module {m_name!r} "
                f"({_desc_m(m)}); expected 'gdn' or 'gqa'"
            )
        idx = counts.get(m_name, 0)
        counts[m_name] = idx + 1
        branch_indices.append(idx)
        names.append(m_name)

    num_att = counts.get("gqa", 0)
    if num_att == 0:
        raise MegakernelConfigError(
            f"Schedule.mixer: layer 0 has {names[0]!r} ({_desc_m(mixer[0])}); "
            "expected at least one 'gqa' layer in schedule"
        )
    if num_att == num_layers:
        return 1, tuple(branch_indices)
    if declared_period and declared_period >= 2 and num_layers % declared_period == 0:
        period = declared_period
    else:
        divs = [p for p in range(2, num_layers + 1) if num_layers % p == 0]
        period = max(
            divs,
            key=lambda p: (
                sum(
                    n == ("gqa" if (i + 1) % p == 0 else "gdn")
                    for i, n in enumerate(names)
                ),
                -abs(num_layers // p - num_att),
            ),
        )
    for lyr, n in enumerate(names):
        exp = "gqa" if (lyr + 1) % period == 0 else "gdn"
        if n != exp:
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has {n!r} ({_desc_m(mixer[lyr])}), "
                f"expected {exp!r} for period={period}"
            )
    return period, tuple(branch_indices)


@dataclass(frozen=True)
class ResolvedSchedule:
    """Resolved per-layer schedule for a concrete `num_layers` model."""

    num_layers: int
    mixer: tuple[ModuleOrBinding, ...]
    ffn: tuple[ModuleOrBinding, ...]
    pre_norm: tuple[ModuleOrBinding | None, ...]
    post_norm: tuple[ModuleOrBinding | None, ...]
    att_every: int
    mixer_branch_indices: tuple[int, ...]

    def layer_cache_decls(self, g: Any) -> list[KVCacheDecl | RecurrentCacheDecl]:
        """Return the per-layer cache declarations along `self.mixer`."""
        out: list[KVCacheDecl | RecurrentCacheDecl] = []
        for lyr, m in enumerate(self.mixer):
            mod = (
                m.module_cls(geometry=g, prefix=m.prefix, **dict(m.options))
                if isinstance(m, ModuleBinding)
                else m
            )
            decl = mod.cache_decl(g)
            if decl is None:
                raise MegakernelConfigError(
                    f"Schedule.mixer: {mod.name!r} at layer {lyr} has no cache_decl()"
                )
            out.append(decl)
        return out


@dataclass(frozen=True)
class Schedule:
    """Declarative layer schedule (`mixer`, `ffn`, norms, and per-layer `overrides`)."""

    mixer: Any
    ffn: Any
    pre_norm: ModuleOrBinding | None = None
    post_norm: ModuleOrBinding | None = None
    overrides: Mapping[int, Mapping[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_slot_spec(self.mixer, "mixer", allow_none=False)
        _validate_slot_spec(self.ffn, "ffn", allow_none=False)
        _validate_slot_spec(self.pre_norm, "pre_norm", allow_none=True)
        _validate_slot_spec(self.post_norm, "post_norm", allow_none=True)
        if not isinstance(self.overrides, Mapping):
            raise MegakernelConfigError(
                f"Schedule.overrides must be a mapping, got {self.overrides!r}"
            )
        frozen_ov: dict[int, FrozenDict] = {}
        for lyr, ov in self.overrides.items():
            if not isinstance(lyr, int) or isinstance(lyr, bool) or lyr < 0:
                raise MegakernelConfigError(
                    f"Schedule.overrides: layer index {lyr!r} is out of range (>= 0)"
                )
            if not isinstance(ov, Mapping) or not ov:
                raise MegakernelConfigError(
                    f"Schedule.overrides[{lyr}]: value {ov!r} must be non-empty"
                )
            for slot_k, slot_v in ov.items():
                if slot_k not in _VALID_OVERRIDE_SLOTS:
                    raise MegakernelConfigError(
                        f"Schedule.overrides[{lyr}]: unknown {slot_k!r} ({slot_v!r})"
                    )
                allow_none = slot_k in ("pre_norm", "post_norm")
                if (not allow_none and not _is_module_or_binding(slot_v)) or (
                    allow_none
                    and slot_v is not None
                    and not _is_module_or_binding(slot_v)
                ):
                    raise MegakernelConfigError(
                        f"Schedule.overrides[{lyr}][{slot_k!r}]: "
                        f"layer {lyr} has no valid module (got {slot_v!r})"
                    )
            frozen_ov[lyr] = FrozenDict(ov)
        object.__setattr__(self, "overrides", FrozenDict(frozen_ov))

    def validate_bounds(self, num_layers: int) -> None:
        """Validate `num_layers` and `overrides` layer indices in `0..num_layers-1`."""
        if (
            not isinstance(num_layers, int)
            or isinstance(num_layers, bool)
            or num_layers < 1
        ):
            raise MegakernelConfigError(
                f"Schedule: num_layers must be an integer >= 1, got {num_layers!r}"
            )
        for lyr, ov in self.overrides.items():
            if lyr < 0 or lyr >= num_layers:
                raise MegakernelConfigError(
                    f"Schedule.overrides: layer index {lyr} is out of range "
                    f"for num_layers={num_layers} (0..{num_layers - 1})"
                )

    def resolve(self, hf_config: Any, num_layers: int) -> ResolvedSchedule:
        """Resolve per-layer bindings for `num_layers` layers against `hf_config`."""
        self.validate_bounds(num_layers)
        mix_l = _resolve_slot_spec(self.mixer, hf_config, num_layers, "mixer")
        ffn_l = _resolve_slot_spec(self.ffn, hf_config, num_layers, "ffn")
        pre_l = _resolve_slot_spec(
            self.pre_norm, hf_config, num_layers, "pre_norm", allow_none=True
        )
        post_l = _resolve_slot_spec(
            self.post_norm, hf_config, num_layers, "post_norm", allow_none=True
        )
        decl_p: int | None = self.mixer.n if isinstance(self.mixer, Every) else None
        for lyr, ov in self.overrides.items():
            if "mixer" in ov:
                mix_l[lyr] = ov["mixer"]
            if "ffn" in ov:
                ffn_l[lyr] = ov["ffn"]
            if "pre_norm" in ov:
                pre_l[lyr] = ov["pre_norm"]
            if "post_norm" in ov:
                post_l[lyr] = ov["post_norm"]
        mix_t = tuple(m for m in mix_l if m is not None)
        ffn_t = tuple(m for m in ffn_l if m is not None)
        att_every, b_idx = _derive_att_every_and_branch_indices(
            mix_t, declared_period=decl_p
        )
        return ResolvedSchedule(
            num_layers, mix_t, ffn_t, tuple(pre_l), tuple(post_l), att_every, b_idx
        )


__all__ = ["Every", "FromHFLayerTypes", "ResolvedSchedule", "Schedule"]

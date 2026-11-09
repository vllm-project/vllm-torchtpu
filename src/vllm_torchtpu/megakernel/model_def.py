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
"""Declarative `ModelDef` base class, derived artifacts, and model registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import jax

from . import derive, pool, validate
from .decls import (
    FrozenDict,
    KVCacheDecl,
    ModuleBinding,
    RecurrentCacheDecl,
    _sub_cfg,
)
from .errors import EligibilityResult, MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import FromHF, LMHead, ModelGeometry
from .modules.gdn import GDNModule
from .pool import PoolBackend, PoolLayout
from .profile import Profile, get_profile, model_default_profile, register_profile
from .reference import composed_reference_step, spec_reference_step
from .schedule import (
    Every,
    FromHFLayerTypes,
    ModuleOrBinding,
    ResolvedSchedule,
    Schedule,
)
from .spec import ModelSpec, Module


class _BoundFromHF:
    """`FromHF` bound to a `ModelDef` subclass; callable to derive geometry."""

    __slots__ = ("_from_hf", "_model_cls")

    def __init__(self, from_hf: FromHF, model_cls: type[ModelDef]) -> None:
        self._from_hf, self._model_cls = from_hf, model_cls

    @property
    def fields(self) -> Mapping[str, Any]:
        return self._from_hf.fields

    def resolve(self, hf_config: Any) -> FrozenDict:
        return self._from_hf.resolve(hf_config)

    def __call__(self, hf_config: Any = None, **kw: Any) -> ModelGeometry:
        return self._model_cls.derive_geometry(hf_config, **kw)

    def __eq__(self, other: object) -> Any:
        if isinstance(other, _BoundFromHF):
            return self._from_hf == other._from_hf
        return self._from_hf == other if isinstance(other, FromHF) else NotImplemented

    def __repr__(self) -> str:
        return repr(self._from_hf)


class ModelDef:
    """Declarative model definition base class for the hybrid GQA/GDN + MoE family."""

    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    hf_architectures: ClassVar[tuple[str, ...]] = ()
    hf_model_type: ClassVar[tuple[str, ...]] = ()
    supported_tp_sizes: ClassVar[tuple[int, ...]] = ()
    layer_type_name: ClassVar[str | None] = None
    geometry: ClassVar[Any] = None
    schedule: ClassVar[Schedule | None] = None
    embed: ClassVar[Any] = None
    final_norm: ClassVar[ModuleOrBinding | None] = None
    lm_head: ClassVar[LMHead | None] = None
    profiles: ClassVar[Mapping[str, Profile]] = FrozenDict()
    default_hf_config: ClassVar[Mapping[str, Any] | None] = None
    pool_layout_override: ClassVar[Callable[..., PoolLayout] | None] = None
    geometry_cls: ClassVar[type[ModelGeometry]] = ModelGeometry
    hf_oracle: ClassVar[Callable[..., tuple[Any, Any]] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        defaults = {
            "name": "",
            "display_name": "",
            "hf_architectures": (),
            "hf_model_type": (),
        }
        for k, v in defaults.items():
            if k not in cls.__dict__:
                setattr(cls, k, v)

    @classmethod
    def _from_hf_decl(cls) -> FromHF:
        raw = cls.__dict__.get("geometry") or getattr(cls, "geometry", None)
        if isinstance(raw, _BoundFromHF):
            return raw._from_hf
        if isinstance(raw, FromHF):
            return raw
        mname = validate.model_label(cls)
        raise MegakernelConfigError(
            f"ModelDef {mname} must declare `geometry = FromHF(...)`"
        )

    @classmethod
    def _effective_hf_config(cls, hf_config: Any = None) -> Any:
        if hf_config is not None or cls.default_hf_config is not None:
            return hf_config if hf_config is not None else cls.default_hf_config
        raise MegakernelConfigError(
            f"ModelDef {validate.model_label(cls)!r} has no default_hf_config; "
            "pass an explicit hf_config"
        )

    @classmethod
    def _find_binding(
        cls, target_cls: type[Module], sched: ResolvedSchedule
    ) -> ModuleBinding | None:
        for b in (*sched.mixer, *sched.ffn):
            if isinstance(b, ModuleBinding) and issubclass(b.module_cls, target_cls):
                return b
        return None

    @classmethod
    def derive_geometry(cls, hf_config: Any = None, **kw: Any) -> ModelGeometry:
        """Derive `ModelGeometry` from `cls.geometry`, bindings, and `schedule`."""
        return derive.derive_model_geometry(cls, hf_config, **kw)

    @classmethod
    def resolve_schedule(cls, hf_config_or_geom: Any = None) -> ResolvedSchedule:
        """Resolve per-layer schedule for `hf_config_or_geom`."""
        return derive.resolve_model_schedule(cls, hf_config_or_geom)

    @classmethod
    def layer_cache_decls(
        cls, hf_config_or_geom: Any = None
    ) -> list[KVCacheDecl | RecurrentCacheDecl]:
        """Return the per-layer cache declaration sequence for `hf_config_or_geom`."""
        g = cls.derive_geometry(hf_config_or_geom)
        return cls.resolve_schedule(g).layer_cache_decls(g)

    @classmethod
    def pool_layout(cls, hf_config_or_geom: Any = None, *, R: int) -> PoolLayout:
        """Derive `PoolLayout` from per-layer `cache_decl()` + schedule grouping."""
        g = cls.derive_geometry(hf_config_or_geom)
        if cls.pool_layout_override is not None:
            if not callable(cls.pool_layout_override):
                mname = validate.model_label(cls)
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: pool_layout_override not callable"
                )
            return cls.pool_layout_override(g, R=R)
        return pool.derive_pool_layout(cls.layer_cache_decls(g), R=R)

    @classmethod
    def pool_backend(cls, hf_config_or_geom: Any = None, *, R: int) -> PoolBackend:
        """Create the `PoolBackend` for `hf_config_or_geom`."""
        g = cls.derive_geometry(hf_config_or_geom)
        return pool.make_pool_backend(
            cls.pool_layout(g, R=R),
            check=getattr(g, "check", None),
            rec_chunk=GDNModule.chunk_rows,
        )

    @classmethod
    def to_model_spec(cls, hf_config_or_geom: Any = None, **kw: Any) -> ModelSpec:
        """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s, deduping shared modules)."""
        return derive.derive_model_spec(cls, hf_config_or_geom, **kw)

    @classmethod
    def shrink_hf_config(cls, hf_config: Any = None, **kw: Any) -> dict[str, Any]:
        """Return a shrunk HF config dict (`periods` cycles, legal heads/experts)."""
        return derive.shrink_model_hf_config(cls, hf_config, **kw)

    @classmethod
    def shrink(cls, hf_config_or_geom: Any = None, **kw: Any) -> ModelGeometry:
        """Return a tiny `ModelGeometry` with `periods` cycles that passes checks."""
        return cls.derive_geometry(hf_config_or_geom).shrink(**kw)

    @classmethod
    def default_profile(cls) -> Profile:
        """Return the registered default `Profile` for this model."""
        return model_default_profile(validate.model_label(cls))

    @classmethod
    def get_profile(cls, name: str | None = None) -> Profile:
        """Resolve a profile by `cls.profiles` alias key or global profile name."""
        key = (name or "default").strip()
        if key in (cls.profiles or {}):
            return cls.profiles[key]
        return cls.default_profile() if key == "default" else get_profile(key)

    @classmethod
    def validate_hf_config(cls, hf_config: Any) -> EligibilityResult:
        """Validate `hf_config` against this model's schedule and geometry."""
        return validate.validate_model_hf_config(cls, hf_config)

    @classmethod
    def reference_step(
        cls,
        g: ModelGeometry,
        ex: ExecSpec,
        pos: jax.Array,
        hidden: jax.Array,
        logical_weights: Mapping[str, Any],
        logical_states: Mapping[str, Any],
        *,
        paged_kv: bool | None = None,
        pool_io: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Composed pure-JAX multi-rank reference along `schedule`."""
        args = (g, ex, pos, hidden, logical_weights, logical_states)
        kw = dict(paged_kv=paged_kv, pool_io=pool_io)
        if not cls.schedule:
            return spec_reference_step(cls.to_model_spec(g, R=ex.R), *args, **kw)
        sched = cls.resolve_schedule(g)
        validate.validate_norms(
            cls, sched, None, g.eps, expected_zc=g.zero_centered_norm
        )
        return composed_reference_step(sched, cls.final_norm, *args, **kw)


_raw_geom_shrink = ModelGeometry.shrink


def _from_hf_get(
    self: FromHF, obj: Any, owner: type[Any] | None = None
) -> FromHF | _BoundFromHF:
    if (
        isinstance(owner, type)
        and issubclass(owner, ModelDef)
        and owner is not ModelDef
    ):
        return _BoundFromHF(self, owner)
    return self


def _geom_shrink(self: ModelGeometry, **kw: Any) -> ModelGeometry:
    shrunk = _raw_geom_shrink(self, **kw)
    if (mdef := getattr(self, "_model_def", None)) is not None:
        object.__setattr__(shrunk, "_model_def", mdef)
    return shrunk


def _geom_to_model_spec(self: ModelGeometry) -> ModelSpec:
    """Convert this geometry into its declarative `ModelSpec`."""
    if (mdef := getattr(self, "_model_def", None)) is not None:
        return mdef.to_model_spec(self)
    raise MegakernelConfigError(
        "ModelGeometry is not bound to a ModelDef; call ModelDef.to_model_spec(g)"
    )


type.__setattr__(FromHF, "__get__", _from_hf_get)
type.__setattr__(ModelGeometry, "shrink", _geom_shrink)
type.__setattr__(ModelGeometry, "to_model_spec", _geom_to_model_spec)

_MODEL_REGISTRY: dict[str, type[ModelDef]] = {}
_ARCH_REGISTRY: dict[str, type[ModelDef]] = {}
_MODEL_TYPE_REGISTRY: dict[str, type[ModelDef]] = {}


def register_model(
    *,
    name: str,
    hf_architectures: Sequence[str] = (),
    hf_model_type: str | Sequence[str] | None = None,
    replace: bool = False,
) -> Callable[[type[ModelDef]], type[ModelDef]]:
    """Class decorator registering a declarative `ModelDef` subclass."""
    archs = tuple(hf_architectures)
    mtypes = (
        ()
        if hf_model_type is None
        else (
            (hf_model_type,) if isinstance(hf_model_type, str) else tuple(hf_model_type)
        )
    )

    def _decorator(cls: type[ModelDef]) -> type[ModelDef]:
        if name in _MODEL_REGISTRY and not replace:
            if _MODEL_REGISTRY[name] is cls:
                return cls
            prev_cls = _MODEL_REGISTRY[name].__name__
            raise MegakernelConfigError(
                f"Model {name!r} is already registered by {prev_cls}"
            )
        for kind, reg, keys in (
            ("HF architecture", _ARCH_REGISTRY, archs),
            ("HF model_type", _MODEL_TYPE_REGISTRY, mtypes),
        ):
            for k in keys:
                prev = reg.get(k)
                if (
                    prev is not None
                    and prev is not cls
                    and prev.name != name
                    and not replace
                ):
                    raise MegakernelConfigError(
                        f"{kind} {k!r} is already registered to {prev.name!r}"
                    )
        validate.validate_model_def(cls, name, archs, mtypes, ModelDef)
        if replace:
            old_cls = _MODEL_REGISTRY.get(name)
            for reg in (_ARCH_REGISTRY, _MODEL_TYPE_REGISTRY):
                for k in [rk for rk, rv in reg.items() if rv in (cls, old_cls)]:
                    del reg[k]
        cls.name, cls.hf_architectures, cls.hf_model_type = name, archs, mtypes
        cls.profiles = FrozenDict(cls.profiles or {})
        for prof in cls.profiles.values():
            register_profile(prof, replace=replace)
        _MODEL_REGISTRY[name] = cls
        for a in archs:
            _ARCH_REGISTRY[a] = cls
        for mt in mtypes:
            _MODEL_TYPE_REGISTRY[mt] = cls
        return cls

    return _decorator


def list_models() -> tuple[str, ...]:
    """Return sorted names of all registered `ModelDef`s."""
    return tuple(sorted(_MODEL_REGISTRY))


def find_model(target: Any) -> type[ModelDef] | None:
    """Look up a `ModelDef` by name, HF architecture, `model_type`, or `hf_config`."""
    if isinstance(target, type) and issubclass(target, ModelDef):
        return target
    if (mdef := getattr(target, "_model_def", None)) is not None:
        return mdef
    if isinstance(target, str):
        key = target.strip()
        return (
            _MODEL_REGISTRY.get(key)
            or _ARCH_REGISTRY.get(key)
            or _MODEL_TYPE_REGISTRY.get(key)
        )
    if target is not None:
        cands: list[str] = []
        archs = _sub_cfg(target, "architectures") or ()
        if isinstance(archs, Sequence) and not isinstance(archs, (str, bytes)):
            cands.extend(str(a) for a in archs if a)
        for root in (target, _sub_cfg(target, "text_config")):
            if root is not None and (mt := _sub_cfg(root, "model_type")):
                cands.append(str(mt))
        for cand in cands:
            if (found := find_model(cand)) is not None:
                return found
    return None


def get_model(target: Any) -> type[ModelDef]:
    """Resolve a `ModelDef` by name, HF architecture, `model_type`, or `hf_config`."""
    found = find_model(target)
    if found is None:
        raise KeyError(
            f"Unknown megakernel model {target!r}; available: {list(list_models())}"
        )
    return found


__all__ = [
    "Every",
    "FromHF",
    "FromHFLayerTypes",
    "LMHead",
    "ModelDef",
    "ModelGeometry",
    "ResolvedSchedule",
    "Schedule",
    "find_model",
    "get_model",
    "list_models",
    "register_model",
]

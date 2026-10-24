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

from . import pool
from .decls import (
    FrozenDict,
    KVCacheDecl,
    ModuleBinding,
    RecurrentCacheDecl,
    _sub_cfg,
)
from .derive import (
    derive_model_geometry,
    derive_model_spec,
    resolve_model_schedule,
    shrink_model_hf_config,
)
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import FromHF, LMHead, ModelGeometry, _BoundFromHF
from .modules.gdn import GDNModule
from .pool import PoolBackend, PoolLayout
from .profile import Profile, get_profile, model_default_profile, register_profile
from .reference import composed_reference_step
from .schedule import (
    Every,
    FromHFLayerTypes,
    ModuleOrBinding,
    ResolvedSchedule,
    Schedule,
)
from .spec import ModelSpec, Module
from .validate import model_label, validate_model_def, validate_norms


class ModelDef:
    """Declarative model definition base class for the hybrid GQA/GDN + MoE family."""

    _IS_MODEL_DEF: ClassVar[bool] = True
    name: ClassVar[str] = ""
    hf_architectures: ClassVar[tuple[str, ...]] = ()
    hf_model_type: ClassVar[tuple[str, ...]] = ()
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
    # A segment-composed model runs as several kernel calls per step (e.g. one
    # per span between attention cores that run outside the kernel), so it has
    # no whole-model `schedule`. It overrides `derive_geometry`,
    # `to_model_spec` and `pool_layout` itself, and whole-model drivers refuse
    # it.
    segment_composed: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for attr, empty in (
            ("name", ""),
            ("hf_architectures", ()),
            ("hf_model_type", ()),
        ):
            if attr not in cls.__dict__:
                setattr(cls, attr, empty)

    @classmethod
    def _from_hf_decl(cls) -> FromHF:
        raw = cls.__dict__.get("geometry") or getattr(cls, "geometry", None)
        if isinstance(raw, _BoundFromHF):
            return raw._from_hf
        if isinstance(raw, FromHF):
            return raw
        raise MegakernelConfigError(
            f"ModelDef {model_label(cls)} must declare `geometry = FromHF(...)`"
        )

    @classmethod
    def _effective_hf_config(cls, hf_config: Any = None) -> Any:
        if hf_config is not None:
            return hf_config
        if cls.default_hf_config is not None:
            return cls.default_hf_config
        raise MegakernelConfigError(
            f"ModelDef {model_label(cls)!r} has no default_hf_config; "
            "pass an explicit hf_config"
        )

    @classmethod
    def _find_binding(
        cls, target_cls: type[Module], sched: ResolvedSchedule
    ) -> ModuleBinding | None:
        for seq in (sched.mixer, sched.ffn):
            for b in seq:
                if isinstance(b, ModuleBinding) and issubclass(
                    b.module_cls, target_cls
                ):
                    return b
        return None

    @classmethod
    def derive_geometry(
        cls,
        hf_config: Any = None,
        *,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
    ) -> Any:
        """Derive `ModelGeometry` from `cls.geometry`, bindings, and `schedule`."""
        return derive_model_geometry(cls, hf_config, moe_tp=moe_tp, e_slots=e_slots)

    @classmethod
    def resolve_schedule(cls, hf_config_or_geom: Any = None) -> ResolvedSchedule:
        """Resolve per-layer schedule for `hf_config_or_geom`."""
        return resolve_model_schedule(cls, hf_config_or_geom)

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
                raise MegakernelConfigError(
                    f"ModelDef {model_label(cls)!r}: pool_layout_override not callable"
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
    def to_model_spec(
        cls,
        hf_config_or_geom: Any = None,
        *,
        gdn_module: Module | None = None,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
        R: int = 8,
    ) -> ModelSpec:
        """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s, deduping shared modules)."""
        return derive_model_spec(
            cls,
            hf_config_or_geom,
            gdn_module=gdn_module,
            moe_tp=moe_tp,
            e_slots=e_slots,
            R=R,
        )

    @classmethod
    def shrink_hf_config(
        cls,
        hf_config: Any = None,
        *,
        periods: int = 2,
        R: int = 8,
        ex: ExecSpec | None = None,
    ) -> dict[str, Any]:
        """Return a shrunk HF config dict (`periods` cycles, legal heads/experts)."""
        return shrink_model_hf_config(cls, hf_config, periods=periods, R=R, ex=ex)

    @classmethod
    def shrink(
        cls,
        hf_config_or_geom: Any = None,
        *,
        periods: int = 2,
        R: int = 8,
        ex: ExecSpec | None = None,
    ) -> ModelGeometry:
        """Return a tiny `ModelGeometry` with `periods` cycles that passes checks."""
        return cls.derive_geometry(hf_config_or_geom).shrink(
            periods=periods, R=R, ex=ex
        )

    @classmethod
    def default_profile(cls) -> Profile:
        """Return the registered default `Profile` for this model."""
        return model_default_profile(model_label(cls))

    @classmethod
    def get_profile(cls, name: str | None = None) -> Profile:
        """Resolve a profile by `cls.profiles` alias key or global profile name."""
        key = (name or "default").strip()
        if key in (cls.profiles or {}):
            return cls.profiles[key]
        return cls.default_profile() if key == "default" else get_profile(key)

    @classmethod
    def reference_step(
        cls,
        g: ModelGeometry,
        ex: ExecSpec,
        pos: jax.Array,
        hidden: jax.Array,
        logical_weights: dict[str, Any],
        logical_states: dict[str, Any],
        *,
        paged_kv: bool | None = None,
        pool_io: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Composed pure-JAX multi-rank reference along `schedule`."""
        sched = cls.resolve_schedule(g)
        validate_norms(cls, sched, None, g.eps, expected_zc=g.zero_centered_norm)
        return composed_reference_step(
            sched,
            cls.final_norm,
            g,
            ex,
            pos,
            hidden,
            logical_weights,
            logical_states,
            paged_kv=paged_kv,
            pool_io=pool_io,
        )


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
    if hf_model_type is None:
        mtypes: tuple[str, ...] = ()
    elif isinstance(hf_model_type, str):
        mtypes = (hf_model_type,)
    else:
        mtypes = tuple(hf_model_type)

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
        validate_model_def(cls, name, archs, mtypes, ModelDef)
        if replace:
            old_cls = _MODEL_REGISTRY.get(name)
            for reg in (_ARCH_REGISTRY, _MODEL_TYPE_REGISTRY):
                for k in [rk for rk, rv in reg.items() if rv in (cls, old_cls)]:
                    del reg[k]
        cls.name = name
        cls.hf_architectures = archs
        cls.hf_model_type = mtypes
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

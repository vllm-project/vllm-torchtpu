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
"""Stdlib-only module parameter, cache, and geometry declaration primitives."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .module import Module


class _RequiredType:
    """Singleton sentinel marking a required geometry field with no default."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "REQUIRED"

    def __reduce__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _RequiredType()


def _freeze_val(v: Any) -> Any:
    if isinstance(v, Mapping):
        return v if isinstance(v, FrozenDict) else FrozenDict(v)
    return tuple(_freeze_val(x) for x in v) if isinstance(v, (list, tuple)) else v


class FrozenDict(Mapping[str, Any]):
    """Order-independent hashable, picklable, deep-copyable immutable mapping."""

    __slots__ = ("_dict",)
    _dict: dict[str, Any]

    def __init__(self, data: Any = ()) -> None:
        d = {k: _freeze_val(v) for k, v in dict(data).items()}
        object.__setattr__(self, "_dict", d)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __hash__(self) -> int:
        return hash(frozenset(self._dict.items()))

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Mapping):
            return NotImplemented
        od = other if isinstance(other, FrozenDict) else FrozenDict(other)
        return self._dict == od._dict

    def __repr__(self) -> str:
        return f"FrozenDict({self._dict!r})"

    def __reduce__(self) -> tuple[Any, Any]:
        return (FrozenDict, (tuple(self._dict.items()),))


@dataclass(frozen=True)
class KVCacheDecl:
    """KV-page cache requirement declared by an attention module."""

    hd: int
    kvh: int = 1
    page_size: int = 128
    page_dtype: Any = None

    @property
    def kv_page_shape(self) -> tuple[int, int, int, int]:
        return (2 * self.kvh, self.hd // 4, 4, self.page_size)


@dataclass(frozen=True)
class RecurrentCacheDecl:
    """Recurrent + causal-conv slot requirement declared by a recurrent module."""

    kh: int
    vh: int
    KD: int = 128
    VD: int = 128
    rec_dtype: Any = None
    rec_pages_per_slot: int | None = None
    conv_taps: int = 3
    conv_dtype: Any = None
    conv_pages_per_slot: int | None = None


@dataclass(frozen=True)
class ParamInit:
    """Declarative synthetic initialization parameters for an `HFParamDecl`."""

    mean: float = 0.0
    std: float = 0.05


@dataclass(frozen=True)
class ParamTransform:
    """Declarative transform rules for a canonical module parameter."""

    fuse_into: str | None = None
    fuse_order: int = 0
    fuse_axis: int = -1
    split_from: str | None = None
    split_slice: tuple[int, int] | None = None
    transpose: tuple[int, ...] | None = None
    pad: tuple[tuple[int, int], ...] | None = None
    quantize: str | None = None
    scale_for: str | None = None
    shard_axis: int | None = None
    kv_repeat: bool = False
    replicate_ranks: bool = False
    head_reduce: str | None = None
    channel_splits: tuple[int, ...] | None = None
    flatten_tail: bool = False
    expert_axis: int | None = None
    small_field: str | None = None

    def __post_init__(self) -> None:
        if self.transpose is not None and (
            not isinstance(self.transpose, tuple)
            or not all(type(i) is int for i in self.transpose)
        ):
            raise TypeError("transpose must be a permutation tuple or None")


_TRANSFORM_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(ParamTransform))


@dataclass(frozen=True)
class VLLMAlias:
    """Mapping from a live vLLM module parameter/submodule to canonical names."""

    source: str
    targets: tuple[str, ...]
    scale_targets: tuple[str, ...] = ()
    scale_source: str | None = None
    shard_axis: int | None = None
    already_sharded: bool = True
    already_fp8: bool = True
    flipped: bool = False


@dataclass(frozen=True)
class HFParamDecl:
    """Canonical parameter declaration (`shape` in MK layout, `hf_shape` in raw HF)."""

    name: str
    hf_name: str
    shape: tuple[int, ...]
    dtype: Any
    hf_shape: tuple[int, ...] | None = None
    transform: ParamTransform = ParamTransform()
    vllm_aliases: tuple[VLLMAlias, ...] = ()
    init: ParamInit | None = None

    def __post_init__(self) -> None:
        perm = self.transform.transpose
        if perm is not None and sorted(perm) != list(range(len(self.shape))):
            raise ValueError(f"not a valid rank-{len(self.shape)} permutation")
        if self.hf_shape is None:
            hf_sh = (
                tuple(self.shape[perm.index(i)] for i in range(len(perm)))
                if perm is not None
                else self.shape
            )
            object.__setattr__(self, "hf_shape", hf_sh)


def apply_decl_overrides(decls: Sequence[HFParamDecl], geom: Any) -> list[HFParamDecl]:
    """Apply `geom.transforms` overrides to a module's `HFParamDecl` list."""
    transforms = getattr(geom, "transforms", None)
    if not transforms:
        return list(decls)
    inv_rename = {v: k for k, v in (getattr(geom, "rename", {}) or {}).items()}
    out: list[HFParamDecl] = []
    for d in decls:
        ov = transforms.get(d.name, transforms.get(inv_rename.get(d.name, "")))
        if ov is None:
            out.append(d)
        elif isinstance(ov, ParamTransform):
            out.append(dataclasses.replace(d, transform=ov))
        elif isinstance(ov, Mapping):
            tf = dataclasses.replace(d.transform, **dict(ov))
            out.append(dataclasses.replace(d, transform=tf))
        else:
            raise TypeError(f"Invalid transforms[{d.name!r}]: {type(ov).__name__}")
    return out


def combined_hf_param_shapes(
    decls: Sequence[HFParamDecl],
) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Derive unsharded combined `(shape, dtype)` for uniform `fuse_into` groups."""
    groups: dict[str, list[HFParamDecl]] = {}
    for d in decls:
        if d.transform.fuse_into is not None:
            groups.setdefault(d.transform.fuse_into, []).append(d)
    out: dict[str, tuple[tuple[int, ...], Any]] = {}
    for target, members in groups.items():
        ms = sorted(members, key=lambda m: m.transform.fuse_order)
        first, ndim = ms[0], len(ms[0].shape)
        axis = first.transform.fuse_axis % ndim
        if all(
            len(m.shape) == ndim
            and (m.transform.fuse_axis % ndim) == axis
            and all(m.shape[i] == first.shape[i] for i in range(ndim) if i != axis)
            and m.transform.pad is None
            for m in ms
        ):
            dim = sum(m.shape[axis] for m in ms)
            sh = tuple(dim if i == axis else first.shape[i] for i in range(ndim))
            out[target] = (sh, first.dtype)
    return out


def _sub_cfg(c: Any, name: str) -> Any:
    return c.get(name) if isinstance(c, Mapping) else getattr(c, name, None)


def _lookup_raw_key(cfg: Any, key: str) -> Any:
    if cfg is None:
        return REQUIRED
    tc = _sub_cfg(cfg, "text_config")
    roots = (cfg, tc) if tc is not None else (cfg,)
    subs = [
        sub
        for r in roots
        for s in ("rope_parameters", "rope_scaling")
        if (sub := _sub_cfg(r, s)) is not None
    ]
    return next(
        (v for c in (*roots, *subs) if (v := _sub_cfg(c, key)) is not None), REQUIRED
    )


def _resolve_field_spec(
    hf_config: Any,
    canon: str,
    spec_val: Any,
    fallback: Any = REQUIRED,
    aliases: Sequence[str] = (),
    *,
    explicit: bool = False,
) -> Any:
    if isinstance(spec_val, (int, float, bool)):
        return spec_val
    if (
        isinstance(spec_val, tuple)
        and len(spec_val) == 2
        and isinstance(spec_val[0], str)
    ):
        hf_key, default = spec_val
    elif isinstance(spec_val, str):
        hf_key, default = spec_val, (REQUIRED if explicit else fallback)
    else:
        return spec_val
    for cand in (hf_key,) if explicit else (hf_key, canon, *aliases):
        if (found := _lookup_raw_key(hf_config, cand)) is not REQUIRED:
            return found
    if default is not REQUIRED:
        return default
    raise ValueError(f"Cannot resolve field {canon!r} (missing HF key {hf_key!r})")


def validate_rename_and_transforms(
    module_cls: type[Module],
    rename: Mapping[str, str],
    transforms: Mapping[str, Any],
    *,
    param_name_override: str | None = None,
) -> None:
    """Validate `rename` and `transforms` keys against `module_cls.PARAM_NAMES`."""
    cls_n = module_cls.__name__
    extra = {param_name_override} if param_name_override else set()
    allowed = set(getattr(module_cls, "PARAM_NAMES", ())) | extra
    seen: set[str] = set()
    for k, v in rename.items():
        if allowed and k not in allowed:
            raise ValueError(f"{cls_n}.bind got unknown rename key {k!r}")
        if not isinstance(v, str) or not v:
            raise ValueError(f"{cls_n}.bind invalid rename[{k!r}]={v!r}")
        if (v in allowed and v != k) or v in seen:
            raise ValueError(f"{cls_n}.bind rename target collision {v!r}")
        seen.add(v)
    allowed_tf = allowed | seen
    for k, v in transforms.items():
        if allowed_tf and k not in allowed_tf:
            raise ValueError(f"{cls_n}.bind got unknown transforms key {k!r}")
        if isinstance(v, Mapping):
            if bad := sorted(set(v) - _TRANSFORM_FIELD_NAMES):
                raise ValueError(f"{cls_n}.bind transforms[{k!r}] unknown fields {bad}")
        elif not isinstance(v, ParamTransform):
            raise TypeError(f"{cls_n}.bind transforms[{k!r}] invalid type")


@dataclass(frozen=True)
class ResolvedModuleGeometry:
    """Resolved module geometry and options from `ModuleBinding.resolve(hf_config)`."""

    module_cls: type[Module]
    prefix: str
    values: Mapping[str, Any] = field(default_factory=FrozenDict)
    options: Mapping[str, Any] = field(default_factory=FrozenDict)
    rename: Mapping[str, str] = field(default_factory=FrozenDict)
    transforms: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for attr in ("values", "options", "rename", "transforms"):
            object.__setattr__(self, attr, FrozenDict(getattr(self, attr)))

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "values")
        options = object.__getattribute__(self, "options")
        module_cls = object.__getattribute__(self, "module_cls")
        cands = (
            name,
            module_cls.FIELD_ALIASES.get(name),
            *(a for a, t in module_cls.FIELD_ALIASES.items() if t == name),
        )
        for key in cands:
            if key is not None and (key in values or key in options):
                return values[key] if key in values else options[key]
        if (fn := module_cls.GEOMETRY_METHODS.get(name)) is not None:
            return lambda *a, _fn=fn, **kw: _fn(self, *a, **kw)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def geometry(self) -> ResolvedModuleGeometry:
        return self

    def to_module(self) -> Module:
        return self.module_cls._from_resolved(self)

    def hf_params(self, prefix: str | None = None) -> list[HFParamDecl]:
        return self.to_module().hf_params(
            self.prefix if prefix is None else prefix, self
        )

    def cache_decl(self) -> KVCacheDecl | RecurrentCacheDecl | None:
        return self.to_module().cache_decl(self)

    def reference(self, params: Any, x: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.to_module().reference(params, x, state, **kwargs)


@dataclass(frozen=True)
class ModuleBinding:
    """Frozen declarative module binding returned by `Module.bind(...)`."""

    module_cls: type[Module]
    prefix: str
    fields: Mapping[str, Any] = field(default_factory=FrozenDict)
    options: Mapping[str, Any] = field(default_factory=FrozenDict)
    rename: Mapping[str, str] = field(default_factory=FrozenDict)
    transforms: Mapping[str, Any] = field(default_factory=FrozenDict)
    explicit_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for attr in ("fields", "options", "rename", "transforms"):
            object.__setattr__(self, attr, FrozenDict(getattr(self, attr)))
        object.__setattr__(self, "explicit_fields", frozenset(self.explicit_fields))

    def resolve(self, hf_config: Any) -> ResolvedModuleGeometry:
        return self.module_cls._resolve_binding(self, hf_config)

    def to_module(self, hf_config: Any) -> Module:
        return self.resolve(hf_config).to_module()

    def hf_params(self, hf_config: Any, prefix: str | None = None) -> list[HFParamDecl]:
        return self.resolve(hf_config).hf_params(
            self.prefix if prefix is None else prefix
        )

    def cache_decl(self, hf_config: Any) -> KVCacheDecl | RecurrentCacheDecl | None:
        return self.resolve(hf_config).cache_decl()


def normalize_weight_dtypes(
    raw: Any,
    allowed: Sequence[str],
    module_name: str,
    err_cls: type[Exception] = ValueError,
) -> FrozenDict:
    """Validate and normalize per-weight dtype policies (`'fp8' | 'bf16'`)."""
    out = {k: "fp8" for k in allowed}
    if raw is None:
        return FrozenDict(out)
    if not isinstance(raw, Mapping):
        raise err_cls(
            f"{module_name} weight_dtypes must be a Mapping, got {type(raw).__name__}"
        )
    allowed_set = set(allowed)
    for k, v in raw.items():
        if k not in allowed_set:
            raise err_cls(
                f"{module_name} weight_dtypes got unknown weight {k!r}; "
                f"supported: {sorted(allowed_set)}"
            )
        if v in ("fp8", "fp8_channel", "per_channel_fp8"):
            out[k] = "fp8"
        elif v == "bf16":
            out[k] = "bf16"
        else:
            raise err_cls(
                f"{module_name} weight_dtypes[{k!r}] must be 'fp8' or 'bf16', got {v!r}"
            )
    return FrozenDict(out)

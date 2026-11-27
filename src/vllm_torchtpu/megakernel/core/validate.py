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
"""`ModelDef` registration and norm consistency validation helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from . import decls, schedule
from .decls import REQUIRED, ModuleBinding
from .errors import EligibilityResult, MegakernelConfigError
from .geometry import FromHF
from .modules.moe import MoEModule
from .modules.norm import NormModule
from .profile import Profile
from .schedule import ModuleOrBinding, ResolvedSchedule
from .spec import Module

_ROPE_KEYS = ("rope_parameters", "rope_scaling")
_SKIP_KEYS = frozenset(
    {
        "architectures",
        "model_type",
        "rope_parameters",
        "rope_scaling",
        "text_config",
        "torch_dtype",
        "transformers_version",
    }
)
NormSlotPair = tuple[Sequence[ModuleOrBinding | None], Sequence[ModuleOrBinding | None]]


def _cfg_map(c: Any) -> dict[str, Any]:
    if isinstance(c, Mapping):
        return dict(c)
    d = getattr(c, "__dict__", None)
    return {k: v for k, v in d.items() if k[:1] != "_"} if isinstance(d, dict) else {}


def _merge_partial_root(
    base: dict[str, Any], present: Mapping[str, Any], ropes: Mapping[str, Any]
) -> dict[str, Any]:
    out = {**base, **present}
    for rk in _ROPE_KEYS:
        if rk in base or rk in ropes:
            br = _cfg_map(base.get(rk))
            flat = {k: present[k] for k in br if k in present}
            out[rk] = {**br, **flat, **ropes.get(rk, {})}
    return out


def _is_int(v: Any) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_clean_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip()) and v.strip() == v


def _hf_key(v: Any) -> str:
    return v[0] if isinstance(v, tuple) else str(v)


def model_label(cls: Any, model_name: str | None = None) -> str:
    """Return the human-readable model label for `cls`."""
    own = cls.__dict__.get("name") if isinstance(cls, type) else None
    return model_name or (str(own) if own else getattr(cls, "__name__", str(cls)))


def describe_binding(
    m: ModuleOrBinding | None, expected: ModuleOrBinding | None = None
) -> str:
    """Format a `Module` or `ModuleBinding` for error messages."""
    if m is None:
        return "None"
    label = f"{schedule._module_name_of(m)}/{schedule._module_cls_of(m).__name__}"
    m_opts = {k: v for k, v in m.options.items() if v is not None}
    if expected is not None:
        e_opts = {k: v for k, v in expected.options.items() if v is not None}
        parts = [
            f"{k}={m_opts.get(k)!r} (expected {e_opts.get(k)!r})"
            for k in sorted(set(m_opts) | set(e_opts))
            if m_opts.get(k) != e_opts.get(k)
        ]
        if parts:
            return f"{label}(prefix={m.prefix!r}, {', '.join(parts)})"
    parts = [f"{k}={v!r}" for k, v in sorted(m_opts.items())]
    tail = f", {', '.join(parts)}" if parts else ""
    return f"{label}(prefix={m.prefix!r}{tail})"


def norm_zc_and_eps(
    m: ModuleOrBinding,
    merged: dict[str, Any] | None,
    default_eps: float | None = None,
) -> tuple[bool, float | None]:
    """Extract `(zero_centered, eps)` from a norm `Module` or `ModuleBinding`."""
    if not issubclass(schedule._module_cls_of(m), NormModule):
        raise MegakernelConfigError(
            f"Expected NormModule binding, got {describe_binding(m)}"
        )
    zc = bool(m.options.get("zero_centered", False))
    if isinstance(m, ModuleBinding) and "eps" in m.explicit_fields:
        if merged is not None:
            return zc, float(m.resolve(merged).eps)
        raw_eps = m.fields.get("eps")
        if isinstance(raw_eps, (int, float)) and not isinstance(raw_eps, bool):
            return zc, float(raw_eps)
    elif isinstance(m, NormModule) and m.eps is not None:
        return zc, float(m.eps)
    return zc, default_eps


def validate_norms(
    cls: Any,
    sched: ResolvedSchedule | NormSlotPair,
    merged: dict[str, Any] | None,
    eps: float | None = None,
    expected_zc: bool | None = None,
    *,
    model_name: str | None = None,
) -> bool:
    """Validate `pre_norm`, `post_norm`, and `final_norm` presence and agreement."""
    pfx = f"ModelDef {model_label(cls, model_name)!r}"
    is_res = isinstance(sched, ResolvedSchedule)
    pre_s, post_s = (sched.pre_norm, sched.post_norm) if is_res else sched
    entries: list[tuple[str, ModuleOrBinding]] = []
    for slot_name, seq in (("pre_norm", pre_s), ("post_norm", post_s)):
        for i, m in enumerate(seq):
            if m is None:
                raise MegakernelConfigError(
                    f"{pfx}: layer {i} slot {slot_name!r} is None "
                    f"(kernel requires {slot_name} on every layer)"
                )
            entries.append((f"layer {i} slot {slot_name!r}", m))
    if cls.final_norm is None:
        raise MegakernelConfigError(
            f"{pfx}: final_norm is None (kernel requires final_norm)"
        )
    entries.append(("final_norm", cls.final_norm))
    first_loc, first_m = entries[0]
    first_zc, _ = norm_zc_and_eps(first_m, merged, eps)
    target_zc = first_zc if expected_zc is None else expected_zc
    zc_owner = first_loc if expected_zc is None else "model"
    for loc, m in entries:
        zc_i, eps_i = norm_zc_and_eps(m, merged, eps)
        if zc_i != target_zc:
            raise MegakernelConfigError(
                f"{pfx}: {loc} ({describe_binding(m)}) has zero_centered={zc_i}, "
                f"disagreeing with {zc_owner} zero_centered={target_zc}"
            )
        if eps is not None and eps_i != eps:
            raise MegakernelConfigError(
                f"{pfx}: {loc} ({describe_binding(m)}) has eps={eps_i}, "
                f"disagreeing with model eps={eps}"
            )
    return target_zc


def _infer_static_layers(cls: Any, from_hf: FromHF) -> int | None:
    ls = from_hf.fields.get("layers")
    raw_l = (
        decls._lookup_raw_key(cls.default_hf_config, _hf_key(ls))
        if cls.default_hf_config is not None
        else None
    )
    for cand in (raw_l, ls, ls[1] if isinstance(ls, tuple) else None):
        if _is_int(cand):
            return int(cand)
    mix = cls.schedule.mixer if isinstance(cls.schedule, schedule.Schedule) else None
    if isinstance(mix, Sequence) and not isinstance(mix, (str, bytes)):
        return len(mix)
    return None


def validate_model_def(
    cls: Any,
    name: str,
    archs: tuple[str, ...],
    model_types: tuple[str, ...],
    base_cls: type,
) -> None:
    """Validate `cls` at `@register_model` decoration time."""
    if not _is_clean_str(name):
        raise MegakernelConfigError(f"register_model: invalid model name {name!r}")
    pfx = f"register_model({name!r})"
    if not issubclass(cls, base_cls):
        raise MegakernelConfigError(f"{pfx}: {cls.__name__} must subclass ModelDef")
    from_hf = cls._from_hf_decl()
    for req_key in ("hidden", "layers", "eps"):
        if req_key not in from_hf.fields:
            raise MegakernelConfigError(
                f"{pfx}: FromHF is missing required key {req_key!r}"
            )
    if not isinstance(cls.schedule, schedule.Schedule):
        raise MegakernelConfigError(
            f"{pfx}: `schedule` must be a Schedule instance, got {cls.schedule!r}"
        )
    if cls.pool_layout_override is not None and not callable(cls.pool_layout_override):
        raise MegakernelConfigError(f"{pfx}: pool_layout_override must be callable")
    for s_name in ("pre_norm", "post_norm"):
        if getattr(cls.schedule, s_name) is None:
            raise MegakernelConfigError(
                f"{pfx}: layer 0 slot {s_name!r} is None (kernel requires {s_name})"
            )
    if cls.final_norm is None:
        raise MegakernelConfigError(
            f"{pfx}: final_norm is None (kernel requires final_norm)"
        )

    static_l = _infer_static_layers(cls, from_hf)
    if static_l is not None:
        sch, lt_cls = cls.schedule, schedule.FromHFLayerTypes
        sch.validate_bounds(static_l)
        slots = (sch.mixer, sch.ffn, sch.pre_norm, sch.post_norm)
        if cls.default_hf_config is not None or not any(
            isinstance(s, lt_cls) for s in slots
        ):
            sched = sch.resolve(cls.default_hf_config or {}, static_l)
            validate_norms(cls, sched, None, eps=None, model_name=name)
        elif not any(isinstance(s, lt_cls) for s in slots[2:]):
            pre_l = sch.resolve_slot("pre_norm", {}, static_l, allow_none=True)
            post_l = sch.resolve_slot("post_norm", {}, static_l, allow_none=True)
            validate_norms(cls, (pre_l, post_l), None, eps=None, model_name=name)

    if cls.default_hf_config is not None:
        try:
            g_def = cls.derive_geometry(cls.default_hf_config)
            cls.pool_layout(g_def, R=8)
        except MegakernelConfigError as exc:
            if name not in str(exc):
                raise MegakernelConfigError(f"{pfx}: {exc}") from exc
            raise
        except Exception as exc:
            raise MegakernelConfigError(
                f"{pfx}: invalid default_hf_config: {exc}"
            ) from exc

    for alias, prof in (cls.profiles or {}).items():
        if not _is_clean_str(alias):
            raise MegakernelConfigError(f"{pfx}: invalid profile alias key {alias!r}")
        if not isinstance(prof, Profile):
            raise MegakernelConfigError(
                f"{pfx}: profiles[{alias!r}] must be a Profile, got {prof!r}"
            )
        if prof.model is not None and prof.model != name:
            raise MegakernelConfigError(
                f"{pfx}: profile {prof.name!r} targets model {prof.model!r}, "
                f"expected {name!r}"
            )
    for kind, seq in (("hf_architectures", archs), ("hf_model_type", model_types)):
        for entry in seq:
            if not _is_clean_str(entry):
                raise MegakernelConfigError(f"{pfx}: invalid {kind} entry {entry!r}")


def _resolved_ffn_bindings(
    cls: Any, hf_config: Any, from_hf: FromHF
) -> list[ModuleOrBinding]:
    sched = cls.schedule
    if not isinstance(sched, schedule.Schedule) or sched.ffn is None:
        return []
    l_spec = from_hf.fields.get("layers", "num_hidden_layers")
    num_l = _infer_static_layers(cls, from_hf)
    try:
        raw_l = decls._resolve_field_spec(hf_config, "layers", l_spec, explicit=True)
        if not isinstance(raw_l, bool) and int(raw_l) >= 1:
            num_l = int(raw_l)
    except (ValueError, TypeError):
        pass
    if num_l is not None and num_l >= 1:
        try:
            return [
                m for m in sched.resolve_slot("ffn", hf_config, num_l) if m is not None
            ]
        except MegakernelConfigError:
            pass
    spec = sched.ffn
    if isinstance(spec, (ModuleBinding, Module)):
        out: list[ModuleOrBinding] = [spec]
    elif isinstance(spec, schedule.Every):
        out = [spec.last] if spec.rest is None else [spec.rest, spec.last]
    elif isinstance(spec, schedule.FromHFLayerTypes):
        out = list(spec.mapping.values())
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        out = [x for x in spec if isinstance(x, (ModuleBinding, Module))]
    else:
        out = []
    out.extend(ov["ffn"] for ov in sched.overrides.values() if ov.get("ffn"))
    return out


def validate_model_hf_config(cls: Any, hf_config: Any) -> EligibilityResult:
    """Validate `hf_config` against `cls` (`ModelDef` schedule + geometry)."""
    if hf_config is None:
        return EligibilityResult(True)
    dis = f"Megakernel disabled for {cls.__name__}"
    from_hf = cls._from_hf_decl()
    for ffn in _resolved_ffn_bindings(cls, hf_config, from_hf):
        m_cls = schedule._module_cls_of(ffn)
        if not issubclass(m_cls, MoEModule):
            continue
        if not isinstance(ffn, ModuleBinding) and ffn.geometry is not None:
            exp_key, valid_e = "E", _is_int(ffn.geometry.E) and ffn.geometry.E > 0
        else:
            is_b = isinstance(ffn, ModuleBinding)
            def_e = m_cls.GEOMETRY_FIELDS["experts"][0]
            exp_spec = ffn.fields.get("experts", "num_experts") if is_b else def_e
            exp_key = _hf_key(exp_spec)
            als = tuple(a for a, t in m_cls.FIELD_ALIASES.items() if t == "experts")
            is_exp = is_b and "experts" in ffn.explicit_fields
            try:
                raw_e = decls._resolve_field_spec(
                    hf_config, "experts", exp_spec, REQUIRED, als, explicit=is_exp
                )
                valid_e = _is_int(raw_e) and raw_e > 0
            except (ValueError, TypeError):
                valid_e = False
        if not valid_e:
            msg = (
                f"{dis}: {cls.__name__} supports MoE checkpoints only "
                f"(missing or non-positive {exp_key!r}); skipping."
            )
            return EligibilityResult(False, "non_moe_checkpoint", msg)
    hid_key = _hf_key(from_hf.fields.get("hidden", "hidden_size"))
    has_hid = decls._lookup_raw_key(hf_config, hid_key) not in (None, REQUIRED)
    def_root = _cfg_map(cls.default_hf_config)
    def_tc = _cfg_map(decls._sub_cfg(cls.default_hf_config, "text_config"))
    f_keys = {_hf_key(v) for v in from_hf.fields.values() if isinstance(v, str | tuple)}
    def_keys = set(def_root) | set(def_tc)
    for rk in _ROPE_KEYS:
        def_keys |= set(_cfg_map(def_root.get(rk))) | set(_cfg_map(def_tc.get(rk)))
    p_geom = {
        k: val
        for k in (f_keys | def_keys) - _SKIP_KEYS
        if (val := decls._lookup_raw_key(hf_config, k)) not in (None, REQUIRED)
    }
    hf_tc = decls._sub_cfg(hf_config, "text_config")
    ropes = {}
    for rk in _ROPE_KEYS:
        sub_a = _cfg_map(decls._sub_cfg(hf_config, rk))
        sub_b = _cfg_map(decls._sub_cfg(hf_tc, rk))
        if sub_a or sub_b:
            ropes[rk] = {**sub_a, **sub_b}
    if has_hid or p_geom or ropes:
        if has_hid or cls.default_hf_config is None:
            cfg_for_geom: Any = hf_config
        else:
            cfg_for_geom = _merge_partial_root(def_root, p_geom, ropes)
            if def_tc:
                cfg_for_geom["text_config"] = _merge_partial_root(def_tc, p_geom, ropes)
        try:
            g_hf = cls.derive_geometry(cfg_for_geom)
        except (MegakernelConfigError, ValueError, TypeError, KeyError) as exc:
            return EligibilityResult(
                False, "invalid_geometry", f"{dis}: {exc}; skipping."
            )
        if cls.default_hf_config is not None:
            g_def = cls.derive_geometry(cls.default_hf_config)
            if g_hf != g_def:
                diffs = [
                    f"{f.name}={getattr(g_hf, f.name)!r} "
                    f"(expected {getattr(g_def, f.name)!r})"
                    for f in dataclasses.fields(g_def)
                    if getattr(g_hf, f.name) != getattr(g_def, f.name)
                ]
                msg = (
                    f"{dis}: hf_config geometry differs in "
                    f"{', '.join(diffs)}; skipping."
                )
                return EligibilityResult(False, "geometry_mismatch", msg)
    return EligibilityResult(True)


__all__ = [
    "describe_binding",
    "model_label",
    "norm_zc_and_eps",
    "validate_model_def",
    "validate_model_hf_config",
    "validate_norms",
]

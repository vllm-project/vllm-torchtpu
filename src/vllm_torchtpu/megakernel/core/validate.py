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
from collections.abc import Sequence
from typing import Any

from .decls import REQUIRED, ModuleBinding, _lookup_raw_key, _resolve_field_spec
from .errors import EligibilityResult, MegakernelConfigError
from .geometry import FromHF
from .modules.moe import MoEModule
from .modules.norm import NormModule
from .profile import Profile
from .schedule import (
    Every,
    FromHFLayerTypes,
    ModuleOrBinding,
    ResolvedSchedule,
    Schedule,
    _module_cls_of,
    _module_name_of,
    _resolve_slot_spec,
)
from .spec import Module


def model_label(cls: Any, model_name: str | None = None) -> str:
    """Return the human-readable model label for `cls`."""
    if model_name:
        return model_name
    own = cls.__dict__.get("name") if isinstance(cls, type) else None
    return str(own) if own else getattr(cls, "__name__", str(cls))


def describe_binding(
    m: ModuleOrBinding | None, expected: ModuleOrBinding | None = None
) -> str:
    """Format a `Module` or `ModuleBinding` for error messages."""
    if m is None:
        return "None"
    label = f"{_module_name_of(m)}/{_module_cls_of(m).__name__}"
    m_opts = {k: v for k, v in m.options.items() if v is not None}
    if expected is not None:
        exp_opts = {k: v for k, v in expected.options.items() if v is not None}
        diff = {
            k: (m_opts.get(k), exp_opts.get(k))
            for k in sorted(set(m_opts) | set(exp_opts))
            if m_opts.get(k) != exp_opts.get(k)
        }
        if diff:
            opts_str = ", ".join(
                f"{k}={got!r} (expected {exp!r})" for k, (got, exp) in diff.items()
            )
            return f"{label}(prefix={m.prefix!r}, {opts_str})"
    if m_opts:
        opts_str = ", ".join(f"{k}={v!r}" for k, v in sorted(m_opts.items()))
        return f"{label}(prefix={m.prefix!r}, {opts_str})"
    return f"{label}(prefix={m.prefix!r})"


def norm_zc_and_eps(
    m: ModuleOrBinding,
    merged: dict[str, Any] | None,
    default_eps: float | None = None,
) -> tuple[bool, float | None]:
    """Extract `(zero_centered, eps)` from a norm `Module` or `ModuleBinding`."""
    if not issubclass(_module_cls_of(m), NormModule):
        raise MegakernelConfigError(
            f"Expected NormModule binding, got {describe_binding(m)}"
        )
    zc = bool(m.options.get("zero_centered", False))
    if isinstance(m, ModuleBinding) and "eps" in m.explicit_fields:
        if merged is not None:
            return zc, float(m.resolve(merged).eps)
        if isinstance(m.fields.get("eps"), (int, float)):
            return zc, float(m.fields["eps"])
    elif not isinstance(m, ModuleBinding) and getattr(m, "eps", None) is not None:
        return zc, float(m.eps)
    return zc, default_eps


def validate_norms(
    cls: Any,
    sched: ResolvedSchedule,
    merged: dict[str, Any] | None,
    eps: float | None = None,
    expected_zc: bool | None = None,
    *,
    model_name: str | None = None,
) -> bool:
    """Validate `pre_norm`, `post_norm`, and `final_norm` presence and agreement."""
    mname = model_label(cls, model_name)
    entries: list[tuple[str, ModuleOrBinding]] = []
    for slot_name, seq in (
        ("pre_norm", sched.pre_norm),
        ("post_norm", sched.post_norm),
    ):
        for i, m in enumerate(seq):
            if m is None:
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: layer {i} slot {slot_name!r} is None "
                    f"(kernel requires {slot_name} on every layer)"
                )
            entries.append((f"layer {i} slot {slot_name!r}", m))
    if cls.final_norm is None:
        raise MegakernelConfigError(
            f"ModelDef {mname!r}: final_norm is None (kernel requires final_norm)"
        )
    entries.append(("final_norm", cls.final_norm))
    first_loc, first_m = entries[0]
    first_zc, _ = norm_zc_and_eps(first_m, merged, eps)
    target_zc = first_zc if expected_zc is None else expected_zc
    for loc, m in entries:
        zc_i, eps_i = norm_zc_and_eps(m, merged, eps)
        if zc_i != target_zc:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({describe_binding(m)}) has "
                f"zero_centered={zc_i}, disagreeing with {first_loc} "
                f"zero_centered={target_zc}"
            )
        if eps is not None and eps_i != eps:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({describe_binding(m)}) has "
                f"eps={eps_i}, disagreeing with model eps={eps}"
            )
    return target_zc


def _infer_static_layers(cls: Any, from_hf: FromHF) -> int | None:
    layers_spec = from_hf.fields.get("layers")
    if cls.default_hf_config is not None:
        key = layers_spec[0] if isinstance(layers_spec, tuple) else str(layers_spec)
        raw_l = _lookup_raw_key(cls.default_hf_config, key)
        if isinstance(raw_l, int) and not isinstance(raw_l, bool):
            return raw_l
    if isinstance(layers_spec, int) and not isinstance(layers_spec, bool):
        return layers_spec
    if (
        isinstance(layers_spec, tuple)
        and isinstance(layers_spec[1], int)
        and not isinstance(layers_spec[1], bool)
    ):
        return layers_spec[1]
    if (
        isinstance(cls.schedule, Schedule)
        and isinstance(cls.schedule.mixer, Sequence)
        and not isinstance(cls.schedule.mixer, (str, bytes))
    ):
        return len(cls.schedule.mixer)
    return None


def validate_model_def(
    cls: Any,
    name: str,
    archs: tuple[str, ...],
    model_types: tuple[str, ...],
    base_cls: type,
) -> None:
    """Validate `cls` at `@register_model` decoration time."""
    if not isinstance(name, str) or not name.strip() or name.strip() != name:
        raise MegakernelConfigError(f"register_model: invalid model name {name!r}")
    if not issubclass(cls, base_cls):
        raise MegakernelConfigError(
            f"register_model({name!r}): {cls.__name__} must subclass ModelDef"
        )
    from_hf = cls._from_hf_decl()
    for req_key in ("hidden", "layers", "eps"):
        if req_key not in from_hf.fields:
            raise MegakernelConfigError(
                f"register_model({name!r}): FromHF is missing required key {req_key!r}"
            )
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(
            f"register_model({name!r}): `schedule` must be a Schedule instance, "
            f"got {cls.schedule!r}"
        )
    if cls.pool_layout_override is not None and not callable(cls.pool_layout_override):
        raise MegakernelConfigError(
            f"register_model({name!r}): pool_layout_override must be callable"
        )
    for slot_name, val in (
        ("pre_norm", cls.schedule.pre_norm),
        ("post_norm", cls.schedule.post_norm),
    ):
        if val is None:
            raise MegakernelConfigError(
                f"register_model({name!r}): layer 0 slot {slot_name!r} is None "
                f"(kernel requires {slot_name})"
            )
    if cls.final_norm is None:
        raise MegakernelConfigError(
            f"register_model({name!r}): final_norm is None (kernel requires final_norm)"
        )

    static_l = _infer_static_layers(cls, from_hf)
    if static_l is not None:
        cls.schedule.validate_bounds(static_l)
        if cls.default_hf_config is not None or not isinstance(
            cls.schedule.mixer, FromHFLayerTypes
        ):
            sched = cls.schedule.resolve(cls.default_hf_config or {}, static_l)
            if cls.default_hf_config is None:
                validate_norms(cls, sched, None, eps=None, model_name=name)

    if cls.default_hf_config is not None:
        try:
            g_def = cls.derive_geometry(cls.default_hf_config)
            cls.pool_layout(g_def, R=8)
        except MegakernelConfigError as exc:
            if name not in str(exc):
                raise MegakernelConfigError(f"register_model({name!r}): {exc}") from exc
            raise
        except Exception as exc:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid default_hf_config: {exc}"
            ) from exc

    for alias, prof in (cls.profiles or {}).items():
        if not isinstance(alias, str) or not alias.strip() or alias.strip() != alias:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid profile alias key {alias!r}"
            )
        if not isinstance(prof, Profile):
            raise MegakernelConfigError(
                f"register_model({name!r}): profiles[{alias!r}] must be a Profile, "
                f"got {prof!r}"
            )
        if prof.model is not None and prof.model != name:
            raise MegakernelConfigError(
                f"register_model({name!r}): profile {prof.name!r} targets model "
                f"{prof.model!r}, expected {name!r}"
            )
    for kind, seq in (("hf_architectures", archs), ("hf_model_type", model_types)):
        for entry in seq:
            if not isinstance(entry, str) or not entry.strip():
                raise MegakernelConfigError(
                    f"register_model({name!r}): invalid {kind} entry {entry!r}"
                )


def _resolved_ffn_bindings(
    cls: Any, hf_config: Any, from_hf: FromHF
) -> list[ModuleOrBinding]:
    sched = cls.schedule
    if not isinstance(sched, Schedule) or sched.ffn is None:
        return []
    l_spec = from_hf.fields.get("layers", "num_hidden_layers")
    num_l: int | None = None
    try:
        raw_l = _resolve_field_spec(hf_config, "layers", l_spec)
        if not isinstance(raw_l, bool) and int(raw_l) >= 1:
            num_l = int(raw_l)
    except (ValueError, TypeError):
        pass
    if num_l is None:
        num_l = _infer_static_layers(cls, from_hf)
    if num_l is not None and num_l >= 1:
        try:
            ffn_l = _resolve_slot_spec(sched.ffn, hf_config, num_l, "ffn")
            for lyr, ov in sched.overrides.items():
                if 0 <= lyr < num_l and "ffn" in ov:
                    ffn_l[lyr] = ov["ffn"]
            return [m for m in ffn_l if m is not None]
        except MegakernelConfigError:
            pass
    spec = sched.ffn
    if isinstance(spec, (ModuleBinding, Module)):
        out: list[ModuleOrBinding] = [spec]
    elif isinstance(spec, Every):
        out = [spec.last] if spec.rest is None else [spec.rest, spec.last]
    elif isinstance(spec, FromHFLayerTypes):
        out = list(spec.mapping.values())
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        out = [x for x in spec if isinstance(x, (ModuleBinding, Module))]
    else:
        out = []
    for ov in sched.overrides.values():
        if "ffn" in ov and ov["ffn"] is not None:
            out.append(ov["ffn"])
    return out


def validate_model_hf_config(cls: Any, hf_config: Any) -> EligibilityResult:
    """Validate `hf_config` against `cls` (`ModelDef` schedule + geometry)."""
    if hf_config is None:
        return EligibilityResult(True)
    from_hf = cls._from_hf_decl()
    for ffn in _resolved_ffn_bindings(cls, hf_config, from_hf):
        m_cls = _module_cls_of(ffn)
        if not (
            issubclass(m_cls, MoEModule) and getattr(m_cls, "routed_experts", True)
        ):
            continue
        if isinstance(ffn, ModuleBinding):
            exp_spec = ffn.fields.get("experts", "num_experts")
            exp_key = exp_spec[0] if isinstance(exp_spec, tuple) else str(exp_spec)
            is_exp = "experts" in ffn.explicit_fields
            aliases = getattr(m_cls, "FIELD_ALIASES", {})
            exp_aliases = tuple(a for a, t in aliases.items() if t == "experts")
            try:
                raw_e = _resolve_field_spec(
                    hf_config,
                    "experts",
                    exp_spec,
                    fallback=REQUIRED,
                    aliases=exp_aliases,
                    explicit=is_exp,
                )
                valid_e = (
                    isinstance(raw_e, int) and not isinstance(raw_e, bool) and raw_e > 0
                )
            except (ValueError, TypeError):
                valid_e = False
        else:
            exp_key = "num_experts"
            raw_e = getattr(ffn, "E", getattr(ffn, "experts", 0))
            valid_e = (
                isinstance(raw_e, int) and not isinstance(raw_e, bool) and raw_e > 0
            )
        if not valid_e:
            return EligibilityResult(
                False,
                "non_moe_checkpoint",
                f"Megakernel disabled for {cls.__name__}: {cls.__name__} "
                f"supports MoE checkpoints only "
                f"(missing or non-positive {exp_key!r}); skipping.",
            )
    hid_spec = from_hf.fields.get("hidden", "hidden_size")
    hid_key = hid_spec[0] if isinstance(hid_spec, tuple) else str(hid_spec)
    has_hid = _lookup_raw_key(hf_config, hid_key) not in (None, REQUIRED)
    geom_keys = {
        (v[0] if isinstance(v, tuple) else str(v))
        for v in from_hf.fields.values()
        if isinstance(v, (str, tuple))
    } | set(cls.default_hf_config or ())
    geom_keys -= {"architectures", "model_type", "torch_dtype", "transformers_version"}
    present_geom = {
        k: val
        for k in geom_keys
        if (val := _lookup_raw_key(hf_config, k)) not in (None, REQUIRED)
    }
    if has_hid or present_geom:
        cfg_for_geom = (
            hf_config
            if (has_hid or cls.default_hf_config is None)
            else {**dict(cls.default_hf_config), **present_geom}
        )
        try:
            g_hf = cls.derive_geometry(cfg_for_geom)
        except (MegakernelConfigError, ValueError, KeyError) as exc:
            return EligibilityResult(
                False,
                "invalid_geometry",
                f"Megakernel disabled for {cls.__name__}: {exc}; skipping.",
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
                return EligibilityResult(
                    False,
                    "geometry_mismatch",
                    f"Megakernel disabled for {cls.__name__}: hf_config geometry "
                    f"differs in {', '.join(diffs)}; skipping.",
                )
    return EligibilityResult(True)


__all__ = [
    "describe_binding",
    "model_label",
    "norm_zc_and_eps",
    "validate_model_def",
    "validate_model_hf_config",
    "validate_norms",
]

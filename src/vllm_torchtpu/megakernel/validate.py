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

from collections.abc import Sequence
from typing import Any

from .decls import ModuleBinding, _lookup_raw_key
from .errors import MegakernelConfigError
from .geometry import FromHF
from .modules.norm import NormModule
from .profile import Profile
from .schedule import (
    FromHFLayerTypes,
    ModuleOrBinding,
    ResolvedSchedule,
    Schedule,
    _module_cls_of,
    _module_name_of,
)


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


def _validate_segment_composed(cls: Any, name: str, base_cls: type) -> None:
    """A segment-composed model brings its own geometry, specs and pool layout."""
    for meth in ("derive_geometry", "to_model_spec", "pool_layout"):
        if getattr(cls, meth).__func__ is getattr(base_cls, meth).__func__:
            raise MegakernelConfigError(
                f"register_model({name!r}): a segment-composed model must "
                f"override {meth}"
            )
    if cls.default_hf_config is not None:
        try:
            cls.derive_geometry(cls.default_hf_config)
        except MegakernelConfigError as exc:
            raise MegakernelConfigError(f"register_model({name!r}): {exc}") from exc


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
    if getattr(cls, "segment_composed", False):
        _validate_segment_composed(cls, name, base_cls)
        return
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


__all__ = [
    "describe_binding",
    "model_label",
    "norm_zc_and_eps",
    "validate_model_def",
    "validate_norms",
]

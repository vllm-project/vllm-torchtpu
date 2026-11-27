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
"""`ModelDef` derivation: geometry, `ModelSpec`, schedule resolution, and `shrink`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from . import decls, schedule, validate
from .decls import REQUIRED, _sub_cfg
from .errors import MegakernelConfigError
from .geometry import ModelGeometry
from .modules.gdn import GDNModule
from .modules.gqa import GQAModule
from .modules.moe import MoEModule
from .pool import PoolLayout
from .schedule import FromHFLayerTypes, ModuleOrBinding, ResolvedSchedule, Schedule
from .spec import LayerType, ModelSpec, Module, SwitchSlot


def _pub_vars(obj: Any) -> dict[str, Any]:
    return {k: v for k, v in vars(obj).items() if k[:1] != "_" and v is not None}


def _hf_str_key(v: Any) -> str | None:
    k = v[0] if isinstance(v, tuple) else v
    return k if isinstance(k, str) else None


def _merged_cfg(hf_config: Any, extras: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    for src in (hf_config, _sub_cfg(hf_config, "text_config")):
        if isinstance(src, Mapping):
            base.update(src)
        elif src is not None:
            base.update(_pub_vars(src))
    for k, v in extras.items():
        if decls._lookup_raw_key(hf_config, k) is REQUIRED and v is not None:
            base[k] = v
    return base


def instantiate_spec_module(
    m: ModuleOrBinding, *, pool_layout: PoolLayout | None = None, g: Any = None
) -> Module:
    """Instantiate a `ModelSpec` `Module` from a `ModuleBinding` or return `m`."""
    if isinstance(m, Module):
        if isinstance(m, MoEModule) and g is not None:
            m._check_option_consistency(g)
            if m.geometry != g:
                m_kw = dict(geometry=g, prefix=m.prefix, shared_gate=m.shared_gate)
                return type(m)(
                    **m_kw,
                    moe_tp=m._explicit_moe_tp,
                    e_slots=m._explicit_e_slots,
                    weight_dtypes=m.options.get("weight_dtypes"),
                )
        return m
    opts = {k: v for k, v in m.options.items() if v is not None}
    if issubclass(m.module_cls, GQAModule):
        return m.module_cls(pool_layout=pool_layout, prefix=m.prefix, **opts)
    g_gate = getattr(g, "gdn_gate", None)
    if issubclass(m.module_cls, GDNModule) and (
        (g_gate is not None and g_gate != "silu") or "gate" in opts
    ):
        opts["gate"] = str(opts["gate"] if g_gate is None else g_gate)
    if issubclass(m.module_cls, MoEModule) and g is not None:
        return m.module_cls(geometry=g, prefix=m.prefix, **opts)
    return m.module_cls(prefix=m.prefix, **opts)


def derive_model_geometry(
    cls: Any,
    hf_config: Any = None,
    *,
    moe_tp: bool | None = None,
    e_slots: int | None = None,
) -> ModelGeometry:
    """Derive hybrid GQA/GDN + MoE `ModelGeometry` from `cls` and `hf_config`."""
    if isinstance(hf_config, ModelGeometry):
        rep: dict[str, Any] = {
            k: fn(v)
            for k, v, fn in (("moe_tp", moe_tp, bool), ("e_slots", e_slots, int))
            if v is not None
        }
        g = dataclasses.replace(hf_config, **rep)
        object.__setattr__(g, "_model_def", cls)
        return g

    mname = validate.model_label(cls)
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(f"ModelDef {mname!r}: `schedule` must be Schedule")
    cfg = cls._effective_hf_config(hf_config)
    top = cls._from_hf_decl().resolve(cfg)
    L, H, eps = int(top["layers"]), int(top["hidden"]), float(top["eps"])
    gdn_l2_eps = float(top["gdn_l2_eps"]) if "gdn_l2_eps" in top else 1e-6
    sched = cls.schedule.resolve(cfg, L)
    merged = _merged_cfg(
        cfg,
        {
            "hidden_size": H,
            "rms_norm_eps": eps,
            "rope_theta": top.get("rope_theta"),
            "partial_rotary_factor": top.get("rope_partial"),
            "gdn_gate": top.get("gdn_gate"),
        },
    )
    for root in (cfg, _sub_cfg(cfg, "text_config")):
        for rope_key in ("rope_parameters", "rope_scaling"):
            rope_sub = _sub_cfg(root, rope_key) if root is not None else None
            if rope_sub is not None:
                rtype = _sub_cfg(rope_sub, "rope_type")
                if rtype is None:
                    rtype = _sub_cfg(rope_sub, "type")
                if rtype not in (None, "default"):
                    raise MegakernelConfigError(
                        f"ModelDef {mname!r}: unsupported {rope_key} "
                        f"rope_type={rtype!r}; megakernel requires unscaled "
                        "('default') RoPE"
                    )
    gqa_b = cls._find_binding(GQAModule, sched)
    if gqa_b is None:
        raise MegakernelConfigError(f"ModelDef {mname!r} has no GQAModule in schedule")
    gqa_g = gqa_b.resolve(merged)
    NH, NKV, HD = int(gqa_g.NH), int(gqa_g.NKV), int(gqa_g.HD)
    if "rot" in gqa_b.explicit_fields or "rope_partial" not in top:
        ROT = int(gqa_g.ROT)
    else:
        ROT = int(round(HD * float(top["rope_partial"])))
    theta = float(top["rope_theta"]) if "rope_theta" in top else float(gqa_g.theta)

    gdn_b = cls._find_binding(GDNModule, sched)
    if gdn_b is not None:
        gdn_g = gdn_b.resolve(merged)
        KH, VH, KD, VD = int(gdn_g.KH), int(gdn_g.VH), int(gdn_g.KD), int(gdn_g.VD)
        if "gdn_l2_eps" in gdn_b.explicit_fields or "gdn_l2_eps" not in top:
            gdn_l2_eps = float(gdn_g.gdn_l2_eps)
        gdn_gate = str(gdn_g.gate)
    else:
        KH = VH = KD = VD = 0
        gdn_gate = str(top.get("gdn_gate", "silu"))

    moe_b = cls._find_binding(MoEModule, sched)
    if moe_b is None:
        raise MegakernelConfigError(f"ModelDef {mname!r} has no MoEModule in schedule")
    exp_spec = moe_b.fields.get("experts", "num_experts")
    exp_key = _hf_str_key(exp_spec) or "experts"
    exp_als = tuple(
        a for a, t in moe_b.module_cls.FIELD_ALIASES.items() if t == "experts"
    )
    try:
        raw_e = decls._resolve_field_spec(
            merged,
            "experts",
            exp_spec,
            REQUIRED,
            exp_als,
            explicit="experts" in moe_b.explicit_fields,
        )
    except ValueError:
        raw_e = REQUIRED
    if not (isinstance(raw_e, int) and not isinstance(raw_e, bool) and raw_e > 0):
        raise MegakernelConfigError(
            f"{cls.__name__} supports MoE checkpoints only "
            f"(missing or non-positive {exp_key!r})"
        )
    moe_g = moe_b.resolve(merged)
    eff_tp = bool(moe_g.moe_tp) if moe_tp is None else bool(moe_tp)
    eff_slots = int(moe_g.e_slots) if e_slots is None else int(e_slots)
    zc = validate.validate_norms(cls, sched, merged, eps, model_name=mname)
    g_kw: dict[str, Any] = dict(
        L=L, att_every=sched.att_every, H=H, NH=NH, NKV=NKV, HD=HD, ROT=ROT
    )
    g_kw.update(KH=KH, VH=VH, KD=KD, VD=VD, E=int(moe_g.E), TOPK=int(moe_g.TOPK))
    g_kw.update(I=int(moe_g.I), IS=int(moe_g.IS), moe_tp=eff_tp, e_slots=eff_slots)
    g_kw.update(eps=eps, gdn_l2_eps=gdn_l2_eps, theta=theta, zero_centered_norm=zc)
    g = cls.geometry_cls(**g_kw, gdn_gate=gdn_gate)
    object.__setattr__(g, "_model_def", cls)
    return g


def derive_model_spec(
    cls: Any,
    hf_config_or_geom: Any = None,
    *,
    gdn_module: Module | None = None,
    moe_tp: bool | None = None,
    e_slots: int | None = None,
    R: int | None = 8,
) -> ModelSpec:
    """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s), failing closed on overrides."""
    g = cls.derive_geometry(hf_config_or_geom, moe_tp=moe_tp, e_slots=e_slots)
    sched = cls.resolve_schedule(g)
    mname = validate.model_label(cls)
    ovs = cls.schedule.overrides if isinstance(cls.schedule, Schedule) else {}
    validate.validate_norms(
        cls, sched, None, g.eps, expected_zc=g.zero_centered_norm, model_name=mname
    )
    for slot_name, seq in (
        ("pre_norm", sched.pre_norm),
        ("post_norm", sched.post_norm),
    ):
        canon_n = next(
            (b for i, b in enumerate(seq) if slot_name not in ovs.get(i, {})), seq[0]
        )
        for lyr, b in enumerate(seq):
            if b != canon_n:
                got_d = validate.describe_binding(b, canon_n)
                exp_d = validate.describe_binding(canon_n)
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: layer {lyr} slot {slot_name!r} binding "
                    f"{got_d} cannot be represented in ModelSpec (expected {exp_d})"
                )
    p = sched.att_every
    canon_att = next(
        (sched.mixer[i] for i in range(p - 1, g.L, p) if "mixer" not in ovs.get(i, {})),
        sched.mixer[p - 1],
    )
    canon_rec = next(
        (
            sched.mixer[i]
            for i in range(g.L)
            if (i + 1) % p != 0 and "mixer" not in ovs.get(i, {})
        ),
        sched.mixer[0],
    )
    for lyr, b in enumerate(sched.mixer):
        exp_m = canon_att if (lyr + 1) % p == 0 else canon_rec
        if b != exp_m:
            got_d = validate.describe_binding(b, exp_m)
            exp_d = validate.describe_binding(exp_m)
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'mixer' binding {got_d} "
                f"cannot be represented in ModelSpec (expected {exp_d})"
            )
    canon_ffn = next(
        (
            b
            for i, b in enumerate(sched.ffn)
            if "ffn" not in ovs.get(i, {})
            and issubclass(schedule._module_cls_of(b), MoEModule)
        ),
        None,
    )
    for lyr, b in enumerate(sched.ffn):
        if canon_ffn is None or b != canon_ffn:
            got_d = validate.describe_binding(b, canon_ffn)
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'ffn' binding {got_d} "
                "cannot be represented in ModelSpec"
            )
    play = None if R is None else cls.pool_layout(g, R=R)
    branches: list[Module] = []
    counts: list[int] = []
    for b in sched.mixer[:p]:
        mod = (
            gdn_module
            if gdn_module is not None
            and issubclass(schedule._module_cls_of(b), GDNModule)
            else instantiate_spec_module(b, pool_layout=play, g=g)
        )
        if branches and branches[-1].name == mod.name:
            counts[-1] += 1
        else:
            branches.append(mod)
            counts.append(1)
    mixer_slot = SwitchSlot("mixer", tuple(branches), tuple(counts))
    assert canon_ffn is not None
    ffn_mod = instantiate_spec_module(canon_ffn, g=g)
    own_lt = cls.__dict__.get("layer_type_name") if isinstance(cls, type) else None
    lt_name = (
        own_lt
        or (cls.layer_type_name if cls.__dict__.get("name") else None)
        or f"{mname.replace('_', '')}_layer"
    )
    return ModelSpec(
        name=mname,
        config=g,
        layer_type=LayerType(name=lt_name, slots=(mixer_slot, ffn_mod)),
        cache_backend=None if R is None else cls.pool_backend(g, R=R),
    )


def _synth_layer_types(cls: Any, g: ModelGeometry) -> tuple[str, list[str]] | None:
    if not isinstance(cls.schedule, Schedule) or not isinstance(
        cls.schedule.mixer, FromHFLayerTypes
    ):
        return None
    if g.L <= 0 or g.att_every <= 0 or g.L % g.att_every != 0:
        raise MegakernelConfigError(f"Invalid layers L={g.L}, att_every={g.att_every}")
    m = cls.schedule.mixer.mapping
    att_k = next(
        (k for k, b in m.items() if issubclass(schedule._module_cls_of(b), GQAModule)),
        None,
    )
    rec_k = next(
        (k for k, b in m.items() if issubclass(schedule._module_cls_of(b), GDNModule)),
        None,
    )
    if att_k is None or (g.att_every > 1 and rec_k is None):
        raise MegakernelConfigError(
            f"FromHFLayerTypes mapping {dict(m)!r} cannot build att_every={g.att_every}"
        )
    seq = ([rec_k or att_k] * (g.att_every - 1) + [att_k]) * (g.L // g.att_every)
    return cls.schedule.mixer.field, seq


def resolve_model_schedule(cls: Any, hf_config_or_geom: Any = None) -> ResolvedSchedule:
    """Resolve per-layer schedule for `hf_config_or_geom`."""
    mname = validate.model_label(cls)
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(f"ModelDef {mname!r}: `schedule` must be Schedule")
    if isinstance(hf_config_or_geom, ModelGeometry):
        g = hf_config_or_geom
        synth = _synth_layer_types(cls, g)
        return cls.schedule.resolve({synth[0]: synth[1]} if synth else {}, g.L)
    cfg = cls._effective_hf_config(hf_config_or_geom)
    return cls.schedule.resolve(cfg, int(cls._from_hf_decl().resolve(cfg)["layers"]))


def hf_config_for_geometry(
    cls: Any, g: ModelGeometry, hf_config: Any = None, *, max_vocab: int | None = None
) -> dict[str, Any]:
    """Return an HF config dict updated to match `g` via `FromHF` and bindings."""
    out = _merged_cfg(cls._effective_hf_config(hf_config), {})
    updates: dict[str, Any] = {}
    for k, v in cls._from_hf_decl().fields.items():
        if (hf_k := _hf_str_key(v)) is not None:
            if k == "layers":
                updates[hf_k] = int(g.L)
            elif k == "hidden":
                updates[hf_k] = int(g.H)
            elif k == "gdn_gate":
                updates[hf_k] = str(g.gdn_gate)
            elif k == "vocab" and max_vocab is not None:
                updates[hf_k] = min(int(out.get(hf_k, max_vocab)), int(max_vocab))
    sched = cls.resolve_schedule(g)
    gqa_m = {"heads": int(g.NH), "kv_heads": int(g.NKV), "head_dim": int(g.HD)}
    gdn_m: dict[str, Any] = {"k_heads": int(g.KH), "v_heads": int(g.VH)}
    gdn_m.update(k_dim=int(g.KD), v_dim=int(g.VD))
    gdn_b = cls._find_binding(GDNModule, sched)
    if gdn_b and (
        "gate" in gdn_b.explicit_fields
        or gdn_b.options.get("gate")
        or g.gdn_gate != "silu"
    ):
        gdn_m["gate"] = str(g.gdn_gate)
    moe_m = {"experts": g.E, "top_k": g.TOPK, "inter": g.I, "shared_inter": g.IS}
    for b, mapping in (
        (cls._find_binding(GQAModule, sched), gqa_m),
        (gdn_b, gdn_m),
        (cls._find_binding(MoEModule, sched), {k: int(v) for k, v in moe_m.items()}),
    ):
        if b is not None:
            for canon, val in mapping.items():
                if (hf_k := _hf_str_key(b.fields.get(canon))) is not None:
                    updates[hf_k] = val
    if (synth := _synth_layer_types(cls, g)) is not None:
        updates[synth[0]] = synth[1]
    out.update(updates)
    tc = out.get("text_config")
    if isinstance(tc, Mapping):
        out["text_config"] = {**tc, **updates}
    elif tc is not None:
        out["text_config"] = {**_pub_vars(tc), **updates}
    return out


def shrink_model_hf_config(
    cls: Any, hf_config: Any = None, *, periods: int = 2, R: int = 8, ex: Any = None
) -> dict[str, Any]:
    """Return a shrunk HF config dict (`periods` cycles, legal heads/experts)."""
    cfg = cls._effective_hf_config(hf_config)
    g_tiny = cls.derive_geometry(cfg).shrink(periods=periods, R=R, ex=ex)
    return hf_config_for_geometry(cls, g_tiny, cfg)


__all__ = [
    "derive_model_geometry",
    "derive_model_spec",
    "hf_config_for_geometry",
    "instantiate_spec_module",
    "resolve_model_schedule",
    "shrink_model_hf_config",
]

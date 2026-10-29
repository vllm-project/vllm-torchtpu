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

from .decls import REQUIRED, _lookup_raw_key, _sub_cfg
from .errors import MegakernelConfigError
from .geometry import ModelGeometry
from .modules.gdn import GDNModule
from .modules.gqa import GQAModule
from .modules.moe import MoEModule
from .modules.moe.params import resolve_expert_quant
from .pool import PoolLayout
from .schedule import (
    FromHFLayerTypes,
    ModuleOrBinding,
    ResolvedSchedule,
    Schedule,
    _module_cls_of,
)
from .spec import LayerType, ModelSpec, Module, SwitchSlot
from .validate import describe_binding, model_label, validate_norms


def _merged_cfg(hf_config: Any, extras: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    tc = _sub_cfg(hf_config, "text_config")
    for src in (hf_config, tc):
        if isinstance(src, Mapping):
            base.update(src)
        elif src is not None:
            base.update(
                {
                    k: v
                    for k, v in vars(src).items()
                    if not k.startswith("_") and v is not None
                }
            )
    for k, v in extras.items():
        if _lookup_raw_key(hf_config, k) is REQUIRED and v is not None:
            base[k] = v
    return base


def instantiate_spec_module(
    m: ModuleOrBinding, *, pool_layout: PoolLayout | None = None
) -> Module:
    """Instantiate a `ModelSpec` `Module` from a `ModuleBinding` or return `m`."""
    if isinstance(m, Module):
        return m
    opts = {k: v for k, v in m.options.items() if v is not None}
    if issubclass(m.module_cls, GQAModule):
        return m.module_cls(pool_layout=pool_layout, prefix=m.prefix, **opts)
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
        rep: dict[str, Any] = {}
        if moe_tp is not None:
            rep["moe_tp"] = bool(moe_tp)
        if e_slots is not None:
            rep["e_slots"] = int(e_slots)
        g = dataclasses.replace(hf_config, **rep)
        object.__setattr__(g, "_model_def", cls)
        return g

    mname = model_label(cls)
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(f"ModelDef {mname!r}: `schedule` must be Schedule")
    cfg = cls._effective_hf_config(hf_config)
    top = cls._from_hf_decl().resolve(cfg)
    L = int(top["layers"])
    H = int(top["hidden"])
    eps = float(top["eps"])
    gdn_l2_eps = float(top["gdn_l2_eps"]) if "gdn_l2_eps" in top else 1e-6
    sched = cls.schedule.resolve(cfg, L)
    merged = _merged_cfg(
        cfg,
        {
            "hidden_size": H,
            "rms_norm_eps": eps,
            "rope_theta": top.get("rope_theta"),
            "partial_rotary_factor": top.get("rope_partial"),
        },
    )
    for rope_key in ("rope_parameters", "rope_scaling"):
        rope_sub = _sub_cfg(cfg, rope_key) or merged.get(rope_key)
        if rope_sub is not None:
            rtype = (
                rope_sub.get("rope_type", rope_sub.get("type"))
                if isinstance(rope_sub, Mapping)
                else getattr(rope_sub, "rope_type", getattr(rope_sub, "type", None))
            )
            if rtype not in (None, "default"):
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: unsupported {rope_key} rope_type={rtype!r}; "
                    "megakernel requires unscaled ('default') RoPE"
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
        KH, VH = int(gdn_g.KH), int(gdn_g.VH)
        KD, VD = int(gdn_g.KD), int(gdn_g.VD)
        if "gdn_l2_eps" in gdn_b.explicit_fields or "gdn_l2_eps" not in top:
            gdn_l2_eps = float(gdn_g.gdn_l2_eps)
    else:
        KH = VH = KD = VD = 0

    moe_b = cls._find_binding(MoEModule, sched)
    if moe_b is None:
        raise MegakernelConfigError(f"ModelDef {mname!r} has no MoEModule in schedule")
    exp_spec = moe_b.fields.get("experts", "num_experts")
    exp_key = exp_spec[0] if isinstance(exp_spec, tuple) else exp_spec
    if isinstance(exp_key, str):
        raw_e = _lookup_raw_key(merged, exp_key)
        if (
            raw_e is REQUIRED
            or raw_e is None
            or (isinstance(raw_e, (int, float)) and raw_e <= 0)
        ):
            raise MegakernelConfigError(
                f"{cls.__name__} supports MoE checkpoints only "
                f"(missing or non-positive {exp_key!r})"
            )
    moe_g = moe_b.resolve(merged)
    eff_tp = bool(moe_g.moe_tp) if moe_tp is None else bool(moe_tp)
    eff_slots = int(moe_g.e_slots) if e_slots is None else int(e_slots)
    eff_quant = resolve_expert_quant(
        hf_config, merged, moe_b.options.get("expert_quant"), moe_g.expert_quant
    )
    zc = validate_norms(cls, sched, merged, eps, model_name=mname)
    g = cls.geometry_cls(
        L=L,
        att_every=sched.att_every,
        H=H,
        NH=NH,
        NKV=NKV,
        HD=HD,
        ROT=ROT,
        KH=KH,
        VH=VH,
        KD=KD,
        VD=VD,
        E=int(moe_g.E),
        TOPK=int(moe_g.TOPK),
        I=int(moe_g.I),
        IS=int(moe_g.IS),
        moe_tp=eff_tp,
        e_slots=eff_slots,
        expert_quant=eff_quant,
        eps=eps,
        gdn_l2_eps=gdn_l2_eps,
        theta=theta,
        zero_centered_norm=zc,
    )
    object.__setattr__(g, "_model_def", cls)
    return g


def derive_model_spec(
    cls: Any,
    hf_config_or_geom: Any = None,
    *,
    gdn_module: Module | None = None,
    moe_tp: bool | None = None,
    e_slots: int | None = None,
    R: int = 8,
) -> ModelSpec:
    """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s), failing closed on overrides."""
    g = cls.derive_geometry(hf_config_or_geom, moe_tp=moe_tp, e_slots=e_slots)
    sched = cls.resolve_schedule(g)
    mname = model_label(cls)
    ovs = cls.schedule.overrides if isinstance(cls.schedule, Schedule) else {}
    validate_norms(
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
                got_desc = describe_binding(b, canon_n)
                exp_desc = describe_binding(canon_n)
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: layer {lyr} slot {slot_name!r} binding "
                    f"{got_desc} cannot be represented in ModelSpec "
                    f"(expected {exp_desc})"
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
            got_desc = describe_binding(b, exp_m)
            exp_desc = describe_binding(exp_m)
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'mixer' binding {got_desc} "
                f"cannot be represented in ModelSpec (expected {exp_desc})"
            )
    canon_ffn = next(
        (
            b
            for i, b in enumerate(sched.ffn)
            if "ffn" not in ovs.get(i, {})
            and issubclass(_module_cls_of(b), MoEModule)
            and getattr(_module_cls_of(b), "routed_experts", True)
        ),
        None,
    )
    for lyr, b in enumerate(sched.ffn):
        if canon_ffn is None or b != canon_ffn:
            got_desc = describe_binding(b, canon_ffn)
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'ffn' binding {got_desc} "
                "cannot be represented in ModelSpec"
            )
    play = cls.pool_layout(g, R=R)
    branches: list[Module] = []
    counts: list[int] = []
    for b in sched.mixer[:p]:
        mod = (
            gdn_module
            if gdn_module is not None and issubclass(_module_cls_of(b), GDNModule)
            else instantiate_spec_module(b, pool_layout=play)
        )
        if branches and branches[-1].name == mod.name:
            counts[-1] += 1
        else:
            branches.append(mod)
            counts.append(1)
    mixer_slot = SwitchSlot("mixer", tuple(branches), tuple(counts))
    ffn_mod = instantiate_spec_module(canon_ffn)
    own_lt = cls.__dict__.get("layer_type_name") if isinstance(cls, type) else None
    lt_name = own_lt or (
        cls.layer_type_name
        if cls.__dict__.get("name")
        else f"{mname.replace('_', '')}_layer"
    )
    return ModelSpec(
        name=mname,
        config=g,
        layer_type=LayerType(name=lt_name, slots=(mixer_slot, ffn_mod)),
        cache_backend=cls.pool_backend(g, R=R),
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
        (k for k, b in m.items() if issubclass(_module_cls_of(b), GQAModule)), None
    )
    rec_k = next(
        (k for k, b in m.items() if issubclass(_module_cls_of(b), GDNModule)), None
    )
    if att_k is None or (g.att_every > 1 and rec_k is None):
        raise MegakernelConfigError(
            f"FromHFLayerTypes mapping {dict(m)!r} cannot build att_every={g.att_every}"
        )
    seq = ([rec_k or att_k] * (g.att_every - 1) + [att_k]) * (g.L // g.att_every)
    return cls.schedule.mixer.field, seq


def resolve_model_schedule(cls: Any, hf_config_or_geom: Any = None) -> ResolvedSchedule:
    """Resolve per-layer schedule for `hf_config_or_geom`."""
    mname = model_label(cls)
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(
            f"ModelDef {mname!r}: `schedule` must be a Schedule"
        )
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
        hf_k = v[0] if isinstance(v, tuple) else v
        if isinstance(hf_k, str):
            if k == "layers":
                updates[hf_k] = int(g.L)
            elif k == "hidden":
                updates[hf_k] = int(g.H)
            elif k == "vocab" and max_vocab is not None:
                v_sz = int(getattr(g, "V", out.get(hf_k, max_vocab)))
                updates[hf_k] = min(v_sz, int(max_vocab))
    sched = cls.resolve_schedule(g)
    gqa_m = {"heads": int(g.NH), "kv_heads": int(g.NKV), "head_dim": int(g.HD)}
    gdn_m = {
        "k_heads": int(g.KH),
        "v_heads": int(g.VH),
        "k_dim": int(g.KD),
        "v_dim": int(g.VD),
    }
    moe_m = {
        "experts": int(g.E),
        "top_k": int(g.TOPK),
        "inter": int(g.I),
        "shared_inter": int(g.IS),
    }
    for b, mapping in (
        (cls._find_binding(GQAModule, sched), gqa_m),
        (cls._find_binding(GDNModule, sched), gdn_m),
        (cls._find_binding(MoEModule, sched), moe_m),
    ):
        if b is not None:
            for canon, val in mapping.items():
                spec_v = b.fields.get(canon)
                hf_k = spec_v[0] if isinstance(spec_v, tuple) else spec_v
                if isinstance(hf_k, str):
                    updates[hf_k] = val
    if (synth := _synth_layer_types(cls, g)) is not None:
        updates[synth[0]] = synth[1]
    out.update(updates)
    if isinstance(out.get("text_config"), Mapping):
        out["text_config"] = {**out["text_config"], **updates}
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

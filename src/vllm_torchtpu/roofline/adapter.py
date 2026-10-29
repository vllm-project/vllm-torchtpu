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
"""Megakernel Plan and Profile adapter for the decode roofline library."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

from .chips import CHIPS, ChipSpec
from .geometry import ModelGeometry
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    _attn_layer_collectives,
    _embed_collectives,
    _lm_head_collectives,
    _moe_layer_collectives,
    _split_matmul_flops,
    ep_busiest_experts,
    experts_touched,
)
from .step import StepFloor, _combine_modules, step_floor


@dataclass(frozen=True)
class PlanReconciliation:
    """Per-module byte and FLOP delta between Plan floor and standalone step_floor."""

    module: str
    plan_hbm_bytes: float
    standalone_hbm_bytes: float
    delta_hbm_bytes: float
    plan_flops: float
    standalone_flops: float
    delta_flops: float


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return optional attribute when present on ``obj``."""
    if obj is None or name not in dir(obj):
        return default
    return operator.attrgetter(name)(obj)


def _dtype_bytes(dt: Any) -> float:
    """Return element size in bytes from a JAX/NumPy dtype, class, or string."""
    if type(dt) in (int, float):
        return float(dt)
    name = str(_attr(dt, "__name__", dt)).lower()
    if "mxfp4" in name:
        return 0.53125
    if "nvfp4" in name:
        return 0.5625
    if "float4" in name or "fp4" in name or "int4" in name:
        return 0.5
    if "float8" in name or "fp8" in name or "int8" in name or "uint8" in name:
        return 1.0
    if "bfloat16" in name or "bf16" in name or "float16" in name or "fp16" in name:
        return 2.0
    if "float32" in name or "fp32" in name or "int32" in name:
        return 4.0
    if "itemsize" in dir(dt):
        return float(dt.itemsize)
    if "dtype" in dir(dt) and "itemsize" in dir(dt.dtype):
        return float(dt.dtype.itemsize)
    raise ValueError(f"Unrecognized dtype in Megakernel Plan: {dt!r}")


def _resolve_cache_decl(mod: Any, model_spec: Any) -> Any:
    fn = _attr(mod, "cache_decl", None)
    if fn is None:
        return None
    if not callable(fn):
        return fn
    cfg = _attr(model_spec, "config", None)
    try:
        return fn(cfg)
    except TypeError:
        return fn()


def _derive_parallel(
    p: Any,
    ex: Any,
    geom: ModelGeometry,
    r_val: int,
    weight_decls: tuple[Any, ...],
) -> Parallel:
    tiles = _attr(ex, "tiles", None)
    rsag = bool(_attr(tiles, "rsag", False))
    hier = int(_attr(tiles, "hier", 0))
    if rsag:
        ar_algo = "two_shot"
    elif hier == 0:
        ar_algo = "one_shot"
    else:
        raise NotImplementedError(
            f"Tiles.hier={hier}: only the one-shot (hier=0) and rsag all-reduce"
            " wire models are priced"
        )

    mp = _attr(p, "module_params", None) or {}
    gqa_p = mp.get("gqa") if type(mp) is dict else None
    is_split = bool(_attr(gqa_p, "SPLIT", False))
    g_val = max(1, r_val // geom.num_kv_heads) if geom.num_kv_heads > 0 else 1
    kv_split = g_val if is_split else 1
    kv_merge = "exchange" if is_split else "ag_rs"

    e13 = next(
        (w for w in weight_decls if w.name == "e13" and _attr(w, "lead_dims", ())),
        None,
    )
    if (
        e13 is not None
        and r_val > 1
        and geom.num_experts > 0
        and int(e13.lead_dims[0]) == geom.num_experts // r_val
    ):
        moe_mode = "ep"
    else:
        moe_mode = "tp_slice"

    return Parallel(
        tp=r_val,
        kv_split=kv_split,
        kv_merge=kv_merge,
        moe=moe_mode,
        ar_algo=ar_algo,
    )


def floor_from_plan(
    p: Any,
    ex: Any = None,
    *,
    geom: ModelGeometry,
    par: Parallel | None = None,
    B: int | None = None,
    S: int | None = None,
    T: int | None = None,
    state_writes: int | None = None,
    chip: ChipSpec = CHIPS["tpu7x"],
    dt: Dtypes,
    collective_latency_s: float = 0.0,
    include_plan_extras: bool = True,
) -> StepFloor:
    """Derive a StepFloor directly from a megakernel.planner.Plan and ExecSpec."""
    t_val = int(T if T is not None else _attr(ex, "T", 1))
    r_val = int(par.tp if par is not None else _attr(ex, "R", 1))
    ex_b = _attr(ex, "B", None)
    if B is not None:
        b_val = int(B)
    elif ex_b is not None:
        b_val = max(1, int(ex_b) // max(1, t_val))
    else:
        b_val = 1
    s_val = int(S if S is not None else _attr(ex, "ctx", 0))
    sw_val = int(state_writes if state_writes is not None else 1)
    m_rows = b_val * t_val

    raw_weights = _attr(p, "weight_decls", ())
    if type(raw_weights) is dict:
        weight_decls = tuple(raw_weights.values())
    else:
        weight_decls = tuple(raw_weights)

    eff_par = (
        par if par is not None else _derive_parallel(p, ex, geom, r_val, weight_decls)
    )

    module_counts: dict[str, int] = dict(_attr(p, "module_layer_counts", {}))
    model_spec = _attr(p, "model_spec", None)
    all_mods_fn = _attr(model_spec, "all_modules", None)
    raw_mods = all_mods_fn() if callable(all_mods_fn) else ()
    mod_map: dict[str, Any] = {m.name: m for m in raw_mods}

    hidden = geom.hidden
    for wd in weight_decls:
        if wd.name in ("att_in", "lin_in", "router"):
            hidden = int(wd.K)
            break

    weight_owners: dict[str, str] = dict(_attr(p, "weight_owners", {}))
    by_owner: dict[str, list[Any]] = {}
    for wd in weight_decls:
        owner = _attr(wd, "owner", None) or weight_owners.get(wd.name, "unknown")
        by_owner.setdefault(owner, []).append(wd)

    modules: list[ModuleCost] = [
        ModuleCost(
            name="embed",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=_embed_collectives(m_rows, hidden, eff_par),
        )
    ]

    q_per_rank = max(1, geom.num_heads // r_val)

    # 1. GQA module
    n_gqa = module_counts.get("gqa", 0)
    if n_gqa > 0:
        gqa_mod = mod_map.get("gqa")
        cd = _resolve_cache_decl(gqa_mod, model_spec) if gqa_mod is not None else None
        if cd is None or _attr(cd, "page_dtype", None) is None:
            raise ValueError(
                "Module 'gqa' requires a cache_decl() with non-None page_dtype"
            )
        gqa_hbm = 0.0
        gqa_fp8 = 0.0
        gqa_bf16 = 0.0
        for wd in by_owner.get("gqa", ()):
            db = _dtype_bytes(wd.dtype)
            elems = float(n_gqa) * float(wd.K) * float(wd.N)
            gqa_hbm += elems * db
            fp8_f, bf16_f = _split_matmul_flops(m_rows, elems, db, dt.act)
            gqa_fp8 += fp8_f
            gqa_bf16 += bf16_f

        kvh = int(cd.kvh)
        hd = int(cd.hd)
        kv_b = _dtype_bytes(cd.page_dtype)
        kv_bytes = float(b_val * kvh * hd * 2 * s_val * n_gqa) * kv_b / eff_par.kv_split
        gqa_hbm += kv_bytes
        gqa_bf16 += float(n_gqa) * (4.0 * m_rows * q_per_rank * s_val * hd)

        gqa_colls = (
            _attn_layer_collectives(
                b_val,
                t_val,
                hidden,
                q_per_rank,
                hd,
                eff_par,
                is_full_attention=True,
            )
            * n_gqa
            if _attr(gqa_mod, "reduction", "allreduce") == "allreduce"
            else ()
        )
        modules.append(
            ModuleCost(
                name="gqa",
                hbm_bytes=gqa_hbm,
                fp8_flops=gqa_fp8,
                bf16_flops=gqa_bf16,
                collectives=gqa_colls,
            )
        )

    # 2. GDN module
    n_gdn = module_counts.get("gdn", 0)
    if n_gdn > 0:
        gdn_mod = mod_map.get("gdn")
        cd = _resolve_cache_decl(gdn_mod, model_spec) if gdn_mod is not None else None
        if (
            cd is None
            or _attr(cd, "rec_dtype", None) is None
            or _attr(cd, "conv_dtype", None) is None
        ):
            raise ValueError(
                "Module 'gdn' requires a cache_decl() with "
                "non-None rec_dtype and conv_dtype"
            )
        gdn_hbm = 0.0
        gdn_fp8 = 0.0
        gdn_bf16 = 0.0
        kh = int(cd.kh)
        vh = int(cd.vh)
        if r_val > 1 and vh == geom.linear_v_heads:
            vh = max(1, vh // r_val)
            kh = max(1, kh // r_val)
        kd = int(cd.KD)
        vd = int(cd.VD)
        for wd in by_owner.get("gdn", ()):
            db = _dtype_bytes(wd.dtype)
            n_cols = float(wd.N)
            if not include_plan_extras and wd.name == "lin_in":
                ba_pad = max(2 * vh, 128)
                n_cols -= float(ba_pad)
            elems = float(n_gdn) * float(wd.K) * n_cols
            gdn_hbm += elems * db
            fp8_f, bf16_f = _split_matmul_flops(m_rows, elems, db, dt.act)
            gdn_fp8 += fp8_f
            gdn_bf16 += bf16_f

        state_b = _dtype_bytes(cd.rec_dtype)
        conv_b = _dtype_bytes(cd.conv_dtype)
        conv_taps = int(cd.conv_taps)
        conv_k = conv_taps + 1
        qkv_r = 2 * kh * kd + vh * vd
        if not include_plan_extras:
            aux_elems = float(n_gdn * (hidden * 2 * vh + conv_k * qkv_r))
            gdn_hbm += aux_elems * dt.w_aux
            aux_fp8, aux_bf16 = _split_matmul_flops(m_rows, aux_elems, dt.w_aux, dt.act)
            gdn_fp8 += aux_fp8
            gdn_bf16 += aux_bf16
        gdn_bf16 += float(m_rows * 4 * vh * kd * vd * n_gdn)
        state_elems = float(vh * kd * vd)
        conv_state_elems = float(qkv_r * conv_taps)
        gdn_hbm += (
            n_gdn
            * float(b_val)
            * (1.0 + sw_val)
            * (state_elems * state_b + conv_state_elems * conv_b)
        )

        gdn_colls = (
            _attn_layer_collectives(
                b_val,
                t_val,
                hidden,
                q_per_rank,
                geom.head_dim,
                eff_par,
                is_full_attention=False,
            )
            * n_gdn
            if _attr(gdn_mod, "reduction", "allreduce") == "allreduce"
            else ()
        )
        modules.append(
            ModuleCost(
                name="gdn",
                hbm_bytes=gdn_hbm,
                fp8_flops=gdn_fp8,
                bf16_flops=gdn_bf16,
                collectives=gdn_colls,
            )
        )

    # 3. MoE module
    n_moe = module_counts.get("moe", 0)
    if n_moe > 0:
        moe_mod = mod_map.get("moe")
        moe_hbm = 0.0
        moe_hbm_min = 0.0
        moe_fp8 = 0.0
        moe_bf16 = 0.0
        e_global = geom.num_experts
        top_k = geom.top_k
        e_d_global = experts_touched(e_global, top_k, m_rows)
        for wd in by_owner.get("moe", ()):
            if wd.name == "es" and not include_plan_extras:
                continue
            db = _dtype_bytes(wd.dtype)
            elems = float(wd.K) * float(wd.N)
            lead = _attr(wd, "lead_dims", ())
            if not lead:
                w_bytes = float(n_moe) * elems * db
                moe_hbm += w_bytes
                moe_hbm_min += w_bytes
                rows_eff = (
                    math.ceil(m_rows / r_val)
                    if eff_par.residual == "sp" and wd.name in ("sh13", "sh2")
                    else m_rows
                )
                fp8_f, bf16_f = _split_matmul_flops(
                    rows_eff, float(n_moe) * elems, db, dt.act
                )
                moe_fp8 += fp8_f
                moe_bf16 += bf16_f
            else:
                e_rank = int(lead[0])
                scale = float(e_rank) / float(max(1, e_global))
                if e_rank == e_global:
                    e_d_rank = e_d_global
                    e_min_rank = float(min(e_global, top_k)) if m_rows > 0 else 0.0
                elif r_val > 1 and e_rank == e_global // r_val:
                    e_d_rank = ep_busiest_experts(e_global, top_k, m_rows, r_val)
                    e_min_rank = float(math.ceil(top_k / r_val)) if m_rows > 0 else 0.0
                else:
                    # EPLB slot count: load balancing targets the mean active experts.
                    e_d_rank = e_d_global * scale
                    e_min_rank = (
                        float(min(e_global, top_k)) * scale if m_rows > 0 else 0.0
                    )
                moe_hbm += float(n_moe) * e_d_rank * elems * db
                moe_hbm_min += float(n_moe) * e_min_rank * elems * db
                if wd.name in ("e13", "e2"):
                    act_elems = float(n_moe * top_k) * scale * elems
                    fp8_f, bf16_f = _split_matmul_flops(m_rows, act_elems, db, dt.act)
                    moe_fp8 += fp8_f
                    moe_bf16 += bf16_f

        moe_colls = (
            _moe_layer_collectives(m_rows, hidden, e_global, top_k, eff_par) * n_moe
            if moe_mod is None
            or _attr(moe_mod, "reduction", "allreduce") == "allreduce"
            else ()
        )
        modules.append(
            ModuleCost(
                name="moe",
                hbm_bytes=moe_hbm,
                fp8_flops=moe_fp8,
                bf16_flops=moe_bf16,
                collectives=moe_colls,
                hbm_bytes_min=moe_hbm_min,
            )
        )

    # 4. Smalls per-layer F32 table (when include_plan_extras=True)
    if include_plan_extras:
        # One F32 row of smalls_width per layer, as planner.weight_shapes()["smalls"].
        modules.append(
            ModuleCost(
                name="smalls",
                hbm_bytes=float(p.model_spec.config.L * p.smalls_width * 4),
                fp8_flops=0.0,
                bf16_flops=0.0,
            )
        )

    # 5. LM head from ModelGeometry.vocab and Dtypes.lm_head
    if geom.vocab > 0 and hidden > 0:
        lm_elems = float(geom.vocab * hidden // r_val)
        lm_fp8, lm_bf16 = _split_matmul_flops(m_rows, lm_elems, dt.lm_head, dt.act)
        modules.append(
            ModuleCost(
                name="lm_head",
                hbm_bytes=lm_elems * dt.lm_head,
                fp8_flops=lm_fp8,
                bf16_flops=lm_bf16,
                collectives=_lm_head_collectives(m_rows, hidden, geom.vocab, eff_par),
            )
        )

    return _combine_modules(
        tuple(modules), chip, par=eff_par, collective_latency_s=collective_latency_s
    )


def floor_for_profile(
    model_spec: Any,
    profile: Any,
    *,
    geom: ModelGeometry,
    par: Parallel | None = None,
    B: int,
    S: int,
    T: int = 1,
    R: int,
    granularity: int = 2048,
    state_writes: int | None = None,
    chip: ChipSpec = CHIPS["tpu7x"],
    dt: Dtypes,
    collective_latency_s: float = 0.0,
    include_plan_extras: bool = True,
) -> StepFloor:
    """Build ExecSpec + Plan from Megakernel ModelSpec and Profile, then StepFloor."""
    import importlib

    exec_spec_mod = importlib.import_module("vllm_torchtpu.megakernel.exec_spec")
    ExecSpec = exec_spec_mod.ExecSpec
    planner_mod = importlib.import_module("vllm_torchtpu.megakernel.planner")
    mk_plan = planner_mod.plan

    gran = max(1, int(granularity))
    plan_ctx = max(gran, ((max(1, int(S)) + gran - 1) // gran) * gran)
    base = ExecSpec(R=R, ctx=plan_ctx)
    ex = profile.exec_for(base, B=int(B) * int(T), T=int(T))
    p = mk_plan(model_spec, ex)
    return floor_from_plan(
        p,
        ex,
        geom=geom,
        par=par,
        B=B,
        S=S,
        T=T,
        state_writes=state_writes,
        chip=chip,
        dt=dt,
        collective_latency_s=collective_latency_s,
        include_plan_extras=include_plan_extras,
    )


def reconcile_plan_vs_standalone(
    p: Any,
    ex: Any,
    geom: ModelGeometry,
    par: Parallel | None = None,
    shape: StepShape | None = None,
    dt: Dtypes | None = None,
    chip: ChipSpec = CHIPS["tpu7x"],
    include_plan_extras: bool = False,
) -> tuple[PlanReconciliation, ...]:
    """Compare floor_from_plan against standalone step_floor module-by-module."""
    if shape is None or dt is None:
        raise ValueError(
            "reconcile_plan_vs_standalone requires explicit 'shape' and 'dt'"
        )
    raw_weights = _attr(p, "weight_decls", ())
    weight_decls = (
        tuple(raw_weights.values())
        if type(raw_weights) is dict
        else tuple(raw_weights)
    )
    r_val = int(par.tp if par is not None else _attr(ex, "R", 1))
    eff_par = (
        par if par is not None else _derive_parallel(p, ex, geom, r_val, weight_decls)
    )
    plan_sf = floor_from_plan(
        p,
        ex,
        geom=geom,
        par=eff_par,
        B=shape.batch,
        S=shape.context,
        T=shape.rows_per_seq,
        state_writes=shape.state_writes,
        chip=chip,
        dt=dt,
        include_plan_extras=include_plan_extras,
    )
    std_sf = step_floor(geom, chip, eff_par, shape, dt)
    plan_map = {m.name: m for m in plan_sf.modules if m.name != "embed"}
    std_map = {m.name: m for m in std_sf.modules if m.name != "embed"}
    names = list(dict.fromkeys([*std_map.keys(), *plan_map.keys()]))
    rows: list[PlanReconciliation] = []
    for name in names:
        pm = plan_map.get(name)
        sm = std_map.get(name)
        pb = pm.hbm_bytes if pm is not None else 0.0
        sb = sm.hbm_bytes if sm is not None else 0.0
        pf = (pm.fp8_flops + pm.bf16_flops) if pm is not None else 0.0
        sf = (sm.fp8_flops + sm.bf16_flops) if sm is not None else 0.0
        rows.append(
            PlanReconciliation(
                module=name,
                plan_hbm_bytes=pb,
                standalone_hbm_bytes=sb,
                delta_hbm_bytes=pb - sb,
                plan_flops=pf,
                standalone_flops=sf,
                delta_flops=pf - sf,
            )
        )
    return tuple(rows)

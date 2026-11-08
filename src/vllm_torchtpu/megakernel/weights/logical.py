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
"""All-ranks declarative HF sharding (`load_logical_weights`) and packing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from ..decls import HFParamDecl, ModuleBinding, ResolvedModuleGeometry
from ..exec_spec import ExecSpec
from ..model_def import ModelDef
from ..module import Module
from ..modules.norm import NormModule
from ..modules.transforms import _validate_decl_and_array, apply_hf_transforms
from ..ops import I32
from ..spec import ModelSpec
from .eplb import default_eplb_cores, encode_eplb_ecopy
from .spec_inputs import _coerce_canonical_hf_entry, coerce_canonical_hf_entry
from .spec_pack import pack_spec_logical_weights, switch_branch_for_layer


def _resolve_norm_mod(
    binding_or_mod: Any, g: Any, *, default_small_field: str | None = None
) -> Module | None:
    if binding_or_mod is None or isinstance(binding_or_mod, Module):
        if (
            isinstance(binding_or_mod, NormModule)
            and binding_or_mod.small_field is None
            and default_small_field is not None
        ):
            return NormModule(
                binding_or_mod.prefix,
                geometry=binding_or_mod.geometry or g,
                zero_centered=binding_or_mod.zero_centered,
                eps=binding_or_mod.eps,
                param_name=binding_or_mod.param_name,
                small_field=default_small_field,
            )
        return binding_or_mod
    if isinstance(binding_or_mod, ModuleBinding):
        b = binding_or_mod
        vals = {
            "hidden": int(g.H),
            "H": int(g.H),
            "eps": float(getattr(g, "eps", 1e-6)),
        }
        opts = {
            **b.options,
            "small_field": b.options.get("small_field") or default_small_field,
        }
        res = ResolvedModuleGeometry(
            b.module_cls, b.prefix, vals, opts, b.rename, b.transforms
        )
        return b.module_cls._from_resolved(res)
    return None


def resolve_spec_and_norms(
    model_def: Any, ex: ExecSpec | None = None, g: Any = None
) -> tuple[ModelSpec, Any, Module | None, Module | None, Module | None]:
    """Resolve `(ModelSpec, geometry, pre_norm, post_norm, final_norm)`."""
    if not (isinstance(model_def, type) and issubclass(model_def, ModelDef)):
        raise TypeError(f"Expected a ModelDef subclass, got {model_def!r}")
    geom = model_def.geometry() if g is None else g
    ms = model_def.to_model_spec(geom, R=None if ex is None else ex.R)
    sched = model_def.schedule
    pre_n = (
        _resolve_norm_mod(sched.pre_norm, geom, default_small_field="in_norm")
        if sched is not None
        else None
    )
    post_n = (
        _resolve_norm_mod(sched.post_norm, geom, default_small_field="post_norm")
        if sched is not None
        else None
    )
    final_n = _resolve_norm_mod(model_def.final_norm, geom, default_small_field=None)
    return ms, geom, pre_n, post_n, final_n


def load_logical_weights(
    model_def: Any,
    ex: ExecSpec,
    hf: Mapping[str, Any],
    *,
    g: Any = None,
    e_cores: list[list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Shard unsharded HF weights across `R = ex.R` ranks from `hf_params()`."""
    ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(model_def, ex, g=g)
    geom.check(ex)
    R, L = ex.R, int(geom.L)
    er, ew = int(geom.ER(R)), int(geom.EW(R))
    moe_tp = bool(getattr(geom, "moe_tp", True))

    mods_l = [m for m in (pre_n, post_n) if m is not None] + list(ms.all_modules())
    per_layer = [d for m in mods_l for d in m.hf_params(g=geom)]
    final_d = final_n.hf_params(g=geom) if final_n is not None else []
    decls = [*per_layer, *final_d]

    fuse_members: dict[str, list[HFParamDecl]] = {}
    for d in decls:
        if d.transform.fuse_into is not None:
            fuse_members.setdefault(d.transform.fuse_into, []).append(d)
    for grp in fuse_members.values():
        grp.sort(key=lambda m: m.transform.fuse_order)

    canon_hf = {
        d.name: _coerce_canonical_hf_entry(
            d, hf, fuse_members, has_layer_dim=(d not in final_d)
        )
        for d in decls
    }

    if not moe_tp and ew > er:
        e_cores = default_eplb_cores(geom, R) if e_cores is None else e_cores
        known_names = {d.name for d in decls}
        non_exp = [d for d in decls if d.transform.expert_axis is None]
        exp_decls = [d for d in decls if d.transform.expert_axis is not None]
        logical_weights: dict[str, Any] = apply_hf_transforms(non_exp, canon_hf, R=R)
        idx = jnp.maximum(jnp.asarray(e_cores, dtype=I32), 0)
        l_idx = jnp.arange(L, dtype=I32)[:, None, None]
        exp_sharded: dict[str, jax.Array] = {}
        exp_fuse: dict[str, list[HFParamDecl]] = {}
        for d in exp_decls:
            arr_d = canon_hf[d.name]
            _validate_decl_and_array(d, arr_d, known_names)
            if arr_d.shape != (L, *d.shape):
                raise ValueError(
                    f"EPLB {d.name!r} shape {arr_d.shape} != {(L, *d.shape)}"
                )
            gathered = arr_d[l_idx, idx]
            exp_sharded[d.name] = jnp.transpose(
                gathered, (1, 0, *range(2, gathered.ndim))
            )
            if d.transform.fuse_into is not None:
                exp_fuse.setdefault(d.transform.fuse_into, []).append(d)
            else:
                logical_weights[d.name] = exp_sharded[d.name]
        for target, members in exp_fuse.items():
            ms_sorted = sorted(members, key=lambda m: m.transform.fuse_order)
            logical_weights[target] = jnp.concatenate(
                [exp_sharded[m.name] for m in ms_sorted],
                axis=ms_sorted[0].transform.fuse_axis,
            )
        logical_weights["e_cores"] = e_cores
        logical_weights["ecopy"] = encode_eplb_ecopy(geom, R, e_cores)
        return logical_weights

    return apply_hf_transforms(decls, canon_hf, R=R)


def pack_logical_weights(
    model_def: Any,
    ex: ExecSpec,
    logical_weights: Mapping[str, Any],
    *,
    g: Any = None,
    merged_tiles: bool = False,
) -> dict[str, Any]:
    """Pack all-ranks `logical_weights` into tiled megakernel weights (`weights_mk`)."""
    ms, _, pre_n, post_n, final_n = resolve_spec_and_norms(model_def, ex, g=g)
    return pack_spec_logical_weights(
        ms,
        ex,
        logical_weights,
        pre_norm=pre_n,
        post_norm=post_n,
        final_norm=final_n,
        merged_tiles=merged_tiles,
    )


def load_hf(
    model_def: Any, ex: ExecSpec, hf: Mapping[str, Any], *, g: Any = None, **kw: Any
) -> dict[str, Any]:
    """Load and pack unsharded HF weights across `R = ex.R` ranks."""
    mt = bool(kw.pop("merged_tiles", False))
    logical = load_logical_weights(model_def, ex, hf, g=g, **kw)
    return pack_logical_weights(model_def, ex, logical, g=g, merged_tiles=mt)


__all__ = [
    "coerce_canonical_hf_entry",
    "load_hf",
    "load_logical_weights",
    "pack_logical_weights",
    "pack_spec_logical_weights",
    "resolve_spec_and_norms",
    "switch_branch_for_layer",
]

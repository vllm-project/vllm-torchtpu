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
"""`ModelSpec`-driven weight packing (`pack_spec_logical_weights`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .. import planner
from ..exec_spec import ExecSpec
from ..module import Module
from ..modules.norm import NormModule
from ..ops import F32
from ..spec import ModelSpec, SwitchSlot
from .eplb import resolve_mapping_ecopy


def switch_branch_for_layer(slot: SwitchSlot, lyr: int) -> tuple[Module, int]:
    """Return `(active_branch_module, ordinal_layer_index)` for `slot` at `lyr`."""
    cyc_idx, rem = divmod(lyr, slot.period)
    acc = 0
    for branch, count in slot.active_pairs:
        if rem < acc + count:
            return branch, cyc_idx * count + (rem - acc)
        acc += count
    raise AssertionError(f"Unreachable SwitchSlot index for layer {lyr}")


def pack_spec_logical_weights(
    model_spec: ModelSpec,
    ex: ExecSpec,
    logical_weights: Mapping[str, Any],
    *,
    pre_norm: Module | None = None,
    post_norm: Module | None = None,
    final_norm: Module | None = None,
    merged_tiles: bool = False,
) -> dict[str, Any]:
    """Pack all-ranks `logical_weights` for a `ModelSpec` into `weights_mk`."""
    ms, geom = model_spec, model_spec.config
    p = planner.plan(ms, ex)
    R, L = ex.R, int(geom.L)
    lay = p.smalls_layout
    sm = jnp.zeros((R, L, 1, p.smalls_width), dtype=F32)

    def set_sm(name: str, val: jax.Array, lyr: int | None = None) -> None:
        nonlocal sm
        o, w = lay[name]
        vw = min(w, val.shape[-1])
        assert val.shape[-1] <= w, (name, val.shape[-1], w)
        if lyr is None:
            sh = (R, L, val.shape[-1])
            v = jnp.broadcast_to(val[None], sh) if val.ndim == 2 else val
            sm = sm.at[:, :, 0, o : o + vw].set(v[:, :, :vw].astype(F32))
        else:
            sh = (R, val.shape[-1])
            v = jnp.broadcast_to(val[None], sh) if val.ndim == 1 else val
            sm = sm.at[:, lyr, 0, o : o + vw].set(v[:, :vw].astype(F32))

    populated_sm: set[str] = set()

    def small_pairs(mod: Module) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        seen_sfs: set[str] = set()
        out: list[tuple[str, str]] = []
        if mod.has_hf_params:
            for d in mod.hf_params(g=geom):
                sf_name = d.transform.small_field
                if sf_name is not None:
                    if sf_name not in lay:
                        raise ValueError(
                            f"{mod.name!r} {d.name!r} small_field={sf_name!r} "
                            "not in lay"
                        )
                    pair = (sf_name, d.transform.fuse_into or d.name)
                    if pair not in seen:
                        seen.add(pair)
                        seen_sfs.add(sf_name)
                        out.append(pair)
        if not isinstance(mod, NormModule):
            for sf in mod.smalls(geom, ex):
                if sf.name == "ecopy" or sf.name in seen_sfs or sf.name in populated_sm:
                    continue
                if not mod.has_hf_params or sf.name in logical_weights:
                    out.append((sf.name, sf.name))
                    seen_sfs.add(sf.name)
        return out

    mods_l = [m for m in (pre_norm, post_norm) if m is not None]
    mods_l.extend(s for s in ms.layer_type.slots if isinstance(s, Module))
    for mod in mods_l:
        for sm_field, l_key in small_pairs(mod):
            if l_key not in logical_weights:
                raise ValueError(f"Missing logical weight {l_key!r} for {sm_field!r}")
            set_sm(sm_field, logical_weights[l_key])
            populated_sm.add(sm_field)

    if (
        "ecopy" in lay
        and (ec := resolve_mapping_ecopy(logical_weights, None, geom, R)) is not None
    ):
        set_sm("ecopy", ec)

    for slot in ms.layer_type.slots:
        if isinstance(slot, SwitchSlot):
            b_pairs = {b.name: small_pairs(b) for b in slot.active_branches}
            for lyr in range(L):
                b_mod, ord_idx = switch_branch_for_layer(slot, lyr)
                for sm_field, l_key in b_pairs[b_mod.name]:
                    if l_key not in logical_weights:
                        raise ValueError(f"Missing {l_key!r} for {sm_field!r}")
                    arr = logical_weights[l_key]
                    sl_l = (
                        (arr[lyr] if arr.shape[0] == L else arr[ord_idx])
                        if arr.ndim == 2
                        else (arr[:, lyr] if arr.shape[1] == L else arr[:, ord_idx])
                    )
                    set_sm(sm_field, sl_l, lyr=lyr)
                    populated_sm.add(sm_field)

    for mod in ms.all_modules():
        if p.module_layer_counts.get(mod.name, 0) > 0:
            for sf in mod.smalls(geom, ex):
                if sf.name != "ecopy" and sf.name not in populated_sm:
                    raise ValueError(f"{mod.name!r} smalls() {sf.name!r} unmapped")

    def tile_w(arr: jax.Array, kind: str) -> jax.Array:
        *lead, k_in, n_in = arr.shape
        nt_, tk_, exp_n = p.nt[kind], p.tk[kind], p.kinds[kind][1]
        assert nt_ * tk_ == k_in and n_in == exp_n, (kind, k_in, nt_, tk_, n_in, exp_n)
        return arr if merged_tiles else arr.reshape((*lead, nt_, tk_, n_in))

    weights_mk: dict[str, Any] = {}
    for mod in ms.all_modules():
        if (n_layers := p.module_layer_counts.get(mod.name, 0)) == 0:
            continue
        for wd in mod.weights(geom, ex):
            raw = (
                jnp.concatenate([logical_weights[s] for s in wd.fuse_from], axis=-1)
                if wd.fuse_from
                else logical_weights[wd.name]
            )
            if wd.stream_ring in ("static", "dynamic"):
                if wd.layout == "stacked":
                    if isinstance(raw, (tuple, list)):
                        raw = jnp.stack(raw, axis=1)
                    if raw.ndim == 1 + len(wd.lead_dims) + 2:
                        sh = (R, *raw.shape)
                        raw = jnp.broadcast_to(raw[None, ...], sh).astype(wd.dtype)
                    weights_mk[wd.name] = tile_w(raw, wd.name)
                else:
                    is_seq = isinstance(raw, (tuple, list))
                    src = raw if is_seq else [raw[:, i] for i in range(n_layers)]
                    weights_mk[wd.name] = tuple(tile_w(x, wd.name) for x in src)
            else:
                sh = (
                    (R, n_layers, *wd.lead_dims, wd.K * wd.N)
                    if (merged_tiles and wd.K == 1 and len(wd.lead_dims) >= 1)
                    else (R, n_layers, *wd.lead_dims, wd.K, wd.N)
                )
                weights_mk[wd.name] = raw.reshape(sh).astype(wd.dtype)

    if "smalls" in p.w_names:
        weights_mk["smalls"] = sm
    if "final_norm" in p.w_names:
        if final_norm is not None and final_norm.has_hf_params:
            fn_decl = final_norm.hf_params(g=geom)[0]
            fn_key = fn_decl.name if fn_decl.name in logical_weights else "final_norm"
            fn_dt = fn_decl.dtype
        else:
            fn_key, fn_dt = "final_norm", F32
        if fn_key in logical_weights:
            fn_raw = logical_weights[fn_key]
            fn_3d = (
                fn_raw[None, None, :]
                if fn_raw.ndim == 1
                else (fn_raw[:, None, :] if fn_raw.ndim == 2 else fn_raw)
            )
            weights_mk["final_norm"] = jnp.broadcast_to(
                fn_3d, (R, 1, int(geom.H))
            ).astype(fn_dt)
    return weights_mk


__all__ = ["pack_spec_logical_weights", "switch_branch_for_layer"]

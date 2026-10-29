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
"""`ModelSpec`-driven synthetic weight and input generators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel
from ..decls import HFParamDecl
from ..exec_spec import ExecSpec
from ..modules.transforms import apply_hf_transforms
from ..ops import BF16, F32, FP8, I32
from ..pool.config import POOL_STATE_NAME
from ..spec import ModelSpec, SwitchSlot
from .eplb import default_eplb_cores, encode_eplb_ecopy
from .logical import _coerce_canonical_hf_entry
from .spec_pack import pack_spec_logical_weights, switch_branch_for_layer


def _module_layer_counts(ms: Any, L: int) -> dict[str, int]:
    counts: dict[str, int] = {m.name: 0 for m in ms.all_modules()}
    for s in ms.layer_type.slots:
        if isinstance(s, SwitchSlot):
            for lyr in range(L):
                b_mod, _ = switch_branch_for_layer(s, lyr)
                counts[b_mod.name] = counts.get(b_mod.name, 0) + 1
        else:
            counts[s.name] = counts.get(s.name, 0) + L
    return counts


def random_spec_logical_weights(
    key: jax.Array, model_spec: ModelSpec, ex: ExecSpec
) -> dict[str, Any]:
    """Synthesize all-ranks logical weights directly from a `ModelSpec`."""
    geom = model_spec.config
    geom.check(ex)
    R, L = ex.R, int(geom.L)
    counts = _module_layer_counts(model_spec, L)
    active_mods = [
        (m, counts[m.name])
        for m in model_spec.all_modules()
        if counts.get(m.name, 0) > 0
    ]
    n_keys = 32 + sum(
        (len(m.hf_params(g=geom)) if m.has_hf_params else 0)
        + 4 * len(m.weights(geom, ex))
        + len(m.smalls(geom, ex))
        for m, _ in active_mods
    )
    it = iter(jax.random.split(key, max(32, n_keys)))

    def _draw_hf(d: HFParamDecl, lead: tuple[int, ...]) -> jax.Array:
        hf_sh = d.shape if d.hf_shape is None else d.hf_shape
        sh = (*lead, *hf_sh)
        if d.transform.scale_for is not None:
            return jnp.abs(jax.random.normal(next(it), sh, dtype=F32)) * 0.02 + 0.01
        if d.transform.quantize is not None or jnp.dtype(d.dtype) == jnp.dtype(FP8):
            return (jax.random.normal(next(it), sh, dtype=F32) * 0.25).astype(FP8)
        mean = 0.0 if d.init is None else float(d.init.mean)
        std = 0.05 if d.init is None else float(d.init.std)
        raw = jax.random.normal(next(it), sh, dtype=d.dtype) * std
        return jnp.ones(sh, dtype=d.dtype) + raw if mean == 1.0 else raw + mean

    logical_weights: dict[str, Any] = {}
    moe_tp = bool(getattr(geom, "moe_tp", True))
    has_eplb = (
        not moe_tp
        and hasattr(geom, "EW")
        and hasattr(geom, "ER")
        and int(geom.EW(R)) > int(geom.ER(R))
    )

    for mod, n_l in active_mods:
        if mod.has_hf_params:
            decls = mod.hf_params(g=geom)
            hf_mod: dict[str, jax.Array] = {}
            split_sources: dict[str, jax.Array] = {}
            fuse_members: dict[str, list[HFParamDecl]] = {}
            for d in decls:
                if d.transform.fuse_into is not None:
                    fuse_members.setdefault(d.transform.fuse_into, []).append(d)
                if d.transform.split_from is not None:
                    if d.transform.split_from not in split_sources:
                        split_sources[d.transform.split_from] = _draw_hf(d, (n_l,))
                    hf_mod[d.name] = split_sources[d.transform.split_from]
                else:
                    hf_mod[d.name] = _draw_hf(d, (n_l,))
            for grp in fuse_members.values():
                grp.sort(key=lambda m: m.transform.fuse_order)
            canon_hf = {
                d.name: _coerce_canonical_hf_entry(
                    d, hf_mod, fuse_members, has_layer_dim=True
                )
                for d in decls
            }
            if has_eplb and any(d.transform.expert_axis is not None for d in decls):
                e_cores = default_eplb_cores(geom, R)
                non_exp = [d for d in decls if d.transform.expert_axis is None]
                exp_decls = [d for d in decls if d.transform.expert_axis is not None]
                logical_weights.update(apply_hf_transforms(non_exp, canon_hf, R=R))
                idx = jnp.maximum(jnp.asarray(e_cores, dtype=I32), 0)
                l_idx = jnp.arange(n_l, dtype=I32)[:, None, None]
                exp_sharded: dict[str, jax.Array] = {}
                exp_fuse: dict[str, list[HFParamDecl]] = {}
                for d in exp_decls:
                    gathered = canon_hf[d.name][l_idx, idx]
                    exp_sharded[d.name] = jnp.transpose(
                        gathered, (1, 0, *range(2, gathered.ndim))
                    )
                    if d.transform.fuse_into is not None:
                        exp_fuse.setdefault(d.transform.fuse_into, []).append(d)
                    else:
                        logical_weights[d.name] = exp_sharded[d.name]
                for target, members in exp_fuse.items():
                    ms_s = sorted(members, key=lambda m: m.transform.fuse_order)
                    logical_weights[target] = jnp.concatenate(
                        [exp_sharded[m.name] for m in ms_s],
                        axis=ms_s[0].transform.fuse_axis,
                    )
                logical_weights["e_cores"] = e_cores
                logical_weights["ecopy"] = encode_eplb_ecopy(geom, R, e_cores)
            else:
                logical_weights.update(apply_hf_transforms(decls, canon_hf, R=R))
        else:
            for wd in mod.weights(geom, ex):
                t_names = wd.fuse_from if wd.fuse_from else (wd.name,)
                n_part = wd.N // len(t_names)
                is_fp8 = jnp.dtype(wd.dtype) == jnp.dtype(FP8)
                for t_name in t_names:
                    if t_name in logical_weights:
                        continue
                    if wd.layout == "stacked":
                        sh = (R, n_l, *wd.lead_dims, wd.K, n_part)
                        logical_weights[t_name] = (
                            (jax.random.normal(next(it), sh, dtype=F32) * 0.25).astype(
                                FP8
                            )
                            if is_fp8
                            else (
                                jax.random.normal(next(it), sh, dtype=F32) * 0.05
                            ).astype(wd.dtype)
                        )
                    else:
                        sh = (R, *wd.lead_dims, wd.K, n_part)
                        logical_weights[t_name] = tuple(
                            (jax.random.normal(next(it), sh, dtype=F32) * 0.25).astype(
                                FP8
                            )
                            if is_fp8
                            else (
                                jax.random.normal(next(it), sh, dtype=F32) * 0.05
                            ).astype(wd.dtype)
                            for _ in range(n_l)
                        )

        mapped_sfs = (
            {
                d.transform.small_field
                for d in mod.hf_params(g=geom)
                if d.transform.small_field is not None
                and (d.transform.fuse_into or d.name) in logical_weights
            }
            if mod.has_hf_params
            else set()
        )
        norm_base = 0.0 if getattr(geom, "zero_centered_norm", False) else 1.0
        for sf in mod.smalls(geom, ex):
            if sf.name == "ecopy":
                if "ecopy" not in logical_weights and has_eplb:
                    e_cores = default_eplb_cores(geom, R)
                    logical_weights["e_cores"] = e_cores
                    logical_weights["ecopy"] = encode_eplb_ecopy(geom, R, e_cores)
                continue
            if sf.name in mapped_sfs or sf.name in logical_weights:
                continue
            if sf.name in ("in_norm", "post_norm"):
                logical_weights[sf.name] = (
                    norm_base
                    + jax.random.normal(next(it), (L, sf.width), dtype=F32) * 0.05
                )
            elif sf.name == mod.input_norm or sf.name.endswith("norm"):
                base_n = (
                    norm_base
                    + jax.random.normal(next(it), (1, n_l, sf.width), dtype=F32) * 0.05
                )
                logical_weights[sf.name] = jnp.broadcast_to(base_n, (R, n_l, sf.width))
            elif sf.name.endswith("_s"):
                logical_weights[sf.name] = (
                    jnp.abs(jax.random.normal(next(it), (R, n_l, sf.width), dtype=F32))
                    * 0.02
                    + 0.01
                )
            else:
                logical_weights[sf.name] = (
                    jax.random.normal(next(it), (R, n_l, sf.width), dtype=F32) * 0.05
                )

    if "final_norm" not in logical_weights:
        fn_base = 0.0 if getattr(geom, "zero_centered_norm", False) else 1.0
        logical_weights["final_norm"] = (
            fn_base + jax.random.normal(next(it), (int(geom.H),), dtype=F32) * 0.05
        )
    return logical_weights


def random_spec_inputs(
    key: jax.Array,
    model_spec: ModelSpec,
    ex: ExecSpec,
    *,
    logical_weights: dict[str, Any] | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate logical `(pos, hidden, w_log, states)` from `model_spec`."""
    geom = model_spec.config
    geom.check(ex)
    counts = _module_layer_counts(model_spec, int(geom.L))
    mods = [m for m in model_spec.all_modules() if counts.get(m.name, 0) > 0]
    keys = jax.random.split(key, 2 + len(mods))
    k_w, k_hid, *k_mods = keys

    base_pos = jnp.arange(1, ex.NS + 1, dtype=I32) * min(16, max(1, ex.ctx // 4))
    pos = (
        base_pos
        if ex.T == 1
        else (base_pos[:, None] + jnp.arange(ex.T, dtype=I32)[None, :]).reshape(ex.B)
    )
    hidden = jax.random.normal(k_hid, (ex.B, int(geom.H)), dtype=BF16) * 0.1
    w_log = (
        random_spec_logical_weights(k_w, model_spec, ex)
        if logical_weights is None
        else dict(logical_weights)
    )

    logical_states: dict[str, Any] = {}
    for k_m, mod in zip(k_mods, mods):
        n_l = counts[mod.name]
        mod_st = dict(mod.init_logical_states(k_m, geom, ex, n_l))
        for sd in mod.states(geom, ex):
            if sd.name in (POOL_STATE_NAME, "page_table") or sd.name in mod_st:
                continue
            sh = (ex.R, n_l, *sd.shape) if sd.per_layer else (ex.R, *sd.shape)
            mod_st[sd.name] = jnp.zeros(sh, dtype=sd.dtype)
        logical_states.update(mod_st)
    return pos, hidden, w_log, logical_states


def pack_spec_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: Mapping[str, Any],
    logical_states: Mapping[str, Any],
    model_spec: ModelSpec,
    ex: ExecSpec,
    *,
    mesh: Any = None,
    pre_norm: Any = None,
    post_norm: Any = None,
    final_norm: Any = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical weights and states for a `ModelSpec` into megakernel layout."""
    geom = model_spec.config
    counts = _module_layer_counts(model_spec, int(geom.L))
    R, B, M = ex.R, ex.B, ex.M
    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, int(geom.H))
    weights_mk = pack_spec_logical_weights(
        model_spec,
        ex,
        logical_weights,
        pre_norm=pre_norm,
        post_norm=post_norm,
        final_norm=final_norm,
    )

    states_mk: dict[str, Any] = {}
    for mod in model_spec.all_modules():
        if counts.get(mod.name, 0) > 0:
            states_mk.update(mod.pack_logical_states(logical_states, geom, ex))
            for sd in mod.states(geom, ex):
                if sd.name not in states_mk and sd.name in logical_states:
                    states_mk[sd.name] = kernel.fold(logical_states[sd.name])

    if mesh is not None:
        rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))
        pos, hid_mk = jax.device_put(pos, rep), jax.device_put(hid_mk, tp)

        def _put(d: dict[str, Any]) -> dict[str, Any]:
            return {
                k: tuple(jax.device_put(x, tp) for x in v)
                if isinstance(v, tuple)
                else jax.device_put(v, tp)
                for k, v in d.items()
            }

        weights_mk, states_mk = _put(weights_mk), _put(states_mk)
    return pos, hid_mk, weights_mk, states_mk


__all__ = [
    "pack_spec_inputs",
    "random_spec_inputs",
    "random_spec_logical_weights",
]

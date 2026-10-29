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
"""Composed multi-rank reference step along a resolved `Schedule`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .decls import ModuleBinding
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import ModelGeometry
from .modules.gdn import GDNModule
from .modules.gqa import GQAModule
from .modules.norm import NormModule
from .ops import BF16, F32, rmsnorm, rnd, rup
from .pool.config import cache_exec
from .schedule import ModuleOrBinding, ResolvedSchedule
from .spec import ModelSpec, Module, SwitchSlot
from .validate import norm_zc_and_eps


def _make_ref_module(m: ModuleOrBinding | None, g: ModelGeometry) -> Module | None:
    if m is None or isinstance(m, Module):
        return m
    if not isinstance(m, ModuleBinding):
        raise MegakernelConfigError(f"Expected Module or ModuleBinding, got {m!r}")
    if issubclass(m.module_cls, NormModule):
        zc, eps_val = norm_zc_and_eps(m, None, g.eps)
        return m.module_cls(
            m.prefix,
            geometry=g,
            zero_centered=zc,
            eps=eps_val,
            param_name=m.options.get("param_name"),
        )
    opts = {k: v for k, v in m.options.items() if v is not None}
    return m.module_cls(geometry=g, prefix=m.prefix, **opts)


def prepare_reference_step_state(
    logical_states: Mapping[str, Any],
    g: Any,
    ex: ExecSpec,
    *,
    pool_io: bool = False,
    include_gdn_defaults: bool = True,
) -> tuple[dict[str, Any], Any, Any, int, int, int, int]:
    """Prepare mutable `step_state` dict (including GDN `conv_out`/`rec_out`)."""
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    # Forward all logical state keys into `step_state` so custom module states
    # reach `reference` / `module_oracle` unmodified; `_compare_state_dict`
    # requires the kernel to return every state key present in `ref_st`.
    step_state: dict[str, Any] = dict(logical_states)
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    vh = KD = VD = 0
    conv_in = rec_in = None
    has_gdn = "conv" in logical_states or "rec" in logical_states
    ngdn = (
        int(logical_states["conv"].shape[1])
        if "conv" in logical_states
        else (
            int(logical_states["rec"].shape[1])
            if "rec" in logical_states
            else (int(getattr(g, "NGDN", 0)) if include_gdn_defaults else 0)
        )
    )
    if ngdn > 0 and (has_gdn or include_gdn_defaults):
        lay_gdn = GDNModule.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD, lay_gdn.QKV
        conv_in = logical_states.get(
            "conv", jnp.zeros((R, ngdn, NS, 3, QKV), dtype=F32)
        )
        if pool_io:
            conv_in = conv_in.astype(BF16).astype(F32)
        rec_in = logical_states.get(
            "rec", jnp.zeros((R, ngdn, NS, vh, KD, VD), dtype=BF16)
        ).reshape(R, ngdn, NS, vh, KD, VD)
        step_state["conv"], step_state["rec"] = conv_in, rec_in
        step_state["conv_out"] = jnp.zeros(
            (R, ngdn, CR, QKV) if T == 1 else (R, ngdn, NS, HR, QKV), dtype=F32
        )
        step_state["rec_out"] = jnp.zeros((R, ngdn, B * vh, KD, VD), dtype=BF16)
    return step_state, conv_in, rec_in, ngdn, vh, KD, VD


def finalize_reference_step_state(
    step_state: Mapping[str, Any],
    conv_in: Any,
    rec_in: Any,
    ngdn: int,
    vh: int,
    KD: int,
    VD: int,
    ex: ExecSpec,
    *,
    pool_io: bool = False,
) -> dict[str, Any]:
    """Finalize output state dict from `step_state` after reference execution."""
    R, T, NS = ex.R, ex.T, ex.NS
    HR = rup(3 + T, 8)
    out_states: dict[str, Any] = {
        k: v
        for k, v in step_state.items()
        if k not in ("conv", "rec", "conv_out", "rec_out")
    }
    if ngdn > 0 and conv_in is not None and rec_in is not None:
        conv_out, rec_out = step_state["conv_out"], step_state["rec_out"]
        if pool_io:
            conv_out = conv_out.astype(BF16).astype(F32)
        if ex.VER:
            pad_spec = ((0, 0), (0, 0), (0, 0), (0, HR - conv_in.shape[-2]), (0, 0))
            conv_pad = (
                jnp.pad(conv_in, pad_spec)
                if conv_in.ndim == 5 and conv_in.shape[-2] < HR
                else conv_in
            )
            out_states["conv"] = conv_pad
            out_states["rec"] = rec_in.reshape(R, ngdn, NS * vh, KD, VD)
            out_states["conv_ckpt"], out_states["rec_ckpt"] = conv_out, rec_out
        else:
            out_states["conv"], out_states["rec"] = conv_out, rec_out
    return out_states


def composed_reference_step(
    sched: ResolvedSchedule,
    final_norm: ModuleOrBinding | None,
    g: ModelGeometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    *,
    paged_kv: bool | None = None,
    pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:
    """Execute hybrid GQA/GDN + MoE module `reference()` along `sched`."""
    B, L = ex.B, g.L
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    step_state, conv_in, rec_in, ngdn, vh, KD, VD = prepare_reference_step_state(
        logical_states, g, ex, pool_io=pool_io, include_gdn_defaults=True
    )

    res = hidden[:B].astype(F32)
    for lyr in range(L):
        pre_mod = _make_ref_module(sched.pre_norm[lyr], g)
        if pre_mod is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'pre_norm' is None (kernel requires pre_norm)"
            )
        pre_key = (
            pre_mod.renamed_param_name(pre_mod.prefix, g)
            if isinstance(pre_mod, NormModule)
            else "in_norm"
        )
        pre_w = logical_weights[pre_key if pre_key in logical_weights else "in_norm"]
        x_mix = pre_mod.reference(pre_w[lyr], res, g=g)[0]
        mix_mod = _make_ref_module(sched.mixer[lyr], g)
        if mix_mod is None:
            raise MegakernelConfigError(f"Layer {lyr} has no mixer module")
        b_idx = sched.mixer_branch_indices[lyr]
        if isinstance(mix_mod, GQAModule):
            mix_kw = {
                "pos": pos,
                "att_idx": b_idx,
                "layer_idx": lyr,
                "paged_kv": use_paged_kv,
            }
        elif isinstance(mix_mod, GDNModule):
            mix_kw = {"gdn_idx": b_idx, "layer_idx": lyr, "multi_layer": True}
        else:
            mix_kw = {"branch_idx": b_idx, "layer_idx": lyr}
        mix_delta, step_state = mix_mod.reference(
            logical_weights, x_mix, step_state, g=g, ex=ex, **mix_kw
        )
        res = rnd(res + mix_delta)
        post_mod = _make_ref_module(sched.post_norm[lyr], g)
        if post_mod is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'post_norm' is None (kernel requires post_norm)"
            )
        post_key = (
            post_mod.renamed_param_name(post_mod.prefix, g)
            if isinstance(post_mod, NormModule)
            else "post_norm"
        )
        post_w = logical_weights[
            post_key if post_key in logical_weights else "post_norm"
        ]
        x_ffn = post_mod.reference(post_w[lyr], res, g=g)[0]
        ffn_mod = _make_ref_module(sched.ffn[lyr], g)
        if ffn_mod is None:
            raise MegakernelConfigError(f"Layer {lyr} has no ffn module")
        ffn_delta, step_state = ffn_mod.reference(
            logical_weights, x_ffn, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = rnd(res + ffn_delta)

    fn_mod = _make_ref_module(final_norm, g)
    if fn_mod is None:
        raise MegakernelConfigError("final_norm is None (kernel requires final_norm)")
    fn_key = (
        fn_mod.renamed_param_name(fn_mod.prefix, g)
        if isinstance(fn_mod, NormModule)
        else "final_norm"
    )
    fn_w = logical_weights[fn_key if fn_key in logical_weights else "final_norm"]
    out_b = fn_mod.reference(fn_w, res, g=g, out_dtype=BF16)[0]
    out_states = finalize_reference_step_state(
        step_state, conv_in, rec_in, ngdn, vh, KD, VD, ex, pool_io=pool_io
    )
    return out_b, out_states


def spec_reference_step(
    spec: ModelSpec,
    g: Any,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    *,
    paged_kv: bool | None = None,
    pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:
    """Execute module `reference()` along `spec.layer_type.slots`."""
    from .weights.spec_pack import switch_branch_for_layer

    B, L = ex.B, int(g.L)
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    has_gdn = any(isinstance(m, GDNModule) for m in spec.all_modules())
    step_state, conv_in, rec_in, ngdn, vh, KD, VD = prepare_reference_step_state(
        logical_states, g, ex, pool_io=pool_io, include_gdn_defaults=has_gdn
    )
    res = hidden[:B].astype(F32)
    for lyr in range(L):
        for slot in spec.layer_type.slots:
            mod, b_idx = (
                switch_branch_for_layer(slot, lyr)
                if isinstance(slot, SwitchSlot)
                else (slot, lyr)
            )
            if mod.input_norm is not None:
                in_w = logical_weights[mod.input_norm]
                in_w_l = (
                    in_w[0, lyr]
                    if in_w.ndim == 3
                    else (in_w[lyr] if in_w.ndim == 2 else in_w)
                )
                x_in: Any = rmsnorm(
                    res,
                    in_w_l,
                    eps=float(g.eps),
                    zero_centered=bool(g.zero_centered_norm),
                ).astype(BF16)
            else:
                x_in = res
            if isinstance(mod, GQAModule):
                kw: dict[str, Any] = {
                    "pos": pos,
                    "att_idx": b_idx,
                    "layer_idx": lyr,
                    "paged_kv": use_paged_kv,
                }
            elif isinstance(mod, GDNModule):
                kw = {"gdn_idx": b_idx, "layer_idx": lyr, "multi_layer": True}
            else:
                kw = {"branch_idx": b_idx, "layer_idx": lyr, "pos": pos}
            sub_out, step_state = mod.reference(
                logical_weights, x_in, step_state, g=g, ex=ex, **kw
            )
            sub_arr = jnp.asarray(sub_out).astype(F32)
            res = rnd(res + sub_arr) if mod.reduction == "allreduce" else sub_arr
    fn_w = logical_weights["final_norm"]
    fn_w_0 = fn_w[0] if fn_w.ndim >= 2 else fn_w
    out_b = rmsnorm(
        res,
        fn_w_0,
        eps=float(g.eps),
        zero_centered=bool(g.zero_centered_norm),
    ).astype(BF16)
    out_states = finalize_reference_step_state(
        step_state, conv_in, rec_in, ngdn, vh, KD, VD, ex, pool_io=pool_io
    )
    return out_b, out_states


__all__ = [
    "composed_reference_step",
    "finalize_reference_step_state",
    "prepare_reference_step_state",
    "spec_reference_step",
]

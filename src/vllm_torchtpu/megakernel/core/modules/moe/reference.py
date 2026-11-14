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
"""`MoEModule` pure-JAX reference forward pass (`moe_reference`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec
from ...ops import BF16, F32, FP8, round_bf16


def moe_reference(
    params: Mapping[str, Any],
    x: Any,
    state: Any = None,
    *,
    g: Any,
    ex: ExecSpec,
    layer_idx: int = 0,
    moe_tp: bool | None = None,
) -> tuple[jax.Array, Any]:
    """Pure-JAX multi-rank MoE forward for one layer."""
    R, B = ex.R, ex.B
    H = int(g.H)
    E = int(g.E)
    TOPK = int(g.TOPK)
    IS = int(g.IS)
    I = int(g.I)  # noqa: E741
    is_tp = bool(g.moe_tp) if moe_tp is None else bool(moe_tp)
    ISR = IS // R
    IR = I // R if is_tp else I
    ER = E // R
    rename = getattr(g, "rename", None) or {}

    def _get(canon: str) -> Any:
        key = rename.get(canon, canon)
        return params[key] if key in params else params[canon]

    def _get_opt(canon: str) -> Any:
        key = rename.get(canon, canon)
        return params[key] if key in params else params.get(canon)

    x_moe = x[:B].astype(F32)
    xb_moe = x_moe.astype(BF16).astype(F32)

    router_all = _get("router")
    if router_all.ndim == 4:
        w_router = router_all[0, layer_idx].astype(F32)
    elif router_all.ndim == 3:
        w_router = (
            router_all[layer_idx]
            if router_all.shape[0] != R or layer_idx > 0
            else router_all[0]
        ).astype(F32)
    else:
        w_router = router_all.astype(F32)
    logits = xb_moe @ w_router
    probs = jax.nn.softmax(logits, axis=-1)
    top_vals, top_idx = lax.top_k(probs, TOPK)
    top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    sh13_all = _get_opt("sh13")
    if sh13_all is None:
        sh13_all = jnp.concatenate([_get("sh_gate_proj"), _get("sh_up_proj")], axis=-1)
    sh13_l = sh13_all[:, layer_idx] if sh13_all.ndim == 4 else sh13_all
    sh13_s_all: Any = None
    if jnp.dtype(sh13_all.dtype) == jnp.dtype(FP8):
        sh13_s_all = _get_opt("sh13_s")
        if sh13_s_all is None:
            sg_s = _get_opt("sh_gate_proj_s")
            su_s = _get_opt("sh_up_proj_s")
            if sg_s is not None and su_s is not None:
                sh13_s_all = jnp.concatenate([sg_s, su_s], axis=-1)
        if sh13_s_all is None:
            raise MegakernelConfigError(
                "moe_reference requires 'sh13_s' when 'sh13' is float8_e4m3fn"
            )
    sh13_s_l = (
        None
        if sh13_s_all is None
        else (sh13_s_all[:, layer_idx] if sh13_s_all.ndim == 3 else sh13_s_all)
    )
    sh2_all = _get("sh2")
    sh2_l = sh2_all[:, layer_idx] if sh2_all.ndim == 4 else sh2_all
    sh2_s_all: Any = None
    if jnp.dtype(sh2_all.dtype) == jnp.dtype(FP8):
        sh2_s_all = _get_opt("sh2_s")
        if sh2_s_all is None:
            raise MegakernelConfigError(
                "moe_reference requires 'sh2_s' when 'sh2' is float8_e4m3fn"
            )
    sh2_s_l = (
        None
        if sh2_s_all is None
        else (sh2_s_all[:, layer_idx] if sh2_s_all.ndim == 3 else sh2_s_all)
    )

    e13_all = _get_opt("e13")
    if e13_all is None:
        e13_all = jnp.concatenate([_get("e_gate_proj"), _get("e_up_proj")], axis=-1)
    e13_l_all = e13_all[:, layer_idx] if e13_all.ndim == 5 else e13_all
    e13_s_all: Any = None
    if jnp.dtype(e13_all.dtype) == jnp.dtype(FP8):
        e13_s_all = _get_opt("e13_s")
        if e13_s_all is None:
            eg_s = _get_opt("e_gate_proj_s")
            eu_s = _get_opt("e_up_proj_s")
            if eg_s is not None and eu_s is not None:
                e13_s_all = jnp.concatenate([eg_s, eu_s], axis=-1)
        if e13_s_all is None:
            raise MegakernelConfigError(
                "moe_reference requires 'e13_s' when 'e13' is float8_e4m3fn"
            )
    e13_s_l_all = (
        None
        if e13_s_all is None
        else (e13_s_all[:, layer_idx] if e13_s_all.ndim == 4 else e13_s_all)
    )
    e2_all = _get("e2")
    e2_l_all = e2_all[:, layer_idx] if e2_all.ndim == 5 else e2_all
    e2_s_all: Any = None
    if jnp.dtype(e2_all.dtype) == jnp.dtype(FP8):
        e2_s_all = _get_opt("e2_s")
        if e2_s_all is None:
            raise MegakernelConfigError(
                "moe_reference requires 'e2_s' when 'e2' is float8_e4m3fn"
            )
    e2_s_l_all = (
        None
        if e2_s_all is None
        else (e2_s_all[:, layer_idx] if e2_s_all.ndim == 4 else e2_s_all)
    )

    multi_layer = sh13_all.ndim == 4 or e13_all.ndim == 5 or router_all.ndim == 4
    sh_gate_all = _get("sh_gate")
    if sh_gate_all.ndim == 3:
        sh_gate_l = sh_gate_all[:, layer_idx]
    elif sh_gate_all.ndim == 2 and multi_layer:
        sh_gate_l = jnp.broadcast_to(sh_gate_all[layer_idx][None, :], (R, H))
    elif sh_gate_all.ndim == 2 and sh_gate_all.shape[0] == R:
        sh_gate_l = sh_gate_all
    elif sh_gate_all.ndim == 2:
        sh_gate_l = jnp.broadcast_to(sh_gate_all[layer_idx][None, :], (R, H))
    else:
        sh_gate_l = jnp.broadcast_to(sh_gate_all[None, :], (R, H))

    moe_partials = []
    for r in range(R):
        sh_gate = sh_gate_l[r]
        sg = jax.nn.sigmoid(jnp.sum(x_moe * sh_gate[None, :], axis=-1, keepdims=True))
        w_sh13 = sh13_l[r].astype(F32)
        g13 = (
            (xb_moe @ w_sh13) * sh13_s_l[r]
            if sh13_s_l is not None
            else (xb_moe @ w_sh13)
        )
        mid_sh = (
            round_bf16(
                round_bf16(jax.nn.silu(round_bf16(g13[:, :ISR])))
                * round_bf16(g13[:, ISR:])
            )
            .astype(BF16)
            .astype(F32)
        )
        w_sh2 = sh2_l[r].astype(F32)
        ysh = (mid_sh @ w_sh2) * sh2_s_l[r] if sh2_s_l is not None else (mid_sh @ w_sh2)
        part_r = ysh * sg

        e13_l = e13_l_all[r]
        e13_s_l = None if e13_s_l_all is None else e13_s_l_all[r]
        e2_l = e2_l_all[r]
        e2_s_l = None if e2_s_l_all is None else e2_s_l_all[r]

        for b in range(B):
            for k in range(TOPK):
                e = int(top_idx[b, k])
                if not is_tp and (e // ER) != r:
                    continue
                w_slot = e if is_tp else (e - r * ER)
                w13_e = e13_l[w_slot].astype(F32)
                w2_e = e2_l[w_slot].astype(F32)
                h_e = (
                    (xb_moe[b : b + 1] @ w13_e) * e13_s_l[w_slot]
                    if e13_s_l is not None
                    else (xb_moe[b : b + 1] @ w13_e)
                )
                mid_e = (
                    round_bf16(
                        round_bf16(jax.nn.silu(round_bf16(h_e[:, :IR])))
                        * round_bf16(h_e[:, IR:])
                    )
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = (
                    round_bf16((mid_e @ w2_e) * e2_s_l[w_slot])
                    if e2_s_l is not None
                    else round_bf16(mid_e @ w2_e)
                )
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)

        moe_partials.append(part_r.astype(BF16).astype(F32))

    delta = jnp.sum(jnp.stack(moe_partials, axis=0), axis=0)
    return delta, state

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

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, FP8, rnd


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
    is_tp = bool(getattr(g, "moe_tp", True)) if moe_tp is None else bool(moe_tp)
    ISR = IS // R
    IR = I // R if is_tp else I
    ER = E // R

    x_moe = x[:B].astype(F32)
    xb_moe = x_moe.astype(BF16).astype(F32)

    router_all = params["router"]
    w_router = (router_all[layer_idx] if router_all.ndim == 3 else router_all).astype(
        F32
    )
    logits = xb_moe @ w_router
    probs = jax.nn.softmax(logits, axis=-1)
    top_vals, top_idx = lax.top_k(probs, TOPK)
    top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    sh_gate_all = params["sh_gate"]
    if sh_gate_all.ndim == 3:
        sh_gate_l = sh_gate_all[:, layer_idx]
    elif sh_gate_all.ndim == 2 and sh_gate_all.shape[0] == R:
        sh_gate_l = sh_gate_all
    elif sh_gate_all.ndim == 2:
        sh_gate_l = jnp.broadcast_to(sh_gate_all[layer_idx][None, :], (R, H))
    else:
        sh_gate_l = jnp.broadcast_to(sh_gate_all[None, :], (R, H))

    sh13_all = params["sh13"]
    sh13_l = sh13_all[:, layer_idx] if sh13_all.ndim == 4 else sh13_all
    sh13_s_all = (
        params.get("sh13_s") if jnp.dtype(sh13_all.dtype) == jnp.dtype(FP8) else None
    )
    sh13_s_l = (
        None
        if sh13_s_all is None
        else (sh13_s_all[:, layer_idx] if sh13_s_all.ndim == 3 else sh13_s_all)
    )
    sh2_all = params["sh2"]
    sh2_l = sh2_all[:, layer_idx] if sh2_all.ndim == 4 else sh2_all
    sh2_s_all = (
        params.get("sh2_s") if jnp.dtype(sh2_all.dtype) == jnp.dtype(FP8) else None
    )
    sh2_s_l = (
        None
        if sh2_s_all is None
        else (sh2_s_all[:, layer_idx] if sh2_s_all.ndim == 3 else sh2_s_all)
    )

    e13_all = params["e13"]
    e13_l_all = e13_all[:, layer_idx] if e13_all.ndim == 5 else e13_all
    e13_s_all = (
        params.get("e13_s") if jnp.dtype(e13_all.dtype) == jnp.dtype(FP8) else None
    )
    e13_s_l_all = (
        None
        if e13_s_all is None
        else (e13_s_all[:, layer_idx] if e13_s_all.ndim == 4 else e13_s_all)
    )
    e2_all = params["e2"]
    e2_l_all = e2_all[:, layer_idx] if e2_all.ndim == 5 else e2_all
    e2_s_all = params.get("e2_s") if jnp.dtype(e2_all.dtype) == jnp.dtype(FP8) else None
    e2_s_l_all = (
        None
        if e2_s_all is None
        else (e2_s_all[:, layer_idx] if e2_s_all.ndim == 4 else e2_s_all)
    )

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
            rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
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
                    rnd(rnd(jax.nn.silu(rnd(h_e[:, :IR]))) * rnd(h_e[:, IR:]))
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = (
                    rnd((mid_e @ w2_e) * e2_s_l[w_slot])
                    if e2_s_l is not None
                    else rnd(mid_e @ w2_e)
                )
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)

        moe_partials.append(part_r.astype(BF16).astype(F32))

    delta = jnp.sum(jnp.stack(moe_partials, axis=0), axis=0)
    return delta, state

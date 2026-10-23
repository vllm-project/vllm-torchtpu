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
"""`MoEBlock` pure-JAX reference forward pass (`moe_reference`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, FP8, I32, U8, rnd, rup


def dequant_nvfp4_weight(w_u8: jax.Array, n_logical: int) -> jax.Array:
    """Dequantize a `[K // 2, n_raw] uint8` NVFP4 matrix to `[K, n_logical] f32`."""
    k_half = w_u8.shape[0]
    packed = w_u8[:, :n_logical]
    scale_cols = n_logical // 8
    bscale_u8 = w_u8[:, n_logical : n_logical + scale_cols]
    bscale = (
        lax.bitcast_convert_type(bscale_u8, jnp.float8_e4m3fn)
        .astype(BF16)
        .reshape(k_half // 8, n_logical)
    )

    def _dec(nib):
        nib_i = nib.astype(I32)
        sign = jnp.where((nib_i & 8) != 0, jnp.bfloat16(-1.0), jnp.bfloat16(1.0))
        exp = (nib_i >> 1) & 3
        mant = (nib_i & 1).astype(BF16) * jnp.bfloat16(0.5)
        pow2 = jnp.left_shift(jnp.int32(1), jnp.maximum(exp - 1, 0)).astype(BF16)
        mag = jnp.where(exp == 0, mant, pow2 * (jnp.bfloat16(1.0) + mant))
        return sign * mag

    lo = _dec(packed & U8(0x0F))
    hi = _dec((packed >> U8(4)) & U8(0x0F))
    w_bf16 = jnp.stack([lo, hi], axis=1).reshape(2 * k_half, n_logical)
    return (w_bf16 * jnp.repeat(bscale, 16, axis=0)).astype(F32)


def random_nvfp4_weight(
    key: jax.Array, lead_shape: tuple[int, ...], k_dim: int, n_dim: int
) -> jax.Array:
    """Random `[*lead, K // 2, n_raw] uint8` NVFP4 matrix for `dequant_nvfp4_weight`."""
    k_half, (k_lo, k_hi, k_s) = k_dim // 2, jax.random.split(key, 3)
    nib_shape = (*lead_shape, k_half, n_dim)
    lo = jax.random.randint(k_lo, nib_shape, 0, 8, dtype=U8)
    hi = jax.random.randint(k_hi, nib_shape, 0, 8, dtype=U8)
    s_shape = (*lead_shape, k_dim // 16, n_dim)
    bscale = (jnp.abs(jax.random.normal(k_s, s_shape, dtype=F32)) * 0.1 + 0.05).astype(
        FP8
    )
    bscale_u8 = lax.bitcast_convert_type(bscale, U8).reshape(
        *lead_shape, k_half, n_dim // 8
    )
    pad = rup(n_dim // 8, 128) - n_dim // 8
    return jnp.concatenate(
        [lo | (hi << U8(4)), bscale_u8, jnp.zeros((*lead_shape, k_half, pad), U8)],
        axis=-1,
    )


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
    sh_rep = max(1, R // max(1, IS // 128))
    ISR = IS // max(1, R // sh_rep)
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
    sh13_s_all = params["sh13_s"]
    sh13_s_l = sh13_s_all[:, layer_idx] if sh13_s_all.ndim == 3 else sh13_s_all
    sh2_all = params["sh2"]
    sh2_l = sh2_all[:, layer_idx] if sh2_all.ndim == 4 else sh2_all
    sh2_s_all = params["sh2_s"]
    sh2_s_l = sh2_s_all[:, layer_idx] if sh2_s_all.ndim == 3 else sh2_s_all

    e13_all = params["e13"]
    e13_l_all = e13_all[:, layer_idx] if e13_all.ndim == 5 else e13_all
    e13_s_all = params["e13_s"]
    e13_s_l_all = e13_s_all[:, layer_idx] if e13_s_all.ndim == 4 else e13_s_all
    e2_all = params["e2"]
    e2_l_all = e2_all[:, layer_idx] if e2_all.ndim == 5 else e2_all
    e2_s_all = params["e2_s"]
    e2_s_l = e2_s_all[:, layer_idx] if e2_s_all.ndim == 4 else e2_s_all

    moe_partials = []
    for r in range(R):
        sh_gate = sh_gate_l[r]
        sg = jax.nn.sigmoid(jnp.sum(x_moe * sh_gate[None, :], axis=-1, keepdims=True))
        w_sh13 = sh13_l[r].astype(F32)
        sh13_s = sh13_s_l[r]
        g13 = (xb_moe @ w_sh13) * sh13_s
        mid_sh = (
            rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
            .astype(BF16)
            .astype(F32)
        )
        w_sh2 = sh2_l[r].astype(F32)
        sh2_s = sh2_s_l[r] * (1.0 / sh_rep) if sh_rep > 1 else sh2_s_l[r]
        part_r = (mid_sh @ w_sh2) * sh2_s * sg

        e13_l = e13_l_all[r]
        e13_s_l = e13_s_l_all[r]
        e2_l = e2_l_all[r]
        e2_s_r = e2_s_l[r]

        for b in range(B):
            for k in range(TOPK):
                e = int(top_idx[b, k])
                if not is_tp and (e // ER) != r:
                    continue
                w_slot = e if is_tp else (e - r * ER)
                w13_e = (
                    dequant_nvfp4_weight(e13_l[w_slot], 2 * IR)
                    if e13_l.dtype == U8
                    else e13_l[w_slot].astype(F32)
                )
                w2_e = (
                    dequant_nvfp4_weight(e2_l[w_slot], H)
                    if e2_l.dtype == U8
                    else e2_l[w_slot].astype(F32)
                )
                s13 = e13_s_l[w_slot]
                s2 = e2_s_r[w_slot]
                h_e = (xb_moe[b : b + 1] @ w13_e) * s13
                mid_e = (
                    rnd(rnd(jax.nn.silu(rnd(h_e[:, :IR]))) * rnd(h_e[:, IR:]))
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = rnd((mid_e @ w2_e) * s2)
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)

        moe_partials.append(part_r.astype(BF16).astype(F32))

    delta = jnp.sum(jnp.stack(moe_partials, axis=0), axis=0)
    return delta, state

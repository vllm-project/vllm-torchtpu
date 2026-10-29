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
"""Stateless Tensor-in / Tensor-out GDN and KDA linear recurrence primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .base import BF16, F32, rnd


def gdn_recurrence_step(
    state: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta_raw: jax.Array,
    a_raw: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    *,
    gnorm_weight: jax.Array | None = None,
    z_raw: jax.Array | None = None,
    decay_type: str = "head",
    decay_law: str = "softplus",
    decay_bound: float = 8.0,
    gate_type: str = "silu",
    l2_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Advance `[vh, KD, VD]` state under GDN softplus or KDA bounded decay."""
    vh, kd, vd = state.shape
    kh = q.shape[0]
    if kh <= 0 or vh % kh != 0 or q.shape != (kh, kd) or k.shape != (kh, kd):
        raise ValueError(f"incompatible q={q.shape}, k={k.shape}, state={state.shape}")
    if v.shape != (vh, vd):
        raise ValueError(f"expected v shape {(vh, vd)}, got {v.shape}")
    if decay_type not in ("head", "key") or decay_law not in (
        "softplus",
        "bounded_sigmoid",
    ):
        raise ValueError(
            f"unsupported decay_type={decay_type!r}, decay_law={decay_law!r}"
        )
    if gate_type not in ("silu", "sigmoid"):
        raise ValueError(f"unsupported gate_type={gate_type!r}")
    qf = q.astype(F32)
    kf = k.astype(F32)
    vf = v.astype(F32)
    qf = qf * lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + l2_eps) * (kd**-0.5)
    kf = kf * lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + l2_eps)
    expand = vh // kh
    if expand > 1:
        qf = jnp.broadcast_to(qf[:, None, :], (kh, expand, kd)).reshape(vh, kd)
        kf = jnp.broadcast_to(kf[:, None, :], (kh, expand, kd)).reshape(vh, kd)
    qc, kc = qf[:, :, None], kf[:, :, None]
    beta = rnd(jax.nn.sigmoid(beta_raw.astype(F32))).reshape(vh, 1)
    gate_in = a_raw.astype(F32) + dt_bias.astype(F32)
    rate = jnp.exp(a_log.astype(F32))
    log_decay = (
        -F32(decay_bound) * jax.nn.sigmoid(rate * gate_in)
        if decay_law == "bounded_sigmoid"
        else -rate * jax.nn.softplus(gate_in)
    )
    eg = (
        jnp.exp(log_decay.reshape(vh, kd))[:, :, None]
        if decay_type == "key"
        else jnp.exp(log_decay.reshape(vh))[:, None, None]
    )
    s_f32 = state.astype(F32) * eg
    s_new = s_f32 + kc * ((vf - jnp.sum(s_f32 * kc, axis=1)) * beta)[:, None, :]
    out = rnd(jnp.sum(s_new * qc, axis=1))
    next_state = s_new.astype(BF16)
    if gnorm_weight is None:
        return next_state, out
    val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + norm_eps)
    gw = gnorm_weight.astype(BF16)
    val = ((gw[None, :] if gw.ndim == 1 else gw) * val.astype(BF16)).astype(F32)
    if z_raw is None:
        return next_state, val
    zz = z_raw.astype(F32).reshape(vh, vd)
    gate_val = jax.nn.sigmoid(zz) if gate_type == "sigmoid" else jax.nn.silu(zz)
    return next_state, rnd(val * gate_val)


def kda_step(
    state: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta_raw: jax.Array,
    a_raw: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    *,
    gnorm_weight: jax.Array | None = None,
    z_raw: jax.Array | None = None,
    decay_bound: float = 8.0,
    gate_type: str = "sigmoid",
    l2_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """KDA per-key bounded-sigmoid decay step with RMS-sigmoid output gating."""
    return gdn_recurrence_step(
        state,
        q,
        k,
        v,
        beta_raw,
        a_raw,
        a_log,
        dt_bias,
        gnorm_weight=gnorm_weight,
        z_raw=z_raw,
        decay_type="key",
        decay_law="bounded_sigmoid",
        decay_bound=decay_bound,
        gate_type=gate_type,
        l2_eps=l2_eps,
        norm_eps=norm_eps,
    )

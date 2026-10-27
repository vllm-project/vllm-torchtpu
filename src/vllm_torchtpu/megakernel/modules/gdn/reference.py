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
"""`GDNModule` pure-JAX reference forward pass (`gdn_reference`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, rnd, rup
from .params import gdn_lin_in_layout, resolve_gdn_options


def gdn_reference(
    params: Mapping[str, Any],
    x: Any,
    state: Mapping[str, Any],
    *,
    g: Any,
    ex: ExecSpec,
    gdn_idx: int = 0,
    layer_idx: int | None = None,
    multi_layer: bool = True,
    packed_conv: bool = False,
    flat_rec: bool = False,
    options: Mapping[str, Any] | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Pure-JAX multi-rank GDN forward for one linear-attention layer (`gdn_idx`).

    State layout parameters are explicit:
    - `multi_layer`: when `True` (default), `state["conv"]` and `state["rec"]`
      have a leading `NGDN` layer dimension at axis 1 (`[R, NGDN, ...]`),
      indexed by `gdn_idx`. When `False`, `state["conv"]` and `state["rec"]`
      are single-layer tensors (`[R, ...]`).
    - `packed_conv`: when `True`, single-layer `conv` is packed as
      `[R, >=3*B, QKV]`; when `False` (default), `conv` has shape
      `[R, (NGDN,) NS, 3, QKV]`.
    - `flat_rec`: when `True`, `rec` flattens `(NS, vh)` into `NS * vh`; when
      `False` (default), `rec` has separate `NS, vh` axes.
    """
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    o_gdn = int(gdn_idx)
    lyr = o_gdn if layer_idx is None else int(layer_idx)
    L = int(getattr(g, "L", 1))
    opts = resolve_gdn_options(options or {}, g)
    decay_type, decay_law = opts["decay_type"], opts["decay_law"]
    gate_type, decay_bound = opts["gate_type"], opts["decay_bound"]
    lay_gdn = gdn_lin_in_layout(g, ex, decay_type=decay_type)
    kh, vh, KD, VD = lay_gdn.kh, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    QKV, Z, BA, A_OFF = lay_gdn.QKV, lay_gdn.Z, lay_gdn.BA, lay_gdn.A_OFF
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    expand = vh // kh
    eps = float(getattr(g, "eps", 1e-6))
    l2_eps = float(getattr(g, "gdn_l2_eps", 1e-6))
    xb = x[:B].astype(BF16).astype(F32)

    def _slice_sm(arr: Any) -> Any:
        if arr.ndim == 2:
            return arr
        return (
            arr[:, lyr] if arr.shape[1] == L and lyr < arr.shape[1] else arr[:, o_gdn]
        )

    w_lin_in_all = params["lin_in"]
    w_lin_out_all = params["lin_out"]
    w_lin_in_l = w_lin_in_all[:, o_gdn] if w_lin_in_all.ndim == 4 else w_lin_in_all
    w_lin_out_l = w_lin_out_all[:, o_gdn] if w_lin_out_all.ndim == 4 else w_lin_out_all

    lin_in_s_l = _slice_sm(
        params["lin_in_s"] if "lin_in_s" in params else params["in_s"]
    )
    lin_out_s_l = _slice_sm(
        params["lin_out_s"] if "lin_out_s" in params else params["out_s"]
    )
    conv_w_l = _slice_sm(params["conv"])
    alog_l = _slice_sm(params["alog"])
    dtb_l = _slice_sm(params["dtb"])
    gnorm_l = _slice_sm(params["gnorm"])

    conv_in_raw = state["conv"]
    rec_in_raw = state["rec"]
    has_multi_out = "conv_out" in state and "rec_out" in state

    conv_slice = conv_in_raw[:, o_gdn] if multi_layer else conv_in_raw
    if packed_conv:
        conv_in_l = jnp.swapaxes(conv_slice[:, : 3 * B].reshape(R, 3, B, QKV), 1, 2)
    else:
        conv_in_l = conv_slice

    rec_slice = rec_in_raw[:, o_gdn] if multi_layer else rec_in_raw
    rec_in_l = rec_slice.reshape(R, NS, vh, KD, VD) if flat_rec else rec_slice

    if has_multi_out:
        conv_out = state["conv_out"]
        rec_out = state["rec_out"]
    else:
        conv_out = (
            jnp.zeros((R, CR, QKV), dtype=F32)
            if T == 1
            else jnp.zeros((R, NS, HR, QKV), dtype=F32)
        )
        rec_out = jnp.zeros((R, B * vh, KD, VD), dtype=BF16)

    gdn_partials = []
    for r in range(R):
        w_in = w_lin_in_l[r].astype(F32)
        in_s = lin_in_s_l[r]
        ybuf = rnd((xb @ w_in) * in_s)

        cw = conv_w_l[r]
        taps = [cw[t * QKV : (t + 1) * QKV] for t in range(4)]
        if T == 1:
            s0 = conv_in_l[r, :, 0, :]
            s1 = conv_in_l[r, :, 1, :]
            s2 = conv_in_l[r, :, 2, :]
            x_c = ybuf[:, :QKV]
            u = rnd(
                jax.nn.silu(s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x_c * taps[3])
            )
            conv_next = jnp.concatenate([s1, s2, x_c], axis=0)
            if has_multi_out:
                conv_out = conv_out.at[r, o_gdn, : 3 * B, :].set(conv_next)
            else:
                conv_out = conv_out.at[r, : 3 * B, :].set(conv_next)
        else:
            us = []
            for s_idx in range(NS):
                hist = jnp.zeros((HR, QKV), dtype=F32)
                hist = hist.at[0:3, :].set(conv_in_l[r, s_idx, 0:3, :])
                hist = hist.at[3 : 3 + T, :].set(
                    ybuf[s_idx * T : (s_idx + 1) * T, :QKV]
                )
                if has_multi_out:
                    conv_out = conv_out.at[r, o_gdn, s_idx].set(hist)
                else:
                    conv_out = conv_out.at[r, s_idx].set(hist)
                u_s = rnd(
                    jax.nn.silu(
                        hist[0:T] * taps[0]
                        + hist[1 : 1 + T] * taps[1]
                        + hist[2 : 2 + T] * taps[2]
                        + hist[3 : 3 + T] * taps[3]
                    )
                )
                us.append(u_s)
            u = jnp.concatenate(us, axis=0)

        is_key_decay = decay_type == "key"
        a_dim = vh * KD if is_key_decay else vh
        alog = alog_l[r, :vh]
        exp_alog = jnp.exp(alog)[:, None] if is_key_decay else jnp.exp(alog)
        dtb = dtb_l[r, :a_dim]
        gnw = gnorm_l[r, :VD].astype(BF16)
        mix_r = []
        rec_r_list = []
        rec_prev = rec_in_l[r, 0]
        for b in range(B):
            s_idx, t_idx = divmod(b, T)
            st_cur = (
                rec_in_l[r, s_idx].astype(F32) if t_idx == 0 else rec_prev.astype(F32)
            )
            q = u[b, : kh * KD].reshape(kh, KD)
            k = u[b, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = u[b, 2 * kh * KD :].reshape(vh, VD)
            q = (
                q
                * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
                * (KD**-0.5)
            )
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
            qc = jnp.repeat(q, expand, axis=0)[:, :, None]
            kc_ = jnp.repeat(k, expand, axis=0)[:, :, None]
            beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
            a_val = ybuf[b, A_OFF : A_OFF + a_dim]
            a_shifted = (a_val + dtb).reshape(vh, KD) if is_key_decay else (a_val + dtb)
            if decay_law == "bounded_sigmoid":
                decay = -F32(decay_bound) * jax.nn.sigmoid(exp_alog * a_shifted)
            else:
                decay = -exp_alog * jax.nn.softplus(a_shifted)
            eg = (
                jnp.exp(decay)[:, :, None]
                if is_key_decay
                else jnp.exp(decay)[:, None, None]
            )
            st_cur = st_cur * eg
            pred = jnp.sum(st_cur * kc_, axis=1)
            delta = (v - pred) * beta[:, None]
            st_cur = st_cur + kc_ * delta[:, None, :]
            out_b = rnd(jnp.sum(st_cur * qc, axis=1))
            rec_prev = st_cur.astype(BF16)
            rec_r_list.append(rec_prev)
            val = out_b * lax.rsqrt(
                jnp.mean(out_b * out_b, axis=-1, keepdims=True) + eps
            )
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            gate_val = jax.nn.sigmoid(zz) if gate_type == "sigmoid" else jax.nn.silu(zz)
            mix_r.append(rnd(val * gate_val).reshape(Z))

        rec_cat = jnp.concatenate(rec_r_list, axis=0)
        if has_multi_out:
            rec_out = rec_out.at[r, o_gdn].set(rec_cat)
        else:
            rec_out = rec_out.at[r].set(rec_cat)
        mix_r_arr = jnp.stack(mix_r, axis=0).astype(BF16).astype(F32)
        w_out = w_lin_out_l[r].astype(F32)
        out_s = lin_out_s_l[r]
        yo = (mix_r_arr @ w_out) * out_s
        gdn_partials.append(yo.astype(BF16).astype(F32))

    delta_out = jnp.sum(jnp.stack(gdn_partials, axis=0), axis=0)
    if has_multi_out:
        return delta_out, {**state, "conv_out": conv_out, "rec_out": rec_out}
    if ex.VER:
        conv_pad = (
            jnp.pad(
                conv_in_l,
                ((0, 0), (0, 0), (0, HR - conv_in_l.shape[-2]), (0, 0)),
            )
            if conv_in_l.shape[-2] < HR
            else conv_in_l
        )
        return delta_out, {
            **state,
            "conv": conv_pad,
            "rec": rec_in_l.reshape(R, NS * vh, KD, VD),
            "conv_ckpt": conv_out,
            "rec_ckpt": rec_out,
        }
    return delta_out, {**state, "conv": conv_out, "rec": rec_out}

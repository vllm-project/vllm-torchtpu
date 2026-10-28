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
"""`GQAModule` pure-JAX reference forward pass (`gqa_reference`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, FP8, rmsnorm, rnd, rope_partial
from ...pool.config import cache_exec


def gqa_reference(
    params: Mapping[str, Any],
    x: Any,
    state: Mapping[str, Any],
    *,
    g: Any,
    ex: ExecSpec,
    pos: Any,
    att_idx: int = 0,
    layer_idx: int | None = None,
    paged_kv: bool | None = None,
    module_name: str = "gqa",
) -> tuple[jax.Array, dict[str, Any]]:
    """Pure-JAX multi-rank GQA forward for one attention layer (`att_idx`)."""
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    a = int(att_idx)
    lyr = a if layer_idx is None else int(layer_idx)
    L = int(getattr(g, "L", 1))
    nh = int(g.NH) // R
    HD = int(g.HD)
    ROT = int(getattr(g, "ROT", HD))
    theta = float(g.theta)
    eps = float(getattr(g, "eps", 1e-6))
    zc = bool(getattr(g, "zero_centered_norm", True))
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    rope_round = not ex.resolve_tiles(g).module(module_name).rope_fp32

    half = ROT // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=F32)[None, :] * 2.0 / ROT))
    xb = x[:B].astype(BF16).astype(F32)

    def _slice_sm(arr: Any) -> Any:
        if arr.ndim == 2:
            return arr
        return arr[:, lyr] if arr.shape[1] == L and lyr < arr.shape[1] else arr[:, a]

    w_att_in_all = params["att_in"]
    w_att_out_all = params["att_out"]
    w_att_in_l = w_att_in_all[:, a] if w_att_in_all.ndim == 4 else w_att_in_all
    w_att_out_l = w_att_out_all[:, a] if w_att_out_all.ndim == 4 else w_att_out_all

    in_scaled = jnp.dtype(w_att_in_all.dtype) == jnp.dtype(FP8)
    out_scaled = jnp.dtype(w_att_out_all.dtype) == jnp.dtype(FP8)
    has_in_s = in_scaled and ("att_in_s" in params or "in_s" in params)
    has_out_s = out_scaled and ("att_out_s" in params or "out_s" in params)
    att_in_s_l = (
        _slice_sm(params["att_in_s"] if "att_in_s" in params else params["in_s"])
        if has_in_s
        else None
    )
    att_out_s_l = (
        _slice_sm(params["att_out_s"] if "att_out_s" in params else params["out_s"])
        if has_out_s
        else None
    )
    qn_l = _slice_sm(params["qn"])
    kn_l = _slice_sm(params["kn"])

    kc_in = state["kc"]
    vc_in = state["vc"]
    multi_layer_kv = kc_in.ndim == 5
    kc_a = kc_in[:, a] if multi_layer_kv else kc_in
    vc_a = vc_in[:, a] if multi_layer_kv else vc_in
    if use_paged_kv:
        kc_a = jnp.clip(kc_a.astype(F32), -448.0, 448.0).astype(FP8).astype(BF16)
        vc_a = jnp.clip(vc_a.astype(F32), -448.0, 448.0).astype(FP8).astype(BF16)

    att_partials = []
    knew_ranks = []
    vnew_ranks = []
    koff = nh * 2 * HD

    for r in range(R):
        w_in = w_att_in_l[r].astype(F32)
        y = (
            rnd((xb @ w_in) * att_in_s_l[r])
            if att_in_s_l is not None
            else rnd(xb @ w_in)
        )

        qn = qn_l[r]
        kn = kn_l[r]

        qg = y[:, :koff].reshape(B, nh, 2 * HD)
        gate = qg[:, :, HD:].reshape(B, nh * HD)
        q_all = []
        k_new_r = []
        v_new_r = []
        for b in range(B):
            ang = pos[b].astype(F32) * inv
            cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
            qb = rmsnorm(qg[b, :, :HD], qn, eps, zero_centered=zc)
            kb = rmsnorm(y[b : b + 1, koff : koff + HD], kn, eps, zero_centered=zc)
            vb = y[b : b + 1, koff + HD : koff + 2 * HD]
            qb = rnd(
                rope_partial(qb, cos, sin, ROT, round_intermediates=rope_round)
                * (HD**-0.5)
            )
            kb = rope_partial(kb, cos, sin, ROT, round_intermediates=rope_round)
            if use_paged_kv:
                kb = jnp.clip(kb, -448.0, 448.0).astype(FP8).astype(F32)
                vb = jnp.clip(vb, -448.0, 448.0).astype(FP8).astype(F32)
            q_all.append(qb)
            k_new_r.append(kb[0])
            v_new_r.append(vb[0])
        q_r = jnp.stack(q_all, axis=0)
        kn_r = jnp.stack(k_new_r, axis=0)
        vn_r = jnp.stack(v_new_r, axis=0)
        knew_ranks.append(kn_r)
        vnew_ranks.append(vn_r)

        mix_r = []
        for b in range(B):
            sq, tb = divmod(b, T)
            plen = pos[sq * T]
            k_cache = kc_a[r, sq].astype(F32)
            v_cache = vc_a[r, sq].astype(F32)
            idx = jnp.arange(ctx)
            scores_c = jnp.where(
                idx[None, :] < plen,
                q_r[b].astype(BF16).astype(F32) @ k_cache.T,
                -1e30,
            )
            new_k = jnp.stack(
                [knew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)],
                axis=0,
            )
            new_v = jnp.stack(
                [vnew_ranks[r][sq * T + j].astype(F32) for j in range(tb + 1)],
                axis=0,
            )
            scores_n = q_r[b] @ new_k.T
            scores = jnp.concatenate([scores_c, scores_n], axis=-1)
            vals = jnp.concatenate([v_cache, new_v], axis=0)
            probs = jax.nn.softmax(scores, axis=-1)
            out_b = probs @ vals
            oh = rnd(out_b).reshape(nh * HD)
            gh = rnd(jax.nn.sigmoid(gate[b]))
            mix_r.append(rnd(oh * gh))
        mix_r_arr = jnp.stack(mix_r, axis=0).astype(BF16).astype(F32)
        w_out = w_att_out_l[r].astype(F32)
        yo = (
            (mix_r_arr @ w_out) * att_out_s_l[r]
            if att_out_s_l is not None
            else (mix_r_arr @ w_out)
        )
        att_partials.append(yo.astype(BF16).astype(F32))

    for r in range(R):
        for s_idx in range(NS):
            b0 = s_idx * T
            kc_a = kc_a.at[r, s_idx, pos[b0] : pos[b0] + T].set(
                knew_ranks[r][b0 : b0 + T].astype(BF16)
            )
            vc_a = vc_a.at[r, s_idx, pos[b0] : pos[b0] + T].set(
                vnew_ranks[r][b0 : b0 + T].astype(BF16)
            )

    kc_out = kc_in.at[:, a].set(kc_a) if multi_layer_kv else kc_a
    vc_out = vc_in.at[:, a].set(vc_a) if multi_layer_kv else vc_a
    delta = jnp.sum(jnp.stack(att_partials, axis=0), axis=0)
    return delta, {**state, "kc": kc_out, "vc": vc_out}

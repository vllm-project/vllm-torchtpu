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
"""Random Qwen3.8 segment inputs and their packing into segment-kernel operands.

Shared by the CPU kernel-vs-reference test and the on-TPU segment benchmark.
GDN / MoE weights and states come from `qwen35.random_logical_inputs` on the
segment's inner geometry; HC parameters, the attention output and `o_proj`
are added here. All per-rank arrays carry a leading `R` axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..blocks.gdn import GDNBlock
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, rup
from ..spec import ModelSpec
from . import qwen35
from .qwen38 import SEGMENT_LAYERS, Qwen38Geometry
from .qwen38_pack import pack_weights


@jax.tree_util.register_dataclass
@dataclass
class SegmentInputs:
    pos: Any
    x: Any  # [B, NHC*H] residual
    hc: dict[str, dict[str, Any]]  # sublayer -> {norm, down, up, inj} stacked
    w_l: dict[str, Any]  # library logical GDN / MoE weights
    st_l: dict[str, Any]  # library logical GDN states
    attn: Any  # [R, B, nh*HD]
    o_proj: Any  # [R, nh*HD, H]


def _hc_params(key: jax.Array, g: Qwen38Geometry, n: int) -> dict[str, jax.Array]:
    W = g.NHC * g.H
    k = jax.random.split(key, 4)
    return {
        "norm": jax.random.normal(k[0], (n, W), F32) * 0.05,
        "down": (jax.random.normal(k[1], (n, W, g.LOWRANK), F32) * 0.03).astype(BF16),
        "up": (jax.random.normal(k[2], (n, g.LOWRANK, W), F32) * 0.1).astype(BF16),
        "inj": (jax.random.normal(k[3], (n, W, g.NHC), F32) * 0.3).astype(BF16),
    }


def random_segment_inputs(
    key: jax.Array,
    g: Qwen38Geometry,
    ex: ExecSpec,
    L: int = SEGMENT_LAYERS,
    *,
    gain: float = 1.0,
    zero: tuple[str, ...] = (),
    router_gain: float = 1.0,
    head: bool = False,
) -> SegmentInputs:
    """Random inputs whose sublayers each move the residual visibly.

    The library's random output scales give near-zero sublayer deltas, so the
    output projection scales are raised; a test against such inputs would
    otherwise compare the residual with itself. `gain` scales those output
    gains (larger geometries need smaller ones).
    """
    R, B = ex.R, ex.B
    keys = jax.random.split(key, 7)
    pos, _, w_l, st_l = qwen35.random_logical_inputs(keys[0], g.inner(R, L, head), ex)
    gains = {"e2_s": 2000.0 * gain, "sh2_s": 400.0 * gain, "lin_out_s": 50.0 * gain}
    # `zero` silences a sublayer kind ("gdn", "moe") to isolate the others.
    for k in {"gdn": ("lin_out_s",), "moe": ("e2_s", "sh2_s")}.get(zero[0], ()) if zero else ():
        gains[k] = 0.0
    w_l = {k: v * gains[k] if k in gains else v for k, v in w_l.items()}
    # A larger router spreads the top-k logits so rounding drift upstream of the
    # router cannot flip a selection (`router_gain`); the library default is 1.
    w_l["router"] = (w_l["router"].astype(F32) * router_gain).astype(w_l["router"].dtype)
    hc = {
        "qsa_out": _hc_params(keys[1], g, 1),
        "gdn": _hc_params(keys[2], g, L if head else L - 1),
        "moe": _hc_params(keys[3], g, L),
    }
    K = g.NH // R * g.HD
    attn = (jax.random.normal(keys[4], (R, B, K), F32) * 0.5).astype(BF16)
    o_proj = (jax.random.normal(keys[5], (R, K, g.H), F32) * 0.05).astype(BF16)
    x = jax.random.normal(keys[6], (B, g.NHC * g.H), F32).astype(BF16)
    return SegmentInputs(pos, x, hc, w_l, st_l, attn, o_proj)


def _rep(x: jax.Array, R: int) -> jax.Array:
    return jnp.broadcast_to(x[None], (R, *x.shape))


def hc_weight_arrays(
    name: str, p: dict[str, Any], g: Qwen38Geometry, R: int, *, mix: bool
) -> dict[str, Any]:
    """Streamed HC weights of sublayer `name` (replicated over ranks)."""
    n, W, h = p["down"].shape[0], g.NHC * g.H, g.H
    if not mix:
        hinj = jnp.zeros((n, W, 128), BF16).at[:, :, : g.NHC].set(p["inj"])
        return {f"{name}_hinj": _rep(hinj, R)}
    n_down, k_up = rup(g.LOWRANK + g.NHC, 128), rup(g.LOWRANK, 128)
    hdn = jnp.zeros((n, W, n_down), BF16)
    hdn = hdn.at[:, :, : g.LOWRANK].set(p["down"])
    hdn = hdn.at[:, :, g.LOWRANK : g.LOWRANK + g.NHC].set(p["inj"])
    hup = jnp.zeros((n, k_up, W), BF16).at[:, : g.LOWRANK].set(p["up"])
    out = {f"{name}_hdn": _rep(hdn, R)}
    for b in range(g.NHC):
        out[f"{name}_hup{b}"] = _rep(hup[..., b * h : (b + 1) * h], R)
    return out


def _rows(arr: jax.Array, rows: tuple[int, ...], L: int) -> jax.Array:
    out = jnp.zeros((arr.shape[0], L, arr.shape[-1]), F32)
    return out.at[:, list(rows)].set(arr.astype(F32))


def pack_segment(
    spec: ModelSpec, g: Qwen38Geometry, ex: ExecSpec, inp: SegmentInputs, head: bool = False
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """`(weights, states, attn)` per-rank operands (leading `R`, not yet folded)."""
    R, B, T, L = ex.R, ex.B, ex.T, spec.config.L
    w_l, st_l, hc = inp.w_l, inp.st_l, inp.hc
    W = g.NHC * g.H
    weights = {
        "router": _rep(w_l["router"], R),
        "sh13": w_l["sh13"],
        "sh2": w_l["sh2"],
        "e13": tuple(w_l["e13"][:, j] for j in range(L)),
        "e2": tuple(w_l["e2"][:, j] for j in range(L)),
        "es": jnp.concatenate([w_l["e13_s"], w_l["e2_s"]], axis=-1)[:, :, :, None, :],
        **hc_weight_arrays("moe", hc["moe"], g, R, mix=True),
    }
    smalls: dict[str, Any] = {
        "moe_hcn": hc["moe"]["norm"],
        **{k: w_l[k] for k in ("post_norm", "sh_gate", "sh13_s", "sh2_s", "ecopy") if k in w_l},
    }
    if not head:
        weights["o_proj"] = inp.o_proj[:, None]
        weights.update(hc_weight_arrays("qsa_out", hc["qsa_out"], g, R, mix=False))
        smalls["qsa_out_hcn"] = _rows(_rep(hc["qsa_out"]["norm"], R), (0,), L)
    states: dict[str, Any] = {}
    n_gdn = L if head else L - 1
    if n_gdn > 0:
        gdn_rows = tuple(range(L - n_gdn, L))
        weights.update(
            lin_in=w_l["lin_in"],
            lin_out=w_l["lin_out"],
            **hc_weight_arrays("gdn", hc["gdn"], g, R, mix=True),
        )
        smalls.update(
            gdn_hcn=_rows(_rep(hc["gdn"]["norm"], R), gdn_rows, L),
            in_s=_rows(w_l["lin_in_s"], gdn_rows, L),
            out_s=_rows(w_l["lin_out_s"], gdn_rows, L),
            **{k: _rows(w_l[k], gdn_rows, L) for k in ("conv", "alog", "dtb", "gnorm")},
        )
        lay = GDNBlock.lin_in_layout(g.inner(R, L, head), ex)
        if T > 1:
            hr = rup(3 + T, 8)
            conv = jnp.pad(st_l["conv"], ((0, 0),) * 3 + ((0, hr - 3), (0, 0)))
        else:
            cr = rup(3 * B, 8)
            c3 = jnp.swapaxes(st_l["conv"], 2, 3).reshape(R, n_gdn, 3 * B, lay.QKV)
            conv = jnp.pad(c3, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0)))
        states["conv"] = conv.astype(F32)
        states["rec"] = (
            st_l["rec"].reshape(R, n_gdn, ex.NS * lay.vh, lay.KD, lay.VD).astype(BF16)
        )
    assert W == spec.config.H
    attn = jnp.pad(inp.attn, ((0, 0), (0, ex.M - B), (0, 0)))
    return pack_weights(spec, ex, weights, smalls), states, attn

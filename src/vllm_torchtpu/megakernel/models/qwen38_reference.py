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
"""Pure-JAX reference of one Qwen3.8 segment, composed from block references.

Inputs follow the library's per-rank logical layout (`qwen35.random_logical_inputs`
keys for the GDN / MoE weights and states) plus:

- `hc[name]`: per-sublayer HC params `{norm, down, up, inj}` stacked over the
  kernel layers the sublayer is active in (`qsa_out`: 1, `gdn`: L-1, `moe`: L);
- `attn [R, B, nh*HD]` and `o_proj [R, nh*HD, H]` for the QSA output projection.
"""

from __future__ import annotations

from typing import Any

import jax

from ..exec_spec import ExecSpec
from ..modules.gdn.reference import gdn_reference
from ..modules.hc.reference import hc_combine, hc_mix, hc_normalize
from ..modules.moe.reference import moe_reference
from ..modules.qsa_out import QSAOutModule
from .qwen38 import Qwen38Geometry


def _layer(p: dict[str, Any], i: int) -> dict[str, Any]:
    return {k: v[i] for k, v in p.items()}


def segment_reference(
    g: Qwen38Geometry,
    ex: ExecSpec,
    x: jax.Array,
    hc: dict[str, dict[str, Any]],
    w_l: dict[str, Any],
    st_l: dict[str, Any],
    attn: jax.Array,
    o_proj: jax.Array,
    *,
    L: int,
    moe_inputs: list | None = None,
    head: bool = False,
) -> tuple[jax.Array, dict[int, Any]]:
    """Return `(new residual [B, NHC*H], {gdn_idx: new GDN states})`.

    `moe_inputs`, when a list, receives each layer's MoE input `[B, H]`.
    """
    gi = g.inner(ex.R, L, head)
    nhc, eps = g.NHC, g.eps
    x = x[: ex.B]
    gdn_states: dict[int, Any] = {}
    for j in range(L):
        if j == 0 and not head:
            p = _layer(hc["qsa_out"], 0)
            xn = hc_normalize(x, p["norm"], nhc, eps)
            delta, _ = QSAOutModule().reference({"attn": attn, "o_proj": o_proj}, None)
        else:
            o = j if head else j - 1
            p = _layer(hc["gdn"], o)
            mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
            delta, gdn_states[o] = gdn_reference(
                w_l, mixed, st_l, g=gi, ex=ex, gdn_idx=o, layer_idx=j
            )
        x = hc_combine(delta, x, xn, p, nhc=nhc)
        p = _layer(hc["moe"], j)
        mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
        if moe_inputs is not None:
            moe_inputs.append(mixed)
        delta, _ = moe_reference(w_l, mixed, None, g=gi, ex=ex, layer_idx=j)
        x = hc_combine(delta, x, xn, p, nhc=nhc)
    return x, gdn_states

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
"""`ModelSpec` of the whole-step kernel: `L` layers of `[GDN, GDN, GDN, QSA]` groups.

Unlike `qwen38.segment_spec`, the QSA layer is a full layer (`QSAFullBlock`
under an HC sublayer that mixes in-kernel), so the layer loop runs straight
through every group with no XLA work between QSA cores. Layer `l` is a GDN
layer when `l % att_every < att_every - 1`, the QSA layer otherwise; every layer
has the HC(MoE) sublayer. `L = 48` is the full model; the prototype uses `L = 8`
(two groups).
"""

from __future__ import annotations

from typing import Any

from ..blocks.hc import HCSublayer
from ..models.qwen38 import (
    QWEN38_SEGMENT,
    Qwen38Geometry,
    Qwen38GDN,
    Qwen38MoE,
    Qwen38ResidualGeometry,
)
from ..spec import LayerType, ModelSpec, SwitchSlot
from .qsa_full import QSAFullBlock

WHOLESTEP = "qwen38_wholestep"


def wholestep_spec(
    g: Qwen38Geometry,
    R: int,
    *,
    L: int = 8,
    pool_gdn: bool = False,
    qsa: dict[str, Any] | None = None,
) -> ModelSpec:
    """`L` (a multiple of `att_every`) layers starting at a group boundary (layer 0 = GDN)."""
    assert L % g.att_every == 0, (L, g.att_every)
    if pool_gdn:
        from ..models.qwen38_gdn_pool import Qwen38PoolGDN as gdn_cls
    else:
        gdn_cls = Qwen38GDN
    gi = g.inner(R, L)
    hc = dict(inner_geometry=gi, nhc=g.NHC, lowrank=g.LOWRANK)
    gdn = HCSublayer(gdn_cls(geometry=gi), ar_slot=0, **hc)
    attn = HCSublayer(QSAFullBlock(**(qsa or {})), ar_slot=0, **hc)
    # Counts are per cycle (period = att_every): branch 0 runs at l % period < c0.
    mixer = SwitchSlot("mixer", branches=(gdn, attn), branch_counts=(g.att_every - 1, 1))
    moe = HCSublayer(Qwen38MoE(geometry=gi, moe_tp=False), ar_slot=1, emit_residual=True, **hc)
    return ModelSpec(
        name=WHOLESTEP,
        config=Qwen38ResidualGeometry(L=L, H=g.NHC * g.H, eps=g.eps),
        layer_type=LayerType(name=QWEN38_SEGMENT, slots=(mixer, moe)),
    )

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
"""Synthetic hybrid (`GDN` + `GQA` + `MoE`) `ModelSpec` fixtures and knob tables."""

from __future__ import annotations

from typing import Any

from tests.megakernel.megakernel_fakes import (
    TINY_TILES as _FAKE_TINY_TILES,
)
from tests.megakernel.megakernel_fakes import (
    FakeGeometry,
    tile_field,
    with_tile_fields,
)
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.modules.gqa import GQAModule
from vllm_torchtpu.megakernel.modules.moe import MoEModule
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec, SwitchSlot

TINY_TILES = _FAKE_TINY_TILES.with_modules({"gqa": {"kv_blk": 256, "nkv": 3}})
SynthHybrid = FakeGeometry


def synth_hybrid_spec(
    g: FakeGeometry | None = None, *, R: int = 8, **overrides: Any
) -> ModelSpec:
    """Build a synthetic hybrid `ModelSpec` (`GDN` + `GQA` + `MoE` + `PoolBackend`)."""
    if g is None:
        mtp = bool(overrides.get("moe_tp", False))
        base = dict(
            L=4, att_every=2, H=256, NH=max(2, R), NKV=min(2, R), HD=256, ROT=64
        )
        base.update(
            KH=max(2, R), VH=max(2, R), KD=128, VD=128, E=128, TOPK=2, moe_tp=mtp
        )
        base.update(
            {"I": max(256, 128 * R) if mtp else 256, "IS": max(256, 128 * R)}
            | overrides
        )
        g = FakeGeometry(**base)
    gqa, gdn, moe = GQAModule(), GDNModule(), MoEModule()
    decls = [
        gqa.cache_decl(g) if (i + 1) % g.att_every == 0 else gdn.cache_decl(g)
        for i in range(g.L)
    ]
    play = pool.derive_pool_layout(decls, R=R)
    gqa_p, gdn_p = GQAModule(pool_layout=play), GDNModule(pool_layout=play)
    br, cnt = (
        ((gqa_p,), (1,)) if g.att_every == 1 else ((gdn_p, gqa_p), (g.att_every - 1, 1))
    )
    mixer = SwitchSlot("mixer", br, cnt)
    pb = pool.make_pool_backend(
        play,
        check=getattr(g, "check", None),
        rec_chunk=GDNModule.chunk_rows,
    )
    return ModelSpec(
        "synth_hybrid", g, LayerType("synth_layer", (mixer, moe)), cache_backend=pb
    )


def synth_full_scale_spec(*, moe_tp: bool = False, R: int = 8) -> ModelSpec:
    """Build a 60-layer full-scale synthetic `ModelSpec` for VMEM fitting tests."""
    kw = dict(L=60, att_every=4, H=4096, NH=32, NKV=2, HD=256, ROT=64, KH=16, VH=64)
    return synth_hybrid_spec(E=512, TOPK=10, I=1024, IS=1024, moe_tp=moe_tp, R=R, **kw)


EXACT_FLIPS: list[tuple[str, str, Any, tuple[int, int], dict[str, Any]]] = [
    ("nbd", "nbd", 8, (2, 4), {}),
    ("nkv", "gqa.nkv", 4, (2, 4), {}),
    ("lazy", "lazy", False, (2, 4), {}),
    ("pair_kv", "gqa.pair_kv", True, (2, 4), {}),
    ("kv_ering", "gqa.kv_ering", True, (2, 4), {}),
    ("gqa_qv", "gqa.gqa_qv", True, (2, 4), {}),
    ("rel_fetch", "rel_fetch", True, (2, 4), {}),
    ("bounds_checks", "bounds_checks", True, (2, 4), {}),
    ("vmem_mib", "vmem_mib", 48, (2, 4), {}),
    ("smem_mib", "smem_mib", 2, (2, 4), {}),
    ("moe_group_rows", "moe.moe_group_rows", 0, (2, 4), {"moe.moe_unroll": True}),
    (
        "moe_group_rows-pipe_ep",
        "moe.moe_group_rows",
        0,
        (2, 4),
        {"moe.pipe_ep": True, "ebank": 64 << 10, "nbe": 6, "geom_kw": {"I": 512}},
    ),
    ("gdn_bufs-wide", "gdn.gdn_bufs", 2, (4, 24), {}),
    ("pair_kv-wide", "gqa.pair_kv", True, (4, 24), {}),
    ("kv_ering-wide", "gqa.kv_ering", True, (4, 24), {}),
    ("gqa_qv-wide", "gqa.gqa_qv", True, (4, 24), {}),
]

__all__ = [
    "EXACT_FLIPS",
    "SynthHybrid",
    "TINY_TILES",
    "synth_full_scale_spec",
    "synth_hybrid_spec",
    "tile_field",
    "with_tile_fields",
]
